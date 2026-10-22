# 03 - Root cause of "incoherent actions" on the deployed π0.5 policy: non-persistent buffers never rebuilt

> Evidence tags: **✅ measured**, **🟡 strong inference**, **❓ unverified**.
> Source material: internal note 2026-09-21 (deployment-side diagnosis), and the internal
> deployment-repository diagnostics note 2026-09-20 quoted in the project logbook.

## 0. Symptom and one-sentence cause

A π0.5 checkpoint with **absolute actions** produced visibly incoherent motion when served on the
robot side, while the team's reference deployment package was known to operate correctly.

Cause: our fork of `kuavo_deploy/utils/policy_loader.py` was missing
`_restore_pi_nonpersistent_buffers`. The policy is constructed on the `meta` device and then
allocated with `to_empty`, which leaves non-persistent buffers (Gemma RoPE `inv_freq`, SigLIP
`position_ids`) filled with uninitialized memory. The checkpoint does not contain them, and the
streaming loader only copies tensors present in the state dict, so nothing ever overwrites them.
Positional encoding is therefore corrupt from the first forward pass.

## 1. Mechanism ✅

Both loading paths (full checkpoint and PEFT adapter) funnel through
`_load_pi_streaming` -> `_construct_empty_pi`:

1. the policy is built under `torch.device("meta")` to save memory;
2. `policy.to_empty(device)` allocates real storage for every parameter and buffer, **without
   initialization**;
3. the loader copies the checkpoint tensor-by-tensor, keyed by state-dict names.

Non-persistent buffers are excluded from `state_dict` by definition, so step 3 never touches them.
This is not a bug in the checkpoint or in the model code - it is a property of the
`meta` + `to_empty` loading pattern, and it is usually hidden because the training script never
reloads a model this way.

The deployed model at the time was a PEFT checkpoint with
`type=pi05, chunk_size=20, n_action_steps=20, use_relative_actions=false` (i.e. absolute actions).

## 2. Direct evidence of corrupt buffers ✅

The two affected modules were rebuilt from config in isolation and compared:

```
GemmaRotaryEmbedding(cfg)
  reference inv_freq      = [1.0000, 0.9306, 0.8660, 0.8058, 0.7499, 0.6978]
  meta -> to_empty(cpu)   = [1.4013e-45, 1.4013e-45, 0, 1.3873e-43, 0, 0]
  equal? False

SiglipVisionEmbeddings(cfg)
  reference position_ids  = [0, 1, 2, 3, 4, 5, 6, 7]
  meta -> to_empty(cpu)   = [2314861380256428066, 4485057601257611296, ...]
  equal? False
```

Both values come from the same mechanism; the exact garbage depends on allocator state and is not
guaranteed to be deterministic (uninitialized memory is not guaranteed to be zero). The practical
consequence is stable: positional frequencies and image-patch indices are wrong, so every action
chunk is generated from a corrupted positional prior.

## 3. Fix ✅

Port `_restore_pi_nonpersistent_buffers` from the reference deployment package into the fork, and
call it immediately after `policy.to_empty(device=...)`:

1. before returning from `_construct_empty_pi`, call `_restore_pi_nonpersistent_buffers(policy)`;
2. the function rebuilds known non-persistent buffers from `config` (Gemma `inv_freq`, SigLIP
   `position_ids`) and **fails closed on unknown buffer types**, so a future framework upgrade
   cannot silently reintroduce uninitialized buffers.

Procedure used on the deployment host (all paths relative to the deployment workspace):

```bash
cd <deploy-workspace>/kuavo_deploy/utils
cp -a policy_loader.py policy_loader.py.bak_before_nonpersistent_fix_$(date +%Y%m%d_%H%M%S)
cp <reference-package>/kuavo_deploy/utils/policy_loader.py policy_loader.py
diff policy_loader.py <reference-package>/kuavo_deploy/utils/policy_loader.py   # must be identical
```

Equivalently, two edits to the local file: add the function, add its call after `to_empty`.

Verification performed:

- ✅ `python -m py_compile kuavo_deploy/utils/policy_loader.py` passes;
- ✅ byte-identical `diff` against the reference package's loader;
- ✅ `git diff --stat` against the fork's HEAD shows only the function plus its call site;
- ✅ rebuilding the isolated Gemma / SigLIP modules through the `meta -> to_empty` path and calling
  the restore function gives `torch.allclose` / `equal` against the reference values;
- ✅ launch logs contain the new line reporting how many non-persistent positional buffers were
  rebuilt.

Rollback is the backup copy above.

## 4. Offline before/after measurement

The internal deployment-repository diagnostics note records a six-frame offline replay on the same
batch of dataset frames, before and after the fix:

| metric | before | after |
|---|---:|---:|
| six-frame action prediction MAE | **0.267916 rad** | **0.014516 rad** |
| close-gripper prediction on previously failing frames | not saturated | close to 1.0 |

Two boundaries must be stated with this number:

- it is an **offline** replay against demonstration actions, not a robot run. The diagnostics note
  explicitly says the robot was not connected and no on-robot inference was started;
- the raw before/after logs and the summary JSON live in the deployment machine's diagnostics
  directory and are **not included in this repository**. The number is therefore reported with
  provenance, not independently reproducible here (see `PROVENANCE.md`). In the evidence scheme of
  this repo it is tagged **🟡**: measured internally, but the measurement cannot be re-run from the
  published material.

## 5. Other differences to the reference package (not the root cause)

The fork and the reference package also differed in several smaller ways. They are recorded so that
later debugging does not blame them for an unrelated symptom:

| file / area | our fork | reference package | effect |
|---|---|---|---|
| `config.py` error message | prints `[height, width]` | prints `[width, height]` | message only |
| `kuavo_env/KuavoBaseRosEnv.py` | `h, w = resize_wh`, declares `Box(shape=(3,848,480))` | `w, h = resize_wh`, matches real `(3,480,848)` | observation-space declaration only, unused by inference |
| `utils/obs_buffer.py` | additionally keeps old `/camera/...` topic names | only `/cam_h,/cam_l,/cam_r` | harmless compatibility code |
| `src/scripts/script.py` | `run` performs a two-stage arm homing first | `run` starts from the current pose | behavior difference; a bad initial pose can worsen "incoherent" motion |
| deploy YAML | `async_control_hz: 10.0`, low watermark 15, timeout 2.0, RTC max delay 8 | `0.0`, 5, 1.0, 3 | only active with `async_inference: true`, which was false |
| `kuavo_service/server.py` (client mode) | manual pre/post-processing | uses `config_inference` | only active in `policy_type: client` |

There is also a harmless reciprocal confusion around `resize_wh = [848, 480]`: `cv2.resize` treats
`dsize` as `(width, height)`, matching the checkpoint's `[3, 480, 848]`, so the resize value is
correct; only the observation-space declaration in `KuavoBaseRosEnv` reads it in the wrong order.

## 6. Lessons

- **"Loss converges" and "weights were loaded" are independent facts.** This bug and the silent
  state-dict failure in [01](01_pi05_integration_contract.md) are the same class: a plausible
  training curve with a corrupt runtime.
- A private fork of a vendor deployment package does not receive upstream fixes. Diff
  `kuavo_deploy/` against the reference package on every model or framework change, with
  `policy_loader.py` first.
- Prefer a loading path with an explicit post-allocation invariant check (rebuild buffers, then
  assert equality against config) over one that looks correct by inspection.
- ❓ The full end-to-end check (load the 14 GB model, run inference on the robot) is not recorded.
  The buffer-level and offline-replay checks closed the mechanism, not the robot behavior.
