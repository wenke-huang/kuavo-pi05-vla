# 01 - The π0.5 integration contract: actions, state, cameras, and the failures that stay silent

> Evidence tags used throughout this repo: **✅ measured**, **🟡 strong inference**, **❓ unverified**.
> Source material: internal engineering notes (2026-09-18 / 2026-09-19) written while reading the
> LeRobot π0.5 policy source at commit `ecc02f3` (submodule `lerobot` v0.5.1-40-ga07f22e2) and running
> the first four training jobs on a 4×A100-40G node.

## 0. Why this note exists

Before the first real training run, two things looked inconsistent with the data:

1. The demonstrations carry **16-D joint actions**, while the π0.5 base model is declared with
   `max_action_dim = 32`.
2. The base model config names its cameras `base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb`,
   while the sponsor-provided dataset uses `head_cam_h / wrist_cam_l / wrist_cam_r`.

The expectation was that a mapping layer had to be written. It did not. Both mismatches are handled
inside the framework, but only if the configuration is left in a specific state. This note records
what the code actually does, which configuration keys silently do nothing, and which failure modes
therefore have no error message.

## 1. Action dimension: pad to 32, slice back, and slice the loss too

✅ `pad_vector` right-pads the action to `max_action_dim`; the loss is computed on the padded tensor
and then sliced back to the dataset width before reduction:

```python
# lerobot/policies/pi05/modeling_pi05.py
def pad_vector(vector, new_dim):                      # :141
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))

actions = pad_vector(batch[ACTION], self.config.max_action_dim)   # :1210, 16 -> 32
actions = actions[:, :, :original_action_dim]                    # :1245, inference: 32 -> 16
losses  = losses[:, :, :original_action_dim]                     # :1269, training: 32 -> 16
```

Consequences:

- The 16 dataset joints always occupy slots 0-15; slots 16-31 are zero.
- The padded slots do not enter the loss. They can still influence slots 0-15 through
  `action_in_proj` and attention, so this is "not in the loss", not "statistically irrelevant".
- No data-side padding is needed. The dataset stays 16-D.

## 2. State: there is no state projector, and state is text

✅ The base safetensors header contains 812 tensors and **no `state_proj`**. Only `action_in_proj`
[1024, 32], `action_out_proj` [32, 1024], and `time_mlp_in/out` exist on the action side.
`embed_prefix` has no state argument (`:641`).

π0.5 moves the robot state into the language prompt. A processor step normalizes the state to
[-1, 1], discretizes it into 256 bins, and appends it to the task string:

```python
# lerobot/policies/pi05/processor_pi05.py
@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")   # :51
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    max_state_dim: int = 32
    ...
    discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    full_prompts.append(f"Task: {cleaned_text}, State: {state_str};\nAction: ")
```

Consequences:

- `max_state_dim = 32` is dead weight when a dataset is present: it is only used by
  `validate_features()` when no dataset defines the feature shapes (`configuration_pi05.py:134-139`).
- **Order matters**: normalization must run before discretization. The bin edges assume inputs in
  [-1, 1]; reversing the order does not raise, it silently writes garbage numbers into the prompt.

## 3. Camera keys: `input_features: {}` means "trust the dataset"

✅ `make_policy` fills `output_features` unconditionally from the dataset, but overwrites
`input_features` only when it is empty:

```python
# lerobot/policies/factory.py
cfg.output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}   # :507
if not cfg.input_features:                                                                     # :508
    cfg.input_features = {k: ft for k, ft in features.items() if k not in cfg.output_features}
```

The training YAML is shipped with `policy.input_features: {}`. The empty dict is falsy, so the
dataset's own camera keys win. The base checkpoint's `config.json` is never read during training,
because `from_pretrained(path, config=cfg)` is called with a non-`None` config
(`lerobot/policies/pretrained.py:95`); only `model.safetensors` is loaded. `dataset.rename_map: {}`
is therefore also correct as-is.

One configuration can break all of this: passing `--policy.path=<base>` on the CLI instead of
`policy.pretrained_path` in YAML. That path does read the base `config.json`
(`lerobot/configs/train.py:88-93`), `input_features` becomes non-empty, the dataset no longer fills
it, and image lookup fails with `ValueError: All image features are missing from the batch`.
`--policy.path` and `--policy.type` are also mutually exclusive.

## 4. The one silent trap that remains: camera order

✅ `image_features` preserves the insertion order of `input_features`, which comes from the order of
`features` in the dataset `info.json`. `_preprocess_images()` (`modeling_pi05.py:1156`) appends
images in that order, and the pretrained weights expect fixed slots:

| slot | semantic role | this dataset's feature |
|---:|---|---|
| 0 | main / head view | `observation.images.head_cam_h` ✅ |
| 1 | left wrist | `observation.images.wrist_cam_l` ✅ |
| 2 | right wrist | `observation.images.wrist_cam_r` ✅ |

The dataset happens to be alphabetically ordered in the same way as the semantic order. This is a
coincidence, not a guarantee. The only error the code raises is when **none** of the image keys
match; a wrong order trains silently against the wrong slot semantics. The slot semantics are a
learned positional prior - no parameter name in the checkpoint exposes them.

> 🟡 The mapping "0 = head, 1 = left wrist, 2 = right wrist" is inferred from the openpi / base
> checkpoint convention. No line in the LeRobot source asserts it. Verify it with a two-step smoke
> test on any new dataset.

## 5. The default LoRA target regex is stale: 38 modules, 1.29 M parameters

✅ The framework's default LoRA targets are built from a regex copied from π0. Three of its five
projection names do not exist in π0.5:

| name in default regex | reality in π0.5 |
|---|---|
| `state_proj` | does not exist (`modeling_pi05.py:1096-1098` skips it explicitly) |
| `action_time_mlp_in` | actual attribute is `time_mlp_in` (`:580`) |
| `action_time_mlp_out` | actual attribute is `time_mlp_out` (`:581`) |

PEFT matches with `re.fullmatch`, so those branches never fire. What actually trains is
`gemma_expert`'s `q_proj`/`v_proj` plus `action_in_proj`/`action_out_proj`:

| configuration | modules | breakdown | trainable LoRA params |
|---|---:|---|---:|
| framework default at `r=16` | 38 | action expert 36, VLM trunk 0, action projections 2 | **1,287,168** (0.031% of 4.14 B) |
| explicit target regex at `r=16` | 256 | expert 126, VLM trunk 126, action projections 4 | 26,641,408 |
| explicit target regex at `r=32` + full vision tower | 256 (+vision) | same 256 modules, rank doubled | 53,282,816 LoRA + 412,442,352 vision = **465,725,168** (10.1% of 4.61 B) |

A later revision of the regex also covers `multi_modal_projector.linear`, giving **257** modules and
replacing the 4 projection-layer targets with 5 (see [02](02_lora_parameter_budget_and_rank_spectrum.md)).

Cross-check: the earlier `r=32` run on the previous cluster reported `num_learnable_params=2,574,336`,
exactly twice the `r=16` value. The count is therefore a reliable fingerprint of the regex that was
actually in effect.

Practical consequence: with the default regex the 2 B PaliGemma language trunk gets **zero**
adapters and the vision tower stays frozen. Loss still decreases, so this is invisible in the loss
curve. Any claim of "we fine-tuned the VLM" needs the parameter count next to it.

## 6. Failure modes with no error message

✅ These are the ones observed or read line-by-line during the first training attempt:

| # | failure | mechanism | how to detect |
|---|---|---|---|
| 1 | random-weight training | `from_pretrained` wraps state-dict loading in a bare `except Exception` (`modeling_pi05.py:1051`), prints one warning, returns a randomly initialized model | grep the log for `All keys loaded successfully!`; a `Warning: Could not load state dict` line means the run is worthless |
| 2 | processor registry drift | the base repo's frozen `policy_preprocessor.json` names a step `relative_actions_processor`, while the installed LeRobot registers the same class as `delta_actions_processor` (`processor/relative_action_processor.py:84`); registry lookup raises `KeyError` after weights load | first-run crash; fix the one string (the step is `enabled: false`, so it is a no-op) |
| 3 | relative-action mask silently wrong | `factory.py:511-515` calls `list(action_names)`; when `meta/info.json` stores `names` as `{"action_names": [...]}` (v3.0 style), `list()` returns `['action_names']` (length 1), broadcast over 16 dims gives an all-True mask -> **16/16 relative** instead of 14/16 | assert `sum(mask)==14` before training; expected mask is `RRRRRRRARRRRRRRA` (dims 7 and 15 absolute) |
| 4 | `--policy.path` trap | see section 3 | image lookup `ValueError` at start |
| 5 | vision tower silently reverts at deploy | `modules_to_save` wraps the weight; if `set_adapter` is not called at load time, `ModulesToSaveWrapper` forwards to the **original frozen module** | compare deployed vs training-weight outputs on a fixed batch; loss alone cannot detect it |

## 7. Other risks recorded but not yet verified

These came from the same static review and are marked 🟡 / ❓; they are relevant when interpreting
any loss curve from this codebase:

- 🟡 Episode-padding frames are not masked out of the loss: with `chunk_size=50` at 10 Hz, the last
  50 frames of every episode are trained to repeat the final action (~6% of frames). `PI05Policy.forward`
  applies no `action_is_pad` mask.
- 🟡 848×480 images are letterboxed to 224×224, so ~44% of the visual tokens are black padding
  (content is 224×126). Raising `image_resolution` cannot help: the positional embedding is fixed at
  256 patches.
- 🟡 `tokenizer_max_length` in YAML is a no-op; the tokenizer processor is inherited from the base
  preprocessor JSON.
- ❓ Whether the camera-slot convention holds for a new dataset has to be re-checked. Retraining a
  two-step smoke test is the cheapest proof.

## 8. Review method and residual uncertainty

The source-reading conclusions in sections 1-4 were split into six falsifiable claims and each was
handed to an independent reviewer with the instruction to find a counter-example, file and line.
None was falsified. The `--policy.path` trap was found by that review, not by the first read.

Residual uncertainty: all of sections 1-4 come from reading source plus that review, **not from an
end-to-end instrumented run**. The recommended closure is a 2-step smoke run with real data that
prints the resolved feature names, the padded action width, and the prompt string. Until then the
slot convention (section 4) stays 🟡.
