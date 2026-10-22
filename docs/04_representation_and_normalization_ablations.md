# 04 - Representation, normalization, and rank ablations: what a training loss can and cannot decide

> Evidence tags: **✅ measured**, **🟡 strong inference**, **❓ unverified**.
> Source material: internal training logbook (run #1, run #2, normalization comparison, 2026-09-19
> / 2026-09-20 / 2026-09-23 notes). Loss curves behind the plots are in [`data/`](../data/).

## 0. Scope

This note collects the controlled comparisons that shaped the final recipe, and states explicitly
where a comparison is not controlled. The rule used throughout: **training loss, offline action
error, and task success are three different measurements**. Only the first one is present in this
repository as raw data; the other two are mentioned only where a source exists.

## 1. Run inventory

| # | task | action representation | adapter | hardware | steps | terminal train loss | source |
|---|---|---|---|---:|---:|---|---|
| 1 | tray placement | absolute | default LoRA r=16, 38 modules, vision frozen | 4×A100-40G | 25,000 | **0.081** | ✅ `data/tray_absolute_default_lora.csv` |
| 2 | tray placement | relative-except-claw (14/16) | same as #1 | 4×A100-40G | 25,000 | **0.054** | ✅ launch log quote (no exported csv) |
| 3 | express parcel weighing | relative-except-claw | LoRA r=32 + vision full (module count not recorded for this run) | 4×RTX 5880 | 30,000 | **0.015** | ✅ `data/kuaidi_rel_30k.csv` |
| 4 | part sorting | relative-except-claw | same as #3 | 4×RTX 5880 | 30,000 | **0.010** | ✅ `data/cixilingjian_rel_30k.csv` |
| 5 | socket insertion | absolute | same as the 257-module relative socket configuration | 4×RTX4090-24G | 28,900 logged | **0.019** | ✅ `data/socket_absolute_b12.csv` |

Also trained but **not included** in this repository: the final-recipe tray run (10-GPU internal
node), a relative-action socket run (35k steps, 4×A100), and two reduced-rank socket runs that were
still in progress when the project snapshot was taken. Their raw logs are not part of the project
export, so no number from them is quoted here.

> Cross-task loss comparison is not meaningful. Within task #1, runs #1 and #2 are the only
> deliberately single-variable pair in the project.

## 2. Absolute vs relative actions (runs #1 and #2)

**Design.** Same 500-episode tray dataset, same base checkpoint, same LoRA config (default r=16,
38 modules, 1,287,168 trainable parameters), same optimizer schedule (2.5e-4 -> 2.5e-5 cosine,
warmup 1000), same batch (16 per card, 64 global), same seed, same step count. Only the action
representation changed:

- run #1: absolute joint targets, `use_relative_actions: false`;
- run #2: `use_relative_actions: true` with `relative_exclude_joints: ["claw"]`, giving a
  14/16 relative mask (`RRRRRRRARRRRRRRA`: dims 7 and 15, the two gripper channels, stay absolute).

Run #2 read from an isolated metadata root: `data/`, `videos/`, `images/` symlinked, `meta/` copied,
relative statistics recomputed offline. The source root's `stats.json` and `info.json` hashes were
verified unchanged afterwards. This isolation is a hard rule: the shared dataset root is owned by
the sponsor and must never be mutated in place.

**Result.**

| run | terminal loss @ 25k | wall clock |
|---|---:|---:|
| #1 absolute | 0.081 | 8 h 37 m 53 s |
| #2 relative-except-claw | 0.054 | 8 h 36 m 03 s |

**What this does and does not show.** ✅ The pair is a clean single-variable experiment. ❓ It does
**not** show that relative actions are better: the two losses live on different scales because the
prediction target was redefined. Comparing them as if 0.054 < 0.081 meant "better policy" is the
classic mistake this project tried to avoid. The decision between representations has to be made
downstream, on deployment success, which is not recorded in this repository.

Related bug worth knowing: before the metadata root was fixed, `meta/info.json` stored
`action.names` as a dict (`{"action_names": [...]}`). The framework calls `list(action_names)`,
gets `['action_names']` (length 1), broadcasts it over 16 dimensions, and produces an **all-True
mask** - i.e. 16/16 relative, including the grippers. It does not raise. The fix is to flatten
`action.names` to a list and assert the expected mask.

## 3. Normalization comparison: default QUANTILES vs MEAN_STD vs QUANTILE10

✅ Same 500-episode tray data, same base, LoRA r=16, absolute actions, 4×A100, batch 16 per card,
`seed=1000`, `compile_model=true`. The only intended variable is `normalization_mapping.STATE/ACTION`:
default `QUANTILES` vs `MEAN_STD` (vs `QUANTILE10`).

| step | default QUANTILES (run #1) | MEAN_STD run | ratio |
|---:|---:|---:|---:|
| 100 | 0.367 | 0.748 | 2.04x |
| 800 | 0.171 | 0.342 | 2.00x |
| 2,000 | 0.121 | 0.246 | 2.03x |
| 4,000 | 0.105 | 0.213 | 2.03x |
| 6,000 | 0.098 | 0.199 | 2.03x |
| 8,000 | 0.092 | 0.188 | 2.04x |
| 10,000 | 0.088 | 0.176 | 2.00x |
| 25,000 | **0.081** | not reached | - |

The default column is reproducible from `data/tray_absolute_default_lora.csv`. The MEAN_STD column
is transcribed from the internal comparison note; the MEAN_STD run itself was stopped at ~10.7k of
60k steps, so its terminal loss is unknown, but the ratio is stable at ~2.0x across every matched
step and its curve had already flattened. Verdict ✅: keep the default `QUANTILES`.

`QUANTILE10` (default architecture) reached 0.661 @100 and 0.308 @800, i.e. ~1.80x the default, then
was terminated by SIGTERM at step 863 during warmup. That is a direction, not a result ❓. One
scheduled QUANTILE10 run never left the CPU because a launch flag was missing; it is excluded from
the comparison rather than counted as evidence.

Interpretation (🟡, not proven): `MEAN_STD` assumes an approximately unimodal Gaussian state/action
distribution, while joint and gripper channels here are multi-modal and partly near-discrete;
quantile scaling is more robust to that. The mechanism was not tested with distribution statistics.
What is proven is narrower: under this data and architecture, `MEAN_STD` fits the same targets
visibly worse at every matched step, which was enough to stop the direction.

## 4. Rank reduction: measured, designed, launched, unfinished

The spectrum measurement in [02](02_lora_parameter_budget_and_rank_spectrum.md) says the language
model wants ~16 and the action expert ~8, while the projector and time MLP must keep 32. A design
with that `rank_pattern` and `alpha = rank` (PEFT scaling 1.0 instead of 0.25) was verified offline
and launched on two machines:

- 257 modules hit: language model 126 at r=16, action expert 126 at r=8, five projection-ish
  modules at r=32;
- LoRA 23,377,920 + vision 412,442,352 = 435,820,272 trainable parameters, observed exactly in both
  launch logs ✅;
- the relative-action version OOMed at batch 12 per card at step 1 and restarted at batch 10.

This validates the parameter accounting and the configuration path. It does **not** answer whether
the reduced rank costs accuracy ❓: the runs had not finished, and the project has no success metric
for that task. Also note that the relative reduced-rank run differs from its A100 baseline in batch
size and hardware, so it is not a strict single-variable comparison even once finished.

## 5. Metric hierarchy used for decisions

The project deliberately keeps three layers separate:

| layer | measurement | available in this repo |
|---|---|---|
| optimization | training loss (MSE on the action chunk, model units) | ✅ raw loss curves in `data/` |
| offline policy quality | action MAE against held-out demonstration frames | ❌ not included |
| task performance | success rate / completion on the physical robot | ❌ not included |

A held-out action-MAE protocol was designed and a parallel team line ran it on a separate
10-GPU internal node; those numbers appear in team records but their raw logs are not part of this
export, so they are not quoted here (see `PROVENANCE.md`, "not included"). The same applies to
on-robot success rates.

Consequence for reading this repository: the loss curves show that optimization worked and that
configurations are distinguishable. They do not rank policies. The deployment note
([03](03_deploy_nonpersistent_buffers.md)) is the only place where an offline action-error number
appears, because that note is about a loader bug and the metric is a before/after check on the same
frames.

## 6. What was believed and later corrected

- ❌ "The default LoRA already adapts the VLM." Corrected by reading the regex: 38 modules, zero
  language-model adapters, vision frozen ([01](01_pi05_integration_contract.md) section 5).
- ❌ "A converged loss means the checkpoint loads correctly." Corrected twice: by the silent
  state-dict failure and by the non-persistent buffer bug.
- ❌ "Lower training loss decides the representation." Corrected before the decision was made:
  absolute and relative losses are on different scales.
- ❌ "MEAN_STD is a safe alternative." Measured 2x worse at every matched step; dropped.
