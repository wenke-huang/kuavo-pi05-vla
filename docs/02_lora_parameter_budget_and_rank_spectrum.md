# 02 - Parameter budget and measured rank requirements of the π0.5 weight update

> Evidence tags: **✅ measured**, **🟡 strong inference**, **❓ unverified**.
> Source material: internal note 2026-09-21 (`各模块LoRA秩实测`), plus the training logbook
> parameter accounting for the 4×RTX 5880 and 4×A100 jobs. Method implemented in
> [`scripts/rank_spectrum.py`](../scripts/rank_spectrum.py).

## 0. The question

The final recipe applies LoRA with `r=32` to 256-257 modules and unfreezes the whole vision
tower. That budget was inherited from a proposal, not measured. Two competing hypotheses:

- **wasteful**: the updates are effectively low-rank, so most of the 32 dimensions carry 1% of the energy;
- **binding**: some modules want more rank, and 32 is a cap we cannot see past.

Both can be tested from the trained weights alone, without extra GPU time.

## 1. What was measured

✅ Two checkpoints from the same relative-action socket task, steps 12000 and 26000:

| item | value |
|---|---|
| PEFT update | `dW = (alpha / r) * B @ A` with `alpha=8`, `r=32`, so the PEFT scaling is 0.25 |
| full-parameter update | `dW = W_finetuned - W_base` for the vision tower (no rank cap, true rank observable) |
| modules | 257 LoRA modules + vision tower; base checkpoint 14.47 GB in fp32 |
| checkpoints | relative-except-claw, vision tower full-parameter, VLM/expert/action projections LoRA `r=32` |
| compute | plain SVD in numpy; the vision tower's full-parameter SVD dominated the ~7 min runtime |

The two checkpoints produced the same per-group verdict, which is why the numbers below are quoted
once and not per checkpoint.

## 2. Method

For a weight update `dW` with singular values `s_1 >= ... >= s_m`:

- cumulative spectral energy `E(k) = sum_{i<=k} s_i^2 / sum_i s_i^2`;
- effective rank `k_eta = min {k : E(k) >= eta}` with `eta = 0.99`;
- **Eckart-Young**: the best rank-k approximation has relative Frobenius error exactly
  `sqrt(1 - E(k))`.

`E(k)` is invariant to the global scalar `alpha / r`. The energy threshold answers "how many
directions do I need to keep 99% of the update energy"; the Eckart-Young bound answers "what
relative error does a rank-k adapter have to live with".

## 3. Results ✅

| group | structure | E(8) / E(16), mean | k@eta=0.99, median / max | rank the update needs |
|---|---|---:|---:|---|
| **vision tower** | full-parameter | - / - | **704 / 1026** (full rank 1152) | **approximately full rank** -> keep full-parameter |
| **language model** | LoRA | 0.978 / 0.996 | **8-11 / 25** | **16** covers the worst layer, 32 is headroom |
| **action expert** | LoRA | 0.993 / 0.999 | **5-6 / 15-17** | **8** already covers the worst layer |
| action projections (4 layers) | LoRA | 0.93 / 0.99 | 14 / 19-21 | 16-24 |
| `multi_modal_projector.linear` | LoRA | 0.74 / 0.88 | **30 / 30** (cap 32) | **at the cap** |

Anomaly: in language-model layer 17, the MLP `down/gate/up` LoRA tensors are all exactly zero, i.e.
never trained. The cause was not investigated ❓. It does not move the group verdict, but it means
the per-layer picture is not perfectly clean.

## 4. Reading the table

- ✅ **Rank is not the bottleneck for the language model and the action expert.** Their spectra
  saturate quickly (`E(24) ~ 0.999`), so `r=32` is genuinely surplus. The language model needs
  ~16 to cover its worst layer; the action expert needs ~8.
- ✅ **The vision tower update is close to full rank.** Capturing 99% of it needs a median of 704
  directions out of 1152. A LoRA adapter would need `r ~ 600-1000` to represent this update, which
  is not a sensible budget. Full-parameter tuning is the only practical option for the vision path;
  this is a property of the update, not a preference.
- 🟡 **`multi_modal_projector` is the only module group that may want more rank.** `E(24) = 0.96`
  and `k@0.99` is 30 for both checkpoints: the energy is spread over ~30 directions and the current
  rank 32 is only two directions above the measured requirement. It is a candidate, not a finding.
- 🟡 Derived Eckart-Young error bounds from the group-mean energies (approximate, since the means
  hide the worst layers): language model at `r=16` ~6.3%, action expert at `r=8` ~8.4%, projector at
  `r=24` ~20%. The projector's bound is an order of magnitude worse than the other groups.

> Important negative reading: a saturated spectrum is a **necessary**, not sufficient, signal that
> more rank would help. A large fraction of the update energy can be fitting noise, or compensating
> for another layer. Frobenius energy is not success rate. The correct acceptance criterion is task
> success, which this repo does not contain.

## 5. Follow-up experiments that were designed but not run

Hypothesis: the projector is under-parameterized at `r=32`, and raising its rank or training it
full-parameter yields a small gain.

| option | change | pro | con |
|---|---|---|---|
| **A** (preferred) | add `multi_modal_projector` to `full_training_modules` | removes the rank cap entirely; ~2 M parameters; the vision path already runs in fp32 | changes the "LoRA vs full" nature of the run |
| **B** | PEFT `rank_pattern` sets the projector to 64 | single-variable change | needs a retrain and a second spectrum pass |

Decision rule (the part that is easy to get wrong): judge by **success rate**, not by the spectrum.
Then re-measure `k@0.99` against the new cap. If it sticks at the new cap, the old 32 really was
binding; if it stays near 30, 32 was already enough.

Two implementation cautions recorded before any such run:

- if the language model and action expert are later reduced to 16 / 8, the `alpha_pattern` must be
  changed with them so that `alpha / r` stays constant; otherwise the scaling change contaminates
  the comparison;
- the projector (~30) and `time_mlp_in` (~20) must keep their own `r=32`; they cannot follow the
  reduction.

## 6. The reduced-rank design that was actually launched ✅

After the measurement, a two-machine A/B was started on the socket task: language model `r=32 -> 16`,
action expert `r=32 -> 8`, projector and time MLP kept at 32, and **`alpha = rank`** so that the PEFT
scaling changes from 0.25 to 1.0. Both versions were retrained from the base weights (a new rank
cannot load an existing r=32 adapter).

The offline parameter check was exact and matched both machines' launch logs:

| item | value |
|---|---|
| modules hit by the regex | 257: LM 126 at r=16, expert 126 at r=8, action_proj + time_mlp + projector 5 at r=32 |
| LoRA parameters | 23,377,920 |
| vision tower full-parameter | 412,442,352 |
| **expected / observed trainable total** | **435,820,272 (436 M)** on both machines ✅ |
| cross-check | the earlier `r=32`, `alpha=8` recipe reported 466 M learnable parameters |

Status: both runs reached step ~100+ without errors; the relative version hit a step-1 OOM at batch
12 per card and was restarted at batch 10. **No final accuracy or success-rate comparison exists for
this reduced-rank design** ❓. It is a validated parameter-count design and an unfinished experiment,
not a result.

## 7. Reproducing the analysis

```bash
# self-test: synthetic rank-8 matrix, recovery of singular values, k@0.99,
# Eckart-Young bound, LoRA scaling, and the numpy safetensors reader
python scripts/rank_spectrum.py --selftest

# on a PEFT adapter plus its base checkpoint (weights are not public, see NOTICE.md)
python scripts/rank_spectrum.py --adapter adapter_model.safetensors \
    --base model.safetensors --lora-alpha 8 --lora-r 32
```

The script reads tensors with a minimal numpy-only safetensors parser (fp32/fp16/bf16), pairs
`lora_A`/`lora_B` tensors, treats `modules_to_save` tensors as full-parameter updates, and prints
`E(8)`, `E(16)`, `E(24)`, `k@0.99` and the rank-24 Eckart-Young bound per module.

## 8. Boundaries

- ❓ The spectrum is task- and data-dependent. These numbers belong to one relative-action socket
  task; another task requires a re-measure.
- ❓ `k@0.99` for LM/expert was measured under a rank-32 cap. `E(24) ~ 0.999` shows the spectrum is
  saturated well before that cap, so the measurement is informative, but the claim "r=16/8 does not
  hurt accuracy" was not verified by a completed reduced-rank run.
- ❓ Whether more projector rank helps is completely untested. The note that proposed it exists
  precisely to be run later.
- ❓ Multi-task use invalidates the single-task conclusion: the source paper's point is that fixed
  ranks interfere across tasks, and the per-task requirement can change.
