# PROVENANCE

Every number used in the README and in `docs/` is listed here with its source. The goal is that a
reviewer can re-derive or re-check any claim without access to the private material.

## Source naming convention

| tag | meaning |
|---|---|
| `[R]` | file in this repository |
| `[I:note]` | internal engineering note, part of the project export but not published; cited by file name, section, and line |
| `[I:logbook]` | internal training logbook (`训练记录与调参手册.md`), 1548 lines, cited with line numbers |
| `[I:record]` | internal project work record (`工作记录.md`), 2345 lines, cited with line numbers |
| `[I:export]` | extracted run artifacts (`outputs/...` in the project export); citing CSV row numbers means the data row index including the header as row 1 |
| `[D]` | derivation in this document |
| `[X]` | excluded / not included, with reason |

Line numbers are those of the snapshot used to build this repository (2026-10-21). They are given so
the original note can be opened at the exact place; they are not stable across edits of the private
notes.

The raw source of every `data/*.csv` file is the project export's full-resolution CSV, which was
extracted from training logs. Those full CSVs and logs are not published (see `NOTICE.md`). The
downsampling rule and source mapping are in [`data/README.md`](data/README.md).

---

## 1. Repository artifacts and their sources

| artifact | source | method |
|---|---|---|
| `data/tray_absolute_default_lora.csv` | `[I:export] outputs/pi05_loss_curves_20260918/scene2_run1_pi05_loss_history.csv` (250 data rows, run `pi05_scene2_500ep_lora_25k_4gpu_20260918_200102`) | every 5th row + last; `run,step,loss` |
| `data/kuaidi_rel_30k.csv` | `[I:export] outputs/pi05_loss_curves_20261013_kuaidi_cixi/kuaidi_rel_30k_loss_history.csv` (1500 data rows, run `pi05_kuaidi_rel_30k_20261010_172447`) | every 10th row + last |
| `data/cixilingjian_rel_30k.csv` | `[I:export] outputs/pi05_loss_curves_20261013_kuaidi_cixi/cixilingjian_rel_30k_loss_history.csv` (1500 data rows, run `pi05_cixilingjian_rel_30k_20261011_200136`) | every 10th row + last |
| `data/socket_absolute_b12.csv` | `[I:export] outputs/pi05_loss_curves_20260923_socket_abs_b12/socket_abs_b12_loss_history.csv` (1445 data rows) | every 20th row + last |
| `figures/01_tray_default_lora_25k.png` | `data/tray_absolute_default_lora.csv` | `scripts/plot_loss.py` |
| `figures/02_kuaidi_cixilingjian_rel_30k.png` | `data/kuaidi_rel_30k.csv`, `data/cixilingjian_rel_30k.csv` | `scripts/plot_loss.py` |
| `figures/03_socket_absolute_b12.png` | `data/socket_absolute_b12.csv` | `scripts/plot_loss.py` |
| `scripts/rank_spectrum.py` | `[I:note] 各模块LoRA秩实测` section 2 (method); standard Eckart-Young-Mirsky result | re-implemented in numpy, verified by `--selftest` |

---

## 2. README numbers

### 2.1 Project context

| claim in README | value | source |
|---|---|---|
| four official task names | express parcel weighing and placement / part sorting / tray placement / plug connection | `[I:record]` L2241 (official names from the competition document) |
| data provided by the sponsor, not collected or converted by us | - | `[I:record]` L577; also `[I:record]` L585 (isolated data root to avoid polluting the shared dataset) |
| 500 demonstrations per task | 500 episodes | `[I:logbook]` L1548 (`dataset.num_episodes=500`); `[I:note] 归一化对比` L12 ("500 ep"); `[I:note] cixi_socket场景实测` L23 |
| 10 Hz | 10 fps | `[I:note] cixi_socket场景实测` L23; `[I:note] 特征映射机制` L253 (fps=10) |
| 16-D state/action (7+1 per arm) | 16 | `[I:note] cixi_socket场景实测` L25; `[I:note] 特征映射机制` L12, L33 |
| three 848×480 cameras | `head_cam_h`, `wrist_cam_l`, `wrist_cam_r` | `[I:note] cixi_socket场景实测` L26; `[I:note] 特征映射机制` L127-132, L255-257 |
| π0.5 = PaliGemma VLM + Gemma action expert, 4.14 B base | 4,144,691,984 total parameters | `[I:logbook]` L442 (2B PaliGemma trunk, Gemma action expert); L1548 (`num_total_params=4,144,691,984`) |

### 2.2 Parameter budget

| claim | value | source |
|---|---|---|
| default LoRA modules / trainable params | 38 modules; 1,287,168; 0.031% | `[I:logbook]` L442, L472, L985; `[I:note] 提案参数可行性裁定` L214 (table), L217 (cross-check) |
| default LoRA breakdown | action expert 36 (q/v), VLM trunk 0, action projections 2 | `[I:note] 提案参数可行性裁定` L186-225; `[I:note] 特征映射机制` L235-247 |
| cross-check: older r=32 run = exactly 2x | 2,574,336 | `[I:note] 提案参数可行性裁定` L217; `[I:logbook]` L506 |
| explicit regex: 256 modules in the first revision | 126 LM + 126 expert + 4 projection-ish | `[I:logbook]` L671; `[I:note] 提案参数可行性裁定` L214 |
| explicit regex at r=16 (256 modules) trainable LoRA | 26,641,408 | `[I:note] 提案参数可行性裁定` L214 (table) |
| explicit regex after adding the multimodal projector: 257 modules | 126 LM + 126 expert + 5 projection-ish | `[I:note] rk16e8双机对照` L77; `[I:note] 各模块LoRA秩实测` L22 |
| final recipe (256 modules) trainable params | 53,282,816 LoRA + 412,442,352 vision = **465,725,168** (10.1%) | `[I:logbook]` L663-673 (launch log lines quoted); composition L671. The 257-module revision has no separately recorded total; it is not quoted. |
| reduced-rank recipe trainable params | 23,377,920 + 412,442,352 = **435,820,272** | `[I:note] rk16e8双机对照` L77, L79 (both machines' logs) |
| rounding cross-check for the r32 recipe | ~466 M | `[I:logbook]` L662 (`num_learnable_params=465725168 (466M)`) |

### 2.3 Loss curves (all from `data/` unless noted)

| claim | value | source |
|---|---|---|
| tray run #1, absolute, default LoRA, 25k | 0.367 @100 → **0.081** @25k | `[R] data/tray_absolute_default_lora.csv` row 2 (step 100, 0.367) and last row (step 25000, 0.081); `[I:export]` source csv rows 2 and 251 |
| tray run #1 hardware / steps | 4×A100-40G, 25,000 steps | `[I:logbook]` L17 |
| tray run #1 wall clock | 8 h 37 m 53 s | `[I:logbook]` L63, L255 |
| tray run #2, relative-except-claw, default LoRA, 25k | **0.054** @25k | `[I:logbook]` L18; final log line quoted at L271 (`loss:0.054`) |
| tray #1 vs #2 design | same data/base/LoRA/batch/seed/steps; only action representation differs | `[I:logbook]` L41-42, L282-299; `[I:record]` L668 ("the only true single-variable comparison is run#1 vs run#2") |
| express parcel, relative, 30k | 0.430 → **0.015**; min 0.013 | `[R] data/kuaidi_rel_30k.csv` row 2 and last row; min computed over `[R]`; `[I:export]` source csv rows 2, 1501; `[I:record]` L1042-1043 |
| part sorting, relative, 30k | 0.373 → **0.010**; min 0.009 | `[R] data/cixilingjian_rel_30k.csv`; `[I:export]` source csv rows 2, 1501; `[I:record]` L1042-1043 |
| socket insertion, absolute, 28.9k logged | 0.380 → **0.019** | `[R] data/socket_absolute_b12.csv` row 2 (step 20, 0.38) and last row (step 28900, 0.019); `[I:export]` source csv rows 2, 1446 |
| express/part hardware and recipe | 4×RTX 5880; LoRA r=32 + full vision tower | `[I:record]` L1047-1048 (config as stated in the run's own figure title), L1149 (machine spec); loss/step/lr independently derived from the csv |
| express/part LR and batch | peak LR 2.5e-4 @500 → 2.5e-5; global batch 64 (16×4) | `[I:record]` L1044-1045 |
| socket hardware / batch | 4×RTX4090-24G, batch 12 per card | `[I:logbook]` L27 (S5-2) |
| tray #1 absolute-vs-relative pair, only representation differs | - | `[I:logbook]` L41-42, L282-299 |

### 2.4 Rank spectrum

| claim | value | source |
|---|---|---|
| method: `E(k)`, `k@0.99`, Eckart-Young bound | - | `[I:note] 各模块LoRA秩实测` L16; standard result; implemented and self-tested in `[R] scripts/rank_spectrum.py` |
| checkpoints measured | relative-except-claw, vision full, LoRA r=32, steps 12000 and 26000 | `[I:note] 各模块LoRA秩实测` L20-22 |
| vision tower | k@0.99 median 704 / max 1026; full rank 1152 | `[I:note] 各模块LoRA秩实测` L28 |
| language model | k@0.99 median 8-11 / max 25; E(8)/E(16) = 0.978/0.996 | `[I:note] 各模块LoRA秩实测` L29 |
| action expert | k@0.99 median 5-6 / max 15-17; E(8)/E(16) = 0.993/0.999 | `[I:note] 各模块LoRA秩实测` L30 |
| action projections | k@0.99 median 14 / max 19-21; E(8)/E(16) = 0.93/0.99 | `[I:note] 各模块LoRA秩实测` L31 |
| multimodal projector | k@0.99 30 / 30 at cap 32; E(8)/E(16) = 0.74/0.88; E(24) = 0.96 | `[I:note] 各模块LoRA秩实测` L32, L40 |
| "LoRA would need r ≈ 600-1000 for vision" | - | `[I:note] 各模块LoRA秩实测` L39 |
| reduced-rank design: LM 16 / expert 8 / projector+time_mlp 32, alpha = rank | - | `[I:note] rk16e8双机对照` L11, L51-64 |
| reduced-rank design batch OOM and restart | batch 12 → 10 per card | `[I:note] rk16e8双机对照` L22, L86 |
| follow-up projector options were designed, not run | - | `[I:note] 各模块LoRA秩实测` L43-57; readme statement "never run" follows `[I:note]` L76 (❓ completely unverified) |

### 2.5 Deployment

| claim | value | source |
|---|---|---|
| symptom | incoherent actions with an absolute-action π0.5 checkpoint | `[I:note] 部署侧pi05动作不连贯` L4 |
| root cause | missing `_restore_pi_nonpersistent_buffers`; `meta` construction + `to_empty` + non-persistent buffers absent from `state_dict` | `[I:note] 部署侧pi05动作不连贯` L12-14, L19-31 |
| affected buffers | Gemma RoPE `inv_freq`, SigLIP `position_ids` | `[I:note] 部署侧pi05动作不连贯` L14, L49-50 |
| Gemma `inv_freq`: reference vs after `to_empty` | `[1.0000, 0.9306, 0.8660, 0.8058, 0.7499, 0.6978]` vs `[1.4013e-45, 1.4013e-45, 0, 1.3873e-43, 0, 0]` | `[I:note] 部署侧pi05动作不连贯` L40-42 |
| SigLIP `position_ids` after `to_empty` | huge garbage integers, e.g. `2314861380256428066` | `[I:note] 部署侧pi05动作不连贯` L44-46 |
| deployed model config at the time | `type=pi05`, `chunk_size=20`, `n_action_steps=20`, `use_relative_actions=false` | `[I:note] 部署侧pi05动作不连贯` L30-31 |
| fix location | `kuavo_deploy/utils/policy_loader.py` in the deployment fork | `[I:note] 部署侧pi05动作不连贯` L12, L56-74 |
| verification of fix | compile pass, byte-identical to reference package, two-line git diff, buffer equality, rebuild log line | `[I:note] 部署侧pi05动作不连贯` L76-84 |
| offline six-frame MAE before → after | **0.267916 → 0.014516 rad** | `[I:record]` L2083, L2100-2101; figure from the internal diagnostics note `docs/diagnostics/π0.5部署修复说明_2026-09-20.md` section "验证结果" |
| caveat: offline only, robot not connected | - | `[I:record]` L2083 ("明确「没有连接机器人、没有启动真机推理」") |
| end-to-end robot check not recorded | - | `[I:note] 部署侧pi05动作不连贯` L116 |
| residual fork-vs-reference differences | 6 rows | `[I:note] 部署侧pi05动作不连贯` L96-108 |

### 2.6 Normalization comparison

| claim | value | source |
|---|---|---|
| MEAN_STD / default loss ratio at matched steps | 2.04x, 2.00x, 2.03x, 2.03x, 2.03x, 2.04x, 2.00x at steps 100 / 800 / 2000 / 4000 / 6000 / 8000 / 10000 | `[I:note] 归一化对比` L34-43; conclusion restated L78 |
| default column is reproducible | 0.367 @100 ... 0.081 @25000 | `[R] data/tray_absolute_default_lora.csv` |
| MEAN_STD run stopped early | ~10.7k of 60k steps | `[I:note] 归一化对比` L49, L157 |
| QUANTILE10 early ratio | 1.80x at steps 100 and 800; terminated at step 863 | `[I:note] 归一化对比` L55-62, L80 |
| one QUANTILE10 launch was invalid (ran on CPU) | - | `[I:note] 归一化对比` L69-70 |

---

## 3. Facts in `docs/` beyond the README

The four notes contain more detail than the README table. The additional facts and their sources:

| fact | source |
|---|---|
| action pad 16→32, slice back at inference and in the loss; locations `modeling_pi05.py:141, 1210, 1245, 1269` | `[I:note] 特征映射机制` L21-40 |
| no `state_proj` in the base safetensors header (812 tensors); `embed_prefix` has no state argument | `[I:note] 特征映射机制` L44-49 |
| state normalized to [-1,1], discretized to 256 bins, written into the prompt; normalize-before-discretize order | `[I:note] 特征映射机制` L51-69 |
| `input_features: {}` makes dataset camera keys win; base `config.json` not read; `rename_map: {}` correct | `[I:note] 特征映射机制` L71-117 |
| camera slot order 0=head, 1=left wrist, 2=right wrist; only failure mode is "all keys missing" | `[I:note] 特征映射机制` L119-136; convention status 🟡 per L317 |
| `--policy.path` trap | `[I:note] 特征映射机制` L150-156 |
| silent state-dict load failure (bare `except Exception`) | `[I:note] 特征映射机制` L217-233 |
| default target regex has three dead names; 38 modules / 1,287,168 | `[I:note] 特征映射机制` L235-247; `[I:note] 提案参数可行性裁定` L186-225 |
| preprocessor registry drift (`relative_actions_processor` vs `delta_actions_processor`) | `[I:note] 特征映射机制` L277-311 |
| `action.names` dict bug -> 16/16 mask instead of 14/16 | `[I:note] 特征映射机制` L269-275; `[I:logbook]` L413-415; `[I:note] rk16e8双机对照` L70 |
| episode-padding frames not masked (~6%) | `[I:note] 特征映射机制` L249-253 |
| 848×480 letterboxed to 224×224, ~44% padding tokens | `[I:note] 特征映射机制` L255-259 |
| `tokenizer_max_length` is a no-op | `[I:note] 特征映射机制` L265-267 |
| adversarial review of six claims, none falsified | `[I:note] 特征映射机制` L193-211 |
| vision path pinned to fp32; `modules_to_save` semantics; deployment-side `set_adapter` caveat | `[I:note] 提案参数可行性裁定` L126-183 |
| alpha not settable in the old code path (`dataclasses.asdict` filter); single LR group; AdamW absorbs scaling | `[I:note] 提案参数可行性裁定` L58-122 |
| relative mask = 14/16, grippers absolved; `['claw']` vs explicit names equivalent after name fix | `[I:note] 提案参数可行性裁定` L229-238 |
| dataset format is LeRobot v3.0 | `[I:note] cixi_socket场景实测` L21 |
| relative data root isolation: symlink data/videos/images, copy meta, recompute stats, verify source hashes unchanged | `[I:note] rk16e8双机对照` L66-73; `[I:logbook]` L298 (run#2 root), L328-346 (`recompute_stats`, source root verified byte-identical), L371 (14/16 mask check) |
| `relative_dims=14/16 (excluded=2)`, 105,768 chunks on the socket root | `[I:note] rk16e8双机对照` L71 |
| measured rank table per module group and per-checkpoint agreement | `[I:note] 各模块LoRA秩实测` L24-34 |
| LM layer 17 MLP adapters all zero | `[I:note] 各模块LoRA秩实测` L34 |
| E(24) ≈ 0.999 saturation; "energy is not success rate" | `[I:note] 各模块LoRA秩实测` L38-41 |
| reduced-rank parameter check, 257 modules, both machines 435,820,272 | `[I:note] rk16e8双机对照` L75-79 |
| relative reduced-rank run not a strict single variable (batch/hardware also differ) | `[I:note] rk16e8双机对照` L90-94 |
| residual deploy differences (config message, resize_wh, obs_buffer topics, startup behavior, async YAML, client mode) | `[I:note] 部署侧pi05动作不连贯` L96-108 |
| residual diff values quoted in `docs/03`: `async_control_hz: 10.0`, declared shape `(3,848,480)` vs real `(3,480,848)` | `[I:note] 部署侧pi05动作不连贯` L103, L100 |
| base checkpoint size in `docs/02` | 14.47 GB fp32 | `[I:note] 各模块LoRA秩实测` L69 |
| held-out action-MAE protocol exists in team records; raw logs not in this export | `[I:record]` L706-723, L1185-1199 (values) — see section 4 |
| nine static-analysis risks in the integration path | `[I:note] 特征映射机制` L217-316 |

---

## 4. Excluded: numbers that appear in application material but are not used here

These numbers exist in the resume / baseline report / team records. They are **not quoted in the
README** because no reproducible artifact for them is part of this export. They are listed so that
their absence is a documented decision, not an oversight.

| number (as used elsewhere) | status | why it is not used |
|---|---|---|
| training loss `0.09 → 0.006` | `[X]` | `0.006` appears only in the baseline report's table 6-2, from a parallel 10-GPU run whose checkpoints and raw logs are not in our hands (`[I:record]` L642-644, L727, L1206-1209). `0.09` is a rounded description of the 25k default-LoRA run, whose measured terminal loss is 0.081 ([R] `data/`); it is not a matched pair with 0.006. |
| held-out action MAE `0.0101 rad` (and the 0.0115 → 0.0101 sequence) | `[X]` | first-hand values live in a held-out evaluation log on an internal 10-GPU node, produced by a parallel team line; the log and the evaluation script are not part of this export (`[I:record]` L706-707, L1185-1192). The values match the baseline report, but this repository cannot reproduce or publish them. |
| left/right channel consistency `0.004 / 0.016 vs 0.76 rad` | `[X]` | only source is the baseline report's figure 6-3 (exact values 0.0043 / 0.0164 vs 0.7607 / 0.7668). The original output was not found on the internal cluster; `[I:record]` L1204-1205 states it "still has only the baseline report as its source". Rounded values in the resume (`0.004 / 0.016 / 0.76`) are therefore not treated as measured here. |
| on-robot success "3 of 4 tasks completed autonomously" | `[X]` | no success log, score sheet, or evaluation script is part of this export; the only on-robot note is a qualitative description of the tray task in the baseline report (`[I:record]` L605). |
| "5 non-persistent buffers were rebuilt" | `[X]` | this count comes only from a commit-message quote (`[I:record]` L2077), not from the engineering note, which documents two buffer classes (RoPE `inv_freq`, SigLIP `position_ids`). The exact count is framework-version dependent, so the repo says "the buffers" instead of a number. |
| full 30k-step tray run with the final recipe (0.042 → 0.006, holdout MAE, channel consistency) | `[X]` | same parallel 10-GPU line as above; no checkpoint, no raw log in this export (`[I:record]` L642-644). |

The held-out evaluation script and the five-checkpoint evaluation are documented in the team
record, so the work existed; the decision here is about what can be *published with a source in this
repository*, not about whether the work was done.

---

## 5. Derivations

| derived value | derivation |
|---|---|
| 0.031% trainable share | 1,287,168 / 4,144,691,984 = 3.106e-4 |
| 10.1% trainable share | 465,725,168 / 4,609,129,984 = 0.1010 |
| final recipe parameter sum | 53,282,816 + 412,442,352 = 465,725,168 |
| reduced-rank parameter sum | 23,377,920 + 412,442,352 = 435,820,272 |
| `2,574,336 = 2 × 1,287,168` | the old r=32 run on the previous cluster; doubling `r` doubles LoRA parameters at fixed module coverage |
| Eckart-Young bounds quoted in `docs/02` | `sqrt(1-E(k))` at the group-mean energies: LM `r=16`: `sqrt(0.004)` = 6.3%; expert `r=8`: `sqrt(0.007)` = 8.4%; projector `r=24`: `sqrt(0.04)` = 20%. These are approximate because the mean energy hides the worst layers. |
| normalization ratios in `docs/04` | MEAN_STD / default at each matched step from `[I:note] 归一化对比` L34-43: 0.748/0.367, 0.342/0.171, 0.246/0.121, 0.213/0.105, 0.199/0.098, 0.188/0.092, 0.176/0.088 |
| QUANTILE10 ratio | 0.661/0.367 = 1.80x; 0.308/0.171 = 1.80x (`[I:note] 归一化对比` L55-60) |
| run#1 per-1k statistics used to describe the plateau | computed from `[R] data/tray_absolute_default_lora.csv` (downsampled) and `[I:export]` full csv; the full csv has 10 points per 1k steps |
| "lost 1 point per 500 steps" in `figures/` captions | `data/README.md` downsample rule |
| `0.268` in the README | rounding of the measured 0.267916 rad (`[I:record]` L2083) |
| `1.29 M` in `docs/01` | rounding of 1,287,168 (`[I:logbook]` L442) |
| `4.61 B` in `docs/01` | total parameters 4,609,129,984 (`[I:logbook]` L673) |
| step counts quoted in `docs/` (1,000 / 2,000 / 4,000 / 6,000 / 8,000 / 10,000 / 25,000 / 28,900 / 30,000) | `step` column of `[R] data/*.csv`; the last row of each file gives the run's last logged step |
| `k@0.99 = 6` in the selftest | synthetic singular values `[8, 6, 4, 3, 2, 1, 0.5, 0.25]`; cumulative energy through 6 = 130/130.3125 = 0.9976 ≥ 0.99 |
| vision "near full rank" | k@0.99 median 704 out of full rank 1152, i.e. 61% of the spectrum needed for 99% energy |

---

## 6. Desensitization

The original engineering notes were left untouched (they live in a read-only source directory with
automatic commit hooks). The public `docs/` are rewrites. The following classes of information were
removed or replaced before publication:

| class | action |
|---|---|
| internal IPv4 addresses and host aliases of training/deployment machines | removed; replaced by `<training-cluster>`, `<deploy-host>` |
| SSH ports, usernames, host account strings, VPN/tunnel details | removed |
| cloud-GPU rental host names and instance identifiers | removed; replaced by hardware class (e.g. 4×A100-40G) |
| credentials, tokens, API keys, wandb URLs and account names | removed |
| vendor-internal repository paths and absolute dataset/checkpoint paths (`/root/...`, `/home/...`) | removed; replaced by `<dataset-root>`, `<deploy-workspace>` |
| teammate names, initials, personal emails, GitHub accounts, private fork names | removed |
| internal hostnames embedded in CSV extraction metadata | not copied; `data/` keeps only `run,step,loss` columns |
| competition-internal file structure beyond public task names | not reproduced |

Verification performed on this repository:

```bash
# IPv4 addresses
grep -rEn '([0-9]{1,3}\.){3}[0-9]{1,3}' --exclude-dir=.git .
# SSH / account strings / absolute host paths
grep -rEni 'ssh -p|root@|/root/|/home/|/data/' --exclude-dir=.git .
# e-mail addresses and account handles
grep -rEn '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' --exclude-dir=.git .
```

These return no match in the published tree. The literal internal strings that were searched for
are deliberately not reproduced here: publishing a denylist of host names and account handles would
itself disclose them. With the audit text excluded (`--exclude=PROVENANCE.md`), all three commands
return an empty result.

---

## 7. How to re-audit this repository

1. `python scripts/rank_spectrum.py --selftest` - checks the math used in `docs/02`.
2. `python scripts/plot_loss.py` - regenerates `figures/` from `data/`; compare with the committed
   PNGs (bitwise identical on the same matplotlib version, visually identical otherwise).
3. Take any row of section 2 and open the cited note/CSV; the line number points at the original
   statement, and every loss value is a direct data row.
4. For numbers in section 4, the exclusion reason is recorded next to them; do not re-introduce
   them into the README without a new source.
