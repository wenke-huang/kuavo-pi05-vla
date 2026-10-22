# data/

Downsampled training-loss histories. One row per logged point, columns `run,step,loss`.

The source CSVs were extracted from the training logs (`log_freq = 100` for the tray run,
`log_freq = 20` for the 30k runs, `log_freq = 20` for the socket run). This directory keeps a
reduced version so the repository stays small and the figures in `../figures/` can be regenerated
offline.

| file | source run | source rows | kept | downsample rule |
|---|---|---:|---:|---|
| `tray_absolute_default_lora.csv` | tray placement, absolute actions, default LoRA r=16, 25k steps | 250 | 51 | every 5th row (1 point per 500 steps) + last row |
| `kuaidi_rel_30k.csv` | express parcel weighing, relative-except-claw, r=32 + vision full, 30k steps | 1,500 | 151 | every 10th row (1 point per 200 steps) + last row |
| `cixilingjian_rel_30k.csv` | part sorting, relative-except-claw, r=32 + vision full, 30k steps | 1,500 | 151 | every 10th row (1 point per 200 steps) + last row |
| `socket_absolute_b12.csv` | socket insertion, absolute actions, r=32 + vision full, batch 12, 28.9k logged steps | 1,445 | 74 | every 20th row (1 point per 400 steps) + last row |

The rule is always: keep row `i` when `i % stride == 0`, and always keep the final row, so the
visible start and end values are the true start and end of each run. Downsampling never
interpolates and never averages, so no value in these files is synthesized.

The full-resolution CSVs stay in the project's internal export, together with the raw log excerpts.
They are not published because the runs are part of a competition collaboration and the logs contain
machine and workspace details that are not ours to publish. See `../NOTICE.md`. `../PROVENANCE.md`
maps every number used in the README and docs back to the source file, section, or derivation.

Loss units are model units and are **not comparable across tasks or across action
representations**. Do not overlay these curves to rank runs.
