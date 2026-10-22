# NOTICE

## What is not in this repository

**Dataset.** The demonstrations used in this work were provided by the challenge sponsor (Leju
Robotics) under the competition terms. They are not redistributed here. No data files, no dataset
statistics, and no dataset screenshots are included. Anything describing dataset structure in the
docs is limited to what is needed to explain the model-side integration.

**Model weights.** The π0.5 base checkpoint, the LoRA adapters, and the fine-tuned checkpoints are
not published. No weight files, no adapter files, and no checkpoint directories are included.

**Training and deployment logs.** Raw logs, launch scripts, cluster connection details, and
deployment host information are not published. The `data/` CSVs are downsampled extracts; see
`data/README.md`.

**Third-party and team code.** The training and deployment code belongs to the team and to the
frameworks it builds on. No third-party source code is redistributed here. The scripts in
`scripts/` are written for this repository and have no dependency on the private codebase. The
internal repository names, hostnames, addresses, ports, credentials, and teammate identities that
appeared in the original engineering notes have been removed or replaced with placeholders.

## Third-party components referenced, not redistributed

This work builds on LeRobot, PEFT, PyTorch, and the π0.5 / openpi model family. Their names, code,
and licenses belong to their respective authors and are referenced here only for context. No
license from those projects is claimed or extended by this repository.

## Status of the numbers

Every quantitative claim in the README and in `docs/` is mapped to a source file, note, section, or
derivation in `PROVENANCE.md`. Claims that could not be traced are listed there as *not included*
and do not appear in the README. The evidence tags used in the docs are:

- ✅ measured in a run or direct instrumentation,
- 🟡 strong inference from code or data, not directly validated,
- ❓ unverified hypothesis or open item.

## License

The write-ups, figures and data extracts are licensed under CC BY 4.0; the scripts in `scripts/`
are licensed under the MIT License. See [`LICENSE`](LICENSE) for the exact scope of each. The
licenses cover the material in this repository only. They do not extend to the dataset, the model
weights, the team's training and deployment code, or any third-party component named above, none of
which is redistributed here.

## Contact

Wenke Huang, Harbin Institute of Technology, Shenzhen. Open an issue in this repository.
