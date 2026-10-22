#!/usr/bin/env python3
"""Reproduce the loss-curve figures from the downsampled CSVs in data/.

The training logs behind these curves were parsed once into CSV (see
data/README.md for the downsampling rule). This script only reads those CSVs
and draws the figures, so it runs offline with no access to the training
cluster.

Usage:
    python scripts/plot_loss.py
    python scripts/plot_loss.py --outdir /tmp/figs
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data"
DEFAULT_FIG = ROOT / "figures"

# Neutral labels only: the loss is the training objective in model units and is
# not comparable across tasks or across action representations.
FIGS = {
    "01_tray_default_lora_25k.png": dict(
        csv="tray_absolute_default_lora.csv",
        title="Tray placement, absolute actions, default LoRA (r=16, 38 modules)",
        steps_text="step 100 to 25000, one point per 500 steps",
    ),
    "03_socket_absolute_b12.png": dict(
        csv="socket_absolute_b12.csv",
        title="Socket insertion, absolute actions, LoRA r=32 + full vision tower",
        steps_text="step 20 to 28900, one point per 400 steps",
    ),
}

TWO_PANEL = dict(
    csvs=["kuaidi_rel_30k.csv", "cixilingjian_rel_30k.csv"],
    titles=[
        "Express parcel weighing, relative actions, LoRA r=32 + full vision tower",
        "Part sorting, relative actions, LoRA r=32 + full vision tower",
    ],
    steps_text="step 20 to 30000, one point per 200 steps",
    outname="02_kuaidi_cixilingjian_rel_30k.png",
)


def load(path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            steps.append(int(float(row["step"])))
            losses.append(float(row["loss"]))
    if not steps:
        raise SystemExit(f"empty csv: {path}")
    return steps, losses


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_xlabel("training step")
    ax.set_ylabel("training loss (model units)")


def draw(ax: plt.Axes, steps: list[int], losses: list[float], title: str) -> None:
    ax.plot(steps, losses, color="#1f4e79", linewidth=1.4)
    ax.scatter([steps[-1]], [losses[-1]], color="#c0504d", s=22, zorder=3)
    ax.annotate(
        f"{losses[-1]:.3f} @ {steps[-1]}",
        xy=(steps[-1], losses[-1]),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color="#c0504d",
    )
    ax.set_title(title, fontsize=10, loc="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )

    for outname, spec in FIGS.items():
        steps, losses = load(args.data / spec["csv"])
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        draw(ax, steps, losses, spec["title"])
        style_axis(ax)
        fig.text(0.01, -0.02, spec["steps_text"], fontsize=7.5, color="#555555")
        out = args.outdir / outname
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}  ({steps[-1]} steps, final loss {losses[-1]:.3f})")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    for ax, name, title in zip(axes, TWO_PANEL["csvs"], TWO_PANEL["titles"]):
        steps, losses = load(args.data / name)
        draw(ax, steps, losses, title)
        style_axis(ax)
        ax.set_xlabel("")
    axes[-1].set_xlabel("training step")
    fig.text(0.01, -0.01, TWO_PANEL["steps_text"], fontsize=7.5, color="#555555")
    out = args.outdir / TWO_PANEL["outname"]
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
