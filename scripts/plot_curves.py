# scripts/plot_curves.py
"""Render runs/curves.png from runs/loss.csv (+ runs/eval.csv if present)."""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> dict:
    cols: dict = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return cols


def main() -> None:
    runs = Path("runs")
    loss = read_csv(runs / "loss.csv")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    steps = loss.get("step", [])
    axes[0].plot(steps, loss.get("loss", []), linewidth=0.7)
    axes[0].set_ylabel("total loss")
    axes[0].set_title("tmt-torch training")
    for key in ("l_var", "l_pred", "l_ce", "l_stop"):
        if key in loss:
            axes[1].plot(steps, loss[key], linewidth=0.7, label=key)
    axes[1].legend(fontsize=8)
    axes[1].set_ylabel("components")
    if (runs / "eval.csv").exists():
        ev = read_csv(runs / "eval.csv")
        axes[2].plot(ev.get("step", []), ev.get("bpb", []), "o-",
                     markersize=3, label="bpb")
        axes[2].legend(fontsize=8)
    else:
        axes[2].text(0.5, 0.5, "no eval.csv yet", ha="center")
    axes[2].set_ylabel("held-out bpb")
    axes[2].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(runs / "curves.png", dpi=100)
    print(f"wrote {runs / 'curves.png'}")


if __name__ == "__main__":
    main()
