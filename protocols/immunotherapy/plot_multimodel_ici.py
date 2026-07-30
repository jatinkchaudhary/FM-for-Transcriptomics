#!/usr/bin/env python3
"""Plot primary multi-model immunotherapy results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    results = args.results.resolve()
    frame = pd.read_csv(results / "multimodel_loco_summary.csv")
    frame = frame[
        frame["method"].eq("Raw_common_measured")
        | frame["method"].str.endswith("__completed")
    ].copy()
    labels = {
        "Raw_common_measured": "Raw",
        "Txn_Jatin__completed": "Txn_Jatin",
        "Txn_Jatin_OSDR_LoRA__completed": "Txn OSDR LoRA",
        "BRIDGE__completed": "BRIDGE",
        "BulkFormer_37M__completed": "BulkFormer 37M",
        "BulkFormer_50M__completed": "BulkFormer 50M",
        "BulkFormer_93M__completed": "BulkFormer 93M",
        "BulkFormer_127M__completed": "BulkFormer 127M",
        "BulkFormer_147M__completed": "BulkFormer 147M",
    }
    frame["label"] = frame["method"].map(labels)
    frame = frame.sort_values("macro_auroc")
    colors = ["#087E8B" if value == "Raw" else "#4C78A8" for value in frame["label"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), sharey=True)
    for axis, column, title in (
        (axes[0], "macro_auroc", "Macro AUROC"),
        (axes[1], "macro_auprc", "Macro AUPRC"),
        (axes[2], "macro_brier", "Macro Brier (lower is better)"),
    ):
        axis.barh(frame["label"], frame[column], color=colors)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
        for index, value in enumerate(frame[column]):
            axis.text(value, index, f" {value:.3f}", va="center", fontsize=8)
    axes[0].set_xlim(0.60, 0.66)
    axes[1].set_xlim(0.48, 0.52)
    axes[2].set_xlim(0.24, 0.25)
    fig.suptitle("Nested leave-one-cohort-out immunotherapy benchmark", fontsize=15)
    fig.tight_layout()
    fig.savefig(results / "multimodel_primary_comparison.png", dpi=240)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
