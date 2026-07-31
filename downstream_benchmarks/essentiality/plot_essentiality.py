#!/usr/bin/env python3
"""Create manuscript-oriented DepMap essentiality figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.results / "essentiality_metrics.csv").sort_values("auprc")
    labels = data["model"].str.replace("Txn_Jatin", "ESPRESSO", regex=False)
    colors = [
        "#0072B2" if model == "Txn_Jatin" else
        "#56B4E9" if model == "Txn_Jatin_contextual" else
        "#009E73" if model == "BRIDGE" else
        "#D55E00" if model.startswith("BulkFormer") else
        "#777777" if model in {"Mean_expression", "Gene_length"} else
        "#CC79A7"
        for model in data["model"]
    ]
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.8), sharey=True)
    for axis, metric, title in [
        (axes[0], "auroc", "AUROC"),
        (axes[1], "auprc", "AUPRC"),
    ]:
        low = data[metric] - data[f"{metric}_ci_low"]
        high = data[f"{metric}_ci_high"] - data[metric]
        axis.barh(y, data[metric], color=colors, alpha=0.9)
        axis.errorbar(
            data[metric], y, xerr=np.vstack([low, high]), fmt="none",
            ecolor="#222222", capsize=2, linewidth=1,
        )
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_xlabel(f"{title} (95% bootstrap CI)")
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlim(0.45, 0.96)
    axes[1].set_xlim(0.10, 0.80)
    fig.suptitle(
        "Cold-gene prediction of DepMap common essentiality",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5, 0.015,
        "13,342 shared screened genes; 5-fold gene-disjoint nested CV; 2,000 bootstrap replicates",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    for suffix in ("png", "pdf"):
        fig.savefig(args.results / f"essentiality_comparison.{suffix}", dpi=300)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
