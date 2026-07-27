#!/usr/bin/env python3
"""Select the OSDR PEFT candidate to benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted(args.sweep_dir.glob("candidates/*/heldout_test_metrics.csv")):
        frame = pd.read_csv(metrics_path)
        model = str(frame.loc[frame["model"] != "Txn_Jatin_original", "model"].iloc[0])
        mse = float(frame[(frame["model"] == model) & (frame["metric"] == "MSE")]["value"].iloc[0])
        auroc = float(frame[(frame["model"] == model) & (frame["metric"] == "AUROC")]["value"].iloc[0])
        f1 = float(frame[(frame["model"] == model) & (frame["metric"] == "F1_macro")]["value"].iloc[0])
        rows.append({"model": model, "path": str(metrics_path.parent), "MSE": mse, "AUROC": auroc, "F1_macro": f1})

    if not rows:
        raise RuntimeError("no candidate metrics found")
    table = pd.DataFrame(rows)
    table["mse_rank"] = table["MSE"].rank(method="min", ascending=True)
    table["auroc_rank"] = table["AUROC"].rank(method="min", ascending=False)
    table["mean_rank"] = (table["mse_rank"] + table["auroc_rank"]) / 2.0
    table = table.sort_values(["mean_rank", "auroc_rank", "mse_rank", "model"]).reset_index(drop=True)
    winner = table.iloc[0].to_dict()

    args.sweep_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.sweep_dir / "candidate_summary.csv", index=False)
    (args.sweep_dir / "winner_selection.json").write_text(
        json.dumps(
            {
                "selection_rule": "minimum mean rank across held-out masked reconstruction MSE (lower better) and flight-vs-ground AUROC (higher better); ties prefer AUROC",
                "winner": winner,
                "candidates": table.to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"winner": winner}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
