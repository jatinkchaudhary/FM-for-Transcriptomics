#!/usr/bin/env python3
"""Aggregate currently available benchmark outputs into presentation-ready tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from masked_benchmark_common import read_protocol, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    protocol = read_protocol(run_dir)
    output = run_dir / "results" / "current"
    output.mkdir(parents=True, exist_ok=True)

    imputation_rows = []
    for path in (run_dir / "results" / "imputation").glob("*/*_summary.json"):
        imputation_rows.append(json.loads(path.read_text(encoding="utf-8")))
    unsupported = run_dir / "results" / "unsupported_imputation_rows.csv"
    if unsupported.exists():
        imputation_rows.extend(pd.read_csv(unsupported).to_dict("records"))
    imputation = pd.DataFrame(imputation_rows)
    if not imputation.empty:
        imputation.to_csv(output / "all_imputation_results.csv", index=False)
        complete = imputation[imputation["supported"].astype(str).str.lower().eq("true")]
        if not complete.empty:
            means = (
                complete.groupby(["model", "dataset"], as_index=False)[
                    ["pcc_global", "spearman_global", "mse", "mae", "auroc_macro", "auprc_macro"]
                ]
                .mean()
                .sort_values(["dataset", "pcc_global"], ascending=[True, False])
            )
            means.to_csv(output / "imputation_means_across_masks.csv", index=False)

    rf_rows = []
    raw = run_dir / "results" / "raw_rf_baseline" / "raw_rf_summary.csv"
    if raw.exists():
        raw_frame = pd.read_csv(raw)
        raw_frame.insert(0, "model", "Raw_expression_control")
        raw_frame.insert(2, "mask_seed", "not_masked")
        rf_rows.extend(raw_frame.to_dict("records"))
    for path in (run_dir / "results" / "masked_rf").glob("*/masked_rf_summary.csv"):
        rf_rows.extend(pd.read_csv(path).to_dict("records"))
    rf = pd.DataFrame(rf_rows)
    if not rf.empty:
        rf.to_csv(output / "all_cancer_rf_results.csv", index=False)

    ready = sorted(path.stem for path in (run_dir / "status").glob("*.READY"))
    metric_complete = sorted(
        path.name.removesuffix(".METRICS_COMPLETE")
        for path in (run_dir / "status").glob("*.METRICS_COMPLETE")
    )
    rf_complete = sorted(
        path.name.removesuffix(".RF_COMPLETE")
        for path in (run_dir / "status").glob("*.RF_COMPLETE")
    )
    summary = {
        "supported_models": [
            name for name, spec in protocol["models"].items() if spec["supported"]
        ],
        "gpu_inference_ready": ready,
        "metrics_complete": metric_complete,
        "masked_rf_complete": rf_complete,
        "imputation_rows": len(imputation),
        "cancer_rf_rows": len(rf),
    }
    write_json(output / "progress_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
