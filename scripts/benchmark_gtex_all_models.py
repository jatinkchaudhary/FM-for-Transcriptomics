#!/usr/bin/env python3
"""Benchmark every Studio model on the prepared GTEx whole-gene mask."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(truth) & np.isfinite(prediction)
    x = truth[valid].astype(np.float64)
    y = prediction[valid].astype(np.float64)
    if not len(x):
        return {"n_values": 0, "mse": np.nan, "mae": np.nan, "pearson": np.nan}
    correlation = np.corrcoef(x, y)[0, 1] if np.std(x) and np.std(y) else np.nan
    return {
        "n_values": int(len(x)),
        "mse": float(np.mean((y - x) ** 2)),
        "mae": float(np.mean(np.abs(y - x))),
        "pearson": float(correlation),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True)
    parser.add_argument("--masked", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = args.api.rstrip("/")

    masked = pd.read_csv(args.masked, index_col=0, na_values=["NA"])
    truth = pd.read_csv(args.truth, index_col=0).loc[masked.index, masked.columns]
    missing = masked.isna().to_numpy()
    values = masked.fillna(0).to_numpy(dtype=np.float32)
    masked_gene_rows = np.flatnonzero(missing.all(axis=1))
    models = requests.get(api + "/api/models", timeout=30).json()["models"]
    predictions: dict[str, np.ndarray] = {}
    supported_rows = []
    all_rows = []

    for model in models:
        name = model["id"]
        if not model["imputation_supported"]:
            all_rows.append(
                {
                    "model": name,
                    "label": model["label"],
                    "status": "NaN: embedding-only model has no expression decoder",
                    "native_masked_genes": 0,
                    "native_mse": np.nan,
                    "native_mae": np.nan,
                    "native_pearson": np.nan,
                    "elapsed_seconds": 0,
                }
            )
            continue
        payload = {
            "model": name,
            "genes": masked.index.tolist(),
            "samples": masked.columns.tolist(),
            "matrix": values.tolist(),
            "missing": missing.tolist(),
            "input_scale": "log1p",
        }
        started = time.time()
        print(f"[START] {name}", flush=True)
        response = requests.post(api + "/api/impute", json=payload, timeout=1800)
        elapsed = time.time() - started
        if response.status_code != 200:
            message = response.json().get("message", response.text[:300])
            all_rows.append(
                {
                    "model": name,
                    "label": model["label"],
                    "status": f"failed: {message}",
                    "native_masked_genes": 0,
                    "native_mse": np.nan,
                    "native_mae": np.nan,
                    "native_pearson": np.nan,
                    "elapsed_seconds": elapsed,
                }
            )
            print(f"[FAILED] {name}: {message}", flush=True)
            continue
        result = response.json()
        completed = np.asarray(
            [
                [np.nan if value is None else value for value in row]
                for row in result["imputed"]
            ],
            dtype=np.float32,
        )
        prediction = completed[masked_gene_rows]
        native_truth = truth.to_numpy(dtype=np.float32)[masked_gene_rows]
        native = metrics(native_truth, prediction)
        available_rows = np.isfinite(prediction).all(axis=1)
        predictions[name] = prediction
        supported_rows.append((name, available_rows))
        row = {
            "model": name,
            "label": model["label"],
            "status": "complete",
            "native_masked_genes": int(available_rows.sum()),
            "native_mse": native["mse"],
            "native_mae": native["mae"],
            "native_pearson": native["pearson"],
            "elapsed_seconds": elapsed,
            "matched_genes": result.get("matched_genes"),
            "model_gene_count": result.get("model_gene_count"),
        }
        all_rows.append(row)
        np.savez_compressed(
            args.output_dir / f"{name}_masked_predictions.npz",
            genes=masked.index.to_numpy()[masked_gene_rows],
            samples=masked.columns.to_numpy(),
            prediction=prediction,
            truth=native_truth,
        )
        print("[DONE] " + json.dumps(row), flush=True)

    common = np.ones(len(masked_gene_rows), dtype=bool)
    for _, available in supported_rows:
        common &= available
    common_truth = truth.to_numpy(dtype=np.float32)[masked_gene_rows][common]
    row_by_model = {row["model"]: row for row in all_rows}
    sample_rows = []
    for name, _ in supported_rows:
        prediction = predictions[name][common]
        common_result = metrics(common_truth, prediction)
        row_by_model[name].update(
            {
                "common_masked_genes": int(common.sum()),
                "common_mse": common_result["mse"],
                "common_mae": common_result["mae"],
                "common_pearson": common_result["pearson"],
            }
        )
        for sample_index, sample in enumerate(masked.columns):
            result = metrics(common_truth[:, sample_index], prediction[:, sample_index])
            sample_rows.append({"model": name, "sample": sample, **result})

    summary = pd.DataFrame(all_rows)
    summary.to_csv(args.output_dir / "gtex_all_model_summary.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(
        args.output_dir / "gtex_all_model_sample_metrics.csv", index=False
    )
    pd.DataFrame(
        {"gene": masked.index.to_numpy()[masked_gene_rows][common]}
    ).to_csv(args.output_dir / "common_masked_genes.csv", index=False)

    plotted = summary[summary["status"].eq("complete")].sort_values("common_mse")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(plotted["label"], plotted["common_mse"], color="#147d92")
    axes[0].invert_yaxis()
    axes[0].set(xlabel="MSE (lower is better)", title="GTEx whole-gene imputation")
    axes[1].barh(plotted["label"], plotted["common_pearson"], color="#5669a8")
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Pearson r (higher is better)", title="Masked-value correlation")
    fig.tight_layout()
    fig.savefig(args.output_dir / "gtex_all_model_comparison.png", dpi=220)
    plt.close(fig)
    manifest = {
        "samples": len(masked.columns),
        "input_genes": len(masked),
        "masked_genes": len(masked_gene_rows),
        "common_masked_genes": int(common.sum()),
        "models_registered": len(models),
        "decoders_completed": int(summary["status"].eq("complete").sum()),
        "embedding_only_nan": int(summary["status"].str.startswith("NaN").sum()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
