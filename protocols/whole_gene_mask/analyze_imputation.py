#!/usr/bin/env python3
"""Calculate masked-expression metrics for one completed model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from masked_benchmark_common import load_mask, read_protocol, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def truth_for_mask(protocol: dict, dataset: str, genes: list[str]) -> np.ndarray:
    if dataset == "TCGA":
        frame = pd.read_parquet(protocol["datasets"]["TCGA"]["matrix"], columns=genes)
        return np.log1p(frame.to_numpy(dtype=np.float32, copy=False))
    payload = np.load(protocol["datasets"]["OSDR"]["matrix"], allow_pickle=True)
    source_genes = [str(value).upper() for value in payload["genes"]]
    source_index = {gene: index for index, gene in enumerate(source_genes)}
    indices = [source_index[gene] for gene in genes]
    return payload["X"][:, indices].astype(np.float32, copy=False)


def pearson_axis(truth: np.ndarray, prediction: np.ndarray, axis: int) -> np.ndarray:
    truth_centered = truth - truth.mean(axis=axis, keepdims=True)
    prediction_centered = prediction - prediction.mean(axis=axis, keepdims=True)
    numerator = np.sum(truth_centered * prediction_centered, axis=axis)
    denominator = np.sqrt(
        np.sum(truth_centered * truth_centered, axis=axis)
        * np.sum(prediction_centered * prediction_centered, axis=axis)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def calculate_metrics(
    model: str,
    dataset: str,
    seed: int,
    genes: list[str],
    truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    if prediction.shape != truth.shape:
        raise RuntimeError(f"Prediction shape {prediction.shape} != truth shape {truth.shape}")
    finite = np.isfinite(truth) & np.isfinite(prediction)
    if not finite.all():
        truth_flat = truth[finite]
        prediction_flat = prediction[finite]
    else:
        truth_flat = truth.ravel()
        prediction_flat = prediction.ravel()
    pcc_global = float(np.corrcoef(truth_flat, prediction_flat)[0, 1])
    spearman_global = float(spearmanr(truth_flat, prediction_flat).statistic)
    pcc_gene = pearson_axis(truth, prediction, axis=0)
    pcc_sample = pearson_axis(truth, prediction, axis=1)
    threshold = float(np.log(2.0))
    aurocs = np.full(len(genes), np.nan)
    auprcs = np.full(len(genes), np.nan)
    for index in range(len(genes)):
        binary = truth[:, index] >= threshold
        if binary.any() and (~binary).any():
            aurocs[index] = roc_auc_score(binary, prediction[:, index])
            auprcs[index] = average_precision_score(binary, prediction[:, index])
    per_gene = pd.DataFrame(
        {
            "model": model,
            "dataset": dataset,
            "mask_seed": seed,
            "gene_symbol": genes,
            "pcc": pcc_gene,
            "mse": np.mean((prediction - truth) ** 2, axis=0),
            "mae": np.mean(np.abs(prediction - truth), axis=0),
            "auroc": aurocs,
            "auprc": auprcs,
            "expressed_prevalence": np.mean(truth >= threshold, axis=0),
        }
    )
    summary = {
        "model": model,
        "dataset": dataset,
        "mask_seed": seed,
        "supported": True,
        "status": "complete",
        "samples": int(truth.shape[0]),
        "masked_genes": int(truth.shape[1]),
        "pcc_global": pcc_global,
        "pcc_gene_macro": float(np.nanmean(pcc_gene)),
        "pcc_sample_macro": float(np.nanmean(pcc_sample)),
        "spearman_global": spearman_global,
        "mse": float(np.mean((prediction_flat - truth_flat) ** 2)),
        "mae": float(np.mean(np.abs(prediction_flat - truth_flat))),
        "auroc_macro": float(np.nanmean(aurocs)),
        "auprc_macro": float(np.nanmean(auprcs)),
        "auc_eligible_genes": int(np.isfinite(aurocs).sum()),
        "reason": "",
    }
    return summary, per_gene


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    protocol = read_protocol(run_dir)
    output = run_dir / "results" / "imputation" / args.model
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dataset in ("TCGA", "OSDR"):
        for seed in protocol["masking"]["seeds"]:
            destination = output / f"{dataset}_seed_{seed}_summary.json"
            if destination.exists():
                summaries.append(json.loads(destination.read_text(encoding="utf-8")))
                continue
            genes = load_mask(run_dir, seed)
            truth = truth_for_mask(protocol, dataset, genes)
            prediction_path = (
                run_dir
                / "predictions"
                / args.model
                / dataset
                / f"seed_{seed}_predictions.npy"
            )
            prediction = np.load(prediction_path, mmap_mode="r")
            summary, per_gene = calculate_metrics(
                args.model, dataset, seed, genes, truth, prediction
            )
            per_gene.to_csv(
                output / f"{dataset}_seed_{seed}_per_gene.csv.gz",
                index=False,
                compression="gzip",
            )
            write_json(destination, summary)
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    pd.DataFrame(summaries).to_csv(output / "imputation_summary.csv", index=False)
    (run_dir / "status" / f"{args.model}.METRICS_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
