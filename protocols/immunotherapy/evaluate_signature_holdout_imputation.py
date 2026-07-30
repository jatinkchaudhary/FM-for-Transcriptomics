#!/usr/bin/env python3
"""Mask measured immune-signature genes and evaluate decoder recovery."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr, spearmanr

from run_provenance_not_response import SIGNATURES

MODELS = ("Txn_Jatin", "BulkFormer_147M")
COHORTS = ("gide", "riaz", "hugo", "rose")


def request(session, url, payload):
    for attempt in range(4):
        try:
            response = session.post(url, json=payload, timeout=(30, 900))
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def correlation(truth, prediction):
    finite = np.isfinite(truth) & np.isfinite(prediction)
    if finite.sum() < 3 or np.std(truth[finite]) == 0 or np.std(prediction[finite]) == 0:
        return np.nan, np.nan
    return pearsonr(truth[finite], prediction[finite]).statistic, spearmanr(
        truth[finite], prediction[finite]
    ).statistic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    signature_lookup = {
        gene: signature for signature, genes in SIGNATURES.items() for gene in genes
    }
    target_union = set(signature_lookup)
    session = requests.Session()
    rows = []
    for model in MODELS:
        for cohort in COHORTS:
            path = args.frozen / f"{cohort}__{model}.npz"
            with np.load(path, allow_pickle=False) as data:
                genes = data["genes"].astype(str)
                sample_ids = data["sample_ids"].astype(str)
                raw = np.asarray(data["raw"], dtype=np.float32)
                measured = np.asarray(data["measured_gene_mask"], dtype=bool)
            targets = np.asarray(
                [measured[i] and gene.upper() in target_union for i, gene in enumerate(genes)]
            )
            if not targets.any():
                continue
            for start in range(0, len(sample_ids), args.batch_size):
                end = min(start + args.batch_size, len(sample_ids))
                values = raw[start:end]
                missing = np.broadcast_to(targets, values.shape)
                result = request(session, f"{args.api}/api/impute", {
                    "model": model,
                    "genes": genes.tolist(),
                    "samples": sample_ids[start:end].tolist(),
                    "matrix": values.T.tolist(),
                    "missing": missing.T.tolist(),
                    "input_scale": "log1p",
                })
                imputed = np.asarray(
                    [[np.nan if value is None else float(value) for value in row]
                     for row in result["imputed"]],
                    dtype=np.float32,
                ).T
                for local_i, sample_id in enumerate(sample_ids[start:end]):
                    for gene_i in np.flatnonzero(targets):
                        rows.append({
                            "model": model, "cohort": cohort.title(), "sample_id": sample_id,
                            "gene": genes[gene_i], "signature": signature_lookup[genes[gene_i].upper()],
                            "truth_log1p_tpm": float(values[local_i, gene_i]),
                            "imputed_log1p_tpm": float(imputed[local_i, gene_i]),
                        })
                print(f"{model} {cohort}: {end}/{len(sample_ids)}", flush=True)
    predictions = pd.DataFrame(rows)
    predictions.to_csv(args.output / "signature_gene_holdout_predictions.csv", index=False)
    summary = []
    for keys, frame in predictions.groupby(["model", "cohort", "signature"], dropna=False):
        pearson, spearman = correlation(
            frame.truth_log1p_tpm.to_numpy(), frame.imputed_log1p_tpm.to_numpy()
        )
        summary.append({
            "model": keys[0], "cohort": keys[1], "signature": keys[2],
            "n_values": len(frame), "n_genes": frame.gene.nunique(),
            "pearson": pearson, "spearman": spearman,
            "mae_log1p_tpm": np.mean(np.abs(frame.truth_log1p_tpm - frame.imputed_log1p_tpm)),
        })
    for model, frame in predictions.groupby("model"):
        pearson, spearman = correlation(
            frame.truth_log1p_tpm.to_numpy(), frame.imputed_log1p_tpm.to_numpy()
        )
        summary.append({
            "model": model, "cohort": "ALL", "signature": "ALL",
            "n_values": len(frame), "n_genes": frame.gene.nunique(),
            "pearson": pearson, "spearman": spearman,
            "mae_log1p_tpm": np.mean(np.abs(frame.truth_log1p_tpm - frame.imputed_log1p_tpm)),
        })
    pd.DataFrame(summary).to_csv(
        args.output / "signature_gene_holdout_accuracy.csv", index=False
    )
    (args.output / "signature_holdout_protocol.json").write_text(json.dumps({
        "models": list(MODELS), "cohorts": list(COHORTS),
        "masking": "All measured genes in the fixed Phase-1 immune signatures were masked together.",
        "input_scale": "log1p(TPM)", "observed_non-target_genes_retained": True,
        "metrics": ["Pearson", "Spearman", "MAE_log1p_TPM"],
        "note": "This tests measured-gene holdout recovery; it does not use response labels.",
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
