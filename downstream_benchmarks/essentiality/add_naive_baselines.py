#!/usr/bin/env python3
"""Add mean-expression and gene-length controls on the frozen gene folds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.provenance import sha256
from shared.statistics import stratified_metric_bootstrap, stratified_paired_bootstrap

from run_essentiality import parse_entrez


def oof_univariate(values, labels, fold_ids):
    scores = np.full(len(labels), np.nan)
    for fold in sorted(np.unique(fold_ids)):
        train = fold_ids != fold
        test = fold_ids == fold
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
                random_state=42,
            ),
        )
        model.fit(values[train, None], labels[train])
        scores[test] = model.predict_proba(values[test, None])[:, 1]
    return scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--gene-info", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()

    predictions = pd.read_csv(args.results / "out_of_fold_predictions.csv")
    reference = predictions.loc[predictions["model"].eq("Txn_Jatin")].sort_values(
        "entrez"
    )
    genes = reference["entrez"].to_numpy(dtype=int)
    labels = reference["label"].to_numpy(dtype=int)
    folds = reference["fold"].to_numpy(dtype=int)

    expression = pd.read_csv(args.expression, index_col=0)
    expression_genes = np.asarray(
        [parse_entrez(column) for column in expression.columns], dtype=object
    )
    valid = np.asarray([value is not None for value in expression_genes])
    means = np.nanmean(expression.iloc[:, valid].to_numpy(dtype=np.float32), axis=0)
    mean_by_entrez = dict(zip(expression_genes[valid].astype(int), means))

    gene_info = pd.read_csv(args.gene_info)
    symbol_to_entrez = {}
    for column in expression.columns:
        entrez = parse_entrez(column)
        if entrez is not None:
            symbol_to_entrez[str(column).rsplit(" (", 1)[0].upper()] = entrez
    length_by_entrez = {
        symbol_to_entrez[str(row.gene_symbol).upper()]: float(row.gene_length)
        for row in gene_info.itertuples()
        if str(row.gene_symbol).upper() in symbol_to_entrez
        and pd.notna(row.gene_length)
    }
    missing_expression = [gene for gene in genes if gene not in mean_by_entrez]
    missing_length = [gene for gene in genes if gene not in length_by_entrez]
    expression_median = float(np.nanmedian(list(mean_by_entrez.values())))
    length_median = float(np.nanmedian(list(length_by_entrez.values())))

    controls = {
        "Mean_expression": np.asarray(
            [mean_by_entrez.get(gene, expression_median) for gene in genes]
        ),
        "Gene_length": np.log1p(
            np.asarray(
                [length_by_entrez.get(gene, length_median) for gene in genes],
                dtype=float,
            )
        ),
    }
    metrics = pd.read_csv(args.results / "essentiality_metrics.csv")
    deltas = pd.read_csv(args.results / "paired_deltas_vs_espresso.csv")
    espresso_scores = reference["probability"].to_numpy(dtype=float)
    new_prediction_rows = []
    new_metrics = []
    new_deltas = []
    for name, values in controls.items():
        scores = oof_univariate(values, labels, folds)
        estimate = stratified_metric_bootstrap(
            labels, scores, replicates=args.bootstraps, seed=42
        )
        new_metrics.append(
            {
                "model": name,
                "genes": len(genes),
                "positives": int(labels.sum()),
                "negatives": int((labels == 0).sum()),
                **estimate,
                "selected_C_by_fold": "1.0|1.0|1.0|1.0|1.0",
            }
        )
        new_deltas.append(
            {
                "reference": "Txn_Jatin",
                "comparator": name,
                **stratified_paired_bootstrap(
                    labels,
                    espresso_scores,
                    scores,
                    replicates=args.bootstraps,
                    seed=42,
                ),
            }
        )
        new_prediction_rows.extend(
            {
                "entrez": gene,
                "label": label,
                "model": name,
                "probability": score,
                "fold": fold,
            }
            for gene, label, score, fold in zip(genes, labels, scores, folds)
        )
    metrics = pd.concat([metrics, pd.DataFrame(new_metrics)], ignore_index=True)
    deltas = pd.concat([deltas, pd.DataFrame(new_deltas)], ignore_index=True)
    predictions = pd.concat(
        [predictions, pd.DataFrame(new_prediction_rows)], ignore_index=True
    )
    metrics.to_csv(args.results / "essentiality_metrics.csv", index=False)
    deltas.to_csv(args.results / "paired_deltas_vs_espresso.csv", index=False)
    predictions.to_csv(args.results / "out_of_fold_predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "Mean_expression",
                "source": str(args.expression),
                "source_sha256": sha256(args.expression),
                "dimension": 1,
                "missing_median_imputed": len(missing_expression),
            },
            {
                "model": "Gene_length",
                "source": str(args.gene_info),
                "source_sha256": sha256(args.gene_info),
                "dimension": 1,
                "missing_median_imputed": len(missing_length),
            },
        ]
    ).to_csv(args.results / "naive_baseline_provenance.csv", index=False)
    print(metrics.sort_values("auroc", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
