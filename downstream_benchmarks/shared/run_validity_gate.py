#!/usr/bin/env python3
"""Embed TCGA and gate native adapters before clinical/integration benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

from sample_adapters import (
    native_embedding,
    pca_embedding,
    raw_embedding,
    save_embedding,
)


def load_tcga(matrix_path: Path, metadata_path: Path, maximum_per_class: int):
    expression = pd.read_parquet(matrix_path)
    metadata = pd.read_csv(metadata_path)
    if expression.shape[0] != len(metadata):
        raise ValueError("TCGA expression and metadata row counts differ")
    sample_col = next(
        col for col in ("file_id", "sample_id", "id") if col in metadata.columns
    )
    label_col = next(
        col for col in ("project_id", "cancer_type", "label") if col in metadata.columns
    )
    metadata = metadata.copy()
    metadata["sample_id"] = metadata[sample_col].astype(str)
    metadata["label"] = metadata[label_col].astype(str)
    keep = []
    for _, group in metadata.groupby("label"):
        keep.extend(group.index[:maximum_per_class].tolist())
    expression = expression.iloc[keep].copy()
    expression.index = metadata.loc[keep, "sample_id"].to_numpy()
    return expression, metadata.loc[keep].reset_index(drop=True)


def gate(embedding: np.ndarray, labels: np.ndarray):
    encoded = LabelEncoder().fit_transform(labels)
    scaled = StandardScaler().fit_transform(embedding)
    silhouette = float(silhouette_score(scaled, encoded, sample_size=min(2000, len(encoded)), random_state=42))
    folds = StratifiedKFold(5, shuffle=True, random_state=42)
    classifier = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
    probability = cross_val_predict(
        make_pipeline(StandardScaler(), classifier),
        embedding,
        encoded,
        cv=folds,
        method="predict_proba",
        n_jobs=5,
    )
    truth = label_binarize(encoded, classes=np.arange(probability.shape[1]))
    auroc = float(roc_auc_score(truth, probability, average="macro", multi_class="ovr"))
    return silhouette, auroc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=["Txn_Jatin", "BRIDGE", "BulkFormer_147M"])
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=250)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expression, metadata = load_tcga(args.matrix, args.metadata, args.max_per_class)
    gate_path = args.output / "validity_gate.csv"
    rows = (
        pd.read_csv(gate_path).to_dict("records")
        if args.skip_baselines and gate_path.exists()
        else []
    )
    methods = [
        (
            model,
            lambda frame, m=model: native_embedding(frame, m, args.runtime, args.config),
        )
        for model in args.models
    ]
    if not args.skip_baselines:
        methods.extend([("PCA64", pca_embedding), ("raw_log1p_TPM", raw_embedding)])
    for name, extractor in methods:
        rows = [row for row in rows if row.get("model") != name]
        row = {"model": name}
        try:
            embedding, provenance = extractor(expression)
            save_embedding(args.output / f"{name}.npz", embedding, expression.index, provenance)
            silhouette, auroc = gate(embedding, metadata["label"].to_numpy())
            row.update(
                status="pass" if silhouette > 0 and auroc >= 0.90 else "fail",
                silhouette=silhouette,
                macro_auroc=auroc,
                dimension=embedding.shape[1],
                error="",
            )
            (args.output / f"{name}.provenance.json").write_text(
                json.dumps(provenance, indent=2), encoding="utf-8"
            )
        except Exception as error:
            row.update(status="fail", silhouette=np.nan, macro_auroc=np.nan, dimension=np.nan, error=repr(error))
        rows.append(row)
        pd.DataFrame(rows).to_csv(gate_path, index=False)
        print(rows[-1], flush=True)


if __name__ == "__main__":
    main()
