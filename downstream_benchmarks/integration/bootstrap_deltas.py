#!/usr/bin/env python3
"""Paired stratified bootstrap for sample-level integration neighborhood scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def local_scores(embedding, metadata):
    x = StandardScaler().fit_transform(embedding)
    neighbors = NearestNeighbors(n_neighbors=31).fit(x).kneighbors(return_distance=False)[:, 1:]
    study = metadata["study"].to_numpy()
    tissue = metadata["tissue"].to_numpy()
    batch, biology = [], []
    for row in neighbors:
        _, counts = np.unique(study[row], return_counts=True)
        probability = counts / counts.sum()
        ilisi = 1 / np.sum(probability**2)
        batch.append(ilisi - 1)
        _, counts = np.unique(tissue[row], return_counts=True)
        probability = counts / counts.sum()
        clisi = 1 / np.sum(probability**2)
        biology.append(1 / clisi)
    batch = np.asarray(batch)
    biology = np.asarray(biology)
    return batch, biology, 0.4 * batch + 0.6 * biology


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    metadata = pd.read_csv(args.result_dir / "cohort_metadata.csv")
    methods = {}
    for name in ("Txn_Jatin", "BulkFormer_147M"):
        methods[name] = local_scores(
            np.load(args.result_dir / f"{name}.npz", allow_pickle=False)["embeddings"],
            metadata,
        )
    strata_labels = metadata["study"].astype(str) + "_" + metadata["tissue"].astype(str)
    strata = [np.flatnonzero(strata_labels.to_numpy() == value) for value in strata_labels.unique()]
    rng = np.random.default_rng(42)
    names = ("local_batch_mixing", "local_biology_preservation", "local_combined")
    rows = []
    for metric_index, metric in enumerate(names):
        left, right = methods["Txn_Jatin"][metric_index], methods["BulkFormer_147M"][metric_index]
        boot = []
        for _ in range(2000):
            index = np.concatenate([rng.choice(group, len(group), replace=True) for group in strata])
            boot.append(float(np.mean(left[index] - right[index])))
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "reference": "BulkFormer_147M",
                "method": "Txn_Jatin",
                "paired_delta": float(np.mean(left - right)),
                "ci_low": low,
                "ci_high": high,
                "bootstrap_replicates": 2000,
                "seed": 42,
            }
        )
    pd.DataFrame(rows).to_csv(args.result_dir / "paired_delta_vs_espresso.csv", index=False)


if __name__ == "__main__":
    main()
