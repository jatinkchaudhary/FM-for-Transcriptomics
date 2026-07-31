#!/usr/bin/env python3
"""PVCA-style weighted PC variance partition for immunotherapy representations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from run_provenance_not_response import load_inputs, signature_features


def one_hot(values):
    levels = sorted(np.unique(values))
    return np.column_stack([(values == level).astype(float) for level in levels[1:]])


def r_squared(y, design):
    fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    total = np.square(y - y.mean()).sum()
    return 1 - np.square(y - fitted).sum() / total if total else 0.0


def partition(matrix, cohort, cancer, response):
    finite = np.isfinite(matrix).all(axis=0)
    matrix = matrix[:, finite]
    scaled = StandardScaler().fit_transform(matrix)
    pca = PCA(
        n_components=min(20, len(matrix) - 1, matrix.shape[1]), random_state=42
    ).fit(scaled)
    scores = pca.transform(scaled)
    weights = pca.explained_variance_ratio_
    weights /= weights.sum()
    intercept = np.ones((len(matrix), 1))
    cohort_x = np.column_stack([intercept, one_hot(cohort)])
    cohort_cancer_x = np.column_stack([cohort_x, one_hot(cancer)])
    full_x = np.column_stack([cohort_cancer_x, response])
    rows = []
    for component, (values, weight) in enumerate(zip(scores.T, weights), 1):
        cohort_r2 = r_squared(values, cohort_x)
        cancer_r2 = max(0.0, r_squared(values, cohort_cancer_x) - cohort_r2)
        full_r2 = r_squared(values, full_x)
        response_r2 = max(0.0, full_r2 - r_squared(values, cohort_cancer_x))
        rows.append((component, weight, cohort_r2, cancer_r2, response_r2, max(0, 1 - full_r2)))
    values = np.asarray([row[1:] for row in rows])
    return {
        "components": len(rows),
        "cohort_fraction": np.sum(values[:, 0] * values[:, 1]),
        "cancer_after_cohort_fraction": np.sum(values[:, 0] * values[:, 2]),
        "response_after_cohort_cancer_fraction": np.sum(values[:, 0] * values[:, 3]),
        "residual_fraction": np.sum(values[:, 0] * values[:, 4]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    genes, _, metadata, raw, masks, completed, _ = load_inputs(args.frozen, args.clinical)
    common = np.logical_and.reduce(masks)
    limited, _ = signature_features(raw, genes, common)
    representations = {
        "Raw_common": raw[:, common],
        **completed,
        **{f"Assay_limited_{key}": value for key, value in limited.items()},
    }
    for model in ("Txn_Jatin", "BulkFormer_147M"):
        scores, _ = signature_features(completed[model], genes)
        representations.update({
            f"Harmonized_{model}_{key}": value for key, value in scores.items()
        })
    rows = []
    for name, matrix in representations.items():
        rows.append({
            "representation": name,
            **partition(
                matrix, metadata.cohort.to_numpy(str),
                metadata.cancer_type.to_numpy(str), metadata.label.to_numpy(float),
            ),
            "note": "Cancer is nested within cohort and has zero identifiable incremental variance after cohort.",
        })
    pd.DataFrame(rows).to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
