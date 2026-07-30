"""Shared paired uncertainty estimates for downstream benchmarks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def stratified_paired_bootstrap(
    labels,
    left_scores,
    right_scores,
    *,
    replicates: int = 2000,
    seed: int = 42,
):
    labels = np.asarray(labels, dtype=int)
    left_scores = np.asarray(left_scores, dtype=float)
    right_scores = np.asarray(right_scores, dtype=float)
    classes = np.unique(labels)
    if classes.tolist() != [0, 1]:
        raise ValueError("paired classification bootstrap requires binary labels")
    rng = np.random.default_rng(seed)
    deltas = {"auroc": [], "auprc": []}
    class_indices = [np.flatnonzero(labels == value) for value in classes]
    for _ in range(replicates):
        index = np.concatenate(
            [rng.choice(group, len(group), replace=True) for group in class_indices]
        )
        sampled = labels[index]
        deltas["auroc"].append(
            roc_auc_score(sampled, left_scores[index])
            - roc_auc_score(sampled, right_scores[index])
        )
        deltas["auprc"].append(
            average_precision_score(sampled, left_scores[index])
            - average_precision_score(sampled, right_scores[index])
        )
    result = {}
    for metric, values in deltas.items():
        values = np.asarray(values)
        result[f"delta_{metric}"] = (
            (roc_auc_score(labels, left_scores) - roc_auc_score(labels, right_scores))
            if metric == "auroc"
            else (
                average_precision_score(labels, left_scores)
                - average_precision_score(labels, right_scores)
            )
        )
        result[f"{metric}_ci_low"] = float(np.quantile(values, 0.025))
        result[f"{metric}_ci_high"] = float(np.quantile(values, 0.975))
    result["bootstrap_replicates"] = replicates
    result["seed"] = seed
    return result


def stratified_metric_bootstrap(
    labels,
    scores,
    *,
    replicates: int = 2000,
    seed: int = 42,
):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == value) for value in (0, 1)]
    values = {"auroc": [], "auprc": []}
    for _ in range(replicates):
        index = np.concatenate(
            [rng.choice(group, len(group), replace=True) for group in class_indices]
        )
        sampled = labels[index]
        values["auroc"].append(roc_auc_score(sampled, scores[index]))
        values["auprc"].append(average_precision_score(sampled, scores[index]))
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auroc_ci_low": float(np.quantile(values["auroc"], 0.025)),
        "auroc_ci_high": float(np.quantile(values["auroc"], 0.975)),
        "auprc": float(average_precision_score(labels, scores)),
        "auprc_ci_low": float(np.quantile(values["auprc"], 0.025)),
        "auprc_ci_high": float(np.quantile(values["auprc"], 0.975)),
        "bootstrap_replicates": replicates,
        "seed": seed,
    }
