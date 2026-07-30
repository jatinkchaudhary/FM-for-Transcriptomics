#!/usr/bin/env python3
"""Compare all validated decoders on the same held-out ICI cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from evaluate_atlas_augmented_ici import (
    COHORTS,
    CS,
    atlas_scores,
    parse_gmt,
)
from extract_multimodel_ici_via_api import MODELS


EMBEDDING_ONLY = (
    "Txn_Jatin_contextual",
    "ESM2_PCA512_prior",
    "ESM3",
    "Geneformer",
    "scGPT",
)


def transform(X, train, test, top_variance):
    train_matrix = X[train]
    test_matrix = X[test]
    if top_variance:
        selected = np.argsort(np.var(train_matrix, axis=0))[-min(1000, X.shape[1]):]
        train_matrix = train_matrix[:, selected]
        test_matrix = test_matrix[:, selected]
    scaler = StandardScaler().fit(train_matrix)
    return scaler.transform(train_matrix), scaler.transform(test_matrix)


def fit_predict(X, y, train, test, c_value, top_variance):
    X_train, X_test = transform(X, train, test, top_variance)
    model = LogisticRegression(
        C=c_value,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
    model.fit(X_train, y[train])
    return model.predict_proba(X_test)[:, 1]


def tune_calibrate(X, y, groups, outer_train, outer_test, top_variance):
    candidates = []
    train_groups = sorted(np.unique(groups[outer_train]))
    for c_value in CS:
        scores = []
        for held_out in train_groups:
            valid = outer_train & (groups == held_out)
            train = outer_train & (groups != held_out)
            probability = fit_predict(X, y, train, valid, c_value, top_variance)
            scores.append(roc_auc_score(y[valid], probability))
        candidates.append((float(np.mean(scores)), c_value))
    selected_c = sorted(candidates, key=lambda value: (-value[0], value[1]))[0][1]

    outer_indices = np.flatnonzero(outer_train)
    positions = {index: position for position, index in enumerate(outer_indices)}
    oof = np.zeros(len(outer_indices), dtype=float)
    for held_out in train_groups:
        valid = outer_train & (groups == held_out)
        train = outer_train & (groups != held_out)
        probability = fit_predict(X, y, train, valid, selected_c, top_variance)
        for index, value in zip(np.flatnonzero(valid), probability):
            oof[positions[index]] = value
    test_probability = fit_predict(
        X, y, outer_train, outer_test, selected_c, top_variance
    )
    eps = 1e-5
    clipped = np.clip(oof, eps, 1 - eps)
    oof_logit = np.log(clipped / (1 - clipped))

    def objective(offset):
        probability = 1 / (1 + np.exp(-(oof_logit + offset)))
        return log_loss(y[outer_train], probability)

    offset = float(minimize_scalar(objective, bounds=(-6, 6), method="bounded").x)
    clipped_test = np.clip(test_probability, eps, 1 - eps)
    test_logit = np.log(clipped_test / (1 - clipped_test))
    return 1 / (1 + np.exp(-(test_logit + offset))), selected_c


def metrics(frame):
    y = frame["label"].to_numpy()
    p = frame["probability"].to_numpy()
    return {
        "n": len(frame),
        "responders": int(y.sum()),
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
    }


def paired_bootstrap(predictions, method, reference="Raw_common_measured", n_boot=500):
    columns = ["sample_id", "held_out_cohort", "label", "probability"]
    left = predictions.loc[predictions["method"].eq(method), columns].rename(
        columns={"probability": "model_probability"}
    )
    right = predictions.loc[
        predictions["method"].eq(reference), ["sample_id", "probability"]
    ].rename(columns={"probability": "reference_probability"})
    paired = left.merge(right, on="sample_id", validate="one_to_one")

    def macro(frame, probability):
        aurocs, auprcs = [], []
        for _, cohort in frame.groupby("held_out_cohort"):
            aurocs.append(roc_auc_score(cohort["label"], cohort[probability]))
            auprcs.append(average_precision_score(cohort["label"], cohort[probability]))
        return float(np.mean(aurocs)), float(np.mean(auprcs))

    model_observed = macro(paired, "model_probability")
    reference_observed = macro(paired, "reference_probability")
    rng = np.random.default_rng(42)
    auroc_deltas, auprc_deltas = [], []
    cohort_arrays = []
    for _, frame in paired.groupby("held_out_cohort"):
        cohort_arrays.append(
            (
                frame["label"].to_numpy(),
                frame["model_probability"].to_numpy(),
                frame["reference_probability"].to_numpy(),
            )
        )
    for _ in range(n_boot):
        model_aurocs, model_auprcs = [], []
        reference_aurocs, reference_auprcs = [], []
        for labels, model_probability, reference_probability in cohort_arrays:
            indices = np.concatenate(
                [
                    rng.choice(np.flatnonzero(labels == value), int((labels == value).sum()), replace=True)
                    for value in (0, 1)
                ]
            )
            sampled_labels = labels[indices]
            model_aurocs.append(roc_auc_score(sampled_labels, model_probability[indices]))
            model_auprcs.append(average_precision_score(sampled_labels, model_probability[indices]))
            reference_aurocs.append(roc_auc_score(sampled_labels, reference_probability[indices]))
            reference_auprcs.append(average_precision_score(sampled_labels, reference_probability[indices]))
        auroc_deltas.append(float(np.mean(model_aurocs) - np.mean(reference_aurocs)))
        auprc_deltas.append(float(np.mean(model_auprcs) - np.mean(reference_auprcs)))
    return {
        "method": method,
        "reference": reference,
        "delta_macro_auroc": model_observed[0] - reference_observed[0],
        "auroc_ci_low": float(np.quantile(auroc_deltas, 0.025)),
        "auroc_ci_high": float(np.quantile(auroc_deltas, 0.975)),
        "delta_macro_auprc": model_observed[1] - reference_observed[1],
        "auprc_ci_low": float(np.quantile(auprc_deltas, 0.025)),
        "auprc_ci_high": float(np.quantile(auprc_deltas, 0.975)),
        "bootstrap_replicates": n_boot,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--ici-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = args.frozen.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    clinical = pd.read_csv(
        args.ici_root / "prepared" / "clinical_harmonized.tsv", sep="\t"
    ).set_index("Index", drop=False)
    model_matrices = {}
    ordered_ids = []
    genes = None
    raw_parts = []
    measured_masks = []
    for model in MODELS:
        parts = []
        model_ids = []
        for cohort in ("gide", "riaz", "hugo", "rose"):
            with np.load(frozen / f"{cohort}__{model}.npz", allow_pickle=False) as data:
                cohort_genes = data["genes"].astype(str)
                if genes is None:
                    genes = cohort_genes
                elif not np.array_equal(genes, cohort_genes):
                    raise ValueError("Submitted gene order differs")
                ids = data["sample_ids"].astype(str)
                parts.append(np.asarray(data["completed"], dtype=np.float32))
                model_ids.extend(ids.tolist())
                if model == MODELS[0]:
                    raw_parts.append(np.asarray(data["raw"], dtype=np.float32))
                    measured_masks.append(
                        np.asarray(data["measured_gene_mask"], dtype=bool)
                    )
                    ordered_ids.extend(ids.tolist())
        if model_ids != ordered_ids:
            raise ValueError(f"Sample order differs for {model}")
        model_matrices[model] = np.concatenate(parts)
    raw = np.concatenate(raw_parts)
    strict = np.logical_and.reduce(
        [np.isfinite(matrix).all(axis=0) for matrix in model_matrices.values()]
    )
    strict_genes = genes[strict]
    raw = raw[:, strict]
    common_measured = np.logical_and.reduce(measured_masks)[strict]
    raw_common = raw[:, common_measured]
    model_matrices = {name: matrix[:, strict] for name, matrix in model_matrices.items()}
    gene_index = {gene.upper(): index for index, gene in enumerate(strict_genes)}
    gene_sets = parse_gmt(
        args.ici_root / "config" / "MSigDB_Hallmark_2020.gmt", gene_index
    )

    features = {"Raw_common_measured": (raw_common, True)}
    for model, matrix in model_matrices.items():
        features[f"{model}__completed"] = (matrix, True)
        _, scores = atlas_scores(matrix, gene_sets)
        features[f"{model}__atlas"] = (scores, False)

    metadata = clinical.loc[ordered_ids]
    y = metadata["label"].to_numpy(dtype=int)
    groups = metadata["cohort"].to_numpy(dtype=str)
    predictions = []
    for method, (X, top_variance) in features.items():
        for held_out in COHORTS:
            test = groups == held_out
            train = ~test
            probability, selected_c = tune_calibrate(
                X, y, groups, train, test, top_variance
            )
            for index, value in zip(np.flatnonzero(test), probability):
                predictions.append({
                    "method": method,
                    "held_out_cohort": held_out,
                    "sample_id": ordered_ids[index],
                    "label": int(y[index]),
                    "probability": float(value),
                    "selected_C": selected_c,
                })
            print(f"{method}: held out {held_out}", flush=True)
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(output / "multimodel_loco_predictions.csv", index=False)

    rows = []
    for (method, cohort), frame in predictions.groupby(["method", "held_out_cohort"]):
        rows.append({"method": method, "scope": "cohort", "cohort": cohort, **metrics(frame)})
    for method, frame in predictions.groupby("method"):
        rows.append({"method": method, "scope": "pooled", "cohort": "ALL", **metrics(frame)})
    metric_frame = pd.DataFrame(rows)
    metric_frame.to_csv(output / "multimodel_loco_metrics.csv", index=False)
    summary = (
        metric_frame.query("scope == 'cohort'")
        .groupby("method", as_index=False)
        .agg(
            macro_auroc=("auroc", "mean"),
            macro_auprc=("auprc", "mean"),
            macro_brier=("brier", "mean"),
            macro_log_loss=("log_loss", "mean"),
        )
        .merge(
            metric_frame.query("scope == 'pooled'").drop(columns=["scope", "cohort"]),
            on="method",
        )
        .sort_values(["macro_auroc", "macro_auprc"], ascending=False)
    )
    summary.to_csv(output / "multimodel_loco_summary.csv", index=False)
    pd.DataFrame(
        [
            paired_bootstrap(predictions, f"{model}__completed")
            for model in MODELS
        ]
    ).to_csv(output / "paired_bootstrap_vs_raw.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": model,
                "patient_level_track": "unsupported",
                "reason": "No validated bulk-expression decoder or frozen bulk sample encoder",
            }
            for model in EMBEDDING_ONLY
        ]
    ).to_csv(output / "unsupported_embedding_only_models.csv", index=False)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "patients": len(y),
                "cohorts": list(COHORTS),
                "decoder_models": list(MODELS),
                "strict_shared_genes": int(strict.sum()),
                "raw_common_measured_genes": int(common_measured.sum()),
                "primary_track": "completed expression; top 1000 variance genes selected inside training folds",
                "secondary_track": "fixed within-sample rank atlas scores",
                "outer_validation": "leave one entire cohort out",
                "inner_tuning": "leave one training cohort out",
                "calibration": "monotone intercept-only, fitted on inner OOF predictions",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
