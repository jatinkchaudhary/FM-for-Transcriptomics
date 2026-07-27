#!/usr/bin/env python3
"""Leakage-safe initial immunotherapy response evaluation for Txn_Jatin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


SEED = 42
COHORTS = ("Gide", "Riaz", "Hugo", "Rose")
DISPLAY = {
    "txn_context_mean": "Txn context (contrastive)",
    "txn_hidden_all": "Txn hidden (all pooling)",
    "raw_common_topvar": "Raw expression (common genes)",
    "hallmark_common": "Hallmark means (common genes)",
    "txn_completed_topvar": "Txn-completed expression",
    "txn_completed_hallmark": "Txn-completed Hallmark means",
}
COLORS = {
    "txn_context_mean": "#087E8B",
    "txn_hidden_all": "#E4572E",
    "raw_common_topvar": "#2E5AAC",
    "hallmark_common": "#7A5195",
    "txn_completed_topvar": "#D1495B",
    "txn_completed_hallmark": "#3C8D5A",
}


class TopVarianceSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 1000):
        self.k = k

    def fit(self, X, y=None):
        variance = np.nanvar(np.asarray(X), axis=0)
        k = min(int(self.k), X.shape[1])
        self.indices_ = np.argpartition(variance, -k)[-k:]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.indices_]


def parse_gmt(path: Path, genes: list[str]) -> tuple[list[str], list[np.ndarray]]:
    gene_index = {gene.upper(): i for i, gene in enumerate(genes)}
    names = []
    members = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            idx = sorted(
                {gene_index[g.upper()] for g in fields[2:] if g.upper() in gene_index}
            )
            if len(idx) >= 5:
                names.append(fields[0])
                members.append(np.asarray(idx, dtype=int))
    if not names:
        raise ValueError(f"No Hallmark gene sets mapped from {path}")
    return names, members


def pathway_means(X: np.ndarray, members: list[np.ndarray]) -> np.ndarray:
    return np.column_stack([X[:, idx].mean(axis=1) for idx in members]).astype(
        np.float32
    )


def estimator(method: str, c_value: float) -> Pipeline:
    steps = []
    if method == "txn_context_mean":
        steps.append(("l2_normalize", Normalizer(norm="l2")))
    if method.endswith("topvar"):
        steps.append(("top_variance", TopVarianceSelector(k=1000)))
    steps.extend(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    return Pipeline(steps)


def metric_values(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, probability >= 0.5)
        ),
        "brier": float(brier_score_loss(y, probability)),
        "prevalence": float(np.mean(y)),
    }


def bootstrap_ci(
    y: np.ndarray, probability: np.ndarray, n_boot: int = 1000
) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    aurocs = []
    auprcs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y[idx], probability[idx]))
        auprcs.append(average_precision_score(y[idx], probability[idx]))
    return {
        "auroc_ci_low": float(np.quantile(aurocs, 0.025)),
        "auroc_ci_high": float(np.quantile(aurocs, 0.975)),
        "auprc_ci_low": float(np.quantile(auprcs, 0.025)),
        "auprc_ci_high": float(np.quantile(auprcs, 0.975)),
        "bootstrap_valid": int(len(aurocs)),
    }


def choose_c(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    method: str,
    candidates=(0.01, 0.1, 1.0, 10.0),
) -> tuple[float, list[dict]]:
    records = []
    unique_groups = sorted(np.unique(groups).tolist())
    for c_value in candidates:
        scores = []
        for held_out in unique_groups:
            train = groups != held_out
            valid = groups == held_out
            if len(np.unique(y[train])) < 2 or len(np.unique(y[valid])) < 2:
                continue
            model = estimator(method, c_value)
            model.fit(X[train], y[train])
            pred = model.predict_proba(X[valid])[:, 1]
            scores.append(float(roc_auc_score(y[valid], pred)))
        mean_score = float(np.mean(scores)) if scores else float("nan")
        records.append(
            {
                "C": float(c_value),
                "inner_macro_auroc": mean_score,
                "inner_folds": int(len(scores)),
            }
        )
    valid = [r for r in records if np.isfinite(r["inner_macro_auroc"])]
    if not valid:
        return 1.0, records
    valid.sort(key=lambda r: (-r["inner_macro_auroc"], r["C"]))
    return float(valid[0]["C"]), records


def paired_delta_bootstrap(
    predictions: pd.DataFrame, method: str, reference: str, n_boot: int = 3000
) -> dict:
    left = predictions.loc[
        predictions["method"].eq(method), ["sample_id", "label", "probability"]
    ].rename(columns={"probability": "method_probability"})
    right = predictions.loc[
        predictions["method"].eq(reference), ["sample_id", "probability"]
    ].rename(columns={"probability": "reference_probability"})
    paired = left.merge(right, on="sample_id", validate="one_to_one")
    y = paired["label"].to_numpy(dtype=int)
    p_model = paired["method_probability"].to_numpy()
    p_reference = paired["reference_probability"].to_numpy()
    observed = roc_auc_score(y, p_model) - roc_auc_score(y, p_reference)
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(
            roc_auc_score(y[idx], p_model[idx])
            - roc_auc_score(y[idx], p_reference[idx])
        )
    return {
        "method": method,
        "reference": reference,
        "delta_auroc": float(observed),
        "delta_ci_low": float(np.quantile(deltas, 0.025)),
        "delta_ci_high": float(np.quantile(deltas, 0.975)),
        "bootstrap_valid": int(len(deltas)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results"
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    clinical = pd.read_csv(root / "prepared" / "clinical_harmonized.tsv", sep="\t")
    clinical = clinical.set_index("Index", drop=False)
    arrays: dict[str, list[np.ndarray]] = {
        "txn_context_mean": [],
        "txn_hidden_all": [],
        "raw_aligned": [],
        "txn_completed": [],
    }
    measured_masks = []
    ordered_metadata = []
    genes = None
    for cohort in COHORTS:
        path = results / f"{cohort.lower()}_txn_embeddings.npz"
        with np.load(path, allow_pickle=False) as data:
            sample_ids = data["sample_ids"].astype(str)
            cohort_genes = data["genes"].astype(str).tolist()
            if genes is None:
                genes = cohort_genes
            elif genes != cohort_genes:
                raise ValueError("Model gene order differs across cohort files")
            arrays["txn_context_mean"].append(
                np.asarray(data["txn_context_mean"], dtype=np.float32)
            )
            arrays["txn_hidden_all"].append(
                np.asarray(data["txn_hidden_all"], dtype=np.float32)
            )
            arrays["raw_aligned"].append(
                np.asarray(data["aligned_log1p_tpm"], dtype=np.float32)
            )
            arrays["txn_completed"].append(
                np.asarray(data["txn_completed_log1p_tpm"], dtype=np.float32)
            )
            measured_masks.append(np.asarray(data["measured_gene_mask"], dtype=bool))
        missing = [x for x in sample_ids if x not in clinical.index]
        if missing:
            raise ValueError(f"{cohort} embedding IDs absent from clinical data: {missing}")
        frame = clinical.loc[sample_ids].copy()
        if not frame["cohort"].eq(cohort).all():
            raise ValueError(f"{cohort} clinical order mismatch")
        ordered_metadata.append(frame)

    metadata = pd.concat(ordered_metadata, axis=0)
    combined = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    common_mask = np.logical_and.reduce(measured_masks)
    common_genes = [gene for gene, keep in zip(genes, common_mask) if keep]
    raw_common = combined["raw_aligned"][:, common_mask]
    completed = combined["txn_completed"]
    common_names, common_members = parse_gmt(
        root / "config" / "MSigDB_Hallmark_2020.gmt", common_genes
    )
    completed_names, completed_members = parse_gmt(
        root / "config" / "MSigDB_Hallmark_2020.gmt", genes
    )
    features = {
        "txn_context_mean": combined["txn_context_mean"],
        "txn_hidden_all": combined["txn_hidden_all"],
        "raw_common_topvar": raw_common,
        "hallmark_common": pathway_means(raw_common, common_members),
        "txn_completed_topvar": completed,
        "txn_completed_hallmark": pathway_means(completed, completed_members),
    }
    y = metadata["label"].to_numpy(dtype=int)
    groups = metadata["cohort"].to_numpy(dtype=str)
    cancers = metadata["cancer_type"].to_numpy(dtype=str)
    sample_ids = metadata["Index"].to_numpy(dtype=str)

    pd.DataFrame(
        features["hallmark_common"], index=sample_ids, columns=common_names
    ).to_csv(results / "hallmark_common_features.csv", index_label="sample_id")
    pd.DataFrame(
        features["txn_completed_hallmark"],
        index=sample_ids,
        columns=completed_names,
    ).to_csv(
        results / "txn_completed_hallmark_features.csv", index_label="sample_id"
    )
    cohort_characteristics = (
        metadata.groupby(["cohort", "cancer_type"], as_index=False)
        .agg(
            patients=("label", "size"),
            responders=("label", "sum"),
            response_rate=("label", "mean"),
        )
        .sort_values("cohort")
    )
    cohort_characteristics["nonresponders"] = (
        cohort_characteristics["patients"] - cohort_characteristics["responders"]
    )
    cohort_characteristics.to_csv(results / "cohort_characteristics.csv", index=False)

    analysis_sets = {
        "four_cohort_loco": np.ones(len(y), dtype=bool),
        "melanoma_loco": cancers == "SKCM",
    }
    prediction_records = []
    tuning_records = []
    for analysis_name, inclusion in analysis_sets.items():
        for method, X_all in features.items():
            X = X_all[inclusion]
            y_set = y[inclusion]
            group_set = groups[inclusion]
            ids_set = sample_ids[inclusion]
            cancer_set = cancers[inclusion]
            for held_out in sorted(np.unique(group_set)):
                train = group_set != held_out
                test = group_set == held_out
                selected_c, tuning = choose_c(
                    X[train], y_set[train], group_set[train], method
                )
                for record in tuning:
                    tuning_records.append(
                        {
                            "analysis": analysis_name,
                            "method": method,
                            "held_out_cohort": held_out,
                            **record,
                            "selected": bool(record["C"] == selected_c),
                        }
                    )
                model = estimator(method, selected_c)
                model.fit(X[train], y_set[train])
                probability = model.predict_proba(X[test])[:, 1]
                for i, p in zip(np.where(test)[0], probability):
                    prediction_records.append(
                        {
                            "analysis": analysis_name,
                            "method": method,
                            "held_out_cohort": held_out,
                            "sample_id": ids_set[i],
                            "cancer_type": cancer_set[i],
                            "label": int(y_set[i]),
                            "probability": float(p),
                            "selected_C": float(selected_c),
                        }
                    )
                print(
                    f"{analysis_name} {method} held_out={held_out} C={selected_c}",
                    flush=True,
                )

    predictions = pd.DataFrame(prediction_records)
    predictions.to_csv(results / "loco_predictions.csv", index=False)
    pd.DataFrame(tuning_records).to_csv(results / "nested_tuning.csv", index=False)

    metric_records = []
    for (analysis, method, cohort), frame in predictions.groupby(
        ["analysis", "method", "held_out_cohort"], sort=False
    ):
        values = metric_values(frame["label"].to_numpy(), frame["probability"].to_numpy())
        ci = bootstrap_ci(frame["label"].to_numpy(), frame["probability"].to_numpy())
        metric_records.append(
            {
                "analysis": analysis,
                "method": method,
                "scope": "held_out_cohort",
                "cohort": cohort,
                "n": int(len(frame)),
                "responders": int(frame["label"].sum()),
                **values,
                **ci,
            }
        )
    for (analysis, method), frame in predictions.groupby(["analysis", "method"]):
        values = metric_values(frame["label"].to_numpy(), frame["probability"].to_numpy())
        ci = bootstrap_ci(frame["label"].to_numpy(), frame["probability"].to_numpy())
        metric_records.append(
            {
                "analysis": analysis,
                "method": method,
                "scope": "pooled_loco",
                "cohort": "ALL",
                "n": int(len(frame)),
                "responders": int(frame["label"].sum()),
                **values,
                **ci,
            }
        )
    metrics = pd.DataFrame(metric_records)
    metrics.to_csv(results / "loco_metrics.csv", index=False)

    macro = (
        metrics.loc[metrics["scope"].eq("held_out_cohort")]
        .groupby(["analysis", "method"], as_index=False)
        .agg(
            macro_auroc=("auroc", "mean"),
            macro_auprc=("auprc", "mean"),
            macro_balanced_accuracy=("balanced_accuracy", "mean"),
            macro_brier=("brier", "mean"),
            cohorts=("cohort", "nunique"),
        )
    )
    pooled = metrics.loc[metrics["scope"].eq("pooled_loco")].drop(
        columns=["scope", "cohort"]
    )
    summary = macro.merge(pooled, on=["analysis", "method"], suffixes=("", "_pooled"))
    summary.to_csv(results / "loco_summary.csv", index=False)

    delta_records = []
    for analysis in analysis_sets:
        subset = predictions.loc[predictions["analysis"].eq(analysis)]
        for method in features:
            if method == "raw_common_topvar":
                continue
            delta = paired_delta_bootstrap(subset, method, "raw_common_topvar")
            delta["analysis"] = analysis
            delta_records.append(delta)
    deltas = pd.DataFrame(delta_records)
    deltas.to_csv(results / "paired_delta_vs_raw.csv", index=False)

    within_records = []
    within_methods = (
        "txn_context_mean",
        "raw_common_topvar",
        "hallmark_common",
    )
    for cohort in COHORTS:
        keep = groups == cohort
        y_cohort = y[keep]
        ids_cohort = sample_ids[keep]
        splitter = RepeatedStratifiedKFold(
            n_splits=5, n_repeats=1, random_state=SEED
        )
        splits = list(splitter.split(np.zeros(len(y_cohort)), y_cohort))
        for method in within_methods:
            X_all = features[method]
            X = X_all[keep]
            sums = np.zeros(len(y_cohort), dtype=float)
            counts = np.zeros(len(y_cohort), dtype=int)
            for train, test in splits:
                model = estimator(method, 1.0)
                model.fit(X[train], y_cohort[train])
                sums[test] += model.predict_proba(X[test])[:, 1]
                counts[test] += 1
            probability = sums / np.maximum(counts, 1)
            for i, p in enumerate(probability):
                within_records.append(
                    {
                        "cohort": cohort,
                        "method": method,
                        "sample_id": ids_cohort[i],
                        "label": int(y_cohort[i]),
                        "probability": float(p),
                        "repeat_predictions": int(counts[i]),
                    }
                )
            print(f"within-cohort {cohort} {method}", flush=True)
    within_predictions = pd.DataFrame(within_records)
    within_predictions.to_csv(results / "within_cohort_predictions.csv", index=False)
    within_metrics = []
    for (cohort, method), frame in within_predictions.groupby(["cohort", "method"]):
        values = metric_values(frame["label"].to_numpy(), frame["probability"].to_numpy())
        ci = bootstrap_ci(frame["label"].to_numpy(), frame["probability"].to_numpy())
        within_metrics.append(
            {
                "cohort": cohort,
                "method": method,
                "n": int(len(frame)),
                "responders": int(frame["label"].sum()),
                **values,
                **ci,
            }
        )
    within_metrics = pd.DataFrame(within_metrics)
    within_metrics.to_csv(results / "within_cohort_metrics.csv", index=False)

    # Cohort composition.
    composition = cohort_characteristics.set_index("cohort").loc[list(COHORTS)]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(composition))
    ax.bar(x, composition["nonresponders"], color="#B8C2CC", label="Nonresponder")
    ax.bar(
        x,
        composition["responders"],
        bottom=composition["nonresponders"],
        color="#087E8B",
        label="Responder",
    )
    ax.set_xticks(x, composition.index)
    ax.set_ylabel("Pretreatment patients")
    ax.set_title("Initial public ICI cohorts")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "cohort_composition.png", dpi=220)
    plt.close(fig)

    # Per-cohort LOCO heatmap.
    heat_source = metrics.loc[
        metrics["analysis"].eq("four_cohort_loco")
        & metrics["scope"].eq("held_out_cohort")
    ]
    heat = heat_source.pivot(index="method", columns="cohort", values="auroc").reindex(
        index=list(features), columns=list(COHORTS)
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    image = ax.imshow(heat.to_numpy(), vmin=0.25, vmax=0.75, cmap="RdBu_r")
    for row in range(heat.shape[0]):
        for col in range(heat.shape[1]):
            value = heat.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=10)
    ax.set_xticks(np.arange(len(COHORTS)), COHORTS)
    ax.set_yticks(
        np.arange(len(features)), [DISPLAY[x] for x in heat.index], fontsize=9
    )
    ax.set_title("Leave-one-cohort-out response AUROC")
    fig.colorbar(image, ax=ax, label="AUROC")
    fig.tight_layout()
    fig.savefig(figures / "loco_auroc_heatmap.png", dpi=220)
    plt.close(fig)

    # Pooled LOCO ROC.
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    pooled_predictions = predictions.loc[
        predictions["analysis"].eq("four_cohort_loco")
    ]
    for method in features:
        frame = pooled_predictions.loc[pooled_predictions["method"].eq(method)]
        false_positive, true_positive, _ = roc_curve(frame["label"], frame["probability"])
        auc = roc_auc_score(frame["label"], frame["probability"])
        ax.plot(
            false_positive,
            true_positive,
            lw=2,
            color=COLORS[method],
            label=f"{DISPLAY[method]} ({auc:.2f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7A7A7A", lw=1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Pooled out-of-cohort predictions")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(figures / "pooled_loco_roc.png", dpi=220)
    plt.close(fig)

    # Diagnostic within-cohort AUROC.
    within_heat = within_metrics.pivot(
        index="method", columns="cohort", values="auroc"
    ).reindex(index=list(within_methods), columns=list(COHORTS))
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    image = ax.imshow(within_heat.to_numpy(), vmin=0.4, vmax=1.0, cmap="YlGnBu")
    for row in range(within_heat.shape[0]):
        for col in range(within_heat.shape[1]):
            ax.text(
                col,
                row,
                f"{within_heat.iloc[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
            )
    ax.set_xticks(np.arange(len(COHORTS)), COHORTS)
    ax.set_yticks(
        np.arange(len(within_methods)),
        [DISPLAY[x] for x in within_heat.index],
        fontsize=9,
    )
    ax.set_title("Five-fold within-cohort response AUROC (diagnostic)")
    fig.colorbar(image, ax=ax, label="AUROC")
    fig.tight_layout()
    fig.savefig(figures / "within_cohort_auroc_heatmap.png", dpi=220)
    plt.close(fig)

    # Txn context geometry.
    context = StandardScaler().fit_transform(features["txn_context_mean"])
    coords = PCA(n_components=2, random_state=SEED).fit_transform(context)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    cohort_colors = dict(
        zip(COHORTS, ("#087E8B", "#E4572E", "#2E5AAC", "#7A5195"))
    )
    for cohort in COHORTS:
        for label, marker in ((0, "o"), (1, "^")):
            keep = (groups == cohort) & (y == label)
            ax.scatter(
                coords[keep, 0],
                coords[keep, 1],
                s=32,
                alpha=0.78,
                marker=marker,
                color=cohort_colors[cohort],
                edgecolor="white",
                linewidth=0.35,
                label=f"{cohort} {'R' if label else 'NR'}",
            )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Frozen Txn_Jatin context geometry")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "txn_context_pca.png", dpi=220)
    plt.close(fig)

    # Response-score distributions for primary representation.
    primary = pooled_predictions.loc[
        pooled_predictions["method"].eq("txn_context_mean")
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.8), sharey=True)
    rng = np.random.default_rng(SEED)
    for ax, cohort in zip(axes, COHORTS):
        frame = primary.loc[primary["held_out_cohort"].eq(cohort)]
        for label, position, color in ((0, 0, "#A9B4BE"), (1, 1, "#087E8B")):
            values = frame.loc[frame["label"].eq(label), "probability"].to_numpy()
            ax.boxplot(
                values,
                positions=[position],
                widths=0.55,
                patch_artist=True,
                boxprops={"facecolor": color, "alpha": 0.65},
                medianprops={"color": "black"},
            )
            jitter = rng.normal(position, 0.055, size=len(values))
            ax.scatter(jitter, values, s=11, color=color, alpha=0.75)
        ax.set_xticks([0, 1], ["NR", "R"])
        ax.set_title(cohort)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Out-of-cohort response probability")
    fig.suptitle("Txn_Jatin frozen-transfer scores", y=1.02)
    fig.tight_layout()
    fig.savefig(figures / "txn_response_scores_by_cohort.png", dpi=220)
    plt.close(fig)

    with pd.ExcelWriter(results / "ici_initial_results.xlsx", engine="openpyxl") as writer:
        cohort_characteristics.to_excel(writer, sheet_name="cohorts", index=False)
        summary.to_excel(writer, sheet_name="LOCO_summary", index=False)
        metrics.to_excel(writer, sheet_name="LOCO_metrics", index=False)
        predictions.to_excel(writer, sheet_name="LOCO_predictions", index=False)
        deltas.to_excel(writer, sheet_name="delta_vs_raw", index=False)
        within_metrics.to_excel(writer, sheet_name="within_metrics", index=False)
        pd.DataFrame(tuning_records).to_excel(writer, sheet_name="nested_tuning", index=False)

    metadata_output = {
        "seed": SEED,
        "primary_endpoint": "RECIST-derived binary response: R (CR/PR) vs NR (SD/PD)",
        "primary_validation": "leave one entire cohort out",
        "secondary_validation": (
            "single five-fold within-cohort CV for three primary comparators; "
            "diagnostic only"
        ),
        "classifier": "class-weighted L2 logistic regression",
        "C_selection": (
            "nested leave-one-training-cohort-out AUROC over [0.01, 0.1, 1, 10]"
        ),
        "raw_expression_selector": (
            "top 1,000 variance genes fitted only in each training split"
        ),
        "hallmark_sets_common": int(len(common_names)),
        "hallmark_sets_completed": int(len(completed_names)),
        "common_measured_model_genes": int(common_mask.sum()),
        "patients": int(len(y)),
        "responders": int(y.sum()),
        "cohorts": list(COHORTS),
        "features": {key: list(value.shape) for key, value in features.items()},
    }
    (results / "evaluation_metadata.json").write_text(
        json.dumps(metadata_output, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(within_metrics[["cohort", "method", "auroc", "auprc"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
