#!/usr/bin/env python3
"""Quantify cohort nuisance and exploratory Hallmark response biology."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


COHORTS = ("Gide", "Riaz", "Hugo", "Rose")
FEATURE_LABELS = {
    "txn_context_mean": "Txn context",
    "txn_hidden_all": "Txn hidden all",
    "raw_common_topvar": "Raw common genes",
    "hallmark_common": "Hallmark means",
    "txn_completed_topvar": "Txn-completed expression",
}


def load_evaluation_module(path: Path):
    spec = importlib.util.spec_from_file_location("ici_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def eta_squared(X: np.ndarray, labels: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    grand = X.mean(axis=0)
    total = np.square(X - grand).sum()
    between = 0.0
    for label in np.unique(labels):
        group = X[labels == label]
        between += len(group) * np.square(group.mean(axis=0) - grand).sum()
    return float(between / total) if total > 0 else float("nan")


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def hedges_g(responder: np.ndarray, nonresponder: np.ndarray) -> tuple[float, float]:
    n1, n0 = len(responder), len(nonresponder)
    pooled_variance = (
        (n1 - 1) * np.var(responder, ddof=1)
        + (n0 - 1) * np.var(nonresponder, ddof=1)
    ) / max(n1 + n0 - 2, 1)
    if pooled_variance <= 0:
        return 0.0, float((n1 + n0) / (n1 * n0))
    d = (np.mean(responder) - np.mean(nonresponder)) / np.sqrt(pooled_variance)
    correction = 1.0 - 3.0 / max(4.0 * (n1 + n0) - 9.0, 1.0)
    g = correction * d
    variance = (n1 + n0) / (n1 * n0) + g * g / max(
        2.0 * (n1 + n0 - 2), 1.0
    )
    return float(g), float(variance)


def random_effects_meta(effects: np.ndarray, variances: np.ndarray) -> dict:
    fixed_weights = 1.0 / variances
    fixed_mean = np.sum(fixed_weights * effects) / np.sum(fixed_weights)
    q_value = np.sum(fixed_weights * np.square(effects - fixed_mean))
    df = len(effects) - 1
    denominator = np.sum(fixed_weights) - (
        np.sum(np.square(fixed_weights)) / np.sum(fixed_weights)
    )
    tau_squared = max(0.0, (q_value - df) / denominator) if denominator > 0 else 0.0
    weights = 1.0 / (variances + tau_squared)
    mean = np.sum(weights * effects) / np.sum(weights)
    standard_error = np.sqrt(1.0 / np.sum(weights))
    z_score = mean / standard_error if standard_error > 0 else 0.0
    p_value = 2.0 * norm.sf(abs(z_score))
    i_squared = max(0.0, (q_value - df) / q_value) if q_value > 0 else 0.0
    return {
        "meta_hedges_g": float(mean),
        "meta_se": float(standard_error),
        "ci_low": float(mean - 1.96 * standard_error),
        "ci_high": float(mean + 1.96 * standard_error),
        "p_value": float(p_value),
        "tau_squared": float(tau_squared),
        "i_squared": float(i_squared),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results"
    figures = root / "figures"
    evaluation = load_evaluation_module(root / "scripts" / "evaluate_ici_transfer.py")

    clinical = pd.read_csv(root / "prepared" / "clinical_harmonized.tsv", sep="\t")
    clinical = clinical.set_index("Index", drop=False)
    context_parts = []
    hidden_parts = []
    raw_parts = []
    completed_parts = []
    masks = []
    metadata_parts = []
    genes = None
    for cohort in COHORTS:
        with np.load(
            results / f"{cohort.lower()}_txn_embeddings.npz", allow_pickle=False
        ) as data:
            sample_ids = data["sample_ids"].astype(str)
            context_parts.append(data["txn_context_mean"].astype(np.float32))
            hidden_parts.append(data["txn_hidden_all"].astype(np.float32))
            raw_parts.append(data["aligned_log1p_tpm"].astype(np.float32))
            completed_parts.append(
                data["txn_completed_log1p_tpm"].astype(np.float32)
            )
            masks.append(data["measured_gene_mask"].astype(bool))
            if genes is None:
                genes = data["genes"].astype(str).tolist()
        metadata_parts.append(clinical.loc[sample_ids])
    metadata = pd.concat(metadata_parts)
    cohort_labels = metadata["cohort"].to_numpy(dtype=str)
    response_labels = metadata["label"].to_numpy(dtype=int)
    common_mask = np.logical_and.reduce(masks)
    raw_common = np.concatenate(raw_parts)[:, common_mask]
    completed = np.concatenate(completed_parts)
    hallmark = pd.read_csv(results / "hallmark_common_features.csv", index_col=0)
    hallmark = hallmark.loc[metadata["Index"]]

    features = {
        "txn_context_mean": np.concatenate(context_parts),
        "txn_hidden_all": np.concatenate(hidden_parts),
        "raw_common_topvar": raw_common,
        "hallmark_common": hallmark.to_numpy(dtype=np.float32),
        "txn_completed_topvar": completed,
    }

    diagnostics = []
    cohort_code = pd.Categorical(cohort_labels, categories=COHORTS).codes
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, X in features.items():
        if name.endswith("topvar"):
            selector = evaluation.TopVarianceSelector(k=1000)
            descriptive_X = selector.fit_transform(X)
            classifier_steps = [
                ("top_variance", evaluation.TopVarianceSelector(k=1000))
            ]
        else:
            descriptive_X = X
            classifier_steps = []
        descriptive_scaled = StandardScaler().fit_transform(descriptive_X)
        dimensions = min(20, descriptive_scaled.shape[0] - 1, descriptive_scaled.shape[1])
        coordinates = PCA(n_components=dimensions, random_state=42).fit_transform(
            descriptive_scaled
        )
        classifier = Pipeline(
            classifier_steps
            + [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=20, random_state=42)),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        )
        predicted = cross_val_predict(
            classifier, X, cohort_code, cv=splitter, method="predict"
        )
        cohort_centered = descriptive_scaled.copy()
        for cohort in COHORTS:
            keep = cohort_labels == cohort
            cohort_centered[keep] -= cohort_centered[keep].mean(axis=0)
        diagnostics.append(
            {
                "feature": name,
                "dimensions_for_geometry": int(dimensions),
                "cohort_cv_accuracy": float(accuracy_score(cohort_code, predicted)),
                "cohort_cv_macro_f1": float(
                    f1_score(cohort_code, predicted, average="macro")
                ),
                "cohort_silhouette": float(
                    silhouette_score(coordinates, cohort_labels)
                ),
                "response_silhouette": float(
                    silhouette_score(coordinates, response_labels)
                ),
                "cohort_eta_squared": eta_squared(
                    descriptive_scaled, cohort_labels
                ),
                "response_eta_squared_after_cohort_centering": eta_squared(
                    cohort_centered, response_labels
                ),
            }
        )
    diagnostics = pd.DataFrame(diagnostics)
    diagnostics.to_csv(results / "representation_nuisance_diagnostics.csv", index=False)

    association_records = []
    for pathway in hallmark.columns:
        for cohort in COHORTS:
            keep = cohort_labels == cohort
            values = hallmark.loc[keep, pathway].to_numpy(dtype=float)
            labels = response_labels[keep]
            responder = values[labels == 1]
            nonresponder = values[labels == 0]
            effect, variance = hedges_g(responder, nonresponder)
            test = mannwhitneyu(
                responder, nonresponder, alternative="two-sided", method="auto"
            )
            association_records.append(
                {
                    "pathway": pathway,
                    "cohort": cohort,
                    "responders": int(len(responder)),
                    "nonresponders": int(len(nonresponder)),
                    "mean_responder": float(np.mean(responder)),
                    "mean_nonresponder": float(np.mean(nonresponder)),
                    "hedges_g": effect,
                    "variance": variance,
                    "auc_responder_higher": float(test.statistic / (len(responder) * len(nonresponder))),
                    "p_value": float(test.pvalue),
                }
            )
    associations = pd.DataFrame(association_records)
    associations["q_value_within_cohort"] = np.nan
    for cohort in COHORTS:
        keep = associations["cohort"].eq(cohort)
        associations.loc[keep, "q_value_within_cohort"] = bh_adjust(
            associations.loc[keep, "p_value"].to_numpy()
        )
    associations.to_csv(results / "hallmark_response_associations.csv", index=False)

    meta_records = []
    for pathway, frame in associations.groupby("pathway"):
        meta = random_effects_meta(
            frame["hedges_g"].to_numpy(), frame["variance"].to_numpy()
        )
        sign = np.sign(meta["meta_hedges_g"])
        concordant = int((np.sign(frame["hedges_g"].to_numpy()) == sign).sum())
        meta_records.append(
            {
                "pathway": pathway,
                "cohorts": int(len(frame)),
                "direction_concordant_cohorts": concordant,
                **meta,
            }
        )
    meta = pd.DataFrame(meta_records)
    meta["q_value"] = bh_adjust(meta["p_value"].to_numpy())
    meta = meta.sort_values(["q_value", "p_value"])
    meta.to_csv(results / "hallmark_random_effects_meta.csv", index=False)

    # Nuisance-versus-response geometry.
    plot_frame = diagnostics.set_index("feature").loc[list(FEATURE_LABELS)]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(plot_frame))
    width = 0.36
    ax.bar(
        x - width / 2,
        plot_frame["cohort_silhouette"],
        width,
        color="#D1495B",
        label="Cohort/platform",
    )
    ax.bar(
        x + width / 2,
        plot_frame["response_silhouette"],
        width,
        color="#087E8B",
        label="Response",
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(
        x, [FEATURE_LABELS[value] for value in plot_frame.index], rotation=18, ha="right"
    )
    ax.set_ylabel("Silhouette score in 20-PC space")
    ax.set_title("Representation geometry is organized by cohort, not response")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "cohort_vs_response_geometry.png", dpi=220)
    plt.close(fig)

    # Most consistent pathway effects.
    top = meta.sort_values("p_value").head(12).copy()
    top_pathways = top["pathway"].tolist()[::-1]
    effect_matrix = associations.pivot(
        index="pathway", columns="cohort", values="hedges_g"
    ).loc[top_pathways, list(COHORTS)]
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    image = ax.imshow(effect_matrix.to_numpy(), vmin=-1.2, vmax=1.2, cmap="RdBu_r")
    for row in range(effect_matrix.shape[0]):
        for col in range(effect_matrix.shape[1]):
            ax.text(
                col,
                row,
                f"{effect_matrix.iloc[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    ax.set_xticks(np.arange(len(COHORTS)), COHORTS)
    ax.set_yticks(
        np.arange(len(top_pathways)),
        [x.replace("HALLMARK_", "").replace("_", " ").title() for x in top_pathways],
        fontsize=8,
    )
    ax.set_title("Exploratory response-associated Hallmark effects")
    fig.colorbar(image, ax=ax, label="Hedges g (R minus NR)")
    fig.tight_layout()
    fig.savefig(figures / "hallmark_effect_heterogeneity.png", dpi=220)
    plt.close(fig)

    output = {
        "cohort_classifier_best_accuracy": float(
            diagnostics["cohort_cv_accuracy"].max()
        ),
        "cohort_classifier_txn_context_accuracy": float(
            diagnostics.set_index("feature").loc[
                "txn_context_mean", "cohort_cv_accuracy"
            ]
        ),
        "meta_pathways_fdr_below_0_05": int((meta["q_value"] < 0.05).sum()),
        "minimum_meta_q_value": float(meta["q_value"].min()),
        "interpretation": (
            "Exploratory only; pathway tests are not independent validation and "
            "must not be treated as biomarker discovery."
        ),
    }
    with pd.ExcelWriter(
        results / "ici_initial_results.xlsx",
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        diagnostics.to_excel(writer, sheet_name="nuisance_geometry", index=False)
        meta.to_excel(writer, sheet_name="Hallmark_meta", index=False)
        associations.to_excel(writer, sheet_name="Hallmark_by_cohort", index=False)
        for filename, sheet_name in (
            ("pretraining_overlap_summary.csv", "pretraining_overlap"),
            ("clean_no_pretraining_overlap_summary.csv", "clean_LOCO_summary"),
            ("clean_no_pretraining_overlap_deltas.csv", "clean_delta_vs_raw"),
        ):
            path = results / filename
            if path.exists():
                pd.read_csv(path).to_excel(writer, sheet_name=sheet_name, index=False)
    (results / "biology_diagnostic_summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(diagnostics.to_string(index=False))
    print(meta.head(12).to_string(index=False))
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
