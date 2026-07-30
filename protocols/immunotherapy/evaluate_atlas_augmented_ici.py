#!/usr/bin/env python3
"""Atlas-augmented, calibrated leave-one-cohort-out ICI evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


SEED = 42
COHORTS = ("Gide", "Riaz", "Hugo", "Rose")
CS = (0.01, 0.1, 1.0, 10.0)
METHODS = (
    "raw",
    "txn",
    "atlas",
    "raw_atlas",
    "txn_atlas",
    "raw_txn_atlas",
)

# Curated immune programs supplement MSigDB Hallmarks. They are fixed before
# label access and are used only to construct within-sample rank scores.
IMMUNE_SIGNATURES = {
    "ATLAS_T_CELL_INFLAMED": ("CCL5", "CD27", "CD274", "CD8A", "CMKLR1", "CXCL9", "CXCR6", "HLA-DQA1", "HLA-DRB1", "IDO1", "LAG3", "NKG7", "PDCD1LG2", "PSMB10", "STAT1", "TIGIT"),
    "ATLAS_CYTOLYTIC": ("GZMA", "GZMB", "GZMH", "GNLY", "NKG7", "PRF1"),
    "ATLAS_ANTIGEN_PRESENTATION": ("B2M", "HLA-A", "HLA-B", "HLA-C", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1", "HLA-DRA", "HLA-DRB1", "TAP1", "TAP2"),
    "ATLAS_IFNG_AXIS": ("CXCL9", "CXCL10", "CXCL11", "GBP1", "IDO1", "IFNG", "IRF1", "JAK1", "JAK2", "STAT1"),
    "ATLAS_EXHAUSTION": ("CTLA4", "HAVCR2", "LAG3", "PDCD1", "TIGIT", "TOX"),
    "ATLAS_TREG": ("CCR8", "CTLA4", "FOXP3", "IL2RA", "IKZF2", "TNFRSF18"),
    "ATLAS_MYeloid_SUPPRESSION": ("ARG1", "CD163", "CSF1R", "IL10", "MRC1", "S100A8", "S100A9", "TGFB1"),
    "ATLAS_TGF_BETA": ("COL1A1", "COL1A2", "COL3A1", "SERPINE1", "SMAD3", "TGFB1", "TGFBR1", "TGFBR2"),
    "ATLAS_ANGIOGENESIS": ("ANGPT2", "ESM1", "FLT1", "KDR", "PECAM1", "VWF"),
}


def parse_gmt(path: Path, gene_index: dict[str, int]) -> dict[str, np.ndarray]:
    sets: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            idx = sorted({gene_index[g.upper()] for g in fields[2:] if g.upper() in gene_index})
            if len(idx) >= 5:
                sets[fields[0]] = np.asarray(idx, dtype=int)
    for name, genes in IMMUNE_SIGNATURES.items():
        idx = sorted({gene_index[g] for g in genes if g in gene_index})
        if len(idx) >= 3:
            sets[name] = np.asarray(idx, dtype=int)
    return sets


def row_ranks(X: np.ndarray) -> np.ndarray:
    order = np.argsort(X, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(X.shape[0])[:, None]
    ranks[rows, order] = np.arange(X.shape[1], dtype=np.float32)
    return ranks / max(X.shape[1] - 1, 1)


def atlas_scores(X: np.ndarray, sets: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    ranks = row_ranks(X)
    names = list(sets)
    scores = np.column_stack([ranks[:, sets[name]].mean(axis=1) for name in names])
    return names, scores.astype(np.float32)


def load_features(root: Path):
    clinical = pd.read_csv(root / "prepared" / "clinical_harmonized.tsv", sep="\t").set_index("Index", drop=False)
    blocks = {"txn": [], "raw": []}
    metadata = []
    genes = None
    masks = []
    for cohort in COHORTS:
        with np.load(root / "results" / f"{cohort.lower()}_txn_embeddings.npz", allow_pickle=False) as data:
            ids = data["sample_ids"].astype(str)
            cohort_genes = data["genes"].astype(str).tolist()
            genes = cohort_genes if genes is None else genes
            if genes != cohort_genes:
                raise ValueError("Gene order differs across cohorts")
            blocks["txn"].append(np.asarray(data["txn_context_mean"], dtype=np.float32))
            blocks["raw"].append(np.asarray(data["aligned_log1p_tpm"], dtype=np.float32))
            masks.append(np.asarray(data["measured_gene_mask"], dtype=bool))
        metadata.append(clinical.loc[ids])
    meta = pd.concat(metadata)
    common = np.logical_and.reduce(masks)
    raw = np.concatenate(blocks["raw"])[:, common]
    common_genes = [g for g, keep in zip(genes, common) if keep]
    gene_index = {g.upper(): i for i, g in enumerate(common_genes)}
    sets = parse_gmt(root / "config" / "MSigDB_Hallmark_2020.gmt", gene_index)
    names, atlas = atlas_scores(raw, sets)
    return (
        meta,
        raw,
        np.concatenate(blocks["txn"]),
        atlas,
        names,
        common_genes,
    )


def transform_blocks(raw, txn, atlas, train, test, groups, method):
    output_train, output_test = [], []
    if "raw" in method:
        variance = np.var(raw[train], axis=0)
        selected = np.argsort(variance)[-min(1000, raw.shape[1]):]
        output_train.append(raw[train][:, selected])
        output_test.append(raw[test][:, selected])
    if "txn" in method:
        # Unlabelled cohort centering removes cohort offsets. Test labels are
        # never accessed, but this is explicitly a transductive normalization.
        tr = txn[train].copy()
        te = txn[test].copy()
        for cohort in np.unique(groups[train]):
            mask = groups[train] == cohort
            tr[mask] -= tr[mask].mean(axis=0, keepdims=True)
        te -= te.mean(axis=0, keepdims=True)
        output_train.append(tr)
        output_test.append(te)
    if "atlas" in method:
        output_train.append(atlas[train])
        output_test.append(atlas[test])
    X_train = np.concatenate(output_train, axis=1)
    X_test = np.concatenate(output_test, axis=1)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def fit_predict(raw, txn, atlas, y, groups, train, test, method, c_value):
    X_train, X_test = transform_blocks(raw, txn, atlas, train, test, groups, method)
    model = LogisticRegression(
        C=c_value, solver="liblinear", class_weight="balanced", max_iter=5000, random_state=SEED
    )
    model.fit(X_train, y[train])
    return model.predict_proba(X_test)[:, 1]


def tune_and_calibrate(raw, txn, atlas, y, groups, outer_train, outer_test, method):
    train_groups = sorted(np.unique(groups[outer_train]))
    candidates = []
    for c_value in CS:
        fold_scores = []
        for held_out in train_groups:
            inner_test = outer_train & (groups == held_out)
            inner_train = outer_train & (groups != held_out)
            p = fit_predict(raw, txn, atlas, y, groups, inner_train, inner_test, method, c_value)
            fold_scores.append(roc_auc_score(y[inner_test], p))
        candidates.append((float(np.mean(fold_scores)), c_value))
    selected_c = sorted(candidates, key=lambda x: (-x[0], x[1]))[0][1]
    oof = np.zeros(int(outer_train.sum()), dtype=float)
    outer_indices = np.flatnonzero(outer_train)
    position = {idx: pos for pos, idx in enumerate(outer_indices)}
    for held_out in train_groups:
        inner_test = outer_train & (groups == held_out)
        inner_train = outer_train & (groups != held_out)
        p = fit_predict(raw, txn, atlas, y, groups, inner_train, inner_test, method, selected_c)
        for idx, value in zip(np.flatnonzero(inner_test), p):
            oof[position[idx]] = value
    test_probability = fit_predict(raw, txn, atlas, y, groups, outer_train, outer_test, method, selected_c)
    eps = 1e-5
    oof_logit = np.log(np.clip(oof, eps, 1 - eps) / (1 - np.clip(oof, eps, 1 - eps)))
    test_clipped = np.clip(test_probability, eps, 1 - eps)
    test_logit = np.log(test_clipped / (1 - test_clipped))
    # Intercept-only recalibration is monotone and cannot reverse a biological
    # ranking when heterogeneous inner cohorts suggest a negative Platt slope.
    def calibration_loss(offset):
        probability = 1 / (1 + np.exp(-(oof_logit + offset)))
        return log_loss(y[outer_train], probability)

    offset = float(minimize_scalar(calibration_loss, bounds=(-6, 6), method="bounded").x)
    calibrated = 1 / (1 + np.exp(-(test_logit + offset)))
    return calibrated, selected_c


def metric_row(frame: pd.DataFrame) -> dict:
    y = frame["label"].to_numpy()
    p = frame["probability"].to_numpy()
    clipped = np.clip(p, 1e-5, 1 - 1e-5)
    calibration_model = LogisticRegression(solver="lbfgs").fit(
        np.log(clipped / (1 - clipped))[:, None], y
    )
    return {
        "n": len(frame),
        "responders": int(y.sum()),
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "calibration_intercept": float(calibration_model.intercept_[0]),
        "calibration_slope": float(calibration_model.coef_[0, 0]),
    }


def hedges_g(values: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    a, b = values[labels == 1], values[labels == 0]
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0, float("inf")
    d = (a.mean() - b.mean()) / pooled
    correction = 1 - 3 / (4 * (len(a) + len(b)) - 9)
    g = float(d * correction)
    variance = (len(a) + len(b)) / (len(a) * len(b)) + g * g / (2 * (len(a) + len(b) - 2))
    return g, float(variance)


def random_effects(group: pd.DataFrame) -> pd.Series:
    effects = group["hedges_g"].to_numpy()
    variances = group["variance"].to_numpy()
    fixed_weights = 1 / variances
    fixed = np.sum(fixed_weights * effects) / np.sum(fixed_weights)
    q_value = np.sum(fixed_weights * (effects - fixed) ** 2)
    df = len(effects) - 1
    denominator = np.sum(fixed_weights) - np.sum(fixed_weights**2) / np.sum(fixed_weights)
    tau2 = max(0.0, (q_value - df) / denominator) if denominator > 0 else 0.0
    weights = 1 / (variances + tau2)
    estimate = np.sum(weights * effects) / np.sum(weights)
    standard_error = np.sqrt(1 / np.sum(weights))
    z_value = estimate / standard_error
    p_value = float(2 * norm.sf(abs(z_value)))
    return pd.Series({
        "meta_hedges_g": estimate,
        "ci_low": estimate - 1.96 * standard_error,
        "ci_high": estimate + 1.96 * standard_error,
        "p_value": p_value,
        "tau2": tau2,
        "positive_cohorts": int((effects > 0).sum()),
        "negative_cohorts": int((effects < 0).sum()),
        "effect_sd": float(np.std(effects, ddof=1)),
    })


def bh_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    meta, raw, txn, atlas, pathway_names, common_genes = load_features(root)
    y = meta["label"].to_numpy(dtype=int)
    groups = meta["cohort"].to_numpy(dtype=str)
    sample_ids = meta["Index"].to_numpy(dtype=str)

    records = []
    for method in METHODS:
        for held_out in COHORTS:
            test = groups == held_out
            train = ~test
            probability, selected_c = tune_and_calibrate(raw, txn, atlas, y, groups, train, test, method)
            for idx, p in zip(np.flatnonzero(test), probability):
                records.append({
                    "method": method, "held_out_cohort": held_out, "sample_id": sample_ids[idx],
                    "label": int(y[idx]), "probability": float(p), "selected_C": selected_c,
                })
            print(f"{method}: held out {held_out}", flush=True)
    predictions = pd.DataFrame(records)
    predictions.to_csv(output / "atlas_loco_predictions.csv", index=False)

    metrics = []
    for (method, cohort), frame in predictions.groupby(["method", "held_out_cohort"]):
        metrics.append({"method": method, "scope": "cohort", "cohort": cohort, **metric_row(frame)})
    for method, frame in predictions.groupby("method"):
        metrics.append({"method": method, "scope": "pooled", "cohort": "ALL", **metric_row(frame)})
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(output / "atlas_loco_metrics.csv", index=False)
    summary = (
        metrics.query("scope == 'cohort'")
        .groupby("method", as_index=False)
        .agg(macro_auroc=("auroc", "mean"), macro_auprc=("auprc", "mean"), macro_brier=("brier", "mean"))
        .merge(metrics.query("scope == 'pooled'").drop(columns=["scope", "cohort"]), on="method")
        .sort_values("macro_auroc", ascending=False)
    )
    summary.to_csv(output / "atlas_loco_summary.csv", index=False)

    effects = []
    for cohort in COHORTS:
        mask = groups == cohort
        for col, pathway in enumerate(pathway_names):
            effect, variance = hedges_g(atlas[mask, col], y[mask])
            effects.append({
                "cohort": cohort, "pathway": pathway,
                "hedges_g": effect, "variance": variance,
            })
    effects = pd.DataFrame(effects)
    consistency = (
        effects.groupby("pathway")[["hedges_g", "variance"]]
        .apply(random_effects)
        .reset_index()
    )
    consistency["q_value"] = bh_adjust(consistency["p_value"].to_numpy())
    consistency["direction_concordance"] = consistency[["positive_cohorts", "negative_cohorts"]].max(axis=1) / len(COHORTS)
    consistency = consistency.sort_values(["q_value", "direction_concordance", "meta_hedges_g"], ascending=[True, False, False])
    effects.to_csv(output / "atlas_pathway_effects_by_cohort.csv", index=False)
    consistency.to_csv(output / "atlas_pathway_consistency.csv", index=False)

    evidence = {
        "patients": len(y),
        "cohorts": list(COHORTS),
        "common_genes": len(common_genes),
        "atlas_pathways": len(pathway_names),
        "validation": "nested leave-one-cohort-out",
        "calibration": "Platt scaling fit only to inner leave-cohort-out predictions",
        "cohort_normalization": "unlabelled within-cohort centering for Txn features (transductive)",
        "best_method": summary.iloc[0].to_dict(),
        "top_concordant_pathways": consistency.head(12).to_dict(orient="records"),
        "llm_role": "Evidence summarization only; no labels or LLM outputs enter prediction.",
    }
    (output / "atlas_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    (output / "llm_prompt.txt").write_text(
        "Summarize the following immunotherapy benchmark evidence. Distinguish "
        "discrimination, calibration, and pathway consistency. Do not claim "
        "clinical validity, causality, or patient-level treatment advice.\n\n"
        + json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
