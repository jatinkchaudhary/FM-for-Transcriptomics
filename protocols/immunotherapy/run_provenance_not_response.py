#!/usr/bin/env python3
"""Run the four-cohort immunotherapy provenance-not-response audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED = 42
COHORTS = ("Gide", "Riaz", "Hugo", "Rose")
MODELS = (
    "Txn_Jatin",
    "Txn_Jatin_OSDR_LoRA",
    "BRIDGE",
    "BulkFormer_37M",
    "BulkFormer_50M",
    "BulkFormer_93M",
    "BulkFormer_127M",
    "BulkFormer_147M",
)
CS = (0.01, 0.1, 1.0, 10.0)

# Fixed before outcome analysis. GEP and CYT are verbatim published signatures;
# the remaining compact panels are named biological references, not clinical tests.
SIGNATURES = {
    "T_cell_inflamed_GEP_Ayers18": (
        "CCL5", "CD27", "CD274", "CD8A", "CMKLR1", "CXCL9", "CXCR6", "HLA-DQA1",
        "HLA-DRB1", "HLA-E", "IDO1", "LAG3", "NKG7", "PDCD1LG2", "PSMB10",
        "STAT1", "TIGIT", "TAGAP",
    ),
    "IFNG_6": ("CXCL9", "CXCL10", "IDO1", "IFNG", "HLA-DRA", "STAT1"),
    "CYT_Rooney": ("GZMA", "PRF1"),
    "CD8_effector": ("CD8A", "CD8B", "CXCL9", "CXCL10", "GZMA", "GZMB", "IFNG", "PRF1"),
    "Exhaustion": ("PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT"),
    "TLS": ("CCL19", "CCL21", "CXCL13", "CXCR5", "CD79A", "MS4A1", "MZB1", "SDC1"),
    "Proliferation": ("MKI67", "PCNA", "MCM2", "MCM4", "MCM6", "TOP2A", "TYMS"),
    "Purity_inverse_immune_stromal": (
        "PTPRC", "CD3D", "CD3E", "CD8A", "COL1A1", "COL1A2", "DCN", "VIM",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(frozen: Path, clinical_path: Path):
    clinical = pd.read_csv(clinical_path, sep="\t").set_index("Index", drop=False)
    genes = None
    ordered_ids: list[str] = []
    raw_parts, masks = [], []
    completed: dict[str, list[np.ndarray]] = {model: [] for model in MODELS}
    provenance = []
    for model in MODELS:
        model_ids = []
        for cohort in ("gide", "riaz", "hugo", "rose"):
            path = frozen / f"{cohort}__{model}.npz"
            with np.load(path, allow_pickle=False) as data:
                current_genes = data["genes"].astype(str)
                ids = data["sample_ids"].astype(str)
                if genes is None:
                    genes = current_genes
                elif not np.array_equal(genes, current_genes):
                    raise ValueError(f"gene order mismatch in {path}")
                completed[model].append(np.asarray(data["completed"], dtype=np.float32))
                model_ids.extend(ids.tolist())
                if model == MODELS[0]:
                    raw_parts.append(np.asarray(data["raw"], dtype=np.float32))
                    masks.append(np.asarray(data["measured_gene_mask"], dtype=bool))
                    ordered_ids.extend(ids.tolist())
                provenance.append({
                    "representation": model,
                    "cohort": cohort.title(),
                    "artifact": path.name,
                    "sha256": sha256(path),
                    "genes": len(current_genes),
                    "samples": len(ids),
                    "preprocessing": "log1p(TPM); observed values retained; assay-missing genes decoder-filled",
                })
        if model_ids != ordered_ids:
            raise ValueError(f"sample order mismatch for {model}")
    assert genes is not None
    metadata = clinical.loc[ordered_ids].copy()
    return (
        genes,
        ordered_ids,
        metadata,
        np.concatenate(raw_parts),
        masks,
        {model: np.concatenate(parts) for model, parts in completed.items()},
        pd.DataFrame(provenance),
    )


def rank_signature(
    matrix: np.ndarray, indices: list[int], reference_indices: np.ndarray
) -> np.ndarray:
    if not indices:
        return np.full(len(matrix), np.nan)
    values = matrix[:, reference_indices]
    order = np.argsort(np.argsort(values, axis=1), axis=1)
    position = {gene_index: rank_index for rank_index, gene_index in enumerate(reference_indices)}
    target_positions = [position[index] for index in indices if index in position]
    return order[:, target_positions].mean(axis=1) / max(1, values.shape[1] - 1)


def signature_features(matrix, genes, allowed=None):
    index = {gene.upper(): i for i, gene in enumerate(genes)}
    reference_indices = (
        np.arange(len(genes)) if allowed is None else np.flatnonzero(allowed)
    )
    output, coverage = {}, []
    for name, members in SIGNATURES.items():
        present = [g for g in members if g in index and (allowed is None or allowed[index[g]])]
        output[name] = rank_signature(
            matrix, [index[g] for g in present], reference_indices
        )[:, None]
        coverage.append({
            "signature": name,
            "requested_genes": len(members),
            "available_genes": len(present),
            "missing_genes": len(members) - len(present),
            "available_gene_symbols": ";".join(present),
            "missing_gene_symbols": ";".join(sorted(set(members) - set(present))),
        })
    return output, coverage


def design(train_x, test_x, mode):
    if mode == "topvar":
        selected = np.argsort(np.var(train_x, axis=0))[-min(1000, train_x.shape[1]):]
        train_x, test_x = train_x[:, selected], test_x[:, selected]
    elif mode == "pca":
        selected = np.argsort(np.var(train_x, axis=0))[-min(3000, train_x.shape[1]):]
        scaler = StandardScaler().fit(train_x[:, selected])
        train_scaled = scaler.transform(train_x[:, selected])
        test_scaled = scaler.transform(test_x[:, selected])
        pca = PCA(n_components=min(64, len(train_x) - 1), random_state=SEED).fit(train_scaled)
        return pca.transform(train_scaled), pca.transform(test_scaled)
    scaler = StandardScaler().fit(train_x)
    return scaler.transform(train_x), scaler.transform(test_x)


def fit_probability(x, y, train, test, c_value, mode):
    train_x, test_x = design(x[train], x[test], mode)
    model = LogisticRegression(
        C=c_value, solver="liblinear", class_weight="balanced",
        max_iter=5000, random_state=SEED,
    ).fit(train_x, y[train])
    return model.predict_proba(test_x)[:, 1]


def tune_calibrate(x, y, groups, outer_train, outer_test, mode):
    train_groups = sorted(np.unique(groups[outer_train]))
    candidates = []
    for c_value in CS:
        scores = []
        for held_out in train_groups:
            valid = outer_train & (groups == held_out)
            train = outer_train & ~valid
            scores.append(roc_auc_score(y[valid], fit_probability(x, y, train, valid, c_value, mode)))
        candidates.append((float(np.mean(scores)), c_value))
    selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[0][1]
    outer_indices = np.flatnonzero(outer_train)
    positions = {index: position for position, index in enumerate(outer_indices)}
    oof = np.zeros(len(outer_indices))
    for held_out in train_groups:
        valid = outer_train & (groups == held_out)
        train = outer_train & ~valid
        probability = fit_probability(x, y, train, valid, selected, mode)
        for index, value in zip(np.flatnonzero(valid), probability):
            oof[positions[index]] = value
    test_probability = fit_probability(x, y, outer_train, outer_test, selected, mode)
    eps = 1e-5
    logit = np.log(np.clip(oof, eps, 1 - eps) / np.clip(1 - oof, eps, 1 - eps))

    def objective(offset):
        return log_loss(y[outer_train], 1 / (1 + np.exp(-(logit + offset))))

    offset = minimize_scalar(objective, bounds=(-6, 6), method="bounded").x
    test_logit = np.log(
        np.clip(test_probability, eps, 1 - eps) /
        np.clip(1 - test_probability, eps, 1 - eps)
    )
    return 1 / (1 + np.exp(-(test_logit + offset))), selected


def evaluate(features, y, groups, clean_mask):
    rows = []
    for method, (matrix, mode) in features.items():
        if not np.isfinite(matrix).all():
            continue
        for held_out in COHORTS:
            test = groups == held_out
            probability, selected = tune_calibrate(matrix, y, groups, ~test, test, mode)
            for index, value in zip(np.flatnonzero(test), probability):
                rows.append({
                    "method": method, "held_out_cohort": held_out, "sample_index": index,
                    "label": int(y[index]), "probability": float(value),
                    "selected_C": selected, "pretraining_overlap_clean": bool(clean_mask[index]),
                })
    return pd.DataFrame(rows)


def macro_metrics(frame):
    values = []
    for _, cohort in frame.groupby("held_out_cohort"):
        if cohort.label.nunique() < 2:
            continue
        values.append((
            roc_auc_score(cohort.label, cohort.probability),
            average_precision_score(cohort.label, cohort.probability),
            brier_score_loss(cohort.label, cohort.probability),
        ))
    return np.mean(values, axis=0)


def bootstrap_summary(predictions, replicates=2000):
    rng = np.random.default_rng(SEED)
    rows = []
    clean = predictions.loc[predictions.pretraining_overlap_clean].copy()
    evaluable = [
        cohort for cohort, frame in clean.groupby("held_out_cohort")
        if frame.label.nunique() == 2
    ]
    clean = clean.loc[clean.held_out_cohort.isin(evaluable)]
    scoped = pd.concat([
        predictions.assign(scope="full"),
        clean.assign(scope="overlap_clean_evaluable_cohorts"),
    ])
    for (method, scope), frame in scoped.groupby(["method", "scope"]):
        observed = macro_metrics(frame)
        draws = []
        for _ in range(replicates):
            sampled = []
            for _, stratum in frame.groupby(["held_out_cohort", "label"]):
                sampled.append(stratum.iloc[rng.integers(0, len(stratum), len(stratum))])
            draws.append(macro_metrics(pd.concat(sampled)))
        draws = np.asarray(draws)
        rows.append({
            "method": method, "scope": scope, "n": frame.sample_index.nunique(),
            "macro_auroc": observed[0], "auroc_ci_low": np.quantile(draws[:, 0], .025),
            "auroc_ci_high": np.quantile(draws[:, 0], .975),
            "macro_auprc": observed[1], "auprc_ci_low": np.quantile(draws[:, 1], .025),
            "auprc_ci_high": np.quantile(draws[:, 1], .975),
            "macro_brier": observed[2], "bootstrap_replicates": replicates,
        })
    return pd.DataFrame(rows)


def paired_deltas(predictions, reference="Raw_common_measured", replicates=2000):
    rng = np.random.default_rng(SEED)
    rows = []
    ref = predictions[predictions.method == reference][
        ["sample_index", "probability"]
    ].rename(columns={"probability": "reference_probability"})
    for method in sorted(set(predictions.method) - {reference}):
        frame = predictions[predictions.method == method].merge(ref, on="sample_index")
        reference_frame = frame.copy()
        reference_frame["probability"] = reference_frame.reference_probability
        observed = macro_metrics(frame)[0:2] - macro_metrics(reference_frame)[0:2]
        draws = []
        for _ in range(replicates):
            sampled = []
            for _, stratum in frame.groupby(["held_out_cohort", "label"]):
                sampled.append(stratum.iloc[rng.integers(0, len(stratum), len(stratum))])
            sample = pd.concat(sampled)
            model = macro_metrics(sample)
            reference_frame = sample.copy()
            reference_frame["probability"] = reference_frame.reference_probability
            draws.append(model[:2] - macro_metrics(reference_frame)[:2])
        draws = np.asarray(draws)
        rows.append({
            "method": method, "reference": reference,
            "delta_macro_auroc": observed[0],
            "auroc_ci_low": np.quantile(draws[:, 0], .025),
            "auroc_ci_high": np.quantile(draws[:, 0], .975),
            "delta_macro_auprc": observed[1],
            "auprc_ci_low": np.quantile(draws[:, 1], .025),
            "auprc_ci_high": np.quantile(draws[:, 1], .975),
        })
    return pd.DataFrame(rows)


def eta_squared(matrix, labels):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    matrix = StandardScaler().fit_transform(matrix)
    grand = matrix.mean(axis=0)
    total = np.square(matrix - grand).sum()
    between = sum(
        len(part) * np.square(part.mean(axis=0) - grand).sum()
        for value in np.unique(labels)
        for part in [matrix[labels == value]]
    )
    return float(between / total) if total else 0.0


def nuisance_table(features, groups, cancer, response):
    rows = []
    for name, (matrix, _) in features.items():
        if matrix.shape[1] > 64:
            matrix = PCA(n_components=min(20, matrix.shape[0] - 1), random_state=SEED).fit_transform(
                StandardScaler().fit_transform(matrix)
            )
        centered = matrix.copy()
        for cohort in np.unique(groups):
            centered[groups == cohort] -= centered[groups == cohort].mean(axis=0)
        cohort_eta = eta_squared(matrix, groups)
        cancer_eta = eta_squared(matrix, cancer)
        response_eta = eta_squared(matrix, response)
        residual_response = eta_squared(centered, response)
        rows.append({
            "representation": name, "dimensions_for_geometry": matrix.shape[1],
            "cohort_eta_squared": cohort_eta, "cancer_eta_squared_marginal": cancer_eta,
            "response_eta_squared": response_eta,
            "response_eta_squared_after_cohort_centering": residual_response,
            "residual_fraction": max(0.0, 1 - cohort_eta - residual_response),
            "identifiability_note": "Cancer is confounded with cohort: Rose is BLCA; all other cohorts are SKCM.",
        })
    return pd.DataFrame(rows)


def permutation_floor(predictions, replicates=1000):
    rng = np.random.default_rng(SEED)
    reference = predictions[predictions.method == "Raw_common_measured"].copy()
    draws = []
    for replicate in range(replicates):
        shuffled = reference.copy()
        shuffled["label"] = shuffled.groupby("held_out_cohort").label.transform(
            lambda values: rng.permutation(values.to_numpy())
        )
        metric = macro_metrics(shuffled)
        draws.append({"replicate": replicate, "macro_auroc": metric[0], "macro_auprc": metric[1]})
    return pd.DataFrame(draws)


def plot_variance(nuisance, output):
    frame = nuisance.sort_values("cohort_eta_squared", ascending=False)
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(10, max(5, .35 * len(frame))))
    ax.barh(y - .18, frame.cohort_eta_squared, height=.34, label="Cohort eta-squared")
    ax.barh(
        y + .18, frame.response_eta_squared_after_cohort_centering,
        height=.34, label="Residual response eta-squared",
    )
    ax.set_yticks(y, frame.representation)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of representation variance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "provenance_vs_response_variance.png", dpi=220)
    fig.savefig(output / "provenance_vs_response_variance.pdf")
    plt.close(fig)


def plot_geometry(matrix, groups, response, output):
    coords = PCA(n_components=2, random_state=SEED).fit_transform(StandardScaler().fit_transform(matrix))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for value in np.unique(groups):
        mask = groups == value
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=22, alpha=.8, label=value)
    for value, label in ((0, "Non-responder"), (1, "Responder")):
        mask = response == value
        axes[1].scatter(coords[mask, 0], coords[mask, 1], s=22, alpha=.8, label=label)
    axes[0].set_title("Same embedding, colored by cohort")
    axes[1].set_title("Same embedding, colored by response")
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "same_embedding_cohort_vs_response.png", dpi=220)
    fig.savefig(output / "same_embedding_cohort_vs_response.pdf")
    plt.close(fig)
    return pd.DataFrame({
        "sample_index": np.arange(len(matrix)), "PC1": coords[:, 0], "PC2": coords[:, 1],
        "cohort": groups, "response": response,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--historical-predictions", type=Path, required=True)
    parser.add_argument("--overlap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    genes, ids, metadata, raw, masks, completed, provenance = load_inputs(
        args.frozen, args.clinical
    )
    y = metadata.label.to_numpy(int)
    groups = metadata.cohort.to_numpy(str)
    cancer = metadata.cancer_type.to_numpy(str)
    common = np.logical_and.reduce(masks)
    overlap = pd.read_csv(args.overlap, sep="\t")
    clean_ids = set(
        overlap.loc[
            ~overlap.present_in_archs4_pretraining_metadata.astype(bool),
            "evaluation_sample_id",
        ].astype(str)
    )
    clean_mask = np.asarray([sample_id in clean_ids for sample_id in ids])

    features = {
        "Raw_common_measured": (raw[:, common], "topvar"),
        "PCA64_raw_common": (raw[:, common], "pca"),
    }
    limited_signatures, coverage = signature_features(raw, genes, common)
    features.update({f"Signature_assay_limited__{k}": (v, "fixed") for k, v in limited_signatures.items()})
    for model, matrix in completed.items():
        features[f"{model}__completed"] = (matrix, "topvar")
    for model in ("Txn_Jatin", "BulkFormer_147M"):
        harmonized, _ = signature_features(completed[model], genes)
        features.update({f"Signature_harmonized_{model}__{k}": (v, "fixed") for k, v in harmonized.items()})

    predictions = evaluate(features, y, groups, clean_mask)
    predictions["sample_id"] = predictions.sample_index.map(dict(enumerate(ids)))
    predictions.to_csv(args.output / "loco_predictions.csv", index=False)
    summary = bootstrap_summary(predictions, args.bootstraps)
    summary.to_csv(args.output / "loco_summary_with_ci.csv", index=False)
    paired_deltas(predictions, replicates=args.bootstraps).to_csv(
        args.output / "paired_deltas_vs_raw.csv", index=False
    )
    permutation_floor(predictions).to_csv(args.output / "permutation_floor.csv", index=False)

    coverage_rows = []
    for cohort, mask in zip(COHORTS, masks):
        _, rows = signature_features(raw, genes, mask)
        for row in rows:
            row["cohort"] = cohort
            coverage_rows.append(row)
    pd.DataFrame(coverage_rows).to_csv(args.output / "signature_panel_coverage.csv", index=False)

    geometry_features = {
        "Raw_common_PCA20": (raw[:, common], "pca"),
        **{f"{model}_completed_PCA20": (matrix, "pca") for model, matrix in completed.items()},
        **{f"Assay_limited_{k}": (v, "fixed") for k, v in limited_signatures.items()},
    }
    for model in ("Txn_Jatin", "BulkFormer_147M"):
        scores, _ = signature_features(completed[model], genes)
        geometry_features.update({f"Harmonized_{model}_{k}": (v, "fixed") for k, v in scores.items()})
    nuisance = nuisance_table(geometry_features, groups, cancer, y)
    nuisance.to_csv(args.output / "representation_nuisance_diagnostics.csv", index=False)
    plot_variance(nuisance, args.output)
    coords = plot_geometry(completed["Txn_Jatin"], groups, y, args.output)
    coords["sample_id"] = coords.sample_index.map(dict(enumerate(ids)))
    coords.to_csv(args.output / "espresso_geometry_coordinates.csv", index=False)
    provenance.to_csv(args.output / "representation_provenance.csv", index=False)

    historical = pd.read_csv(args.historical_predictions)
    historical.to_csv(args.output / "phase0_frozen_decoder_null_predictions.csv", index=False)
    protocol = {
        "seed": SEED,
        "patients": len(ids),
        "responders": int(y.sum()),
        "cohorts": list(COHORTS),
        "outer_validation": "leave-one-cohort-out",
        "inner_tuning": "leave-one-training-cohort-out",
        "calibration": "intercept-only monotone calibration fitted on inner OOF predictions",
        "primary_metrics": ["macro_auroc", "macro_auprc"],
        "bootstrap": f"{args.bootstraps} cohort-and-label-stratified paired replicates",
        "permutation_floor": "1000 response-label shuffles within held-out cohort",
        "pretraining_overlap_clean_definition": "patients absent from exact human ARCHS4 accession match",
        "signature_scoring": "within-sample mean rank over available signature genes",
        "signature_definitions": {key: list(value) for key, value in SIGNATURES.items()},
        "cancer_variance_warning": "Cancer is not separately identifiable from cohort because Rose is BLCA and the other cohorts are SKCM.",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(summary.query("scope == 'full'").sort_values("macro_auroc", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
