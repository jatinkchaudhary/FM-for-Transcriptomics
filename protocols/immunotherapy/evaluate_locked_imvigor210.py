#!/usr/bin/env python3
"""Single-shot preregistered evaluation on ExperimentHub EH6677."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from experimenthub import ExperimentHubRegistry
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from run_provenance_not_response import COHORTS, SEED, load_inputs


def extract_eh6677(cache):
    record = ExperimentHubRegistry(cache_dir=cache).load("EH6677")
    obj = record["Mariathasan2018_PDL1_treatment"]["attributes"]
    genes = np.asarray(obj["NAMES"]["data"], dtype=str)
    coldata = obj["colData"]["attributes"]
    colnames = coldata["listData"]["attributes"]["names"]["data"]
    columns = {
        name: np.asarray(value["data"])
        for name, value in zip(colnames, coldata["listData"]["data"])
    }
    columns["sample_id"] = np.asarray(coldata["rownames"]["data"], dtype=str)
    clinical = pd.DataFrame(columns)
    assays = obj["assays"]["attributes"]["data"]["attributes"]["listData"]
    assay_names = list(assays["attributes"]["names"]["data"])
    tpm_assay = assays["data"][assay_names.index("tpm")]
    tpm = np.column_stack([np.asarray(column["data"], dtype=float) for column in tpm_assay["data"]])
    keep = clinical.BOR.astype(str).isin(["R", "NR"]).to_numpy()
    return genes, tpm[:, keep].T, clinical.loc[keep].reset_index(drop=True)


def impute(api, model_genes, source_genes, tpm, sample_ids, batch_size=16):
    source_index = {gene.upper(): i for i, gene in enumerate(source_genes)}
    aligned = np.zeros((len(tpm), len(model_genes)), dtype=np.float32)
    measured = np.zeros(len(model_genes), dtype=bool)
    for model_i, gene in enumerate(model_genes):
        source_i = source_index.get(gene.upper())
        if source_i is not None:
            aligned[:, model_i] = np.log1p(np.maximum(tpm[:, source_i], 0))
            measured[model_i] = True
    completed = np.full_like(aligned, np.nan)
    session = requests.Session()
    for start in range(0, len(aligned), batch_size):
        end = min(start + batch_size, len(aligned))
        values = aligned[start:end]
        missing = np.broadcast_to(~measured, values.shape)
        payload = {
            "model": "Txn_Jatin", "genes": model_genes.tolist(),
            "samples": sample_ids[start:end].tolist(), "matrix": values.T.tolist(),
            "missing": missing.T.tolist(), "input_scale": "log1p",
        }
        for attempt in range(4):
            try:
                response = session.post(f"{api}/api/impute", json=payload, timeout=(30, 900))
                response.raise_for_status()
                result = response.json()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        completed[start:end] = np.asarray(
            [[np.nan if value is None else float(value) for value in row]
             for row in result["imputed"]], dtype=np.float32
        ).T
        print(f"IMvigor210 imputation: {end}/{len(aligned)}", flush=True)
    return aligned, completed, measured


def domain_standardize(train_x, groups):
    pooled_mean = train_x.mean(axis=0)
    pooled_sd = np.maximum(train_x.std(axis=0), 1e-6)
    output = np.zeros_like(train_x)
    for group in np.unique(groups):
        mask = groups == group
        mean = train_x[mask].mean(axis=0)
        sd = np.maximum(train_x[mask].std(axis=0), 1e-6)
        output[mask] = (train_x[mask] - mean) / sd * pooled_sd + pooled_mean
    return output


def fit_locked(dev_x, external_x, y, groups):
    selected = np.argsort(np.var(dev_x, axis=0))[-min(1000, dev_x.shape[1]):]
    scaler = StandardScaler().fit(dev_x[:, selected])
    dev_scaled = scaler.transform(dev_x[:, selected])
    external_scaled = scaler.transform(external_x[:, selected])
    pca = PCA(n_components=min(64, len(dev_x) - 1), random_state=SEED).fit(dev_scaled)
    dev_pca = domain_standardize(pca.transform(dev_scaled), groups)
    external_pca = pca.transform(external_scaled)
    candidates, oof_by_c = [], {}
    for c_value in (.01, .1, 1., 10.):
        oof = np.zeros(len(y))
        scores = []
        for held_out in COHORTS:
            valid = groups == held_out
            train = ~valid
            model = LogisticRegression(
                C=c_value, solver="liblinear", class_weight="balanced",
                max_iter=5000, random_state=SEED,
            ).fit(dev_pca[train], y[train])
            oof[valid] = model.predict_proba(dev_pca[valid])[:, 1]
            scores.append(roc_auc_score(y[valid], oof[valid]))
        candidates.append((np.mean(scores), c_value))
        oof_by_c[c_value] = oof
    selected_c = sorted(candidates, key=lambda item: (-item[0], item[1]))[0][1]
    oof = np.clip(oof_by_c[selected_c], 1e-5, 1 - 1e-5)
    oof_logit = np.log(oof / (1 - oof))

    def objective(offset):
        return log_loss(y, 1 / (1 + np.exp(-(oof_logit + offset))))

    offset = float(minimize_scalar(objective, bounds=(-6, 6), method="bounded").x)
    model = LogisticRegression(
        C=selected_c, solver="liblinear", class_weight="balanced",
        max_iter=5000, random_state=SEED,
    ).fit(dev_pca, y)
    probability = np.clip(model.predict_proba(external_pca)[:, 1], 1e-5, 1 - 1e-5)
    logit = np.log(probability / (1 - probability))
    return 1 / (1 + np.exp(-(logit + offset))), selected_c, offset, selected


def bootstrap(labels, model_p, raw_p, replicates=2000):
    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(replicates):
        indices = np.concatenate([
            rng.choice(np.flatnonzero(labels == value), (labels == value).sum(), replace=True)
            for value in (0, 1)
        ])
        row = {}
        for name, probability in (("espresso", model_p), ("raw", raw_p)):
            row[f"{name}_auroc"] = roc_auc_score(labels[indices], probability[indices])
            row[f"{name}_auprc"] = average_precision_score(labels[indices], probability[indices])
        row["delta_auroc"] = row["espresso_auroc"] - row["raw_auroc"]
        row["delta_auprc"] = row["espresso_auprc"] - row["raw_auprc"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--eh-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model_genes, _, metadata, dev_raw, masks, completed, provenance = load_inputs(
        args.frozen, args.clinical
    )
    external_genes, external_tpm, external_clinical = extract_eh6677(args.eh_cache)
    external_raw, external_completed, external_measured = impute(
        args.api, model_genes, external_genes, external_tpm,
        external_clinical.sample_id.astype(str).to_numpy(),
    )
    y_dev = metadata.label.to_numpy(int)
    groups = metadata.cohort.to_numpy(str)
    y_external = external_clinical.BOR.eq("R").to_numpy(int)
    finite = np.isfinite(completed["Txn_Jatin"]).all(axis=0) & np.isfinite(external_completed).all(axis=0)
    espresso_p, espresso_c, espresso_offset, selected = fit_locked(
        completed["Txn_Jatin"][:, finite], external_completed[:, finite], y_dev, groups
    )
    raw_common = np.logical_and.reduce(masks) & external_measured
    raw_p, raw_c, raw_offset, raw_selected = fit_locked(
        dev_raw[:, raw_common], external_raw[:, raw_common], y_dev, groups
    )
    predictions = external_clinical[["sample_id", "pat_id", "BOR", "TMB"]].copy()
    predictions["label"] = y_external
    predictions["espresso_probability"] = espresso_p
    predictions["raw_probability"] = raw_p
    predictions.to_csv(args.output / "locked_imvigor210_predictions.csv", index=False)
    draws = bootstrap(y_external, espresso_p, raw_p)
    draws.to_csv(args.output / "locked_imvigor210_bootstrap.csv", index=False)

    def metric(probability, prefix):
        return {
            "method": prefix, "n": len(y_external), "responders": int(y_external.sum()),
            "auroc": roc_auc_score(y_external, probability),
            "auprc": average_precision_score(y_external, probability),
            "brier": brier_score_loss(y_external, probability),
            "auroc_ci_low": draws[f"{prefix}_auroc"].quantile(.025),
            "auroc_ci_high": draws[f"{prefix}_auroc"].quantile(.975),
            "auprc_ci_low": draws[f"{prefix}_auprc"].quantile(.025),
            "auprc_ci_high": draws[f"{prefix}_auprc"].quantile(.975),
        }
    summary = pd.DataFrame([metric(espresso_p, "espresso"), metric(raw_p, "raw")])
    summary["delta_auroc_vs_raw"] = [draws.delta_auroc.mean(), 0]
    summary["delta_auroc_ci_low"] = [draws.delta_auroc.quantile(.025), 0]
    summary["delta_auroc_ci_high"] = [draws.delta_auroc.quantile(.975), 0]
    summary["delta_auprc_vs_raw"] = [draws.delta_auprc.mean(), 0]
    summary["delta_auprc_ci_low"] = [draws.delta_auprc.quantile(.025), 0]
    summary["delta_auprc_ci_high"] = [draws.delta_auprc.quantile(.975), 0]
    summary.to_csv(args.output / "locked_imvigor210_summary.csv", index=False)
    (args.output / "locked_imvigor210_protocol.json").write_text(json.dumps({
        "resource": "ExperimentHub EH6677", "samples": len(y_external),
        "responders": int(y_external.sum()), "external_genes": len(external_genes),
        "model_genes": len(model_genes), "external_measured_model_genes": int(external_measured.sum()),
        "espresso_finite_genes": int(finite.sum()), "raw_common_genes": int(raw_common.sum()),
        "espresso_selected_C": espresso_c, "espresso_calibration_offset": espresso_offset,
        "raw_selected_C": raw_c, "raw_calibration_offset": raw_offset,
        "feature_selection": "top 1000 development variance genes",
        "adaptation": "diagonal training-domain CORAL; external statistics unused",
        "external_labels_used_for_fit": False, "seed": SEED,
        "accession_overlap": "EH6677 exposes internal SAM identifiers, not GEO/archive accessions; exact ARCHS4 accession matching is not identifiable from this distribution.",
    }, indent=2) + "\n")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
