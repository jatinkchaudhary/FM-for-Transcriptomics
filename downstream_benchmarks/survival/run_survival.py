#!/usr/bin/env python3
"""Patient-grouped TCGA-CDR survival benchmark using frozen representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


ALPHAS = np.logspace(-2, 1, 7)


def cindex(event, time, risk):
    return float(concordance_index_censored(event.astype(bool), time, risk)[0])


def stratified_bootstrap(event, time, predictions, repetitions=2000, seed=42):
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(event == value) for value in np.unique(event)]
    scores = []
    for _ in range(repetitions):
        index = np.concatenate(
            [rng.choice(group, len(group), replace=True) for group in strata]
        )
        try:
            scores.append(cindex(event[index], time[index], predictions[index]))
        except ValueError:
            continue
    point = cindex(event, time, predictions)
    return point, *np.quantile(scores, [0.025, 0.975])


def nested_oof(x, event, time, strata):
    outer = StratifiedKFold(5, shuffle=True, random_state=42)
    splits = list(outer.split(x, strata))

    def run_fold(fold, train, test):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train])
        x_test = scaler.transform(x[test])
        inner_strata = strata[train]
        inner = StratifiedKFold(3, shuffle=True, random_state=100 + fold)
        alpha_scores = np.zeros(len(ALPHAS))
        alpha_counts = np.zeros(len(ALPHAS))
        for fit, validation in inner.split(x_train, inner_strata):
            # Coxnet overflows below 0.1 for the 15k-feature raw baseline. Those
            # candidates remain in the global grid but are numerically invalid
            # for p > 5,000 and are excluded before repeated failed fits.
            offset = 2 if x_train.shape[1] > 5000 else 0
            while True:
                try:
                    path = CoxnetSurvivalAnalysis(
                        l1_ratio=0.5,
                        alphas=ALPHAS[offset:],
                        max_iter=100000,
                        fit_baseline_model=False,
                    ).fit(
                        x_train[fit],
                        Surv.from_arrays(event[train][fit], time[train][fit]),
                    )
                    break
                except ArithmeticError:
                    offset += 1
                    if offset == len(ALPHAS):
                        raise
            for j in range(path.coef_.shape[1]):
                risk = x_train[validation] @ path.coef_[:, j]
                alpha_scores[offset + j] += cindex(
                    event[train][validation], time[train][validation], risk
                )
                alpha_counts[offset + j] += 1
        valid_scores = np.divide(
            alpha_scores,
            alpha_counts,
            out=np.full_like(alpha_scores, -np.inf),
            where=alpha_counts > 0,
        )
        best = int(np.argmax(valid_scores))
        final_index = best
        while True:
            try:
                model = CoxnetSurvivalAnalysis(
                    l1_ratio=0.5,
                    alphas=[ALPHAS[final_index]],
                    max_iter=100000,
                    fit_baseline_model=False,
                ).fit(x_train, Surv.from_arrays(event[train], time[train]))
                break
            except ArithmeticError:
                final_index += 1
                if final_index == len(ALPHAS):
                    raise
        return test, model.predict(x_test), float(ALPHAS[final_index])

    fitted = Parallel(n_jobs=min(5, len(splits)), prefer="processes")(
        delayed(run_fold)(fold, train, test)
        for fold, (train, test) in enumerate(splits)
    )
    prediction = np.full(len(x), np.nan)
    chosen = []
    for test, risk, alpha in fitted:
        prediction[test] = risk
        chosen.append(alpha)
    return prediction, chosen


def load_embedding(path):
    payload = np.load(path, allow_pickle=False)
    ids = payload["sample_ids"].astype(str)
    return pd.DataFrame(payload["embeddings"], index=ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--gdc-map", required=True, type=Path)
    parser.add_argument("--cdr", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata)
    gdc = pd.read_csv(args.gdc_map)
    cdr = pd.read_excel(args.cdr, sheet_name="TCGA-CDR")
    cdr = cdr.rename(columns={"bcr_patient_barcode": "patient_id", "type": "cancer"})
    sample_col = next(c for c in ("file_id", "sample_id", "id") if c in metadata)
    metadata["sample_id"] = metadata[sample_col].astype(str)
    file_patient = (
        gdc[["file_id", "patient_id"]].drop_duplicates("file_id").set_index("file_id")["patient_id"]
    )
    metadata["patient_id"] = metadata["sample_id"].map(file_patient)
    metadata = metadata.dropna(subset=["patient_id"]).drop_duplicates("patient_id")
    cohort = metadata.merge(cdr, on="patient_id", how="inner")
    project_col = next(c for c in ("project_id", "cancer_type") if c in cohort)
    cohort["cancer"] = cohort[project_col].str.replace("TCGA-", "", regex=False)
    gate = pd.read_csv(args.gate)
    allowed = set(gate.loc[gate["status"].eq("pass"), "model"])
    models = {
        p.stem: load_embedding(p)
        for p in args.embedding_dir.glob("*.npz")
        if p.stem in allowed
    }
    priority = ["Txn_Jatin", "BulkFormer_147M", "PCA64", "raw_log1p_TPM"]
    models = dict(sorted(models.items(), key=lambda item: priority.index(item[0]) if item[0] in priority else 99))
    result_path = args.output / "survival_cindex.csv"
    prediction_path = args.output / "oof_predictions.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    predictions = pd.read_csv(prediction_path).to_dict("records") if prediction_path.exists() else []
    completed = {(row["endpoint"], row["scope"], row["model"]) for row in rows}
    for endpoint in ("OS", "PFI"):
        endpoint_data = cohort.loc[
            cohort[endpoint].isin([0, 1])
            & pd.to_numeric(cohort[f"{endpoint}.time"], errors="coerce").gt(0)
        ].copy()
        endpoint_data["event"] = endpoint_data[endpoint].astype(bool)
        endpoint_data["time"] = pd.to_numeric(endpoint_data[f"{endpoint}.time"])
        scopes = ["pan_cancer"] + sorted(endpoint_data["cancer"].unique())
        for scope in scopes:
            subset = (
                endpoint_data
                if scope == "pan_cancer"
                else endpoint_data.loc[endpoint_data["cancer"].eq(scope)]
            )
            if len(subset) < 80 or subset["event"].sum() < 20:
                continue
            ids = subset["sample_id"].astype(str)
            for model_name, embedding in models.items():
                if (endpoint, scope, model_name) in completed:
                    continue
                shared = ids.isin(embedding.index)
                data = subset.loc[shared].copy()
                x = embedding.loc[data["sample_id"]].to_numpy(dtype=np.float32)
                event = data["event"].to_numpy()
                time = data["time"].to_numpy()
                if scope == "pan_cancer":
                    strata = data["cancer"].astype(str) + "_" + data["event"].astype(int).astype(str)
                else:
                    strata = data["event"].astype(int).astype(str)
                risk, chosen = nested_oof(x, event, time, strata.to_numpy())
                point, low, high = stratified_bootstrap(event, time, risk)
                rows.append(
                    {
                        "endpoint": endpoint,
                        "scope": scope,
                        "model": model_name,
                        "n": len(data),
                        "events": int(event.sum()),
                        "cindex": point,
                        "ci_low": low,
                        "ci_high": high,
                        "selected_alphas": json.dumps(chosen),
                    }
                )
                predictions.extend(
                    {
                        "endpoint": endpoint,
                        "scope": scope,
                        "model": model_name,
                        "sample_id": sid,
                        "event": int(ev),
                        "time": tm,
                        "risk": value,
                    }
                    for sid, ev, tm, value in zip(data["sample_id"], event, time, risk)
                )
                pd.DataFrame(rows).to_csv(result_path, index=False)
                pd.DataFrame(predictions).to_csv(prediction_path, index=False)
                print(rows[-1], flush=True)
    pred = pd.DataFrame(predictions)
    deltas = []
    for (endpoint, scope), group in pred.groupby(["endpoint", "scope"]):
        wide = group.pivot(index="sample_id", columns="model", values="risk")
        if "Txn_Jatin" not in wide:
            continue
        truth = group.drop_duplicates("sample_id").set_index("sample_id").loc[wide.index]
        event, time = truth["event"].to_numpy(bool), truth["time"].to_numpy(float)
        rng = np.random.default_rng(42)
        strata = [np.flatnonzero(event == value) for value in np.unique(event)]
        for model in wide.columns:
            if model == "Txn_Jatin":
                continue
            valid = wide[["Txn_Jatin", model]].dropna()
            loc = wide.index.get_indexer(valid.index)
            observed = cindex(event[loc], time[loc], valid["Txn_Jatin"]) - cindex(
                event[loc], time[loc], valid[model]
            )
            boot = []
            local_event, local_time = event[loc], time[loc]
            local_strata = [np.flatnonzero(local_event == value) for value in np.unique(local_event)]
            for _ in range(1000):
                index = np.concatenate([rng.choice(s, len(s), replace=True) for s in local_strata])
                boot.append(
                    cindex(local_event[index], local_time[index], valid["Txn_Jatin"].to_numpy()[index])
                    - cindex(local_event[index], local_time[index], valid[model].to_numpy()[index])
                )
            low, high = np.quantile(boot, [0.025, 0.975])
            deltas.append(
                {"endpoint": endpoint, "scope": scope, "competitor": model, "delta_cindex": observed, "ci_low": low, "ci_high": high}
            )
    delta = pd.DataFrame(deltas)
    delta.to_csv(args.output / "paired_delta_vs_espresso.csv", index=False)
    if not delta.empty:
        plot = delta.loc[delta["scope"].eq("pan_cancer")].copy()
        fig, ax = plt.subplots(figsize=(8, max(3, len(plot) * 0.45)))
        y = np.arange(len(plot))
        ax.errorbar(plot["delta_cindex"], y, xerr=[plot["delta_cindex"] - plot["ci_low"], plot["ci_high"] - plot["delta_cindex"]], fmt="o", color="#0072B2")
        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(y, plot["endpoint"] + " vs " + plot["competitor"])
        ax.set_xlabel("Paired delta C-index (ESPRESSO - comparator)")
        fig.tight_layout()
        fig.savefig(args.output / "survival_forest.png", dpi=240)
        fig.savefig(args.output / "survival_forest.pdf")


if __name__ == "__main__":
    main()
