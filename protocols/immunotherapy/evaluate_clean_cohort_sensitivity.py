#!/usr/bin/env python3
"""LOCO sensitivity excluding all patients found in pretraining metadata."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COHORTS = ("Gide", "Riaz", "Hugo", "Rose")
CLEAN_COHORTS = ("Gide", "Riaz", "Rose")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("ici_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results"
    evaluation = load_module(root / "scripts" / "evaluate_ici_transfer.py")
    clinical = pd.read_csv(root / "prepared" / "clinical_harmonized.tsv", sep="\t")
    clinical = clinical.set_index("Index", drop=False)

    loaded = {}
    masks = []
    for cohort in COHORTS:
        with np.load(
            results / f"{cohort.lower()}_txn_embeddings.npz", allow_pickle=False
        ) as data:
            loaded[cohort] = {
                "sample_ids": data["sample_ids"].astype(str),
                "txn_context_mean": data["txn_context_mean"].astype(np.float32),
                "txn_hidden_all": data["txn_hidden_all"].astype(np.float32),
                "raw": data["aligned_log1p_tpm"].astype(np.float32),
                "completed": data["txn_completed_log1p_tpm"].astype(np.float32),
            }
            masks.append(data["measured_gene_mask"].astype(bool))
    common_mask = np.logical_and.reduce(masks)
    hallmark = pd.read_csv(results / "hallmark_common_features.csv", index_col=0)
    completed_hallmark = pd.read_csv(
        results / "txn_completed_hallmark_features.csv", index_col=0
    )

    metadata_parts = []
    feature_parts = {
        "txn_context_mean": [],
        "txn_hidden_all": [],
        "raw_common_topvar": [],
        "hallmark_common": [],
        "txn_completed_topvar": [],
        "txn_completed_hallmark": [],
    }
    for cohort in CLEAN_COHORTS:
        data = loaded[cohort]
        ids = data["sample_ids"]
        metadata_parts.append(clinical.loc[ids])
        feature_parts["txn_context_mean"].append(data["txn_context_mean"])
        feature_parts["txn_hidden_all"].append(data["txn_hidden_all"])
        feature_parts["raw_common_topvar"].append(data["raw"][:, common_mask])
        feature_parts["hallmark_common"].append(
            hallmark.loc[ids].to_numpy(dtype=np.float32)
        )
        feature_parts["txn_completed_topvar"].append(data["completed"])
        feature_parts["txn_completed_hallmark"].append(
            completed_hallmark.loc[ids].to_numpy(dtype=np.float32)
        )
    metadata = pd.concat(metadata_parts)
    features = {
        key: np.concatenate(parts, axis=0) for key, parts in feature_parts.items()
    }
    y = metadata["label"].to_numpy(dtype=int)
    groups = metadata["cohort"].to_numpy(dtype=str)
    sample_ids = metadata["Index"].to_numpy(dtype=str)

    prediction_records = []
    tuning_records = []
    for method, X in features.items():
        for held_out in CLEAN_COHORTS:
            train = groups != held_out
            test = groups == held_out
            selected_c, tuning = evaluation.choose_c(
                X[train], y[train], groups[train], method
            )
            for record in tuning:
                tuning_records.append(
                    {
                        "method": method,
                        "held_out_cohort": held_out,
                        **record,
                        "selected": bool(record["C"] == selected_c),
                    }
                )
            model = evaluation.estimator(method, selected_c)
            model.fit(X[train], y[train])
            probability = model.predict_proba(X[test])[:, 1]
            for i, value in zip(np.where(test)[0], probability):
                prediction_records.append(
                    {
                        "method": method,
                        "held_out_cohort": held_out,
                        "sample_id": sample_ids[i],
                        "label": int(y[i]),
                        "probability": float(value),
                        "selected_C": float(selected_c),
                    }
                )
            print(f"{method} held_out={held_out} C={selected_c}", flush=True)
    predictions = pd.DataFrame(prediction_records)
    predictions.to_csv(
        results / "clean_no_pretraining_overlap_predictions.csv", index=False
    )
    pd.DataFrame(tuning_records).to_csv(
        results / "clean_no_pretraining_overlap_tuning.csv", index=False
    )

    metrics = []
    for method, frame in predictions.groupby("method"):
        values = evaluation.metric_values(
            frame["label"].to_numpy(), frame["probability"].to_numpy()
        )
        ci = evaluation.bootstrap_ci(
            frame["label"].to_numpy(), frame["probability"].to_numpy()
        )
        per_cohort = []
        for cohort, cohort_frame in frame.groupby("held_out_cohort"):
            cohort_values = evaluation.metric_values(
                cohort_frame["label"].to_numpy(),
                cohort_frame["probability"].to_numpy(),
            )
            per_cohort.append(cohort_values["auroc"])
        metrics.append(
            {
                "method": method,
                "n": int(len(frame)),
                "responders": int(frame["label"].sum()),
                "macro_auroc": float(np.mean(per_cohort)),
                **values,
                **ci,
            }
        )
    metrics = pd.DataFrame(metrics).sort_values("auroc", ascending=False)
    metrics.to_csv(
        results / "clean_no_pretraining_overlap_summary.csv", index=False
    )
    deltas = []
    for method in features:
        if method == "raw_common_topvar":
            continue
        delta = evaluation.paired_delta_bootstrap(
            predictions, method, "raw_common_topvar"
        )
        deltas.append(delta)
    pd.DataFrame(deltas).to_csv(
        results / "clean_no_pretraining_overlap_deltas.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    plot = metrics.sort_values("auroc")
    y_pos = np.arange(len(plot))
    ax.barh(
        y_pos,
        plot["auroc"],
        color=[
            evaluation.COLORS.get(method, "#6C757D") for method in plot["method"]
        ],
    )
    ax.errorbar(
        plot["auroc"],
        y_pos,
        xerr=[
            plot["auroc"] - plot["auroc_ci_low"],
            plot["auroc_ci_high"] - plot["auroc"],
        ],
        fmt="none",
        ecolor="#222222",
        capsize=3,
    )
    ax.axvline(0.5, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks(
        y_pos, [evaluation.DISPLAY[x] for x in plot["method"]], fontsize=8
    )
    ax.set_xlim(0.2, 0.8)
    ax.set_xlabel("Pooled clean-cohort LOCO AUROC (95% bootstrap CI)")
    ax.set_title("Sensitivity excluding Hugo pretraining overlap")
    fig.tight_layout()
    fig.savefig(
        root / "figures" / "clean_no_pretraining_overlap_auroc.png", dpi=220
    )
    plt.close(fig)
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
