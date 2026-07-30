#!/usr/bin/env python3
"""Parallel uncertainty aggregation and final figures for the provenance audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from run_provenance_not_response import (
    COHORTS,
    SEED,
    load_inputs,
    nuisance_table,
    plot_geometry,
    plot_variance,
    signature_features,
)


def macro_arrays(labels, probabilities, cohorts):
    values = []
    for cohort in np.unique(cohorts):
        mask = cohorts == cohort
        if np.unique(labels[mask]).size < 2:
            continue
        values.append((
            roc_auc_score(labels[mask], probabilities[mask]),
            average_precision_score(labels[mask], probabilities[mask]),
            brier_score_loss(labels[mask], probabilities[mask]),
        ))
    return np.mean(values, axis=0)


def bootstrap_method(method, scope, frame, replicates):
    labels = frame.label.to_numpy(int)
    probabilities = frame.probability.to_numpy(float)
    cohorts = frame.held_out_cohort.to_numpy(str)
    strata = [
        np.flatnonzero((cohorts == cohort) & (labels == label))
        for cohort in np.unique(cohorts) for label in (0, 1)
        if np.any((cohorts == cohort) & (labels == label))
    ]
    rng = np.random.default_rng(SEED + sum(map(ord, method + scope)))
    observed = macro_arrays(labels, probabilities, cohorts)
    draws = np.empty((replicates, 3))
    for replicate in range(replicates):
        indices = np.concatenate([
            stratum[rng.integers(0, len(stratum), len(stratum))] for stratum in strata
        ])
        draws[replicate] = macro_arrays(labels[indices], probabilities[indices], cohorts[indices])
    return {
        "method": method, "scope": scope, "n": frame.sample_index.nunique(),
        "macro_auroc": observed[0], "auroc_ci_low": np.quantile(draws[:, 0], .025),
        "auroc_ci_high": np.quantile(draws[:, 0], .975),
        "macro_auprc": observed[1], "auprc_ci_low": np.quantile(draws[:, 1], .025),
        "auprc_ci_high": np.quantile(draws[:, 1], .975),
        "macro_brier": observed[2], "bootstrap_replicates": replicates,
    }


def paired_method(method, frame, reference_frame, replicates):
    paired = frame.merge(
        reference_frame[["sample_index", "probability"]],
        on="sample_index", suffixes=("", "_reference"),
    )
    labels = paired.label.to_numpy(int)
    cohorts = paired.held_out_cohort.to_numpy(str)
    model_p = paired.probability.to_numpy(float)
    ref_p = paired.probability_reference.to_numpy(float)
    observed = macro_arrays(labels, model_p, cohorts) - macro_arrays(labels, ref_p, cohorts)
    strata = [
        np.flatnonzero((cohorts == cohort) & (labels == label))
        for cohort in np.unique(cohorts) for label in (0, 1)
        if np.any((cohorts == cohort) & (labels == label))
    ]
    rng = np.random.default_rng(SEED + sum(map(ord, method)))
    draws = np.empty((replicates, 2))
    for replicate in range(replicates):
        indices = np.concatenate([
            stratum[rng.integers(0, len(stratum), len(stratum))] for stratum in strata
        ])
        draws[replicate] = (
            macro_arrays(labels[indices], model_p[indices], cohorts[indices])[:2] -
            macro_arrays(labels[indices], ref_p[indices], cohorts[indices])[:2]
        )
    return {
        "method": method, "reference": reference_frame.method.iloc[0],
        "delta_macro_auroc": observed[0],
        "auroc_ci_low": np.quantile(draws[:, 0], .025),
        "auroc_ci_high": np.quantile(draws[:, 0], .975),
        "delta_macro_auprc": observed[1],
        "auprc_ci_low": np.quantile(draws[:, 1], .025),
        "auprc_ci_high": np.quantile(draws[:, 1], .975),
        "bootstrap_replicates": replicates,
    }


def summarize(predictions, output, prefix, replicates, jobs):
    clean = predictions[predictions.pretraining_overlap_clean].copy()
    evaluable = [
        cohort for cohort, frame in clean.groupby("held_out_cohort")
        if frame.label.nunique() == 2
    ]
    scopes = {
        "full": predictions,
        "overlap_clean_evaluable_cohorts": clean[clean.held_out_cohort.isin(evaluable)],
    }
    tasks = [
        (method, scope, frame)
        for scope, scoped in scopes.items()
        for method, frame in scoped.groupby("method")
    ]
    summary = Parallel(n_jobs=jobs, verbose=5)(
        delayed(bootstrap_method)(method, scope, frame, replicates)
        for method, scope, frame in tasks
    )
    pd.DataFrame(summary).to_csv(output / f"{prefix}_summary_with_ci.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--historical-predictions", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.prediction_dir / "loco_predictions.csv")
    summarize(predictions, args.output, "loco", args.bootstraps, args.jobs)
    reference = predictions[predictions.method == "Raw_common_measured"]
    deltas = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(paired_method)(method, frame, reference, args.bootstraps)
        for method, frame in predictions.groupby("method")
        if method != "Raw_common_measured"
    )
    pd.DataFrame(deltas).to_csv(args.output / "paired_deltas_vs_raw.csv", index=False)

    invariance = pd.read_csv(args.prediction_dir / "invariance_loco_predictions.csv")
    summarize(invariance, args.output, "invariance_loco", args.bootstraps, args.jobs)
    adaptation_deltas = []
    for representation, representation_frame in invariance.groupby("representation"):
        reference = representation_frame[representation_frame.adaptation == "unadapted"].copy()
        reference["method"] = f"{representation}__unadapted"
        for adaptation, frame in representation_frame.groupby("adaptation"):
            if adaptation == "unadapted":
                continue
            adaptation_deltas.append((representation, adaptation, frame, reference))
    rows = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(paired_method)(frame.method.iloc[0], frame, reference, args.bootstraps)
        for _, _, frame, reference in adaptation_deltas
    )
    for row, (representation, adaptation, _, _) in zip(rows, adaptation_deltas):
        row["representation"] = representation
        row["adaptation"] = adaptation
    pd.DataFrame(rows).to_csv(args.output / "invariance_paired_deltas.csv", index=False)

    genes, ids, metadata, raw, masks, completed, provenance = load_inputs(
        args.frozen, args.clinical
    )
    common = np.logical_and.reduce(masks)
    limited, _ = signature_features(raw, genes, common)
    geometry = {
        "Raw_common": (raw[:, common], "pca"),
        **{
            f"{name}_completed": (values[:, np.isfinite(values).all(axis=0)], "pca")
            for name, values in completed.items()
        },
        **{f"Assay_limited_{name}": (values, "fixed") for name, values in limited.items()},
    }
    for model in ("Txn_Jatin", "BulkFormer_147M"):
        scores, _ = signature_features(completed[model], genes)
        geometry.update({
            f"Harmonized_{model}_{name}": (values, "fixed")
            for name, values in scores.items()
        })
    nuisance = nuisance_table(
        geometry, metadata.cohort.to_numpy(str),
        metadata.cancer_type.to_numpy(str), metadata.label.to_numpy(int),
    )
    nuisance.to_csv(args.output / "representation_nuisance_diagnostics.csv", index=False)
    plot_variance(nuisance, args.output)
    coordinates = plot_geometry(
        completed["Txn_Jatin"], metadata.cohort.to_numpy(str),
        metadata.label.to_numpy(int), args.output,
    )
    coordinates["sample_id"] = coordinates.sample_index.map(dict(enumerate(ids)))
    coordinates.to_csv(args.output / "espresso_geometry_coordinates.csv", index=False)
    provenance.to_csv(args.output / "representation_provenance.csv", index=False)
    pd.read_csv(args.historical_predictions).to_csv(
        args.output / "phase0_frozen_decoder_null_predictions.csv", index=False
    )
    (args.output / "finalization_protocol.json").write_text(json.dumps({
        "seed": SEED, "bootstrap_replicates": args.bootstraps,
        "bootstrap_strata": ["held_out_cohort", "response_label"],
        "parallel_jobs": args.jobs,
        "overlap_clean_macro": "Gide, Riaz, and Rose; Hugo excluded because only one clean patient remains and AUROC is undefined.",
        "locked_external_outcomes_read": False,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
