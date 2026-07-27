#!/usr/bin/env python3
"""Run cancer-gene RF analysis on one model's completed TCGA imputations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from masked_benchmark_common import load_mask, read_protocol, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=12)
    return parser.parse_args()


def load_groups(protocol: dict, sample_ids: np.ndarray) -> np.ndarray:
    sheet = pd.read_csv(protocol["datasets"]["TCGA"]["sample_sheet"], sep="\t", dtype=str)
    mapping = dict(zip(sheet["File ID"].astype(str), sheet["Case ID"].astype(str)))
    return np.asarray([mapping.get(sample_id, sample_id) for sample_id in sample_ids])


def evaluate_panel(
    model: str,
    panel: str,
    projects: list[str],
    seed: int,
    expression: np.ndarray,
    genes: list[str],
    metadata: pd.DataFrame,
    groups: np.ndarray,
    trees: int,
    jobs: int,
    output: Path,
) -> dict:
    selected = metadata["project_id"].isin(projects).to_numpy()
    x = expression[selected]
    y = metadata.loc[selected, "tissue_type"].astype(str).str.lower().eq("tumor").astype(int).to_numpy()
    panel_groups = groups[selected]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    fold_rows = []
    for fold, (train, test) in enumerate(splitter.split(x, y, panel_groups), start=1):
        classifier = RandomForestClassifier(
            n_estimators=trees,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=jobs,
            random_state=seed + fold,
        )
        classifier.fit(x[train], y[train])
        probability = classifier.predict_proba(x[test])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        oof[test] = probability
        fold_rows.append(
            {
                "model": model,
                "panel": panel,
                "mask_seed": seed,
                "fold": fold,
                "auroc": roc_auc_score(y[test], probability),
                "f1": f1_score(y[test], prediction),
                "balanced_accuracy": balanced_accuracy_score(y[test], prediction),
            }
        )
    final_classifier = RandomForestClassifier(
        n_estimators=trees,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=jobs,
        random_state=seed,
    )
    final_classifier.fit(x, y)
    importance = pd.DataFrame(
        {
            "model": model,
            "panel": panel,
            "mask_seed": seed,
            "gene_symbol": genes,
            "importance": final_classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(output / f"{panel}_seed_{seed}_gene_importance.csv.gz", index=False)
    pd.DataFrame(fold_rows).to_csv(output / f"{panel}_seed_{seed}_folds.csv", index=False)
    folds = pd.DataFrame(fold_rows)
    return {
        "model": model,
        "panel": panel,
        "mask_seed": seed,
        "supported": True,
        "samples": int(len(y)),
        "patients": int(len(np.unique(panel_groups))),
        "oof_auroc": float(roc_auc_score(y, oof)),
        "fold_auroc_mean": float(folds["auroc"].mean()),
        "fold_auroc_std": float(folds["auroc"].std(ddof=1)),
        "fold_f1_mean": float(folds["f1"].mean()),
        "fold_balanced_accuracy_mean": float(folds["balanced_accuracy"].mean()),
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    protocol = read_protocol(run_dir)
    output = run_dir / "results" / "masked_rf" / args.model
    output.mkdir(parents=True, exist_ok=True)
    genes = pd.read_csv(run_dir / "config" / "shared_gene_universe.csv")[
        "gene_symbol"
    ].astype(str).str.upper().tolist()
    gene_index = {gene: index for index, gene in enumerate(genes)}
    tcga = pd.read_parquet(protocol["datasets"]["TCGA"]["matrix"], columns=genes)
    original = np.log1p(tcga.to_numpy(dtype=np.float32, copy=False))
    metadata = pd.read_parquet(protocol["datasets"]["TCGA"]["metadata"]).copy()
    metadata.index = metadata["file_id"].astype(str)
    metadata = metadata.reindex(tcga.index.astype(str))
    groups = load_groups(protocol, tcga.index.astype(str).to_numpy())
    summaries = []
    for seed in protocol["masking"]["seeds"]:
        mask = load_mask(run_dir, seed)
        columns = np.asarray([gene_index[gene] for gene in mask], dtype=np.int64)
        prediction = np.load(
            run_dir
            / "predictions"
            / args.model
            / "TCGA"
            / f"seed_{seed}_predictions.npy",
            mmap_mode="r",
        )
        completed = original.copy()
        completed[:, columns] = prediction
        for panel, projects in protocol["cancer_panels"].items():
            summary_path = output / f"{panel}_seed_{seed}_summary.json"
            if summary_path.exists():
                summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
                continue
            summary = evaluate_panel(
                args.model,
                panel,
                projects,
                seed,
                completed,
                genes,
                metadata,
                groups,
                args.trees,
                args.jobs,
                output,
            )
            write_json(summary_path, summary)
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
        del completed
    frame = pd.DataFrame(summaries)
    frame.to_csv(output / "masked_rf_summary.csv", index=False)

    importance_files = sorted(output.glob("*_gene_importance.csv.gz"))
    if importance_files:
        importance = pd.concat(
            [pd.read_csv(path) for path in importance_files], ignore_index=True
        )
        aggregate = (
            importance.groupby(["model", "panel", "gene_symbol"], as_index=False)["importance"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values(["panel", "mean"], ascending=[True, False])
        )
        aggregate.to_csv(output / "gene_importance_across_masks.csv", index=False)
    (run_dir / "status" / f"{args.model}.RF_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
