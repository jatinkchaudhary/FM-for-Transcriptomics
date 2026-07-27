#!/usr/bin/env python3
"""Run the lightweight raw-expression cancer RF baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold

from masked_benchmark_common import read_protocol, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=12)
    return parser.parse_args()


def patient_groups(protocol: dict, sample_ids: np.ndarray) -> np.ndarray:
    sheet = pd.read_csv(protocol["datasets"]["TCGA"]["sample_sheet"], sep="\t", dtype=str)
    mapping = dict(zip(sheet["File ID"].astype(str), sheet["Case ID"].astype(str)))
    return np.asarray([mapping.get(sample_id, sample_id) for sample_id in sample_ids])


def run_panel(
    name: str,
    projects: list[str],
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
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260723)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    fold_rows = []
    for fold, (train, test) in enumerate(splitter.split(x, y, panel_groups), start=1):
        classifier = RandomForestClassifier(
            n_estimators=trees,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=jobs,
            random_state=20260723 + fold,
        )
        classifier.fit(x[train], y[train])
        probability = classifier.predict_proba(x[test])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        oof[test] = probability
        fold_rows.append(
            {
                "panel": name,
                "fold": fold,
                "samples": len(test),
                "patients": len(np.unique(panel_groups[test])),
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
        random_state=20260723,
    )
    final_classifier.fit(x, y)
    importance = pd.DataFrame(
        {"gene_symbol": genes, "importance": final_classifier.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(output / f"{name}_raw_gene_importance.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / f"{name}_raw_folds.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": metadata.index.to_numpy()[selected],
            "truth_tumor": y,
            "oof_probability": oof,
        }
    ).to_csv(output / f"{name}_raw_oof_predictions.csv", index=False)

    false_positive, true_positive, _ = roc_curve(y, oof)
    auroc = roc_auc_score(y, oof)
    plt.figure(figsize=(6.4, 5.2))
    plt.plot(false_positive, true_positive, color="#1f6f5b", linewidth=2, label=f"Raw expression (AUC {auroc:.4f})")
    plt.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{name.replace('_', ' ')}: tumor versus normal")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output / f"{name}_raw_roc.png", dpi=180)
    plt.close()

    folds = pd.DataFrame(fold_rows)
    return {
        "panel": name,
        "projects": projects,
        "samples": int(len(y)),
        "patients": int(len(np.unique(panel_groups))),
        "tumor_samples": int(y.sum()),
        "normal_samples": int((1 - y).sum()),
        "oof_auroc": float(auroc),
        "fold_auroc_mean": float(folds["auroc"].mean()),
        "fold_auroc_std": float(folds["auroc"].std(ddof=1)),
        "fold_f1_mean": float(folds["f1"].mean()),
        "fold_balanced_accuracy_mean": float(folds["balanced_accuracy"].mean()),
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output = run_dir / "results" / "raw_rf_baseline"
    output.mkdir(parents=True, exist_ok=True)
    protocol = read_protocol(run_dir)

    universe = pd.read_csv(run_dir / "config" / "shared_gene_universe.csv")[
        "gene_symbol"
    ].astype(str).str.upper().tolist()
    tcga = pd.read_parquet(protocol["datasets"]["TCGA"]["matrix"], columns=universe)
    expression = np.log1p(tcga.to_numpy(dtype=np.float32, copy=False))
    metadata = pd.read_parquet(protocol["datasets"]["TCGA"]["metadata"]).copy()
    metadata.index = metadata["file_id"].astype(str)
    metadata = metadata.reindex(tcga.index.astype(str))
    if metadata[["project_id", "tissue_type"]].isna().any().any():
        raise RuntimeError("TCGA metadata did not align with expression sample IDs")
    groups = patient_groups(protocol, tcga.index.astype(str).to_numpy())

    summaries = []
    for name, projects in protocol["cancer_panels"].items():
        summaries.append(
            run_panel(
                name,
                projects,
                expression,
                universe,
                metadata,
                groups,
                args.trees,
                args.jobs,
                output,
            )
        )
    pd.DataFrame(summaries).to_csv(output / "raw_rf_summary.csv", index=False)
    write_json(output / "raw_rf_summary.json", {"panels": summaries})
    (run_dir / "status" / "RAW_RF_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "panels": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
