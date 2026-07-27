#!/usr/bin/env python3
"""Validate packaged ICI feasibility outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results"
    checks = {}

    clinical = pd.read_csv(root / "prepared" / "clinical_harmonized.tsv", sep="\t")
    checks["clinical_239_unique"] = (
        len(clinical) == 239 and clinical["Index"].nunique() == 239
    )
    checks["clinical_labels_complete"] = set(clinical["label"]) == {0, 1}

    predictions = pd.read_csv(results / "loco_predictions.csv")
    primary = predictions.loc[predictions["analysis"].eq("four_cohort_loco")]
    primary_counts = primary.groupby("method")["sample_id"].agg(["size", "nunique"])
    checks["primary_predictions_complete"] = bool(
        (primary_counts["size"] == 239).all()
        and (primary_counts["nunique"] == 239).all()
    )
    checks["primary_probabilities_valid"] = bool(
        np.isfinite(primary["probability"]).all()
        and primary["probability"].between(0, 1).all()
    )

    clean = pd.read_csv(results / "clean_no_pretraining_overlap_predictions.csv")
    clean_counts = clean.groupby("method")["sample_id"].agg(["size", "nunique"])
    checks["clean_predictions_complete"] = bool(
        (clean_counts["size"] == 213).all()
        and (clean_counts["nunique"] == 213).all()
    )
    checks["clean_excludes_hugo"] = "Hugo" not in set(clean["held_out_cohort"])

    overlap = pd.read_csv(results / "pretraining_overlap_summary.csv")
    overlap_map = dict(zip(overlap["cohort"], overlap["patients_present"]))
    checks["pretraining_overlap_exact"] = overlap_map == {
        "Gide": 0,
        "Hugo": 25,
        "Riaz": 0,
        "Rose": 0,
    }

    workbook = pd.ExcelFile(results / "ici_initial_results.xlsx")
    required_sheets = {
        "cohorts",
        "LOCO_summary",
        "LOCO_metrics",
        "LOCO_predictions",
        "delta_vs_raw",
        "within_metrics",
        "nested_tuning",
        "nuisance_geometry",
        "Hallmark_meta",
        "Hallmark_by_cohort",
        "pretraining_overlap",
        "clean_LOCO_summary",
        "clean_delta_vs_raw",
    }
    checks["workbook_sheets_complete"] = required_sheets.issubset(
        set(workbook.sheet_names)
    )

    figure_paths = sorted((root / "figures").glob("*.png"))
    figure_valid = len(figure_paths) >= 9
    figure_dimensions = {}
    for path in figure_paths:
        with Image.open(path) as image:
            figure_dimensions[path.name] = list(image.size)
            figure_valid = figure_valid and image.width >= 1000 and image.height >= 700
    checks["figures_valid"] = bool(figure_valid)

    for filename in (
        "extraction_metadata.json",
        "evaluation_metadata.json",
        "biology_diagnostic_summary.json",
        "pretraining_overlap_summary.json",
    ):
        json.loads((results / filename).read_text(encoding="utf-8"))
    json.loads((root / "prepared" / "preparation_qc.json").read_text(encoding="utf-8"))
    json.loads((root / "config" / "protocol.json").read_text(encoding="utf-8"))
    checks["json_valid"] = True

    checks["documents_present"] = all(
        (root / filename).exists()
        for filename in (
            "README.md",
            "STATUS.md",
            "INITIAL_IMMUNOTHERAPY_REPORT.md",
        )
    )
    checks["provisional_outputs_flagged"] = all(
        "REJECTED" in path.name
        for path in list(results.glob("*provisional*"))
        + list((root / "logs").glob("*provisional*"))
    )

    output = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "primary_prediction_rows": int(len(primary)),
        "clean_prediction_rows": int(len(clean)),
        "workbook_sheets": workbook.sheet_names,
        "figure_dimensions": figure_dimensions,
    }
    (results / "validation_report.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
