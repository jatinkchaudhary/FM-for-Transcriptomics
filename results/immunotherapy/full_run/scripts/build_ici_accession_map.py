#!/usr/bin/env python3
"""Map evaluated ICI patients to GEO/SRA accessions for pretraining audits."""

from __future__ import annotations

import argparse
import csv
import gzip
import re
from pathlib import Path

import pandas as pd


def series_samples(path: Path) -> list[tuple[str, str]]:
    titles = None
    accessions = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_title"):
                titles = next(csv.reader([line], delimiter="\t"))[1:]
            elif line.startswith("!Sample_geo_accession"):
                accessions = next(csv.reader([line], delimiter="\t"))[1:]
    if titles is None or accessions is None or len(titles) != len(accessions):
        raise ValueError(f"Cannot parse sample metadata from {path}")
    return list(zip(titles, accessions))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "raw"
    output = []

    riaz = pd.read_csv(raw / "Riaz_clinical.tsv", sep="\t")
    riaz_lookup = {
        re.sub(r"-\d+$", "", title): accession
        for title, accession in series_samples(
            raw / "GSE91061_series_matrix.txt.gz"
        )
    }
    for _, row in riaz.iterrows():
        output.append(
            {
                "cohort": "Riaz",
                "evaluation_sample_id": row["Index"],
                "archive_accession": riaz_lookup[row["Index"]],
            }
        )

    hugo = pd.read_csv(raw / "Hugo_clinical.tsv", sep="\t")
    hugo_series = series_samples(raw / "GSE78220_series_matrix.txt.gz")
    for _, row in hugo.iterrows():
        patient = str(row["Sample_id"]).upper()
        matches = [
            accession
            for title, accession in hugo_series
            if re.match(r"^PT\d+", title.upper()).group(0) == patient
        ]
        if not matches:
            raise ValueError(f"No Hugo accession for {patient}")
        for accession in matches:
            output.append(
                {
                    "cohort": "Hugo",
                    "evaluation_sample_id": row["Index"],
                    "archive_accession": accession,
                }
            )

    rose = pd.read_csv(raw / "Rose_clinical.tsv", sep="\t")
    for _, row in rose.iterrows():
        output.append(
            {
                "cohort": "Rose",
                "evaluation_sample_id": row["Index"],
                "archive_accession": row["geo"],
            }
        )

    gide = pd.read_csv(raw / "Gide_clinical.tsv", sep="\t")
    for _, row in gide.iterrows():
        output.append(
            {
                "cohort": "Gide",
                "evaluation_sample_id": row["Index"],
                "archive_accession": row["rnaseq_id"],
            }
        )

    frame = pd.DataFrame(output)
    frame.to_csv(root / "prepared" / "evaluation_accessions.tsv", sep="\t", index=False)
    print(
        frame.groupby("cohort").agg(
            evaluated_patients=("evaluation_sample_id", "nunique"),
            accessions=("archive_accession", "nunique"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
