#!/usr/bin/env python3
"""ARCHS4 accession manifest and exact test-set overlap certification."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ACCESSION = re.compile(r"\b(?:GSM|GSE|SRR|SRX|SRS|ERP|DRR|ERR)\d+\b", re.I)


def normalize(value: object) -> set[str]:
    return {match.upper() for match in ACCESSION.findall(str(value))}


def build_manifest(metadata_csv: str | Path, output_csv: str | Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_csv, dtype=str)
    accessions = set()
    for column in metadata.columns:
        for value in metadata[column].dropna():
            accessions.update(normalize(value))
    manifest = pd.DataFrame({"accession": sorted(accessions)})
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False)
    return manifest


def check_overlap(
    manifest_csv: str | Path,
    test_ids: list[str],
    output_json: str | Path | None = None,
) -> dict:
    manifest = set(pd.read_csv(manifest_csv)["accession"].astype(str).str.upper())
    normalized = set()
    for value in test_ids:
        normalized.update(normalize(value))
    overlap = sorted(manifest & normalized)
    result = {
        "status": "clean" if not overlap else "overlap_detected",
        "training_accessions": len(manifest),
        "test_accessions": len(normalized),
        "overlap_count": len(overlap),
        "overlap_accessions": overlap,
        "manifest_sha256": hashlib.sha256(Path(manifest_csv).read_bytes()).hexdigest(),
        "method": "exact normalized GEO/SRA accession intersection",
    }
    if output_json:
        Path(output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
