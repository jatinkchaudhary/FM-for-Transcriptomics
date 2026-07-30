#!/usr/bin/env python3
"""Exercise Txn_Jatin imputation -> atlas -> Ollama on one full profile."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np


def post(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atlas = np.load(args.atlas, allow_pickle=False)
    genes = atlas["genes"].astype(str).tolist()
    values = atlas["expression"][0].astype(float)
    rng = np.random.default_rng(42)
    masked = np.zeros(len(genes), dtype=bool)
    masked[rng.choice(len(genes), size=round(0.15 * len(genes)), replace=False)] = True
    matrix = [[None if masked[index] else round(float(value), 6)] for index, value in enumerate(values)]
    missing = [[bool(value)] for value in masked]
    started = time.time()
    imputed = post(
        f"{args.api}/api/impute",
        {
            "model": "Txn_Jatin",
            "genes": genes,
            "samples": ["GTEx_adipose_masked_15pct"],
            "matrix": matrix,
            "missing": missing,
            "input_scale": "log1p",
        },
        600,
    )
    imputation_seconds = time.time() - started
    completed = [
        [
            float(imputed["imputed"][index][0])
            if masked[index]
            else float(values[index])
        ]
        for index in range(len(genes))
    ]
    started = time.time()
    analysis = post(
        f"{args.api}/api/atlas",
        {
            "genes": genes,
            "samples": ["GTEx_adipose_masked_15pct"],
            "matrix": completed,
            "missing": [[False] for _ in genes],
            "input_scale": "log1p",
        },
        900,
    )
    atlas_seconds = time.time() - started
    args.output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    summary = {
        "masked_genes": int(masked.sum()),
        "imputation_seconds": round(imputation_seconds, 3),
        "atlas_language_seconds": round(atlas_seconds, 3),
        "imputation": {
            "matched_genes": imputed["matched_genes"],
            "model_gene_count": imputed["model_gene_count"],
            "unresolved_genes": len(imputed["unresolved_genes"]),
            "warnings": imputed["warnings"],
        },
        "best_match": analysis["sample_results"][0]["matches"][0],
        "species_evidence": analysis["sample_results"][0]["species_evidence"],
        "language_head": {
            "status": analysis["language_head"]["status"],
            "model": analysis["language_head"]["model"],
            "characters": len(analysis["language_head"].get("text", "")),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
