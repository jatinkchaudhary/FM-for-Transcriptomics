#!/usr/bin/env python3
"""Generate deterministic TCGA-derived upload matrices for UI/API testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_gene_list(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    column = next(
        (name for name in frame.columns if name.lower() in {"gene", "symbol", "gene_symbol"}),
        frame.columns[0],
    )
    return frame[column].dropna().astype(str).str.upper().drop_duplicates().tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcga", type=Path, required=True)
    parser.add_argument("--genes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--files", type=int, default=50)
    parser.add_argument("--genes-per-file", type=int, default=50)
    parser.add_argument("--samples-per-file", type=int, default=8)
    parser.add_argument("--mask-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.tcga)
    shared = sorted(set(read_gene_list(args.genes)).intersection(frame.columns))
    if len(shared) < args.genes_per_file:
        raise RuntimeError(f"Only {len(shared)} shared genes are available")
    manifests = []
    for file_index in range(args.files):
        genes = rng.choice(shared, size=args.genes_per_file, replace=False).tolist()
        sample_rows = rng.choice(frame.index.to_numpy(), size=args.samples_per_file, replace=False)
        matrix = frame.loc[sample_rows, genes].T.astype(float)
        matrix.columns = [f"TCGA_UI_{file_index + 1:02d}_S{i + 1:02d}" for i in range(len(sample_rows))]
        truth = matrix.copy()
        count = matrix.shape[0] * matrix.shape[1]
        flat_mask = np.zeros(count, dtype=bool)
        flat_mask[rng.choice(count, size=max(1, round(count * args.mask_fraction)), replace=False)] = True
        mask = flat_mask.reshape(matrix.shape)
        matrix = matrix.mask(mask)
        name = f"tcga_random_panel_{file_index + 1:02d}_50genes.csv"
        truth_name = f"tcga_random_panel_{file_index + 1:02d}_truth.csv"
        matrix.rename_axis("gene").to_csv(args.output / name, na_rep="NA", float_format="%.6f")
        truth.rename_axis("gene").to_csv(args.output / truth_name, float_format="%.6f")
        manifests.append(
            {
                "file": name,
                "truth_file": truth_name,
                "genes": len(genes),
                "samples": len(sample_rows),
                "masked_cells": int(mask.sum()),
                "mask_fraction": float(mask.mean()),
                "input_scale": "raw TPM",
                "source_sample_ids": [str(value) for value in sample_rows],
            }
        )
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "files": manifests,
                "scientific_scope": (
                    "Parser/API smoke tests. Fifty-gene panels expose less than 1% of each "
                    "decoder vocabulary and are outside the 15% masking training regime."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(manifests)} matrices and truth files to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
