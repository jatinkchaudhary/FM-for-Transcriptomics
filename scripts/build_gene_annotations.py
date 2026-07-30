#!/usr/bin/env python3
"""Rebuild MyGene annotations without rebuilding the expression atlas."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_production_atlas import mygene_annotations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.genes)
    column = next(
        (name for name in frame.columns if name.lower() in {"gene", "symbol", "gene_symbol"}),
        frame.columns[0],
    )
    genes = frame[column].astype(str).str.upper().tolist()
    rows = mygene_annotations(genes, args.output)
    print(f"received {len(rows):,} annotation responses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
