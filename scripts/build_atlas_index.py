#!/usr/bin/env python3
"""Build the inspectable atlas NPZ consumed by the Studio backend."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-scale", choices=("raw", "log1p"), default="raw")
    args = parser.parse_args()

    frame = pd.read_parquet(args.expression) if args.expression.suffix == ".parquet" else pd.read_csv(args.expression, index_col=0)
    metadata = pd.read_csv(args.metadata)
    required = {"reference_id", "species", "tissue"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"metadata requires columns: {sorted(required)}")
    common = [value for value in metadata["reference_id"].astype(str) if value in frame.columns]
    if not common:
        raise ValueError("no metadata reference_id values match expression columns")
    matrix = frame.loc[:, common].T.to_numpy(dtype=np.float32)
    if args.input_scale == "raw":
        matrix = np.log1p(np.maximum(matrix, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        genes=frame.index.astype(str).str.upper().to_numpy(dtype=str),
        reference_ids=np.asarray(common, dtype=str),
        expression=matrix,
    )
    print(f"wrote {args.output}: {matrix.shape[0]:,} references x {matrix.shape[1]:,} genes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
