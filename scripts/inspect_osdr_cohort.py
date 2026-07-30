#!/usr/bin/env python3
"""Audit a prepared OSDR NPZ and its companion metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    payload = np.load(args.matrix, allow_pickle=True)
    arrays = {
        key: {
            "shape": list(payload[key].shape),
            "dtype": str(payload[key].dtype),
            "examples": payload[key].reshape(-1)[:3].astype(str).tolist(),
        }
        for key in payload.files
    }
    metadata = pd.read_csv(args.metadata)
    report = {
        "matrix": str(args.matrix),
        "arrays": arrays,
        "metadata": {
            "path": str(args.metadata),
            "shape": list(metadata.shape),
            "columns": metadata.columns.tolist(),
            "examples": metadata.head(3).fillna("").to_dict("records"),
            "unique_counts": {
                column: int(metadata[column].nunique(dropna=True))
                for column in metadata.columns
                if metadata[column].nunique(dropna=True) <= 100
            },
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
