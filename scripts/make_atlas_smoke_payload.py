#!/usr/bin/env python3
"""Create a complete API request from one known atlas reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=int, default=0)
    args = parser.parse_args()
    atlas = np.load(args.atlas, allow_pickle=False)
    genes = atlas["genes"].astype(str).tolist()
    reference = int(args.reference)
    values = atlas["expression"][reference].astype(float)
    payload = {
        "genes": genes,
        "samples": [str(atlas["reference_ids"][reference])],
        "matrix": [[round(float(value), 6)] for value in values],
        "missing": [[False] for _ in genes],
        "input_scale": "log1p",
    }
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {args.output}: {len(genes):,} genes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
