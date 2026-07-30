#!/usr/bin/env python3
"""Build a public-cohort expression reference for edge baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


COHORTS = ("gide", "riaz", "hugo", "rose")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    blocks = []
    sample_ids = []
    genes = None
    for cohort in COHORTS:
        path = args.input_dir / f"{cohort}__Txn_Jatin.npz"
        with np.load(path, allow_pickle=False) as payload:
            current_genes = payload["genes"].astype(str)
            if genes is None:
                genes = current_genes
            elif not np.array_equal(genes, current_genes):
                raise ValueError(f"gene order differs for {cohort}")
            raw = payload["raw"].astype(np.float32)
            blocks.append(raw)
            sample_ids.extend(f"{cohort}:{value}" for value in payload["sample_ids"].astype(str))

    matrix = np.vstack(blocks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        expression=matrix,
        genes=genes,
        sample_ids=np.asarray(sample_ids),
    )
    metadata = {
        "cohorts": list(COHORTS),
        "samples": int(matrix.shape[0]),
        "genes": int(matrix.shape[1]),
        "role": "independent public immunotherapy-cohort co-expression baseline",
        "label_usage": "none",
        "missing_values": int((~np.isfinite(matrix)).sum()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
