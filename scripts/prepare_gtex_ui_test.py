#!/usr/bin/env python3
"""Prepare a diverse, reproducible GTEx matrix for Studio pipeline testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


TARGET_TISSUES = [
    "Brain - Cerebellar Hemisphere",
    "Heart - Left Ventricle",
    "Liver",
    "Lung",
    "Muscle - Skeletal",
    "Skin - Sun Exposed (Lower leg)",
    "Whole Blood",
    "Adipose - Visceral (Omentum)",
    "Colon - Transverse",
    "Kidney - Cortex",
    "Pancreas",
    "Thyroid",
]


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--canonical-genes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gene_table = pd.read_csv(args.canonical_genes)
    gene_column = "gene_symbol" if "gene_symbol" in gene_table else gene_table.columns[-1]
    canonical = gene_table[gene_column].astype(str).str.upper().tolist()
    with h5py.File(args.source, "r") as handle:
        source_genes = decode(handle["meta/genes"][:])
        tissues = decode(handle["meta/smtsd"][:])
        sample_ids = decode(handle["meta/sampid"][:])
        runs = decode(handle["meta/run"][:])
        rin = np.asarray(handle["meta/smrin"][:], dtype=float)
        freeze = decode(handle["meta/smafrze"][:])

        selected = []
        for tissue in TARGET_TISSUES:
            candidates = np.flatnonzero(tissues == tissue)
            if not len(candidates):
                continue
            preferred = candidates[freeze[candidates] == "USE ME"]
            pool = preferred if len(preferred) else candidates
            quality = np.nan_to_num(rin[pool], nan=-1.0)
            selected.append(int(pool[np.argmax(quality)]))

        gene_index = {gene.upper(): index for index, gene in enumerate(source_genes)}
        missing = [gene for gene in canonical if gene not in gene_index]
        available = [gene for gene in canonical if gene in gene_index]
        columns = np.asarray([gene_index[gene] for gene in available], dtype=int)
        counts = np.stack(
            [np.asarray(handle["data/expression"][index, :])[columns] for index in selected]
        ).astype(np.float64, copy=False)

    totals = counts.sum(axis=1, keepdims=True)
    log1p_cpm = np.log1p(counts * (1_000_000.0 / np.maximum(totals, 1.0))).astype(
        np.float32
    )
    labels = [
        f"GTEx_{tissues[index].replace(' ', '_').replace('/', '-')}_{sample_ids[index]}"
        for index in selected
    ]
    truth = pd.DataFrame(log1p_cpm.T, index=available, columns=labels)
    rng = np.random.default_rng(args.seed)
    masked_indices = np.sort(
        rng.choice(
            len(available),
            size=round(len(available) * args.mask_ratio),
            replace=False,
        )
    )
    masked = truth.copy()
    masked.iloc[masked_indices, :] = np.nan

    truth_path = args.output_dir / "gtex_diverse_12samples_truth_log1p_cpm.csv"
    masked_path = args.output_dir / "gtex_diverse_12samples_15pct_whole_gene_mask.csv"
    metadata_path = args.output_dir / "gtex_diverse_12samples_metadata.csv"
    truth.rename_axis("gene").to_csv(truth_path, float_format="%.6f")
    masked.rename_axis("gene").to_csv(masked_path, float_format="%.6f", na_rep="NA")
    pd.DataFrame(
        {
            "sample": labels,
            "gtex_sample_id": sample_ids[selected],
            "run": runs[selected],
            "tissue": tissues[selected],
            "rin": rin[selected],
            "source_index": selected,
        }
    ).to_csv(metadata_path, index=False)
    pd.DataFrame({"gene": np.asarray(available)[masked_indices]}).to_csv(
        args.output_dir / "masked_genes.csv", index=False
    )
    pd.DataFrame({"gene": missing}).to_csv(
        args.output_dir / "model_genes_absent_from_gtex_source.csv", index=False
    )
    manifest = {
        "source": str(args.source.resolve()),
        "source_shape": [9662, 25150],
        "samples": len(selected),
        "model_vocabulary_genes": len(canonical),
        "genes": len(available),
        "model_genes_absent_from_source": len(missing),
        "masked_genes": len(masked_indices),
        "mask_ratio": args.mask_ratio,
        "seed": args.seed,
        "normalization": "log1p(CPM), library size over the 16,055 model genes",
        "masking": "same seeded gene set hidden across every sample",
        "ui_input": masked_path.name,
        "truth": truth_path.name,
        "metadata": metadata_path.name,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
