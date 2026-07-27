#!/usr/bin/env python3
"""Freeze the whole-gene masking protocol and create its lightweight artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from masked_benchmark_common import read_genes, write_json


SUPPORTED = [
    "Txn_Jatin",
    "BRIDGE",
    "BulkFormer_37M",
    "BulkFormer_50M",
    "BulkFormer_93M",
    "BulkFormer_127M",
    "BulkFormer_147M",
]
UNSUPPORTED = [
    "ESM2_PCA512_prior",
    "Geneformer",
    "scGPT",
    "Txn_Jatin_contextual",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--tcga", required=True)
    parser.add_argument("--tcga-metadata", required=True)
    parser.add_argument("--tcga-sample-sheet", required=True)
    parser.add_argument("--osdr", required=True)
    parser.add_argument("--txn-genes", required=True)
    parser.add_argument("--bridge-genes", required=True)
    parser.add_argument("--bulk-genes", required=True)
    parser.add_argument("--txn-checkpoint", required=True)
    parser.add_argument("--bridge-checkpoint", required=True)
    parser.add_argument("--bulk-checkpoint-dir", required=True)
    parser.add_argument("--bulkformer-root", required=True)
    parser.add_argument("--train-flash", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--graph-weights", required=True)
    parser.add_argument("--esm2", required=True)
    parser.add_argument("--existing-static-results", required=True)
    parser.add_argument("--existing-sample-results", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260723, 20260724, 20260725])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    for name in ("config", "masks", "results", "predictions", "logs", "status", "resources"):
        (output / name).mkdir(parents=True, exist_ok=True)

    tcga = pd.read_parquet(args.tcga)
    tcga_genes = [str(value).upper() for value in tcga.columns]
    tcga_samples = tcga.index.astype(str).tolist()
    metadata = pd.read_parquet(args.tcga_metadata)
    osdr = np.load(args.osdr, allow_pickle=True)
    osdr_genes = [str(value).upper() for value in osdr["genes"]]

    genes = {
        "TCGA": tcga_genes,
        "OSDR": osdr_genes,
        "Txn_Jatin": read_genes(args.txn_genes),
        "BRIDGE": read_genes(args.bridge_genes),
        "BulkFormer": read_genes(args.bulk_genes),
    }
    shared = set(genes["TCGA"])
    for values in genes.values():
        shared.intersection_update(values)
    universe = [gene for gene in tcga_genes if gene in shared]
    if not universe:
        raise RuntimeError("The shared gene universe is empty")

    pd.DataFrame({"gene_symbol": universe}).to_csv(
        output / "config" / "shared_gene_universe.csv", index=False
    )
    coverage_rows = []
    for name, values in genes.items():
        current = set(values)
        coverage_rows.append(
            {
                "source": name,
                "total_genes": len(values),
                "unique_genes": len(current),
                "shared_genes": len(current.intersection(universe)),
                "missing_from_shared": len(set(universe).difference(current)),
            }
        )
    pd.DataFrame(coverage_rows).to_csv(output / "config" / "gene_coverage.csv", index=False)

    mask_size = int(round(0.15 * len(universe)))
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        selected = sorted(rng.choice(universe, size=mask_size, replace=False).tolist())
        pd.DataFrame({"gene_symbol": selected}).to_csv(
            output / "masks" / f"mask_seed_{seed}.csv", index=False
        )

    capability_rows = []
    for model in SUPPORTED + UNSUPPORTED:
        supported = model in SUPPORTED
        capability_rows.append(
            {
                "model": model,
                "imputation_supported": supported,
                "imputation_output": "native expression decoder" if supported else "NaN",
                "reason": (
                    "Validated checkpoint exposes a per-gene expression prediction head."
                    if supported
                    else "No validated bulk-expression decoder is available for this embedding model."
                ),
            }
        )
    pd.DataFrame(capability_rows).to_csv(output / "config" / "model_capabilities.csv", index=False)

    unsupported_rows = []
    for model in UNSUPPORTED:
        for dataset in ("TCGA", "OSDR"):
            for seed in args.seeds:
                unsupported_rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "mask_seed": seed,
                        "supported": False,
                        "status": "not_applicable",
                        "pcc_global": np.nan,
                        "spearman_global": np.nan,
                        "mse": np.nan,
                        "mae": np.nan,
                        "auroc_macro": np.nan,
                        "auprc_macro": np.nan,
                        "reason": "No validated bulk-expression decoder.",
                    }
                )
    pd.DataFrame(unsupported_rows).to_csv(
        output / "results" / "unsupported_imputation_rows.csv", index=False
    )

    checkpoint_paths = {
        "Txn_Jatin": args.txn_checkpoint,
        "BRIDGE": args.bridge_checkpoint,
    }
    for size in ("37M", "50M", "93M", "127M", "147M"):
        checkpoint_paths[f"BulkFormer_{size}"] = str(
            Path(args.bulk_checkpoint_dir) / f"BulkFormer_{size}.pt"
        )

    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "whole_gene_masking_and_cancer_gene_identification",
        "masking": {
            "type": "whole_gene",
            "fraction": 0.15,
            "same_genes_across_samples": True,
            "seeds": args.seeds,
            "genes_per_seed": mask_size,
            "mask_token": -10.0,
        },
        "shared_gene_count": len(universe),
        "datasets": {
            "TCGA": {
                "matrix": args.tcga,
                "metadata": args.tcga_metadata,
                "sample_sheet": args.tcga_sample_sheet,
                "samples": len(tcga_samples),
                "source_gene_count": len(tcga_genes),
                "input_scale": "log1p(TPM)",
                "binary_expression_threshold": "TPM >= 1",
            },
            "OSDR": {
                "matrix": args.osdr,
                "samples": int(osdr["X"].shape[0]),
                "source_gene_count": len(osdr_genes),
                "input_scale": "prepared log1p(CPM)",
                "binary_expression_threshold": "CPM >= 1",
            },
        },
        "models": {
            name: {
                "supported": name in SUPPORTED,
                "checkpoint": checkpoint_paths.get(name),
            }
            for name in SUPPORTED + UNSUPPORTED
        },
        "paths": {
            "txn_genes": args.txn_genes,
            "bridge_genes": args.bridge_genes,
            "bulk_genes": args.bulk_genes,
            "bulkformer_root": args.bulkformer_root,
            "train_flash": args.train_flash,
            "graph": args.graph,
            "graph_weights": args.graph_weights,
            "esm2": args.esm2,
            "existing_static_results": args.existing_static_results,
            "existing_sample_results": args.existing_sample_results,
        },
        "imputation_metrics": [
            "PCC",
            "Spearman",
            "MSE",
            "MAE",
            "expressed-vs-unexpressed AUROC",
            "expressed-vs-unexpressed AUPRC",
        ],
        "cancer_panels": {
            "2_cancer": ["TCGA-LUAD", "TCGA-LUSC"],
            "5_cancer": ["TCGA-BRCA", "TCGA-KIRC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-SKCM"],
        },
        "cancer_target": "binary tumor versus normal",
        "cancer_probe": {
            "type": "RandomForestClassifier",
            "trees": 300,
            "cross_validation": "5-fold stratified patient-group CV",
            "class_weight": "balanced",
        },
        "reuse_without_recalculation": {
            "github_protocol_static_results": args.existing_static_results,
            "previous_sample_results": args.existing_sample_results,
        },
    }
    write_json(output / "config" / "protocol.json", protocol)

    project_counts = (
        metadata.groupby(["project_id", "tissue_type"], dropna=False)
        .size()
        .reset_index(name="samples")
    )
    project_counts.to_csv(output / "results" / "dataset_counts.csv", index=False)

    report = f"""# Early benchmark artifacts

Generated: {protocol['created_utc']}

## Frozen scope

- Shared TCGA/OSDR/Txn_Jatin/BRIDGE/BulkFormer universe: **{len(universe):,} genes**
- Whole-gene masks: **{mask_size:,} genes (15%)** for each seed: {", ".join(map(str, args.seeds))}
- The same masked genes are hidden in every sample, dataset, and supported model.
- TCGA scale: `log1p(TPM)`; prepared OSDR scale: `log1p(CPM)`.
- Native imputation is supported by 7 models; 4 embedding-only models are explicitly reported as `NaN`.
- Existing GitHub-protocol static gene and gene-pair results are retained without recalculation.

## Cancer-gene panels

- 2 cancer: LUAD + LUSC, binary tumor versus normal.
- 5 cancer: BRCA + KIRC + LUAD + LUSC + SKCM, binary tumor versus normal.
- Probe: 300-tree random forest with patient-grouped five-fold validation.

## Execution order

1. Dataset, overlap, capability, mask, and raw-expression RF baseline artifacts.
2. BulkFormer-37M first, followed by Txn_Jatin and BRIDGE.
3. Remaining BulkFormer checkpoints from 50M through 147M.
4. CPU metrics and RF analysis run concurrently with sequential H100 inference.
"""
    (output / "EARLY_RESULTS.md").write_text(report, encoding="utf-8")
    (output / "status" / "PREPARED").write_text(protocol["created_utc"], encoding="utf-8")
    print(json.dumps({"output": str(output), "shared_genes": len(universe), "mask_size": mask_size}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
