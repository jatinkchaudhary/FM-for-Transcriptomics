#!/usr/bin/env python3
"""Add ESM3 to the frozen individual-GO partial-unfreezing benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


PARTIAL_MODELS = {
    "last1": "BRIDGE_OSDR_last1_dynamic",
    "last2": "BRIDGE_OSDR_last2_dynamic",
    "last3": "BRIDGE_OSDR_last3_dynamic",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--partial-output", type=Path, required=True)
    parser.add_argument("--full-finetune-output", type=Path, required=True)
    parser.add_argument("--reference-embedding-dir", type=Path, required=True)
    parser.add_argument("--go-library", type=Path, required=True)
    parser.add_argument("--esm2-prior", type=Path, required=True)
    parser.add_argument("--esm3-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_npz(path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        keys = [str(value).upper() for value in payload["keys"].tolist()]
        matrix = payload["emb"].astype(np.float32)
        provenance = json.loads(str(payload["provenance"][0]))
        fallback = bool(payload["is_fallback"])
    if fallback:
        raise ValueError("ESM3 embedding is marked as fallback")
    if provenance.get("name") != "ESM3":
        raise ValueError(f"unexpected ESM3 provenance: {provenance}")
    return keys, matrix


def load_esm2_prior(path: Path, master: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        genes = [str(value).upper() for value in payload["genes"].tolist()]
        matrix = payload["embeddings"].astype(np.float32)
        covered = payload["covered"].astype(bool)
    if matrix.shape[0] != len(genes) or covered.shape != (len(genes),):
        raise ValueError(f"invalid ESM2 prior shapes in {path}")
    matrix[~covered] = np.nan
    index = {gene: row for row, gene in enumerate(genes)}
    aligned = np.full((len(master), matrix.shape[1]), np.nan, dtype=np.float32)
    for row, gene in enumerate(master):
        source_row = index.get(gene)
        if source_row is not None:
            aligned[row] = matrix[source_row]
    return aligned


def build_summary(real: pd.DataFrame) -> pd.DataFrame:
    summary = (
        real.groupby(["library", "variant", "model"], as_index=False)
        .agg(
            mean_AUROC=("auroc", "mean"),
            mean_AUPRC=("auprc", "mean"),
            terms=("term", "nunique"),
        )
    )
    summary["AUROC_rank"] = summary.groupby(["library", "variant"])[
        "mean_AUROC"
    ].rank(ascending=False, method="min")
    return summary.sort_values(["library", "variant", "AUROC_rank", "model"])


def main() -> int:
    args = parse_args()
    for field in vars(args):
        value = getattr(args, field)
        if isinstance(value, Path):
            setattr(args, field, value.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    probe = load_module(
        args.project_root / "Txn_Jatin" / "osdr_finetune" / "benchmark_go_kegg.py",
        "osdr_probe_common",
    )
    partial = load_module(
        args.project_root
        / "Txn_Jatin"
        / "osdr_partial_unfreeze"
        / "benchmark_partial_osdr_go_kegg.py",
        "partial_go_common",
    )
    loader_args = SimpleNamespace(
        partial_output=args.partial_output,
        full_finetune_output=args.full_finetune_output,
        reference_embedding_dir=args.reference_embedding_dir,
    )
    master, existing_embeddings = partial.load_all_embeddings(loader_args)
    existing_embeddings["ESM2_PCA512_prior"] = (
        load_esm2_prior(args.esm2_prior, master),
        False,
    )
    esm3_genes, esm3_matrix = load_npz(args.esm3_npz)
    if esm3_genes != master:
        raise RuntimeError("ESM3 gene order differs from the partial benchmark master")

    universe_table = pd.read_csv(
        args.partial_output / "benchmark" / "osdr_gene_universe.csv"
    )
    universe = (
        universe_table.loc[universe_table["eligible"].astype(bool), "gene"]
        .astype(str)
        .str.upper()
        .tolist()
    )
    master_index = {gene: index for index, gene in enumerate(master)}
    rows = [master_index[gene] for gene in universe]
    existing_subset = {
        model: matrix[rows]
        for model, (matrix, _fallback) in existing_embeddings.items()
    }
    common = set.intersection(
        *(
            {
                universe[index]
                for index in range(len(universe))
                if np.isfinite(matrix[index]).all()
            }
            for matrix in existing_subset.values()
        )
    )
    esm3_subset = esm3_matrix[rows]
    universe_index = {gene: index for index, gene in enumerate(universe)}
    esm3_finite = {
        gene
        for gene, index in universe_index.items()
        if np.isfinite(esm3_subset[index]).all()
    }
    esm3_common = common & esm3_finite
    if len(esm3_common) < 30:
        raise RuntimeError(
            f"ESM3-common universe is unexpectedly small: {len(esm3_common)}"
        )

    existing_path = (
        args.partial_output / "benchmark" / "osdr_go_kegg_term_scores.csv"
    )
    existing = pd.read_csv(existing_path)
    existing = existing[
        ~existing["model"]
        .astype(str)
        .str.upper()
        .isin({"ESM2_PCA512_PRIOR", "ESM3"})
    ].copy()
    go_existing = existing[
        (existing["library"] == "GO-BP") & (existing["control"] == "real")
    ]
    go = partial.load_library(
        args.output_dir,
        "GO_Biological_Process_2021",
        args.go_library,
    )
    frames = []
    for benchmark_variant, probe_variant in (("osdr_full", "full"),):
        terms = (
            go_existing.loc[
                go_existing["variant"] == benchmark_variant, "term"
            ]
            .drop_duplicates()
            .tolist()
        )
        if len(terms) != 40:
            raise RuntimeError(
                f"expected 40 frozen GO terms for {benchmark_variant}, got {len(terms)}"
            )
        frozen_library = {term: go[term] for term in terms}
        scores = probe.single_gene_scores(
            {
                "ESM2_PCA512_prior": (
                    existing_subset["ESM2_PCA512_prior"],
                    False,
                ),
                "ESM3": (esm3_subset, False),
            },
            universe,
            frozen_library,
            variant=probe_variant,
            common_genes=common if probe_variant == "common" else None,
            n_terms=40,
            min_positive=12,
        )
        scores["variant"] = benchmark_variant
        scores["library"] = "GO-BP"
        frames.append(scores)

    # A protein model cannot represent non-protein-coding genes. Recompute every
    # model on the strict intersection so the primary ESM3 comparison uses
    # identical genes, labels, and folds rather than mixing universes.
    common_terms = (
        go_existing.loc[
            go_existing["variant"] == "osdr_common", "term"
        ]
        .drop_duplicates()
        .tolist()
    )
    if len(common_terms) != 40:
        raise RuntimeError(
            f"expected 40 frozen common GO terms, got {len(common_terms)}"
        )
    common_library = {term: go[term] for term in common_terms}
    strict_embeddings = {
        model: (matrix, bool(existing_embeddings[model][1]))
        for model, matrix in existing_subset.items()
    }
    strict_embeddings["ESM3"] = (esm3_subset, False)
    strict = probe.single_gene_scores(
        strict_embeddings,
        universe,
        common_library,
        variant="common",
        common_genes=esm3_common,
        n_terms=40,
        min_positive=12,
    )
    strict["variant"] = "osdr_esm3_common"
    strict["library"] = "GO-BP"
    frames.append(strict)

    esm3_rows = pd.concat(frames, ignore_index=True)
    combined = pd.concat([existing, esm3_rows], ignore_index=True, sort=False)
    combined.to_csv(args.output_dir / "osdr_go_term_scores_with_esm3.csv", index=False)
    real = combined[
        (combined["library"] == "GO-BP") & (combined["control"] == "real")
    ].copy()
    summary = build_summary(real)
    summary.to_csv(args.output_dir / "osdr_go_model_summary_with_esm3.csv", index=False)

    base_models = [
        model
        for model in sorted(real["model"].unique())
        if model not in PARTIAL_MODELS.values()
    ]
    experiment_manifest = {}
    for experiment, partial_model in PARTIAL_MODELS.items():
        models = base_models + [partial_model]
        experiment = str(experiment)
        folder = args.output_dir / experiment
        folder.mkdir(parents=True, exist_ok=True)
        long = real[real["model"].isin(models)].copy()
        long.to_csv(folder / "individual_go_term_scores.csv", index=False)
        experiment_summary = build_summary(long)
        experiment_summary.to_csv(folder / "model_summary.csv", index=False)
        for variant in ("osdr_esm3_common", "osdr_full"):
            subset = long[long["variant"] == variant]
            for metric in ("auroc", "auprc"):
                wide = subset.pivot(index="term", columns="model", values=metric)
                wide.to_csv(folder / f"individual_go_{variant}_{metric}.csv")
            winners = (
                subset.loc[subset.groupby("term")["auroc"].idxmax()]
                [["term", "model", "auroc", "auprc", "n_pos"]]
                .sort_values("auroc", ascending=False)
            )
            winners.to_csv(folder / f"individual_go_{variant}_winners.csv", index=False)
        experiment_manifest[experiment] = {
            "partial_model": partial_model,
            "models": models,
            "terms": int(long["term"].nunique()),
            "rows": int(len(long)),
        }

    manifest = {
        "protocol": (
            "Frozen 40 GO Biological Process terms; 5-fold out-of-fold GPU "
            "linear probes; StandardScaler; seed 42; AUROC and AUPRC"
        ),
        "existing_results": str(existing_path),
        "esm2_embedding": str(args.esm2_prior),
        "esm3_embedding": str(args.esm3_npz),
        "common_genes_before_esm3": len(common),
        "esm3_finite_osdr_genes": len(esm3_finite),
        "strict_esm3_common_genes": len(esm3_common),
        "genes_removed_for_strict_esm3_comparison": len(common - esm3_common),
        "experiments": experiment_manifest,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
