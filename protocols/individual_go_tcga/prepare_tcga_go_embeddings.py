#!/usr/bin/env python3
"""Prepare TCGA-restricted model embeddings for the official GitHub GO task."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-embeddings", type=Path, required=True)
    parser.add_argument("--tcga-parquet", type=Path, required=True)
    parser.add_argument("--symbol-map", type=Path, required=True)
    parser.add_argument("--esm3-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_embedding(folder: Path) -> tuple[list[str], np.ndarray]:
    csvs = sorted(folder.glob("*.csv"))
    txts = sorted(folder.glob("*.txt"))
    if len(csvs) != 1 or len(txts) != 1:
        raise RuntimeError(f"expected one CSV and one gene list in {folder}")
    genes = [line.strip() for line in txts[0].read_text().splitlines() if line.strip()]
    values = pd.read_csv(csvs[0], header=None, dtype=np.float32).to_numpy(
        dtype=np.float32
    )
    if values.shape[0] != len(genes) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid embedding in {folder}: {values.shape}")
    return genes, values


def write_embedding(
    root: Path, model: str, genes: list[str], values: np.ndarray
) -> None:
    folder = root / model
    folder.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(values).to_csv(
        folder / f"{model}_emb.csv", header=False, index=False
    )
    (folder / f"{model}_genelist.txt").write_text(
        "\n".join(genes) + "\n", encoding="ascii"
    )


def symbol_to_entrez_map(table: pd.DataFrame) -> dict[str, list[str]]:
    mapped: dict[str, list[str]] = defaultdict(list)
    for row in table.itertuples(index=False):
        symbol = str(row.symbol).upper()
        value = str(row.entrez)
        if value.lower() == "nan" or not value:
            continue
        try:
            entrez = str(int(float(value)))
        except ValueError:
            continue
        if entrez not in mapped[symbol]:
            mapped[symbol].append(entrez)
    return dict(mapped)


def collapse_symbol_npz(
    path: Path, mapping: dict[str, list[str]]
) -> tuple[list[str], np.ndarray, dict]:
    with np.load(path, allow_pickle=True) as payload:
        symbols = [str(value).upper() for value in payload["keys"].tolist()]
        values = payload["emb"].astype(np.float32)
        fallback = bool(payload["is_fallback"])
        provenance = json.loads(str(payload["provenance"].reshape(-1)[0]))
    if fallback:
        raise RuntimeError(f"{path} is marked as a fallback")
    if values.shape[0] != len(symbols):
        raise RuntimeError(f"gene/embedding mismatch in {path}")

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = defaultdict(int)
    for index, symbol in enumerate(symbols):
        vector = values[index]
        if not np.isfinite(vector).all():
            continue
        for entrez in mapping.get(symbol, []):
            if entrez not in sums:
                sums[entrez] = vector.astype(np.float64)
            else:
                sums[entrez] += vector
            counts[entrez] += 1
    genes = list(sums)
    matrix = np.vstack(
        [(sums[gene] / counts[gene]).astype(np.float32) for gene in genes]
    )
    return genes, matrix, provenance


def subset(
    genes: list[str], values: np.ndarray, keep: set[str] | list[str]
) -> tuple[list[str], np.ndarray]:
    keep_set = set(keep)
    rows = [index for index, gene in enumerate(genes) if gene in keep_set]
    return [genes[index] for index in rows], values[rows]


def main() -> int:
    args = parse_args()
    for field in vars(args):
        value = getattr(args, field)
        if isinstance(value, Path):
            setattr(args, field, value.resolve())
    for path in (
        args.source_embeddings,
        args.tcga_parquet,
        args.symbol_map,
        args.esm3_npz,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite prepared embeddings: {args.output_dir}"
        )

    build_dir = args.output_dir.with_name(args.output_dir.name + ".building")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    all_root = build_dir / "tcga_all_genes"
    intersect_root = build_dir / "tcga_strict_intersection"
    all_root.mkdir(parents=True)
    intersect_root.mkdir(parents=True)

    mapping_table = pd.read_csv(args.symbol_map, dtype=str)
    required = {"symbol", "entrez"}
    if not required.issubset(mapping_table.columns):
        raise RuntimeError(f"{args.symbol_map} lacks {required}")
    mapping = symbol_to_entrez_map(mapping_table)

    schema_names = pq.ParquetFile(args.tcga_parquet).schema.names
    tcga_symbols = {
        str(name).upper() for name in schema_names if str(name) != "sample_id"
    }
    if len(tcga_symbols) != 15165:
        raise RuntimeError(f"expected 15,165 TCGA genes, got {len(tcga_symbols)}")
    tcga_entrez = {
        entrez
        for symbol in tcga_symbols
        for entrez in mapping.get(symbol, [])
    }
    if len(tcga_entrez) < 14000:
        raise RuntimeError(
            f"unexpectedly small TCGA Entrez universe: {len(tcga_entrez)}"
        )

    source_manifest_path = args.source_embeddings / "embedding_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_all = args.source_embeddings / "all_genes"
    source_models = sorted(
        path.name
        for path in source_all.iterdir()
        if path.is_dir()
    )
    expected_source = {
        "BRIDGE",
        "BulkFormer_37M",
        "BulkFormer_50M",
        "BulkFormer_93M",
        "BulkFormer_127M",
        "BulkFormer_147M",
        "ESM2_PCA512_prior",
        "Geneformer",
        "Txn_Jatin",
        "Txn_Jatin_contextual",
        "scGPT",
    }
    if set(source_models) != expected_source:
        raise RuntimeError(
            f"source model mismatch: {sorted(set(source_models) ^ expected_source)}"
        )

    model_metadata: dict[str, dict] = {}
    gene_sets: dict[str, set[str]] = {}
    for model in source_models:
        genes, values = load_embedding(source_all / model)
        genes, values = subset(genes, values, tcga_entrez)
        if len(genes) < 12000:
            raise RuntimeError(f"{model} has only {len(genes)} TCGA-overlap genes")
        write_embedding(all_root, model, genes, values)
        gene_sets[model] = set(genes)
        metadata = dict(source_manifest["models"].get(model, {}))
        metadata.update(
            {
                "representation": (
                    "frozen embedding preserved from the completed pinned GitHub run"
                ),
                "tcga_entrez_rows": len(genes),
                "dimensions": int(values.shape[1]),
            }
        )
        model_metadata[model] = metadata
        print(f"[prepare] {model}: {values.shape}", flush=True)

    extra_npzs = {"ESM3": args.esm3_npz}
    for model, path in extra_npzs.items():
        genes, values, provenance = collapse_symbol_npz(path, mapping)
        genes, values = subset(genes, values, tcga_entrez)
        if len(genes) < 12000:
            raise RuntimeError(f"{model} has only {len(genes)} TCGA-overlap genes")
        write_embedding(all_root, model, genes, values)
        gene_sets[model] = set(genes)
        model_metadata[model] = {
            "source": str(path),
            "source_sha256": sha256(path),
            "provenance": provenance,
            "representation": "native frozen ESM3 sequence embedding",
            "tcga_entrez_rows": len(genes),
            "dimensions": int(values.shape[1]),
        }
        print(f"[prepare] {model}: {values.shape}", flush=True)

    if len(gene_sets) != 12:
        raise RuntimeError(f"expected 12 models, got {len(gene_sets)}")

    official_folder = args.source_embeddings / "intersect" / "BRIDGE"
    official_txt = next(official_folder.glob("*.txt"))
    official_order = [
        line.strip() for line in official_txt.read_text().splitlines() if line.strip()
    ]
    strict_set = set.intersection(*gene_sets.values()) & set(official_order)
    strict_order = [gene for gene in official_order if gene in strict_set]
    if len(strict_order) < 8500:
        raise RuntimeError(
            f"strict 13-model TCGA intersection is too small: {len(strict_order)}"
        )

    for model in sorted(gene_sets):
        genes, values = load_embedding(all_root / model)
        index = {gene: row for row, gene in enumerate(genes)}
        rows = np.asarray([index[gene] for gene in strict_order], dtype=np.int64)
        write_embedding(intersect_root, model, strict_order, values[rows])
        model_metadata[model]["strict_intersection_rows"] = len(strict_order)

    pd.DataFrame({"gene_symbol": sorted(tcga_symbols)}).to_csv(
        build_dir / "tcga_gene_universe_symbols.csv", index=False
    )
    pd.DataFrame({"entrez_id": sorted(tcga_entrez)}).to_csv(
        build_dir / "tcga_gene_universe_entrez.csv", index=False
    )
    manifest = {
        "protocol": "ylaboratory/gene-embedding-benchmarks GO holdout protocol",
        "tcga_definition": (
            "Processed TCGA TPM matrix gene columns mapped from HGNC symbols to "
            "human Entrez identifiers using the original GitHub-run mapping cache"
        ),
        "tcga_parquet": str(args.tcga_parquet),
        "tcga_parquet_sha256": sha256(args.tcga_parquet),
        "tcga_symbol_genes": len(tcga_symbols),
        "tcga_entrez_genes": len(tcga_entrez),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256(source_manifest_path),
        "strict_intersection_definition": (
            "TCGA Entrez genes AND all 12 local model vocabularies AND the pinned "
            "official 38-model reference intersection"
        ),
        "strict_intersection_genes": len(strict_order),
        "models": model_metadata,
    }
    (build_dir / "prepared_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_dir.replace(args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
