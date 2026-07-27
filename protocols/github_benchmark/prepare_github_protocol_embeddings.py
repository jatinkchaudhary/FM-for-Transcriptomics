#!/usr/bin/env python3
"""Convert benchmark NPZ exports to the Zhong et al. Entrez/CSV layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import mygene
import numpy as np
import pandas as pd

# NumPy 2.x pickles refer to ``numpy._core``; expose the equivalent 1.24
# module while reading the previously exported object-valued metadata arrays.
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mapping(symbols: list[str], cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_csv(cache_path, dtype={"symbol": str, "entrez": str})
        if set(symbols).issubset(set(cached["symbol"])):
            return cached

    client = mygene.MyGeneInfo()
    records = []
    for start in range(0, len(symbols), 1000):
        chunk = symbols[start : start + 1000]
        response = client.querymany(
            chunk,
            scopes="symbol",
            fields="entrezgene",
            species="human",
            as_dataframe=False,
            verbose=False,
        )
        for item in response:
            symbol = str(item.get("query", "")).upper()
            entrez = item.get("entrezgene")
            if not symbol or entrez is None or item.get("notfound"):
                continue
            values = entrez if isinstance(entrez, list) else [entrez]
            for value in values:
                try:
                    records.append((symbol, str(int(float(value)))))
                except (TypeError, ValueError):
                    continue

    mapping = pd.DataFrame(records, columns=["symbol", "entrez"]).drop_duplicates()
    missing = sorted(set(symbols) - set(mapping["symbol"]))
    mapping = pd.concat(
        [mapping, pd.DataFrame({"symbol": missing, "entrez": [""] * len(missing)})],
        ignore_index=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(cache_path, index=False)
    return mapping


def collapse_to_entrez(
    symbols: np.ndarray, embeddings: np.ndarray, mapping: pd.DataFrame
) -> tuple[list[str], np.ndarray]:
    mapped = defaultdict(list)
    symbol_map = defaultdict(list)
    for row in mapping.itertuples(index=False):
        if row.entrez and str(row.entrez) != "nan":
            symbol_map[str(row.symbol).upper()].append(str(row.entrez))

    for index, symbol in enumerate(symbols):
        vector = embeddings[index]
        if not np.isfinite(vector).all():
            continue
        for entrez in symbol_map.get(str(symbol).upper(), []):
            mapped[entrez].append(vector.astype(np.float32, copy=False))

    genes = list(mapped)
    values = np.vstack(
        [np.mean(np.vstack(mapped[gene]), axis=0, dtype=np.float64) for gene in genes]
    ).astype(np.float32)
    return genes, values


def write_embedding(root: Path, model: str, genes: list[str], values: np.ndarray) -> None:
    folder = root / model
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values).to_csv(folder / f"{model}_emb.csv", header=False, index=False)
    (folder / f"{model}_genelist.txt").write_text("\n".join(genes) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--esm2-prior", type=Path, required=True)
    parser.add_argument("--official-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.source_dir.glob("*__gene_benchmark__symbol.npz"))
    if not sources:
        raise FileNotFoundError(f"no benchmark NPZ exports under {args.source_dir}")

    raw_models: dict[str, tuple[np.ndarray, np.ndarray, Path, dict]] = {}
    all_symbols = set()
    for source in sources:
        model = source.name.split("__gene_benchmark__symbol.npz")[0]
        with np.load(source, allow_pickle=True) as payload:
            if bool(payload["is_fallback"].item()):
                raise ValueError(f"{model} is a fallback embedding")
            symbols = payload["keys"].astype(str)
            values = payload["emb"].astype(np.float32)
            provenance = json.loads(str(payload["provenance"].reshape(-1)[0]))
        raw_models[model] = (symbols, values, source, provenance)
        all_symbols.update(str(symbol).upper() for symbol in symbols)

    with np.load(args.esm2_prior, allow_pickle=False) as payload:
        esm_symbols = payload["genes"].astype(str)
        esm_values = payload["embeddings"].astype(np.float32)
        covered = payload["covered"].astype(bool)
        esm_values[~covered] = np.nan
    raw_models["ESM2_PCA512_prior"] = (
        esm_symbols,
        esm_values,
        args.esm2_prior,
        {"role": "training-prior ablation", "source": "ESM2 PCA-512"},
    )
    all_symbols.update(str(symbol).upper() for symbol in esm_symbols)

    mapping_path = args.output_dir / "symbol_to_entrez_mygene.csv"
    mapping = load_mapping(sorted(all_symbols), mapping_path)

    converted: dict[str, tuple[list[str], np.ndarray]] = {}
    manifest = {
        "protocol": "ylaboratory/gene-embedding-benchmarks",
        "mapping": "mygene symbol -> human entrezgene; duplicate Entrez rows averaged",
        "models": {},
    }
    for model, (symbols, values, source, provenance) in raw_models.items():
        genes, matrix = collapse_to_entrez(symbols, values, mapping)
        if not genes:
            raise ValueError(f"no Entrez-mapped rows for {model}")
        converted[model] = (genes, matrix)
        write_embedding(args.output_dir / "all_genes", model, genes, matrix)
        manifest["models"][model] = {
            "source": str(source),
            "source_sha256": sha256(source),
            "symbol_rows": int(len(symbols)),
            "entrez_rows": int(len(genes)),
            "dimensions": int(matrix.shape[1]),
            "provenance": provenance,
        }

    common = set.intersection(*(set(genes) for genes, _ in converted.values()))
    official = {
        line.strip()
        for line in args.official_reference.read_text().splitlines()
        if line.strip()
    }
    common_official = sorted(common & official)
    if not common_official:
        raise ValueError("official-reference common intersection is empty")

    for model, (genes, matrix) in converted.items():
        index = {gene: i for i, gene in enumerate(genes)}
        rows = np.vstack([matrix[index[gene]] for gene in common_official])
        write_embedding(args.output_dir / "intersect", model, common_official, rows)

    manifest["common_across_local_models"] = len(common)
    manifest["official_reference_genes"] = len(official)
    manifest["intersect_genes"] = len(common_official)
    manifest["intersect_definition"] = (
        "intersection(all local models) AND official 38-model reference intersection"
    )
    (args.output_dir / "embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
