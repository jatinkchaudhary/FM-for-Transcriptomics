#!/usr/bin/env python3
"""Append one symbol-keyed NPZ embedding to an existing GitHub protocol layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# NumPy 2.x object arrays may pickle module paths as ``numpy._core``.
# The GitHub benchmark env uses NumPy 1.x, where the same objects live under
# ``numpy.core``. Expose aliases before reading object-valued NPZ fields.
import sys

sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(values).to_csv(temporary, header=False, index=False)
    temporary.replace(path)


def read_provenance(payload: np.lib.npyio.NpzFile) -> dict:
    if "provenance" not in payload:
        return {}
    raw = payload["provenance"].reshape(-1)[0]
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {"raw_provenance": str(raw)}


def collapse_to_entrez(
    symbols: np.ndarray, embeddings: np.ndarray, mapping: pd.DataFrame
) -> tuple[list[str], np.ndarray]:
    symbol_map: dict[str, list[str]] = defaultdict(list)
    for row in mapping.itertuples(index=False):
        entrez = str(row.entrez)
        if entrez and entrez.lower() != "nan":
            symbol_map[str(row.symbol).upper()].append(entrez)

    mapped: dict[str, list[np.ndarray]] = defaultdict(list)
    for index, symbol in enumerate(symbols):
        vector = embeddings[index]
        if not np.isfinite(vector).all():
            continue
        for entrez in symbol_map.get(str(symbol).upper(), []):
            mapped[entrez].append(vector.astype(np.float32, copy=False))

    genes = list(mapped)
    if not genes:
        raise ValueError("no finite rows mapped to Entrez IDs")
    values = np.vstack(
        [np.mean(np.vstack(mapped[gene]), axis=0, dtype=np.float64) for gene in genes]
    ).astype(np.float32)
    return genes, values


def write_embedding(root: Path, model: str, genes: list[str], values: np.ndarray) -> None:
    folder = root / model
    atomic_csv(folder / f"{model}_emb.csv", values)
    atomic_text(folder / f"{model}_genelist.txt", "\n".join(genes) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--source-output", type=Path)
    parser.add_argument("--embeddings-root", type=Path, required=True)
    parser.add_argument("--intersection-genelist", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    with np.load(args.source_npz, allow_pickle=True) as payload:
        symbols = payload["keys"].astype(str)
        values = payload["emb"].astype(np.float32)
        fallback = bool(payload["is_fallback"].item()) if "is_fallback" in payload else False
        provenance = read_provenance(payload)
    if fallback:
        raise ValueError(f"{args.source_npz} is marked as a fallback embedding")
    if values.ndim != 2 or len(symbols) != values.shape[0]:
        raise ValueError("source NPZ must contain rank-2 emb rows matching keys")

    original_name = provenance.get("name")
    provenance = {
        **provenance,
        "name": args.model,
        "original_provenance_name": original_name,
        "retagged_for": "OSDR fine-tuned Txn_Jatin GitHub gene-level benchmark",
        "source_symbol_npz": str(args.source_npz),
        "source_symbol_npz_sha256": sha256(args.source_npz),
    }

    source_for_manifest = args.source_npz
    if args.source_output is not None:
        args.source_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.source_output.with_name(args.source_output.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            keys=symbols,
            emb=values,
            is_fallback=np.asarray(False),
            provenance=np.asarray([json.dumps(provenance, sort_keys=True)], dtype=object),
        )
        temporary.replace(args.source_output)
        source_for_manifest = args.source_output

    mapping = pd.read_csv(
        args.embeddings_root / "symbol_to_entrez_mygene.csv",
        dtype={"symbol": str, "entrez": str},
    )
    genes, entrez_values = collapse_to_entrez(symbols, values, mapping)
    write_embedding(args.embeddings_root / "all_genes", args.model, genes, entrez_values)

    intersection = [
        line.strip()
        for line in args.intersection_genelist.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    row_index = {gene: index for index, gene in enumerate(genes)}
    missing = [gene for gene in intersection if gene not in row_index]
    if missing:
        raise ValueError(f"{args.model} is missing {len(missing)} intersection genes")
    intersect_values = np.vstack([entrez_values[row_index[gene]] for gene in intersection])
    write_embedding(args.embeddings_root / "intersect", args.model, intersection, intersect_values)

    manifest_path = args.embeddings_root / "embedding_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "protocol": "ylaboratory/gene-embedding-benchmarks",
            "mapping": "mygene symbol -> human entrezgene; duplicate Entrez rows averaged",
            "models": {},
        }
    manifest["models"][args.model] = {
        "source": str(source_for_manifest),
        "source_sha256": sha256(source_for_manifest),
        "symbol_rows": int(len(symbols)),
        "entrez_rows": int(len(genes)),
        "intersect_rows": int(len(intersection)),
        "dimensions": int(entrez_values.shape[1]),
        "provenance": provenance,
    }
    manifest["model_count"] = len(manifest["models"])
    manifest["intersect_definition"] = (
        "same fixed intersection genelist used by the prior GitHub protocol comparison"
    )
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    report = {
        "model": args.model,
        "symbol_rows": int(len(symbols)),
        "entrez_rows": int(len(genes)),
        "intersect_rows": int(len(intersection)),
        "dimensions": int(entrez_values.shape[1]),
        "source": str(source_for_manifest),
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
