#!/usr/bin/env python3
"""Append Txn_Jatin's saved contextual table to a prepared protocol run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


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


def checkpoint_gene_weight(payload: dict) -> np.ndarray | None:
    state = payload.get("model_state_dict", {})
    candidates = [
        value
        for key, value in state.items()
        if key == "gene_embedding.weight" or key.endswith(".gene_embedding.weight")
    ]
    if len(candidates) != 1:
        return None
    return candidates[0].detach().float().cpu().numpy()


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
            mapped[entrez].append(vector)

    genes = list(mapped)
    values = np.vstack(
        [np.mean(np.vstack(mapped[gene]), axis=0, dtype=np.float64) for gene in genes]
    ).astype(np.float32)
    return genes, values


def write_embedding(
    root: Path, model: str, genes: list[str], values: np.ndarray
) -> None:
    folder = root / model
    atomic_csv(folder / f"{model}_emb.csv", values)
    atomic_text(folder / f"{model}_genelist.txt", "\n".join(genes) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--canonical-genes", type=Path, required=True)
    parser.add_argument("--static-source", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--embeddings-root", type=Path, required=True)
    parser.add_argument("--model", default="Txn_Jatin_contextual")
    parser.add_argument(
        "--context-key",
        default="contextual_gene_embedding",
        choices=("contextual_gene_embedding", "training_contextual_gene_embedding"),
    )
    args = parser.parse_args()

    with np.load(args.static_source, allow_pickle=True) as static_payload:
        if bool(static_payload["is_fallback"].item()):
            raise ValueError("Txn_Jatin static source is a fallback embedding")
        symbols = static_payload["keys"].astype(str)
        static_values = static_payload["emb"].astype(np.float32)

    canonical_frame = pd.read_csv(args.canonical_genes)
    if "gene_symbol" not in canonical_frame:
        raise ValueError("canonical gene table has no gene_symbol column")
    canonical_symbols = canonical_frame["gene_symbol"].astype(str).to_numpy()
    canonical_index = {
        symbol.upper(): index for index, symbol in enumerate(canonical_symbols)
    }
    if len(canonical_index) != len(canonical_symbols):
        raise ValueError("canonical gene symbols are not unique after uppercasing")
    missing_symbols = [
        symbol for symbol in symbols if symbol.upper() not in canonical_index
    ]
    if missing_symbols:
        raise ValueError(
            f"static export has {len(missing_symbols)} symbols absent from canonical genes"
        )
    selected_rows = np.array(
        [canonical_index[symbol.upper()] for symbol in symbols], dtype=np.int64
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.context_key not in checkpoint or checkpoint[args.context_key] is None:
        raise KeyError(f"checkpoint does not contain {args.context_key}")
    full_contextual = checkpoint[args.context_key].detach().float().cpu().numpy()
    if full_contextual.shape[0] != len(canonical_symbols):
        raise ValueError(
            f"context rows {full_contextual.shape[0]} != canonical rows "
            f"{len(canonical_symbols)}"
        )
    if full_contextual.ndim != 2 or not np.isfinite(full_contextual).all():
        raise ValueError("contextual table is not a finite rank-2 matrix")
    contextual = full_contextual[selected_rows]

    weight = checkpoint_gene_weight(checkpoint)
    static_match = None
    if weight is not None and weight.shape[0] == len(canonical_symbols):
        selected_weight = weight[selected_rows]
        norm = np.linalg.norm(selected_weight, axis=1, keepdims=True)
        normalized_weight = selected_weight / np.maximum(norm, 1e-12)
        static_match = float(np.max(np.abs(normalized_weight - static_values)))
        if static_match > 5e-3:
            raise ValueError(
                "checkpoint does not match the Txn_Jatin static export "
                f"(max normalized-weight error {static_match:.6g})"
            )

    sample_key = (
        "contextual_gene_embedding_samples"
        if args.context_key == "contextual_gene_embedding"
        else "training_contextual_gene_embedding_samples"
    )
    provenance = {
        "role": "contextual representation",
        "source": "saved frozen-model mean contextual gene states",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "context_key": args.context_key,
        "context_samples": int(checkpoint.get(sample_key, 0)),
        "canonical_genes": str(args.canonical_genes),
        "canonical_rows": int(len(canonical_symbols)),
        "static_source": str(args.static_source),
        "static_checkpoint_match_max_abs_error": static_match,
    }
    del checkpoint, weight, full_contextual

    args.source_output.parent.mkdir(parents=True, exist_ok=True)
    source_tmp = args.source_output.with_name(args.source_output.name + ".tmp.npz")
    np.savez_compressed(
        source_tmp,
        keys=symbols,
        emb=contextual.astype(np.float32, copy=False),
        is_fallback=np.array(False),
        provenance=np.array(json.dumps(provenance, sort_keys=True)),
    )
    source_tmp.replace(args.source_output)

    mapping = pd.read_csv(
        args.embeddings_root / "symbol_to_entrez_mygene.csv",
        dtype={"symbol": str, "entrez": str},
    )
    genes, values = collapse_to_entrez(symbols, contextual, mapping)
    if not genes:
        raise ValueError("contextual table has no Entrez-mapped rows")
    write_embedding(args.embeddings_root / "all_genes", args.model, genes, values)

    reference_dir = args.embeddings_root / "intersect" / "Txn_Jatin"
    reference_gene_file = next(reference_dir.glob("*.txt"))
    intersection = [
        line.strip() for line in reference_gene_file.read_text().splitlines() if line.strip()
    ]
    row_index = {gene: index for index, gene in enumerate(genes)}
    missing = [gene for gene in intersection if gene not in row_index]
    if missing:
        raise ValueError(f"contextual table is missing {len(missing)} intersection genes")
    intersect_values = np.vstack([values[row_index[gene]] for gene in intersection])
    write_embedding(
        args.embeddings_root / "intersect", args.model, intersection, intersect_values
    )

    manifest_path = args.embeddings_root / "embedding_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["models"][args.model] = {
        "source": str(args.source_output),
        "source_sha256": sha256(args.source_output),
        "symbol_rows": int(len(symbols)),
        "entrez_rows": int(len(genes)),
        "dimensions": int(values.shape[1]),
        "provenance": provenance,
    }
    manifest["model_count"] = len(manifest["models"])
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    report = {
        "model": args.model,
        "symbol_rows": int(len(symbols)),
        "entrez_rows": int(len(genes)),
        "intersect_rows": int(len(intersection)),
        "dimensions": int(values.shape[1]),
        "finite": True,
        "provenance": provenance,
    }
    report_path = args.embeddings_root.parent / "results" / "contextual_append.json"
    atomic_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
