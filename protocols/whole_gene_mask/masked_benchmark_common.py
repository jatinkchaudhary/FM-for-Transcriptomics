#!/usr/bin/env python3
"""Shared utilities for the whole-gene masking benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


MASK_TOKEN = -10.0


def read_genes(path: str | Path) -> list[str]:
    frame = pd.read_csv(path)
    for column in ("gene_symbol", "symbol", "gene", "genes"):
        if column in frame.columns:
            values = frame[column]
            break
    else:
        values = frame.iloc[:, -1]
    return [str(value).strip().upper() for value in values.dropna()]


def read_protocol(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "config" / "protocol.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_source_matrix(protocol: dict, dataset: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return expression, upper-case genes, and sample IDs on the model input scale."""
    if dataset == "TCGA":
        frame = pd.read_parquet(protocol["datasets"]["TCGA"]["matrix"])
        genes = [str(value).upper() for value in frame.columns]
        sample_ids = frame.index.astype(str).to_numpy()
        values = np.log1p(frame.to_numpy(dtype=np.float32, copy=False))
        return values.astype(np.float32, copy=False), genes, sample_ids
    if dataset == "OSDR":
        payload = np.load(protocol["datasets"]["OSDR"]["matrix"], allow_pickle=True)
        values = payload["X"].astype(np.float32, copy=False)
        genes = [str(value).upper() for value in payload["genes"]]
        if "sample_ids" in payload:
            sample_ids = payload["sample_ids"].astype(str)
        else:
            sample_ids = np.asarray([f"OSDR_{index}" for index in range(len(values))])
        return values, genes, sample_ids
    raise ValueError(f"Unsupported dataset: {dataset}")


def align_matrix(
    values: np.ndarray,
    source_genes: list[str],
    target_genes: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    source_index = {gene: index for index, gene in enumerate(source_genes)}
    target_index = {gene: index for index, gene in enumerate(target_genes)}
    output = np.zeros((values.shape[0], len(target_genes)), dtype=np.float32)
    source_columns = []
    target_columns = []
    for out_index, gene in enumerate(target_genes):
        in_index = source_index.get(gene)
        if in_index is not None:
            source_columns.append(in_index)
            target_columns.append(out_index)
    output[:, np.asarray(target_columns)] = values[:, np.asarray(source_columns)]
    return output, target_index


def load_mask(run_dir: str | Path, seed: int) -> list[str]:
    path = Path(run_dir) / "masks" / f"mask_seed_{seed}.csv"
    return pd.read_csv(path)["gene_symbol"].astype(str).str.upper().tolist()


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
