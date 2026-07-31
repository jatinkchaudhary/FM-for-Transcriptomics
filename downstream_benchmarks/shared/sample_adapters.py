#!/usr/bin/env python3
"""Strict native sample-embedding adapters for downstream benchmarks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runtime(runtime_path: str | Path, config_path: str | Path):
    runtime_path = Path(runtime_path)
    if str(runtime_path.parent) not in sys.path:
        sys.path.insert(0, str(runtime_path.parent))
    spec = importlib.util.spec_from_file_location("espresso_model_runtime", runtime_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    public_models = [
        {"id": name, "label": name, "imputation_supported": True}
        for name in config["models"]
    ]
    return module.ModelRuntime(Path(config_path), public_models)


def align_expression(
    expression: pd.DataFrame, model_genes: list[str], fill: float = -2.0
) -> np.ndarray:
    """Align sample x gene log-expression without silently standardizing."""
    source = {str(g).upper(): i for i, g in enumerate(expression.columns)}
    aligned = np.full((len(expression), len(model_genes)), fill, dtype=np.float32)
    for j, gene in enumerate(model_genes):
        i = source.get(str(gene).upper())
        if i is not None:
            aligned[:, j] = expression.iloc[:, i].to_numpy(dtype=np.float32)
    return aligned


def native_embedding(
    expression: pd.DataFrame,
    model_name: str,
    runtime_path: str | Path,
    config_path: str | Path,
) -> tuple[np.ndarray, dict]:
    """Extract a native frozen sample representation; never substitute a fallback."""
    runtime = load_runtime(runtime_path, config_path)
    canonical = runtime.ensure_loaded(model_name)
    aligned = align_expression(expression, runtime.model_genes)
    if canonical.startswith("BulkFormer_"):
        import torch

        chunks = []
        batch_size = 16
        for start in range(0, len(aligned), batch_size):
            tensor = torch.from_numpy(aligned[start : start + batch_size]).to(runtime.device)
            with torch.inference_mode(), torch.autocast(
                device_type=runtime.device.type,
                dtype=torch.bfloat16,
                enabled=False,
            ):
                hidden = runtime.model(tensor, mask_prob=0.0, output_expr=False)
                if isinstance(hidden, (tuple, list)):
                    hidden = hidden[0]
                pooled = hidden.float().mean(dim=1)
            chunks.append(pooled.cpu().numpy())
        embeddings = np.concatenate(chunks).astype(np.float32)
    else:
        embeddings = runtime._embed_aligned(aligned).astype(np.float32)
    checkpoint = runtime.config["models"][canonical]["checkpoint"]
    provenance = {
        "model": canonical,
        "kind": "native_frozen_sample_embedding",
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "preprocessing": "input log1p-TPM; canonical gene alignment; absent=-2 mask token",
        "pooling": "mean gene hidden states",
        "n_samples": int(len(expression)),
        "input_genes": int(expression.shape[1]),
        "model_genes": int(len(runtime.model_genes)),
        "dimension": int(embeddings.shape[1]),
    }
    runtime.unload()
    return embeddings, provenance


def raw_embedding(expression: pd.DataFrame) -> tuple[np.ndarray, dict]:
    values = expression.to_numpy(dtype=np.float32)
    return values, {
        "model": "raw_log1p_TPM",
        "kind": "expression_baseline",
        "preprocessing": "log1p-TPM",
        "n_samples": int(values.shape[0]),
        "input_genes": int(values.shape[1]),
        "dimension": int(values.shape[1]),
    }


def pca_embedding(
    expression: pd.DataFrame, dimension: int = 64
) -> tuple[np.ndarray, dict]:
    values = expression.to_numpy(dtype=np.float32)
    embedding = PCA(n_components=dimension, random_state=42).fit_transform(
        StandardScaler().fit_transform(values)
    )
    return embedding.astype(np.float32), {
        "model": "PCA64",
        "kind": "expression_baseline",
        "preprocessing": "log1p-TPM; gene z-score; PCA fit on benchmark cohort",
        "n_samples": int(values.shape[0]),
        "input_genes": int(values.shape[1]),
        "dimension": int(embedding.shape[1]),
    }


def save_embedding(
    output: str | Path,
    embeddings: np.ndarray,
    sample_ids: list[str] | np.ndarray,
    provenance: dict,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        sample_ids=np.asarray(sample_ids, dtype=str),
        provenance=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
