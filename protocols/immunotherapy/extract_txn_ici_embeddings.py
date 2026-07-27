#!/usr/bin/env python3
"""Extract frozen Txn_Jatin sample representations on CUDA."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_model(checkpoint: Path, train_flash: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    state = payload["model_state_dict"]
    spec = importlib.util.spec_from_file_location("txn_train_flash", train_flash)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    model = module.ExpressionPerformer(
        num_genes=int(state["gene_embedding.weight"].shape[0]),
        hidden_dim=int(config["hidden_dim"]),
        n_heads=int(config["num_heads"]),
        n_layers=int(config["num_layers"]),
        ffn_dim=int(config["ffn_dim"]),
        ree_base=float(config["ree_base"]),
        mask_token_id=float(config.get("mask_token", -10)),
        feature_type=config.get("feature_type", "flash"),
        compute_type=config.get("compute_type", "iter"),
        include_species_embedding=bool(config.get("include_species_embedding", False)),
        num_species=int(config.get("architecture", {}).get("num_species", 2)),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model.to(device).eval(), payload, config


def read_gene_order(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    for column in ("gene_symbol", "symbol", "gene"):
        if column in frame.columns:
            return frame[column].astype(str).str.upper().tolist()
    raise ValueError(f"No gene-symbol column in {path}")


def align_input(
    path: Path, gene_order: list[str], mask_token: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        source = np.asarray(data["X"], dtype=np.float32)
        sample_ids = data["sample_ids"].astype(str)
        source_genes = data["genes"].astype(str)
    source_index = {gene.upper(): i for i, gene in enumerate(source_genes)}
    raw_aligned = np.zeros((source.shape[0], len(gene_order)), dtype=np.float32)
    model_input = np.full(
        (source.shape[0], len(gene_order)), mask_token, dtype=np.float32
    )
    covered_positions = []
    source_positions = []
    for out_index, gene in enumerate(gene_order):
        source_index_value = source_index.get(gene)
        if source_index_value is not None:
            covered_positions.append(out_index)
            source_positions.append(source_index_value)
    raw_aligned[:, covered_positions] = source[:, source_positions]
    model_input[:, covered_positions] = source[:, source_positions]
    measured_gene_mask = np.zeros(len(gene_order), dtype=bool)
    measured_gene_mask[covered_positions] = True
    coverage = {
        "source_genes": int(len(source_genes)),
        "model_genes": int(len(gene_order)),
        "covered_model_genes": int(len(covered_positions)),
        "missing_model_genes": int(len(gene_order) - len(covered_positions)),
        "coverage_fraction": float(len(covered_positions) / len(gene_order)),
    }
    return model_input, raw_aligned, sample_ids, measured_gene_mask, coverage


def extract(
    model,
    matrix: np.ndarray,
    mask_token: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    gene_identity = model.gene_embedding.weight.detach()
    context_mean_parts = []
    hidden_mean_parts = []
    hidden_all_parts = []
    reconstruction_parts = []
    for start in range(0, len(matrix), batch_size):
        xb = torch.from_numpy(matrix[start : start + batch_size]).to(
            device, non_blocking=True
        )
        with torch.inference_mode(), torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            hidden = model.encode_hidden(xb)
            context = hidden - gene_identity.to(dtype=hidden.dtype).unsqueeze(0)
            observed = (xb != mask_token).to(dtype=hidden.dtype)
            context_mean = (context * observed.unsqueeze(-1)).sum(dim=1)
            context_mean = context_mean / observed.sum(dim=1, keepdim=True).clamp(
                min=1.0
            )
            hidden_mean = hidden.mean(dim=1)
            hidden_all = (
                hidden.max(dim=1).values
                + hidden_mean
                + hidden.median(dim=1).values
            )
            reconstruction = model.output_map(hidden).squeeze(-1)
        context_mean_parts.append(context_mean.float().cpu().numpy())
        hidden_mean_parts.append(hidden_mean.float().cpu().numpy())
        hidden_all_parts.append(hidden_all.float().cpu().numpy())
        reconstruction_parts.append(reconstruction.float().cpu().numpy())
        print(
            f"embedded {min(start + batch_size, len(matrix))}/{len(matrix)}",
            flush=True,
        )
    return {
        "txn_context_mean": np.concatenate(context_mean_parts).astype(np.float32),
        "txn_hidden_mean": np.concatenate(hidden_mean_parts).astype(np.float32),
        "txn_hidden_all": np.concatenate(hidden_all_parts).astype(np.float32),
        "txn_reconstruction": np.concatenate(reconstruction_parts).astype(np.float32),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--canonical-genes", type=Path, required=True)
    parser.add_argument("--train-flash", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this extraction")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model, payload, config = load_model(args.checkpoint, args.train_flash, device)
    mask_token = float(config.get("mask_token", -10))
    gene_order = read_gene_order(args.canonical_genes)
    if len(gene_order) != model.gene_embedding.weight.shape[0]:
        raise ValueError("Canonical gene order does not match checkpoint")

    coverage = {}
    for cohort in ("gide", "riaz", "hugo", "rose"):
        matrix, raw_aligned, sample_ids, measured_gene_mask, cohort_coverage = align_input(
            args.input_dir / f"{cohort}_log1p_tpm.npz", gene_order, mask_token
        )
        print(f"{cohort}: {matrix.shape}, {cohort_coverage}", flush=True)
        representations = extract(
            model, matrix, mask_token, args.batch_size, device
        )
        completed = raw_aligned.copy()
        completed[:, ~measured_gene_mask] = representations["txn_reconstruction"][
            :, ~measured_gene_mask
        ]
        np.savez_compressed(
            args.output_dir / f"{cohort}_txn_embeddings.npz",
            sample_ids=sample_ids,
            genes=np.asarray(gene_order, dtype=str),
            measured_gene_mask=measured_gene_mask,
            aligned_log1p_tpm=raw_aligned,
            txn_completed_log1p_tpm=completed,
            **representations,
        )
        coverage[cohort] = cohort_coverage

    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_val_loss": float(payload.get("val_loss", np.nan)),
        "model_name": config.get("model_name", "Txn_Jatin"),
        "num_genes": int(len(gene_order)),
        "hidden_dim": int(config["hidden_dim"]),
        "num_layers": int(config["num_layers"]),
        "sample_contrastive_weight": float(
            config.get("sample_contrastive_weight", 0.0)
        ),
        "input_scale": "log1p(TPM)",
        "representations": {
            "txn_context_mean": (
                "measured-gene mean of final hidden state minus static gene embedding; "
                "matches the representation used by the sample contrastive objective"
            ),
            "txn_hidden_mean": "mean final hidden state over genes",
            "txn_hidden_all": "max + mean + median final hidden state over genes",
            "txn_reconstruction": "decoder output for all genes",
            "txn_completed_log1p_tpm": (
                "observed expression retained; assay-missing genes filled by decoder"
            ),
        },
        "missing_gene_input": f"mask token {mask_token}, not biological zero",
        "batch_size": int(args.batch_size),
        "amp_dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "coverage": coverage,
        "elapsed_seconds": float(time.time() - started),
    }
    (args.output_dir / "extraction_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
