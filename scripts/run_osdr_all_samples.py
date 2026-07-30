#!/usr/bin/env python3
"""Evaluate the selected OSDR-adapted Txn_Jatin model on every OSDR sample."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pearson_rows(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    x = truth - truth.mean(axis=1, keepdims=True)
    y = prediction - prediction.mean(axis=1, keepdims=True)
    denominator = np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1))
    return np.divide(
        (x * y).sum(axis=1),
        denominator,
        out=np.full(len(truth), np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--peft-script", type=Path)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    archive = np.load(args.matrix, allow_pickle=False)
    values = archive["X"].astype(np.float32)
    genes = archive["genes"].astype(str)
    metadata = pd.read_csv(args.metadata)
    if len(values) != len(metadata):
        raise RuntimeError("Matrix and metadata sample counts differ")

    peft_module = load_module(
        "osdr_peft_runtime",
        args.peft_script
        or args.project_root / "protocols/osdr_peft/osdr_peft_finetune.py",
    )
    common = peft_module.load_common_module(args.project_root)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_class = common.load_expression_performer(args.project_root)
    peft = checkpoint.get("peft", {})
    method = peft.get("method")
    build_checkpoint = checkpoint
    if method == "lora":
        source_checkpoint = peft.get("source_checkpoint")
        if not source_checkpoint:
            raise RuntimeError("LoRA checkpoint does not record its source checkpoint")
        build_checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    model = common.build_model(
        model_class, build_checkpoint, len(genes), torch.device("cuda")
    )
    if method == "lora":
        runtime_args = SimpleNamespace(
            method="lora",
            unfreeze_last_n=int(peft.get("unfreeze_last_n", 0)),
            lora_rank=int(peft.get("lora_rank", 4)),
            lora_alpha=float(peft.get("lora_alpha", 8.0)),
            lora_dropout=float(peft.get("lora_dropout", 0.0)),
            lora_target=str(peft.get("lora_target", "attn")),
            lora_layer_scope=str(peft.get("lora_layer_scope", "all")),
        )
        peft_module.configure_trainability(model, runtime_args)
        model.to("cuda")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    rng = np.random.default_rng(args.seed)
    masked_indices = np.sort(
        rng.choice(len(genes), size=round(len(genes) * args.mask_ratio), replace=False)
    )
    np.save(args.output_dir / "masked_gene_indices.npy", masked_indices)
    pd.DataFrame({"gene": genes[masked_indices]}).to_csv(
        args.output_dir / "masked_genes.csv", index=False
    )
    prediction = np.empty((len(values), len(masked_indices)), dtype=np.float32)
    mask_token = float(checkpoint["config"].get("mask_token", -10))
    started = time.time()
    for start in range(0, len(values), args.batch_size):
        stop = min(start + args.batch_size, len(values))
        tensor = torch.from_numpy(values[start:stop]).to("cuda", non_blocking=True)
        tensor[:, masked_indices] = mask_token
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            output = model(tensor)
        prediction[start:stop] = output[:, masked_indices].float().cpu().numpy()
        if stop == len(values) or stop % 160 == 0:
            progress = {
                "samples_complete": stop,
                "samples_total": len(values),
                "elapsed_seconds": time.time() - started,
            }
            (args.output_dir / "progress.json").write_text(
                json.dumps(progress, indent=2) + "\n"
            )
            print(json.dumps(progress), flush=True)

    truth = values[:, masked_indices]
    errors = prediction - truth
    sample_metrics = metadata.copy()
    sample_metrics["masked_mse"] = np.mean(errors * errors, axis=1)
    sample_metrics["masked_mae"] = np.mean(np.abs(errors), axis=1)
    sample_metrics["masked_pearson"] = pearson_rows(truth, prediction)
    sample_metrics.to_csv(args.output_dir / "sample_metrics.csv", index=False)
    np.savez_compressed(
        args.output_dir / "masked_predictions.npz",
        prediction=prediction,
        truth=truth,
        masked_indices=masked_indices,
        genes=genes,
    )
    summary = {
        "samples": len(values),
        "genes": len(genes),
        "masked_genes": len(masked_indices),
        "mask_ratio": args.mask_ratio,
        "checkpoint": str(args.checkpoint),
        "masked_mse": float(np.mean(errors * errors)),
        "masked_mae": float(np.mean(np.abs(errors))),
        "median_sample_pearson": float(np.nanmedian(sample_metrics["masked_pearson"])),
        "mean_sample_pearson": float(np.nanmean(sample_metrics["masked_pearson"])),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
