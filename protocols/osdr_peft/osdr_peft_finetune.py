#!/usr/bin/env python3
"""Parameter-efficient OSDR fine-tuning for Txn_Jatin."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.dropout(x) @ self.lora_a.t()
        update = update @ self.lora_b.t()
        return self.base(x) + update * self.scaling


def load_common_module(project_root: Path):
    path = project_root / "Txn_Jatin" / "osdr_finetune" / "finetune_bridge_osdr.py"
    spec = importlib.util.spec_from_file_location("osdr_finetune_common", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shared-data-dir", type=Path, required=True)
    parser.add_argument("--reference-embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--method",
        choices=("head_only", "bitfit_norm", "last_n", "lora"),
        required=True,
    )
    parser.add_argument("--unfreeze-last-n", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target", choices=("attn", "attn_ffn"), default="attn")
    parser.add_argument("--lora-layer-scope", default="all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def load_shared_data(shared_data_dir: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    with np.load(shared_data_dir / "prepared_osdr_bridge_input.npz", allow_pickle=False) as payload:
        X = payload["X"].astype(np.float32)
        genes = payload["genes"].astype(str)
    metadata = pd.read_csv(shared_data_dir / "prepared_osdr_metadata.csv")
    split = pd.read_csv(shared_data_dir / "osdr_group_split.csv")
    if not np.array_equal(metadata["sample_id"].astype(str), split["sample_id"].astype(str)):
        raise RuntimeError("shared OSDR metadata and split order differ")
    metadata = metadata.copy()
    metadata["split"] = split["split"].astype(str)
    return X, metadata, genes


def freeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def parse_lora_layers(scope: str, total_layers: int) -> list[int]:
    value = str(scope).strip().lower()
    if value == "all":
        return list(range(total_layers))
    if value.startswith("last"):
        count = int(value.replace("last", ""))
        return list(range(max(0, total_layers - count), total_layers))
    raise ValueError(f"unsupported lora layer scope: {scope}")


def replace_with_lora(module: nn.Module, attribute: str, rank: int, alpha: float, dropout: float) -> None:
    base = getattr(module, attribute)
    if not isinstance(base, nn.Linear):
        raise TypeError(f"{attribute} is not nn.Linear")
    setattr(module, attribute, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))


def configure_trainability(model: nn.Module, args: argparse.Namespace) -> dict:
    freeze_all(model)
    total_layers = len(model.layers)
    details: dict[str, object] = {"method": args.method}

    if args.method == "head_only":
        for parameter in model.output_map.parameters():
            parameter.requires_grad = True

    elif args.method == "bitfit_norm":
        for name, parameter in model.named_parameters():
            if name.endswith(".bias") or ".norm" in name or name.startswith("output_map."):
                parameter.requires_grad = True

    elif args.method == "last_n":
        if args.unfreeze_last_n <= 0 or args.unfreeze_last_n >= total_layers:
            raise ValueError("last_n must unfreeze between 1 and total_layers - 1")
        start = total_layers - args.unfreeze_last_n
        for layer_index in range(start, total_layers):
            for parameter in model.layers[layer_index].parameters():
                parameter.requires_grad = True
        for parameter in model.output_map.parameters():
            parameter.requires_grad = True
        details.update({"unfreeze_last_n": args.unfreeze_last_n, "first_unfrozen_layer": start})

    elif args.method == "lora":
        layers = parse_lora_layers(args.lora_layer_scope, total_layers)
        for layer_index in layers:
            layer = model.layers[layer_index]
            for attribute in ("q_proj", "k_proj", "v_proj", "out_proj"):
                replace_with_lora(layer, attribute, args.lora_rank, args.lora_alpha, args.lora_dropout)
            if args.lora_target == "attn_ffn":
                layer.ffn[0] = LoRALinear(layer.ffn[0], args.lora_rank, args.lora_alpha, args.lora_dropout)
                layer.ffn[2] = LoRALinear(layer.ffn[2], args.lora_rank, args.lora_alpha, args.lora_dropout)
        for parameter in model.output_map.parameters():
            parameter.requires_grad = True
        details.update(
            {
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_target": args.lora_target,
                "lora_layer_scope": args.lora_layer_scope,
                "lora_layers": layers,
            }
        )

    trainable_components: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            component = name.split(".", 1)[0]
            if component == "layers":
                parts = name.split(".")
                component = ".".join(parts[:2])
            trainable_components[component] = trainable_components.get(component, 0) + parameter.numel()

    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    total = int(sum(p.numel() for p in model.parameters()))
    details.update(
        {
            "trainable_components": trainable_components,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "total_parameters": total,
            "trainable_fraction": trainable / max(1, total),
        }
    )
    if trainable <= 0:
        raise RuntimeError("no trainable parameters configured")
    return details


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.shared_data_dir = args.shared_data_dir.resolve()
    args.reference_embedding_dir = args.reference_embedding_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    common = load_common_module(args.project_root)
    X, metadata, cached_genes = load_shared_data(args.shared_data_dir)
    canonical_genes = common.read_canonical_genes(args.checkpoint)
    if cached_genes.tolist() != canonical_genes:
        raise RuntimeError("prepared OSDR matrix does not match checkpoint gene order")

    train_idx = np.flatnonzero(metadata["split"].eq("train").to_numpy())
    val_idx = np.flatnonzero(metadata["split"].eq("validation").to_numpy())
    test_idx = np.flatnonzero(metadata["split"].eq("test").to_numpy())
    development_idx = np.flatnonzero(metadata["split"].isin(["train", "validation"]).to_numpy())

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ExpressionPerformer = common.load_expression_performer(args.project_root)
    model = common.build_model(ExpressionPerformer, checkpoint, len(canonical_genes), device)
    trainability = configure_trainability(model, args)
    # LoRA modules are inserted after the base model has moved to CUDA; move the
    # replaced modules and their new adapter parameters onto the active device.
    model.to(device)
    (args.output_dir / "trainability.json").write_text(json.dumps(trainability, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"model": args.model_name, **trainability}, indent=2, sort_keys=True), flush=True)

    mask_token = float(checkpoint["config"].get("mask_token", -10))
    baseline_test_mean, baseline_test_sd = common.evaluate_reconstruction(
        model, X[test_idx], args.batch_size, args.mask_ratio, mask_token, device, args.seed + 10000
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[train_idx])),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(1, int(total_steps * 0.05))

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    best_path = args.output_dir / "best.pt"
    best_val = float("inf")
    patience = 0
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        generator = torch.Generator(device=device).manual_seed(args.seed + epoch)
        total_loss = 0.0
        samples_seen = 0
        for batch_index, (xb,) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = common.masked_loss(model, xb, args.mask_ratio, mask_token, generator)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item()) * len(xb)
            samples_seen += len(xb)
            if batch_index % 50 == 0:
                print(
                    f"[TRAIN {args.model_name}] epoch={epoch} batch={batch_index}/{len(train_loader)} "
                    f"loss={total_loss/samples_seen:.6f}",
                    flush=True,
                )
        train_loss = total_loss / max(samples_seen, 1)
        val_loss, val_sd = common.evaluate_reconstruction(
            model, X[val_idx], args.batch_size, args.mask_ratio, mask_token, device, args.seed + 20000, repeats=2
        )
        row = {
            "epoch": epoch,
            "train_masked_mse": train_loss,
            "validation_masked_mse": val_loss,
            "validation_sd": val_sd,
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
        print("[EPOCH] " + json.dumps(row), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(
                {
                    "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "config": checkpoint["config"],
                    "val_loss": val_loss,
                    "epoch": epoch,
                    "peft": {
                        **trainability,
                        "source_checkpoint": str(args.checkpoint),
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "mask_ratio": args.mask_ratio,
                        "seed": args.seed,
                        "split": "shared accession-grouped OSDR split",
                    },
                },
                best_path,
            )
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[EARLY STOP] patience={args.patience}", flush=True)
                break

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    test_mean, test_sd = common.evaluate_reconstruction(
        model, X[test_idx], args.batch_size, args.mask_ratio, mask_token, device, args.seed + 10000
    )
    contextual = common.extract_contextual_gene_table(model, X[development_idx], args.context_batch_size, device)
    master_genes, aligned = common.align_to_reference(contextual, canonical_genes, args.reference_embedding_dir)
    provenance = {
        "name": args.model_name,
        "source_checkpoint": str(args.checkpoint),
        "fine_tuned_checkpoint": str(best_path),
        "dynamic_embedding": True,
        "context_dataset": "shared OSDR development partition only",
        "context_samples": int(len(development_idx)),
        "held_out_test_samples": int(len(test_idx)),
        "gene_embedding_mode": "contextual mean",
        "normalization": "log1p_cpm",
        "shape": list(aligned.shape),
        "embedding_sha256": sha256_array(aligned),
        "best_validation_loss": float(best["val_loss"]),
        "best_epoch": int(best["epoch"]),
        **trainability,
    }
    embedding_path = args.output_dir / f"{args.model_name}__gene_benchmark__symbol.npz"
    np.savez_compressed(
        embedding_path,
        keys=np.asarray(master_genes, dtype=object),
        emb=aligned,
        is_fallback=np.asarray(False),
        provenance=np.asarray([json.dumps(provenance, sort_keys=True)], dtype=object),
    )
    (args.output_dir / "embedding_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    development_labels = metadata.iloc[development_idx]["label"].to_numpy()
    test_labels = metadata.iloc[test_idx]["label"].to_numpy()
    development_embedding = common.extract_sample_embeddings(model, X[development_idx], args.context_batch_size, device)
    test_embedding = common.extract_sample_embeddings(model, X[test_idx], args.context_batch_size, device)
    f1, auroc = common.classification_metrics(development_embedding, development_labels, test_embedding, test_labels)
    metrics = pd.DataFrame(
        [
            {"model": args.model_name, "evaluation": "masked_reconstruction", "metric": "MSE", "value": test_mean, "sd": test_sd, "test_samples": len(test_idx)},
            {"model": args.model_name, "evaluation": "flight_vs_ground", "metric": "F1_macro", "value": f1, "sd": np.nan, "test_samples": len(test_idx)},
            {"model": args.model_name, "evaluation": "flight_vs_ground", "metric": "AUROC", "value": auroc, "sd": np.nan, "test_samples": len(test_idx)},
            {"model": "Txn_Jatin_original", "evaluation": "masked_reconstruction", "metric": "MSE", "value": baseline_test_mean, "sd": baseline_test_sd, "test_samples": len(test_idx)},
        ]
    )
    metrics.to_csv(args.output_dir / "heldout_test_metrics.csv", index=False)
    (args.output_dir / "run.status").write_text("0\n")
    print(metrics.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
