#!/usr/bin/env python3
"""Run whole-gene masked expression inference for one native decoder."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from masked_benchmark_common import (
    MASK_TOKEN,
    align_matrix,
    load_mask,
    load_source_matrix,
    read_genes,
    read_protocol,
    write_json,
)


BULK_CONFIGS = {
    "BulkFormer_37M": {"dim": 128, "p_repeat": 1},
    "BulkFormer_50M": {"dim": 256, "p_repeat": 2},
    "BulkFormer_93M": {"dim": 512, "p_repeat": 6},
    "BulkFormer_127M": {"dim": 640, "p_repeat": 8},
    "BulkFormer_147M": {"dim": 640, "p_repeat": 12},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def import_expression_performer(path: str):
    spec = importlib.util.spec_from_file_location("masked_benchmark_train_flash", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model definition from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ExpressionPerformer


def load_bridge_family(protocol: dict, model_name: str, device: torch.device):
    checkpoint_path = protocol["models"][model_name]["checkpoint"]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("model", payload))
    config = payload["config"]
    model_class = import_expression_performer(protocol["paths"]["train_flash"])
    num_genes = int(state["gene_embedding.weight"].shape[0])
    model = model_class(
        num_genes=num_genes,
        hidden_dim=int(config["hidden_dim"]),
        n_heads=int(config["num_heads"]),
        n_layers=int(config["num_layers"]),
        ffn_dim=int(config["ffn_dim"]),
        ree_base=float(config["ree_base"]),
        mask_token_id=float(config.get("mask_token", MASK_TOKEN)),
        feature_type=config.get("feature_type", "sqr"),
        compute_type=config.get("compute_type", "iter"),
        include_species_embedding=bool(config.get("include_species_embedding", False)),
        num_species=int(config.get("architecture", {}).get("num_species", config.get("num_species", 2))),
    )
    model.load_state_dict(state, strict=False)
    gene_path = (
        protocol["paths"]["txn_genes"]
        if model_name == "Txn_Jatin"
        else protocol["paths"]["bridge_genes"]
    )
    genes = read_genes(gene_path)
    if len(genes) != num_genes:
        raise RuntimeError(f"{model_name} genes={len(genes)} checkpoint={num_genes}")
    return model.to(device).eval(), genes


def load_bulkformer(protocol: dict, model_name: str, device: torch.device):
    root = protocol["paths"]["bulkformer_root"]
    if root not in sys.path:
        sys.path.insert(0, root)
    from torch_sparse import SparseTensor
    from utils.BulkFormer import BulkFormer

    graph_edges = torch.load(protocol["paths"]["graph"], map_location="cpu", weights_only=False)
    graph_weights = torch.load(
        protocol["paths"]["graph_weights"], map_location="cpu", weights_only=False
    )
    graph = SparseTensor(
        row=graph_edges[1], col=graph_edges[0], value=graph_weights
    ).t().to(device)
    esm2 = torch.load(protocol["paths"]["esm2"], map_location="cpu", weights_only=False)
    config = BULK_CONFIGS[model_name]
    model = BulkFormer(
        dim=config["dim"],
        graph=graph,
        gene_emb=esm2,
        gene_length=20010,
        bins=0,
        gb_repeat=1,
        p_repeat=config["p_repeat"],
        bin_head=12,
        full_head=8,
    )
    raw_state = torch.load(
        protocol["models"][model_name]["checkpoint"], map_location="cpu", weights_only=False
    )
    state = OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in raw_state.items()
    )
    model.load_state_dict(state, strict=True)
    genes = read_genes(protocol["paths"]["bulk_genes"])
    if len(genes) != 20010:
        raise RuntimeError(f"BulkFormer gene vocabulary has {len(genes)} genes, expected 20010")
    return model.to(device).eval(), genes


def infer(
    model,
    model_name: str,
    values: np.ndarray,
    masked_indices: np.ndarray,
    batch_size: int,
    progress_path: Path,
) -> np.ndarray:
    output = np.empty((len(values), len(masked_indices)), dtype=np.float32)
    current_batch = max(1, batch_size)
    offset = 0
    started = time.time()
    while offset < len(values):
        stop = min(len(values), offset + current_batch)
        try:
            tensor = torch.from_numpy(values[offset:stop]).to("cuda", non_blocking=True)
            tensor[:, masked_indices] = MASK_TOKEN
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=not model_name.startswith("BulkFormer_"),
            ):
                if model_name.startswith("BulkFormer_"):
                    prediction = model(tensor, mask_prob=0.15, output_expr=True)
                else:
                    prediction = model(tensor)
            output[offset:stop] = prediction[:, masked_indices].float().cpu().numpy()
            offset = stop
            if offset == len(values) or offset % max(25, current_batch * 10) == 0:
                write_json(
                    progress_path,
                    {
                        "samples_complete": offset,
                        "samples_total": len(values),
                        "batch_size": current_batch,
                        "elapsed_seconds": time.time() - started,
                    },
                )
            del tensor, prediction
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_batch == 1:
                raise
            current_batch = max(1, current_batch // 2)
            print(f"CUDA OOM: reducing batch size to {current_batch}", flush=True)
    return output


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    protocol = read_protocol(run_dir)
    if not protocol["models"].get(args.model, {}).get("supported"):
        raise RuntimeError(f"{args.model} has no supported expression decoder")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for decoder inference")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    if args.model.startswith("BulkFormer_"):
        model, model_genes = load_bulkformer(protocol, args.model, device)
    else:
        model, model_genes = load_bridge_family(protocol, args.model, device)

    model_output = run_dir / "predictions" / args.model
    model_output.mkdir(parents=True, exist_ok=True)
    model_started = time.time()
    for dataset in ("TCGA", "OSDR"):
        source_values, source_genes, sample_ids = load_source_matrix(protocol, dataset)
        aligned, model_index = align_matrix(source_values, source_genes, model_genes)
        del source_values
        dataset_output = model_output / dataset
        dataset_output.mkdir(parents=True, exist_ok=True)
        np.save(dataset_output / "sample_ids.npy", sample_ids)
        for seed in protocol["masking"]["seeds"]:
            destination = dataset_output / f"seed_{seed}_predictions.npy"
            if destination.exists():
                print(f"Skipping existing {destination}", flush=True)
                continue
            masked_genes = load_mask(run_dir, seed)
            missing = [gene for gene in masked_genes if gene not in model_index]
            if missing:
                raise RuntimeError(
                    f"{args.model} is missing {len(missing)} masked shared-universe genes"
                )
            masked_indices = np.asarray([model_index[gene] for gene in masked_genes], dtype=np.int64)
            predictions = infer(
                model,
                args.model,
                aligned,
                masked_indices,
                args.batch_size,
                run_dir / "status" / f"{args.model}_{dataset}_{seed}.json",
            )
            np.save(destination, predictions)
            write_json(
                dataset_output / f"seed_{seed}_manifest.json",
                {
                    "model": args.model,
                    "dataset": dataset,
                    "seed": seed,
                    "samples": len(predictions),
                    "masked_genes": len(masked_genes),
                    "prediction_file": str(destination),
                    "elapsed_model_seconds": time.time() - model_started,
                },
            )
            print(f"Completed {args.model} {dataset} seed={seed}", flush=True)
        del aligned
    marker = run_dir / "status" / f"{args.model}.READY"
    marker.write_text("ready\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "model": args.model,
                "status": "ready_for_analysis",
                "elapsed_seconds": time.time() - model_started,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
