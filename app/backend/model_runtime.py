#!/usr/bin/env python3
"""Lazy, one-model-at-a-time GPU runtime for validated expression decoders."""

from __future__ import annotations

import csv
import gc
import importlib.util
import json
import math
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    from .downstream_analysis import build_live_analysis
except ImportError:
    from downstream_analysis import build_live_analysis


MASK_TOKEN = -10.0
MAX_REQUEST_GENES = 20010
MAX_REQUEST_SAMPLES = 512
INFERENCE_BATCH_SAMPLES = 32
EMBEDDING_BATCH_SAMPLES = 8
BULK_CONFIGS = {
    "BulkFormer_37M": {"dim": 128, "p_repeat": 1},
    "BulkFormer_50M": {"dim": 256, "p_repeat": 2},
    "BulkFormer_93M": {"dim": 512, "p_repeat": 6},
    "BulkFormer_127M": {"dim": 640, "p_repeat": 8},
    "BulkFormer_147M": {"dim": 640, "p_repeat": 12},
}
ALIASES = {
    "BulkFormer-37M": "BulkFormer_37M",
    "BulkFormer-50M": "BulkFormer_50M",
    "BulkFormer-93M": "BulkFormer_93M",
    "BulkFormer-127M": "BulkFormer_127M",
    "BulkFormer-147M": "BulkFormer_147M",
    "ESM2": "ESM2_PCA512_prior",
}


class RequestError(ValueError):
    """A client request is malformed."""


class UnsupportedModelError(ValueError):
    """The selected model has no validated expression decoder."""


class LoRALinear(nn.Module):
    """The exact adapter module used by the OSDR PEFT experiment."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.dropout(value) @ self.lora_a.t()
        update = update @ self.lora_b.t()
        return self.base(value) + update * self.scaling


def read_genes(path: str) -> list[str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise RuntimeError(f"Empty gene file: {path}")
    header = [cell.strip().lower() for cell in rows[0]]
    index = next(
        (i for i, name in enumerate(header) if name in {"gene", "symbol", "gene_symbol"}),
        0,
    )
    values = rows[1:] if header[index] in {"gene", "symbol", "gene_symbol"} else rows
    return [row[index].strip().upper() for row in values if len(row) > index and row[index].strip()]


def import_expression_performer(path: str):
    spec = importlib.util.spec_from_file_location("studio_train_flash", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model definition: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ExpressionPerformer


class ModelRuntime:
    """Loads one decoder on demand and serializes GPU access."""

    def __init__(self, config_path: Path, public_models: list[dict[str, Any]]):
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        package_path = self.config.get("python_packages")
        if package_path and package_path not in sys.path:
            sys.path.insert(0, package_path)
        self.public_models = {row["id"]: row for row in public_models}
        self.lock = threading.Lock()
        self.model = None
        self.model_name: str | None = None
        self.model_genes: list[str] = []
        self.gene_index: dict[str, int] = {}
        self.device = torch.device(self.config.get("device", "cuda"))
        self.bulk_graph = None
        self.bulk_esm2 = None

    def normalize_name(self, name: str) -> str:
        return ALIASES.get(name, name)

    def unload(self) -> None:
        self.model = None
        self.model_name = None
        self.model_genes = []
        self.gene_index = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_bridge_family(self, name: str):
        model_config = self.config["models"][name]
        payload = torch.load(model_config["checkpoint"], map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload.get("model", payload))
        config = payload["config"]
        model_class = import_expression_performer(self.config["train_flash"])
        count = int(state["gene_embedding.weight"].shape[0])
        model = model_class(
            num_genes=count,
            hidden_dim=int(config["hidden_dim"]),
            n_heads=int(config["num_heads"]),
            n_layers=int(config["num_layers"]),
            ffn_dim=int(config["ffn_dim"]),
            ree_base=float(config["ree_base"]),
            mask_token_id=float(config.get("mask_token", MASK_TOKEN)),
            feature_type=config.get("feature_type", "sqr"),
            compute_type=config.get("compute_type", "iter"),
            include_species_embedding=bool(config.get("include_species_embedding", False)),
            num_species=int(
                config.get("architecture", {}).get(
                    "num_species", config.get("num_species", 2)
                )
            ),
        )
        peft = payload.get("peft", {})
        if peft.get("method") == "lora":
            for layer_index in peft["lora_layers"]:
                layer = model.layers[int(layer_index)]
                for attribute in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    base = getattr(layer, attribute)
                    setattr(
                        layer,
                        attribute,
                        LoRALinear(
                            base,
                            peft["lora_rank"],
                            peft["lora_alpha"],
                            peft.get("lora_dropout", 0.0),
                        ),
                    )
                if peft.get("lora_target") == "attn_ffn":
                    layer.ffn[0] = LoRALinear(
                        layer.ffn[0],
                        peft["lora_rank"],
                        peft["lora_alpha"],
                        peft.get("lora_dropout", 0.0),
                    )
                    layer.ffn[2] = LoRALinear(
                        layer.ffn[2],
                        peft["lora_rank"],
                        peft["lora_alpha"],
                        peft.get("lora_dropout", 0.0),
                    )
        model.load_state_dict(state, strict=False)
        genes = read_genes(model_config["genes"])
        if len(genes) != count:
            raise RuntimeError(f"{name}: gene file has {len(genes)} rows; checkpoint has {count}")
        return model.to(self.device).eval(), genes

    def _load_bulkformer(self, name: str):
        root = self.config["bulkformer_root"]
        if root not in sys.path:
            sys.path.insert(0, root)
        from torch_sparse import SparseTensor
        from utils.BulkFormer import BulkFormer

        if self.bulk_graph is None:
            edges = torch.load(
                self.config["bulkformer_graph"], map_location="cpu", weights_only=False
            )
            weights = torch.load(
                self.config["bulkformer_graph_weights"],
                map_location="cpu",
                weights_only=False,
            )
            self.bulk_graph = SparseTensor(
                row=edges[1], col=edges[0], value=weights
            ).t().to(self.device)
        if self.bulk_esm2 is None:
            self.bulk_esm2 = torch.load(
                self.config["bulkformer_esm2"], map_location="cpu", weights_only=False
            )
        architecture = BULK_CONFIGS[name]
        model = BulkFormer(
            dim=architecture["dim"],
            graph=self.bulk_graph,
            gene_emb=self.bulk_esm2,
            gene_length=20010,
            bins=0,
            gb_repeat=1,
            p_repeat=architecture["p_repeat"],
            bin_head=12,
            full_head=8,
        )
        raw_state = torch.load(
            self.config["models"][name]["checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        state = OrderedDict(
            (key[7:] if key.startswith("module.") else key, value)
            for key, value in raw_state.items()
        )
        model.load_state_dict(state, strict=True)
        genes = read_genes(self.config["bulkformer_genes"])
        if len(genes) != 20010:
            raise RuntimeError(f"BulkFormer gene file has {len(genes)} rows")
        return model.to(self.device).eval(), genes

    def ensure_loaded(self, raw_name: str) -> str:
        name = self.normalize_name(raw_name)
        public = self.public_models.get(name)
        if public is None:
            raise RequestError(f"Unknown model: {raw_name}")
        if not public["imputation_supported"]:
            raise UnsupportedModelError(
                f"{public['label']} is embedding-only and has no validated expression decoder; "
                "the benchmark protocol reports imputation as NaN."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this deployment")
        if self.model_name == name and self.model is not None:
            return name
        self.unload()
        if name.startswith("BulkFormer_"):
            self.model, self.model_genes = self._load_bulkformer(name)
        else:
            self.model, self.model_genes = self._load_bridge_family(name)
        self.model_name = name
        self.gene_index = {gene: i for i, gene in enumerate(self.model_genes)}
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return name

    @staticmethod
    def _validate_payload(
        payload: dict[str, Any],
        *,
        require_missing: bool = True,
        allow_negative: bool = False,
    ) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
        genes = [str(value).strip().upper() for value in payload.get("genes", [])]
        samples = [str(value) for value in payload.get("samples", [])]
        matrix = payload.get("matrix")
        missing = payload.get("missing")
        if not genes or not samples:
            raise RequestError("genes and samples must be non-empty")
        if len(genes) > MAX_REQUEST_GENES or len(samples) > MAX_REQUEST_SAMPLES:
            raise RequestError(
                f"request limit is {MAX_REQUEST_GENES:,} genes by "
                f"{MAX_REQUEST_SAMPLES:,} samples; inference runs in "
                f"{INFERENCE_BATCH_SAMPLES}-sample GPU batches"
            )
        if len(set(genes)) != len(genes):
            raise RequestError("gene symbols must be unique")
        if not isinstance(matrix, list) or len(matrix) != len(genes):
            raise RequestError("matrix must have one row per gene")
        if not isinstance(missing, list) or len(missing) != len(genes):
            raise RequestError("missing must have one row per gene")
        values = np.zeros((len(genes), len(samples)), dtype=np.float32)
        mask = np.zeros_like(values, dtype=bool)
        for g, (row, mrow) in enumerate(zip(matrix, missing)):
            if len(row) != len(samples) or len(mrow) != len(samples):
                raise RequestError(f"row {g} does not match sample count")
            for s, (value, is_missing) in enumerate(zip(row, mrow)):
                absent = bool(is_missing) or value is None
                mask[g, s] = absent
                if absent:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError) as error:
                    raise RequestError(f"matrix[{g}][{s}] is not numeric") from error
                if not math.isfinite(number) or (number < 0 and not allow_negative):
                    qualifier = "finite" if allow_negative else "finite and non-negative"
                    raise RequestError(f"observed expression values must be {qualifier}")
                values[g, s] = number
        if require_missing and not mask.any():
            raise RequestError("the matrix contains no missing values to impute")
        return genes, samples, values, mask

    @staticmethod
    def _resolve_scale(payload: dict[str, Any], values: np.ndarray, mask: np.ndarray) -> str:
        requested = str(payload.get("input_scale", "auto")).lower()
        if requested in {"raw", "tpm", "cpm", "counts"}:
            return "raw"
        if requested in {"log1p", "log1p_tpm", "log1p_cpm"}:
            return "log1p"
        observed = values[~mask]
        return "raw" if observed.size and float(np.percentile(observed, 99)) > 30 else "log1p"

    def _predict_aligned(self, name: str, aligned: np.ndarray) -> np.ndarray:
        prediction = np.empty_like(aligned, dtype=np.float32)
        for start in range(0, len(aligned), INFERENCE_BATCH_SAMPLES):
            end = min(start + INFERENCE_BATCH_SAMPLES, len(aligned))
            tensor = torch.from_numpy(aligned[start:end]).to(
                self.device, non_blocking=True
            )
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda"
                and not name.startswith("BulkFormer_"),
            ):
                if name.startswith("BulkFormer_"):
                    chunk = self.model(tensor, mask_prob=0.15, output_expr=True)
                else:
                    chunk = self.model(tensor)
            prediction[start:end] = chunk.float().cpu().numpy()
            del tensor, chunk
        return prediction

    def _embed_aligned(self, aligned: np.ndarray) -> np.ndarray:
        sample_embeddings = []
        for start in range(0, len(aligned), EMBEDDING_BATCH_SAMPLES):
            end = min(start + EMBEDDING_BATCH_SAMPLES, len(aligned))
            tensor = torch.from_numpy(aligned[start:end]).to(
                self.device, non_blocking=True
            )
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                hidden = self.model.encode_hidden(tensor)
                chunk = hidden.float().mean(dim=1)
            sample_embeddings.append(chunk.cpu().numpy())
            del tensor, hidden, chunk
        return np.concatenate(sample_embeddings, axis=0)

    def impute(self, payload: dict[str, Any]) -> dict[str, Any]:
        genes, samples, values, missing = self._validate_payload(payload)
        with self.lock:
            name = self.ensure_loaded(str(payload.get("model", "")))
            scale = self._resolve_scale(payload, values, missing)
            model_values = np.log1p(values) if scale == "raw" else values.copy()
            aligned = np.full(
                (len(samples), len(self.model_genes)), MASK_TOKEN, dtype=np.float32
            )
            matched = []
            unresolved = []
            for request_index, gene in enumerate(genes):
                model_index = self.gene_index.get(gene)
                if model_index is None:
                    unresolved.append(gene)
                    continue
                matched.append((request_index, model_index))
                observed = ~missing[request_index]
                aligned[observed, model_index] = model_values[request_index, observed]
            if not matched:
                raise RequestError(
                    "none of the submitted genes are in the selected model vocabulary"
                )

            prediction = self._predict_aligned(name, aligned)
            completed = values.astype(float).tolist()
            confidence = np.ones_like(values, dtype=np.float32).tolist()
            for request_index, model_index in matched:
                for sample_index in np.flatnonzero(missing[request_index]):
                    value = float(prediction[sample_index, model_index])
                    if scale == "raw":
                        value = max(0.0, math.expm1(value))
                    completed[request_index][sample_index] = value
                    confidence[request_index][sample_index] = 0.5
            for request_index, gene in enumerate(genes):
                if gene in unresolved:
                    for sample_index in np.flatnonzero(missing[request_index]):
                        completed[request_index][sample_index] = None
                        confidence[request_index][sample_index] = 0.0

            coverage = len(matched) / len(self.model_genes)
            warnings = []
            if coverage < 0.5:
                warnings.append(
                    f"Only {len(matched):,}/{len(self.model_genes):,} model genes were supplied. "
                    "This sparse-panel request is outside the 15% masking training regime."
                )
            if unresolved:
                warnings.append(
                    f"{len(unresolved)} submitted genes are absent from the selected vocabulary."
                )
            return {
                "model": name,
                "imputed": completed,
                "confidence": confidence,
                "confidence_kind": (
                    "placeholder; checkpoint has no calibrated predictive uncertainty"
                ),
                "input_scale": scale,
                "matched_genes": len(matched),
                "model_gene_count": len(self.model_genes),
                "coverage_fraction": coverage,
                "unresolved_genes": unresolved,
                "warnings": warnings,
                "device": torch.cuda.get_device_name(0),
                "inference_batch_samples": INFERENCE_BATCH_SAMPLES,
            }

    def analyze_downstream(self, payload: dict[str, Any]) -> dict[str, Any]:
        genes, samples, values, missing = self._validate_payload(
            payload,
            require_missing=False,
            allow_negative=True,
        )
        if len(samples) < 2:
            raise RequestError("downstream analysis requires at least two samples")
        groups = [str(value).strip() or "Unlabeled" for value in payload.get("groups", [])]
        if not groups:
            groups = ["Unlabeled"] * len(samples)
        if len(groups) != len(samples):
            raise RequestError("groups must have one value per sample")

        raw_name = str(payload.get("model", ""))
        name = self.normalize_name(raw_name)
        public = self.public_models.get(name)
        if public is None:
            raise RequestError(f"Unknown model: {raw_name}")
        requested_scale = str(payload.get("input_scale", "auto")).lower()
        if np.any(values[~missing] < 0):
            scale = "log1p"
        elif requested_scale in {"raw", "tpm", "cpm", "counts"}:
            scale = "raw"
        elif requested_scale in {"log1p", "log1p_tpm", "log1p_cpm"}:
            scale = "log1p"
        else:
            scale = self._resolve_scale(payload, values, missing)
        model_values = np.log1p(np.maximum(values, 0)) if scale == "raw" else values.copy()

        warnings: list[str] = []
        model_gene_count: int | None = None
        matched_genes = len(genes)
        can_encode = bool(public["imputation_supported"]) and not name.startswith(
            "BulkFormer_"
        )
        if can_encode:
            with self.lock:
                name = self.ensure_loaded(name)
                model_gene_count = len(self.model_genes)
                aligned = np.zeros(
                    (len(samples), model_gene_count), dtype=np.float32
                )
                matched_genes = 0
                for request_index, gene in enumerate(genes):
                    model_index = self.gene_index.get(gene)
                    if model_index is None:
                        continue
                    matched_genes += 1
                    aligned[:, model_index] = model_values[request_index]
                    aligned[missing[request_index], model_index] = MASK_TOKEN
                if not matched_genes:
                    raise RequestError(
                        "none of the submitted genes are in the selected model vocabulary"
                    )
                embeddings = self._embed_aligned(aligned)
            embedding_mode = "mean contextual sample embedding (512-d)"
        else:
            profile = model_values.T.astype(np.float32, copy=True)
            if missing.any():
                profile[missing.T] = 0
            means = profile.mean(axis=0, keepdims=True)
            standard_deviation = profile.std(axis=0, keepdims=True)
            informative = standard_deviation > 1e-6
            profile[:, informative[0]] = (
                profile[:, informative[0]] - means[:, informative[0]]
            ) / standard_deviation[:, informative[0]]
            profile[:, ~informative[0]] = 0
            embeddings = profile
            embedding_mode = "standardized log-expression profile"
            warnings.append(
                f"{public['label']} has no validated contextual sample-embedding "
                "interface in this deployment; retrieval and maps use expression "
                "profiles rather than model embeddings."
            )

        result = build_live_analysis(
            model=name,
            embedding_mode=embedding_mode,
            genes=genes,
            samples=samples,
            groups=groups,
            expression=values,
            embeddings=embeddings,
            matched_genes=matched_genes,
            model_gene_count=model_gene_count,
            initial_warnings=warnings,
        )
        result["input_scale"] = scale
        result["device"] = (
            torch.cuda.get_device_name(0)
            if can_encode and torch.cuda.is_available()
            else "CPU"
        )
        result["embedding_batch_samples"] = (
            EMBEDDING_BATCH_SAMPLES if can_encode else None
        )
        return result
