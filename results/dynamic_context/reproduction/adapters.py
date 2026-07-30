"""
mythos.adapters — uniform, model-agnostic embedding extractors.

Every model is wrapped behind the same interface so the benchmark code never
needs to know which model it is talking to:

    class ModelAdapter:
        name: str
        def gene_embeddings(self, genes) -> np.ndarray   # [G, D], NaN if uncovered
        def sample_embeddings(self, expr_df) -> np.ndarray | None  # [N, D]
        def covered_genes(self) -> set[str]
        def provenance(self) -> dict

Design rules (from the Zhong protocol and our own rigor checklist):
  * Frozen extraction only — eval() + torch.no_grad(); no fine-tuning.
  * CPU-friendly — device='cpu', threads pinned, small batches.
  * Full embedding dimensionality (no PCA before the probe).
  * Honesty — if a checkpoint/repo is missing the adapter logs a WARN and
    falls back to clearly-labelled RANDOM vectors (is_fallback=True), which are
    never cached and are flagged in every figure/table.  Sample-embedding
    extraction for the single-cell models is attempted via their real API and,
    if unavailable, returns None (documented skip) rather than a fabricated
    number.

BRIDGE is the only model whose weights live in this repo; the other three are
loaded from user-provided clones/checkpoints (env vars below).  See README.
"""
from __future__ import annotations

import os
import sys
import json
import importlib.util
import gc
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (WALTER_ROOT, ref_paths, load_cache, save_cache, CFG, rng,
                     SEED, EMB_CACHE)
from .data import load_canonical_genes


def _torch_device():
    """Runtime device for frozen extraction; defaults to CUDA when available."""
    try:
        import torch
        requested = os.environ.get("MYTHOS_DEVICE", "cuda").lower()
        if requested.startswith("cuda") and torch.cuda.is_available():
            return torch.device(requested)
        return torch.device("cpu")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class ModelAdapter:
    name = "base"
    dim = None

    def __init__(self, council=None):
        self.council = council
        self.is_fallback = False
        self._covered = set()
        self._prov = {"name": self.name}

    # ---- interface --------------------------------------------------------- #
    def gene_embeddings(self, genes):
        raise NotImplementedError

    def sample_embeddings(self, expr_df):
        return None

    def covered_genes(self):
        return self._covered

    def provenance(self):
        return self._prov

    # ---- helpers ----------------------------------------------------------- #
    def _fallback_gene(self, genes, dim, why=""):
        self.is_fallback = True
        if self.council:
            self.council.check("ReproEng", f"{self.name} gene extractor", False,
                               f"FALLBACK random vectors -- {why}", warn_only=True)
        seed = (hash((self.name, "gene")) & 0xFFFFFFFF)
        r = np.random.default_rng(seed)
        emb = r.standard_normal((len(genes), dim)).astype(np.float32)
        self._covered = set(str(g).upper() for g in genes)
        self.dim = dim
        return emb

    @staticmethod
    def _align(genes, key2row, table):
        """Build [G, D] aligned to `genes`; NaN rows for uncovered genes."""
        D = table.shape[1]
        out = np.full((len(genes), D), np.nan, dtype=np.float32)
        for i, g in enumerate(genes):
            j = key2row.get(str(g).upper())
            if j is not None and j < table.shape[0]:
                out[i] = table[j]
        return out


# --------------------------------------------------------------------------- #
# BRIDGE (ours) — ExpressionPerformer / SLiMPerformer
# --------------------------------------------------------------------------- #
class BridgeAdapter(ModelAdapter):
    """Loads the BRIDGE checkpoint exactly as RNA-Walter's notebooks do:
    payload['config'] for architecture, payload['model_state_dict'] for weights,
    the canonical 15,165-gene order for alignment.  Gene embeddings come from the
    zero-input hidden-state route; sample embeddings from
    extract_transcriptome_embeddings (mean/max/median pooling)."""
    name = "BRIDGE"

    def __init__(
        self,
        council=None,
        run=None,
        aggregate="all",
        name=None,
        gene_embedding_mode=None,
        sample_embedding_enabled=True,
        checkpoint=None,
    ):
        env_run = os.environ.get("BRIDGE_RUN") or os.environ.get("TXN_JATIN_RUN")
        env_name = os.environ.get("BRIDGE_MODEL_NAME") or os.environ.get("TXN_MODEL_NAME")
        self.name = name or env_name or self.name
        super().__init__(council)
        self.run = run or env_run or "20jo1hdd"
        self.checkpoint_override = (
            Path(checkpoint).expanduser() if checkpoint else None
        )
        self._run_explicit = bool(
            run
            or env_run
            or self.checkpoint_override
            or os.environ.get("BRIDGE_CKPT")
            or os.environ.get("TXN_JATIN_CKPT")
        )
        self.aggregate = aggregate
        self.sample_embedding_enabled = bool(sample_embedding_enabled)
        self.model = None
        self.device = None
        self.gene_embedding_mode = (
            gene_embedding_mode
            or os.environ.get("BRIDGE_GENE_EMBED_MODE", "zero")
        ).strip().lower()
        self.gene_order = load_canonical_genes(council)
        self.gene_index = {str(g).upper(): i for i, g in enumerate(self.gene_order)}
        self._gene_emb_cache = None
        self._checkpoint_gene_table = None
        self._checkpoint_contextual_table = None
        self.config = {}

    @staticmethod
    def _read_gene_order(path):
        try:
            df = pd.read_csv(path)
            for col in ("gene_symbol", "symbol", "gene", "genes"):
                if col in df.columns:
                    vals = df[col].dropna().astype(str).tolist()
                    if vals:
                        return vals
            if len(df.columns):
                vals = df.iloc[:, -1].dropna().astype(str).tolist()
                if vals:
                    return vals
        except Exception:
            return None
        return None

    # ---- model loading ----------------------------------------------------- #
    def _load_model(self):
        if self.model is not None:
            return True
        try:
            import torch
            if WALTER_ROOT and str(WALTER_ROOT) not in sys.path:
                sys.path.insert(0, str(WALTER_ROOT))
            ckpt = None
            ckpt_override = (
                self.checkpoint_override
                or os.environ.get("BRIDGE_CKPT")
                or os.environ.get("TXN_JATIN_CKPT")
            )
            run_dir = None
            if ckpt_override:
                ckpt = Path(ckpt_override).expanduser()
                run_dir = ckpt.parent
                if not ckpt.exists():
                    raise RuntimeError(f"{self.name} checkpoint missing: {ckpt}")
            ckpt_root = ref_paths()["checkpoints_performer"]
            if ckpt is None:
                run_dir = ckpt_root / self.run if ckpt_root else None
                if run_dir and run_dir.exists():
                    cands = (sorted(run_dir.glob("best_model.pt"))
                             or sorted(run_dir.glob("*best*.pt"))
                             or sorted(run_dir.glob("*.pt")))
                    ckpt = cands[-1] if cands else None
            if ckpt is None and WALTER_ROOT:
                flash_root = WALTER_ROOT / "flash_osdr_model"
                explicit_dir = flash_root / self.run
                flash_dirs = [explicit_dir] if explicit_dir.exists() else []
                if not self._run_explicit:
                    flash_dirs.extend(sorted(p for p in flash_root.glob("*") if p.is_dir()))
                seen = set()
                for cand_dir in flash_dirs:
                    if cand_dir in seen:
                        continue
                    seen.add(cand_dir)
                    cands = (sorted(cand_dir.glob("best_model.pt"))
                             or sorted(cand_dir.glob("*best*.pt"))
                             or sorted(cand_dir.glob("latest.pt"))
                             or sorted(cand_dir.glob("*.pt")))
                    if cands:
                        ckpt = cands[-1]
                        run_dir = cand_dir
                        break
            if ckpt is None:
                raise RuntimeError(f"no {self.name} weights found for run '{self.run}' in {run_dir}")
            self.device = _torch_device() or torch.device("cpu")
            payload = torch.load(ckpt, map_location="cpu")
            cfg = payload["config"]
            self.config = dict(cfg)
            if str(cfg.get("feature_type", "")).lower() == "flash":
                train_flash = WALTER_ROOT / "flash_osdr_model" / "train_flash.py"
                spec = importlib.util.spec_from_file_location("bridge_train_flash", train_flash)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                ExpressionPerformer = mod.ExpressionPerformer
            else:
                from train import ExpressionPerformer  # noqa
            sd = payload.get("model_state_dict", payload.get("model", payload))
            num_genes = int(sd["gene_embedding.weight"].shape[0])
            ckpt_gene_order = self._read_gene_order(ckpt.parent / "canonical_genes.csv")
            if ckpt_gene_order and len(ckpt_gene_order) == num_genes:
                self.gene_order = ckpt_gene_order
                self.gene_index = {str(g).upper(): i
                                   for i, g in enumerate(self.gene_order)}
            elif ckpt_gene_order and self.council:
                self.council.check(
                    "Methodologist",
                    f"{self.name} checkpoint canonical gene order",
                    False,
                    f"checkpoint canonical_genes={len(ckpt_gene_order)} ckpt={num_genes}; "
                    "falling back to reference token map",
                    warn_only=True)
            # rigor: assert dictionary matches the checkpoint (Walter bug C)
            if len(self.gene_order) != num_genes:
                if self.council:
                    self.council.check(
                        "Methodologist",
                        "BRIDGE gene-dictionary == checkpoint num_genes",
                        False, f"dict={len(self.gene_order)} ckpt={num_genes}; "
                        "using checkpoint order", warn_only=True)
                # fall back to a positional gene order of the right length
                if len(self.gene_order) > num_genes:
                    self.gene_order = self.gene_order[:num_genes]
                    self.gene_index = {str(g).upper(): i
                                       for i, g in enumerate(self.gene_order)}
            model = ExpressionPerformer(
                num_genes=num_genes,
                hidden_dim=cfg["hidden_dim"], n_heads=cfg["num_heads"],
                n_layers=cfg["num_layers"], ffn_dim=cfg["ffn_dim"],
                ree_base=cfg["ree_base"], mask_token_id=cfg.get("mask_token", -10),
                feature_type=cfg.get("feature_type", "sqr"),
                compute_type=cfg.get("compute_type", "iter"),
                include_species_embedding=cfg.get("include_species_embedding", False),
                num_species=cfg.get("architecture", {}).get(
                    "num_species", cfg.get("num_species", 2)
                ))
            model.load_state_dict(sd, strict=False)
            self.model = model.to(self.device).eval()
            checkpoint_table = payload.get("benchmark_gene_embedding")
            contextual_table = payload.get("contextual_gene_embedding")
            if checkpoint_table is not None and tuple(checkpoint_table.shape[:1]) == (num_genes,):
                self._checkpoint_gene_table = (
                    checkpoint_table.detach().float().cpu().numpy().astype(np.float32)
                )
            if contextual_table is not None and tuple(contextual_table.shape[:1]) == (num_genes,):
                self._checkpoint_contextual_table = (
                    contextual_table.detach().float().cpu().numpy().astype(np.float32)
                )
            self.dim = int(cfg["hidden_dim"])
            self._covered = set(self.gene_index)
            self._prov = dict(name=self.name, checkpoint=str(ckpt),
                              run=self.run, num_genes=num_genes,
                              hidden_dim=self.dim, num_layers=cfg["num_layers"],
                              normalization=cfg.get("normalization", "log1p_tpm"),
                              gene_namespace="HGNC symbol",
                              gene_embedding_mode=self.gene_embedding_mode,
                              device=str(self.device),
                              val_loss=float(payload.get("val_loss", np.nan)))
            if self.council:
                self.council.check("ReproEng", f"{self.name} model loaded", True,
                                   f"{ckpt.name}, dim={self.dim}, genes={num_genes}, device={self.device}")
            return True
        except Exception as e:
            self._prov["error"] = str(e)
            if self.council:
                self.council.check("ReproEng", f"{self.name} model loaded", False,
                                   str(e), warn_only=True)
            return False

    # ---- frozen routines --------------------------------------------------- #
    @staticmethod
    def _encode_hidden(model, x, species_ids=None):
        """Replicate ExpressionPerformer's forward up to the per-gene hidden
        states (no output_map).  x: [B, G] -> hidden [B, G, D]."""
        import torch
        if hasattr(model, "encode_hidden"):
            return model.encode_hidden(x, species_ids)
        _, num_genes = x.shape
        gene_ids = torch.arange(num_genes, device=x.device)
        gene_emb = model.gene_embedding(gene_ids)
        ree_emb = model.ree(x)
        hidden = gene_emb.unsqueeze(0) + ree_emb
        for layer in model.layers:
            if getattr(model, "use_flash", False):
                hidden = layer(hidden)
            else:
                rfs = layer.attention.sample_rfs(x.device)
                hidden = layer.full_forward(hidden, rfs)
        return hidden

    def _checkpoint_embedding_table(self):
        if self._checkpoint_gene_table is None:
            raise RuntimeError(
                f"{self.name} checkpoint does not contain benchmark_gene_embedding"
            )
        self._gene_emb_cache = self._checkpoint_gene_table
        return self._gene_emb_cache

    def _release_model_if_requested(self):
        if os.environ.get(
            "MYTHOS_RELEASE_BRIDGE_MODELS", "0"
        ).strip().lower() not in {"1", "true", "yes", "on"}:
            return
        self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _normalize_context_matrix(self, values, species):
        mode = str(self.config.get("normalization", "already_log1p_tpm")).lower()
        values = np.asarray(values, dtype=np.float32)
        if mode == "counts_to_log1p_tpm":
            data_dir = Path(self.config.get("data_dir", ""))
            lengths_path = data_dir / "gene_lengths.npz"
            if not lengths_path.exists():
                raise RuntimeError(
                    f"checkpoint normalization requires gene lengths: {lengths_path}"
                )
            with np.load(lengths_path, allow_pickle=False) as lengths_npz:
                length_genes = lengths_npz["genes"].astype(str).tolist()
                if length_genes != self.gene_order:
                    raise RuntimeError("context gene lengths do not match checkpoint gene order")
                lengths = lengths_npz["mouse" if species == "mouse" else "human"]
            rates = np.maximum(values, 0.0) / np.maximum(lengths[None, :] / 1000.0, 1e-6)
            tpm = rates / np.maximum(rates.sum(axis=1, keepdims=True), 1e-12) * 1e6
            return np.log1p(tpm).astype(np.float32, copy=False)
        if mode == "log1p_tpm":
            return np.log1p(np.maximum(values, 0.0)).astype(np.float32, copy=False)
        return values

    def _zero_input_gene_table(self):
        """[num_genes, D] gene-identity embeddings via the zero-TPM route."""
        if self._gene_emb_cache is not None:
            return self._gene_emb_cache
        import torch
        G = len(self.gene_order)
        zero = torch.zeros((1, G), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            table = self._encode_hidden(self.model, zero)[0].cpu().numpy()
        self._gene_emb_cache = table.astype(np.float32)
        return self._gene_emb_cache

    def _contextual_gene_table(self):
        """Mean per-gene hidden states over an unlabeled expression calibration set.

        This keeps the BRIDGE/Txn_Jatin model frozen but avoids the all-zero input
        route, which is off the pretraining distribution and weak for static gene
        benchmarks.
        """
        if self._checkpoint_contextual_table is not None:
            self._gene_emb_cache = self._checkpoint_contextual_table
            return self._gene_emb_cache
        cached = load_cache(self.name, "gene_contextual", "symbol")
        if cached:
            return cached[1]
        context_path = (os.environ.get("BRIDGE_CONTEXT_PARQUET")
                        or os.environ.get("TXN_JATIN_CONTEXT_PARQUET"))
        if not context_path:
            raise RuntimeError("set BRIDGE_CONTEXT_PARQUET for contextual gene embeddings")
        path = Path(context_path).expanduser()
        if not path.exists():
            raise RuntimeError(f"context parquet missing: {path}")

        import torch
        from torch.utils.data import DataLoader, TensorDataset

        max_samples = int(os.environ.get("BRIDGE_CONTEXT_MAX_SAMPLES", "1024"))
        batch = int(os.environ.get("BRIDGE_CONTEXT_BATCH", str(CFG.bridge_batch)))
        use_cuda_amp = (
            str(self.device).startswith("cuda")
            and os.environ.get("BRIDGE_CONTEXT_AMP", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        progress_interval = float(
            os.environ.get("BRIDGE_CONTEXT_PROGRESS_SEC", "120")
        )
        total = None
        n_seen = 0
        files_seen = 0
        started = time.time()
        last_progress = started
        self.model.eval()

        if path.is_dir():
            files = sorted((path / "batch_files").glob("*.parquet"))
            if not files:
                files = sorted(path.glob("*.parquet"))
        else:
            files = [path]
        if not files:
            raise RuntimeError(f"context parquet path has no parquet files: {path}")

        checkpoint_root = os.environ.get("BRIDGE_CONTEXT_CHECKPOINT_DIR")
        checkpoint_path = None
        checkpoint_every_files = max(
            1, int(os.environ.get("BRIDGE_CONTEXT_CHECKPOINT_EVERY_FILES", "5"))
        )
        if checkpoint_root:
            safe_name = "".join(
                ch if ch.isalnum() or ch in {"-", "_"} else "_"
                for ch in self.name
            )
            checkpoint_path = Path(checkpoint_root).expanduser() / (
                f"{safe_name}_context_accumulator.pt"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            if checkpoint_path.exists():
                state = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                valid = (
                    state.get("model") == self.name
                    and state.get("context_path") == str(path.resolve())
                    and int(state.get("max_samples", -1)) == max_samples
                    and int(state.get("file_count", -1)) == len(files)
                    and list(state.get("gene_order", [])) == self.gene_order
                )
                if valid:
                    total = state["total"].to(
                        device=self.device, dtype=torch.float32
                    )
                    n_seen = int(state["n_seen"])
                    files_seen = int(state["files_seen"])
                    if self.council:
                        self.council.check(
                            "ReproEng",
                            f"{self.name} contextual accumulator resumed",
                            True,
                            f"{n_seen} samples across {files_seen}/{len(files)} files",
                        )
                elif self.council:
                    self.council.check(
                        "ReproEng",
                        f"{self.name} contextual accumulator compatible",
                        False,
                        f"ignoring stale checkpoint: {checkpoint_path}",
                        warn_only=True,
                    )

        def save_accumulator(complete=False):
            if checkpoint_path is None or total is None:
                return
            tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save(
                {
                    "model": self.name,
                    "context_path": str(path.resolve()),
                    "max_samples": max_samples,
                    "file_count": len(files),
                    "gene_order": self.gene_order,
                    "total": total.detach().to("cpu", dtype=torch.float32),
                    "n_seen": int(n_seen),
                    "files_seen": int(files_seen),
                    "complete": bool(complete),
                    "updated_at_unix": time.time(),
                },
                tmp,
            )
            tmp.replace(checkpoint_path)

        with torch.no_grad():
            for file in files[files_seen:]:
                raw = pd.read_parquet(file)
                idx_hits = sum(str(i).upper() in self.gene_index for i in list(raw.index)[:500])
                col_hits = sum(str(c).upper() in self.gene_index for c in list(raw.columns)[:500])
                Xdf = raw.T if idx_hits > col_hits else raw

                numeric_cols = [c for c in Xdf.columns
                                if str(c).upper() in self.gene_index
                                and pd.api.types.is_numeric_dtype(Xdf[c])]
                if not numeric_cols:
                    continue
                Xdf = Xdf.reindex(columns=self.gene_order, fill_value=0.0)
                if max_samples > 0:
                    remaining = max_samples - n_seen
                    if remaining <= 0:
                        break
                    if len(Xdf) > remaining:
                        Xdf = Xdf.iloc[:remaining]
                X = np.asarray(Xdf.values, dtype=np.float32)
                file_species = "mouse" if file.name.lower().startswith("mouse_") else "human"
                X = self._normalize_context_matrix(X, file_species)
                loader = DataLoader(TensorDataset(torch.tensor(X)),
                                    batch_size=max(1, batch), shuffle=False)
                for (xb,) in loader:
                    xb = xb.to(self.device, non_blocking=True)
                    with torch.amp.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=use_cuda_amp,
                    ):
                        hidden = self._encode_hidden(self.model, xb)
                    part = hidden.detach().float().sum(dim=0)
                    total = part if total is None else total + part
                    n_seen += int(hidden.shape[0])
                    now = time.time()
                    if progress_interval > 0 and now - last_progress >= progress_interval:
                        elapsed = max(now - started, 1e-9)
                        print(
                            f"[dynamic] {self.name}: {n_seen:,} samples from "
                            f"{files_seen + 1}/{len(files)} files "
                            f"({n_seen / elapsed:.1f} samples/s, "
                            f"{elapsed / 60:.1f} min)",
                            flush=True,
                        )
                        last_progress = now
                    if max_samples > 0 and n_seen >= max_samples:
                        break
                files_seen += 1
                if (
                    files_seen % checkpoint_every_files == 0
                    or files_seen == len(files)
                    or (max_samples > 0 and n_seen >= max_samples)
                ):
                    save_accumulator()
                if max_samples > 0 and n_seen >= max_samples:
                    break
        if total is None or n_seen == 0:
            raise RuntimeError(f"context parquet has no numeric gene columns matching {self.name}: {path}")
        table = (total / max(1, n_seen)).detach().cpu().numpy().astype(np.float32)
        save_accumulator(complete=True)
        elapsed = time.time() - started
        self._prov.update(
            dynamic_embedding=True,
            context_samples=int(n_seen),
            context_files=int(files_seen),
            context_path=str(path),
            context_batch_size=int(batch),
            context_amp_bf16=bool(use_cuda_amp),
            context_elapsed_seconds=float(elapsed),
            context_embedding_sha256=hashlib.sha256(
                np.ascontiguousarray(table).view(np.uint8)
            ).hexdigest(),
        )
        save_cache(self.name, "gene_contextual", "symbol", self.gene_order, table, False)
        if self.council:
            self.council.check("CompBio", f"{self.name} contextual gene embeddings",
                               True, f"{n_seen} context samples across {files_seen} files, "
                               f"dim={table.shape[1]}, elapsed={elapsed / 60:.1f} min")
        self._gene_emb_cache = table
        return table

    def gene_embeddings(self, genes):
        checkpoint_modes = {
            "checkpoint", "checkpoint_hybrid", "hybrid", "trained_contextual"
        }
        contextual_modes = {"context", "contextual", "mean_context", "context_mean"}
        cache_kind = (
            "gene_checkpoint"
            if self.gene_embedding_mode in checkpoint_modes
            else ("gene_contextual" if self.gene_embedding_mode in contextual_modes else "gene")
        )
        cached = load_cache(self.name, cache_kind, "symbol")
        if cached and set(genes).issubset(set(cached[0])):
            idx = {g: i for i, g in enumerate(cached[0])}
            return self._align(genes, {str(g).upper(): idx[g] for g in cached[0]},
                               cached[1])
        if not self._load_model():
            return self._fallback_gene(genes, dim=128, why=self._prov.get("error", ""))
        if self.gene_embedding_mode in checkpoint_modes:
            try:
                table = self._checkpoint_embedding_table()
            except Exception as e:
                if self.council:
                    self.council.check(
                        "CompBio",
                        f"{self.name} checkpoint gene embeddings",
                        False,
                        f"{e}; falling back to contextual route",
                        warn_only=True,
                    )
                table = self._contextual_gene_table()
        elif self.gene_embedding_mode in contextual_modes:
            try:
                table = self._contextual_gene_table()
            except Exception as e:
                if self.council:
                    self.council.check("CompBio", f"{self.name} contextual gene embeddings",
                                       False, f"{e}; falling back to zero-input route",
                                       warn_only=True)
                table = self._zero_input_gene_table()
        else:
            table = self._zero_input_gene_table()         # [G_all, D]
        # build a key->row over the canonical order, return aligned to `genes`
        emb = self._align(genes, self.gene_index, table)
        # cache the full canonical table once
        save_cache(self.name, cache_kind, "symbol", self.gene_order, table, False)
        if self.council:
            covered = int(np.isfinite(emb).all(axis=1).sum())
            self.council.check("CompBio", f"{self.name} gene embeddings", True,
                               f"{covered}/{len(genes)} covered, dim={table.shape[1]}")
        self._release_model_if_requested()
        return emb

    def sample_embeddings(self, expr_df):
        """expr_df: samples x genes, values = log1p(TPM).  Reindexed to the
        canonical 15,165-gene order (missing -> 0) before the frozen forward."""
        if not self.sample_embedding_enabled:
            return None
        if not self._load_model():
            if self.council:
                self.council.check("ReproEng", f"{self.name} sample embeddings",
                                   False, "model unavailable", warn_only=True)
            return None
        if (getattr(self.model, "use_flash", False)
                and expr_df.shape[1] > 4000
                and str(self.device).startswith("cpu")):
            if self.council:
                self.council.check(
                    "ReproEng", f"{self.name} sample embeddings", False,
                    "skipped: flash checkpoint sample extraction over "
                    f"{expr_df.shape[1]} genes is CPU-prohibitive; gene embeddings "
                    "were extracted from the real checkpoint",
                    warn_only=True)
            return None
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        X = expr_df.reindex(columns=self.gene_order, fill_value=0.0)
        X = np.asarray(X.values, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(X))
        self.model.eval()
        batch_size = max(1, int(CFG.bridge_batch))
        use_cuda_amp = str(self.device).startswith("cuda")
        while True:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            out = []
            try:
                with torch.inference_mode():
                    for (xb,) in loader:
                        xb = xb.to(self.device, non_blocking=True)
                        with torch.amp.autocast(
                            device_type="cuda",
                            dtype=torch.bfloat16,
                            enabled=use_cuda_amp,
                        ):
                            hidden = self._encode_hidden(self.model, xb)
                            if self.aggregate == "mean":
                                pooled = hidden.mean(dim=1)
                            elif self.aggregate == "max":
                                pooled = hidden.max(dim=1).values
                            elif self.aggregate == "median":
                                pooled = hidden.median(dim=1).values
                            else:  # "all" == max + mean + median (Walter convention)
                                pooled = (
                                    hidden.max(dim=1).values
                                    + hidden.mean(dim=1)
                                    + hidden.median(dim=1).values
                                )
                        out.append(pooled.float().cpu().numpy().astype(np.float32))
                break
            except torch.cuda.OutOfMemoryError:
                if not use_cuda_amp or batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                out.clear()
                torch.cuda.empty_cache()
                if self.council:
                    self.council.check(
                        "ReproEng",
                        f"{self.name} sample embedding batch backoff",
                        True,
                        f"retrying with batch_size={batch_size}",
                        warn_only=True,
                    )
        emb = np.vstack(out)
        if self.council:
            self.council.check("CompBio", f"{self.name} sample embeddings", True,
                               f"{emb.shape[0]} samples x {emb.shape[1]} dims; "
                               f"batch_size={batch_size}; bf16={use_cuda_amp}")
        self._release_model_if_requested()
        return emb


# --------------------------------------------------------------------------- #
# scGPT (HuggingFace integration branch)
# --------------------------------------------------------------------------- #
class ScGPTAdapter(ModelAdapter):
    """Single-cell transformer.  Gene embeddings come from the token embedding
    table; sample embeddings treat each bulk sample as one pseudo-cell via the
    model's cell-embedding API.  HF id via SCGPT_HF_ID (default tdc/scGPT)."""
    name = "scGPT"

    def __init__(self, council=None):
        super().__init__(council)
        self.hf_id = os.environ.get("SCGPT_HF_ID", "tdc/scGPT")
        self.model = None
        self.tok = None
        self.W = None
        self.vocab = None
        self.vocab_revision = os.environ.get(
            "SCGPT_VOCAB_REV", "573c7f7782d8b9975230ba79a54b2497ebc8ab9b")

    def _load(self):
        if self.W is not None and self.vocab is not None:
            return True
        try:
            import json
            import torch
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            device = _torch_device() or torch.device("cpu")
            weights = hf_hub_download(self.hf_id, "model.safetensors")
            vocab_p = hf_hub_download(self.hf_id, "vocab.json",
                                      revision=self.vocab_revision)
            state = load_file(weights, device=str(device))
            W = state.get("gene_encoder.embedding.weight")
            if W is None:
                raise RuntimeError("model.safetensors lacks gene_encoder.embedding.weight")
            self.W = W.detach().cpu().numpy().astype(np.float32)
            with open(vocab_p, encoding="utf-8") as f:
                self.vocab = {str(k).upper(): int(v) for k, v in json.load(f).items()}
            self.dim = int(self.W.shape[1])
            self._covered = set(self.vocab)
            self._prov = dict(name=self.name, hf_id=self.hf_id,
                              weights=str(weights), vocab=str(vocab_p),
                              vocab_revision=self.vocab_revision,
                              embedding_key="gene_encoder.embedding.weight",
                              vocab_size=int(self.W.shape[0]),
                              hidden_dim=int(self.W.shape[1]),
                              gene_namespace="HGNC symbol",
                              device=str(device),
                              note="HF tdc/scGPT weights; gene embeddings loaded directly from safetensors")
            return True
        except Exception as e:
            self._prov["error"] = str(e)
            return False

    def gene_embeddings(self, genes):
        cached = load_cache(self.name, "gene", "symbol")
        if cached and set(genes).issubset(set(cached[0])):
            idx = {str(g).upper(): i for i, g in enumerate(cached[0])}
            return self._align(genes, idx, cached[1])
        if not self._load():
            return self._fallback_gene(genes, dim=512, why=self._prov.get("error", ""))
        try:
            key2row = {}
            for g in genes:
                gid = self.vocab.get(str(g).upper())
                if gid is not None and gid < self.W.shape[0]:
                    key2row[str(g).upper()] = gid
            self.dim = self.W.shape[1]
            self._covered = set(key2row)
            save_cache(self.name, "gene", "symbol",
                       [str(g).upper() for g in genes],
                       self._align(genes, key2row, self.W), False)
            return self._align(genes, key2row, self.W)
        except Exception as e:
            return self._fallback_gene(genes, dim=512, why=str(e))

    def sample_embeddings(self, expr_df):
        """Best-effort pseudo-cell cell embeddings via scgpt.tasks.embed_data.
        Returns None (documented skip) if the scGPT pipeline is unavailable."""
        try:
            import anndata as ad
            from scgpt.tasks import embed_data  # type: ignore
            adata = ad.AnnData(np.expm1(np.asarray(expr_df.values, dtype=np.float32)))
            adata.var_names = [str(g) for g in expr_df.columns]
            model_dir = os.environ.get("SCGPT_MODEL_DIR")
            if not model_dir:
                raise RuntimeError("set SCGPT_MODEL_DIR for cell embeddings")
            out = embed_data(adata, model_dir=model_dir, gene_col="index",
                             obs_to_save=None, batch_size=8, device="cpu",
                             return_new_adata=True)
            emb = np.asarray(out.obsm["X_scGPT"], dtype=np.float32)
            if self.council:
                self.council.check("CompBio", f"{self.name} sample embeddings",
                                   True, f"{emb.shape}")
            return emb
        except Exception as e:
            if self.council:
                self.council.check("CompBio", f"{self.name} sample embeddings",
                                   False, f"skipped (single-cell pipeline): {e}",
                                   warn_only=True)
            return None


# --------------------------------------------------------------------------- #
# BulkFormer (Kang et al. 2025) — native bulk model, most comparable
# --------------------------------------------------------------------------- #
class BulkFormerAdapter(ModelAdapter):
    """Bulk transcriptome foundation model (~150M params, GNN+Performer).  Set
    BULKFORMER_DIR to a clone of KangBoming/BulkFormer with its checkpoint."""
    name = "BulkFormer"

    def __init__(self, council=None, ckpt=None, name=None):
        super().__init__(council)
        if name:
            self.name = name
        self.dir = os.environ.get(
            "BULKFORMER_DIR",
            str((Path.cwd() / "external_models" / "BulkFormer").resolve()))
        self.ckpt = os.environ.get(
            "BULKFORMER_CKPT",
            str((Path(self.dir) / "model" / "bulkformer_37M.pt").resolve())
            if self.dir else "")
        if ckpt is not None:
            self.ckpt = str(ckpt)
        self.model = None
        self.vocab = None
        self.W = None

    def _load(self):
        if self.W is not None and self.vocab is not None:
            return True
        try:
            import torch
            device = _torch_device() or torch.device("cpu")
            if not self.dir:
                raise RuntimeError("set BULKFORMER_DIR to a cloned KangBoming/BulkFormer")
            BF = Path(self.dir)
            ckpt = Path(self.ckpt)
            if not ckpt.exists():
                raise RuntimeError(f"BulkFormer checkpoint missing: {ckpt}")
            vocab_p = BF / "data" / "bulkformer_gene_info.csv"
            if not vocab_p.exists():
                raise RuntimeError(f"BulkFormer gene vocabulary missing: {vocab_p}")
            state = torch.load(ckpt, map_location=device, weights_only=False)
            emb = state["module.gene_emb_onehot_layer.weight"].detach().to(device)
            w1 = state["module.gene_emb_proj.0.weight"].detach().to(device)
            b1 = state["module.gene_emb_proj.0.bias"].detach().to(device)
            w2 = state["module.gene_emb_proj.2.weight"].detach().to(device)
            b2 = state["module.gene_emb_proj.2.bias"].detach().to(device)
            W = torch.nn.functional.linear(
                torch.relu(torch.nn.functional.linear(emb, w1, b1)), w2, b2)
            self.W = W.detach().cpu().numpy().astype(np.float32)
            self.vocab = pd.read_csv(vocab_p)
            self.dim = int(self.W.shape[1])
            self._covered = set(self.vocab["gene_symbol"].astype(str).str.upper())
            self._prov = dict(name=self.name, dir=str(BF), checkpoint=str(ckpt),
                              checkpoint_variant=ckpt.stem,
                              embedding_key="gene_emb_proj(gene_emb_onehot_layer.weight)",
                              vocab=str(vocab_p), hidden_dim=self.dim,
                              gene_namespace="HGNC symbol",
                              device=str(device),
                              note="native bulk model; static learned gene identity embeddings")
            return True
        except Exception as e:
            self._prov["error"] = str(e)
            return False

    def gene_embeddings(self, genes):
        cached = load_cache(self.name, "gene", "symbol")
        if cached and set(genes).issubset(set(cached[0])):
            idx = {str(g).upper(): i for i, g in enumerate(cached[0])}
            return self._align(genes, idx, cached[1])
        if not self._load():
            return self._fallback_gene(genes, dim=1024, why=self._prov.get("error", ""))
        try:
            sym2i = {s: i for i, s in enumerate(
                self.vocab["gene_symbol"].astype(str).str.upper())}
            key2row = {str(g).upper(): sym2i[str(g).upper()]
                       for g in genes if str(g).upper() in sym2i}
            self.dim = self.W.shape[1]
            self._covered = set(key2row)
            emb = self._align(genes, key2row, self.W)
            save_cache(self.name, "gene", "symbol",
                       [str(g).upper() for g in genes], emb, False)
            return emb
        except Exception as e:
            return self._fallback_gene(genes, dim=1024, why=str(e))

    def sample_embeddings(self, expr_df):
        """Native bulk sample embeddings.  Returns None if the repo's inference
        entrypoint is not wired (documented skip — adjust to your clone)."""
        if not self._load():
            return None
        try:
            import torch
            X = np.asarray(expr_df.values, dtype=np.float32)
            device = _torch_device() or torch.device("cpu")
            with torch.no_grad():
                emb = self.model.to(device).encode(torch.tensor(X, device=device)).cpu().numpy()
            if self.council:
                self.council.check("CompBio", f"{self.name} sample embeddings",
                                   True, f"{emb.shape}")
            return emb.astype(np.float32)
        except Exception as e:
            if self.council:
                self.council.check("CompBio", f"{self.name} sample embeddings",
                                   False, f"skipped (wire encode() to your clone): {e}",
                                   warn_only=True)
            return None


# --------------------------------------------------------------------------- #
# Geneformer (jkobject fork) — rank-value encoding, Ensembl IDs
# --------------------------------------------------------------------------- #
class GeneformerAdapter(ModelAdapter):
    """Rank-value single-cell model.  Gene embeddings via the token embedding
    table keyed by Ensembl gene id + Geneformer's gene-median dictionary.  Set
    GENEFORMER_DIR (clone of jkobject/geneformer) and GENEFORMER_MODEL."""
    name = "Geneformer"

    def __init__(self, council=None):
        super().__init__(council)
        self.dir = os.environ.get("GENEFORMER_DIR")
        self.hf_id = os.environ.get("GENEFORMER_HF_ID", "ctheodoris/Geneformer")
        self.model_subdir = os.environ.get("GENEFORMER_MODEL_SUBDIR", "Geneformer-V2-104M")
        self.name2id = None
        self.tok2id = None
        self.W = None

    def _load(self):
        if self.W is not None:
            return True
        try:
            import torch
            import pickle as pk
            from transformers import BertModel
            device = _torch_device() or torch.device("cpu")
            if self.dir:
                GF = Path(self.dir)
                self.name2id = pk.load(open(GF / "geneformer" / "gene_name_id_dict.pkl", "rb"))
                self.tok2id = pk.load(open(GF / "geneformer" / "token_dictionary.pkl", "rb"))
                model_path = os.environ.get("GENEFORMER_MODEL",
                                            str(GF / "Geneformer-V2-104M"))
                source = str(GF)
            else:
                from huggingface_hub import hf_hub_download
                name_p = hf_hub_download(
                    self.hf_id, "geneformer/gene_name_id_dict_gc104M.pkl")
                tok_p = hf_hub_download(
                    self.hf_id, "geneformer/token_dictionary_gc104M.pkl")
                self.name2id = pk.load(open(name_p, "rb"))
                self.tok2id = pk.load(open(tok_p, "rb"))
                model_path = self.hf_id
                source = self.hf_id
            if self.dir:
                mdl = BertModel.from_pretrained(model_path).to(device).eval()
            else:
                mdl = BertModel.from_pretrained(
                    model_path, subfolder=self.model_subdir).to(device).eval()
            self.W = mdl.embeddings.word_embeddings.weight.detach().cpu().numpy()
            self._prov = dict(name=self.name, source=source, model=model_path,
                              model_subdir=self.model_subdir,
                              gene_namespace="Ensembl gene id (rank-value)",
                              device=str(device),
                              note="single-cell model; bulk->rank-value pseudo-cell")
            return True
        except Exception as e:
            self._prov["error"] = str(e)
            return False

    def gene_embeddings(self, genes):
        cached = load_cache(self.name, "gene", "symbol")
        if cached and set(genes).issubset(set(cached[0])):
            idx = {str(g).upper(): i for i, g in enumerate(cached[0])}
            return self._align(genes, idx, cached[1])
        if not self._load():
            return self._fallback_gene(genes, dim=512, why=self._prov.get("error", ""))
        try:
            key2row = {}
            for g in genes:
                ens = self.name2id.get(g) or self.name2id.get(str(g).upper())
                tid = self.tok2id.get(ens) if ens else None
                if tid is not None and tid < self.W.shape[0]:
                    key2row[str(g).upper()] = tid
            self.dim = self.W.shape[1]
            self._covered = set(key2row)
            emb = self._align(genes, key2row, self.W)
            save_cache(self.name, "gene", "symbol",
                       [str(g).upper() for g in genes], emb, False)
            return emb
        except Exception as e:
            return self._fallback_gene(genes, dim=512, why=str(e))

    def sample_embeddings(self, expr_df):
        """Cell embeddings via EmbExtractor over rank-value-encoded pseudo-cells.
        Returns None (documented skip) unless GENEFORMER tokenisation is wired."""
        if self.council:
            self.council.check("CompBio", f"{self.name} sample embeddings", False,
                               "skipped: requires Geneformer tokeniser + EmbExtractor "
                               "on rank-value pseudo-cells (see README caveat)",
                               warn_only=True)
        return None


# --------------------------------------------------------------------------- #
# Registry + harmonisation
# --------------------------------------------------------------------------- #
def _bulkformer_variant_adapters(council=None):
    root = Path(os.environ.get("BULKFORMER_VARIANT_DIR", Path.cwd() / "BulkFormer-main"))
    bf_dir = Path(os.environ.get(
        "BULKFORMER_DIR",
        str((Path.cwd() / "external_models" / "BulkFormer").resolve())))
    variants = [
        ("BulkFormer_37M", "BulkFormer_37M.pt", bf_dir / "model" / "bulkformer_37M.pt"),
        ("BulkFormer_50M", "BulkFormer_50M.pt", None),
        ("BulkFormer_93M", "BulkFormer_93M.pt", None),
        ("BulkFormer_127M", "BulkFormer_127M.pt", None),
        ("BulkFormer_147M", "BulkFormer_147M.pt", None),
    ]
    out = []
    for name, filename, preferred in variants:
        candidates = []
        if preferred is not None:
            candidates.append(preferred)
        candidates.extend([root / filename, bf_dir / "model" / filename])
        ckpt = next((p for p in candidates if p.exists()), candidates[0])
        out.append(BulkFormerAdapter(council, ckpt=ckpt, name=name))
    return out


def build_adapters(council=None) -> dict:
    primary_bridge = BridgeAdapter(council)
    adapters = [primary_bridge]
    baseline_ckpt = os.environ.get("BRIDGE_BASELINE_CKPT")
    baseline_name = os.environ.get("BRIDGE_BASELINE_NAME", "BRIDGE")
    if baseline_ckpt and baseline_name != primary_bridge.name:
        adapters.append(
            BridgeAdapter(
                council,
                name=baseline_name,
                checkpoint=baseline_ckpt,
                gene_embedding_mode=os.environ.get(
                    "BRIDGE_BASELINE_GENE_EMBED_MODE", "zero"
                ),
                sample_embedding_enabled=(
                    os.environ.get(
                        "BRIDGE_BASELINE_SAMPLE_EMBEDDINGS", "1"
                    ).strip().lower()
                    in {"1", "true", "yes", "on"}
                ),
            )
        )
    if os.environ.get("BRIDGE_INCLUDE_CONTEXTUAL_VARIANT", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        adapters.append(
            BridgeAdapter(
                council,
                run=primary_bridge.run,
                name=f"{primary_bridge.name}_contextual",
                gene_embedding_mode="contextual",
                sample_embedding_enabled=False,
            )
        )
    adapters.extend([ScGPTAdapter(council), GeneformerAdapter(council)])
    adapters.extend(_bulkformer_variant_adapters(council))
    return {a.name: a for a in adapters}


def extract_gene_embeddings(adapters, genes, council=None):
    """Return {model: (emb[G,D], is_fallback)} aligned to `genes`."""
    out = {}
    export_enabled = os.environ.get(
        "MYTHOS_EXPORT_EMBEDDINGS", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    export_names = {
        item.strip()
        for item in os.environ.get(
            "MYTHOS_EXPORT_EMBEDDING_MODELS", ""
        ).split(",")
        if item.strip()
    }
    for name, ad in adapters.items():
        emb = ad.gene_embeddings(list(genes))
        out[name] = (emb, ad.is_fallback)
        if export_enabled and (not export_names or name in export_names):
            EMB_CACHE.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                EMB_CACHE / f"{name}__gene_benchmark__symbol.npz",
                keys=np.asarray(list(genes), dtype=object),
                emb=np.asarray(emb, dtype=np.float32),
                is_fallback=bool(ad.is_fallback),
                provenance=np.asarray(
                    [json.dumps(ad.provenance(), sort_keys=True)], dtype=object
                ),
            )
        if council:
            cov = int(np.isfinite(emb).all(axis=1).sum())
            council.info("CompBio", f"{name}: gene coverage",
                         f"{cov}/{len(genes)}  dim={emb.shape[1]} "
                         f"{'[FALLBACK]' if ad.is_fallback else ''}")
    return out


def harmonised_gene_sets(gene_emb, genes):
    """Given {model:(emb,fb)} over a common `genes` ordering, return the index
    masks for the 'common' (all models finite) and 'full' (per-model finite)
    coverage variants."""
    genes = list(genes)
    finite = {m: np.isfinite(e).all(axis=1) for m, (e, _) in gene_emb.items()}
    common_mask = np.ones(len(genes), dtype=bool)
    for m in finite:
        common_mask &= finite[m]
    common_genes = [g for g, k in zip(genes, common_mask) if k]
    return common_genes, finite
