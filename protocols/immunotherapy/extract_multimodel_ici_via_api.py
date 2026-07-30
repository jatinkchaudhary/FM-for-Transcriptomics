#!/usr/bin/env python3
"""Extract completed ICI expression matrices from the deployed H100 decoders."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests


MODELS = (
    "Txn_Jatin",
    "Txn_Jatin_OSDR_LoRA",
    "BRIDGE",
    "BulkFormer_37M",
    "BulkFormer_50M",
    "BulkFormer_93M",
    "BulkFormer_127M",
    "BulkFormer_147M",
)
COHORTS = ("gide", "riaz", "hugo", "rose")


def request_batch(session, url, payload, attempts=4):
    for attempt in range(1, attempts + 1):
        try:
            response = session.post(url, json=payload, timeout=(30, 900))
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == attempts:
                raise
            time.sleep(5 * attempt)
    raise RuntimeError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--txn-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--api",
        default="http://nvidea-h100-2-of-2.bio260281.projects.jetstream-cloud.org:8000",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    manifest = {"api": args.api, "models": {}, "batch_size": args.batch_size}

    for model in MODELS:
        manifest["models"][model] = {}
        for cohort in COHORTS:
            output_path = args.output / f"{cohort}__{model}.npz"
            if output_path.exists():
                with np.load(output_path, allow_pickle=False) as data:
                    manifest["models"][model][cohort] = {
                        "status": "reused",
                        "shape": list(data["completed"].shape),
                        "finite_fraction": float(np.isfinite(data["completed"]).mean()),
                    }
                continue
            source_path = args.txn_results / f"{cohort}_txn_embeddings.npz"
            with np.load(source_path, allow_pickle=False) as data:
                genes = data["genes"].astype(str)
                sample_ids = data["sample_ids"].astype(str)
                raw = np.asarray(data["aligned_log1p_tpm"], dtype=np.float32)
                measured = np.asarray(data["measured_gene_mask"], dtype=bool)
            completed = np.full_like(raw, np.nan)
            unresolved = set()
            for start in range(0, len(sample_ids), args.batch_size):
                end = min(start + args.batch_size, len(sample_ids))
                values = raw[start:end]
                missing = np.broadcast_to(~measured, values.shape)
                payload = {
                    "model": model,
                    "genes": genes.tolist(),
                    "samples": sample_ids[start:end].tolist(),
                    "matrix": values.T.tolist(),
                    "missing": missing.T.tolist(),
                    "input_scale": "log1p",
                }
                result = request_batch(session, f"{args.api}/api/impute", payload)
                matrix = np.asarray(
                    [
                        [np.nan if value is None else float(value) for value in row]
                        for row in result["imputed"]
                    ],
                    dtype=np.float32,
                ).T
                completed[start:end] = matrix
                unresolved.update(result.get("unresolved_genes", []))
                print(
                    f"{model} {cohort}: {end}/{len(sample_ids)} "
                    f"matched={result.get('matched_genes')}",
                    flush=True,
                )
            np.savez_compressed(
                output_path,
                completed=completed,
                raw=raw,
                genes=genes,
                sample_ids=sample_ids,
                measured_gene_mask=measured,
            )
            manifest["models"][model][cohort] = {
                "status": "completed",
                "shape": list(completed.shape),
                "finite_fraction": float(np.isfinite(completed).mean()),
                "unresolved_genes": len(unresolved),
            }
            (args.output / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
    manifest["completed"] = True
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
