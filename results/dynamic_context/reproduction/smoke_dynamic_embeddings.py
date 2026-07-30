from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = Path("/media/volume/TrainingData/txn_jatin_archs4/train_txn_jatin_full")
TXN = Path(
    "/media/volume/TrainingData/txn_jatin_archs4/checkpoints/"
    "Txn_Jatin_A100_b9_full/epoch_00.pt"
)
BRIDGE = Path(
    "/media/volume/TrainingData/txn_jatin_archs4/baselines/"
    "BRIDGE_s66qfh36/best_model.pt"
)
OUTPUT = Path(
    "/media/volume/TrainingData/txn_jatin_archs4/"
    "benchmark_outputs_txn_jatin_dynamic_smoke"
)

os.environ.update(
    {
        "MYTHOS_BENCH_ROOT": str(ROOT),
        "WALTER_DIR": str(ROOT / "RNA Walter"),
        "BULKFORMER_DIR": str(ROOT / "external_models" / "BulkFormer"),
        "BULKFORMER_VARIANT_DIR": str(ROOT / "BulkFormer-main"),
        "BRIDGE_MODEL_NAME": "Txn_Jatin",
        "TXN_JATIN_CKPT": str(TXN),
        "MYTHOS_CANONICAL_GENES": str(DATA / "canonical_genes.csv"),
        "BRIDGE_GENE_EMBED_MODE": "contextual",
        "BRIDGE_CONTEXT_PARQUET": str(DATA),
        "BRIDGE_CONTEXT_MAX_SAMPLES": os.environ.get(
            "BRIDGE_CONTEXT_MAX_SAMPLES", "32"
        ),
        "BRIDGE_CONTEXT_BATCH": os.environ.get("BRIDGE_CONTEXT_BATCH", "16"),
        "BRIDGE_CONTEXT_AMP": "1",
        "BRIDGE_BASELINE_CKPT": str(BRIDGE),
        "BRIDGE_BASELINE_NAME": "BRIDGE",
        "BRIDGE_BASELINE_GENE_EMBED_MODE": "contextual",
        "MYTHOS_RELEASE_BRIDGE_MODELS": "1",
        "MYTHOS_OUTPUT_ROOT": str(OUTPUT),
        "MYTHOS_DEVICE": "cuda",
        "MYTHOS_DISABLE_CACHE": "1",
        "MYTHOS_EXPORT_EMBEDDINGS": "1",
        "MYTHOS_EXPORT_EMBEDDING_MODELS": "",
        "MYTHOS_BENCH_MAX_GENES": "256",
    }
)

from mythos import adapters as A
from mythos import data as D
from mythos.common import Council


def mean_cosine(a, b):
    mask = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    a = a[mask]
    b = b[mask]
    return float(
        np.mean(
            np.sum(a * b, axis=1)
            / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        )
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    council = Council()
    genes = D.load_canonical_genes(council)[:256]
    adapters = A.build_adapters(council)
    dynamic = A.extract_gene_embeddings(adapters, genes, council)

    static_txn_adapter = A.BridgeAdapter(
        council, name="Txn_Jatin_static_check", checkpoint=TXN,
        gene_embedding_mode="zero", sample_embedding_enabled=False
    )
    static_bridge_adapter = A.BridgeAdapter(
        council, name="BRIDGE_static_check", checkpoint=BRIDGE,
        gene_embedding_mode="zero", sample_embedding_enabled=False
    )
    static_txn = static_txn_adapter.gene_embeddings(genes)
    static_bridge = static_bridge_adapter.gene_embeddings(genes)

    report = {}
    for name, adapter in adapters.items():
        emb, fallback = dynamic[name]
        report[name] = {
            "shape": list(emb.shape),
            "finite_rows": int(np.isfinite(emb).all(axis=1).sum()),
            "fallback": bool(fallback),
            "provenance": adapter.provenance(),
        }
    report["dynamic_vs_static"] = {
        "Txn_Jatin_mean_gene_cosine": mean_cosine(dynamic["Txn_Jatin"][0], static_txn),
        "BRIDGE_mean_gene_cosine": mean_cosine(dynamic["BRIDGE"][0], static_bridge),
        "Txn_Jatin_exactly_equal": bool(
            np.array_equal(dynamic["Txn_Jatin"][0], static_txn, equal_nan=True)
        ),
        "BRIDGE_exactly_equal": bool(
            np.array_equal(dynamic["BRIDGE"][0], static_bridge, equal_nan=True)
        ),
    }

    if any(report[name]["fallback"] for name in adapters):
        raise RuntimeError("one or more adapters fell back during dynamic smoke")
    for name in ("Txn_Jatin", "BRIDGE"):
        prov = report[name]["provenance"]
        expected_samples = int(os.environ["BRIDGE_CONTEXT_MAX_SAMPLES"])
        if (
            prov.get("context_samples") != expected_samples
            or not prov.get("dynamic_embedding")
        ):
            raise RuntimeError(f"{name} did not produce verified dynamic embeddings: {prov}")
    if report["dynamic_vs_static"]["Txn_Jatin_exactly_equal"]:
        raise RuntimeError("Txn_Jatin dynamic embedding unexpectedly equals static embedding")
    if report["dynamic_vs_static"]["BRIDGE_exactly_equal"]:
        raise RuntimeError("BRIDGE dynamic embedding unexpectedly equals static embedding")

    (OUTPUT / "dynamic_smoke_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
