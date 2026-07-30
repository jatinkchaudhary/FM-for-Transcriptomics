from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = Path(
    os.environ.get(
        "MYTHOS_OUTPUT_ROOT",
        "/media/volume/TrainingData/txn_jatin_archs4/"
        "benchmark_outputs_txn_jatin_dynamic_vs_all",
    )
)
TXN_DATA = Path(
    os.environ.get(
        "TXN_DATA_DIR",
        "/media/volume/TrainingData/txn_jatin_archs4/train_txn_jatin_full",
    )
)
TXN_CKPT = Path(
    os.environ.get(
        "TXN_JATIN_CKPT",
        "/media/volume/TrainingData/txn_jatin_archs4/checkpoints/"
        "Txn_Jatin_A100_b9_full/epoch_00.pt",
    )
)
BRIDGE_CKPT = Path(
    os.environ.get(
        "BRIDGE_BASELINE_CKPT",
        "/media/volume/TrainingData/txn_jatin_archs4/baselines/"
        "BRIDGE_s66qfh36/best_model.pt",
    )
)


def configure() -> None:
    os.chdir(ROOT)
    os.environ.update(
        {
            "MYTHOS_BENCH_ROOT": str(ROOT),
            "WALTER_DIR": str(ROOT / "RNA Walter"),
            "BULKFORMER_DIR": str(ROOT / "external_models" / "BulkFormer"),
            "BULKFORMER_VARIANT_DIR": str(ROOT / "BulkFormer-main"),
            "BRIDGE_MODEL_NAME": "Txn_Jatin",
            "BRIDGE_RUN": "Txn_Jatin_A100_b9_full",
            "TXN_JATIN_CKPT": str(TXN_CKPT),
            "MYTHOS_CANONICAL_GENES": str(TXN_DATA / "canonical_genes.csv"),
            "BRIDGE_GENE_EMBED_MODE": "contextual",
            "BRIDGE_CONTEXT_PARQUET": str(TXN_DATA),
            "BRIDGE_CONTEXT_MAX_SAMPLES": os.environ.get(
                "BRIDGE_CONTEXT_MAX_SAMPLES", "0"
            ),
            "BRIDGE_CONTEXT_BATCH": os.environ.get("BRIDGE_CONTEXT_BATCH", "96"),
            "BRIDGE_CONTEXT_AMP": "1",
            "BRIDGE_CONTEXT_CHECKPOINT_DIR": os.environ.get(
                "BRIDGE_CONTEXT_CHECKPOINT_DIR",
                "/media/volume/TrainingData/txn_jatin_archs4/"
                "dynamic_context_checkpoints",
            ),
            "BRIDGE_CONTEXT_CHECKPOINT_EVERY_FILES": os.environ.get(
                "BRIDGE_CONTEXT_CHECKPOINT_EVERY_FILES", "5"
            ),
            "BRIDGE_INCLUDE_CONTEXTUAL_VARIANT": "0",
            "BRIDGE_BASELINE_CKPT": str(BRIDGE_CKPT),
            "BRIDGE_BASELINE_NAME": "BRIDGE",
            "BRIDGE_BASELINE_GENE_EMBED_MODE": "contextual",
            "BRIDGE_BASELINE_SAMPLE_EMBEDDINGS": "1",
            "MYTHOS_RELEASE_BRIDGE_MODELS": "1",
            "MYTHOS_OUTPUT_ROOT": str(OUTPUT),
            "MYTHOS_DEVICE": "cuda",
            "MYTHOS_PROBE": "torch",
            "MYTHOS_DISABLE_CACHE": "1",
            "MYTHOS_EXPORT_EMBEDDINGS": "1",
            "MYTHOS_EXPORT_EMBEDDING_MODELS": "",
            "MYTHOS_SKIP_PAIRED": "1",
            "MYTHOS_THREADS": os.environ.get("MYTHOS_THREADS", "32"),
            "MYTHOS_TCGA_MAX": "0",
            "MYTHOS_OSDR_MAX_ACC": "0",
            "MYTHOS_BENCH_MAX_GENES": "200000",
            "MYTHOS_N_TERMS": os.environ.get("MYTHOS_N_TERMS", "40"),
            "MYTHOS_CV_FOLDS": "5",
            "MYTHOS_MAX_PAIRS": "200000",
            "MYTHOS_BRIDGE_BATCH": os.environ.get("MYTHOS_BRIDGE_BATCH", "8"),
            "MYTHOS_TORCH_PROBE_EPOCHS": "80",
            "MPLBACKEND": "Agg",
        }
    )


def validate() -> None:
    required = [
        TXN_CKPT,
        BRIDGE_CKPT,
        TXN_DATA / "canonical_genes.csv",
        TXN_DATA / "batch_files",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing dynamic benchmark inputs: " + ", ".join(missing))


def main() -> int:
    configure()
    validate()
    if os.environ.get("MYTHOS_CLEAN_OUTPUTS", "1") == "1" and OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    from mythos.run_all import main as run_all

    status = int(run_all([]))
    marker = json.loads((TXN_DATA / "PREPARED_FULL_DATA.ok").read_text())
    expected_context_samples = int(marker["samples"])
    audit = {}
    for path in sorted((OUTPUT / "embeddings").glob(
        "*__gene_benchmark__symbol.npz"
    )):
        model = path.name.split("__gene_benchmark__", 1)[0]
        with np.load(path, allow_pickle=True) as payload:
            provenance = json.loads(str(payload["provenance"][0]))
            fallback = bool(payload["is_fallback"])
            shape = list(payload["emb"].shape)
        audit[model] = {
            "shape": shape,
            "fallback": fallback,
            "provenance": provenance,
        }
    if len(audit) != 9:
        raise RuntimeError(f"expected 9 exported model embeddings, found {sorted(audit)}")
    if any(item["fallback"] for item in audit.values()):
        raise RuntimeError("one or more exported model embeddings used fallback vectors")
    for model in ("Txn_Jatin", "BRIDGE"):
        provenance = audit[model]["provenance"]
        if not provenance.get("dynamic_embedding"):
            raise RuntimeError(f"{model} was not exported as a dynamic embedding")
        if int(provenance.get("context_samples", -1)) != expected_context_samples:
            raise RuntimeError(
                f"{model} context sample count "
                f"{provenance.get('context_samples')} != {expected_context_samples}"
            )
    for model, item in audit.items():
        if model not in {"Txn_Jatin", "BRIDGE"} and item["provenance"].get(
            "dynamic_embedding"
        ):
            raise RuntimeError(f"{model} unexpectedly used dynamic embeddings")
    (OUTPUT / "dynamic_embedding_audit.json").write_text(
        json.dumps(audit, indent=2)
    )
    (OUTPUT / "dynamic_run_configuration.json").write_text(
        json.dumps(
            {
                "Txn_Jatin": {
                    "checkpoint": str(TXN_CKPT),
                    "embedding_mode": "contextual",
                },
                "BRIDGE": {
                    "checkpoint": str(BRIDGE_CKPT),
                    "embedding_mode": "contextual",
                },
                "other_models": "unchanged static adapters",
                "context_data": str(TXN_DATA),
                "context_max_samples": int(
                    os.environ["BRIDGE_CONTEXT_MAX_SAMPLES"]
                ),
                "context_batch": int(os.environ["BRIDGE_CONTEXT_BATCH"]),
                "context_amp": "bf16",
                "context_checkpoint_dir": os.environ[
                    "BRIDGE_CONTEXT_CHECKPOINT_DIR"
                ],
                "expected_context_samples": expected_context_samples,
                "benchmark_terms": int(os.environ["MYTHOS_N_TERMS"]),
                "cv_folds": int(os.environ["MYTHOS_CV_FOLDS"]),
                "paired_in_separate_supplement": True,
            },
            indent=2,
        )
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
