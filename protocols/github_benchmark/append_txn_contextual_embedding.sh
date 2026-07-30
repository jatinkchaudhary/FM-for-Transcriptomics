#!/usr/bin/env bash
set -euo pipefail

RUN=/media/volume/AdditionalHeadroom/Txn_Jatin_runs/full_ARCHES4_20epoch_H100_20260624_015238
OUT=$(cat "$RUN/benchmarks/LATEST_GITHUB_PROTOCOL_RUN.txt")
PROTO_PY="$RUN/env/github_protocol_py310/bin/python"
TRAIN_PY=/media/volume/TrainingData/home_data/benchmarking_run/.venv/bin/python
MANIFEST="$OUT/embeddings/embedding_manifest.json"
CHECKPOINT="$RUN/checkpoints/Txn_Jatin_full_ARCHES4_20ep_H100_20260624_015238/best_model.pt"
CANONICAL_GENES="$RUN/checkpoints/Txn_Jatin_full_ARCHES4_20ep_H100_20260624_015238/canonical_genes.csv"
LOG="$OUT/logs/contextual_append.log"

exec >>"$LOG" 2>&1

STATIC_SOURCE=$(
    "$PROTO_PY" - "$MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["models"]["Txn_Jatin"]["source"])
PY
)
SOURCE_OUTPUT="$(dirname "$STATIC_SOURCE")/Txn_Jatin_contextual__gene_benchmark__symbol.npz"

echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "checkpoint=$CHECKPOINT"
echo "static_source=$STATIC_SOURCE"
echo "source_output=$SOURCE_OUTPUT"

env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$TRAIN_PY" \
    "$RUN/scripts/append_txn_contextual_embedding.py" \
    --checkpoint "$CHECKPOINT" \
    --canonical-genes "$CANONICAL_GENES" \
    --static-source "$STATIC_SOURCE" \
    --source-output "$SOURCE_OUTPUT" \
    --embeddings-root "$OUT/embeddings" \
    --model Txn_Jatin_contextual \
    --context-key contextual_gene_embedding

touch "$OUT/results/TXN_JATIN_CONTEXTUAL_READY"
echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
