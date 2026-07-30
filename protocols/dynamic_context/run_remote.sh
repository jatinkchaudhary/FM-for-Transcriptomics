#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/exouser/benchmarking_run/Benchmarking}"
VENV="${VENV:-/home/exouser/benchmarking_run/.venv}"
VOLUME_ROOT="${VOLUME_ROOT:-/media/volume/TrainingData/txn_jatin_archs4}"
OUTPUT_ROOT="${MYTHOS_OUTPUT_ROOT:-$VOLUME_ROOT/benchmark_outputs_txn_jatin_dynamic_vs_all}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs}"
MAIN_LOG="$LOG_ROOT/txn_jatin_dynamic_vs_all_benchmark.log"
MAIN_STATUS="$LOG_ROOT/txn_jatin_dynamic_vs_all_benchmark.status"
PAIR_LOG="$LOG_ROOT/txn_jatin_dynamic_paired_supplement.log"
PAIR_STATUS="$LOG_ROOT/txn_jatin_dynamic_paired_supplement.status"

mkdir -p "$LOG_ROOT"
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export MYTHOS_OUTPUT_ROOT="$OUTPUT_ROOT"
export BRIDGE_CONTEXT_MAX_SAMPLES=0
export BRIDGE_CONTEXT_BATCH="${BRIDGE_CONTEXT_BATCH:-96}"
export BRIDGE_CONTEXT_AMP=1
export BRIDGE_CONTEXT_PROGRESS_SEC=120
export BRIDGE_CONTEXT_CHECKPOINT_DIR="${BRIDGE_CONTEXT_CHECKPOINT_DIR:-$VOLUME_ROOT/dynamic_context_checkpoints}"
export BRIDGE_CONTEXT_CHECKPOINT_EVERY_FILES="${BRIDGE_CONTEXT_CHECKPOINT_EVERY_FILES:-5}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set +e
python run_txn_jatin_dynamic_benchmark.py >"$MAIN_LOG" 2>&1
main_code=$?
set -e
printf '%s\n' "$main_code" >"$MAIN_STATUS"
if [[ "$main_code" -ne 0 ]]; then
  exit "$main_code"
fi

export TXN_TRRUST_TSV="$PROJECT_ROOT/RNA Walter/data/gene_set_libs/trrust.tsv"
set +e
python run_dynamic_paired_supplement.py >"$PAIR_LOG" 2>&1
pair_code=$?
set -e
printf '%s\n' "$pair_code" >"$PAIR_STATUS"
exit "$pair_code"
