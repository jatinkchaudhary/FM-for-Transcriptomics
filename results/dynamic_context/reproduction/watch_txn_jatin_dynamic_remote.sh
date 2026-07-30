#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/exouser/benchmarking_run/Benchmarking}"
VENV="${VENV:-/home/exouser/benchmarking_run/.venv}"
VOLUME_ROOT="${VOLUME_ROOT:-/media/volume/TrainingData/txn_jatin_archs4}"
OUTPUT_ROOT="${MYTHOS_OUTPUT_ROOT:-$VOLUME_ROOT/benchmark_outputs_txn_jatin_dynamic_vs_all}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs}"
WATCH_LOG="$LOG_ROOT/txn_jatin_dynamic_watchdog.log"
RETRY_FILE="$LOG_ROOT/txn_jatin_dynamic_watchdog.retries"
MAIN_STATUS="$LOG_ROOT/txn_jatin_dynamic_vs_all_benchmark.status"
PAIR_STATUS="$LOG_ROOT/txn_jatin_dynamic_paired_supplement.status"
MAX_RETRIES="${MAX_RETRIES:-3}"

mkdir -p "$LOG_ROOT"
touch "$WATCH_LOG"
[[ -f "$RETRY_FILE" ]] || printf '0\n' >"$RETRY_FILE"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >>"$WATCH_LOG"
}

status_value() {
  local path="$1"
  if [[ -f "$path" ]]; then
    tr -d '[:space:]' <"$path"
  fi
}

while true; do
  main_status="$(status_value "$MAIN_STATUS")"
  pair_status="$(status_value "$PAIR_STATUS")"

  if [[ "$main_status" == "0" && "$pair_status" == "0" ]]; then
    log "main and paired supplement completed successfully; watchdog exiting"
    exit 0
  fi

  if pgrep -f "bash scripts/run_txn_jatin_dynamic_benchmark_remote.sh" \
      >/dev/null 2>&1 \
      || pgrep -f "python run_txn_jatin_dynamic_benchmark.py" >/dev/null 2>&1 \
      || pgrep -f "python run_dynamic_paired_supplement.py" >/dev/null 2>&1; then
    sleep 300
    continue
  fi

  retries="$(cat "$RETRY_FILE")"
  if (( retries >= MAX_RETRIES )); then
    log "retry limit reached ($MAX_RETRIES); watchdog exiting"
    exit 1
  fi
  retries=$((retries + 1))
  printf '%s\n' "$retries" >"$RETRY_FILE"

  source "$VENV/bin/activate"
  cd "$PROJECT_ROOT"
  export PYTHONUNBUFFERED=1
  export WANDB_MODE=disabled
  export MYTHOS_OUTPUT_ROOT="$OUTPUT_ROOT"
  export MYTHOS_DEVICE=cuda
  export MYTHOS_PROBE=torch
  export MYTHOS_THREADS=32
  export MYTHOS_CV_FOLDS=5
  export MYTHOS_MAX_PAIRS=200000
  export MPLBACKEND=Agg

  if [[ "$main_status" == "0" ]]; then
    log "paired supplement incomplete; retry $retries/$MAX_RETRIES"
    export TXN_TRRUST_TSV="$PROJECT_ROOT/RNA Walter/data/gene_set_libs/trrust.tsv"
    rm -f "$PAIR_STATUS"
    setsid -f bash -c \
      'python run_dynamic_paired_supplement.py \
       >>logs/txn_jatin_dynamic_paired_supplement.log 2>&1; \
       printf "%s\n" "$?" >logs/txn_jatin_dynamic_paired_supplement.status' \
      </dev/null
  else
    log "main benchmark incomplete; retry $retries/$MAX_RETRIES from accumulator"
    rm -f "$MAIN_STATUS" "$PAIR_STATUS"
    setsid -f bash scripts/run_txn_jatin_dynamic_benchmark_remote.sh \
      >>"$LOG_ROOT/txn_jatin_dynamic_launcher.log" 2>&1 </dev/null
  fi
  sleep 300
done

