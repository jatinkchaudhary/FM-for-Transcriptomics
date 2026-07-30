#!/usr/bin/env bash
set -uo pipefail

BENCH="${1:?benchmark output directory is required}"
RUN="/media/volume/AdditionalHeadroom/Txn_Jatin_runs/full_ARCHES4_20epoch_H100_20260624_015238"
PYTHON="/media/volume/TrainingData/home_data/benchmarking_run/.venv/bin/python"
SCRIPTS="$RUN/scripts"
export PYTHONPATH="$BENCH/python_packages_clean:$SCRIPTS:${PYTHONPATH:-}"
MODELS=(
  BulkFormer_37M
  Txn_Jatin
  BRIDGE
  BulkFormer_50M
  BulkFormer_93M
  BulkFormer_127M
  BulkFormer_147M
)

mkdir -p "$BENCH"/{logs,status}
exec > >(tee -a "$BENCH/logs/orchestrator.log") 2>&1
echo "$$" > "$BENCH/status/orchestrator.pid"
echo "[$(date --iso-8601=seconds)] Benchmark orchestrator started"

(
  "$PYTHON" "$SCRIPTS/run_rf_baseline.py" \
    --run-dir "$BENCH" --trees 300 --jobs 12 \
    > "$BENCH/logs/raw_rf.log" 2>&1
  code=$?
  if [[ $code -ne 0 ]]; then
    echo "$code" > "$BENCH/status/RAW_RF_FAILED"
  fi
) &
RAW_RF_PID=$!
echo "$RAW_RF_PID" > "$BENCH/status/raw_rf.pid"

(
  for model in "${MODELS[@]}"; do
    while [[ ! -f "$BENCH/status/$model.READY" && ! -f "$BENCH/status/$model.FAILED" ]]; do
      sleep 30
    done
    if [[ -f "$BENCH/status/$model.READY" ]]; then
      "$PYTHON" "$SCRIPTS/analyze_imputation.py" \
        --run-dir "$BENCH" --model "$model" \
        > "$BENCH/logs/${model}_metrics.log" 2>&1
      "$PYTHON" "$SCRIPTS/aggregate_masked_results.py" \
        --run-dir "$BENCH" \
        >> "$BENCH/logs/aggregate.log" 2>&1
    fi
  done
) &
METRICS_PID=$!
echo "$METRICS_PID" > "$BENCH/status/metrics_worker.pid"

(
  while kill -0 "$RAW_RF_PID" 2>/dev/null; do
    sleep 30
  done
  for model in "${MODELS[@]}"; do
    while [[ ! -f "$BENCH/status/$model.READY" && ! -f "$BENCH/status/$model.FAILED" ]]; do
      sleep 30
    done
    if [[ -f "$BENCH/status/$model.READY" ]]; then
      "$PYTHON" "$SCRIPTS/run_masked_rf.py" \
        --run-dir "$BENCH" --model "$model" --trees 300 --jobs 12 \
        > "$BENCH/logs/${model}_rf.log" 2>&1
      "$PYTHON" "$SCRIPTS/aggregate_masked_results.py" \
        --run-dir "$BENCH" \
        >> "$BENCH/logs/aggregate.log" 2>&1
    fi
  done
) &
RF_PID=$!
echo "$RF_PID" > "$BENCH/status/rf_worker.pid"

for model in "${MODELS[@]}"; do
  if [[ -f "$BENCH/status/$model.READY" ]]; then
    echo "[$(date --iso-8601=seconds)] Skipping completed inference: $model"
    continue
  fi
  batch_size=8
  if [[ "$model" == "Txn_Jatin" || "$model" == "BRIDGE" ]]; then
    batch_size=4
  fi
  echo "[$(date --iso-8601=seconds)] Starting H100 inference: $model"
  "$PYTHON" "$SCRIPTS/run_decoder_inference.py" \
    --run-dir "$BENCH" --model "$model" --batch-size "$batch_size" \
    > "$BENCH/logs/${model}_inference.log" 2>&1
  code=$?
  if [[ $code -ne 0 ]]; then
    echo "$code" > "$BENCH/status/$model.FAILED"
    echo "[$(date --iso-8601=seconds)] $model failed with exit $code; continuing"
  else
    echo "[$(date --iso-8601=seconds)] Completed H100 inference: $model"
  fi
done

wait "$RAW_RF_PID"
wait "$METRICS_PID"
wait "$RF_PID"
"$PYTHON" "$SCRIPTS/aggregate_masked_results.py" --run-dir "$BENCH"
touch "$BENCH/status/ALL_COMPLETE"
echo "[$(date --iso-8601=seconds)] Benchmark complete"
