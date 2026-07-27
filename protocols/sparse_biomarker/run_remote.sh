#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/media/volume/AdditionalHeadroom/Txn_Jatin_runs/full_ARCHES4_20epoch_H100_20260624_015238/benchmarks/txn_jatin_biomarker_pilot_20260725"
PYTHON="/media/volume/TrainingData/home_data/benchmarking_run/.venv/bin/python"
SCRIPT="$RUN_DIR/scripts/txn_jatin_biomarker_pilot.py"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/status"

run_panel() {
  local panel_size="$1"
  set +e
  "$PYTHON" "$SCRIPT" infer \
    --run-dir "$RUN_DIR" \
    --panel-size "$panel_size" \
    --batch-size 4 \
    2>&1 | tee "$RUN_DIR/logs/inference_panel_${panel_size}.log"
  local inference_code="${PIPESTATUS[0]}"
  set -e
  if [[ "$inference_code" -ne 0 ]] && \
     ! grep -q '"status": "complete"' "$RUN_DIR/status/inference_panel_${panel_size}.json"; then
    return "$inference_code"
  fi
  "$PYTHON" "$SCRIPT" analyze \
    --run-dir "$RUN_DIR" \
    --panel-size "$panel_size" \
    --jobs 12 \
    2>&1 | tee "$RUN_DIR/logs/analysis_panel_${panel_size}.log"
}

run_panel 2000
run_panel 1000
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$RUN_DIR/status/ALL_COMPLETE"
