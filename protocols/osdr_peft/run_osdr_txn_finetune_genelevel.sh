#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-/media/volume/TrainingData/home_data/benchmarking_run/Benchmarking}"
ROOT_RUN="${ROOT_RUN:-/media/volume/AdditionalHeadroom/Txn_Jatin_runs/full_ARCHES4_20epoch_H100_20260624_015238}"
BASE_GITHUB_RUN="${BASE_GITHUB_RUN:-$ROOT_RUN/benchmarks/github_protocol_d132002_20260721_202549}"
RUN_ID="${RUN_ID:-osdr_txn_finetune_genelevel_$(date -u +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$ROOT_RUN/benchmarks/$RUN_ID}"
FT_OUT="$OUT/finetune"
SOURCE_OUT="$OUT/source/Txn_Jatin_OSDR_FT_dynamic__gene_benchmark__symbol.npz"
EMB_ROOT="$OUT/embeddings"
RESULTS="$OUT/results"
LOG_DIR="$OUT/logs"
LOG="$LOG_DIR/pipeline.log"

VENV="${VENV:-/media/volume/TrainingData/home_data/benchmarking_run/.venv/bin/python}"
BENCH_PY="${BENCH_PY:-$ROOT_RUN/env/github_protocol_py310/bin/python}"
CHECKPOINT="${CHECKPOINT:-$ROOT_RUN/checkpoints/Txn_Jatin_full_ARCHES4_20ep_H100_20260624_015238/best_model.pt}"
REFERENCE_SYMBOLS="${REFERENCE_SYMBOLS:-/media/volume/TrainingData/txn_jatin_archs4/benchmark_outputs_txn_jatin_dynamic_vs_all/embeddings}"
INTERSECTION_GENELIST="${INTERSECTION_GENELIST:-$BASE_GITHUB_RUN/embeddings/intersect/Txn_Jatin/Txn_Jatin_genelist.txt}"
MODEL_NAME="${MODEL_NAME:-Txn_Jatin_OSDR_FT_dynamic}"

mkdir -p "$FT_OUT" "$EMB_ROOT" "$RESULTS" "$LOG_DIR" "$OUT/source"
exec > >(tee -a "$LOG") 2>&1

echo "[START] $(date -u -Is) run_id=$RUN_ID"
echo "[CONFIG] project=$PROJECT"
echo "[CONFIG] checkpoint=$CHECKPOINT"
echo "[CONFIG] output=$OUT"
echo "[CONFIG] omitted=gene-pair benchmarks SL/POMBE/TF/NG; ANDES gene-set jobs not run"

cp "$BASE_GITHUB_RUN/embeddings/symbol_to_entrez_mygene.csv" "$EMB_ROOT/symbol_to_entrez_mygene.csv"

sudo -n env \
  PYTHONUNBUFFERED=1 \
  WANDB_MODE=disabled \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$VENV" "$PROJECT/Txn_Jatin/osdr_finetune/finetune_bridge_osdr.py" \
    --project-root "$PROJECT" \
    --bridge-checkpoint "$CHECKPOINT" \
    --reference-embedding-dir "$REFERENCE_SYMBOLS" \
    --output-dir "$FT_OUT" \
    --epochs "${OSDR_FT_EPOCHS:-20}" \
    --patience "${OSDR_FT_PATIENCE:-5}" \
    --batch-size "${OSDR_FT_BATCH_SIZE:-16}" \
    --context-batch-size "${OSDR_CONTEXT_BATCH_SIZE:-64}" \
    --learning-rate "${OSDR_FT_LR:-1e-5}" \
    --weight-decay "${OSDR_FT_WEIGHT_DECAY:-1e-2}" \
    --test-fraction 0.30 \
    --seed "${OSDR_FT_SEED:-42}" \
    --num-workers "${OSDR_FT_WORKERS:-8}"

sudo -n chown -R exouser:exouser "$OUT"

"$BENCH_PY" "$ROOT_RUN/scripts/append_symbol_npz_embedding.py" \
  --source-npz "$FT_OUT/BRIDGE_OSDR_FT_dynamic__gene_benchmark__symbol.npz" \
  --source-output "$SOURCE_OUT" \
  --embeddings-root "$EMB_ROOT" \
  --intersection-genelist "$INTERSECTION_GENELIST" \
  --model "$MODEL_NAME"

"$BENCH_PY" "$ROOT_RUN/scripts/run_official_gene_level_only.py" \
  --python "$BENCH_PY" \
  --benchmark-repo "$ROOT_RUN/references/gene-embedding-benchmarks" \
  --embeddings "$EMB_ROOT" \
  --output "$RESULTS" \
  --helper-dir "$ROOT_RUN/scripts" \
  --models "$MODEL_NAME" \
  --parallel "${GITHUB_GENE_LEVEL_PARALLEL:-4}"

echo "[COMPLETE] $(date -u -Is)"
touch "$OUT/ALL_COMPLETE"
