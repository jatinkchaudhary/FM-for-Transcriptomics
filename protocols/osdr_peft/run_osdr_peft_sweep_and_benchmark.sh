#!/usr/bin/env bash
set -Eeuo pipefail

RUN=/media/volume/AdditionalHeadroom/Txn_Jatin_runs/full_ARCHES4_20epoch_H100_20260624_015238
PROJECT=/media/volume/TrainingData/home_data/benchmarking_run/Benchmarking
VENV=/media/volume/TrainingData/home_data/benchmarking_run/.venv/bin/python
BENCH_PY=$RUN/env/github_protocol_py310/bin/python
BASE_GITHUB_RUN=$RUN/benchmarks/github_protocol_d132002_20260721_202549
SOURCE_CHECKPOINT=$RUN/checkpoints/Txn_Jatin_full_ARCHES4_20ep_H100_20260624_015238/best_model.pt
SHARED_DATA=$RUN/benchmarks/osdr_txn_finetune_genelevel_20260723_233746/finetune
REFERENCE_SYMBOLS=/media/volume/TrainingData/txn_jatin_archs4/benchmark_outputs_txn_jatin_dynamic_vs_all/embeddings
SWEEP_ID=${SWEEP_ID:-osdr_txn_peft_sweep_$(date -u +%Y%m%d_%H%M%S)}
OUT=$RUN/benchmarks/$SWEEP_ID
LOG=$OUT/logs/pipeline.log

mkdir -p "$OUT/logs" "$OUT/candidates" "$OUT/benchmark/source" "$OUT/benchmark/embeddings" "$OUT/benchmark/results"
exec > >(tee -a "$LOG") 2>&1

echo "[START] $(date -u -Is) sweep=$SWEEP_ID"
echo "[INFO] full-unfreeze candidates intentionally excluded"
cp "$BASE_GITHUB_RUN/embeddings/symbol_to_entrez_mygene.csv" "$OUT/benchmark/embeddings/symbol_to_entrez_mygene.csv"

run_candidate() {
  local name="$1"; shift
  local dir="$OUT/candidates/$name"
  mkdir -p "$dir"
  if [[ -s "$dir/run.status" ]]; then
    echo "[SKIP] $name already complete"
    return
  fi
  echo "[CANDIDATE_START] $(date -u -Is) $name"
  "$VENV" "$RUN/scripts/osdr_peft_finetune.py" \
    --project-root "$PROJECT" \
    --checkpoint "$SOURCE_CHECKPOINT" \
    --shared-data-dir "$SHARED_DATA" \
    --reference-embedding-dir "$REFERENCE_SYMBOLS" \
    --output-dir "$dir" \
    --model-name "Txn_Jatin_OSDR_${name}_dynamic" \
    --epochs "${OSDR_PEFT_EPOCHS:-20}" \
    --patience "${OSDR_PEFT_PATIENCE:-5}" \
    --batch-size "${OSDR_PEFT_BATCH_SIZE:-16}" \
    --context-batch-size "${OSDR_CONTEXT_BATCH_SIZE:-64}" \
    --learning-rate "${OSDR_PEFT_LR:-1e-5}" \
    --weight-decay "${OSDR_PEFT_WEIGHT_DECAY:-1e-2}" \
    --seed 42 \
    --num-workers "${OSDR_PEFT_WORKERS:-8}" \
    "$@"
  echo "[CANDIDATE_DONE] $(date -u -Is) $name"
}

run_candidate head_only --method head_only
run_candidate bitfit_norm --method bitfit_norm
run_candidate last1 --method last_n --unfreeze-last-n 1
run_candidate last2 --method last_n --unfreeze-last-n 2
run_candidate last3 --method last_n --unfreeze-last-n 3
run_candidate lora_attn_all_r4 --method lora --lora-rank 4 --lora-alpha 8 --lora-target attn --lora-layer-scope all
run_candidate lora_attn_last6_r8 --method lora --lora-rank 8 --lora-alpha 16 --lora-target attn --lora-layer-scope last6
run_candidate lora_attn_ffn_last3_r4 --method lora --lora-rank 4 --lora-alpha 8 --lora-target attn_ffn --lora-layer-scope last3

"$VENV" "$RUN/scripts/select_osdr_peft_winner.py" --sweep-dir "$OUT"
winner_model=$("$VENV" - <<PY
import json
from pathlib import Path
p=Path("$OUT")/"winner_selection.json"
print(json.loads(p.read_text())["winner"]["model"])
PY
)
winner_dir=$("$VENV" - <<PY
import json
from pathlib import Path
p=Path("$OUT")/"winner_selection.json"
print(json.loads(p.read_text())["winner"]["path"])
PY
)
echo "[WINNER] model=$winner_model dir=$winner_dir"

"$BENCH_PY" "$RUN/scripts/append_symbol_npz_embedding.py" \
  --source-npz "$winner_dir/${winner_model}__gene_benchmark__symbol.npz" \
  --source-output "$OUT/benchmark/source/${winner_model}__gene_benchmark__symbol.npz" \
  --embeddings-root "$OUT/benchmark/embeddings" \
  --intersection-genelist "$BASE_GITHUB_RUN/embeddings/intersect/Txn_Jatin/Txn_Jatin_genelist.txt" \
  --model "$winner_model"

"$BENCH_PY" "$RUN/scripts/run_official_gene_level_only.py" \
  --python "$BENCH_PY" \
  --benchmark-repo "$RUN/references/gene-embedding-benchmarks" \
  --embeddings "$OUT/benchmark/embeddings" \
  --output "$OUT/benchmark/results" \
  --helper-dir "$RUN/scripts" \
  --models "$winner_model" \
  --parallel "${GITHUB_GENE_LEVEL_PARALLEL:-4}"

echo "[COMPLETE] $(date -u -Is)"
touch "$OUT/ALL_COMPLETE"
