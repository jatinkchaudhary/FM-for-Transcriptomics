#!/usr/bin/env bash
set -Eeuo pipefail

RUN=/media/volume/AdditionalHeadroom/Txn_Jatin_runs/full_ARCHES4_20epoch_H100_20260624_015238
OUT=$RUN/benchmarks/osdr_txn_finetune_genelevel_20260723_233746
FT_OUT=$OUT/finetune
SOURCE_OUT=$OUT/source/Txn_Jatin_OSDR_FT_dynamic__gene_benchmark__symbol.npz
EMB_ROOT=$OUT/embeddings
RESULTS=$OUT/results
LOG=$OUT/logs/benchmark_only.log
BENCH_PY=$RUN/env/github_protocol_py310/bin/python
MODEL_NAME=Txn_Jatin_OSDR_FT_dynamic

mkdir -p "$OUT/source" "$EMB_ROOT" "$RESULTS" "$OUT/logs"
exec > >(tee -a "$LOG") 2>&1

echo "[START_BENCHMARK_ONLY] $(date -u -Is)"

"$BENCH_PY" "$RUN/scripts/append_symbol_npz_embedding.py" \
  --source-npz "$FT_OUT/BRIDGE_OSDR_FT_dynamic__gene_benchmark__symbol.npz" \
  --source-output "$SOURCE_OUT" \
  --embeddings-root "$EMB_ROOT" \
  --intersection-genelist "$RUN/benchmarks/github_protocol_d132002_20260721_202549/embeddings/intersect/Txn_Jatin/Txn_Jatin_genelist.txt" \
  --model "$MODEL_NAME"

"$BENCH_PY" "$RUN/scripts/run_official_gene_level_only.py" \
  --python "$BENCH_PY" \
  --benchmark-repo "$RUN/references/gene-embedding-benchmarks" \
  --embeddings "$EMB_ROOT" \
  --output "$RESULTS" \
  --helper-dir "$RUN/scripts" \
  --models "$MODEL_NAME" \
  --parallel "${GITHUB_GENE_LEVEL_PARALLEL:-4}"

echo "[BENCHMARK_ONLY_COMPLETE] $(date -u -Is)"
touch "$OUT/ALL_COMPLETE"
