#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/media/volume/AdditionalHeadroom/Txn_Jatin_runs/full_ARCHES4_20epoch_H100_20260624_015238"
PROJECT="/media/volume/TrainingData/home_data/benchmarking_run/Benchmarking"
PYTHON="/media/volume/TrainingData/home_data/benchmarking_run/.venv/bin/python"
BENCHMARK_REPO="${RUN_ROOT}/references/gene-embedding-benchmarks"
SOURCE="${RUN_ROOT}/benchmarks/github_protocol_d132002_20260721_202549/embeddings"
OUTPUT="${RUN_ROOT}/benchmarks/individual_go_tcga_github_d132002_20260727_v2"
SCRIPTS="${RUN_ROOT}/scripts"
TCGA="${PROJECT}/tcga/tcga_tpm_unstranded_matrix.parquet"
PREPARED="${OUTPUT}/prepared_embeddings"
LOGS="${OUTPUT}/logs"

mkdir -p "${LOGS}" "${OUTPUT}/env"

{
  echo "started_utc=$(date -u -Is)"
  echo "hostname=$(hostname)"
  echo "benchmark_commit=$(git -C "${BENCHMARK_REPO}" rev-parse HEAD)"
  echo "python=$("${PYTHON}" --version 2>&1)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} | tee "${OUTPUT}/run_environment.txt"
"${PYTHON}" -m pip freeze > "${OUTPUT}/env/pip_freeze.txt"

"${PYTHON}" "${SCRIPTS}/prepare_tcga_go_embeddings.py" \
  --source-embeddings "${SOURCE}" \
  --tcga-parquet "${TCGA}" \
  --symbol-map "${SOURCE}/symbol_to_entrez_mygene.csv" \
  --esm3-npz "${RUN_ROOT}/benchmarks/individual_go_esm3_20260725/embeddings/ESM3__gene_benchmark__symbol.npz" \
  --output-dir "${PREPARED}" \
  2>&1 | tee "${LOGS}/embedding_preparation.log"

"${PYTHON}" "${SCRIPTS}/run_tcga_individual_go.py" \
  --python "${PYTHON}" \
  --benchmark-repo "${BENCHMARK_REPO}" \
  --prepared-dir "${PREPARED}" \
  --output-dir "${OUTPUT}" \
  --workers 12 \
  2>&1 | tee "${LOGS}/official_go.log"

echo "completed_utc=$(date -u -Is)" | tee -a "${OUTPUT}/run_environment.txt"

(
  cd "${OUTPUT}"
  find . -type f ! -name 'SHA256SUMS.txt*' -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS.txt.tmp
  mv SHA256SUMS.txt.tmp SHA256SUMS.txt
  sha256sum -c SHA256SUMS.txt >/dev/null
)
echo "checksums=verified"
