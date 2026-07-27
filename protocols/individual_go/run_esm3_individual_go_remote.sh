#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="${RUN_ROOT:-/media/volume/AdditionalHeadroom/Txn_Jatin_runs/full_ARCHES4_20epoch_H100_20260624_015238}"
TRAIN_ROOT="${TRAIN_ROOT:-/media/volume/TrainingData/txn_jatin_archs4}"
OUTPUT="${OUTPUT:-${RUN_ROOT}/benchmarks/individual_go_esm3_20260725}"
SCRIPT_ROOT="${RUN_ROOT}/scripts"
PARTIAL="${TRAIN_ROOT}/bridge_osdr_partial_unfreeze"
REFERENCE="${TRAIN_ROOT}/benchmark_outputs_txn_jatin_dynamic_vs_all/embeddings"
FULL_FT="${TRAIN_ROOT}/bridge_osdr_finetune"
MASTER="${PARTIAL}/baselines/BRIDGE_OSDR_original_dynamic__gene_benchmark__symbol.npz"
ESM2_PRIOR="${TRAIN_ROOT}/train_txn_jatin_full_tpm_v2/esm2_prior_512d.npz"
GO_LIBRARY="${RUN_ROOT}/inputs/go_bp.txt"
ESM3_ENV="${RUN_ROOT}/env/esm3_3_2_1_post1"

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  for candidate in \
    /media/volume/TrainingData/home_data/benchmarking_run/Benchmarking \
    /home/exouser/benchmarking_run/Benchmarking
  do
    if [[ -d "${candidate}/Txn_Jatin" ]]; then
      PROJECT_ROOT="${candidate}"
      break
    fi
  done
fi
: "${PROJECT_ROOT:?Unable to locate the remote Benchmarking project root}"

if [[ -z "${GENE_INFO:-}" ]]; then
  for candidate in \
    "${RUN_ROOT}/inputs/bulkformer_gene_info.csv" \
    "${PROJECT_ROOT}/external_models/BulkFormer/data/bulkformer_gene_info.csv" \
    /home/exouser/benchmarking_run/Benchmarking/external_models/BulkFormer/data/bulkformer_gene_info.csv
  do
    if [[ -s "${candidate}" ]]; then
      GENE_INFO="${candidate}"
      break
    fi
  done
fi
: "${GENE_INFO:?Unable to locate BulkFormer gene/protein sequence metadata}"
mkdir -p "${OUTPUT}/logs" "${OUTPUT}/embeddings" "${OUTPUT}/results" "${RUN_ROOT}/env"
exec > >(tee -a "${OUTPUT}/logs/pipeline.log") 2>&1

status_file="${OUTPUT}/RUNNING"
printf '%s\n' "$(date -u -Is)" > "${status_file}"
rm -f "${OUTPUT}/COMPLETE" "${OUTPUT}/FAILED"
trap 'code=$?; rm -f "${status_file}"; if (( code == 0 )); then date -u -Is > "${OUTPUT}/COMPLETE"; else printf "%s exit=%s\n" "$(date -u -Is)" "${code}" > "${OUTPUT}/FAILED"; fi' EXIT

echo "[START] $(date -u -Is)"
echo "[HOST] $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

for required in "${MASTER}" "${ESM2_PRIOR}" "${GENE_INFO}" "${GO_LIBRARY}" \
  "${PARTIAL}/benchmark/osdr_go_kegg_term_scores.csv"
do
  [[ -s "${required}" ]] || {
    echo "[ERROR] missing required input: ${required}" >&2
    exit 2
  }
done

if [[ ! -x "${ESM3_ENV}/bin/python" ]]; then
  echo "[ENV] creating isolated ESM3 environment"
  python3 -m venv "${ESM3_ENV}"
  "${ESM3_ENV}/bin/python" -m pip install --upgrade pip wheel
fi
if ! "${ESM3_ENV}/bin/python" -c 'import esm, httpx, matplotlib, torch, sklearn, pandas' 2>/dev/null; then
  echo "[ENV] installing pinned ESM3 runtime"
  "${ESM3_ENV}/bin/python" -m pip install \
    "esm==3.2.1.post1" hf_xet httpx matplotlib
fi
"${ESM3_ENV}/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("remote ESM3 environment cannot see the H100")
print(
    f"[CUDA] torch={torch.__version__} device={torch.cuda.get_device_name(0)} "
    f"bf16={torch.cuda.is_bf16_supported()}",
    flush=True,
)
PY

export HF_HOME="${RUN_ROOT}/tmp/huggingface"
export XDG_CACHE_HOME="${RUN_ROOT}/tmp/xdg_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}"

esm3_output="${OUTPUT}/embeddings/ESM3__gene_benchmark__symbol.npz"
if [[ -s "${esm3_output}" ]]; then
  echo "[EXTRACT_SKIP] $(date -u -Is) validated ESM3 artifact already exists"
else
  echo "[EXTRACT] $(date -u -Is) ESM3 sequence embeddings"
  "${ESM3_ENV}/bin/python" "${SCRIPT_ROOT}/extract_esm3_gene_embeddings.py" \
    --gene-info "${GENE_INFO}" \
    --master-npz "${MASTER}" \
    --output "${esm3_output}" \
    --device cuda \
    --token-budget 8192 \
    --max-batch-size 32 \
    --flush-every 10 \
    2>&1 | tee -a "${OUTPUT}/logs/esm3_extraction.log"
fi

echo "[PROBE] $(date -u -Is) individual GO terms for last1/last2/last3"
"${ESM3_ENV}/bin/python" "${SCRIPT_ROOT}/add_esm3_individual_go.py" \
  --project-root "${PROJECT_ROOT}" \
  --partial-output "${PARTIAL}" \
  --full-finetune-output "${FULL_FT}" \
  --reference-embedding-dir "${REFERENCE}" \
  --go-library "${GO_LIBRARY}" \
  --esm2-prior "${ESM2_PRIOR}" \
  --esm3-npz "${esm3_output}" \
  --output-dir "${OUTPUT}/results" \
  2>&1 | tee -a "${OUTPUT}/logs/individual_go_probe.log"

echo "[COMPLETE] $(date -u -Is)"
