#!/usr/bin/env bash
set -euo pipefail

BENCH="${1:?benchmark output directory is required}"
RUN="/media/volume/AdditionalHeadroom/Txn_Jatin_runs/full_ARCHES4_20epoch_H100_20260624_015238"
PROJECT="/media/volume/TrainingData/home_data/benchmarking_run/Benchmarking"
PYTHON="/media/volume/TrainingData/home_data/benchmarking_run/.venv/bin/python"
PIP="$PYTHON -m pip"
SCRIPTS="$RUN/scripts"
PACKAGES="$BENCH/python_packages_clean"

mkdir -p "$BENCH"/{config,masks,results,predictions,logs,status,resources}
mkdir -p "$PACKAGES"
export PYTHONPATH="$PACKAGES:$SCRIPTS:${PYTHONPATH:-}"
exec > >(tee -a "$BENCH/logs/setup.log") 2>&1

echo "[$(date --iso-8601=seconds)] Staging published BulkFormer resources"
if [[ ! -f "$BENCH/resources/G_tcga.pt" || "$(stat -c %s "$BENCH/resources/G_tcga.pt")" != "5276395" ]]; then
  curl --fail --location --retry 5 \
    --output "$BENCH/resources/G_tcga.pt" \
    "https://zenodo.org/api/records/15744294/files/G_tcga.pt/content"
fi
if [[ ! -f "$BENCH/resources/G_tcga_weight.pt" || "$(stat -c %s "$BENCH/resources/G_tcga_weight.pt")" != "1319659" ]]; then
  curl --fail --location --retry 5 \
    --output "$BENCH/resources/G_tcga_weight.pt" \
    "https://zenodo.org/api/records/15744294/files/G_tcga_weight.pt/content"
fi
ln -sfn \
  "/media/volume/TrainingData/txn_jatin_archs4/priors/esm2_feature_concat.pt" \
  "$BENCH/resources/esm2_feature_concat.pt"

echo "[$(date --iso-8601=seconds)] Installing missing inference/report dependencies"
$PIP install --no-input --upgrade --no-deps --target "$PACKAGES" \
  einops axial-positional-embedding hyper-connections local-attention \
  torch-einops-utils performer-pytorch torch-geometric xxhash
$PIP install --no-input --upgrade --no-deps --target "$PACKAGES" \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f "https://data.pyg.org/whl/torch-2.5.1+cu121.html"

echo "[$(date --iso-8601=seconds)] Freezing protocol and masks"
"$PYTHON" "$SCRIPTS/prepare_masked_benchmark.py" \
  --output "$BENCH" \
  --tcga "$PROJECT/tcga/tcga_tpm_unstranded_matrix.parquet" \
  --tcga-metadata "$PROJECT/tcga/tcga_metadata.parquet" \
  --tcga-sample-sheet "$PROJECT/tcga/gdc_sample_sheet.2025-11-16.tsv" \
  --osdr "/media/volume/TrainingData/txn_jatin_archs4/bridge_osdr_finetune/prepared_osdr_bridge_input.npz" \
  --txn-genes "$RUN/checkpoints/Txn_Jatin_full_ARCHES4_20ep_H100_20260624_015238/canonical_genes.csv" \
  --bridge-genes "/media/volume/TrainingData/txn_jatin_archs4/baselines/BRIDGE_s66qfh36/canonical_genes.csv" \
  --bulk-genes "$BENCH/resources/BulkFormer-main/data/bulkformer_gene_info.csv" \
  --txn-checkpoint "$RUN/checkpoints/Txn_Jatin_full_ARCHES4_20ep_H100_20260624_015238/best_model.pt" \
  --bridge-checkpoint "/media/volume/TrainingData/txn_jatin_archs4/baselines/BRIDGE_s66qfh36/best_model.pt" \
  --bulk-checkpoint-dir "$PROJECT/BulkFormer-main" \
  --bulkformer-root "$BENCH/resources/BulkFormer-main" \
  --train-flash "$PROJECT/RNA Walter/flash_osdr_model/train_flash.py" \
  --graph "$BENCH/resources/G_tcga.pt" \
  --graph-weights "$BENCH/resources/G_tcga_weight.pt" \
  --esm2 "$BENCH/resources/esm2_feature_concat.pt" \
  --existing-static-results "$RUN/benchmarks/github_protocol_d132002_20260721_202549" \
  --existing-sample-results "$PROJECT/Txn_Jatin/benchmark/tables/sample_scores.csv" \
  --seeds 20260723 20260724 20260725

"$PYTHON" -m py_compile \
  "$SCRIPTS/masked_benchmark_common.py" \
  "$SCRIPTS/prepare_masked_benchmark.py" \
  "$SCRIPTS/run_rf_baseline.py" \
  "$SCRIPTS/run_decoder_inference.py" \
  "$SCRIPTS/analyze_imputation.py" \
  "$SCRIPTS/run_masked_rf.py" \
  "$SCRIPTS/aggregate_masked_results.py"

echo "[$(date --iso-8601=seconds)] Setup complete"
