# FM for Transcriptomics: Txn_Jatin Study

Reproducibility package for Txn_Jatin, a biologically regularized
transcriptomic foundation model trained on ARCHS4. The repository connects the
training and evaluation protocols, measured result tables, scientific reports,
figures, and the ARCHS4 Imputation Studio.

Model checkpoints and full expression datasets are intentionally excluded.

## Contents

- [Study overview](#study-overview)
- [Repository map](#repository-map)
- [Start here](#start-here)
- [Experiment sequence](#experiment-sequence)
- [Studio and inference API](#studio-and-inference-api)
- [Reproducing the study](#reproducing-the-study)
- [Data and checkpoint requirements](#data-and-checkpoint-requirements)
- [Validation](#validation)
- [Scientific scope](#scientific-scope)

## Study overview

Txn_Jatin extends masked expression reconstruction with three biological
regularizers:

1. ESM2 sequence-prior alignment for gene identity.
2. Expression-conditioned contextual alignment.
3. Sample-level contrastive structure.

The study compares static and contextual Txn_Jatin representations against
BRIDGE, five BulkFormer sizes, ESM2, ESM3, Geneformer, and scGPT. It then tests
whole-gene recovery, cancer classification, OSDR parameter-efficient
fine-tuning, sparse-panel behavior, and exploratory immunotherapy transfer.

The central result is task-specific rather than universal: Txn_Jatin improves
several functional-ranking and TCGA expressed-gene endpoints, while BRIDGE
retains stronger global reconstruction geometry. Sequence controls led an
earlier 40-term linear-probe experiment, but Txn_Jatin led the official
56-term, TCGA-scoped GitHub GO protocol in both reported gene scopes.

## Repository map

```text
.
|-- app/
|   |-- backend/                 # HTTP API and lazy checkpoint runtime
|   |-- data/                    # UI-ready measured-results registry
|   `-- frontend/                # ARCHS4 Imputation Studio
|-- config/                      # Remote checkpoint path configuration
|-- docs/
|   |-- EXPERIMENT_ORDER.md      # Protocols in execution order
|   |-- MODELS_AND_CAPABILITIES.md
|   |-- RESULTS_INDEX.md         # Direct map from questions to result files
|   |-- RESULTS_INTERPRETATION.md
|   `-- figures/
|-- protocols/
|   |-- training/
|   |-- github_benchmark/
|   |-- whole_gene_mask/
|   |-- osdr_peft/
|   |-- individual_go/
|   |-- individual_go_tcga/
|   |-- sparse_biomarker/
|   `-- immunotherapy/
|-- results/                     # Measured CSV, JSON, reports, and figures
|-- scripts/                     # Build, validation, test-data, and deployment
|-- test_data/
|   `-- random_50_gene_panels/   # 50 UI inputs and matching truth matrices
|-- validation/                  # API smoke tests and UI screenshots
|-- requirements.txt
`-- SHA256SUMS.txt
```

## Start here

Choose the route matching your goal:

| Goal | Start with |
|---|---|
| Understand the scientific conclusions | [`docs/RESULTS_INTERPRETATION.md`](docs/RESULTS_INTERPRETATION.md) |
| Find a particular metric or figure | [`docs/RESULTS_INDEX.md`](docs/RESULTS_INDEX.md) |
| Follow experiments chronologically | [`docs/EXPERIMENT_ORDER.md`](docs/EXPERIMENT_ORDER.md) |
| Check which models support imputation | [`docs/MODELS_AND_CAPABILITIES.md`](docs/MODELS_AND_CAPABILITIES.md) |
| Inspect the interactive application | [`app/frontend/README.md`](app/frontend/README.md) |
| Verify repository integrity | [`scripts/validate_bundle.py`](scripts/validate_bundle.py) |

## Experiment sequence

1. **ARCHS4 pretraining:** masked reconstruction plus ESM2, contextual, and
   sample-contrastive objectives.
2. **Epoch-7 comparison:** frozen contextual Txn_Jatin versus original BRIDGE.
3. **Pinned GitHub protocol:** gene-level and gene-pair embedding benchmarks.
4. **Dynamic context:** expression-conditioned representation evaluation.
5. **Whole-gene masking:** the same 15% of genes hidden across TCGA or OSDR
   samples.
6. **Cancer classification:** patient-grouped random forests for 2-cancer and
   5-cancer endpoints.
7. **OSDR PEFT:** gradual unfreezing, BitFit/Norm, and LoRA selected using a
   predeclared MSE/AUROC combined rank.
8. **OSDR individual-GO probe plus ESM3:** an earlier 40-term linear-probe
   comparison retained as a separate experiment.
9. **Sparse biomarker pilot:** 1,000- and 2,000-gene observed panels.
10. **Immunotherapy transfer:** exploratory four-cohort leave-one-cohort-out
    response analysis.
11. **Official TCGA-scoped individual GO:** all 56 terms and all 12 models
    evaluated with the fixed folds, nested-CV SVC, and holdout metrics from
    pinned `ylaboratory/gene-embedding-benchmarks` commit
    `d1320026a2a4ee033d49517f91e2d1c2ccc8df1e`.

Exact protocols and primary outputs are linked in
[`docs/EXPERIMENT_ORDER.md`](docs/EXPERIMENT_ORDER.md).

## Studio and inference API

Install Python 3.10+ dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Start the integrated results UI and API:

```bash
python app/backend/server.py --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. Result exploration works from the packaged
registry. Live inference additionally requires compatible checkpoints and the
external model code described below.

The deployed H100 Studio is available at
[`http://nvidea-h100-2-of-2.bio260281.projects.jetstream-cloud.org:8000/`](http://nvidea-h100-2-of-2.bio260281.projects.jetstream-cloud.org:8000/).
Use the hostname rather than the raw public IP; some institutional web filters
block uncategorized IP-address URLs even when the service itself is healthy.

The service exposes:

- `GET /api/health`
- `GET /api/models`
- `GET /api/experiments`
- `POST /api/impute`
- `GET /results/...`
- `GET /test-data/...`

The H100 runtime loads one decoder at a time and unloads the previous
checkpoint when models are switched. Embedding-only models return an explicit
unsupported response and `NaN`; the UI never substitutes a silent surrogate.

## Reproducing the study

There are three reproducibility levels:

### 1. Audit the reported outputs

All publication-facing result tables are under [`results/`](results/).
Checksums are recorded in [`SHA256SUMS.txt`](SHA256SUMS.txt).

### 2. Rebuild the UI registry

```powershell
python scripts/build_results_registry.py `
  --workspace C:\path\to\Benchmarking `
  --repo C:\path\to\this-repository
```

This assembles existing measured outputs. It does not retrain models or invent
missing values.

### 3. Rerun model experiments

1. Acquire the external datasets and checkpoints.
2. Update `config/model_paths.remote.json`.
3. Run the scripts in [`protocols/`](protocols/) following
   [`docs/EXPERIMENT_ORDER.md`](docs/EXPERIMENT_ORDER.md).
4. Rebuild the result registry.
5. Run repository validation and compare checksums.

The protocol scripts preserve the actual execution order and are grouped by
scientific question rather than by machine or date.

## UI test matrices

[`test_data/random_50_gene_panels/`](test_data/random_50_gene_panels/) contains
50 deterministic TCGA-derived upload files. Every input has:

- 50 shared genes;
- 8 samples;
- 15% randomly masked cells;
- a matching `_truth.csv` matrix with the original raw TPM values.

Regenerate them with:

```powershell
python scripts/generate_ui_test_matrices.py `
  --tcga C:\path\to\tcga_tpm_unstranded_matrix.parquet `
  --genes C:\path\to\shared_txn_tcga_genes.csv `
  --output test_data\random_50_gene_panels
```

These matrices test parsing, gene alignment, model switching, and endpoint
behavior. They are not accuracy benchmarks because 50 genes supply less than
1% of a decoder vocabulary.

## Data and checkpoint requirements

The full workflows use:

- ARCHS4 expression data for pretraining;
- processed TCGA and OSDR matrices for masking and transfer;
- external immunotherapy cohorts for exploratory response analysis;
- Txn_Jatin, BRIDGE, BulkFormer, ESM2/ESM3, Geneformer, and scGPT artifacts.

These files are not redistributed because of size, licensing, and provenance
requirements. The repository contains only small derived UI test matrices.
`config/model_paths.remote.json` contains path mappings, never credentials or
weights.

## Remote deployment

Set the SSH credential only in the process environment:

```powershell
$env:REMOTE_PASS = "<remote passphrase>"
.\scripts\deploy_remote.ps1
```

Do not commit credentials, downloaded datasets, or checkpoints.

## Validation

```bash
python scripts/validate_bundle.py
```

Validation checks the model registry, experiment registry, 50 test matrices,
and absence of checkpoint/data artifacts. It fails if PT, PTH, Safetensors,
NPZ, Parquet, or HDF5 files enter the repository.

The packaged validation evidence includes:

- remote API smoke results for all decoder-capable models;
- explicit unsupported responses for embedding-only controls;
- desktop and mobile screenshots;
- a remote H100 end-to-end UI test.

## Scientific scope

All displayed benchmark values come from included measured result tables.
Sparse-panel, cancer, and immunotherapy analyses remain exploratory. They do
not establish a clinically validated biomarker or prospective treatment
response predictor.

The publication-safe conclusion is that Txn_Jatin is a biologically regularized
expression representation with task-specific gains, not a universal
replacement for BRIDGE or protein-sequence foundation models.
