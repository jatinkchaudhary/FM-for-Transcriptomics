# ESPRESSO Downstream Benchmark Suite

This directory contains the experimental designs, executable analysis code,
measured outputs, figures, and scientific reports for four downstream tests of
ESPRESSO, the final 20-epoch `Txn_Jatin` model. The package preserves qualified
and negative findings as well as positive results.

Datasets, checkpoints, credentials, generated representation arrays, and
duplicate remote working directories are intentionally excluded. The reported
results remain auditable through fold-level or sample-level predictions,
protocol JSON files, provenance tables, and fixed analysis scripts.

## Results At A Glance

| Experiment | Primary endpoint | ESPRESSO result | Main comparison | Interpretation |
|---|---|---:|---|---|
| DepMap essentiality | Gene-disjoint AUPRC | **0.743** (0.724-0.762) | BRIDGE 0.702; BulkFormer-147M 0.562; ESM2 0.601 | Strong common-essential gene signal; AUROC tied BRIDGE but AUPRC was higher |
| Cross-species transfer | Mouse macro-F1 | **0.9419** (0.9222-0.9599) | BulkFormer-147M 0.7970; raw PCA-64 0.6587 | Human-trained tissue probe transferred to mouse without mouse-label fitting |
| TCGA survival | Pan-cancer OS C-index | **0.6747** (0.6564-0.6928) | BulkFormer-147M 0.6476; raw 0.6778; PCA-64 0.6849 | Better than BulkFormer for OS, but statistically tied simple expression baselines |
| Cross-study integration | Prespecified combined score | **0.544** | raw 0.531; Harmony 0.525; BulkFormer-147M 0.503 | Best biology preservation, but weaker study mixing than Harmony/PCA |

Intervals are 95% confidence intervals. Read the experiment reports before
using these values as claims; each report documents scope and limitations.

## Experiment Index

### 1. DepMap Common Essentiality

**Question:** Do frozen gene embeddings encode common-essential gene status?

**Design:** DepMap Public 24Q2 labels; strict 13,342-gene intersection across
models; 1,666 positives and 11,676 pan-neutral negatives; five gene-disjoint
outer folds; four-fold inner regularization selection; class-weighted linear
probe; 2,000 paired label-stratified bootstrap replicates.

- [Detailed report](results/essentiality/REPORT.md)
- [Protocol](results/essentiality/protocol.json)
- [Primary metrics](results/essentiality/essentiality_metrics.csv)
- [Paired deltas](results/essentiality/paired_deltas_vs_espresso.csv)
- [Out-of-fold predictions](results/essentiality/out_of_fold_predictions.csv)
- [Representation provenance](results/essentiality/representation_provenance.csv)
- [Naive-baseline provenance](results/essentiality/naive_baseline_provenance.csv)
- [Figure (PNG)](results/essentiality/essentiality_comparison.png)
- [Figure (PDF)](results/essentiality/essentiality_comparison.pdf)
- [Main analysis](essentiality/run_essentiality.py)
- [Naive baselines](essentiality/add_naive_baselines.py)
- [Figure generation](essentiality/plot_essentiality.py)

### 2. Human-To-Mouse Tissue Transfer

**Question:** Can a probe trained only on human tissue labels transfer through
the shared ortholog representation to an independent mouse cohort?

**Design:** 1,841 human GTEx samples for training; 547 Tabula Muris Senis bulk
samples for testing; 10 matched tissues; 15,556 one-to-one orthologs; human-only
hyperparameter selection; 2,000 mouse-tissue-stratified bootstrap replicates.
The exact ARCHS4 accession overlap was zero.

- [Detailed report](results/cross_species/REPORT.md)
- [Protocol](results/cross_species/protocol.json)
- [Primary metrics](results/cross_species/crossspecies_transfer.csv)
- [Paired deltas](results/cross_species/paired_deltas_vs_espresso.csv)
- [Mouse predictions](results/cross_species/mouse_predictions.csv)
- [Shared ortholog list](results/cross_species/shared_one_to_one_orthologs.csv)
- [ESPRESSO confusion matrix](results/cross_species/confusion_ESPRESSO.csv)
- [BulkFormer confusion matrix](results/cross_species/confusion_BulkFormer_147M.csv)
- [Raw-baseline confusion matrix](results/cross_species/confusion_Raw_ortholog_PCA64.csv)
- [ESPRESSO UMAP coordinates](results/cross_species/umap_ESPRESSO.csv)
- [Raw-baseline UMAP coordinates](results/cross_species/umap_Raw_ortholog_PCA64.csv)
- [Figure (PNG)](results/cross_species/crossspecies_umap.png)
- [Figure (PDF)](results/cross_species/crossspecies_umap.pdf)
- [Analysis](cross_species/run_cross_species.py)

### 3. TCGA Survival And Progression

**Question:** Do sample representations predict overall survival (OS) and
progression-free interval (PFI) in TCGA?

**Design:** One primary-tumor sample per patient; TCGA Clinical Data Resource
outcomes; BRCA, KIRC, LUAD, LUSC, and SKCM; five patient-disjoint outer folds;
three-fold inner Coxnet tuning; out-of-fold Harrell C-index; 2,000
event-stratified metric bootstraps and 1,000 paired delta bootstraps.

- [Detailed report](results/survival/REPORT.md)
- [Protocol](results/survival/protocol.json)
- [C-index results](results/survival/survival_cindex.csv)
- [Paired deltas](results/survival/paired_delta_vs_espresso.csv)
- [Out-of-fold predictions](results/survival/oof_predictions.csv)
- [Forest plot (PNG)](results/survival/survival_forest.png)
- [Forest plot (PDF)](results/survival/survival_forest.pdf)
- [Analysis](survival/run_survival.py)

### 4. Cross-Study Bulk Integration

**Question:** Does the representation preserve tissue biology while reducing
study effects across GTEx and TCGA?

**Design:** Crossed 4-tissue by 2-study recount2 cohort; 1,339 samples; primary
TCGA tumors; at most 200 samples per tissue-study cell; fixed neighborhood,
normalization, balancing, and metric definitions across methods. This is a
documented recount2 substitution, not a recount3 analysis.

- [Detailed report](results/integration/REPORT.md)
- [Protocol](results/integration/protocol.json)
- [Integration metrics](results/integration/integration_metrics.csv)
- [Paired deltas](results/integration/paired_delta_vs_espresso.csv)
- [Cohort metadata](results/integration/cohort_metadata.csv)
- [Tissue-study contingency](results/integration/tissue_study_contingency.csv)
- [UMAP coordinates](results/integration/umap_coordinates.csv)
- [Figure (PNG)](results/integration/integration_umaps.png)
- [Figure (PDF)](results/integration/integration_umaps.pdf)
- [Main analysis](integration/run_integration.py)
- [Neighborhood bootstrap](integration/bootstrap_deltas.py)

## Shared Validity Gate

Only sample adapters that passed the frozen-representation validity gate were
eligible for survival and integration comparisons.

- [Validity-gate report](results/validity_gate/REPORT.md)
- [Validity-gate table](results/validity_gate/validity_gate.csv)
- [Validity-gate implementation](shared/run_validity_gate.py)
- [Sample adapters](shared/sample_adapters.py)
- [Bootstrap utilities](shared/statistics.py)
- [Leakage utilities](shared/leakage.py)
- [Provenance utilities](shared/provenance.py)

The cross-experiment scientific synthesis is in
[`FINAL_REPORT.md`](FINAL_REPORT.md). The original run validation marker is in
[`VALIDATION.txt`](VALIDATION.txt).

## Reproduction

Use Python 3.10+ in the repository environment. The scripts require NumPy,
pandas, scikit-learn, SciPy, Matplotlib, h5py, UMAP, NetworkX, harmonypy, and
scikit-survival; model adapters additionally require the corresponding model
runtime and PyTorch.

Run scripts from this directory so local imports resolve:

```bash
python essentiality/run_essentiality.py --help
python cross_species/run_cross_species.py --help
python shared/run_validity_gate.py --help
python survival/run_survival.py --help
python integration/run_integration.py --help
```

Each command exposes explicit input and output arguments. Required external
inputs include DepMap 24Q2, GTEx and TCGA recount2 matrices, TCGA-CDR outcomes,
Tabula Muris Senis bulk expression and metadata, one-to-one ortholog mappings,
model checkpoints, and generated frozen representations. These are not
redistributed because of size, licensing, and provenance constraints.

## Repository Hygiene

This package deliberately excludes:

- model checkpoints and raw or processed expression datasets;
- `.npz` representation caches that can be regenerated from the model runtime;
- the 5.3 MB ARCHS4 accession working manifest;
- duplicate `remote_sync` outputs;
- `__pycache__`, logs, temporary files, and empty diagnostics.

The accession-overlap result needed to interpret cross-species leakage is
reported in the cross-species report. Result CSVs and protocol JSON files are
the source of truth for all displayed values.
