# Results index

This page maps each scientific question to its primary measured artifacts.
Experiment definitions and execution order are documented in
[`EXPERIMENT_ORDER.md`](EXPERIMENT_ORDER.md).
For the complete script/output archive, including detailed per-run files, see
[`BENCHMARK_ARTIFACTS.md`](BENCHMARK_ARTIFACTS.md) and
[`results/ARTIFACT_INVENTORY.csv`](../results/ARTIFACT_INVENTORY.csv).

## Paper diagrams

**Question:** Where are the publication-ready architecture, pipeline, and
benchmark overview figures?

- [`paper_figures/README.md`](paper_figures/README.md)
- [`paper_figures/figure_01_bridge_architecture.pdf`](paper_figures/figure_01_bridge_architecture.pdf)
- [`paper_figures/figure_02_txn_jatin_architecture.pdf`](paper_figures/figure_02_txn_jatin_architecture.pdf)
- [`paper_figures/figure_03_txn_contextual_construction.pdf`](paper_figures/figure_03_txn_contextual_construction.pdf)
- [`paper_figures/figure_04_experimental_pipeline.pdf`](paper_figures/figure_04_experimental_pipeline.pdf)
- [`paper_figures/figure_05_benchmark_landscape.pdf`](paper_figures/figure_05_benchmark_landscape.pdf)

## Model training

**Question:** Did Txn_Jatin optimize stably over 20 ARCHS4 epochs?

- [`results/training/loss_history.csv`](../results/training/loss_history.csv)

## Txn_Jatin versus BRIDGE

**Question:** How did the early Txn_Jatin representation differ from BRIDGE?

- [`results/epoch7_comparison/comparison_report.md`](../results/epoch7_comparison/comparison_report.md)
- [`figures/txn_jatin_architecture.png`](figures/txn_jatin_architecture.png)
- [`figures/txn_jatin_vs_bridge_architecture.png`](figures/txn_jatin_vs_bridge_architecture.png)

## Pinned GitHub embedding protocol

**Question:** How do the static gene representations compare across models?

- Long table:
  [`results/github_protocol/all_results_long_numeric.csv`](../results/github_protocol/all_results_long_numeric.csv)
- Wide comparison table:
  [`results/github_protocol/all_results_wide_numeric.csv`](../results/github_protocol/all_results_wide_numeric.csv)

The long table is the canonical machine-readable output. It retains the
benchmark family, scope, pair operator, model, AUROC, AUPRC, and term/pair
counts.

## Dynamic contextual representations

**Question:** Does expression-conditioned context improve downstream tasks?

- [`results/dynamic_context/dynamic_full_track_summary.csv`](../results/dynamic_context/dynamic_full_track_summary.csv)
- [`results/dynamic_context/dynamic_overall_ranking.csv`](../results/dynamic_context/dynamic_overall_ranking.csv)
- [`results/dynamic_context/dynamic_story_dashboard.png`](../results/dynamic_context/dynamic_story_dashboard.png)

## Immune regulatory and synthetic-lethal edges

**Question:** Does Txn_Jatin recover cold-gene immune relationships that an
independent co-expression baseline misses?

- [`results/immune_edge_recovery/REPORT.md`](../results/immune_edge_recovery/REPORT.md)
- [`results/immune_edge_recovery/summary_metrics.csv`](../results/immune_edge_recovery/summary_metrics.csv)
- [`results/immune_edge_recovery/paired_bootstrap_vs_coexpression.csv`](../results/immune_edge_recovery/paired_bootstrap_vs_coexpression.csv)
- [`results/immune_edge_recovery/immune_edge_recovery.png`](../results/immune_edge_recovery/immune_edge_recovery.png)

## Whole-gene masking

**Question:** Can a model recover a gene hidden across all samples?

- [`results/whole_gene_mask/imputation_all_models.csv`](../results/whole_gene_mask/imputation_all_models.csv)
- [`results/whole_gene_mask/unsupported_imputation_rows.csv`](../results/whole_gene_mask/unsupported_imputation_rows.csv)
- [`results/whole_gene_mask/protocol.json`](../results/whole_gene_mask/protocol.json)

Embedding-only controls are reported as unsupported/`NaN`; they are not assigned
synthetic decoder outputs.

## Cancer classification

**Question:** Does imputation preserve 2-cancer and 5-cancer classification
signal?

- Masked/imputed models:
  [`results/cancer_rf/masked_rf_all_models.csv`](../results/cancer_rf/masked_rf_all_models.csv)
- Raw-expression reference:
  [`results/cancer_rf/raw_rf_baseline.csv`](../results/cancer_rf/raw_rf_baseline.csv)

## OSDR parameter-efficient fine-tuning

**Question:** Which limited-unfreezing strategy best balances reconstruction
MSE and biological AUROC?

- [`results/osdr_peft/report.md`](../results/osdr_peft/report.md)
- [`results/osdr_peft/candidate_summary.csv`](../results/osdr_peft/candidate_summary.csv)

## Earlier OSDR individual-GO probe

**Question:** Which models captured individual functional annotations under the
earlier 40-term OSDR linear-probe design?

- Model-level summary:
  [`results/individual_go/osdr_go_model_summary_with_esm3.csv`](../results/individual_go/osdr_go_model_summary_with_esm3.csv)
- Term-level scores:
  [`results/individual_go/osdr_go_term_scores_with_esm3.csv`](../results/individual_go/osdr_go_term_scores_with_esm3.csv)
- Run metadata:
  [`results/individual_go/manifest.json`](../results/individual_go/manifest.json)

This archived experiment is separate from the supplied GitHub protocol.

## Official TCGA-scoped individual GO

**Question:** How do all models, including native ESM3, compare on TCGA genes
using the supplied GitHub protocol?

- Full named 1,344-row term table:
  [`results/individual_go_tcga/tables/individual_go_term_scores.csv`](../results/individual_go_tcga/tables/individual_go_term_scores.csv)
- Model ranking:
  [`results/individual_go_tcga/tables/model_summary.csv`](../results/individual_go_tcga/tables/model_summary.csv)
- Winner for each GO term:
  [`results/individual_go_tcga/tables/per_term_winners.csv`](../results/individual_go_tcga/tables/per_term_winners.csv)
- Per-model winner counts:
  [`results/individual_go_tcga/tables/winner_counts.csv`](../results/individual_go_tcga/tables/winner_counts.csv)
- Official GO ID/name map:
  [`results/individual_go_tcga/tables/go_term_names.csv`](../results/individual_go_tcga/tables/go_term_names.csv)
- Protocol and validation report:
  [`results/individual_go_tcga/reports/REPORT.md`](../results/individual_go_tcga/reports/REPORT.md)

This run is pinned to `ylaboratory/gene-embedding-benchmarks` commit
`d1320026a2a4ee033d49517f91e2d1c2ccc8df1e`. TCGA defines the eligible gene
universe; GO remains a gene-level label, so TCGA expression values are not
substituted for the official GO folds.

## Sparse biomarker feasibility

**Question:** Is the checkpoint usable when only 1,000 or 2,000 genes are
observed?

- [`results/sparse_biomarker/report.md`](../results/sparse_biomarker/report.md)
- [`results/sparse_biomarker/panel_1000/`](../results/sparse_biomarker/panel_1000/)
- [`results/sparse_biomarker/panel_2000/`](../results/sparse_biomarker/panel_2000/)

## Immunotherapy transfer

**Question:** Do Txn_Jatin-derived features transfer across independent
immunotherapy cohorts?

- Scientific report:
  [`results/immunotherapy/report.md`](../results/immunotherapy/report.md)
- Leave-one-cohort-out summary:
  [`results/immunotherapy/loco_summary.csv`](../results/immunotherapy/loco_summary.csv)
- Raw-expression paired comparison:
  [`results/immunotherapy/paired_delta_vs_raw.csv`](../results/immunotherapy/paired_delta_vs_raw.csv)
- Hallmark meta-analysis:
  [`results/immunotherapy/hallmark_random_effects_meta.csv`](../results/immunotherapy/hallmark_random_effects_meta.csv)
- Figures:
  [`results/immunotherapy/figures/`](../results/immunotherapy/figures/)

These results are exploratory and do not constitute a validated clinical
response predictor.

### Atlas-augmented immunotherapy follow-up

- Report:
  [`results/immunotherapy/atlas_augmented/REPORT.md`](../results/immunotherapy/atlas_augmented/REPORT.md)
- LOCO summary:
  [`results/immunotherapy/atlas_augmented/atlas_loco_summary.csv`](../results/immunotherapy/atlas_augmented/atlas_loco_summary.csv)
- Pathway consistency:
  [`results/immunotherapy/atlas_augmented/atlas_pathway_consistency.csv`](../results/immunotherapy/atlas_augmented/atlas_pathway_consistency.csv)
- Exact LLM prompt:
  [`results/immunotherapy/atlas_augmented/llm_prompt.txt`](../results/immunotherapy/atlas_augmented/llm_prompt.txt)

The atlas improved raw-model calibration but did not improve macro
held-out-cohort AUROC or AUPRC.

### Multi-model immunotherapy benchmark

- Report:
  [`results/immunotherapy/multimodel/REPORT.md`](../results/immunotherapy/multimodel/REPORT.md)
- Model summary:
  [`results/immunotherapy/multimodel/multimodel_loco_summary.csv`](../results/immunotherapy/multimodel/multimodel_loco_summary.csv)
- Paired uncertainty:
  [`results/immunotherapy/multimodel/paired_bootstrap_vs_raw.csv`](../results/immunotherapy/multimodel/paired_bootstrap_vs_raw.csv)
- Comparison figure:
  [`results/immunotherapy/multimodel/multimodel_primary_comparison.png`](../results/immunotherapy/multimodel/multimodel_primary_comparison.png)

## External GTEx whole-gene recovery

**Question:** Do decoder-capable models recover completely hidden genes in an
independent recount2 GTEx matrix?

- Model summary:
  [`results/gtex_external/gtex_all_model_summary.csv`](../results/gtex_external/gtex_all_model_summary.csv)
- Per-sample metrics:
  [`results/gtex_external/gtex_all_model_sample_metrics.csv`](../results/gtex_external/gtex_all_model_sample_metrics.csv)
- Comparison figure:
  [`results/gtex_external/gtex_all_model_comparison.png`](../results/gtex_external/gtex_all_model_comparison.png)
- Protocol manifest:
  [`results/gtex_external/manifest.json`](../results/gtex_external/manifest.json)

All decoder-capable models were scored on the identical 2,117-gene masked
intersection. Embedding-only Txn_Jatin contextual, ESM2, ESM3, Geneformer, and
scGPT representations remain explicitly unsupported for expression imputation.

## Interactive registry

The frontend reads [`app/data/results_registry.json`](../app/data/results_registry.json).
It packages the same measured values into a chronological interactive view.
