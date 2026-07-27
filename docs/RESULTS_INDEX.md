# Results index

This page maps each scientific question to its primary measured artifacts.
Experiment definitions and execution order are documented in
[`EXPERIMENT_ORDER.md`](EXPERIMENT_ORDER.md).

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

## Individual GO terms and ESM3

**Question:** Which models capture individual functional annotations?

- Model-level summary:
  [`results/individual_go/osdr_go_model_summary_with_esm3.csv`](../results/individual_go/osdr_go_model_summary_with_esm3.csv)
- Term-level scores:
  [`results/individual_go/osdr_go_term_scores_with_esm3.csv`](../results/individual_go/osdr_go_term_scores_with_esm3.csv)
- Run metadata:
  [`results/individual_go/manifest.json`](../results/individual_go/manifest.json)

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

## Interactive registry

The frontend reads [`app/data/results_registry.json`](../app/data/results_registry.json).
It packages the same measured values into a chronological interactive view.
