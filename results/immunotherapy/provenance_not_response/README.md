# Immunotherapy Provenance Study

This folder contains the complete, reportable output of `immunotherapy_provenance_todo.md`. The analysis was run on the remote H100 and synchronized locally on 30 July 2026.

## Result

The strongest positive result is targeted missing-gene recovery: Txn_Jatin reached Pearson **0.9522** and Spearman **0.9428** over 43 held-out immune-signature genes. The locked IMvigor210 clinical test was neutral: raw AUROC **0.7032** versus Txn_Jatin **0.6944**, with paired confidence intervals crossing zero. Read [REPORT.md](REPORT.md) for the full scientific interpretation and limitations.

## Completed order

1. Existing eight-decoder LOCO null retained as Phase 0.
2. Fixed-signature LOCO and permutation floor.
3. Provenance versus response variance partition.
4. Whole-panel signature-gene holdout and harmonization test.
5. Cohort-invariant adaptation screen: CORAL, project-out, DANN.
6. Preregistered, locked IMvigor210 external evaluation.

## Important protocol rule

The external response labels were not used to select features, adaptation, regularization, or calibration. The preregistration was hashed before outcome access. See:

- `LOCKED_EXTERNAL_PREREGISTRATION.md`
- `PREREGISTRATION_SHA256.txt`
- `PREREGISTRATION_FROZEN_UTC.txt`
- `locked_imvigor210_protocol.json`

## Files

All numerical outputs and figures are in `results/`. The most important files are:

- `loco_summary_with_ci.csv`
- `permutation_floor.csv`
- `pvca_variance_partition.csv`
- `signature_gene_holdout_accuracy.csv`
- `harmonized_vs_assay_limited_deltas.csv`
- `invariance_loco_summary_with_ci.csv`
- `invariance_paired_deltas.csv`
- `locked_imvigor210_summary.csv`
- `locked_imvigor210_bootstrap.csv`
- `provenance_vs_response_variance.png`
- `same_embedding_cohort_vs_response.png`

The executable protocol scripts are supplied in the companion reproducibility repository under `protocols/immunotherapy/`. Raw datasets, model files, and downloaded caches are intentionally excluded.
