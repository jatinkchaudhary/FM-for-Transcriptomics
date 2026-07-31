# Locked External Immunotherapy Evaluation Preregistration

**Frozen before outcome access:** 2026-07-30

## Locked Cohort

The single external evaluation cohort is the Mariathasan et al. 2018
IMvigor210 bladder-cancer anti-PD-L1 cohort distributed as Bioconductor
ExperimentHub resource `EH6677` through `easierData`.

The `easierData` source package and metadata may be downloaded and inspected to
identify the resource. Expression values and clinical response outcomes must
not be opened until this document is hashed on the remote execution host.

## Final Pipeline

The selected representation is final 20-epoch `Txn_Jatin` completed expression.
This selection was made using only Gide, Hugo, Riaz, and Rose:

1. Map symbols to the fixed 16,055-gene Txn_Jatin vocabulary.
2. Convert source TPM to `log1p(TPM)`.
3. Preserve observed genes and impute assay-missing vocabulary genes with the
   frozen final Txn_Jatin decoder.
4. Select the 1,000 highest-variance genes using the four development cohorts
   only.
5. Fit scaling and PCA-64 using development samples only.
6. Apply diagonal training-domain CORAL, aligning each development cohort to
   pooled development moments. No external-cohort statistics enter this fit.
7. Select logistic-regression `C` from `[0.01, 0.1, 1, 10]` by
   leave-one-development-cohort-out AUROC.
8. Fit an intercept-only monotone probability calibration from development
   out-of-fold predictions.
9. Refit the classifier on all development samples and score the locked cohort
   once.

## Endpoint And Metrics

- Response positive: complete or partial response.
- Response negative: stable or progressive disease.
- Primary metrics: external-cohort AUROC and AUPRC.
- Secondary metrics: Brier score and calibration intercept.
- Uncertainty: 2,000 response-stratified bootstrap replicates.
- Raw measured expression under the identical development-only pipeline is a
  prespecified comparator.
- No hyperparameter, feature, threshold, signature, or adaptation choice may be
  changed after outcomes are opened.

## Leakage Check

Available IMvigor210 sample accessions will be matched exactly against the human
ARCHS4 pretraining manifest before evaluation. Any overlap will be reported and
an overlap-clean sensitivity estimate added when both response classes remain.

## Interpretation

This is a single-shot external test. A positive result requires useful
discrimination and calibration on IMvigor210 without post-access tuning.
Failure or a null comparison will be retained and reported without replacing
the cohort or modifying the pipeline.
