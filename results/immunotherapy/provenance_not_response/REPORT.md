# Immunotherapy Provenance And Response Report

**Status:** complete for the preregistered analysis plan

**Execution date:** 30 July 2026

**Execution hardware:** remote NVIDIA H100; CPU preprocessing and report generation were run in the same reproducibility environment

## Executive Summary

This study tested whether Txn_Jatin contains response-relevant biology that is not explained by cohort provenance, assay availability, or ordinary expression geometry. The analysis was deliberately staged: existing multi-model results were retained as Phase 0, then the new analyses were executed without using immunotherapy outcomes to choose the external-test pipeline.

The main conclusions are:

1. On the four development cohorts, all tested representations were close to the raw-expression baseline. The final Txn_Jatin representation did not establish a universal improvement in leave-one-cohort-out (LOCO) response prediction.
2. The strongest new capability was accurate targeted recovery of masked immune-signature genes. Txn_Jatin achieved Pearson correlation **0.9522** and Spearman correlation **0.9428** over 43 held-out signature genes, versus **0.4638** and **0.5056** for the available BulkFormer147M comparison.
3. Imputation did not materially improve response transfer after harmonization. The result supports assay harmonization as a technical capability, but not yet as evidence of improved immunotherapy-response prediction.
4. Cohort/provenance explained approximately **64-68%** of weighted representation variance, while response explained approximately **0.3%** after cohort adjustment. Cancer and cohort were perfectly nested in these data, so their independent effects cannot be separated.
5. Training-domain diagonal CORAL was the best adaptation screen for Txn_Jatin: AUROC **0.6750**, compared with **0.6337** for the unadapted rerun. The paired confidence interval crossed zero, so this is encouraging but not conclusive.
6. On the locked external IMvigor210 cohort, raw expression had AUROC **0.7032** and Txn_Jatin had AUROC **0.6944**. Txn_Jatin had slightly higher AUPRC (**0.2925** vs **0.2775**), but all paired uncertainty intervals crossed zero and Txn_Jatin had a worse Brier score. The external test is therefore a neutral result, not evidence of a significant Txn_Jatin advantage.

The scientifically defensible claim is that Txn_Jatin provides strong gene-level biological imputation and a plausible provenance-aware representation, while response prediction remains limited by small, confounded cohorts and does not yet demonstrate a clinical performance advantage over raw expression.

## 1. Study Question And Design

The TODO was designed to separate four explanations for an apparent immunotherapy signal:

- **Response biology:** the representation captures information associated with response.
- **Provenance:** the representation mostly identifies study, assay, or batch.
- **Assay availability:** the representation appears better only because different cohorts measure different genes.
- **Expression geometry:** raw expression or PCA already contains the signal.

The execution order was:

| Phase | Question | Output |
|---|---|---|
| 0 | What did the existing eight-decoder LOCO benchmark show? | Frozen historical null table |
| 1 | What are the signature ceilings and the permutation floor? | LOCO metrics, bootstrap CIs, permutation floor |
| 2 | How much variance is provenance versus response? | PVCA-style partition and nuisance diagnostics |
| 3 | Can the model recover genes missing from an assay, and does harmonization help? | Targeted holdout accuracy and paired deltas |
| 4 | Can training-domain adaptation reduce provenance effects? | CORAL, project-out, and DANN screens |
| 5 | Does the preregistered final pipeline transfer to a locked external cohort? | IMvigor210 single-shot external evaluation |

## 2. Data And Leakage Controls

### Development cohorts

The development analysis used the existing Gide, Hugo, Riaz, and Rose immunotherapy cohorts and their fixed response labels. The clean LOCO sensitivity is limited to the cohorts for which a held-out response class is evaluable. In this collection, cohort and cancer are confounded: Rose is BLCA and the other cohorts are SKCM. Consequently, the independent cancer contribution after cohort is not identifiable.

### Locked external cohort

The external cohort is the Mariathasan et al. 2018 IMvigor210 anti-PD-L1 cohort distributed through Bioconductor ExperimentHub resource `EH6677` in `easierData` 1.18.0:

- 192 samples
- 25 responders and 167 non-responders
- 31,087 expression genes
- response positive: complete or partial response
- response negative: stable or progressive disease

The preregistration was SHA256-frozen on the remote host before expression values and outcomes were opened. The hash and freeze time are preserved in `PREREGISTRATION_SHA256.txt` and `PREREGISTRATION_FROZEN_UTC.txt`. The source and resource hashes are also included in this folder.

The distributed resource exposes internal `SAM...` sample identifiers rather than a valid GEO or ARCHS4 accession field. Therefore an exact ARCHS4 accession-overlap test could not be performed from the distributed object. This is a limitation and is recorded in `locked_imvigor210_protocol.json`; it is not treated as proof of zero overlap.

## 3. Fixed Modeling Protocol

The locked Txn_Jatin pipeline was:

1. Map source symbols to the fixed 16,055-gene Txn_Jatin vocabulary.
2. Convert expression to `log1p(TPM)`.
3. Preserve observed genes and impute assay-missing vocabulary genes with the frozen final Txn_Jatin decoder.
4. Select the 1,000 highest-variance genes using development cohorts only.
5. Fit scaling and PCA-64 using development samples only.
6. Apply diagonal training-domain CORAL using development-domain moments only.
7. Select logistic-regression `C` by development-cohort LOCO AUROC.
8. Fit intercept-only monotone calibration from development out-of-fold predictions.
9. Refit on all development samples and score IMvigor210 once.

The raw-expression comparator used the identical development-only feature selection, scaling, PCA, adaptation, and calibration stages, with no Txn_Jatin imputation. External labels were never used for model selection or adaptation.

## 4. Phase 0: Existing Eight-Decoder LOCO Null

The previously completed eight-decoder benchmark was retained rather than recalculated. Macro metrics are across held-out cohorts.

| Representation | AUROC | AUPRC |
|---|---:|---:|
| Raw expression | 0.6535 | 0.5132 |
| BRIDGE | 0.6501 | 0.5103 |
| Txn_Jatin OSDR LoRA | 0.6496 | 0.5085 |
| BulkFormer127M | 0.6493 | 0.5125 |
| BulkFormer37M | 0.6480 | 0.5112 |
| BulkFormer147M | 0.6449 | 0.5090 |
| BulkFormer50M | 0.6423 | 0.5094 |
| BulkFormer93M | 0.6422 | 0.5081 |
| Txn_Jatin | 0.6360 | 0.5071 |

No decoder had a clear paired advantage over raw expression; the recorded paired confidence intervals included zero. This is the correct null interpretation rather than a ranking claim based on tiny point estimates.

## 5. Phase 1: LOCO And Permutation Floor

The recomputed fixed-signature LOCO results were:

| Representation or signature | AUROC | AUPRC |
|---|---:|---:|
| Raw common measured | 0.6535 | 0.5132 |
| PCA-64 raw | 0.5970 | 0.4743 |
| Txn_Jatin completed | 0.6361 | 0.5054 |
| Txn_Jatin OSDR LoRA completed | 0.6301 | 0.5022 |
| IFNG-6 assay-limited | 0.6564 | 0.5114 |
| T-cell inflamed GEP | 0.6438 | 0.5427 |
| CD8 effector | 0.6480 | 0.5194 |
| Exhaustion | 0.6368 | 0.5188 |
| CYT Rooney | 0.6018 | 0.4836 |
| TLS | 0.6064 | 0.4916 |

The fixed within-cohort label-shuffle floor was:

| Metric | 2.5% | Median | 97.5% |
|---|---:|---:|---:|
| Macro AUROC | 0.4109 | 0.5011 | 0.5929 |
| Macro AUPRC | 0.3452 | 0.4073 | 0.4876 |

This permutation output is a metric null floor using fixed predictions and shuffled labels; it is not 1,000 complete refits of every model. It is therefore used as a calibration reference, not as a substitute for a full refit permutation test.

## 6. Phase 2: Provenance Variance

The PVCA-style weighted principal-component partition found that cohort dominated the measured representation geometry:

| Representation | Cohort fraction | Response after cohort |
|---|---:|---:|
| Raw common | 0.6498 | 0.0030 |
| Txn_Jatin | 0.6444 | 0.0029 |
| Txn_Jatin OSDR LoRA | 0.6480 | 0.0029 |
| BRIDGE | 0.6446 | 0.0029 |
| BulkFormer family | 0.6821-0.6833 | 0.0026-0.0027 |

The cohort-adjusted response fractions are small because the cohorts are small and highly confounded. This does not prove that response biology is absent; it shows that cohort provenance is the dominant measurable axis in this benchmark and that the design has limited power to isolate a response-specific axis.

## 7. Phase 3: Masked Signature-Gene Recovery

The targeted holdout experiment masked the measured immune-signature genes together across samples, then evaluated recovery against the original values. Response labels were not used.

| Model | Held-out genes | Pearson | Spearman | MAE in log1p-TPM |
|---|---:|---:|---:|---:|
| Txn_Jatin | 43 | 0.9522 | 0.9428 | 0.4465 |
| BulkFormer147M | 43 | 0.4638 | 0.5056 | 1.3553 |

This is the clearest positive result in the study. It supports a model-level capability to reconstruct a coherent immune-signature panel from the remaining transcriptome. It does not, by itself, show that imputation improves clinical response prediction.

When the completed expression was used to harmonize signature scores, the paired change relative to assay-limited scores was small and generally negative or near zero. For example:

| Signature | Delta AUROC | 95% CI | Delta AUPRC | 95% CI |
|---|---:|---|---:|---|
| T-cell inflamed GEP | -0.0087 | -0.0230 to 0.0006 | -0.0048 | -0.0128 to 0.0004 |
| IFNG-6 | -0.0024 | -0.0098 to 0.0026 | -0.0015 | -0.0056 to 0.0010 |
| CYT | -0.0010 | not materially different from zero | not material | not material |

The appropriate conclusion is accurate reconstruction without demonstrated response-transfer gain in this dataset.

## 8. Phase 4: Cohort-Invariant Adaptation Screen

The adaptation screen compared raw expression, Txn_Jatin, and BulkFormer representations under three fixed strategies: training-domain diagonal CORAL, project-out, and DANN.

| Txn_Jatin condition | AUROC | AUPRC |
|---|---:|---:|
| Unadapted rerun | 0.6337 | not used as the primary headline |
| Training-domain CORAL | 0.6750 | 0.5565 |
| Project-out | lower than unadapted | not competitive |
| DANN | 0.5556 | not competitive |

The paired Txn_Jatin CORAL improvement was AUROC **+0.0413** with 95% CI **-0.0217 to +0.1029**, and AUPRC **+0.0453** with 95% CI **-0.0243 to +0.1162**. The intervals cross zero. CORAL is therefore a promising prespecified engineering direction, not a confirmed clinical improvement. DANN behaved as a negative control under the fixed settings.

## 9. Phase 5: Locked IMvigor210 External Test

The locked external cohort was scored once after development-only selection and calibration.

| Representation | n | Responders | AUROC | 95% CI | AUPRC | 95% CI | Brier |
|---|---:|---:|---:|---|---:|---|---:|
| Raw expression | 192 | 25 | 0.7032 | 0.5854-0.8065 | 0.2775 | 0.1844-0.4645 | 0.1741 |
| Txn_Jatin | 192 | 25 | 0.6944 | 0.5758-0.8003 | 0.2925 | 0.1830-0.4610 | 0.1887 |

Paired Txn_Jatin minus raw:

- AUROC: **-0.0090**, 95% CI **-0.0285 to +0.0110**
- AUPRC: **+0.0053**, 95% CI **-0.0421 to +0.0542**
- Brier score: Txn_Jatin was higher, indicating worse probabilistic accuracy in this run

The external result is compatible with parity. It does not support selecting Txn_Jatin over raw expression for this endpoint on the basis of current evidence.

## 10. What The Results Mean

### Supported findings

- Txn_Jatin can recover masked immune-signature genes with high correlation.
- The model's completed representation is usable in a fixed downstream pipeline.
- Provenance is a major source of variation in the current immunotherapy data.
- Training-domain CORAL is a reasonable adaptation candidate for future validation.

### Findings not established

- A universal Txn_Jatin advantage over raw expression.
- A significant external IMvigor210 response-prediction improvement.
- A response-specific embedding axis independent of cohort and cancer.
- A clinical biomarker claim from this benchmark alone.

The most defensible manuscript framing is a mechanistic and technical result: Txn_Jatin offers biologically coherent missing-gene recovery and a path toward provenance-aware transfer, while the current response endpoint is underpowered and confounded. The external clinical result should be reported transparently as neutral.

## 11. Limitations And Next Validation

1. The four development cohorts are small, and cohort is nested with cancer. A larger multi-study cohort with crossed cancer and study factors is required to estimate response biology independently of provenance.
2. The locked IMvigor210 resource has opaque internal sample identifiers, preventing a definitive accession-level ARCHS4 overlap test from the distributed object.
3. Only one external cohort was evaluated. A second locked cohort is needed before making a general clinical-transfer claim.
4. The phase-0 eight-decoder table is the previously completed benchmark and is retained for auditability; the new targeted signature and adaptation analyses were run on the representations supported by the remote API and available local artifacts.
5. The permutation floor fixes model predictions rather than refitting each model. It is intentionally lightweight and should not be described as a full nested permutation benchmark.
6. No treatment-effect, survival, or prospective clinical utility claim follows from these experiments.

## 12. Reproducibility Map

All result tables and figures are in `results/`. Protocol metadata and hashes are in the folder root. The executable source scripts are in `scripts/` in the local study bundle and in the reproducibility repository under `protocols/immunotherapy/`.

| Artifact | Purpose |
|---|---|
| `loco_summary_with_ci.csv` | Phase-1 fixed-signature LOCO summary |
| `paired_deltas_vs_raw.csv` | Paired LOCO comparisons |
| `permutation_floor.csv` | Phase-1 label-shuffle floor |
| `pvca_variance_partition.csv` | Phase-2 weighted variance partition |
| `representation_nuisance_diagnostics.csv` | Additional nuisance diagnostics |
| `signature_gene_holdout_accuracy.csv` | Phase-3 masked-gene accuracy |
| `harmonized_vs_assay_limited_deltas.csv` | Phase-3 transfer deltas |
| `invariance_loco_summary_with_ci.csv` | Phase-4 adaptation summary |
| `invariance_paired_deltas.csv` | Phase-4 paired adaptation deltas |
| `locked_imvigor210_summary.csv` | Phase-5 external metrics |
| `locked_imvigor210_bootstrap.csv` | Phase-5 bootstrap replicates |
| `locked_imvigor210_protocol.json` | Locked external protocol and counts |
| `provenance_vs_response_variance.png` | Variance partition figure |
| `same_embedding_cohort_vs_response.png` | Nuisance/response geometry figure |
| `LOCKED_EXTERNAL_PREREGISTRATION.md` | Frozen pre-outcome protocol |

No model weights, raw patient expression matrices, ExperimentHub cache, or response-source package are included in the reproducibility bundle.
