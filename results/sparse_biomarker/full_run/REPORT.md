# Txn_Jatin sparse-panel biomarker pilot

## Claim boundary

This is exploratory internal TCGA evidence. It tests biological-signal recovery; it does not establish a clinically validated biomarker.

## Decision summary

- **Sparse-panel imputation failed.** With 86.3%-93.1% of evaluable genes hidden,
  reconstruction PCC was approximately zero and expressed/unexpressed AUROC was
  approximately 0.50. This masking regime is far outside the model's 15%
  masking curriculum.
- **Some decoder outputs remain cancer-separable, but they are not calibrated
  pathway measurements.** Ten imputed pathway-score features achieved
  five-cancer AUROC 0.9877-0.9930, yet the observed panel itself achieved
  0.9983. Individual pathway correlations were inconsistent and sometimes
  strongly negative.
- **Candidate biomarker ranking was not preserved.** Hidden tumor-normal effect
  correlations and direction agreement were at chance; top-50 overlaps were
  0-7 genes.
- **Current conclusion:** do not use this checkpoint to discover or impute
  biomarkers from 1,000-2,000-gene panels. Its output can be studied as a
  nonlinear representation of the observed panel, but that hypothesis requires
  external-cohort and batch-controlled validation.

## Frozen protocol

- Frozen model: Txn_Jatin final 20-epoch frozen checkpoint.
- Shared Txn_Jatin/TCGA genes: **14,587**.
- Signatures: **10 Hallmark programs**, with every evaluated signature gene hidden from the input.
- Split: 70/30 deterministic SHA256 case-level discovery/test split.
- Panel selection: Rank discovery-only genes by variance*sqrt(TPM>=1 detection); exclude every evaluated Hallmark signature gene.

## Reconstruction

| Observed genes | Hidden genes | Global PCC | Gene-macro PCC | Sample-macro PCC | MSE | Expression AUROC |
|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 12,587 | 0.0093 | -0.0064 | 0.0108 | 7.6082 | 0.5080 |
| 1,000 | 13,587 | 0.0095 | -0.0053 | 0.0094 | 7.9959 | 0.5050 |

## Hidden biological-program recovery

| Panel | Hallmark program | Genes | Pearson | State AUROC |
|---:|---|---:|---:|---:|
| 2,000 | Epithelial Mesenchymal Transition | 187 | 0.3032 | 0.6391 |
| 2,000 | IL-6/JAK/STAT3 Signaling | 70 | 0.2445 | 0.6475 |
| 2,000 | DNA Repair | 131 | 0.2241 | 0.6484 |
| 2,000 | Inflammatory Response | 175 | 0.2091 | 0.6644 |
| 2,000 | Hypoxia | 179 | 0.1554 | 0.5464 |
| 2,000 | Apoptosis | 142 | -0.0323 | 0.5210 |
| 2,000 | Interferon Gamma Response | 149 | -0.0614 | 0.5047 |
| 2,000 | TNF-alpha Signaling via NF-kB | 183 | -0.1825 | 0.4840 |
| 2,000 | E2F Targets | 183 | -0.4353 | 0.2042 |
| 2,000 | G2-M Checkpoint | 180 | -0.4369 | 0.2583 |
| 1,000 | DNA Repair | 131 | 0.4375 | 0.7245 |
| 1,000 | Inflammatory Response | 175 | 0.4184 | 0.7038 |
| 1,000 | IL-6/JAK/STAT3 Signaling | 70 | 0.3436 | 0.6234 |
| 1,000 | Interferon Gamma Response | 149 | 0.0551 | 0.5278 |
| 1,000 | E2F Targets | 183 | 0.0357 | 0.4120 |
| 1,000 | G2-M Checkpoint | 180 | 0.0259 | 0.4799 |
| 1,000 | Apoptosis | 142 | -0.0975 | 0.4212 |
| 1,000 | TNF-alpha Signaling via NF-kB | 183 | -0.2155 | 0.3588 |
| 1,000 | Hypoxia | 179 | -0.2755 | 0.2691 |
| 1,000 | Epithelial Mesenchymal Transition | 187 | -0.7385 | 0.1431 |

## Patient-grouped phenotype signal

| Panel | Features | Endpoint | OOF AUROC | Balanced accuracy |
|---:|---|---|---:|---:|
| 2,000 | full expression hidden pathway scores | five cancer type among tumors | 0.9263 | 0.7378 |
| 2,000 | full expression hidden pathway scores | tumor vs normal four cancers | 0.9671 | 0.9071 |
| 2,000 | Txn Jatin imputed hidden pathway scores | five cancer type among tumors | 0.9877 | 0.9071 |
| 2,000 | Txn Jatin imputed hidden pathway scores | tumor vs normal four cancers | 0.9844 | 0.9440 |
| 2,000 | observed panel expression | five cancer type among tumors | 0.9983 | 0.9706 |
| 2,000 | observed panel expression | tumor vs normal four cancers | 0.9973 | 0.9832 |
| 1,000 | full expression hidden pathway scores | five cancer type among tumors | 0.9263 | 0.7378 |
| 1,000 | full expression hidden pathway scores | tumor vs normal four cancers | 0.9671 | 0.9071 |
| 1,000 | Txn Jatin imputed hidden pathway scores | five cancer type among tumors | 0.9930 | 0.9402 |
| 1,000 | Txn Jatin imputed hidden pathway scores | tumor vs normal four cancers | 0.9523 | 0.8868 |
| 1,000 | observed panel expression | five cancer type among tumors | 0.9983 | 0.9728 |
| 1,000 | observed panel expression | tumor vs normal four cancers | 0.9973 | 0.9862 |

## Hidden-gene effect preservation

| Panel | Cancer | Effect Spearman | Direction concordance | Top-50 overlap |
|---:|---|---:|---:|---:|
| 2,000 | TCGA-BRCA | -0.0135 | 0.4920 | 7/50 |
| 2,000 | TCGA-KIRC | 0.0029 | 0.4913 | 7/50 |
| 2,000 | TCGA-LUAD | 0.0038 | 0.4958 | 7/50 |
| 2,000 | TCGA-LUSC | 0.0070 | 0.4985 | 7/50 |
| 1,000 | TCGA-BRCA | -0.0523 | 0.4836 | 0/50 |
| 1,000 | TCGA-KIRC | -0.0237 | 0.4818 | 2/50 |
| 1,000 | TCGA-LUAD | -0.0291 | 0.4901 | 1/50 |
| 1,000 | TCGA-LUSC | -0.0333 | 0.4921 | 0/50 |

## Interpretation rules

- Global reconstruction PCC is abundance-sensitive; gene-macro PCC is the stricter gene-wise measure.
- Hallmark state AUROC asks whether hidden pathway activity states can be recovered from the observed panel.
- Phenotype probes use patient-grouped out-of-fold predictions; they are sanity checks, not external validation.
- Tumor-normal effect overlap measures preservation of candidate ranking, not clinical utility.
- Survival, treatment response, prospective assay validation, and independent cohorts remain required.
- Because Txn_Jatin output is a deterministic transform of the observed panel,
  high cancer classification does not prove added biological information; the
  appropriate question is whether it improves generalization in an independent
  cohort or platform.
