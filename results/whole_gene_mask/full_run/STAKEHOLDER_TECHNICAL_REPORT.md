# Txn_Jatin Whole-Gene Imputation and Cancer-Signal Benchmark

**Final technical report**  
**Experiment completed:** 23 July 2026  
**Status:** All planned decoder inference, imputation metrics, and cancer random-forest analyses completed successfully.

## Executive summary

This study tested whether pretrained gene-expression models can recover genes
that are completely hidden across all samples. This is a stronger test than
randomly masking different entries in different samples because the model
cannot rely on any observed value of the target gene within the evaluation
cohort. It must infer the gene from the remaining 85% of the transcriptome and
from relationships learned during pretraining.

Seven models with validated expression decoders were evaluated:

- Txn_Jatin
- the original BRIDGE model
- BulkFormer-37M, 50M, 93M, 127M, and 147M

Four embedding-only representations were retained in the result schema but
reported as `NaN` for imputation: ESM2_PCA512_prior, Geneformer, scGPT, and
Txn_Jatin_contextual. They do not expose a validated bulk-expression decoder,
so manufacturing an imputation output would not be scientifically defensible.

This report covers the completed 20-epoch Txn_Jatin checkpoint and the new
whole-gene imputation/cancer-signal experiment. The earlier GitHub-protocol
static gene-embedding and gene-pair results were preserved without
recalculation and remain a separate evaluation surface. They should be used to
discuss static biological organization, while this report should be used to
discuss expression-conditioned recovery and downstream signal preservation.

The principal findings are:

1. **Txn_Jatin was strongest for detecting whether a hidden TCGA gene should
   be expressed.** Its mean macro AUROC was 0.9284, compared with 0.9121 for
   BRIDGE and 0.8885 for the best BulkFormer checkpoint.
2. **BRIDGE was strongest for continuous expression reconstruction.** It had
   the highest global PCC and lowest MSE on both TCGA and OSDR. Its OSDR result
   was especially strong: PCC 0.9026 and macro AUROC 0.9550.
3. **Txn_Jatin and BRIDGE preserve different aspects of the signal.** On TCGA,
   Txn_Jatin had lower MAE and higher AUROC/AUPRC, while BRIDGE had higher PCC
   and lower MSE. Txn_Jatin therefore produced better typical errors and
   expression-state ranking, while BRIDGE better preserved global continuous
   geometry and avoided large squared errors.
4. **Scaling BulkFormer helped on TCGA, but not consistently on OSDR.**
   BulkFormer-147M was the strongest BulkFormer on TCGA, but larger parameter
   counts did not produce monotonic OSDR gains. Domain and preprocessing match
   mattered more than parameter count alone.
5. **All completed matrices retained nearly all tumor-versus-normal
   information.** Random-forest AUROCs remained approximately 0.999 for the
   two-cancer panel and 0.995-0.996 for the five-cancer panel. These results
   demonstrate signal preservation, but they are not sufficiently
   discriminative to rank the models reliably because the raw-expression
   control was already near the performance ceiling.
6. **The experiment supports a targeted, not universal, benefit for the
   Txn_Jatin representation objectives.** The TCGA expression-state result is
   consistent with improved biologically informed ranking, but the OSDR result
   shows that Txn_Jatin is not uniformly superior to reconstruction-focused
   BRIDGE.

## 1. Scientific questions

The experiment addressed four questions:

1. Can a model recover a gene when that gene is absent from every sample?
2. Does Txn_Jatin improve recovery relative to the original BRIDGE model?
3. Does increasing BulkFormer model size improve imputation?
4. Does replacing 15% of TCGA genes with model imputations preserve the signal
   needed to distinguish tumor from normal tissue and identify stable
   predictive genes?

The study did **not** test whether these models can establish causal cancer
drivers, nor did it perform clinical validation.

## 2. Txn_Jatin training and architecture

### 2.1 Training corpus

Txn_Jatin was trained for 20 epochs on the full prepared ARCHS4 corpus:

| Property | Value |
|---|---:|
| Total ARCHS4 samples | 978,212 |
| Human samples | 499,845 |
| Mouse samples | 478,367 |
| Training samples | 958,212 |
| Validation samples | 20,000 |
| Canonical genes | 16,055 |
| Training batch size | 16 |
| Final train reconstruction loss | 0.07614 |
| Final validation loss | 0.07806 |

Expression was normalized using gene-length-adjusted `log1p(TPM)`. The
normalization manifest records GENCODE human release 49 and mouse M38 gene
annotations and a one-to-one MGI mouse/human orthology map.

### 2.2 Backbone

Txn_Jatin and BRIDGE use the same broad ExpressionPerformer backbone:

- 512-dimensional hidden states
- 12 transformer layers
- 8 attention heads
- 2,048-dimensional feed-forward layers
- learned gene-identity embeddings
- rotary expression embeddings
- FlashAttention-compatible attention
- a per-gene expression reconstruction head
- approximately 46 million trainable parameters for Txn_Jatin

Using the same backbone makes the Txn_Jatin-versus-BRIDGE comparison useful,
but the checkpoints still differ in training data preparation, vocabulary, and
training objectives.

### 2.3 What Txn_Jatin adds

The original BRIDGE objective is dominated by masked expression
reconstruction. Txn_Jatin retains this objective and adds representation
constraints:

1. **Masked reconstruction:** 15% of genes are hidden and reconstructed from
   the remaining transcriptome.
2. **ESM2 gene-prior initialization and alignment:** gene-identity embeddings
   were initialized from and regularized toward a 512-dimensional
   ESM2-derived protein-sequence prior. The prior covered 15,666 of 16,055
   genes (97.58%) and retained 98.27% of variance after PCA.
3. **Context-prior alignment:** expression-conditioned hidden states were
   regularized toward the corresponding sequence-informed prior.
4. **Sample-level contrastive learning:** partial views of the same sample
   were encouraged to remain close while different samples were separated.

The configured auxiliary weights were 0.02 for the static gene prior, 0.03 for
the contextual prior, and 0.01 for sample contrastive loss, with contrastive
temperature 0.1.

The intended effect is not merely lower reconstruction error. It is to make a
gene representation preserve sequence identity, transcriptomic context, and
sample-level biological organization simultaneously.

## 3. Benchmark design

### 3.1 Datasets

| Dataset | Samples | Source genes | Input scale | Use |
|---|---:|---:|---|---|
| TCGA | 3,481 | 15,165 | `log1p(TPM)` | Imputation and cancer RF |
| OSDR | 2,099 | 15,165 | prepared `log1p(CPM)` | Imputation |

The TCGA set contains five projects:

| Project | Tumor | Normal |
|---|---:|---:|
| BRCA | 1,118 | 113 |
| KIRC | 542 | 72 |
| LUAD | 542 | 59 |
| LUSC | 511 | 51 |
| SKCM | 472 | 1 |

### 3.2 Shared gene universe

The primary benchmark used the intersection of TCGA, OSDR, Txn_Jatin, BRIDGE,
and BulkFormer:

| Source | Native genes | Shared benchmark genes |
|---|---:|---:|
| TCGA | 15,165 | 14,585 |
| OSDR | 15,165 | 14,585 |
| Txn_Jatin | 16,055 | 14,585 |
| BRIDGE | 15,165 | 14,585 |
| BulkFormer | 20,010 rows / 20,007 unique | 14,585 |

Each model was run in its native vocabulary. Observed genes were aligned by
symbol, genes absent from the evaluation matrix were filled with zero, and
metrics were calculated only on the common masked genes.

### 3.3 Whole-gene masking

- Mask fraction: 15%
- Genes masked per seed: 2,188
- Seeds: 20260723, 20260724, 20260725
- Mask token: -10
- The same target genes were hidden in every sample.
- The same masks were used for every eligible model and both datasets.

This design asks whether learned cross-gene structure can recover a feature
that is wholly unavailable, rather than completing isolated missing entries.

### 3.4 Imputation metrics

The primary metrics were:

- **Global Pearson correlation (PCC):** preservation of continuous linear
  expression geometry across all masked values.
- **Global Spearman correlation:** preservation of rank order.
- **MSE and MAE:** absolute reconstruction error. MSE penalizes occasional
  large errors more strongly than MAE.
- **Macro AUROC and AUPRC:** for each masked gene, the original expression was
  converted to expressed/unexpressed using TPM >= 1 for TCGA and CPM >= 1 for
  OSDR. The imputed continuous value was used as the score. AUROC/AUPRC were
  averaged over genes containing both classes.

Between 1,543 and 1,581 TCGA genes and between 1,337 and 1,346 OSDR genes per
mask were eligible for the binary-state metrics.

Reported uncertainty is the standard deviation across three mask seeds. It is
not a population confidence interval.

### 3.5 Cancer-signal benchmark

Two binary tumor-versus-normal panels were evaluated:

- **Two-cancer panel:** LUAD + LUSC, 1,163 samples from 1,019 patients
- **Five-cancer panel:** BRCA + KIRC + LUAD + LUSC + SKCM, 3,481 samples from
  3,116 patients

For each model and mask:

1. The observed 85% of genes were retained.
2. The hidden 15% were replaced by model predictions.
3. A 300-tree random forest with balanced class weights was evaluated using
   five-fold stratified patient-group cross-validation.
4. A final model was fitted to produce gene-importance rankings.

Grouping by patient prevents samples from the same patient appearing in both
training and test folds.

## 4. Imputation results

### 4.1 TCGA

Mean +/- standard deviation across three whole-gene masks:

| Model | PCC | MSE | MAE | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| BRIDGE | **0.9155 +/- 0.0006** | **0.4112 +/- 0.0040** | 0.4090 +/- 0.0028 | 0.9121 +/- 0.0022 | 0.8490 +/- 0.0060 |
| Txn_Jatin | 0.9087 +/- 0.0067 | 0.4444 +/- 0.0261 | **0.3950 +/- 0.0089** | **0.9284 +/- 0.0018** | **0.8696 +/- 0.0047** |
| BulkFormer-147M | 0.8931 +/- 0.0025 | 0.5138 +/- 0.0205 | 0.4888 +/- 0.0064 | 0.8885 +/- 0.0028 | 0.8233 +/- 0.0089 |
| BulkFormer-127M | 0.8907 +/- 0.0027 | 0.5268 +/- 0.0203 | 0.5025 +/- 0.0050 | 0.8867 +/- 0.0031 | 0.8201 +/- 0.0081 |
| BulkFormer-93M | 0.8782 +/- 0.0043 | 0.5843 +/- 0.0265 | 0.5425 +/- 0.0100 | 0.8677 +/- 0.0010 | 0.8039 +/- 0.0070 |
| BulkFormer-50M | 0.8742 +/- 0.0030 | 0.6048 +/- 0.0213 | 0.5511 +/- 0.0061 | 0.8514 +/- 0.0047 | 0.7916 +/- 0.0098 |
| BulkFormer-37M | 0.8731 +/- 0.0045 | 0.6083 +/- 0.0218 | 0.5364 +/- 0.0076 | 0.8456 +/- 0.0038 | 0.7859 +/- 0.0093 |

**Interpretation**

- BRIDGE had the best PCC and MSE, indicating the strongest preservation of
  continuous expression geometry.
- Txn_Jatin had the best MAE, AUROC, and AUPRC. Its typical absolute error was
  lower, and it was better at ranking whether hidden genes should be active or
  inactive.
- Txn_Jatin exceeded BRIDGE by 0.0164 AUROC and 0.0206 AUPRC, despite trailing
  BRIDGE by 0.0068 PCC.
- The combination of lower MAE but higher MSE for Txn_Jatin indicates that most
  errors were smaller, while a minority of larger errors received a stronger
  MSE penalty.
- BulkFormer improved with scale on TCGA. BulkFormer-147M was the strongest
  BulkFormer, although it did not reach Txn_Jatin or BRIDGE.

### 4.2 OSDR

| Model | PCC | MSE | MAE | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| BRIDGE | **0.9026 +/- 0.0116** | **0.6661 +/- 0.0813** | **0.4352 +/- 0.0164** | **0.9550 +/- 0.0014** | **0.9515 +/- 0.0023** |
| Txn_Jatin | 0.8023 +/- 0.0150 | 1.3079 +/- 0.1034 | 0.7508 +/- 0.0196 | 0.8620 +/- 0.0042 | 0.8796 +/- 0.0016 |
| BulkFormer-127M | 0.7634 +/- 0.0192 | 1.6000 +/- 0.1447 | 0.9475 +/- 0.0412 | 0.8315 +/- 0.0028 | 0.8494 +/- 0.0016 |
| BulkFormer-50M | 0.7593 +/- 0.0140 | 1.5378 +/- 0.0923 | 0.9285 +/- 0.0262 | 0.8397 +/- 0.0021 | 0.8571 +/- 0.0022 |
| BulkFormer-37M | 0.7590 +/- 0.0154 | 1.5376 +/- 0.1046 | 0.9090 +/- 0.0234 | 0.8362 +/- 0.0008 | 0.8526 +/- 0.0026 |
| BulkFormer-147M | 0.7576 +/- 0.0130 | 1.6302 +/- 0.0862 | 0.9611 +/- 0.0197 | 0.8273 +/- 0.0022 | 0.8470 +/- 0.0041 |
| BulkFormer-93M | 0.7527 +/- 0.0142 | 1.6913 +/- 0.0831 | 0.9872 +/- 0.0211 | 0.8340 +/- 0.0021 | 0.8552 +/- 0.0030 |

**Interpretation**

- BRIDGE was decisively strongest on every reported OSDR metric.
- Txn_Jatin remained clearly above all BulkFormer checkpoints but did not
  reproduce its TCGA-relative advantage over BRIDGE.
- BulkFormer scaling was not monotonic. The 127M model had the best BulkFormer
  PCC, while the 50M model had the best BulkFormer AUROC/AUPRC.
- This is evidence that domain and preprocessing compatibility can dominate
  parameter count.
- The OSDR input is prepared `log1p(CPM)`, whereas Txn_Jatin was trained with
  a gene-length-adjusted `log1p(TPM)` pipeline. The BRIDGE vocabulary and the
  prepared OSDR matrix are also closely aligned. These factors may contribute
  to BRIDGE's advantage and should be isolated before attributing the full
  difference to model architecture.

## 5. Cancer-signal results

Mean across the three imputed masks:

| Model | 2-cancer AUROC | 2-cancer balanced accuracy | 5-cancer AUROC | 5-cancer balanced accuracy |
|---|---:|---:|---:|---:|
| Raw expression control | 0.999491 | 0.983091 | 0.995301 | 0.963702 |
| Txn_Jatin | 0.999479 +/- 0.000070 | 0.985930 +/- 0.003646 | 0.996105 +/- 0.000568 | **0.968063 +/- 0.002369** |
| BRIDGE | 0.999387 +/- 0.000017 | **0.988950 +/- 0.000766** | 0.996049 +/- 0.000509 | 0.966799 +/- 0.002700 |
| BulkFormer-147M | 0.999332 +/- 0.000134 | 0.987435 +/- 0.002516 | 0.995768 +/- 0.000419 | 0.966774 +/- 0.001812 |
| BulkFormer-127M | 0.999327 +/- 0.000096 | 0.987593 +/- 0.002371 | 0.995287 +/- 0.001609 | 0.966375 +/- 0.002404 |
| BulkFormer-93M | 0.999393 +/- 0.000022 | **0.988950 +/- 0.000766** | 0.995110 +/- 0.001013 | 0.965136 +/- 0.003049 |
| BulkFormer-50M | 0.999430 +/- 0.000030 | 0.985930 +/- 0.003646 | 0.996158 +/- 0.000806 | 0.965135 +/- 0.001611 |
| BulkFormer-37M | 0.999381 +/- 0.000095 | 0.984319 +/- 0.004746 | **0.996380 +/- 0.000658** | 0.965669 +/- 0.002597 |

The largest mean AUROC difference from the raw control was only 0.00108. On
the two-cancer panel, Txn_Jatin differed from raw expression by -0.000012. On
the five-cancer panel, Txn_Jatin differed by +0.000804.

These tiny differences should be interpreted as equivalence at the current
resolution, not as evidence that imputation improves cancer classification.
Random-forest stochasticity and mask composition are of comparable magnitude.

The scientifically supported conclusion is:

> Replacing 15% of TCGA genes with outputs from any of the seven validated
> decoders preserved essentially all tumor-versus-normal classification signal.

This outcome is reassuring for robustness, but it does not identify a clear
winner because:

- 85% of features were unchanged and directly observed.
- Tumor versus normal is a comparatively easy task in this cohort.
- The classes are imbalanced.
- The five-cancer panel includes only one normal SKCM sample.
- Performance is already near the AUROC ceiling.

## 6. Predictive gene rankings

Random-forest importance was aggregated across the three masks. The following
genes appeared in the top 20 for every decoder:

### Two-cancer consensus, 7 of 7 models

- CLEC3B
- TEK
- SH2D3C
- RS1
- GDF10
- STX11

Additional high-consensus genes included FABP4, VEPH1, AQP4, PECAM1, SFTPC,
GPM6A, TMEM100, CAV1, ADAMTS8, FGR, RAMP3, and GIMAP8.

### Five-cancer consensus, 7 of 7 models

- GPC3
- LRRC2
- CLEC3B
- ALDH1A2
- FOXM1
- ASF1B
- FBLN5
- ANLN
- IQGAP3
- SPC24
- ANGPTL1
- PKMYT1

The two-cancer ranking qualitatively contains many vascular, stromal, immune,
and lung/tissue-composition signals. The five-cancer ranking contains a
prominent cell-cycle and proliferation component. This is a qualitative
interpretation only; no formal enrichment test was performed in this stage.

These rankings are **predictive feature rankings, not causal cancer-gene
claims**. Random-forest impurity importance can favor correlated or
high-variance variables, and tumor/normal differences can reflect tissue
composition, purity, inflammation, or batch effects.

## 7. What the results mean for Txn_Jatin

### 7.1 Supported conclusions

1. Txn_Jatin learned enough conditional transcriptomic structure to recover
   completely hidden genes with high correlation and high expression-state
   AUROC on two external evaluation matrices.
2. On TCGA, Txn_Jatin's sequence/context-informed training is associated with
   the best expressed-versus-unexpressed discrimination among all tested
   decoders.
3. Txn_Jatin preserves tumor/normal signal after replacing 15% of genes with
   model predictions.
4. Txn_Jatin outperformed every BulkFormer checkpoint on both datasets for
   global PCC and macro AUROC.
5. The auxiliary objectives did not yield universal superiority: BRIDGE
   remained stronger for continuous reconstruction and was substantially
   stronger on OSDR.

### 7.2 Mechanistic interpretation

The TCGA pattern is consistent with the intended Txn_Jatin design. ESM2 prior
alignment and contextual/contrastive objectives may help the model place genes
in a biologically coherent representation space, improving whether a gene
should be active under a given transcriptomic context. The reconstruction-only
BRIDGE objective appears better optimized for preserving exact continuous
expression geometry.

This is an interpretation, not proof of causation. The checkpoints were not
trained in a controlled ablation study, so the observed difference cannot be
assigned uniquely to any one auxiliary objective.

### 7.3 Why OSDR differs

At least four explanations remain plausible:

1. The OSDR `log1p(CPM)` scale differs from Txn_Jatin's training-time
   `log1p(TPM)` scale.
2. BRIDGE may have a more favorable vocabulary or preprocessing alignment with
   the prepared OSDR matrix.
3. OSDR biology and technical protocols differ from TCGA and ARCHS4.
4. The Txn_Jatin auxiliary objectives may trade exact reconstruction fidelity
   for representation-level organization.

A matched-normalization ablation is required to distinguish these mechanisms.

## 8. Limitations and risks

1. **Only three mask seeds were used.** Standard deviations measure mask
   sensitivity but are not confidence intervals.
2. **No simple imputation baselines were run in this stage.** Mean, low-rank,
   k-nearest-neighbor, and coexpression baselines are needed to quantify the
   incremental value of deep models.
3. **The checkpoints are not controlled ablations.** They differ in objectives,
   vocabulary, normalization, and potentially training corpus exposure.
4. **BulkFormer TCGA provenance requires care.** The published graph resource
   is named `G_tcga`. Unless sample-level provenance confirms exclusion of
   evaluation samples, TCGA results should be treated as potentially
   transductive rather than a completely untouched external test.
5. **Different numerical precision was required.** Txn_Jatin and BRIDGE used
   BF16 inference on H100. BulkFormer used FP32 because its sparse graph
   operation did not support BF16 in the installed stack.
6. **Zero-filling model-specific genes is an operational alignment choice.**
   It may not be the optimal missing-feature representation for every model.
7. **The cancer task is near ceiling.** It is suitable for checking signal
   preservation, but weak for distinguishing models.
8. **Five-cancer class balance is uneven.** SKCM contributes 472 tumors but
   only one normal sample.
9. **Feature importance is not causal inference.** The rankings require
   external validation and stability analysis.
10. **Threshold dependence:** AUROC/AUPRC labels use TPM or CPM >= 1. Other
    expression thresholds should be tested.

## 9. Recommended next experiments

### Priority 1: controlled Txn_Jatin ablation

Train otherwise identical checkpoints with:

- reconstruction only
- reconstruction + static ESM2 prior
- reconstruction + static/context priors
- full Txn_Jatin objective

This would determine which objective produces the TCGA AUROC gain and whether
it causes the OSDR reconstruction tradeoff.

### Priority 2: matched preprocessing

Run TCGA and OSDR under matched normalization pipelines and include an explicit
normalization calibration step. This is essential before presenting the OSDR
gap as an architectural effect.

### Priority 3: stronger baselines

Add:

- per-gene training mean/median
- PCA or low-rank matrix completion
- k-nearest-neighbor imputation
- coexpression regression
- randomly initialized model

### Priority 4: external validation

Freeze all modeling and evaluate on independently processed cohorts not used
for model, prior, graph, or threshold development.

### Priority 5: cancer-gene discovery protocol

For a stronger gene-discovery claim:

- evaluate classifiers using only masked/imputed genes
- use repeated nested patient-group cross-validation
- add permutation importance and stability selection
- control for tissue, purity, sex, batch, and cancer project
- validate ranked genes in an independent cohort
- perform formal pathway enrichment with a frozen annotation version

## 10. Stakeholder conclusions

The correct high-level message is:

> Txn_Jatin successfully recovers completely hidden genes and provides the
> strongest TCGA expression-state discrimination of the tested models. BRIDGE
> remains the strongest continuous reconstruction model, particularly on
> OSDR. All native decoders preserve tumor-versus-normal signal after 15%
> whole-gene replacement.

The message should **not** be:

> Txn_Jatin is universally the best model, or the RF importance list proves new
> causal cancer genes.

## 11. Suggested presentation structure

1. **Problem:** Can learned biological structure recover an entirely absent
   gene?
2. **Model:** BRIDGE reconstruction backbone plus Txn_Jatin sequence,
   contextual, and contrastive constraints.
3. **Protocol:** 14,585 shared genes, 15% whole-gene masks, three seeds, TCGA
   and OSDR.
4. **Main result:** Txn_Jatin leads TCGA AUROC; BRIDGE leads PCC/MSE and OSDR.
5. **Scale result:** BulkFormer scaling helps TCGA but not OSDR consistently.
6. **Downstream result:** all decoders preserve tumor/normal classification.
7. **Gene rankings:** stable predictive signatures, explicitly not causal.
8. **Decision:** retain Txn_Jatin for biology-oriented representation work,
   retain BRIDGE as the reconstruction reference, and run controlled
   normalization/ablation studies next.

## 12. Reproducibility and source artifacts

The H100 benchmark ran from 20:38:25 to 21:26:22 UTC, approximately 48 minutes.
GPU inference and CPU analysis were parallelized. No model or benchmark failed.

Primary artifacts:

- [Frozen protocol](config/protocol.json)
- [Model capability table](config/model_capabilities.csv)
- [Gene coverage](config/gene_coverage.csv)
- [All imputation rows](results/current/all_imputation_results.csv)
- [Mean imputation comparison](results/current/imputation_means_across_masks.csv)
- [All cancer RF rows](results/current/all_cancer_rf_results.csv)
- [Progress/completion manifest](results/current/progress_summary.json)
- [Raw RF controls](results/raw_rf_baseline/raw_rf_summary.csv)
- Per-gene metrics under `results/imputation/<model>/`
- Per-model RF rankings under `results/masked_rf/<model>/`
- Complete execution logs under `logs/`

Large prediction matrices remain on attached remote storage and were excluded
from the compact local mirror.
