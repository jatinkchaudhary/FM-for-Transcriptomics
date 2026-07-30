# Initial immunotherapy cohort feasibility study

**Model:** Txn_Jatin, final epoch-20 checkpoint  
**Analysis date:** 2026-07-25  
**Status:** Completed exploratory feasibility study; not a clinical validation

## Executive conclusion

This study asked whether a frozen Txn_Jatin model can transfer directly to
pretreatment immune-checkpoint-inhibitor (ICI) response prediction across
independent cohorts.

The answer is currently **no for the frozen sample embedding**. Across four
cohorts and 239 patients, the contrastively trained Txn_Jatin sample vector had
a macro leave-one-cohort-out (LOCO) AUROC of **0.406** and a pooled AUROC of
**0.462** (95% bootstrap CI 0.390-0.536). Raw expression over genes measured in
every cohort achieved macro AUROC **0.654** and pooled AUROC **0.667**
(0.599-0.736). The paired pooled AUROC difference was -0.205
(95% CI -0.299 to -0.115).

Txn_Jatin was more useful as a **gene-completion engine** than as a global
sample vector. Retaining measured expression and filling assay-missing genes
with the decoder produced macro AUROC **0.636** and pooled AUROC **0.659**,
close to raw expression. It did not provide a consistent cohort-level
improvement.

The main failure mode is measurable: cohort/platform identity is perfectly
decodable from the Txn context representation (five-fold accuracy **1.00**),
whereas response separation is weak. The embedding geometry therefore encodes
technical and cohort structure more strongly than a transportable ICI-response
axis.

There is nevertheless a biologically coherent result worth pursuing.
Exploratory random-effects analysis identified higher **interferon-alpha
response**, **interferon-gamma response**, and **allograft-rejection** pathway
scores in responders, with the same effect direction in all four cohorts and
FDR q-values of 0.013, 0.014, and 0.028. These are hypothesis-generating
associations, not independently validated biomarkers, and they arise from
expression pathway scores rather than uniquely from the frozen Txn vector.

## Study question

The prespecified question was:

> Does unsupervised ARCHS4 pretraining create a frozen Txn_Jatin
> representation that predicts ICI response in a cohort not used to train the
> response classifier?

The primary endpoint was binary pretreatment RECIST response:

- Responder (R): complete or partial response
- Nonresponder (NR): stable or progressive disease

No response labels were used during Txn_Jatin representation extraction.

## Cohorts

| Cohort | Cancer | Patients | R | NR | Expression source |
|---|---|---:|---:|---:|---|
| Gide | Melanoma | 73 | 40 | 33 | COMPASS harmonized TPM |
| Riaz | Melanoma | 51 | 10 | 41 | GSE91061 FPKM |
| Hugo | Melanoma | 26 | 14 | 12 | GSE78220 FPKM |
| Rose | Bladder cancer | 89 | 16 | 73 | GSE176307 TPM |
| **Total** | Two cancers | **239** | **80** | **159** | |

Clinical labels were taken from the COMPASS harmonization. FPKM matrices were
converted to per-sample TPM before `log1p`; TPM matrices were transformed with
`log1p`. Riaz Entrez identifiers were mapped with the downloaded NCBI human
gene-info table. Hugo's Pt27A/Pt27B measurements were averaged before TPM
conversion. Every retained clinical patient was matched to expression.

Sources:

- COMPASS paper: https://www.nature.com/articles/s41591-026-04502-7
- COMPASS data portal: https://www.immuno-compass.com/download/
- Riaz/GSE91061: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE91061
- Hugo/GSE78220: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE78220
- Rose/GSE176307: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176307

## Frozen model extraction

The final epoch-20 checkpoint was loaded on the remote NVIDIA H100:

- 16,055 genes
- 512 hidden dimensions
- 12 transformer layers
- Validation reconstruction loss 0.07806
- Sample contrastive objective weight 0.01

Four representations were evaluated:

1. **Txn context:** measured-gene mean of final hidden state minus the static
   gene embedding. This matches the representation used by the training-time
   sample contrastive objective and was L2-normalized before its linear probe.
2. **Txn hidden all:** max + mean + median pooling of final hidden states.
3. **Txn-completed expression:** observed values retained; genes absent from a
   cohort's assay filled with decoder predictions.
4. **Txn-completed Hallmark:** means of 50 Hallmark gene sets calculated from
   the completed matrix.

Genes absent from an assay were passed to Txn_Jatin as the checkpoint mask
token (`-10`), not as biological zero. The contrastive mean included measured
genes only. Model-gene coverage was 84.7% for Gide, 97.2% for Riaz, 93.6% for
Hugo, and 94.4% for Rose.

## Comparators and validation

The raw comparator used 13,523 model genes measured in all four cohorts. The
top 1,000 variance genes were selected within each training split. A second
comparator used means of all 50 Hallmark pathways over the common genes.

Every classifier was a class-weighted L2 logistic regression. Its `C` value
was chosen from 0.01, 0.1, 1, and 10 by leave-one-training-cohort-out AUROC.
The outer test cohort remained untouched during feature selection, scaling,
and hyperparameter selection.

Primary performance is the macro average of four held-out-cohort AUROCs.
Pooled out-of-cohort AUROC is also reported but can be influenced by
cross-cohort probability calibration. Patient bootstrap confidence intervals
used 1,000 replicates. A single five-fold within-cohort analysis was run only
as a signal diagnostic.

## Primary results

| Representation | Macro AUROC | Macro AUPRC | Pooled AUROC (95% CI) | Pooled AUPRC |
|---|---:|---:|---:|---:|
| Raw common-gene expression | **0.654** | **0.513** | **0.667 (0.599-0.736)** | **0.470** |
| Txn-completed expression | 0.636 | 0.505 | 0.659 (0.590-0.727) | 0.462 |
| Hallmark common-gene means | 0.551 | 0.425 | 0.608 (0.542-0.683) | 0.376 |
| Txn-completed Hallmark means | 0.551 | 0.426 | 0.607 (0.540-0.682) | 0.376 |
| Txn context | 0.406 | 0.371 | 0.462 (0.390-0.536) | 0.298 |
| Txn hidden all | 0.452 | 0.356 | 0.368 (0.294-0.437) | 0.264 |

### Held-out cohort AUROC

| Representation | Gide | Riaz | Hugo | Rose |
|---|---:|---:|---:|---:|
| Raw common-gene expression | **0.692** | **0.695** | **0.554** | **0.674** |
| Txn-completed expression | 0.693 | 0.693 | 0.488 | 0.670 |
| Hallmark common-gene means | 0.530 | 0.551 | 0.458 | 0.666 |
| Txn context | 0.494 | 0.361 | 0.214 | 0.557 |
| Txn hidden all | 0.552 | 0.510 | 0.268 | 0.480 |

Txn-completed expression was effectively neutral relative to raw expression in
the four-cohort paired analysis (pooled AUROC difference -0.008, 95% CI -0.018
to 0.002). The frozen context and hidden vectors were significantly worse.

Within-cohort Txn-context AUROCs were 0.600 (Gide), 0.432 (Riaz), 0.619
(Hugo), and 0.423 (Rose). This does not support a stable response axis even
before cross-cohort transfer.

## Pretraining-overlap sensitivity

Exact archive-accession matching against the ARCHS4 training metadata found
25/239 evaluated patients in the pretraining corpus. All 25 were from Hugo;
Gide, Riaz, and Rose had zero exact patient overlap. This is **unlabeled sample
exposure**, not response-label leakage.

A clean sensitivity excluded Hugo from both response-classifier training and
testing:

| Representation | Clean macro AUROC | Clean pooled AUROC (95% CI) |
|---|---:|---:|
| Raw common-gene expression | **0.653** | 0.694 (0.616-0.767) |
| Txn-completed expression | 0.643 | **0.736 (0.655-0.804)** |
| Hallmark common-gene means | 0.636 | 0.682 (0.605-0.753) |
| Txn context | 0.574 | 0.645 (0.571-0.719) |
| Txn hidden all | 0.552 | 0.450 (0.376-0.522) |

Txn completion improved pooled AUROC by 0.042 (paired 95% CI 0.002-0.078),
but its macro AUROC was 0.010 below raw expression and its cohort-specific
effect was inconsistent. The appropriate conclusion is signal preservation
and possible calibration benefit, not a demonstrated discrimination gain.

## Why the frozen representation failed

| Feature | Cohort CV accuracy | Cohort silhouette | Response silhouette | Cohort eta-squared | Response eta-squared after cohort centering |
|---|---:|---:|---:|---:|---:|
| Txn context | 1.000 | 0.420 | 0.063 | 0.568 | 0.0046 |
| Txn hidden all | 1.000 | 0.433 | 0.063 | 0.452 | 0.0036 |
| Raw common genes | 1.000 | 0.485 | 0.055 | 0.670 | 0.0058 |
| Hallmark means | 0.971 | 0.285 | 0.092 | 0.698 | 0.0119 |

The model has learned meaningful transcriptomic structure, but the strongest
sample-level structure in these data is cohort, cancer type, platform, and
processing pipeline. Txn_Jatin's unsupervised objectives did not make that
structure response-invariant. Strong gene-level functional embeddings
therefore do not automatically imply a useful universal patient-level
immunotherapy predictor.

## Exploratory biological signal

Random-effects meta-analysis of responder-versus-nonresponder Hallmark scores
found three pathways below FDR 0.05:

| Pathway | Meta Hedges g | 95% CI | Direction concordance | FDR q |
|---|---:|---:|---:|---:|
| Interferon Alpha Response | 0.551 | 0.256 to 0.847 | 4/4 cohorts | 0.013 |
| Interferon Gamma Response | 0.521 | 0.225 to 0.816 | 4/4 cohorts | 0.014 |
| Allograft Rejection | 0.472 | 0.178 to 0.767 | 4/4 cohorts | 0.028 |

Positive values indicate higher pathway scores in responders. This is
biologically consistent with an inflamed, interferon-active tumor
microenvironment. However, these pathways were tested in the same cohorts used
for model evaluation and are not an independent discovery/validation result.
They should define the next mechanistic hypothesis, not a publication claim.

## Nature-route decision

The simple claim that a frozen Txn_Jatin vector is a universal ICI-response
biomarker is rejected by this experiment. Continuing to optimize only the
linear probe would not address the dominant cohort geometry.

The defensible next route is:

1. Expand to the full accessible COMPASS cohort collection and freeze a
   patient-level manifest before modeling.
2. Remove all evaluation accessions from pretraining or designate completely
   unseen cohorts as locked external tests.
3. Train a response-aware, domain-invariant Txn adapter using LoRA or gradual
   upper-layer unfreezing on training cohorts only.
4. Compare against raw expression, T-cell-inflamed/IFN-gamma signatures,
   Hallmark scores, TMB, PD-L1, and a clinical baseline.
5. Evaluate RECIST response, progression-free survival, overall survival,
   calibration, and decision-curve utility on locked cohorts.
6. Test whether the model discovers an interferon-active responder program
   plus a distinct resistance program, then validate selected genes in
   independent tissue or perturbation data.

For a Nature-level story, predictive improvement must be consistent across
cohorts and accompanied by a mechanistic finding. The current experiment
provides the correct negative control and identifies a plausible immune axis;
it does not yet satisfy either requirement.

## Reproducibility

The experiment folder contains:

- `raw/`: downloaded public expression, clinical, annotation, and GEO metadata
- `prepared/`: harmonized labels, matrices, accession map, and preparation QC
- `scripts/`: preparation, H100 extraction, evaluation, biology, and audit code
- `results/ici_initial_results.xlsx`: consolidated workbook
- `results/`: all predictions, metrics, confidence intervals, pathway tests,
  nuisance diagnostics, and pretraining-overlap tables
- `figures/`: eight presentation-ready figures
- `logs/`: local and remote execution logs

The remote extraction used the final checkpoint without changing model
weights. All supervised evaluation and exploratory biological analyses were
performed locally from frozen outputs.
