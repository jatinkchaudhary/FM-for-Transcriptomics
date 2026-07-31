# DepMap 24Q2 cold-gene essentiality benchmark

**Status:** Complete

**Primary model:** ESPRESSO (`Txn_Jatin`, final 20-epoch static gene table)

**Data:** DepMap Public 24Q2, DOI `10.25452/figshare.plus.25880521.v1`

## Question

Do frozen ESPRESSO gene embeddings encode common-essential gene status better
than BRIDGE, single-cell foundation models, the ESM2 sequence prior, and five
BulkFormer checkpoints?

## Protocol

The positive class is DepMap's inferred common-essential set. The negative class
contains screened genes whose median Chronos effect is at least `-0.2`;
intermediate genes are excluded rather than mislabeled. Evaluation uses the
strict 13,342-gene intersection across every model: 1,666 positives and 11,676
pan-neutral negatives.

Every representation uses the same five stratified, gene-disjoint outer folds,
the same four-fold inner regularization search, and the same class-weighted
linear probe. All reported intervals and ESPRESSO-minus-comparator deltas use
2,000 paired, label-stratified bootstrap replicates (`seed=42`).

## Results

| Model | AUROC (95% CI) | AUPRC (95% CI) |
|---|---:|---:|
| **ESPRESSO** | **0.934 (0.927-0.940)** | **0.743 (0.724-0.762)** |
| BRIDGE | 0.930 (0.923-0.936) | 0.702 (0.680-0.723) |
| Geneformer | 0.929 (0.922-0.935) | 0.699 (0.678-0.720) |
| scGPT | 0.926 (0.919-0.933) | 0.702 (0.680-0.724) |
| ESPRESSO contextual | 0.904 (0.895-0.912) | 0.656 (0.634-0.677) |
| ESM2 PCA-512 prior | 0.890 (0.881-0.899) | 0.601 (0.578-0.623) |
| BulkFormer-147M | 0.888 (0.880-0.897) | 0.562 (0.538-0.587) |
| BulkFormer-93M | 0.887 (0.879-0.895) | 0.553 (0.530-0.577) |
| BulkFormer-50M | 0.881 (0.873-0.888) | 0.540 (0.518-0.563) |
| BulkFormer-127M | 0.877 (0.869-0.886) | 0.524 (0.501-0.548) |
| BulkFormer-37M | 0.861 (0.852-0.870) | 0.487 (0.465-0.512) |
| Mean DepMap expression | 0.859 (0.850-0.867) | 0.470 (0.448-0.492) |
| Gene length | 0.496 (0.481-0.511) | 0.123 (0.118-0.128) |

ESPRESSO exceeds BulkFormer-147M by `+0.045 AUROC` (95% CI
`+0.039` to `+0.052`) and `+0.181 AUPRC` (`+0.157` to `+0.203`).
It exceeds ESM2 by `+0.043 AUROC` (`+0.038` to `+0.049`) and
`+0.142 AUPRC` (`+0.127` to `+0.157`).

Against BRIDGE, the AUROC difference is a statistical tie (`+0.004`, CI
`-0.001` to `+0.008`), while ESPRESSO has significantly higher AUPRC
(`+0.041`, CI `+0.023` to `+0.060`). ESPRESSO also significantly exceeds
scGPT and Geneformer on both metrics, although their AUROC gaps are small.

## Interpretation

This is direct evidence that the 46M-parameter ESPRESSO gene table captures
pan-cancer viability structure beyond protein sequence, mean expression, and
gene length. The AUPRC advantage is particularly relevant because common
essentials comprise only 12.5% of the paired evaluation set.

The result supports representation-level biological content; it does not imply
that ESPRESSO predicts cell-line-specific dependencies or drug response. Those
are separate sample-conditioned tasks.

## Boundaries

- DepMap labels are supervised probe targets and were not used during ESPRESSO
  pretraining.
- Public training-corpus overlap cannot be ruled out equally for every
  competitor; no DepMap labels enter any embedding model in this evaluation.
- Fourteen missing gene-length values and five missing expression values were
  median-imputed without labels to preserve the identical paired gene set.
- The static ESPRESSO table outperforms its corpus-averaged contextual table on
  this gene-identity task; contextual averaging should not be presumed superior
  for every endpoint.
