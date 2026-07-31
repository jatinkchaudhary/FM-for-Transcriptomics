# Cross-study bulk integration

## Design

The cohort is a crossed 4-tissue by 2-study design constructed from uniformly
processed recount2 count matrices: GTEx and TCGA, with breast, kidney, lung, and
skin represented in both studies. It contains 1,339 samples after retaining primary
TCGA tumors and sampling at most 200 observations per tissue-study cell. The exact
contingency table is in `tissue_study_contingency.csv`.

This is an explicit, documented substitution of pinned recount2 matrices for the
recount3 cohort requested in the original plan. It preserves the critical crossed
design but should not be described as a recount3 analysis.

## Results

| Method | iLISI | kBET accept. | cLISI | ASW tissue | ARI | NMI | Batch score | Biology score | Combined |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ESPRESSO | 1.024 | 0.010 | 1.026 | 0.588 | 0.373 | 0.550 | 0.328 | **0.687** | **0.544** |
| Raw log1p-CPM | 1.051 | 0.021 | 1.077 | 0.538 | 0.299 | 0.470 | 0.356 | 0.647 | 0.531 |
| Harmony PCA-64 | 1.235 | **0.149** | 1.402 | 0.514 | 0.214 | 0.395 | **0.461** | 0.567 | 0.525 |
| BulkFormer 147M | 1.038 | 0.016 | 1.063 | 0.563 | 0.248 | 0.370 | 0.335 | 0.615 | 0.503 |
| PCA-64 | 1.219 | 0.136 | 1.396 | 0.513 | 0.075 | 0.271 | 0.451 | 0.515 | 0.489 |

## Interpretation

ESPRESSO best preserved tissue biology and consequently had the highest prespecified
combined score. It did **not** remove study effects strongly: Harmony and PCA mixed
GTEx and TCGA more effectively. This is a trade-off, not a universal integration
win. ESPRESSO appears to retain biologically useful structure while leaving
substantial consortium/platform structure intact.

Against gated BulkFormer, the paired sample-neighborhood bootstrap confirmed higher
local biology preservation (+0.0197, 95% CI 0.0150 to 0.0244) and a small positive
combined difference (+0.0060, 0.0023 to 0.0100), alongside worse local batch mixing
(-0.0144, -0.0207 to -0.0081; 2,000 tissue-study-stratified replicates).

The accompanying UMAPs are colored separately by tissue and study. Neighbor count,
normalization, balancing, and metric definitions were held fixed across all
representations.
