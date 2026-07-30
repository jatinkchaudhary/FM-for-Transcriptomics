# Native sample-embedding validity gate

All models were evaluated on the same 3,481 TCGA samples. The gate was fixed before
survival/integration: cancer-type silhouette must be greater than zero and five-fold
macro-AUROC must be at least 0.90. No fallback embedding was permitted.

| Model | Silhouette | Cancer macro-AUROC | Status |
|---|---:|---:|---|
| Raw log1p-TPM | 0.0493 | 0.9954 | Pass |
| BulkFormer 147M | 0.0362 | 0.9907 | Pass |
| ESPRESSO / Txn_Jatin | 0.0303 | 0.9953 | Pass |
| PCA-64 | 0.0131 | 0.9965 | Pass |
| BRIDGE | -0.0022 | 0.9941 | **Fail** |

BRIDGE was excluded from subsequent sample-level benchmarks because it failed the
predeclared silhouette criterion, despite strong supervised probe performance.
scGPT and Geneformer were not entered because validated native bulk-sample encoders
were unavailable in the pinned remote environment; no surrogate or web fallback was
substituted.
