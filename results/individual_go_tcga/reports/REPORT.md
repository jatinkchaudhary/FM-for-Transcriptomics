# TCGA individual GO benchmark

## Protocol

This run uses the pinned `ylaboratory/gene-embedding-benchmarks` GO protocol at commit `d1320026a2a4ee033d49517f91e2d1c2ccc8df1e`. All 56 official GO terms, fixed nested-CV folds, C selection, holdout sets, and metrics are preserved.

The TCGA scope is defined by genes present in the processed 3,481-sample TCGA TPM matrix. The model representations are the same frozen artifacts used by the completed pinned GitHub run; ESM3 is added as a real frozen sequence embedding. TCGA expression values are not used as GO labels or as a replacement probing protocol.

## Model summary

| scope | model | terms | mean_AUROC | mean_AUPRC | mean_PR_at_10 | AUROC_rank |
|---|---|---|---|---|---|---|
| tcga_all_genes | Txn_Jatin | 56 | 0.8353 | 0.5702 | 0.6732 | 1.0000 |
| tcga_all_genes | ESM2_PCA512_prior | 56 | 0.8250 | 0.5467 | 0.6554 | 2.0000 |
| tcga_all_genes | Txn_Jatin_contextual | 56 | 0.8192 | 0.5211 | 0.6286 | 3.0000 |
| tcga_all_genes | ESM3 | 56 | 0.8101 | 0.4957 | 0.5768 | 4.0000 |
| tcga_all_genes | scGPT | 56 | 0.7823 | 0.4158 | 0.5232 | 5.0000 |
| tcga_all_genes | BRIDGE | 56 | 0.7768 | 0.3861 | 0.4929 | 6.0000 |
| tcga_all_genes | Geneformer | 56 | 0.7603 | 0.3702 | 0.4893 | 7.0000 |
| tcga_all_genes | BulkFormer_147M | 56 | 0.6965 | 0.2601 | 0.3036 | 8.0000 |
| tcga_all_genes | BulkFormer_93M | 56 | 0.6962 | 0.2530 | 0.3071 | 9.0000 |
| tcga_all_genes | BulkFormer_50M | 56 | 0.6928 | 0.2546 | 0.3161 | 10.0000 |
| tcga_all_genes | BulkFormer_37M | 56 | 0.6872 | 0.2466 | 0.3036 | 11.0000 |
| tcga_all_genes | BulkFormer_127M | 56 | 0.6754 | 0.2476 | 0.3179 | 12.0000 |
| tcga_strict_intersection | Txn_Jatin | 56 | 0.8439 | 0.5785 | 0.5857 | 1.0000 |
| tcga_strict_intersection | ESM2_PCA512_prior | 56 | 0.8323 | 0.5547 | 0.5607 | 2.0000 |
| tcga_strict_intersection | Txn_Jatin_contextual | 56 | 0.8230 | 0.5431 | 0.5518 | 3.0000 |
| tcga_strict_intersection | ESM3 | 56 | 0.8126 | 0.4919 | 0.5179 | 4.0000 |
| tcga_strict_intersection | scGPT | 56 | 0.7847 | 0.4139 | 0.4375 | 5.0000 |
| tcga_strict_intersection | BRIDGE | 56 | 0.7839 | 0.4158 | 0.4446 | 6.0000 |
| tcga_strict_intersection | Geneformer | 56 | 0.7621 | 0.3812 | 0.4107 | 7.0000 |
| tcga_strict_intersection | BulkFormer_50M | 56 | 0.7043 | 0.2799 | 0.3107 | 8.0000 |
| tcga_strict_intersection | BulkFormer_93M | 56 | 0.7002 | 0.2674 | 0.3000 | 9.0000 |
| tcga_strict_intersection | BulkFormer_147M | 56 | 0.6995 | 0.2802 | 0.3054 | 10.0000 |
| tcga_strict_intersection | BulkFormer_37M | 56 | 0.6934 | 0.2663 | 0.2839 | 11.0000 |
| tcga_strict_intersection | BulkFormer_127M | 56 | 0.6749 | 0.2670 | 0.2893 | 12.0000 |

## Validation

- Models: **12**
- Official GO terms per model and scope: **56**
- Long-table rows: **1344**
- Missing or non-finite metrics: **0**
- Fallback embeddings: **0**
- Strict shared genes: **9,568**

## Interpretation boundary

GO annotations are gene-level labels. Calling this a TCGA benchmark means that the eligible gene universe is restricted to the processed TCGA matrix; it does not mean that TCGA samples carry GO labels. This distinction keeps the result directly comparable with the supplied GitHub benchmark.
