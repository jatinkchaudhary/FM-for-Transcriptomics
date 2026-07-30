# TCGA-CDR survival and progression benchmark

## Protocol

Primary-tumor RNA-seq samples were deduplicated to one observation per TCGA patient
and linked to the Liu et al. TCGA Clinical Data Resource. Overall survival (OS) and
progression-free interval (PFI) were evaluated separately. The processed expression
matrix contains five cancers, so the analysis reports those five rather than claiming
the approximately ten cancers anticipated in the plan.

- Cancers: BRCA, KIRC, LUAD, LUSC, and SKCM.
- Models: elastic-net Coxnet, identical alpha grid and five outer folds.
- Hyperparameter selection: three-fold inner CV within each outer-training fold.
- Metric: OOF Harrell C-index.
- Uncertainty: 2,000 event-stratified bootstrap replicates for each C-index and
  1,000 paired replicates for each ESPRESSO-minus-comparator delta.
- Representations: only adapters that passed the frozen TCGA validity gate.
- Patient identifiers never cross folds.

The common alpha grid was `0.01` to `10`. Coxnet candidates below `0.1` were
numerically invalid for the 15,165-feature raw baseline and were excluded for that
representation; all valid grid candidates remained available to all models.

## Pan-cancer results

| Representation | OS C-index (95% CI) | PFI C-index (95% CI) |
|---|---:|---:|
| PCA-64 | 0.6849 (0.6665-0.7021) | **0.7003** (0.6833-0.7164) |
| Raw log1p-TPM | **0.6778** (0.6587-0.6964) | 0.6909 (0.6744-0.7085) |
| ESPRESSO | 0.6747 (0.6564-0.6928) | 0.6693 (0.6530-0.6862) |
| BulkFormer 147M | 0.6476 (0.6282-0.6666) | 0.6647 (0.6476-0.6815) |

For OS, ESPRESSO beat BulkFormer by **+0.0271** C-index (paired 95% CI
0.0093 to 0.0450), tied raw expression (-0.0031, -0.0176 to 0.0115), and
tied PCA-64 (-0.0102, -0.0240 to 0.0050).

For PFI, ESPRESSO tied BulkFormer (+0.0047, -0.0108 to 0.0196) but was below
raw expression (-0.0216, -0.0324 to -0.0097) and PCA-64 (-0.0310, -0.0427
to -0.0179).

## Per-cancer pattern

Across the six scopes (pan-cancer plus five cancers), ESPRESSO had a higher point
estimate than BulkFormer in 4/6 OS and 5/6 PFI comparisons. It exceeded raw in 4/6
OS and 3/6 PFI comparisons, but exceeded PCA in only 1/6 OS and 2/6 PFI
comparisons. Cancer-specific intervals are broad, especially in BRCA, and should not
be converted into rank claims without consulting the paired-delta table.

## Conclusion

ESPRESSO carries clinically relevant prognostic signal and materially improves on
BulkFormer for pan-cancer OS, but it does **not** outperform simple expression
baselines universally. The strongest defensible claim is representation efficiency
and competitiveness, not state-of-the-art survival prediction. This null/qualified
result is important for a manuscript because it separates biological embedding
quality from endpoint-specific sufficiency.
