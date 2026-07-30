# Whole-gene masking and cancer-gene benchmark

This directory is the compact local mirror of the active remote run:

`/media/volume/AdditionalHeadroom/Txn_Jatin_runs/full_ARCHES4_20epoch_H100_20260624_015238/benchmarks/whole_gene_mask_2c5c_20260723_132500`

The complete stakeholder-facing scientific interpretation is in
[`STAKEHOLDER_TECHNICAL_REPORT.md`](STAKEHOLDER_TECHNICAL_REPORT.md).

## Frozen protocol

- Shared universe: 14,585 genes present in TCGA, OSDR, Txn_Jatin, BRIDGE, and BulkFormer.
- Masking: the same 2,188 genes (15%) are hidden across all samples for each of three fixed seeds.
- Datasets: processed TCGA `log1p(TPM)` and prepared OSDR `log1p(CPM)`.
- Native decoders: Txn_Jatin, BRIDGE, and BulkFormer 37M/50M/93M/127M/147M.
- Non-decoder models: ESM2, Geneformer, scGPT, and Txn_Jatin_contextual are reported as `NaN` for imputation.
- Cancer task: binary tumor-versus-normal RF for LUAD+LUSC and for BRCA+KIRC+LUAD+LUSC+SKCM.

## Completion

All seven native decoder models completed three TCGA masks, three OSDR masks,
imputation analysis, and the 2-cancer and 5-cancer RF analyses. The four
embedding-only models remain explicit `NaN` entries for imputation.

## Imputation results

Means across three whole-gene masks:

| Model | TCGA PCC | TCGA AUROC | OSDR PCC | OSDR AUROC |
|---|---:|---:|---:|---:|
| BRIDGE | **0.9155** | 0.9121 | **0.9026** | **0.9550** |
| Txn_Jatin | 0.9087 | **0.9284** | 0.8023 | 0.8620 |
| BulkFormer-147M | 0.8931 | 0.8885 | 0.7576 | 0.8273 |
| BulkFormer-127M | 0.8907 | 0.8867 | 0.7634 | 0.8315 |
| BulkFormer-93M | 0.8782 | 0.8677 | 0.7527 | 0.8340 |
| BulkFormer-50M | 0.8742 | 0.8514 | 0.7593 | 0.8397 |
| BulkFormer-37M | 0.8731 | 0.8456 | 0.7590 | 0.8362 |

Detailed completed results:

Txn_Jatin, mean across three masks:

| Dataset | PCC | Spearman | MSE | MAE | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| TCGA | 0.9087 | 0.9115 | 0.4444 | 0.3950 | 0.9284 | 0.8696 |
| OSDR | 0.8023 | 0.7924 | 1.3079 | 0.7508 | 0.8620 | 0.8796 |

BulkFormer-37M, mean across three masks:

| Dataset | PCC | Spearman | MSE | MAE | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| TCGA | 0.8731 | 0.8743 | 0.6083 | 0.5364 | 0.8456 | 0.7859 |
| OSDR | 0.7590 | 0.7492 | 1.5376 | 0.9090 | 0.8362 | 0.8526 |

Raw-expression RF controls:

| Panel | OOF AUROC | Fold F1 | Balanced accuracy |
|---|---:|---:|---:|
| 2 cancer | 0.9995 | 0.9967 | 0.9831 |
| 5 cancer | 0.9953 | 0.9958 | 0.9637 |

The compact tables are under `results/current`. Remote predictions remain on
attached storage and are intentionally excluded from the local mirror.
