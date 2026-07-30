# Multi-model immunotherapy benchmark

**Status:** Complete  
**Patients:** 239  
**Cohorts:** Gide, Riaz, Hugo, and Rose  
**Primary validation:** Nested leave-one-cohort-out

## Protocol

Eight validated expression decoders were run on the H100: Txn_Jatin,
Txn_Jatin OSDR-LoRA, BRIDGE, and BulkFormer 37M/50M/93M/127M/147M. Observed
expression was retained and assay-missing genes were filled by each decoder.

All decoder tracks used the same 15,171-gene finite intersection. The raw
control used the 13,523 genes measured in every cohort, matching the established
immunotherapy protocol. Top-1,000 variance selection, scaling, logistic-
regression tuning, and monotone calibration were fitted without access to the
outer held-out cohort's response labels.

ESM2, ESM3, Geneformer, scGPT, and the precomputed Txn contextual gene matrix
have no validated bulk patient decoder or frozen bulk sample encoder. They are
reported as unsupported rather than replaced by raw expression.

## Primary results

| Model | Macro AUROC | Macro AUPRC | Macro Brier |
|---|---:|---:|---:|
| Raw measured expression | **0.6535** | **0.5132** | 0.2449 |
| BRIDGE | 0.6501 | 0.5103 | 0.2452 |
| Txn_Jatin OSDR-LoRA | 0.6496 | 0.5085 | 0.2466 |
| BulkFormer 127M | 0.6493 | 0.5125 | **0.2432** |
| BulkFormer 37M | 0.6480 | 0.5112 | 0.2445 |
| BulkFormer 147M | 0.6449 | 0.5090 | 0.2447 |
| BulkFormer 50M | 0.6423 | 0.5094 | 0.2435 |
| BulkFormer 93M | 0.6422 | 0.5081 | 0.2439 |
| Txn_Jatin | 0.6360 | 0.5071 | 0.2456 |

No decoder improved both held-out-cohort discrimination endpoints over raw
expression. BRIDGE had the highest decoder AUROC, BulkFormer-127M had the
highest decoder AUPRC and lowest Brier score, and Txn OSDR-LoRA ranked between
them. Model size did not produce monotonic improvement within BulkFormer.

## Paired uncertainty

A 2,000-replicate cohort-and-label-stratified paired bootstrap was used for
the final uncertainty analysis. BRIDGE, Txn OSDR-LoRA, and all BulkFormer
AUROC/AUPRC difference intervals included zero relative to raw expression.
Final Txn_Jatin had a macro AUROC delta of -0.0176 with screening interval
-0.0435 to 0.0056; its AUPRC interval also included zero. Thus, after the
predeclared higher-resolution resampling run, none of the decoder differences
from raw measured expression was statistically resolved by these intervals.

## Cohort behavior

The models were extremely similar within cohorts. For example, held-out Riaz
AUROC ranged from 0.676 for Txn_Jatin to 0.700 for Txn OSDR-LoRA, while
held-out Rose ranged from 0.662 to 0.678. Hugo remained the limiting cohort:
raw expression achieved 0.554, BulkFormer-127M 0.548, BRIDGE 0.536, and
Txn_Jatin 0.500.

## Atlas track

Atlas scores were generated separately from each completed matrix. Their macro
AUROCs ranged from 0.631 to 0.642, below the strongest completed-expression
tracks, and their Brier scores were materially worse. The atlas remains useful
for biological interpretation but does not improve patient-level prediction.

## Conclusion

The result is a near-tie among raw expression, BRIDGE, Txn OSDR-LoRA, and
BulkFormer-127M. Decoder completion mostly preserves response signal; it does
not create a stronger transportable response axis. Txn_Jatin's weaker final
checkpoint result and its recovery after OSDR LoRA suggest that domain
adaptation matters more than the base architecture for this task.

The appropriate next experiment is response-aware, domain-invariant adaptation
using training cohorts only, followed by evaluation on a newly locked cohort.
Model selection on these same four cohorts should now stop to avoid adaptive
overfitting.

## Files

- `multimodel_loco_summary.csv`
- `multimodel_loco_metrics.csv`
- `multimodel_loco_predictions.csv`
- `paired_bootstrap_vs_raw.csv`
- `unsupported_embedding_only_models.csv`
- `protocol.json`
- `multimodel_primary_comparison.png`
