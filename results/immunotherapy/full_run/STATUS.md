# Status

## Completed

- Four public pretreatment ICI cohorts downloaded and fully patient-matched
- 239 patients harmonized to RECIST R/NR labels
- Frozen epoch-20 Txn_Jatin representations extracted on the remote H100
- Missing-assay genes handled with the trained mask token
- Four-cohort nested leave-one-cohort-out evaluation completed
- Raw-expression, Hallmark, Txn-context, pooling, and completion arms compared
- Patient bootstrap confidence intervals and paired deltas completed
- Cohort-nuisance geometry quantified
- Exploratory Hallmark random-effects meta-analysis completed
- Exact ARCHS4 pretraining-accession audit completed
- Clean three-cohort sensitivity excluding Hugo completed
- Workbook, figures, report, scripts, and logs packaged locally

## Decision

The frozen Txn_Jatin sample representation is not ready for publication as an
ICI-response predictor. Txn-completed expression preserves raw-expression
signal, and interferon-related responder biology is a credible next hypothesis.

## Not started

- Response-aware LoRA/domain-invariant adaptation
- Expansion to the full COMPASS cohort collection
- Locked external survival and clinical-utility validation
- Experimental validation of the interferon/resistance program
