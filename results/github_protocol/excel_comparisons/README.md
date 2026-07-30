# GitHub protocol model comparison workbooks

These workbooks were rebuilt from the completed remote results for
`ylaboratory/gene-embedding-benchmarks` commit
`d1320026a2a4ee033d49517f91e2d1c2ccc8df1e`.

## Workbooks

- `01_GitHub_Protocol_All_Models.xlsx`: all 29 benchmark groups, all metrics,
  model summaries, winners, numeric source rows, and protocol notes.
- `02_Gene_Level_All_Models.xlsx`: GO, OMIM, and Temporal GO comparisons.
- `03_Gene_Pair_All_Models.xlsx`: SL, POMBE, TF, and NG comparisons for
  all-gene/intersection scopes and sum/product/concat operators.
- `04_Model_Rankings_and_Winners.xlsx`: model means, ranks, and per-task winners.

All workbooks contain the same 11 selected models. The highest value in every
benchmark row is bold and highlighted; ties are all highlighted. AUROC, AUPRC,
and PR@10 are reported, and higher is better for all three metrics.

The completed retained scope contains 29 benchmark groups and 319 model-level
results with zero failures. ANDES gene-set tasks, including KEGG-GO and
disease-tissue comparisons, were excluded by user request and are not included.
