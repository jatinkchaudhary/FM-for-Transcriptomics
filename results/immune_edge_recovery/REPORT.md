# Immune-program regulatory and synthetic-lethal edge recovery

**Status:** Complete  
**Model:** Final 20-epoch Txn_Jatin static gene embedding  
**Benchmark source:** `ylaboratory/gene-embedding-benchmarks` commit
`d1320026a2a4ee033d49517f91e2d1c2ccc8df1e`  
**Compute:** H100 for edge-head training; CPU for correlation and resampling

## Scientific question

Does Txn_Jatin recover held-out immune regulatory or synthetic-lethal
relationships that cannot be explained by a conventional co-expression
baseline?

The immune scope was fixed before evaluation as the union of the nine
immunotherapy programs used in the preceding atlas analysis. An edge is
immune-scoped when at least one endpoint belongs to this set.

## Leakage control

The supplied GitHub pair folds are gene-disjoint. For each outer fold, both
endpoints of every test pair are held-out genes; training pairs contain only
training genes. Edges crossing the train/test gene partitions are omitted.
This is therefore a cold-gene test rather than a random edge split.

The co-expression baseline is Spearman correlation across 239 independent,
pretreatment samples from the Gide, Riaz, Hugo, and Rose immunotherapy cohorts.
Response labels are never used. Each fold compares:

1. Co-expression alone.
2. Txn_Jatin pair features alone.
3. Txn_Jatin pair features plus co-expression.

All three use the same fixed class-weighted linear edge head. Uncertainty is
estimated with 2,000 paired, label-stratified bootstrap replicates.

## Primary results

| Task and scope | Positives / edges | Co-expression AUROC / AUPRC | Txn_Jatin AUROC / AUPRC | AUROC delta (95% CI) | AUPRC delta (95% CI) |
|---|---:|---:|---:|---:|---:|
| Regulatory, all held-out | 2,504 / 27,488 | 0.511 / 0.092 | **0.788 / 0.304** | **+0.277** (+0.262, +0.291) | **+0.213** (+0.196, +0.229) |
| Regulatory, immune | 13 / 264 | 0.432 / 0.048 | **0.743 / 0.248** | **+0.314** (+0.062, +0.552) | **+0.211** (+0.050, +0.417) |
| Synthetic lethal, all held-out | 611 / 6,174 | 0.582 / 0.127 | 0.605 / 0.140 | +0.022 (-0.011, +0.054) | +0.014 (-0.004, +0.031) |
| Synthetic lethal, immune | 1 / 41 | 0.200 / 0.030 | 0.075 / 0.026 | Not interpretable | Not interpretable |

Adding co-expression to Txn_Jatin did not improve the regulatory result
(immune AUROC 0.732 versus 0.743). This indicates that the recovered signal is
not merely expression correlation measured in these cohorts.

## Regulatory associations missed by co-expression

Five true immune-scoped held-out regulatory associations ranked in the top
decile under Txn_Jatin but outside the co-expression top decile:

| Gene pair | Txn_Jatin percentile | Co-expression percentile |
|---|---:|---:|
| CMKLR1 - TBX18 | 99.7 | 36.1 |
| IL10 - TBX22 | 98.7 | 89.5 |
| NR4A1 - SERPINE1 | 95.6 | 32.4 |
| ZBTB6 - CCR8 | 93.8 | 29.6 |
| ESM1 - SCRT1 | 92.8 | 18.0 |

These are benchmark associations, not newly discovered causal edges. The
upstream GitHub data preparation sorts each TF-target pair, discarding its
original direction. Accordingly, this experiment validates recovery of
regulatory association structure but cannot identify which endpoint regulates
the other.

## Interpretation

The immune-program observation now has a validated downstream capability:
Txn_Jatin generalizes regulatory association structure to genes absent from
edge-head training, and it substantially outperforms a context-matched
co-expression baseline. The result supports the hypothesis that the embedding
contains nonlocal regulatory information beyond pairwise expression
correlation.

The synthetic-lethal result does not support the same claim. The overall point
estimate favors Txn_Jatin only modestly and its paired intervals include zero.
The immune-specific SL subset is underpowered because only one positive edge
survives the cold-gene folds. A dedicated immune-oncology SL resource or
prospective CRISPR screen is required for that question.

## Boundaries

- The edge head is supervised on training edges; Txn_Jatin is not being used as
  a zero-shot causal predictor.
- The comparison establishes discrimination, not mechanism or intervention
  response.
- Txn_Jatin pretraining may include expression studies related to the public
  cohorts, although neither TF nor SL edge labels were used during pretraining.
- The five highlighted associations require independent database or
  perturbational confirmation before being presented as biological findings.

## Files

- `summary_metrics.csv`: AUROC/AUPRC by task, scope, and method.
- `paired_bootstrap_vs_coexpression.csv`: paired deltas and 95% intervals.
- `heldout_edge_predictions.csv`: all fold-level predictions.
- `immune_edges_txn_recovers_coexpression_misses.csv`: top-decile recovered
  regulatory associations.
- `coverage.csv`: source and eligible edge counts.
- `protocol.json`: fixed protocol, hashes, and runtime metadata.
- `immune_edge_recovery.png` and `.pdf`: manuscript-oriented comparison.
