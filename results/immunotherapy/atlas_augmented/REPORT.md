# Atlas-augmented immunotherapy evaluation

**Status:** Complete exploratory analysis  
**Patients:** 239 across Gide, Riaz, Hugo, and Rose  
**Validation:** Nested leave-one-cohort-out (LOCO)

## Question

Can a fixed gene-atlas feature layer and cohort-normalized Txn_Jatin
representation improve held-out-cohort immunotherapy response prediction?

The atlas contained 50 MSigDB Hallmark programs and nine predeclared immune
programs covering T-cell inflammation, cytolysis, antigen presentation,
interferon signaling, exhaustion, Treg, myeloid suppression, TGF-beta, and
angiogenesis. Scores used within-sample expression ranks to reduce dependence
on assay scale. No response labels were used to define these features.

## Leakage controls

- Each outer test cohort was excluded from classifier and hyperparameter
  training.
- Logistic-regression `C` was chosen by inner leave-one-training-cohort-out
  AUROC.
- Raw-expression variance selection and scaling were fitted inside each split.
- Txn features were centered by cohort without response labels. Test-cohort
  centering is therefore transductive but label-free.
- Calibration used a monotone intercept adjustment fitted to inner out-of-
  cohort predictions. It cannot reverse score rankings.
- The LLM receives only the final structured evidence and does not participate
  in prediction, calibration, feature selection, or pathway testing.

## Held-out-cohort results

| Method | Macro AUROC | Macro AUPRC | Macro Brier |
|---|---:|---:|---:|
| Raw expression | **0.6535** | **0.5132** | **0.2449** |
| Raw + atlas | 0.6361 | 0.4755 | 0.2528 |
| Atlas only | 0.6339 | 0.5038 | 0.3007 |
| Raw + Txn + atlas | 0.5145 | 0.3969 | 0.3486 |
| Txn + atlas | 0.4662 | 0.3667 | 0.4705 |
| Cohort-centered Txn | 0.4103 | 0.3446 | 0.4818 |

The atlas did **not** improve the prespecified discrimination endpoints.
Atlas-only features were competitive with raw expression and generalized well
to Gide (AUROC 0.728) and Rose (0.673), but the gain was not consistent:
Hugo remained weak (0.476). Concatenating atlas features with raw expression
reduced macro AUROC by 0.0175 and macro AUPRC by 0.0377.

## Calibration

For the unchanged raw-expression classifier, nested monotone recalibration
reduced macro Brier score from 0.2613 in the original run to 0.2449. This is a
real calibration improvement, while discrimination remained exactly the same
at macro AUROC 0.6535 and macro AUPRC 0.5132.

Pooled AUROC/AUPRC are retained in the output tables but are secondary because
cohort-specific calibration offsets alter cross-cohort score ordering. Macro
held-out-cohort metrics are the primary comparison.

## Pathway consistency

Antigen presentation had the largest positive random-effects estimate
(`g=0.646`) and was positive in all four cohorts. Interferon-alpha response was
also positive in all four (`g=0.542`). T-cell-inflamed, IFN-gamma, exhaustion,
and cytolytic programs were positive in three of four cohorts.

No pathway survived Benjamini-Hochberg correction across the 59 tested
programs (minimum `q=0.297`). These effects are biologically coherent
hypotheses, not validated predictive biomarkers. The earlier mean-expression
Hallmark analysis and this rank-based atlas analysis use different score
definitions, so their q-values should not be interchanged.

## Conclusion

The gene atlas and LLM layer do not solve the frozen-model immunotherapy
problem. They provide:

1. a compact, biologically interpretable predictor that approaches raw
   expression;
2. improved probability calibration for the raw model;
3. reproducible antigen-presentation and interferon hypotheses; and
4. an auditable language-summary interface.

They do not provide a consistent AUROC/AUPRC gain. The limiting factor remains
cross-cohort response transport, especially Hugo, rather than absence of gene
annotations or narrative interpretation. The next model-development step
should use additional training cohorts and a response-aware, domain-invariant
adapter, with one or more cohorts locked until final evaluation.

## Files

- `atlas_loco_predictions.csv`: every outer-cohort probability
- `atlas_loco_metrics.csv`: cohort and pooled discrimination/calibration
- `atlas_loco_summary.csv`: model-level macro summary
- `atlas_pathway_effects_by_cohort.csv`: cohort-specific Hedges g
- `atlas_pathway_consistency.csv`: random-effects estimates and FDR
- `atlas_evidence.json`: structured evidence supplied to the language layer
- `llm_prompt.txt`: exact constrained language-model prompt
