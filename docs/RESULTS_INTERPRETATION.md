# Results interpretation

## What Txn_Jatin improved

Txn_Jatin improved several functional-ranking endpoints. Its static checkpoint
was strong on GO, the epoch-7 representation led BRIDGE in centroid functional
prediction, and the final decoder led TCGA expressed-gene AUROC under whole-gene
masking.

## Where BRIDGE remained stronger

BRIDGE retained better global reconstruction geometry: higher PCC/lower MSE and
stronger OSDR whole-gene recovery. BRIDGE also ranked first in the aggregated
dynamic-context benchmark.

## What sequence controls show

ESM2 and ESM3 led the strict individual-GO experiment. This is coherent with
their direct protein-sequence signal and demonstrates that GO performance alone
does not establish superior expression modeling.

## What PEFT shows

Gradually unfreezing the last three layers minimized OSDR masked MSE, but harmed
flight-vs-ground separability. Last-three-layer attention+FFN LoRA preserved the
best AUROC and was selected by the predefined combined-rank rule. Reconstruction
and biological classification are distinct objectives.

## What sparse panels and immunotherapy show

Literal recovery from 1,000/2,000-gene panels was near chance, so the current
checkpoint is not a sparse clinical-panel imputer. In four immunotherapy
cohorts, raw expression outperformed Txn-completed expression and contextual
features were heavily cohort-encoded. Concordant interferon pathway effects are
biologically plausible exploratory observations, not validated response
biomarkers.

## Publication-safe conclusion

Txn_Jatin is a biologically regularized expression representation with
task-specific gains, not a universal replacement for BRIDGE or sequence
foundation models. The evidence supports a mechanistic representation-learning
paper and motivates prospective, disease-specific adaptation and external
validation.
