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

## What the two GO experiments show

ESM2 and ESM3 led the earlier 40-term OSDR linear-probe experiment, which is
coherent with their direct protein-sequence signal. Under the supplied,
pinned GitHub protocol on TCGA genes, Txn_Jatin instead ranked first across all
56 terms: mean AUROC was 0.8353 on each model's native TCGA genes and 0.8439 on
the identical 9,568-gene intersection. ESM2 ranked second, Txn_Jatin contextual
third, and native 1,536-dimensional ESM3 fourth in both scopes. The reversal
shows that conclusions depend on the annotation set, split construction, probe,
and eligible-gene universe; GO performance alone does not establish superior
expression reconstruction.

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

## What external GTEx recovery shows

On 12 diverse recount2 GTEx samples and the strict 2,117-gene decoder
intersection, Txn_Jatin produced the lowest MSE (0.1326) and highest Pearson
correlation (0.9787). OSDR LoRA was close (MSE 0.1484; Pearson 0.9779), followed
by BRIDGE (MSE 0.2377; Pearson 0.9614). The BulkFormer decoders were materially
weaker under this exact preprocessing and mask. This is evidence of external
reconstruction transfer, not evidence of clinical validity or a fair embedding
benchmark for models without a decoder.

## Publication-safe conclusion

Txn_Jatin is a biologically regularized expression representation with
task-specific gains, not a universal replacement for BRIDGE or sequence
foundation models. The evidence supports a mechanistic representation-learning
paper and motivates prospective, disease-specific adaptation and external
validation.
