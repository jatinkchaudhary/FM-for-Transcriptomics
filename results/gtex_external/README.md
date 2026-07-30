# GTEx all-model whole-gene imputation benchmark

## Protocol

- 12 GTEx samples from distinct tissues.
- 15,313 input genes shared with the Txn_Jatin vocabulary.
- The same seeded 15% whole-gene mask was applied to every sample.
- 2,297 genes were hidden in the Txn_Jatin-native evaluation.
- The strict comparison uses 2,117 masked genes supported by all eight native
  expression decoders.
- Input values are log1p CPM.
- No model-specific post-processing or surrogate imputer was used.

## Main result

| Model | Common MSE | Common MAE | Common Pearson |
|---|---:|---:|---:|
| Txn_Jatin | **0.1326** | **0.2532** | **0.9787** |
| Txn_Jatin OSDR LoRA | 0.1484 | 0.2750 | 0.9779 |
| BRIDGE | 0.2377 | 0.3463 | 0.9614 |
| BulkFormer-50M | 2.4492 | 1.3040 | 0.6151 |
| BulkFormer-37M | 2.7695 | 1.4092 | 0.4182 |
| BulkFormer-147M | 2.8210 | 1.4132 | 0.4293 |
| BulkFormer-93M | 2.9730 | 1.4571 | 0.2476 |
| BulkFormer-127M | 3.0280 | 1.4719 | 0.1970 |

Txn_Jatin contextual, ESM2, ESM3, Geneformer, and scGPT are reported as NaN
because the registered artifacts expose embeddings but no native expression
decoder. Adding an unrelated external imputer would not be a like-for-like
test.

## Interpretation

Txn_Jatin gives the best reconstruction error and correlation on this external
recount2 GTEx panel. The OSDR LoRA adaptation remains close, but slightly
reduces GTEx reconstruction performance, consistent with domain-specific
adaptation. BRIDGE reconstructs the broad expression pattern well but is less
accurate than both Txn_Jatin variants. BulkFormer performance varies by
checkpoint size; larger parameter count does not imply better recovery in this
test.

These 12 samples are a technical external-domain pilot, not a definitive
population-level validation. The samples were selected for tissue diversity
and quality rather than sampled to estimate clinical performance.
