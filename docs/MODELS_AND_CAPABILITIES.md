# Models and capability boundary

## Native expression decoders

- **Txn_Jatin:** 16,055 genes, 512 dimensions.
- **Txn_Jatin OSDR LoRA:** the selected rank-4 last-three-layer adapter,
  16,055 genes, 512 dimensions.
- **BRIDGE:** 15,165 genes, 512 dimensions.
- **BulkFormer 37M/50M/93M/127M/147M:** 20,010-gene decoder vocabulary with
  128/256/512/640/640-dimensional learned gene embeddings.

These models can produce expression estimates for missing genes. The deployed
runtime aligns HGNC symbols to the model vocabulary and leaves unresolved genes
unimputed.

## Embedding-only controls

- **Txn_Jatin contextual**
- **ESM2 PCA-512 prior**
- **ESM3**
- **Geneformer**
- **scGPT**

These artifacts support gene-level or gene-pair embedding benchmarks but do not
expose a validated bulk-expression decoder. Their expression-imputation result
is therefore reported as `NaN`, matching the whole-gene masking protocol.

## Input scope

The native models were trained/evaluated near 15% masking with broad gene
coverage. Supplying only 50, 1,000, or 2,000 genes is a distribution shift.
The API returns a coverage warning whenever fewer than half of the selected
model's genes are supplied.
