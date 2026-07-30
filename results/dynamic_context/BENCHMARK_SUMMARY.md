# Dynamic embedding benchmark summary

Txn_Jatin and BRIDGE used full-corpus expression-conditioned gene embeddings. All seven comparison models remained static.

## Dynamic extraction provenance

- Context corpus: **978,212 ARCHS4 samples** per dynamic model.
- Context batch: **96**, bf16 inference with float32 corpus accumulation.
- No model was fine-tuned.
- Audit requirement: exactly Txn_Jatin and BRIDGE dynamic; all other models static; no fallback embeddings.

| Model | Dynamic | Context samples | Context files | Extraction time |
|---|---:|---:|---:|---:|
| Txn_Jatin | yes | 978,212 | 863 | 12.71 h |
| BRIDGE | yes | 978,212 | 863 | 11.53 h |

## Overall result

| Rank | Model | Dynamic mean | Previous static mean | Delta |
|---:|---|---:|---:|---:|
| 1 | BRIDGE | 0.6437 | 0.6279 | +0.0158 |
| 2 | Txn_Jatin | 0.6357 | 0.6248 | +0.0110 |
| 3 | scGPT | 0.6354 | 0.6354 | +0.0000 |
| 4 | Geneformer | 0.6296 | 0.6298 | -0.0002 |
| 5 | BulkFormer_147M | 0.6242 | 0.6246 | -0.0003 |
| 6 | BulkFormer_50M | 0.6161 | 0.6165 | -0.0004 |
| 7 | BulkFormer_93M | 0.6131 | 0.6139 | -0.0007 |
| 8 | BulkFormer_37M | 0.6113 | 0.6113 | +0.0000 |
| 9 | BulkFormer_127M | 0.6095 | 0.6110 | -0.0016 |

Dynamic BRIDGE ranked **1st overall** across the four full-gene tracks (0.6437). Dynamic Txn_Jatin ranked **2nd overall** (0.6357), narrowly ahead of static scGPT.

## Full-gene benchmark

| Track | Txn_Jatin | Rank | BRIDGE | Rank | Winner |
|---|---:|---:|---:|---:|---|
| gene-set | 0.6386 | 2/9 | 0.6446 | 1/9 | BRIDGE (0.6446) |
| paired | 0.5862 | 3/9 | 0.5888 | 1/9 | BRIDGE (0.5888) |
| single-gene-GO | 0.6570 | 8/9 | 0.6698 | 2/9 | BulkFormer_147M (0.6810) |
| single-gene-disease | 0.6610 | 7/9 | 0.6714 | 3/9 | Geneformer (0.6795) |

## Change from the previous static benchmark

| Model | Track | Dynamic | Static | Delta |
|---|---|---:|---:|---:|
| BRIDGE | gene-set | 0.6446 | 0.6334 | +0.0112 |
| BRIDGE | paired | 0.5888 | 0.5761 | +0.0127 |
| BRIDGE | single-gene-GO | 0.6698 | 0.6495 | +0.0203 |
| BRIDGE | single-gene-disease | 0.6714 | 0.6526 | +0.0188 |
| Txn_Jatin | gene-set | 0.6386 | 0.6271 | +0.0115 |
| Txn_Jatin | paired | 0.5862 | 0.5752 | +0.0110 |
| Txn_Jatin | single-gene-GO | 0.6570 | 0.6442 | +0.0128 |
| Txn_Jatin | single-gene-disease | 0.6610 | 0.6525 | +0.0085 |

## Representation geometry

- Txn_Jatin–BRIDGE CCA increased from **0.792** to **0.895**.
- Dynamic Txn_Jatin's highest CCA similarity is **BRIDGE** at **0.895**.
- Full-corpus contextualization therefore makes Txn_Jatin and BRIDGE substantially more aligned while retaining distinct downstream scores.

## Sample embeddings

Sample embeddings already use expression-conditioned forward passes, so this section checks downstream transfer rather than replacing them with the corpus-mean gene tables.

| Dataset | Split | Feature | F1 | AUROC |
|---|---|---|---:|---:|
| OSDR-spaceflight | group | BRIDGE-emb | 0.5632 | 0.5888 |
| OSDR-spaceflight | group | PCA-64 | 0.5819 | 0.6459 |
| OSDR-spaceflight | group | Txn_Jatin-emb | 0.5765 | 0.6189 |
| OSDR-spaceflight | group | raw_log1p_TPM | 0.6559 | 0.7056 |
| OSDR-spaceflight | stratified | BRIDGE-emb | 0.6794 | 0.7511 |
| OSDR-spaceflight | stratified | PCA-64 | 0.6553 | 0.7202 |
| OSDR-spaceflight | stratified | Txn_Jatin-emb | 0.7032 | 0.7780 |
| OSDR-spaceflight | stratified | raw_log1p_TPM | 0.8167 | 0.8932 |
| TCGA-cancer-type | group | BRIDGE-emb | 0.9527 | 0.9964 |
| TCGA-cancer-type | group | PCA-64 | 0.9655 | 0.9976 |
| TCGA-cancer-type | group | Txn_Jatin-emb | 0.9601 | 0.9969 |
| TCGA-cancer-type | group | raw_log1p_TPM | 0.9715 | 0.9978 |
| TCGA-cancer-type | stratified | BRIDGE-emb | 0.9565 | 0.9970 |
| TCGA-cancer-type | stratified | PCA-64 | 0.9669 | 0.9976 |
| TCGA-cancer-type | stratified | Txn_Jatin-emb | 0.9590 | 0.9971 |
| TCGA-cancer-type | stratified | raw_log1p_TPM | 0.9716 | 0.9979 |

## Caveats

- The dynamic gene table is a corpus mean of frozen per-gene hidden states, not a fine-tuned checkpoint.
- TRRUST paired-gene scores were generated in a separate process from the exact exported embedding matrices.
- Paired-track dynamic-versus-static deltas compare two benchmark runs. Unchanged static controls varied by at most 0.0043 AUROC on that track, so those deltas include small negative-sampling rerun variation.
- SynLethDB and temporal GO availability follow the same caveats as the previous benchmark.
