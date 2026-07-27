# Epoch 7 vs s66qfh36 local comparison

All computations were performed locally on CPU using frozen full-corpus contextual gene embeddings.
The requested `._best_model.pt` is a macOS AppleDouble sidecar; `best_model.pt` was used.

## Checkpoints

| Model | Epoch | Genes | Context samples | Parameters | Train loss | Validation loss | Normalization |
|---|---:|---:|---:|---:|---:|---:|---|
| Txn_Jatin_epoch7 | 7 | 16,055 | 958,212 | 46,049,537 | 0.094959 | 0.095168 | counts_to_log1p_tpm |
| s66qfh36_best | 20 | 14,567 | 978,212 | 45,593,857 | 0.147937 | 0.137804 | log1p_tpm |

Losses are descriptive only because the checkpoints use different data preparation and normalization.

## Gene overlap

- Shared genes: **14,567**
- Epoch-7-only genes: **1,488**
- s66qfh36-only genes: **0**

## Embedding alignment

- Linear CKA: **0.0602**
- Procrustes-aligned same-gene cosine: **0.1857**
- Top-10 neighbor overlap: **0.0238**

## Functional centroid prediction

| library | model | terms | mean_auroc | mean_auprc | shuffled_mean_auroc |
|---|---|---|---|---|---|
| DisGeNET | Txn_Jatin_epoch7 | 100 | 0.6909 | 0.1883 | 0.4998 |
| DisGeNET | s66qfh36_best | 100 | 0.6669 | 0.1761 | 0.4995 |
| GO_BP | Txn_Jatin_epoch7 | 100 | 0.8080 | 0.2563 | 0.5038 |
| GO_BP | s66qfh36_best | 100 | 0.7091 | 0.0751 | 0.4986 |
| KEGG_2021_Human | Txn_Jatin_epoch7 | 100 | 0.8979 | 0.2734 | 0.5035 |
| KEGG_2021_Human | s66qfh36_best | 100 | 0.7330 | 0.0486 | 0.5052 |

## Same-set retrieval

| model | library | n_positive | n_negative | auroc | auprc | mean_positive_cosine | mean_negative_cosine |
|---|---|---|---|---|---|---|---|
| Txn_Jatin_epoch7 | GO_BP | 3000 | 3000 | 0.5998 | 0.6529 | 0.0597 | -0.0020 |
| s66qfh36_best | GO_BP | 3000 | 3000 | 0.6307 | 0.6200 | 0.0695 | -0.0053 |
| Txn_Jatin_epoch7 | Hallmark | 3000 | 3000 | 0.5288 | 0.5462 | 0.0086 | 0.0000 |
| s66qfh36_best | Hallmark | 3000 | 3000 | 0.6521 | 0.6477 | 0.0880 | 0.0052 |
| Txn_Jatin_epoch7 | KEGG_2021_Human | 3000 | 3000 | 0.6049 | 0.6575 | 0.0547 | -0.0016 |
| s66qfh36_best | KEGG_2021_Human | 3000 | 3000 | 0.6186 | 0.6193 | 0.0680 | -0.0021 |

## Headline comparison

- DisGeNET centroid AUROC: epoch 7 leads by **+0.0240**
- GO_BP centroid AUROC: epoch 7 leads by **+0.0989**
- KEGG_2021_Human centroid AUROC: epoch 7 leads by **+0.1650**
- GO_BP same-set AUROC: s66qfh36 leads by **0.0309**
- Hallmark same-set AUROC: s66qfh36 leads by **0.1233**
- KEGG_2021_Human same-set AUROC: s66qfh36 leads by **0.0137**

There is no universal winner: epoch 7 is stronger for centroid-based functional prediction, while s66qfh36 preserves stronger same-set neighborhood ranking.

## Interpretation

Higher AUROC/AUPRC is better. The comparison is paired on the same 14,567 shared genes and uses identical sampled pairs for both models.
