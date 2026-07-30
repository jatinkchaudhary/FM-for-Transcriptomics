## Model architecture and representation differences

The models do not define a "gene embedding" in exactly the same way. The
benchmark therefore freezes each representation at its native dimensionality
and applies the same downstream probes, but the source of the vectors remains
architecturally different.

### Architecture summary

| Model | Core architecture | Checkpoint scale and dimensions | Input representation | Embedding used by this benchmark |
|---|---|---|---|---|
| Txn_Jatin | ExpressionPerformer implemented with dense Flash/SDPA Transformer blocks | 46,050,305 trainable parameters in `epoch_00.pt`; 12 layers, 8 heads, hidden 512, FFN 2,048; 16,057-gene checkpoint vocabulary | Learned gene-identity vector plus sinusoidal Rotary Expression Embedding (REE) of log1p-TPM | Static: hidden states from an all-zero expression input. Dynamic: each gene's hidden state averaged over 978,212 real ARCHS4 profiles |
| BRIDGE | Same ExpressionPerformer family and Flash/SDPA block design | 45,593,601 trainable parameters; 12 layers, 8 heads, hidden 512, FFN 2,048; 15,165-gene checkpoint vocabulary | Learned gene identity plus REE expression features | Static: all-zero-input hidden states. Dynamic: full-corpus mean contextual hidden states over the same 978,212 profiles |
| scGPT | Transformer encoder for gene tokens and continuous expression values, with expression, cell-classification, and optional MVC/batch objectives | 12 layers, 8 heads, hidden 512, FFN 512, maximum sequence 1,536; 60,697-token checkpoint vocabulary | Gene-token embedding plus a continuous value encoder; a CLS token represents the cell | Static 512-dimensional `gene_encoder.embedding.weight`; no contextual forward pass was used for the gene benchmark |
| Geneformer V2-104M | BERT masked-language model over ranked gene tokens | Approximately 104M parameters; 12 layers, 12 heads, hidden 768, FFN 3,072, maximum sequence 4,096; vocabulary 20,275 | Rank-value encoding: genes are ordered by expression after corpus-level median scaling | Static 768-dimensional BERT word-token embedding, mapped from Ensembl IDs to gene symbols |
| BulkFormer-37M | Hybrid graph convolution plus Performer global attention | Hidden 128; one graph block containing one Performer layer; 8 global-attention heads | Expression embedding + projected gene identity + global sample-expression projection | Static projected gene-identity table, before expression, graph, or Performer contextualization |
| BulkFormer-50M | Same BulkFormer hybrid | Hidden 256; one graph block containing two Performer layers; 8 heads | Same as above | Static projected gene-identity table |
| BulkFormer-93M | Same BulkFormer hybrid | Hidden 512; one graph block containing six Performer layers; 8 heads | Same as above | Static projected gene-identity table |
| BulkFormer-127M | Same BulkFormer hybrid | Hidden 640; one graph block containing eight Performer layers; 8 heads | Same as above | Static projected gene-identity table |
| BulkFormer-147M | Same BulkFormer hybrid | Hidden 640; one graph block containing twelve Performer layers; 8 heads | Same as above | Static projected gene-identity table |

The BulkFormer family uses a 20,010-gene native vocabulary. Its forward model
combines a graph-convolution branch with Performer attention, but the frozen
gene vectors used here are
`gene_emb_proj(gene_emb_onehot_layer.weight)`. Consequently, these vectors
capture learned gene identity but do not include sample expression, graph
message passing, or contextual Performer output.

### Txn_Jatin versus BRIDGE

Txn_Jatin deliberately retains the essence of BRIDGE: both sum learned gene
identity with an REE encoding of continuous expression and process the complete
gene sequence through 12 attention/FFN blocks before reconstructing masked
expression values. Their primary differences are training and vocabulary:

- Txn_Jatin was trained for one complete epoch on the harmonized full ARCHS4
  corpus: 968,212 training profiles plus 10,000 validation profiles, 16,057
  genes, batch 9, bf16 AMP, 15% masking, learning rate `1e-4`, and no species
  embedding.
- The BRIDGE reference checkpoint has 15,165 genes and the same 512/2,048,
  12-layer, 8-head architecture. Its configuration targeted up to 20 epochs,
  batch 4, early stopping, and balanced-shard sampling.
- Txn_Jatin therefore changes the learned weights, training coverage, and gene
  vocabulary, but not the fundamental expression-conditioned Transformer
  mechanism.

### Static versus dynamic embeddings

A static gene embedding assigns one fixed vector to a gene without using a real
transcriptome. For scGPT and Geneformer this is the checkpoint token table; for
BulkFormer it is the projected gene-identity table. For the static Txn_Jatin and
BRIDGE comparison, the model was run with an all-zero expression profile and
the resulting per-gene hidden states were retained.

The dynamic Txn_Jatin and BRIDGE embeddings use the same frozen checkpoints but
replace that artificial zero-expression context with real ARCHS4 contexts. For
each profile, the model computes

`hidden(gene, sample) = Transformer(gene identity + REE(expression))`

and the benchmark averages each gene's hidden state across 978,212 profiles.
The resulting matrix remains frozen during downstream probing, but it carries
corpus-level expression context. It is therefore a contextualized gene
embedding, not a separately fine-tuned benchmark representation and not a
different vector for each downstream TCGA/OSDR sample.

This distinction matters most for Txn_Jatin and BRIDGE because their
architecture was trained to place biological information in
expression-conditioned hidden states. The static zero-input route removes that
signal. In contrast, the scGPT, Geneformer, and BulkFormer comparison vectors
were intentionally left static, so they act as controls for the embedding
extraction change.

## All-model static embedding benchmark

The table below is the earlier full-gene benchmark in which **all nine models,
including Txn_Jatin and BRIDGE, used static gene embeddings**. Values are mean
AUROC. "Mean" is the unweighted mean of the four displayed tracks.

| Model | Hallmark gene-set | Paired genes | GO | Disease | Four-track mean |
|---|---:|---:|---:|---:|---:|
| Txn_Jatin | 0.6271 | 0.5752 | 0.6442 | 0.6525 | 0.6248 |
| BRIDGE | 0.6334 | 0.5761 | 0.6495 | 0.6526 | 0.6279 |
| scGPT | 0.6153 | **0.5867** | 0.6690 | 0.6705 | **0.6354** |
| Geneformer | 0.6019 | 0.5723 | 0.6654 | **0.6794** | 0.6298 |
| BulkFormer-37M | 0.5588 | 0.5696 | 0.6558 | 0.6609 | 0.6113 |
| BulkFormer-50M | 0.5688 | 0.5698 | 0.6634 | 0.6640 | 0.6165 |
| BulkFormer-93M | 0.5677 | 0.5604 | 0.6622 | 0.6651 | 0.6138 |
| BulkFormer-127M | 0.5600 | 0.5617 | 0.6620 | 0.6605 | 0.6110 |
| BulkFormer-147M | 0.5816 | 0.5602 | **0.6813** | 0.6752 | 0.6246 |

Under the all-static design, scGPT had the best four-track mean, BRIDGE led the
Hallmark gene-set track, BulkFormer-147M led GO, Geneformer led disease
annotation, and scGPT led the paired-gene aggregate.

## Effect of contextualizing Txn_Jatin and BRIDGE

The direct before/after comparison below changes Txn_Jatin and BRIDGE from
static zero-input embeddings to dynamic full-corpus embeddings. The other
models remain static.

| Model | Track | Static AUROC | Hybrid-run AUROC | Change |
|---|---|---:|---:|---:|
| Txn_Jatin | Hallmark gene-set | 0.6271 | 0.6386 | +0.0115 |
| Txn_Jatin | Paired genes | 0.5752 | 0.5862 | +0.0110 |
| Txn_Jatin | GO | 0.6442 | 0.6570 | +0.0128 |
| Txn_Jatin | Disease | 0.6525 | 0.6610 | +0.0085 |
| Txn_Jatin | Four-track mean | 0.6248 | 0.6357 | +0.0110 |
| BRIDGE | Hallmark gene-set | 0.6334 | 0.6446 | +0.0112 |
| BRIDGE | Paired genes | 0.5761 | 0.5888 | +0.0127 |
| BRIDGE | GO | 0.6495 | 0.6698 | +0.0203 |
| BRIDGE | Disease | 0.6526 | 0.6714 | +0.0188 |
| BRIDGE | Four-track mean | 0.6279 | 0.6437 | +0.0158 |

Dynamic contextualization improved every displayed Txn_Jatin and BRIDGE track.
The larger BRIDGE gains on GO and disease support the original diagnosis that
its zero-input static representation underuses information stored in
expression-conditioned hidden states. In the hybrid comparison, BRIDGE moved
from a four-track mean of 0.6279 to 0.6437 and became the strongest overall
mean performer. Txn_Jatin improved from 0.6248 to 0.6357 and became second.

Scores for the seven static controls are nearly unchanged between runs. Tiny
differences in several probes reflect rerun/supplement variation rather than a
change in their embedding extraction; their gene matrices remained static.

## Architecture and comparison artifacts

- `tables/static_full_track_auroc.csv`: full all-static benchmark for all models
- `tables/hybrid_minus_static_full_track_delta.csv`: model-wise hybrid minus
  static differences
- `tables/hybrid_embedding_provenance.csv`: exact dynamic/static assignment,
  dimensions, checkpoints, and fallback status
- `dynamic_embedding_audit.json`: full-corpus contextual extraction audit

Architecture details were taken from the exact local checkpoint/configuration
and adapter sources used for this analysis:

- `../models/txn_jatin_epoch1/config.json`
- `../models/bridge_reference/best_model.pt`
- `../../RNA Walter/flash_osdr_model/train_flash.py`
- `../../scGPT-main/scgpt/model/model.py` and the local `tdc/scGPT` checkpoint
  configuration
- `../../Geneformer/Geneformer-V2-104M/config.json`
- `../../BulkFormer-main/model/config.py` and `../../BulkFormer-main/utils/`
- `../../mythos/adapters.py`
