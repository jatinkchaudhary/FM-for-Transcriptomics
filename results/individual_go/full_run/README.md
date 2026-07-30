# Individual GO Benchmark With ESM3

Completed remotely on `nvidea-h100-2-of-2` (NVIDIA H100 80GB HBM3) at
2026-07-26 02:27:02 UTC.

## Scope

The three original BRIDGE partial-unfreezing experiments were evaluated
separately:

- `last1`: final Transformer block and reconstruction head trainable
- `last2`: final two Transformer blocks and reconstruction head trainable
- `last3`: final three Transformer blocks and reconstruction head trainable

Each experiment includes its corresponding partial model plus the shared
controls. ESM2 and ESM3 are included in every experiment.

## Protocol

- 40 frozen GO Biological Process terms
- 13,408-gene strict all-model intersection for the primary comparison
- five-fold out-of-fold GPU linear probes
- per-feature `StandardScaler`
- seed 42
- metrics: AUROC and AUPRC
- broader per-model coverage results retained as `osdr_full`

The strict intersection is the primary result because ESM2 and ESM3 represent
protein-coding genes and cannot provide biologically meaningful embeddings for
all non-protein-coding genes.

## ESM3 Artifact

| Item | Value |
|---|---:|
| Model | `esm3-sm-open-v1` |
| Embedding dimension | 1,536 |
| Master genes | 15,916 |
| Genes with protein sequences | 15,529 |
| Maximum sequence length | 1,022 residues |
| Truncated sequences | 2,139 |
| Pooling | Mean final-layer residue embedding, excluding BOS/EOS |

The ESM3 input used the same pinned BulkFormer gene-to-protein sequence
metadata used to define the ESM2 sequence control.

## Primary Strict-Intersection Results

| Rank | Model | Mean AUROC | Mean AUPRC |
|---:|---|---:|---:|
| 1 | ESM2 PCA-512 prior | 0.7803 | 0.2308 |
| 2 | ESM3 | 0.7799 | **0.2415** |
| 3 | BulkFormer-147M | 0.6726 | 0.0778 |
| 4 | BRIDGE last 1 | 0.6622 | 0.0876 |
| 4 in its experiment | BRIDGE last 3 | 0.6621 | 0.0882 |
| 5 in its experiment | BRIDGE last 2 | 0.6617 | 0.0876 |
| 5/4 | scGPT | 0.6618 | 0.0903 |
| 6 | Original BRIDGE OSDR dynamic | 0.6605 | 0.0872 |
| 7 | Fully unfrozen BRIDGE | 0.6589 | 0.0854 |
| 13 | Txn_Jatin OSDR dynamic | 0.6425 | 0.0800 |

ESM2 led mean AUROC by only 0.0004, while ESM3 led mean AUPRC by 0.0108.
In direct term-by-term comparison, ESM3 exceeded ESM2 on 22 of 40 terms.

Across all models, the individual-term winners were:

| Model | GO terms won |
|---|---:|
| ESM3 | 19 |
| ESM2 PCA-512 prior | 16 |
| scGPT | 3 |
| BulkFormer-147M | 2 |

## Partial-Unfreezing Effect

Relative to original BRIDGE on the same strict gene universe:

| Experiment | Mean AUROC | Terms improved | Mean AUROC delta |
|---|---:|---:|---:|
| Last 1 | 0.6622 | 30/40 | +0.001779 |
| Last 2 | 0.6617 | 26/40 | +0.001226 |
| Last 3 | 0.6621 | 25/40 | +0.001662 |

Last 1 retained the strongest aggregate GO signal among the partial models.
Last 3 remained extremely close and was previously the strongest condition
for held-out OSDR flight/ground representation. Last 2 remained the strongest
partial model for the separate KEGG analysis.

## Interpretation

ESM2 and ESM3 dominate this GO benchmark because protein-sequence language
models directly encode conserved domains, motifs, and evolutionary constraints
that are strongly related to GO function. This does not imply that they are
better expression reconstruction or sample-context models. They have no native
bulk-expression decoder and cannot perform the whole-gene expression
imputation benchmark.

The partial-unfreezing conclusion remains stable: updating a small number of
late BRIDGE layers produces modest GO improvements while preserving much more
of the pretrained representation than full unfreezing. Adding the sequence
controls changes the overall leaderboard, but not the ordering of the three
partial-unfreezing conclusions.

## Files

- `results/osdr_go_term_scores_with_esm3.csv`: complete long-form term results
- `results/osdr_go_model_summary_with_esm3.csv`: all-model summaries
- `results/last1/`, `last2/`, `last3/`: experiment-specific matrices, winners,
  and summaries
- `embeddings/ESM3__gene_benchmark__symbol.npz`: final ESM3 embedding artifact
- `embeddings/ESM3__gene_benchmark__symbol.manifest.json`: ESM3 provenance
- `logs/pipeline.log`: complete remote execution log
- `results/manifest.json`: benchmark protocol and model inventory
