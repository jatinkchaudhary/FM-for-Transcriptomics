# Txn_Jatin OSDR Fine-Tuning and Gene-Level Benchmark Report

Generated: 2026-07-24
Remote run: `/media/volume/AdditionalHeadroom/Txn_Jatin_runs/full_ARCHES4_20epoch_H100_20260624_015238/benchmarks/osdr_txn_peft_sweep_20260724_0240`

## Executive Summary

We completed an OSDR-focused fine-tuning sweep for the final 20-epoch Txn_Jatin checkpoint, then benchmarked the selected fine-tuned model using the retained GitHub gene-level protocol. The sweep intentionally avoided fully unfreezing the entire model, because the goal was fine-tuning/adaptation rather than retraining the model from scratch.

The best selected model was:

**`Txn_Jatin_OSDR_lora_attn_ffn_last3_r4_dynamic`**

It was selected by the pre-defined combined ranking rule across:

- held-out 15% masked-gene reconstruction MSE, where lower is better
- held-out flight-vs-ground AUROC, where higher is better

The selected LoRA model achieved the best flight-vs-ground AUROC in the sweep (`0.6547`), while partial unfreezing of the last 3 transformer layers achieved the best reconstruction MSE (`0.3177`). This shows a tradeoff: more direct layer unfreezing improves expression reconstruction, while lightweight LoRA adaptation better preserves sample-level biological separability.

## What Was Done

1. Started from the final Txn_Jatin ARCHS4 checkpoint:
   - checkpoint: `best_model.pt`
   - source training: 20 epochs on ARCHS4
   - final source validation loss: approximately `0.0781`
   - gene vocabulary: approximately `16,055` genes
   - architecture size: approximately `46M` parameters

2. Stopped the earlier full-unfreeze benchmarking path.
   - The full-unfreeze model had strong reconstruction MSE but poor classification AUROC.
   - It was not used for final benchmarking because unfreezing all layers is closer to retraining than parameter-efficient fine-tuning.

3. Ran a controlled OSDR fine-tuning sweep with:
   - head-only tuning
   - BitFit plus normalization tuning
   - gradual unfreezing of the last 1, 2, and 3 transformer layers
   - LoRA adapters over attention projections
   - LoRA adapters over attention plus feed-forward layers

4. Selected the best model using the pre-declared combined rank rule.

5. Converted the selected model into a GitHub-compatible gene embedding file.

6. Ran the requested GitHub-style gene-level benchmarks:
   - GO full
   - GO intersect
   - OMIM full
   - OMIM intersect
   - Temporal GO

7. Omitted the benchmarks requested to be skipped:
   - SL gene-pair
   - POMBE gene-pair
   - TF gene-pair
   - NG gene-pair
   - ANDES

## Fine-Tuning Protocol

### Dataset and Split

The OSDR dataset was prepared once and reused across all candidates. The same accession-grouped split was used for every fine-tuning method, so the comparison is paired and not confounded by different train/test partitions.

- input normalization: `log1p_cpm`
- held-out test samples: `633`
- contextual gene table: computed from the OSDR development partition only, meaning train plus validation, not held-out test
- random seed: `42`

### Masked Reconstruction Task

Each model was fine-tuned on a masked-expression reconstruction objective:

- randomly mask 15% of genes in each expression vector
- replace masked expression values with the mask token
- train the model to predict the original expression values at the masked positions
- evaluate held-out MSE over masked positions

This directly tests whether the model can recover hidden expression values from the observed expression context.

### Flight-vs-Ground Evaluation

For each candidate, sample-level embeddings were extracted and evaluated on OSDR flight-vs-ground classification.

Reported metrics:

- `F1_macro`: class-balanced F1
- `AUROC`: threshold-independent flight-vs-ground separability

This tests whether OSDR adaptation preserves or improves biologically meaningful sample-level structure.

### Training Settings

All candidates used the same base optimization settings:

| Setting | Value |
|---|---:|
| epochs | 20 |
| early stopping patience | 5 |
| batch size | 16 |
| contextual extraction batch size | 64 |
| learning rate | 1e-5 |
| weight decay | 1e-2 |
| masked gene ratio | 0.15 |
| dataloader workers | 8 |
| precision | CUDA mixed precision with bfloat16 autocast |

## Fine-Tuning Candidates

| Candidate | Description |
|---|---|
| `head_only` | Frozen transformer; only output reconstruction head updated |
| `bitfit_norm` | Biases, LayerNorm parameters, and output head updated |
| `last1` | Last 1 transformer layer plus output head updated |
| `last2` | Last 2 transformer layers plus output head updated |
| `last3` | Last 3 transformer layers plus output head updated |
| `lora_attn_all_r4` | LoRA rank 4 on attention projections in all 12 transformer layers |
| `lora_attn_last6_r8` | LoRA rank 8 on attention projections in the last 6 transformer layers |
| `lora_attn_ffn_last3_r4` | LoRA rank 4 on attention and feed-forward projections in the last 3 transformer layers |

## Held-Out OSDR Results

Lower MSE is better. Higher F1 and AUROC are better.

| Model | Masked Reconstruction MSE | F1 Macro | Flight-vs-Ground AUROC |
|---|---:|---:|---:|
| Txn_Jatin original | 1.0904 | not evaluated | not evaluated |
| `head_only` | 1.0401 | 0.5950 | 0.6350 |
| `bitfit_norm` | 0.9447 | 0.5575 | 0.5981 |
| `last1` | 0.6783 | 0.5545 | 0.5955 |
| `last2` | 0.4204 | 0.5330 | 0.5749 |
| `last3` | **0.3177** | 0.5588 | 0.5739 |
| `lora_attn_all_r4` | 0.9359 | 0.5987 | 0.6272 |
| `lora_attn_last6_r8` | 0.9308 | 0.5985 | 0.6161 |
| `lora_attn_ffn_last3_r4` | 0.9648 | **0.6047** | **0.6547** |

### Selection Result

The selected model was:

**`Txn_Jatin_OSDR_lora_attn_ffn_last3_r4_dynamic`**

Selection rule:

> minimum mean rank across held-out masked reconstruction MSE and flight-vs-ground AUROC; ties prefer AUROC.

The selection table showed a tradeoff:

- Best MSE: `last3`, MSE `0.3177`
- Best AUROC: `lora_attn_ffn_last3_r4`, AUROC `0.6547`
- Selected final model: `lora_attn_ffn_last3_r4`, because it had the strongest biological separability and won the tie by AUROC

## Gene-Level Benchmarking Protocol

The selected fine-tuned model was evaluated using the retained GitHub-style gene-level benchmark protocol from `ylaboratory/gene-embedding-benchmarks`.

The selected model embedding was converted into the expected benchmark format:

- model embedding dimension: `512`
- symbol rows: `15,916`
- Entrez-mapped rows: `15,907`
- intersection rows used by the protocol: `9,568`

Benchmark jobs run:

| Benchmark | Scope | Purpose |
|---|---|---|
| GO full | all mapped genes | functional gene ontology recovery |
| GO intersect | shared intersection | apples-to-apples GO comparison on shared genes |
| OMIM full | all mapped genes | disease-gene association recovery |
| OMIM intersect | shared intersection | apples-to-apples OMIM comparison on shared genes |
| Temporal GO | post-2024 holdout | ability to recover newer GO annotations from older structure |

Benchmark jobs omitted by instruction:

- all gene-pair benchmarks: SL, POMBE, TF, NG
- ANDES

## Benchmark Results

Higher AUROC, AUPRC, and PR@10 are better.

| Benchmark | Scope | Terms | AUROC | AUPRC | PR@10 |
|---|---|---:|---:|---:|---:|
| GO full | all genes | 56 | 0.8162 | 0.5089 | 0.6161 |
| GO intersect | intersect | 56 | **0.8278** | **0.5218** | 0.5429 |
| OMIM full | all genes | 103 | 0.6816 | 0.3255 | 0.2757 |
| OMIM intersect | intersect | 103 | 0.6439 | 0.2898 | 0.2456 |
| Temporal GO | intersect | 19 | 0.6000 | 0.2159 | 0.1579 |

## Interpretation

### 1. OSDR adaptation worked, but the best method depends on the endpoint.

Gradual unfreezing, especially `last3`, produced the best masked reconstruction result. This means the model can adapt its expression decoder and high-level representation to recover masked OSDR expression values much more accurately than the original checkpoint.

However, the best masked reconstruction model was not the best flight-vs-ground model. Its AUROC dropped to `0.5739`, suggesting that optimizing reconstruction too strongly may blur the sample-level signal useful for biological classification.

### 2. LoRA preserved biological separability better than layer unfreezing.

The selected LoRA candidate achieved the best flight-vs-ground AUROC:

- `lora_attn_ffn_last3_r4`: AUROC `0.6547`
- original/head-only scale: AUROC around `0.6350`
- last3 unfreezing: AUROC `0.5739`

This suggests that lightweight adapter tuning can inject OSDR-specific information while preserving more of the pretrained Txn_Jatin representation.

### 3. Gene-level GO performance is strong after OSDR fine-tuning.

The selected LoRA model performed well on GO benchmarks:

- GO full AUROC: `0.8162`
- GO intersect AUROC: `0.8278`

This indicates that the fine-tuned contextual gene representation still carries functional gene structure after OSDR adaptation.

### 4. Disease-gene recovery is moderate.

OMIM performance was lower than GO:

- OMIM full AUROC: `0.6816`
- OMIM intersect AUROC: `0.6439`

This is expected because OMIM disease associations are more heterogeneous than GO biological-process labels and may depend on disease-specific genetic mechanisms that are not fully captured by expression reconstruction.

### 5. Temporal GO remains the hardest benchmark.

Temporal GO AUROC was `0.6000`, above random but much lower than standard GO. This benchmark asks whether embeddings recover newer annotations held out by time, which is a stricter test of future functional generalization.

## Slide-Ready Takeaways

- We completed OSDR fine-tuning without fully retraining Txn_Jatin.
- Eight fine-tuning strategies were compared under the same held-out OSDR split.
- The strongest reconstruction model was gradual unfreezing of the last 3 layers: MSE `0.3177`.
- The strongest biologically separable model was LoRA on attention plus FFN in the last 3 layers: AUROC `0.6547`.
- The final benchmarked model was selected by combined MSE/AUROC ranking, with AUROC used to break ties.
- The selected model achieved strong GO recovery: GO intersect AUROC `0.8278`.
- OMIM was moderate and Temporal GO remained difficult, suggesting disease and future-annotation generalization are harder than standard function recovery.

## Suggested Presentation Framing

The cleanest message for stakeholders is:

> Txn_Jatin can be adapted to OSDR without fully retraining the model. Full reconstruction-oriented adaptation improves masked expression recovery, but lightweight LoRA tuning better preserves biological class separability. The selected LoRA-adapted model retains strong gene-function structure, especially on GO benchmarks, while disease and temporal generalization remain harder tasks.

## Caveats

- The selected model is not the model with the best reconstruction MSE. It is the best by the pre-defined combined rank across MSE and AUROC.
- The GitHub-style benchmark was run only for the selected fine-tuned model, not for every PEFT candidate.
- Gene-pair benchmarks and ANDES were intentionally omitted by instruction.
- `nvidia-smi` on the remote machine was unreliable because of an NVML driver/library mismatch, but PyTorch CUDA execution worked and the training completed.

## Key Output Locations

Remote sweep folder:

`/media/volume/AdditionalHeadroom/Txn_Jatin_runs/full_ARCHES4_20epoch_H100_20260624_015238/benchmarks/osdr_txn_peft_sweep_20260724_0240`

Winner selection:

`/media/volume/AdditionalHeadroom/Txn_Jatin_runs/full_ARCHES4_20epoch_H100_20260624_015238/benchmarks/osdr_txn_peft_sweep_20260724_0240/winner_selection.json`

Benchmark summary:

`/media/volume/AdditionalHeadroom/Txn_Jatin_runs/full_ARCHES4_20epoch_H100_20260624_015238/benchmarks/osdr_txn_peft_sweep_20260724_0240/benchmark/results/gene_level_summary_long.csv`

Selected model checkpoint:

`/media/volume/AdditionalHeadroom/Txn_Jatin_runs/full_ARCHES4_20epoch_H100_20260624_015238/benchmarks/osdr_txn_peft_sweep_20260724_0240/candidates/lora_attn_ffn_last3_r4/best.pt`
