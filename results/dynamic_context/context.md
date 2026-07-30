# Dynamic benchmark context

This file is the restart point for a future chat.

## Request

Compute dynamic embeddings for Txn_Jatin and original BRIDGE only, leave all
other benchmark models static, rerun the full benchmark, regenerate all graphs,
and store everything in this folder.

## Completed experiment

Each dynamic model was frozen and forwarded over the complete prepared ARCHS4
corpus:

- 978,212 samples
- 863 parquet files
- bf16 GPU forward pass
- float32 corpus accumulator
- batch size 96, the largest stable A100 setting

The output for each gene is the mean contextual hidden state across the full
corpus. This is a dynamic/context-conditioned gene table, not model
fine-tuning.

Txn_Jatin extraction took 12.71 hours. BRIDGE took 11.53 hours. Atomic recovery
accumulators were written during extraction. The bounded watchdog made zero
retries.

## Audit result

`dynamic_embedding_audit.json` proves:

- Txn_Jatin: dynamic, 978,212 samples, 863 files, no fallback
- BRIDGE: dynamic, 978,212 samples, 863 files, no fallback
- scGPT: static, no fallback
- Geneformer: static, no fallback
- all five BulkFormer variants: static, no fallback

All nine exact benchmark-aligned matrices are under `embeddings/`.

## Results

Full-gene scores:

| Track | Txn_Jatin dynamic | BRIDGE dynamic | Winner |
|---|---:|---:|---|
| Gene-set | 0.6386 | 0.6446 | BRIDGE |
| Paired genes | 0.5862 | 0.5888 | BRIDGE |
| Single-gene GO | 0.6570 | 0.6698 | BulkFormer-147M, 0.6810 |
| Single-gene disease | 0.6610 | 0.6714 | Geneformer, 0.6795 |

Overall four-track means:

- BRIDGE dynamic: 0.6437, rank 1/9
- Txn_Jatin dynamic: 0.6357, rank 2/9
- scGPT static: 0.6354, rank 3/9

Compared with the prior static benchmark, BRIDGE improved by 0.0158 mean
AUROC and Txn_Jatin by 0.0110. Txn_Jatin–BRIDGE CCA increased from 0.792 to
0.895.

## Validation and caveats

- Main benchmark status: 0
- Paired supplement status: 0
- Council: 0 FAIL, 16 WARN, 35 PASS, 30 INFO
- GPU was idle after completion.
- Sample embeddings were already expression-conditioned forwards, so their
  scores match the previous run.
- The paired static/dynamic comparison uses separate benchmark runs; unchanged
  static controls varied by at most 0.0043 AUROC because of negative-pair
  resampling.
- SynLethDB and temporal GO have the same availability caveats as before.

## Files

- `DYNAMIC_BENCHMARK_SUMMARY.md`: detailed written result
- `GRAPHS_AND_RESULTS.md`: every plot and primary table
- `figures/`: original and narrative plots, PNG and PDF
- `tables/`: raw, aggregate, comparison, ranking, and provenance tables
- `embeddings/`: all nine exact matrices used for scoring
- `remote_original/`: untouched pull from the A100
- `reproduction/`: dynamic extraction, benchmark, supplement, and watchdog code
- `REMOTE_SHA256SUMS.txt`: authoritative remote hashes
- `SHA256SUMS.txt`: complete local package hashes

