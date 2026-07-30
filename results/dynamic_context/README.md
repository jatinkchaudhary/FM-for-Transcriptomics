# Txn_Jatin dynamic benchmark

This folder contains the completed full-corpus dynamic/contextual gene
embedding benchmark.

## Design

- Txn_Jatin: dynamic mean per-gene hidden states over all 978,212 ARCHS4
  samples.
- BRIDGE: the same dynamic protocol over all 978,212 samples.
- scGPT, Geneformer, and BulkFormer 37M/50M/93M/127M/147M: unchanged static
  gene embeddings.
- No model was fine-tuned.
- Full TCGA and OSDR sample inputs, all eligible genes, five folds, 40
  GO/disease terms, and the staged TRRUST paired supplement.

## Headline result

Across the four full-gene tracks:

1. Dynamic BRIDGE: 0.6437
2. Dynamic Txn_Jatin: 0.6357
3. Static scGPT: 0.6354

Dynamic contextualization improved every reported Txn_Jatin and BRIDGE gene
track relative to the preceding static benchmark. Txn_Jatin–BRIDGE CCA
similarity increased from 0.792 to 0.895.

## Start here

- [Detailed summary](DYNAMIC_BENCHMARK_SUMMARY.md)
- [Graph and result index](GRAPHS_AND_RESULTS.md)
- [One-page dashboard](figures/dynamic_story_dashboard.png)
- [Completion audit](COMPLETION_AUDIT.md)
- [Restart-safe context](context.md)
- [Exact remote copy](remote_original/)
- [Reproduction code](reproduction/)
- [Integrity verifier](verify_dynamic_benchmark.py)

Both remote benchmark stages exited with status `0`, and the Council reported
0 failures.

