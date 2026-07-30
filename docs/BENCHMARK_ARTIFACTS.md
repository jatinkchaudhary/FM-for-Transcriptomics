# Benchmark code and result archive

The repository publishes the final benchmark code and measured output files
available for this study. The machine-readable
[`results/ARTIFACT_INVENTORY.csv`](../results/ARTIFACT_INVENTORY.csv) records
every file under `protocols/` and `results/`, its byte size, and SHA-256 digest.

## Layout

| Benchmark | Reproduction code | Complete result archive |
|---|---|---|
| Epoch-7 Txn_Jatin versus BRIDGE | `protocols/epoch7_comparison/` | `results/epoch7_comparison/` |
| Pinned GitHub protocol | `protocols/github_benchmark/` | `results/github_protocol/` |
| Dynamic contextual benchmark | `protocols/dynamic_context/` | `results/dynamic_context/` |
| Whole-gene masking and cancer RF | `protocols/whole_gene_mask/` | `results/whole_gene_mask/full_run/` |
| OSDR PEFT | `protocols/osdr_peft/` | `results/osdr_peft/` |
| OSDR individual GO and ESM3 | `protocols/individual_go/` | `results/individual_go/full_run/` |
| Official TCGA individual GO | `protocols/individual_go_tcga/` | `results/individual_go_tcga/` |
| Sparse biomarker pilot | `protocols/sparse_biomarker/` | `results/sparse_biomarker/full_run/` |
| Immunotherapy transfer | `protocols/immunotherapy/` | `results/immunotherapy/full_run/` |
| External GTEx recovery | `scripts/prepare_gtex_ui_test.py`, `scripts/benchmark_gtex_all_models.py` | `results/gtex_external/` |

The shorter tables directly under each result directory remain the canonical
publication-facing summaries. `full_run/` directories preserve detailed
per-mask, per-seed, per-sample, tuning, prediction, diagnostic, configuration,
and figure outputs from the completed run.

## Deliberate exclusions

Model checkpoints, embedding arrays, raw and processed expression matrices,
downloaded source cohorts, caches, and logs are excluded. They are either too
large, externally licensed, or transient. Files whose names contain
`REJECTED` are retained only as an audit trail and must not be interpreted as
accepted scientific results.

The repository also contains launchers for experiments that were deliberately
skipped, including ANDES and some temporal/gene-pair runs. Their presence makes
the execution history reproducible; the absence of a final output means the
experiment was not claimed as completed.
