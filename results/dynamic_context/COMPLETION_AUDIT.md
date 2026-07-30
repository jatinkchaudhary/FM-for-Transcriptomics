# Completion audit

| Requirement | Evidence | Status |
|---|---|---|
| Dynamic Txn_Jatin embeddings | `dynamic_embedding_audit.json`: dynamic=true, 978,212 samples, 863 files | Proven |
| Dynamic BRIDGE embeddings | `dynamic_embedding_audit.json`: dynamic=true, 978,212 samples, 863 files | Proven |
| No dynamics for other models | Audit records dynamic=false and no context samples for all seven comparison models | Proven |
| Benchmark against all models | Leaderboard has nine distinct model columns and four tracks in common/full variants | Proven |
| Full data | Configuration and audit require 978,212 context samples per dynamic model; TCGA/OSDR logs show uncapped sample inputs | Proven |
| No fallback embeddings | All nine audit entries have fallback=false | Proven |
| Paired-gene track | `paired_scores.csv`, supplement summary, and status 0 | Proven |
| Regenerated original graphs | Five original benchmark PNGs exist under `figures/` | Proven |
| Regenerated narrative graphs | Dashboard plus four individual plots exist in PNG and PDF | Proven |
| Requested local folder | All outputs reside under `Txn_Jatin/benchmark_dynamic` | Proven |
| Exact remote preservation | `remote_original/` and `REMOTE_SHA256SUMS.txt`; all hashes verified | Proven |
| Reproducibility | Frozen scripts, environment-independent configuration, logs, matrices, summaries, and hashes included | Proven |
| Successful execution | Main status 0; paired status 0; Council 0 failures | Proven |

The automated verifier `verify_dynamic_benchmark.py` checks these invariants
against the current files rather than relying on this narrative.

