# Production Atlas and Ollama Validation
Validated on the remote H100 on 2026-07-30 UTC.

## Installed Services

- Ollama `0.32.0`
- Model: `gpt-oss:120b`
- Model digest:
  `a951a23b46a1f6093dafee2ea481d634b4e31ac720a8a16f3f91e04f5a40ecd9`
- Parameters: 116.8B
- Quantization: MXFP4
- Model storage: 65 GB
- Service: enabled and active user `systemd` unit
- User linger: enabled
- Bind address: `127.0.0.1:11434`
- Keep-alive: zero; weights unload after each language-head request

Txn_Jatin is unloaded before the language-head stage because the expression
decoder and 120B model cannot safely share a single 80 GB H100.

## Production Atlas

- Version: `production_v1_20260730`
- Genes: 16,057
- Runtime references: 3,120
- GTEx: 9,662 source samples across 54 tissue labels
- TCGA: 11,284 source samples across 33 projects
- ARCHS4: 506 human and 506 mouse references sampled across 64 shards per
  species
- Disease and phenotype sets: 11,700
- Gene annotation queries: 16,057 MyGene.info responses
- Mouse-human mappings: 24,590 from the official MGI
  `HOM_MouseHumanSequence.rpt`
- Runtime index size: 139 MB compressed
- Normalization: counts to library-size CPM to `log1p`
- All generated artifacts passed `SHA256SUMS.txt` verification.

The GTEx and TCGA matrices are the ARCHS4-hosted recount2 releases dated
2017. Their alignment pipeline differs from ARCHS4, which remains a source
domain limitation rather than a hidden implementation detail.

## End-to-End Validation

A GTEx subcutaneous-adipose profile was masked at 15% and passed through the
complete backend:

1. 2,409 genes masked.
2. Txn_Jatin imputation completed in 1.706 seconds.
3. 16,055 model genes matched; two atlas-only genes were explicitly unresolved.
4. Atlas plus `gpt-oss:120b` completed in 24.212 seconds.
5. Best match: GTEx subcutaneous adipose centroid.
6. Cosine similarity: 0.987829.
7. Species evidence: human 1.0.
8. Language head status: `ok`, 2,148 output characters.
9. Backend remained healthy and released the Txn_Jatin checkpoint before
   Ollama generation.

An exact mouse ARCHS4 reference also returned HTTP 200 and ranked itself
first. Species evidence was nearly tied (mouse 0.502, human 0.498), and the
language head reported that ambiguity instead of asserting a confident
species call. This reflects the harmonized cross-species expression space and
must be improved with an independent labelled mouse tissue validation set
before publication use.

## Scientific Boundaries

- Similarities and species weights are not calibrated probabilities.
- TCGA resemblance is not a cancer diagnosis.
- Disease-set overlap is association evidence, not patient-level risk.
- Unlabelled ARCHS4 references cannot support tissue claims.
- Imputed values and observed values must remain distinguishable.
- Clinical use requires independent cohorts, calibration, and prospective
  validation.
