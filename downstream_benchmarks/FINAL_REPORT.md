# ESPRESSO four-task downstream evaluation

## Executive conclusion

ESPRESSO shows its clearest advantages on **gene-level essentiality** and
**human-to-mouse transfer**. It also preserves tissue biology better than BulkFormer
in a crossed bulk-study integration experiment. On TCGA survival, it is competitive
with raw expression and PCA for OS and better than BulkFormer, but it does not
universally improve PFI or simple expression baselines.

The resulting publication position is therefore specific: ESPRESSO is a compact,
biologically structured and cross-species-capable representation, not a universal
replacement for endpoint-tuned expression models.

## 1. DepMap gene essentiality

The benchmark used 13,342 strictly shared genes from DepMap Public 24Q2: 1,666
common essentials and 11,676 pan-neutral genes. A gene-disjoint nested-CV logistic
probe was evaluated with 2,000 paired bootstrap replicates.

| Model | AUROC | AUPRC |
|---|---:|---:|
| ESPRESSO | **0.9335** | **0.7428** |
| BRIDGE | 0.9296 | 0.7019 |
| Geneformer | 0.9286 | 0.6989 |
| scGPT | 0.9264 | 0.7018 |
| ESM2 | 0.8901 | 0.6007 |
| BulkFormer 147M | 0.8881 | 0.5618 |
| Mean expression | 0.8587 | 0.4698 |
| Gene length | 0.4961 | 0.1226 |

ESPRESSO exceeded BulkFormer 147M by +0.0454 AUROC (95% CI 0.0388 to
0.0518) and +0.1810 AUPRC (0.1572 to 0.2033). It exceeded ESM2 by +0.0434
AUROC and +0.1420 AUPRC. Against BRIDGE, AUROC tied (+0.0039, CI crossing
zero) while AUPRC improved by +0.0409 (0.0231 to 0.0600).

**Meaning:** the gain is not explained by sequence alignment alone and is strongest
where positives are rare, as reflected by AUPRC.

## 2. Human-to-mouse tissue transfer

A tissue classifier was selected and trained using 1,841 human GTEx samples only,
then evaluated without tuning on 547 GSE132040 mouse samples across ten tissues and
15,556 one-to-one orthologs. The 947 mouse GEO accessions had zero exact overlap
with the 499,845-accession ARCHS4 training manifest.

| Representation | Mouse macro-F1 | Balanced accuracy | Macro-AUROC |
|---|---:|---:|---:|
| ESPRESSO | **0.9419** | **0.9409** | **0.9927** |
| BulkFormer 147M | 0.7970 | 0.8329 | 0.9853 |
| Raw ortholog PCA-64 | 0.6587 | 0.6696 | 0.9199 |

ESPRESSO improved macro-F1 over BulkFormer by +0.1449 (paired 95% CI
0.1223 to 0.1677) and over raw transfer by +0.2832 (0.2493 to 0.3189).

**Meaning:** this is the strongest differentiating downstream result. It tests
zero-shot species transfer rather than within-species interpolation.

## 3. TCGA-CDR survival

The patient-grouped nested-CV analysis used OS and PFI from TCGA-CDR for the five
cancers available in the processed matrix. All 48 model-by-endpoint-by-scope OOF
results and 36 paired deltas are retained.

| Representation | Pan-cancer OS C-index | Pan-cancer PFI C-index |
|---|---:|---:|
| PCA-64 | **0.6849** | **0.7003** |
| Raw log1p-TPM | 0.6778 | 0.6909 |
| ESPRESSO | 0.6747 | 0.6693 |
| BulkFormer 147M | 0.6476 | 0.6647 |

For OS, ESPRESSO beat BulkFormer by +0.0271 (0.0093 to 0.0450), tied raw
(-0.0031, CI crossing zero), and tied PCA (-0.0102, CI crossing zero). For PFI,
it tied BulkFormer but was below raw and PCA. Across pan-cancer plus five cancers,
ESPRESSO's point estimate exceeded BulkFormer in 4/6 OS and 5/6 PFI scopes.

**Meaning:** ESPRESSO contains prognostic signal and is more useful than BulkFormer
for OS, but expression baselines remain difficult to beat. This supports a qualified
clinical-utility statement, not superiority.

## 4. Cross-study integration

The integration cohort contains 1,339 samples in a crossed breast/kidney/lung/skin
by GTEx/TCGA design. It uses pinned recount2 count matrices; this is explicitly not
the originally proposed recount3 cohort.

| Method | Batch score | Biology score | Combined score |
|---|---:|---:|---:|
| ESPRESSO | 0.328 | **0.687** | **0.544** |
| Raw log1p-CPM | 0.356 | 0.647 | 0.531 |
| Harmony PCA-64 | **0.461** | 0.567 | 0.525 |
| BulkFormer 147M | 0.335 | 0.615 | 0.503 |
| PCA-64 | 0.451 | 0.515 | 0.489 |

The paired local-neighborhood bootstrap versus BulkFormer showed higher biology
preservation (+0.0197, 0.0150 to 0.0244), lower batch mixing (-0.0144,
-0.0207 to -0.0081), and a small positive combined difference (+0.0060,
0.0023 to 0.0100).

**Meaning:** ESPRESSO preserves tissue structure but is not a batch-correction
algorithm. Harmony is better when aggressive batch mixing is the primary goal.

## Validity and leakage controls

- ESPRESSO, BulkFormer 147M, PCA-64, and raw expression passed the TCGA adapter
  gate. BRIDGE failed because silhouette was -0.0022 despite AUROC 0.9941.
- No fallback representation entered a downstream sample benchmark.
- The ARCHS4 manifest contains 499,845 normalized GEO/SRA accessions.
- Checkpoint paths, hashes, gene orders, dimensions, preprocessing, OOF predictions,
  and bootstrap tables are retained beside the aggregate results.

## Limitations

1. scGPT and Geneformer lacked validated native bulk-sample encoders in the pinned
   environment, so they appear in gene-level essentiality but not sample-level tasks.
2. The integration cohort is recount2, not recount3.
3. The TCGA matrix contains five cancers rather than the anticipated ten.
4. Some small per-cancer Cox folds required stronger penalties or reached
   convergence warnings; OOF predictions and selected alphas are retained for audit.
5. Drug response was optional and was not run; DepMap essentiality is the completed
   primary task from that branch.

## Publication interpretation

The coherent mechanistic story is that ESPRESSO's sequence, contextual, and
cross-species objectives improve biological organization at the gene level and make
that organization transferable across species. The survival and integration nulls
are equally informative: learned biological structure does not automatically erase
batch or outperform direct expression on every clinical endpoint. Presenting those
boundaries makes the cross-species and essentiality claims more credible.
