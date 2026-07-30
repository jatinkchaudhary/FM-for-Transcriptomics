**Subject: Txn_Jatin benchmark comparison files and interpretation**

Hi Walt,

I have completed and rebuilt the Excel comparison files for the Txn_Jatin
evaluation. The workbooks contain 11 models, 29 completed benchmark groups,
and 319 model-level results, with no failed jobs. The highest value in each
benchmark row is bolded and highlighted.

The files are:

1. `01_GitHub_Protocol_All_Models.xlsx` - the complete comparison, including
   AUROC, AUPRC, PR@10, model summaries, winners, and the full numeric table.
2. `02_Gene_Level_All_Models.xlsx` - GO, OMIM, and Temporal GO comparisons.
3. `03_Gene_Pair_All_Models.xlsx` - SL, POMBE, TF, and NG comparisons using
   sum, product, and concatenation operators on full and shared gene sets.
4. `04_Model_Rankings_and_Winners.xlsx` - aggregate rankings and the winning
   model for each benchmark.

The evaluation uses the code and supplied data splits from
`ylaboratory/gene-embedding-benchmarks`, pinned at commit
`d1320026a2a4ee033d49517f91e2d1c2ccc8df1e`. The ANDES gene-set analyses,
including KEGG-GO, were intentionally excluded from this delivery.

## ESM2 and the ESM2 control in our workbooks

ESM2 is included in the original GitHub repository. In that study it is one
of the sequence-based reference embeddings evaluated alongside the other gene
representations.

Our column is named `ESM2_PCA512_prior`, rather than simply `ESM2`, because it
is related to, but not identical to, the repository's native ESM2 artifact.
We started with ESM2-derived protein-sequence features and reduced them to 512
dimensions using principal component analysis (PCA). This representation
retains approximately 98.3% of the variance in the source features and covers
approximately 97.6% of the 16,055-gene training vocabulary.

This ESM2-derived table has two roles in our experiment:

- It is a sequence-only control that can be benchmarked independently.
- It provides the biological sequence prior used to guide Txn_Jatin during
  training.

It is important not to describe our `ESM2_PCA512_prior` results as a direct
reproduction of the repository's native `ESM2` results. PCA compresses and
rotates the coordinates. This matters especially for elementwise-product gene
pair benchmarks because elementwise multiplication depends on the exact
coordinate basis.

## What `Txn_Jatin` means in the workbooks

`Txn_Jatin` is the **static gene-identity embedding table** taken from the
final 20-epoch checkpoint. The model was trained on the harmonized ARCHS4
expression collection, containing approximately 978,000 human and mouse
samples and 16,055 canonical genes. Each gene has one learned 512-dimensional
identity vector. The word "static" means that a gene has the same vector
regardless of which biological sample is being considered.

Although the exported table is static, it was not learned as an isolated
lookup table. It was optimized as part of the full transformer using four
training signals:

1. **Masked-expression reconstruction:** During training, approximately 15%
   of the gene-expression values in a sample were hidden. The model had to
   reconstruct those values from the observed genes. This encourages the gene
   embeddings to represent stable co-expression and regulatory relationships.
2. **ESM2 gene-prior alignment:** The learned identity vector for each gene was
   encouraged to remain aligned with its ESM2-derived sequence prior. This
   gives the model a source of biological structure beyond expression alone.
3. **Context-prior alignment:** After expression values and other genes in a
   sample had influenced a gene's hidden state, that expression-conditioned
   state was also encouraged to remain biologically consistent with the ESM2
   prior. This connects the static identity and the dynamic sample context.
4. **Sample-level contrastive learning:** Two partial views of the same sample
   were encouraged to produce similar sample representations, while different
   samples were kept more distinct. This is intended to make the model less
   sensitive to missing genes and more aware of sample-level biological
   structure.

The static `Txn_Jatin` benchmark therefore asks: **What persistent biological
information has been stored in each gene's identity vector after the complete
training process?** The benchmark freezes the table and does not use GO, OMIM,
SL, TF, NG, or other benchmark labels to update it. This is important because
it keeps the downstream evaluation separate from model training.

## What `Txn_Jatin_contextual` means in the workbooks

`Txn_Jatin_contextual` comes from the same final frozen checkpoint, but it is
constructed from the transformer's expression-conditioned hidden states
rather than directly from the static identity table.

For every input sample, a gene's hidden state depends on three things: its
learned identity, its expression value in that sample, and the other observed
genes with which it interacts through transformer attention. The same gene can
therefore have a different hidden state in different samples.

To create one reproducible vector per gene for the GitHub benchmark, we passed
20,000 ARCHS4 samples through the frozen model, accumulated the contextual
hidden states observed for each gene, and averaged them. The resulting table
contains one 512-dimensional vector per gene and can be evaluated by the same
benchmark code as all the static embedding methods.

This contextual table therefore captures:

- the underlying learned gene identity;
- the gene's average expression-conditioned state;
- the average influence of its co-expressed genes and attention context across
  the 20,000 sampled transcriptomes.

The contextual representation asks a different question from the static
table: **What representation does the trained model assign to a gene after
averaging the biological contexts in which that gene was observed?**

Contextualization is not expected to improve every benchmark. Averaging across
many samples can strengthen recurring co-expression or interaction signals,
but it can also smooth rare, tissue-specific, or condition-specific behavior.
This is why I have retained both `Txn_Jatin` and `Txn_Jatin_contextual` in the
workbooks instead of treating one as a replacement for the other. Their
comparison shows which tasks benefit from persistent gene identity and which
benefit from accumulated expression context.

## Why some values differ from the GitHub repository

The benchmark implementation and supplied splits are aligned with the GitHub
repository, but the embedding files are not always the same artifacts used in
the paper. Using the nearest available repository analogues, the mean absolute
AUROC differences across the 29 benchmark rows are approximately:

- scGPT versus repository `SCGPT-HUMAN`: **0.022**
- Geneformer-V2-104M versus repository `GF-12L95M`: **0.036**
- `ESM2_PCA512_prior` versus repository native `ESM2`: **0.061**

The largest ESM2 difference is approximately 0.202 and occurs in an
elementwise-product benchmark. This is consistent with the PCA basis change
described above, rather than indicating that the benchmark protocol changed.

There are several additional sources of difference:

- Our scGPT table comes from a specific `tdc/scGPT` checkpoint and its raw gene
  encoder, whereas the repository reports separately prepared `SCGPT-HUMAN`
  and `SCGPT-PANCANCER` artifacts.
- Our Geneformer model is Geneformer-V2-104M, rather than the older 12-layer
  95M model used as the nearest repository comparison.
- Gene symbols were converted to Entrez identifiers, and duplicate mappings
  were averaged.
- The repository's 38-model reference intersection contains 11,355 genes. Our
  strict shared universe contains 9,568 genes because it is the intersection
  of that reference universe with all 11 locally evaluated models. Before
  applying the repository reference intersection, the local models share
  approximately 14,525 genes.
- BRIDGE, Txn_Jatin, Txn_Jatin_contextual, and the BulkFormer variants are not
  published reference embeddings in the repository, so there are no original
  repository values for those models to reproduce.

For these reasons, the most defensible interpretation is the internally
controlled comparison in the attached workbooks. Every displayed model was
evaluated using the same benchmark implementation, supplied splits, identifier
mapping, and task definitions. An exact reproduction of the paper's baseline
numbers would require the exact Zenodo embedding files and the original
38-model intersection.

Best,

Jatin
