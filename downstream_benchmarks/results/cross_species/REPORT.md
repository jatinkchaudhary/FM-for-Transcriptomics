# Cross-species tissue transfer

## Question

Can a tissue probe learned exclusively from human GTEx samples transfer to an
independent mouse bulk RNA-seq atlas without using mouse labels during training or
model selection?

## Protocol

- Human training set: 1,841 GTEx recount2 samples across 10 matched tissues.
- Mouse test set: all 547 matching Tabula Muris Senis bulk samples from GSE132040.
- Features: 15,556 one-to-one MGI human-mouse orthologs.
- Normalization: length-adjusted TPM followed by `log1p`, using the same human-gene
  lengths for both species.
- Probe: class-weighted multinomial logistic regression. Its regularization was
  selected by five-fold CV on human data only.
- Uncertainty: 2,000 mouse-tissue-stratified bootstrap replicates, seed 42.
- Leakage: the 947 GSE132040 GEO sample accessions were checked against the ARCHS4
  training metadata; the exact intersection was zero.

## Results

| Representation | Mouse macro-F1 (95% CI) | Balanced accuracy (95% CI) | Macro-AUROC (95% CI) |
|---|---:|---:|---:|
| ESPRESSO | 0.9419 (0.9222-0.9599) | 0.9409 (0.9206-0.9593) | 0.9927 (0.9850-0.9986) |
| BulkFormer 147M | 0.7970 (0.7771-0.8183) | 0.8329 (0.8113-0.8532) | 0.9853 (0.9786-0.9915) |
| Raw ortholog expression, PCA-64 | 0.6587 (0.6267-0.6882) | 0.6696 (0.6409-0.6971) | 0.9199 (0.9083-0.9318) |

The paired ESPRESSO-minus-raw macro-F1 difference was **+0.2832** (95% CI
0.2487 to 0.3191). The corresponding balanced-accuracy difference was +0.2713
(0.2386 to 0.3029), and macro-AUROC difference was +0.0729 (0.0612 to 0.0842).
Against BulkFormer 147M, ESPRESSO's paired macro-F1 difference was **+0.1449**
(0.1223 to 0.1677) and its balanced-accuracy difference was +0.1081 (0.0851 to
0.1318).

## Interpretation

ESPRESSO transferred tissue identity across species substantially better than the
matched raw-expression baseline. Because all tuning was restricted to human samples
and the mouse accession overlap was zero, this result supports a genuine
cross-species representation capability rather than mouse-label fitting. It does not
by itself establish transfer for disease phenotypes or unseen tissues.

The confusion matrices, per-sample predictions, paired bootstrap table, and
human/mouse UMAP are stored beside this report.
