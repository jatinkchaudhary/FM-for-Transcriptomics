# Txn_Jatin initial ICI response study

This folder is a self-contained exploratory test of frozen Txn_Jatin
representations in four public pretreatment immunotherapy cohorts.

## Read first

- Full scientific report: `INITIAL_IMMUNOTHERAPY_REPORT.md`
- Consolidated workbook: `results/ici_initial_results.xlsx`
- Primary LOCO summary: `results/loco_summary.csv`
- Clean-cohort sensitivity: `results/clean_no_pretraining_overlap_summary.csv`
- Main figure: `figures/loco_auroc_heatmap.png`
- Biology figure: `figures/hallmark_effect_heterogeneity.png`

## Experiment order

1. `prepare_ici_cohorts.py` downloads/reads cohort files, maps genes, matches
   patients, converts FPKM to TPM, and saves `log1p(TPM)` matrices.
2. `extract_txn_ici_embeddings.py` runs the frozen epoch-20 checkpoint on the
   H100. Assay-missing genes use mask token `-10`.
3. `evaluate_ici_transfer.py` performs nested leave-one-cohort-out response
   evaluation, bootstrapping, and the within-cohort diagnostic.
4. `analyze_ici_biology.py` quantifies cohort nuisance and performs exploratory
   Hallmark random-effects analysis.
5. `build_ici_accession_map.py` maps evaluated patients to archive accessions.
6. `evaluate_clean_cohort_sensitivity.py` excludes Hugo, the only cohort with
   exact pretraining-accession overlap.

## Main result

The frozen Txn context vector does not transfer as an ICI response predictor
(macro LOCO AUROC 0.406 versus 0.654 for raw expression). Txn-completed
expression retains most raw-expression performance (macro AUROC 0.636).
Interferon-alpha, interferon-gamma, and allograft-rejection pathway activity is
higher in responders across all four cohorts, but this remains exploratory.

## Re-run locally

From the repository root:

```powershell
.\.venv\Scripts\python.exe <script> --root Txn_Jatin_20epochs\full_ARCHES4_20epoch_H100_20260624_015238\benchmarks\ici_loco_txn_jatin_20260726
```

The extraction script additionally requires the remote checkpoint,
`canonical_genes.csv`, and `train_flash.py`; see the recorded command in
`logs/remote_extract_corrected_command.log`.

## Interpretation boundary

This is an exploratory feasibility study, not an externally validated
clinical biomarker. Primary evidence is cohort-held-out performance. The
within-cohort and pathway analyses are diagnostic and hypothesis-generating.
