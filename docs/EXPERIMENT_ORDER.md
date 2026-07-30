# Experiment order and protocols

| Order | Experiment | Main protocol | Primary output |
|---:|---|---|---|
| 1 | ARCHS4 pretraining | 15% masked reconstruction on log1p-TPM plus ESM2, context, and sample-contrastive losses | `results/training/loss_history.csv` |
| 2 | Epoch-7 comparison | Frozen contextual embeddings on shared genes | `results/epoch7_comparison/comparison_report.md` |
| 3 | GitHub protocol | Pinned gene-level and gene-pair holdout evaluation | `results/github_protocol/all_results_long_numeric.csv` |
| 4 | Dynamic context | ARCHS4 hidden-state accumulation followed by downstream tracks | `results/dynamic_context/dynamic_overall_ranking.csv` |
| 5 | Whole-gene masking | Same 2,188 genes hidden across samples; three seeds; TCGA and OSDR | `results/whole_gene_mask/imputation_all_models.csv` |
| 6 | Cancer RF | Patient-grouped 5-fold RF on 2-cancer and 5-cancer panels | `results/cancer_rf/masked_rf_all_models.csv` |
| 7 | OSDR PEFT | Head/BitFit/last-1-3/LoRA sweep; accession-grouped split | `results/osdr_peft/candidate_summary.csv` |
| 8 | OSDR individual GO + ESM3 | Earlier 40-term, 5-fold OOF GPU linear probes; strict 13,408-gene comparison | `results/individual_go/osdr_go_term_scores_with_esm3.csv` |
| 9 | Sparse biomarker pilot | 1,000/2,000 observed genes, 86-93% hidden | `results/sparse_biomarker/report.md` |
| 10 | Immunotherapy transfer | Four-cohort nested leave-one-cohort-out evaluation, n=239 | `results/immunotherapy/report.md` |
| 11 | Official TCGA-scoped individual GO | Pinned GitHub commit; 56 terms; fixed folds; nested-CV/holdout SVC; 12 models; native and 9,568-gene strict scopes | `results/individual_go_tcga/tables/individual_go_term_scores.csv` |
| 12 | External GTEx whole-gene recovery | 12 diverse recount2 GTEx samples; log1p-CPM; fixed 15% whole-gene mask; strict 2,117-gene decoder intersection | `results/gtex_external/gtex_all_model_summary.csv` |
| 13 | Atlas-augmented immunotherapy | Fixed immune/pathway atlas; nested calibrated LOCO; six ablations; random-effects pathway consistency | `results/immunotherapy/atlas_augmented/REPORT.md` |

## Recalculation hierarchy

1. Rebuild data-preparation artifacts from the documented external datasets.
2. Train or acquire each checkpoint outside this repository.
3. Set checkpoint and resource paths in `config/model_paths.remote.json`.
4. Execute the scripts under `protocols/` in table order.
5. Run `scripts/build_results_registry.py`.
6. Run `scripts/validate_bundle.py`.

The result-registry builder only assembles existing outputs; it never invents
or recomputes benchmark values.
