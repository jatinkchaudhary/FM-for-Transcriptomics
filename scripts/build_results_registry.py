#!/usr/bin/env python3
"""Build the UI registry and copy publication-facing result artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ROWS = [
    {
        "id": "Txn_Jatin",
        "label": "Txn_Jatin",
        "family": "ARCHS4 transformer",
        "genes": 16055,
        "embedding_dim": 512,
        "imputation_supported": True,
        "note": "Final 20-epoch static checkpoint; reconstruction plus ESM2, contextual, and sample-contrastive objectives.",
    },
    {
        "id": "Txn_Jatin_OSDR_LoRA",
        "label": "Txn_Jatin OSDR LoRA",
        "family": "ARCHS4 transformer + PEFT",
        "genes": 16055,
        "embedding_dim": 512,
        "imputation_supported": True,
        "note": "Rank-4 LoRA on attention and FFN projections in the last three layers; 0.24% trainable parameters.",
    },
    {
        "id": "Txn_Jatin_contextual",
        "label": "Txn_Jatin contextual",
        "family": "context-averaged embedding",
        "genes": 15916,
        "embedding_dim": 512,
        "imputation_supported": False,
        "note": "Expression-conditioned hidden states averaged across 20,000 ARCHS4 samples; no standalone decoder.",
    },
    {
        "id": "BRIDGE",
        "label": "BRIDGE",
        "family": "masked reconstruction transformer",
        "genes": 15165,
        "embedding_dim": 512,
        "imputation_supported": True,
        "note": "Original reconstruction-focused control checkpoint.",
    },
    *[
        {
            "id": f"BulkFormer_{size}",
            "label": f"BulkFormer-{size}",
            "family": "BulkFormer",
            "genes": 20010,
            "embedding_dim": dim,
            "imputation_supported": True,
            "note": "Native bulk-expression decoder and static learned gene embeddings.",
        }
        for size, dim in [("37M", 128), ("50M", 256), ("93M", 512), ("127M", 640), ("147M", 640)]
    ],
    {
        "id": "ESM2_PCA512_prior",
        "label": "ESM2 PCA-512 prior",
        "family": "protein sequence control",
        "genes": 16055,
        "embedding_dim": 512,
        "imputation_supported": False,
        "note": "Protein-sequence embeddings projected to 512 dimensions; benchmark control, no expression decoder.",
    },
    {
        "id": "ESM3",
        "label": "ESM3",
        "family": "protein sequence control",
        "genes": 15529,
        "embedding_dim": 1536,
        "imputation_supported": False,
        "note": "Native frozen ESM3 sequence embeddings; mean-pooled final-layer residues with BOS/EOS excluded.",
    },
    {
        "id": "Geneformer",
        "label": "Geneformer",
        "family": "single-cell foundation model",
        "genes": 15916,
        "embedding_dim": 768,
        "imputation_supported": False,
        "note": "Rank-value pseudo-cell embedding control; no validated bulk-expression decoder.",
    },
    {
        "id": "scGPT",
        "label": "scGPT",
        "family": "single-cell foundation model",
        "genes": 15916,
        "embedding_dim": 512,
        "imputation_supported": False,
        "note": "Gene-encoder embedding control; no validated bulk-expression decoder.",
    },
]


def clean_records(frame: pd.DataFrame) -> list[dict]:
    frame = frame.replace({np.nan: None, np.inf: None, -np.inf: None})
    return frame.to_dict(orient="records")


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def experiment(
    key: str,
    order: int,
    title: str,
    date: str,
    protocol: str,
    conclusion: str,
    primary_metric: str,
    rows: list[dict],
    files: list[str],
) -> dict:
    return {
        "id": key,
        "order": order,
        "title": title,
        "date": date,
        "protocol": protocol,
        "conclusion": conclusion,
        "primary_metric": primary_metric,
        "rows": rows,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    repo = args.repo.resolve()
    data_dir = repo / "app" / "data"
    results_dir = repo / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_root = workspace / "Txn_Jatin_20epochs/full_ARCHES4_20epoch_H100_20260624_015238"
    benchmark_root = run_root / "benchmarks"
    experiments = []

    loss_source = (
        benchmark_root
        / "txn_jatin_biomarker_pilot_20260725/resources/checkpoint/loss_history.csv"
    )
    loss = pd.read_csv(loss_source)
    copy_file(loss_source, results_dir / "training/loss_history.csv")
    training_rows = []
    for _, row in loss.iterrows():
        training_rows.append(
            {
                "label": f"Epoch {int(row['epoch'])}",
                "model": "Txn_Jatin",
                "metric": "validation reconstruction loss",
                "value": float(row["val_recon_loss"]),
                "secondary": float(row["train_recon_loss"]),
                "secondary_label": "train reconstruction loss",
            }
        )
    experiments.append(
        experiment(
            "training",
            1,
            "ARCHS4 pretraining",
            "2026-06-24 to 2026-07-20",
            "Final 20-epoch run over 958,212 training and 20,000 validation samples; 15% random masking; log1p-TPM.",
            "Logged validation reconstruction loss declined from 0.0869 to 0.0781. The checkpoint also optimized ESM2 prior, contextual alignment, and sample-contrastive terms.",
            "validation reconstruction loss",
            training_rows,
            ["training/loss_history.csv"],
        )
    )

    epoch_report = run_root / "comparisons/epoch7_vs_s66qfh36/results/comparison_report.md"
    if epoch_report.exists():
        copy_file(epoch_report, results_dir / "epoch7_comparison/comparison_report.md")
    epoch_rows = [
        {"label": "DisGeNET centroid", "model": "Txn_Jatin epoch 7", "metric": "AUROC", "value": 0.6909},
        {"label": "DisGeNET centroid", "model": "BRIDGE", "metric": "AUROC", "value": 0.6669},
        {"label": "GO-BP centroid", "model": "Txn_Jatin epoch 7", "metric": "AUROC", "value": 0.8080},
        {"label": "GO-BP centroid", "model": "BRIDGE", "metric": "AUROC", "value": 0.7091},
        {"label": "KEGG centroid", "model": "Txn_Jatin epoch 7", "metric": "AUROC", "value": 0.8979},
        {"label": "KEGG centroid", "model": "BRIDGE", "metric": "AUROC", "value": 0.7330},
    ]
    experiments.append(
        experiment(
            "epoch7_comparison",
            2,
            "Epoch-7 checkpoint comparison",
            "2026-07",
            "Frozen full-corpus contextual embeddings; paired on 14,567 shared genes.",
            "Txn_Jatin epoch 7 led centroid functional prediction, while BRIDGE retained stronger same-set neighborhood ranking.",
            "AUROC",
            epoch_rows,
            ["epoch7_comparison/comparison_report.md"],
        )
    )

    github_source = run_root / "reports/github_protocol_excel_comparisons/source/all_results_long_numeric.csv"
    github = pd.read_csv(github_source)
    github["source"] = github["source"].map(lambda value: Path(str(value)).name)
    copy_file(github_source, results_dir / "github_protocol/all_results_long_numeric.csv")
    wide_source = run_root / "reports/github_protocol_excel_comparisons/source/all_results_wide_numeric.csv"
    copy_file(wide_source, results_dir / "github_protocol/all_results_wide_numeric.csv")
    static_rows = []
    for _, row in github.iterrows():
        static_rows.append(
            {
                "label": row["benchmark"],
                "model": row["model"],
                "metric": "AUROC",
                "value": float(row["AUROC"]),
                "secondary": float(row["AUPRC"]),
                "secondary_label": "AUPRC",
                "scope": row["scope"],
                "family": row["family"],
            }
        )
    experiments.append(
        experiment(
            "github_protocol",
            3,
            "Pinned GitHub gene-embedding protocol",
            "2026-07-21",
            "ylaboratory/gene-embedding-benchmarks-compatible holdout evaluation; 11 models, 29 groups, 319 rows. ANDES was excluded.",
            "Txn_Jatin was strongest on GO among learned expression models, while model rankings varied materially across disease and gene-pair domains.",
            "AUROC",
            static_rows,
            [
                "github_protocol/all_results_long_numeric.csv",
                "github_protocol/all_results_wide_numeric.csv",
            ],
        )
    )

    dynamic_root = workspace / "Txn_Jatin/benchmark_dynamic"
    dynamic = pd.read_csv(dynamic_root / "tables/dynamic_overall_ranking.csv")
    copy_file(
        dynamic_root / "tables/dynamic_overall_ranking.csv",
        results_dir / "dynamic_context/dynamic_overall_ranking.csv",
    )
    copy_file(
        dynamic_root / "tables/dynamic_full_track_summary.csv",
        results_dir / "dynamic_context/dynamic_full_track_summary.csv",
    )
    dynamic_rows = [
        {
            "label": "Dynamic full-track mean",
            "model": row["model"],
            "metric": "AUROC",
            "value": float(row["dynamic_full_track_mean"]),
            "secondary": float(row["delta"]),
            "secondary_label": "delta vs static",
        }
        for _, row in dynamic.iterrows()
    ]
    experiments.append(
        experiment(
            "dynamic_context",
            4,
            "Expression-context embedding benchmark",
            "2026-07",
            "Contextual hidden states accumulated across ARCHS4, then evaluated on gene-set, pair, GO, and disease tracks.",
            "BRIDGE ranked first at 0.6437; Txn_Jatin ranked second at 0.6357. Context helped both over their static full-track means.",
            "AUROC",
            dynamic_rows,
            [
                "dynamic_context/dynamic_overall_ranking.csv",
                "dynamic_context/dynamic_full_track_summary.csv",
            ],
        )
    )

    mask_root = benchmark_root / "whole_gene_mask_2c5c_20260723_132500"
    imputation_frames = []
    rf_frames = []
    for model in ["Txn_Jatin", "BRIDGE", "BulkFormer_37M", "BulkFormer_50M", "BulkFormer_93M", "BulkFormer_127M", "BulkFormer_147M"]:
        imp_path = mask_root / f"results/imputation/{model}/imputation_summary.csv"
        rf_path = mask_root / f"results/masked_rf/{model}/masked_rf_summary.csv"
        imputation_frames.append(pd.read_csv(imp_path))
        rf_frames.append(pd.read_csv(rf_path))
    imputation = pd.concat(imputation_frames, ignore_index=True)
    rf = pd.concat(rf_frames, ignore_index=True)
    raw_rf = pd.read_csv(mask_root / "results/raw_rf_baseline/raw_rf_summary.csv")
    (results_dir / "whole_gene_mask").mkdir(parents=True, exist_ok=True)
    (results_dir / "cancer_rf").mkdir(parents=True, exist_ok=True)
    imputation.to_csv(results_dir / "whole_gene_mask/imputation_all_models.csv", index=False)
    rf.to_csv(results_dir / "cancer_rf/masked_rf_all_models.csv", index=False)
    raw_rf.to_csv(results_dir / "cancer_rf/raw_rf_baseline.csv", index=False)
    copy_file(mask_root / "config/protocol.json", results_dir / "whole_gene_mask/protocol.json")
    copy_file(
        mask_root / "results/unsupported_imputation_rows.csv",
        results_dir / "whole_gene_mask/unsupported_imputation_rows.csv",
    )
    imp_mean = (
        imputation.groupby(["model", "dataset"], as_index=False)[
            ["pcc_global", "mse", "auroc_macro", "auprc_macro"]
        ]
        .mean()
        .sort_values(["dataset", "auroc_macro"], ascending=[True, False])
    )
    mask_rows = []
    for _, row in imp_mean.iterrows():
        mask_rows.append(
            {
                "label": f"{row['dataset']} whole-gene mask",
                "model": row["model"],
                "metric": "AUROC",
                "value": float(row["auroc_macro"]),
                "secondary": float(row["pcc_global"]),
                "secondary_label": "global PCC",
                "mse": float(row["mse"]),
            }
        )
    experiments.append(
        experiment(
            "whole_gene_mask",
            5,
            "15% whole-gene masking",
            "2026-07-23",
            "The same 2,188 genes were hidden across all samples on a 14,585-gene common universe; three seeds; TCGA and OSDR.",
            "Txn_Jatin led TCGA expressed-gene AUROC (0.9284); BRIDGE led global PCC/MSE and all OSDR reconstruction metrics.",
            "AUROC",
            mask_rows,
            [
                "whole_gene_mask/imputation_all_models.csv",
                "whole_gene_mask/unsupported_imputation_rows.csv",
                "whole_gene_mask/protocol.json",
            ],
        )
    )

    cancer = (
        rf.groupby(["model", "panel"], as_index=False)[
            ["oof_auroc", "fold_balanced_accuracy_mean"]
        ]
        .mean()
        .sort_values(["panel", "oof_auroc"], ascending=[True, False])
    )
    cancer_rows = [
        {
            "label": str(row["panel"]).replace("_", " "),
            "model": row["model"],
            "metric": "AUROC",
            "value": float(row["oof_auroc"]),
            "secondary": float(row["fold_balanced_accuracy_mean"]),
            "secondary_label": "balanced accuracy",
        }
        for _, row in cancer.iterrows()
    ]
    experiments.append(
        experiment(
            "cancer_rf",
            6,
            "Cancer-gene identification",
            "2026-07-23",
            "Patient-grouped five-fold random forests for 2-cancer and 5-cancer panels, using masked/imputed genes.",
            "All native decoders preserved near-ceiling tumor-versus-normal signal. These are internal TCGA discrimination results, not validated biomarkers.",
            "AUROC",
            cancer_rows,
            ["cancer_rf/masked_rf_all_models.csv", "cancer_rf/raw_rf_baseline.csv"],
        )
    )

    peft_candidates = pd.DataFrame(
        [
            ["head_only", 1.0401, 0.5950, 0.6350],
            ["bitfit_norm", 0.9447, 0.5575, 0.5981],
            ["last1", 0.6783, 0.5545, 0.5955],
            ["last2", 0.4204, 0.5330, 0.5749],
            ["last3", 0.3177, 0.5588, 0.5739],
            ["lora_attn_all_r4", 0.9359, 0.5987, 0.6272],
            ["lora_attn_last6_r8", 0.9308, 0.5985, 0.6161],
            ["lora_attn_ffn_last3_r4", 0.9648, 0.6047, 0.6547],
        ],
        columns=["model", "masked_reconstruction_mse", "f1_macro", "flight_ground_auroc"],
    )
    (results_dir / "osdr_peft").mkdir(parents=True, exist_ok=True)
    peft_candidates.to_csv(results_dir / "osdr_peft/candidate_summary.csv", index=False)
    peft_report = workspace / "Txn_Jatin_presentation_assets_20260720/osdr_peft_finetuning_and_benchmark_report_20260724.md"
    copy_file(peft_report, results_dir / "osdr_peft/report.md")
    peft_rows = [
        {
            "label": "OSDR PEFT candidate",
            "model": row["model"],
            "metric": "AUROC",
            "value": float(row["flight_ground_auroc"]),
            "secondary": float(row["masked_reconstruction_mse"]),
            "secondary_label": "masked MSE",
        }
        for _, row in peft_candidates.iterrows()
    ]
    experiments.append(
        experiment(
            "osdr_peft",
            7,
            "OSDR gradual unfreezing and LoRA",
            "2026-07-24",
            "Accession-grouped split; 20 epochs; head-only, BitFit/Norm, last 1-3 layers, and three LoRA configurations.",
            "Last-3 unfreezing minimized MSE (0.3177); last-3 attention+FFN LoRA maximized flight-vs-ground AUROC (0.6547) and was selected by the predeclared combined-rank rule.",
            "AUROC",
            peft_rows,
            ["osdr_peft/candidate_summary.csv", "osdr_peft/report.md"],
        )
    )

    go_root = benchmark_root / "individual_go_esm3_20260725/results"
    go_summary = pd.read_csv(go_root / "osdr_go_model_summary_with_esm3.csv")
    go_terms = go_root / "osdr_go_term_scores_with_esm3.csv"
    copy_file(go_root / "osdr_go_model_summary_with_esm3.csv", results_dir / "individual_go/osdr_go_model_summary_with_esm3.csv")
    copy_file(go_terms, results_dir / "individual_go/osdr_go_term_scores_with_esm3.csv")
    copy_file(go_root / "manifest.json", results_dir / "individual_go/manifest.json")
    go_full = go_summary[go_summary["variant"] == "osdr_full"].copy()
    go_rows = [
        {
            "label": "40 individual GO-BP terms",
            "model": row["model"],
            "metric": "AUROC",
            "value": float(row["mean_AUROC"]),
            "secondary": float(row["mean_AUPRC"]),
            "secondary_label": "AUPRC",
        }
        for _, row in go_full.iterrows()
    ]
    experiments.append(
        experiment(
            "individual_go_esm3",
            8,
            "Individual GO terms with ESM3",
            "2026-07-25",
            "Forty frozen GO-BP terms; five-fold out-of-fold GPU linear probes; strict 13,408-gene ESM3 comparison.",
            "ESM3 (0.7817) and ESM2 (0.7808) led sequence-aligned GO recovery; BulkFormer-147M led the expression-model group.",
            "AUROC",
            go_rows,
            [
                "individual_go/osdr_go_model_summary_with_esm3.csv",
                "individual_go/osdr_go_term_scores_with_esm3.csv",
            ],
        )
    )

    sparse_root = benchmark_root / "txn_jatin_biomarker_pilot_20260725"
    sparse_rows = []
    for size in (1000, 2000):
        panel = sparse_root / f"results/panel_{size}"
        reconstruction = json.loads((panel / "reconstruction_summary.json").read_text())
        sparse_rows.append(
            {
                "label": f"{size}-gene observed panel",
                "model": "Txn_Jatin",
                "metric": "global PCC",
                "value": float(reconstruction["pcc_global_sampled_cells"]),
                "secondary": float(reconstruction["expressed_auroc_micro_sampled_cells"]),
                "secondary_label": "expression AUROC",
            }
        )
        for name in ["phenotype_prediction.csv", "pathway_recovery.csv", "biomarker_effect_preservation.csv"]:
            copy_file(panel / name, results_dir / f"sparse_biomarker/panel_{size}/{name}")
        copy_file(panel / "reconstruction_summary.json", results_dir / f"sparse_biomarker/panel_{size}/reconstruction_summary.json")
    copy_file(sparse_root / "REPORT.md", results_dir / "sparse_biomarker/report.md")
    experiments.append(
        experiment(
            "sparse_biomarker",
            9,
            "Sparse-panel biomarker feasibility",
            "2026-07-25",
            "Only 1,000 or 2,000 genes observed; 86-93% of the vocabulary hidden; internal TCGA endpoints.",
            "Literal sparse-panel reconstruction failed (PCC about 0.009; expression AUROC about 0.505). Strong phenotype scores did not validate hidden-gene recovery and must not be framed as biomarker discovery.",
            "global PCC",
            sparse_rows,
            [
                "sparse_biomarker/panel_1000/phenotype_prediction.csv",
                "sparse_biomarker/panel_2000/phenotype_prediction.csv",
                "sparse_biomarker/report.md",
            ],
        )
    )

    ici_root = benchmark_root / "ici_loco_txn_jatin_20260726"
    loco = pd.read_csv(ici_root / "results/loco_summary.csv")
    hallmark = pd.read_csv(ici_root / "results/hallmark_random_effects_meta.csv")
    copy_file(ici_root / "results/loco_summary.csv", results_dir / "immunotherapy/loco_summary.csv")
    copy_file(ici_root / "results/paired_delta_vs_raw.csv", results_dir / "immunotherapy/paired_delta_vs_raw.csv")
    copy_file(ici_root / "results/hallmark_random_effects_meta.csv", results_dir / "immunotherapy/hallmark_random_effects_meta.csv")
    copy_file(
        ici_root / "results/clean_no_pretraining_overlap_summary.csv",
        results_dir / "immunotherapy/clean_no_pretraining_overlap_summary.csv",
    )
    copy_file(
        ici_root / "results/representation_nuisance_diagnostics.csv",
        results_dir / "immunotherapy/representation_nuisance_diagnostics.csv",
    )
    copy_file(ici_root / "INITIAL_IMMUNOTHERAPY_REPORT.md", results_dir / "immunotherapy/report.md")
    for figure in (ici_root / "figures").glob("*.png"):
        copy_file(figure, results_dir / "immunotherapy/figures" / figure.name)
    four = loco[loco["analysis"] == "four_cohort_loco"].copy()
    ici_rows = [
        {
            "label": "Four-cohort leave-one-cohort-out",
            "model": row["method"],
            "metric": "macro AUROC",
            "value": float(row["macro_auroc"]),
            "secondary": float(row["macro_auprc"]),
            "secondary_label": "macro AUPRC",
        }
        for _, row in four.iterrows()
    ]
    experiments.append(
        experiment(
            "immunotherapy",
            10,
            "Exploratory immunotherapy transfer",
            "2026-07-26",
            "Four public ICI cohorts, n=239; nested leave-one-cohort-out logistic evaluation; frozen Txn_Jatin features; explicit pretraining-overlap audit.",
            "Raw expression led macro AUROC (0.6535); Txn-completed expression reached 0.6361; context features were dominated by cohort. IFN-alpha and IFN-gamma response pathways were concordantly higher in responders, but the study is exploratory.",
            "macro AUROC",
            ici_rows,
            [
                "immunotherapy/loco_summary.csv",
                "immunotherapy/hallmark_random_effects_meta.csv",
                "immunotherapy/clean_no_pretraining_overlap_summary.csv",
                "immunotherapy/representation_nuisance_diagnostics.csv",
                "immunotherapy/report.md",
            ],
        )
    )

    tcga_go_root = results_dir / "individual_go_tcga"
    tcga_go_summary = pd.read_csv(tcga_go_root / "tables/model_summary.csv")
    tcga_go_manifest = json.loads((tcga_go_root / "manifest.json").read_text())
    tcga_go_rows = [
        {
            "label": (
                "TCGA native genes"
                if row["scope"] == "tcga_all_genes"
                else "TCGA strict intersection"
            ),
            "model": row["model"],
            "metric": "mean AUROC",
            "value": float(row["mean_AUROC"]),
            "secondary": float(row["mean_AUPRC"]),
            "secondary_label": "mean AUPRC",
        }
        for _, row in tcga_go_summary.iterrows()
    ]
    experiments.append(
        experiment(
            "individual_go_tcga_github",
            11,
            "Official individual GO benchmark on TCGA genes",
            "2026-07-27",
            (
                "Pinned ylaboratory/gene-embedding-benchmarks commit "
                f"{tcga_go_manifest['benchmark_commit']}; all 56 official GO terms; "
                "fixed nested-CV folds and holdout SVC; native and 9,568-gene strict "
                "TCGA scopes."
            ),
            (
                "Txn_Jatin ranked first in both scopes (AUROC 0.8353 native; "
                "0.8439 strict), followed by ESM2 and Txn_Jatin contextual. "
                "Native 1,536-dimensional ESM3 ranked fourth."
            ),
            "mean AUROC",
            tcga_go_rows,
            [
                "individual_go_tcga/tables/model_summary.csv",
                "individual_go_tcga/tables/individual_go_term_scores.csv",
                "individual_go_tcga/tables/per_term_winners.csv",
                "individual_go_tcga/reports/REPORT.md",
            ],
        )
    )

    registry = {
        "generated_utc": tcga_go_manifest["completed_utc"],
        "title": "Txn_Jatin experiment registry",
        "claim_boundary": (
            "Research-use-only. Cancer and immunotherapy analyses are exploratory and do not "
            "establish clinically validated biomarkers."
        ),
        "models": MODEL_ROWS,
        "experiments": sorted(experiments, key=lambda row: row["order"]),
        "counts": {
            "models": len(MODEL_ROWS),
            "experiments": len(experiments),
            "github_protocol_rows": len(github),
            "individual_go_rows": len(pd.read_csv(go_terms)),
            "tcga_individual_go_rows": len(
                pd.read_csv(
                    tcga_go_root / "tables/individual_go_term_scores.csv"
                )
            ),
            "whole_gene_mask_rows": len(imputation),
            "ici_patients": 239,
        },
    }
    (data_dir / "results_registry.json").write_text(
        json.dumps(registry, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(registry["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
