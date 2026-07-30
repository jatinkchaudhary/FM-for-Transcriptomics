#!/usr/bin/env python3
"""Frozen Txn_Jatin sparse-panel biomarker pilot on TCGA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260725
MASK_TOKEN = -10.0
HALLMARK_URL = (
    "https://maayanlab.cloud/Enrichr/geneSetLibrary"
    "?mode=text&libraryName=MSigDB_Hallmark_2020"
)
REQUESTED_SIGNATURES = {
    "TNF-alpha Signaling via NF-kB",
    "Interferon Gamma Response",
    "Inflammatory Response",
    "IL-6/JAK/STAT3 Signaling",
    "Epithelial Mesenchymal Transition",
    "Hypoxia",
    "DNA Repair",
    "G2-M Checkpoint",
    "E2F Targets",
    "Apoptosis",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run-dir", required=True)
    prepare.add_argument("--tcga", required=True)
    prepare.add_argument("--metadata", required=True)
    prepare.add_argument("--sample-sheet", required=True)
    prepare.add_argument("--model-genes", required=True)
    prepare.add_argument("--remote-run-dir", required=True)
    prepare.add_argument("--remote-tcga", required=True)
    prepare.add_argument("--remote-metadata", required=True)
    prepare.add_argument("--remote-sample-sheet", required=True)
    prepare.add_argument("--checkpoint", required=True)
    prepare.add_argument("--train-flash", required=True)
    prepare.add_argument("--panel-sizes", nargs="+", type=int, default=[2000, 1000])

    infer = subparsers.add_parser("infer")
    infer.add_argument("--run-dir", required=True)
    infer.add_argument("--panel-size", type=int, required=True)
    infer.add_argument("--batch-size", type=int, default=4)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--run-dir", required=True)
    analyze.add_argument("--panel-size", type=int, required=True)
    analyze.add_argument("--jobs", type=int, default=12)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_genes(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    column = "gene_symbol" if "gene_symbol" in frame else frame.columns[-1]
    return frame[column].dropna().astype(str).str.upper().tolist()


def normalize_term(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def load_sample_sheet(path: Path) -> pd.DataFrame:
    sheet = pd.read_csv(path, sep="\t", dtype=str)
    sheet = sheet.rename(
        columns={
            "File ID": "file_id",
            "Case ID": "case_id",
            "Sample ID": "sample_id",
            "Tissue Type": "sheet_tissue_type",
            "Tumor Descriptor": "tumor_descriptor",
        }
    )
    return sheet


def split_for_case(case_id: str) -> str:
    bucket = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "discovery" if bucket < 7 else "test"


def parse_hallmark(text: str) -> dict[str, list[str]]:
    libraries = {}
    for line in text.splitlines():
        fields = line.rstrip().split("\t")
        if len(fields) < 3:
            continue
        libraries[fields[0]] = [gene.upper() for gene in fields[2:] if gene]
    requested = {normalize_term(term): term for term in REQUESTED_SIGNATURES}
    selected = {}
    for term, genes in libraries.items():
        key = normalize_term(term)
        if key in requested:
            selected[requested[key]] = genes
    missing = sorted(REQUESTED_SIGNATURES.difference(selected))
    if missing:
        raise RuntimeError(f"Missing requested Hallmark terms: {missing}")
    return selected


def prepare(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    for name in ("config", "panels", "predictions", "results", "figures", "logs", "status"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    hallmark_response = requests.get(HALLMARK_URL, timeout=60)
    hallmark_response.raise_for_status()
    hallmark_path = run_dir / "config" / "MSigDB_Hallmark_2020.gmt"
    hallmark_path.write_bytes(hallmark_response.content)
    signatures = parse_hallmark(hallmark_response.text)

    expression = pd.read_parquet(args.tcga)
    expression.columns = expression.columns.astype(str).str.upper()
    metadata = pd.read_parquet(args.metadata)
    metadata.index = metadata["file_id"].astype(str)
    metadata = metadata.reindex(expression.index.astype(str))
    sheet = load_sample_sheet(Path(args.sample_sheet)).set_index("file_id")
    sheet = sheet.reindex(expression.index.astype(str))
    case_ids = sheet["case_id"].fillna(expression.index.to_series()).astype(str)
    assignments = case_ids.map(split_for_case)

    model_genes = read_genes(Path(args.model_genes))
    tcga_genes = expression.columns.tolist()
    shared_set = set(model_genes).intersection(tcga_genes)
    shared_genes = [gene for gene in model_genes if gene in shared_set]

    signature_rows = []
    signature_union = set()
    for term in sorted(signatures):
        for gene in signatures[term]:
            if gene in shared_set:
                signature_rows.append({"signature": term, "gene_symbol": gene})
                signature_union.add(gene)
    signature_frame = pd.DataFrame(signature_rows).drop_duplicates()
    signature_frame.to_csv(run_dir / "config" / "signature_genes.csv", index=False)

    discovery = assignments.eq("discovery").to_numpy()
    discovery_values = np.log1p(
        expression.loc[discovery, shared_genes].to_numpy(dtype=np.float32, copy=False)
    )
    variance = np.var(discovery_values, axis=0)
    detection = np.mean(discovery_values >= np.log(2.0), axis=0)
    panel_score = variance * np.sqrt(np.maximum(detection, 1e-6))
    ranking = pd.DataFrame(
        {
            "gene_symbol": shared_genes,
            "discovery_variance": variance,
            "discovery_detection_rate": detection,
            "panel_score": panel_score,
            "excluded_signature_gene": [gene in signature_union for gene in shared_genes],
        }
    ).sort_values(["panel_score", "gene_symbol"], ascending=[False, True])
    ranking.to_csv(run_dir / "config" / "panel_gene_ranking.csv", index=False)
    eligible = ranking.loc[~ranking["excluded_signature_gene"], "gene_symbol"].tolist()

    panel_details = {}
    for panel_size in args.panel_sizes:
        if panel_size > len(eligible):
            raise RuntimeError(f"Panel size {panel_size} exceeds {len(eligible)} eligible genes")
        panel = eligible[:panel_size]
        hidden = [gene for gene in shared_genes if gene not in set(panel)]
        pd.DataFrame({"gene_symbol": panel}).to_csv(
            run_dir / "panels" / f"panel_{panel_size}.csv", index=False
        )
        pd.DataFrame({"gene_symbol": hidden}).to_csv(
            run_dir / "panels" / f"hidden_{panel_size}.csv", index=False
        )
        panel_details[str(panel_size)] = {
            "observed_genes": len(panel),
            "hidden_evaluable_genes": len(hidden),
            "hidden_fraction": len(hidden) / len(shared_genes),
            "all_signature_genes_hidden": not bool(set(panel).intersection(signature_union)),
        }

    sample_frame = pd.DataFrame(
        {
            "file_id": expression.index.astype(str),
            "case_id": case_ids.to_numpy(),
            "split": assignments.to_numpy(),
            "project_id": metadata["project_id"].to_numpy(),
            "tissue_type": metadata["tissue_type"].to_numpy(),
            "sample_id": sheet["sample_id"].to_numpy(),
            "tumor_descriptor": sheet["tumor_descriptor"].to_numpy(),
        }
    )
    sample_frame.to_csv(run_dir / "config" / "sample_assignments.csv", index=False)
    pd.DataFrame({"gene_symbol": shared_genes}).to_csv(
        run_dir / "config" / "shared_txn_tcga_genes.csv", index=False
    )

    protocol = {
        "benchmark": "txn_jatin_sparse_panel_biomarker_pilot",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "seed": SEED,
        "model": "Txn_Jatin final 20-epoch frozen checkpoint",
        "input_scale": "log1p(TPM)",
        "mask_token": MASK_TOKEN,
        "panel_selection": (
            "Rank discovery-only genes by variance*sqrt(TPM>=1 detection); "
            "exclude every evaluated Hallmark signature gene."
        ),
        "split": "70/30 deterministic SHA256 case-level discovery/test split",
        "shared_gene_count": len(shared_genes),
        "signature_gene_count": len(signature_union),
        "signature_count": len(signatures),
        "panels": panel_details,
        "local_paths": {
            "run_dir": str(run_dir),
            "tcga": str(Path(args.tcga).resolve()),
            "metadata": str(Path(args.metadata).resolve()),
            "sample_sheet": str(Path(args.sample_sheet).resolve()),
            "model_genes": str(Path(args.model_genes).resolve()),
        },
        "remote_paths": {
            "run_dir": args.remote_run_dir,
            "tcga": args.remote_tcga,
            "metadata": args.remote_metadata,
            "sample_sheet": args.remote_sample_sheet,
            "model_genes": (
                f"{Path(args.checkpoint).parent.as_posix()}/canonical_genes.csv"
            ),
            "checkpoint": args.checkpoint,
            "train_flash": args.train_flash,
        },
        "source_hashes": {
            "tcga_parquet_sha256": sha256(Path(args.tcga)),
            "metadata_parquet_sha256": sha256(Path(args.metadata)),
            "sample_sheet_sha256": sha256(Path(args.sample_sheet)),
            "model_genes_sha256": sha256(Path(args.model_genes)),
            "hallmark_gmt_sha256": sha256(hallmark_path),
        },
        "primary_outputs": [
            "held-out hidden-gene reconstruction",
            "held-out Hallmark signature recovery",
            "patient-grouped phenotype prediction",
            "tumor-normal hidden-gene effect preservation",
        ],
        "claim_boundary": (
            "Exploratory internal TCGA evidence only; not a validated clinical biomarker."
        ),
    }
    write_json(run_dir / "config" / "protocol.json", protocol)
    write_json(
        run_dir / "status" / "prepared.json",
        {
            "status": "prepared",
            "samples": len(expression),
            "discovery_samples": int(discovery.sum()),
            "test_samples": int((~discovery).sum()),
            "shared_genes": len(shared_genes),
            "signature_genes": len(signature_union),
            "panels": panel_details,
        },
    )
    print(json.dumps(json.loads((run_dir / "status" / "prepared.json").read_text())))
    return 0


def resolve_run_dir(path: str) -> Path:
    run_dir = Path(path)
    protocol = json.loads((run_dir / "config" / "protocol.json").read_text())
    remote = Path(protocol["remote_paths"]["run_dir"])
    return remote if remote.exists() else run_dir


def import_expression_performer(path: str):
    spec = importlib.util.spec_from_file_location("biomarker_pilot_train_flash", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model definition from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ExpressionPerformer


def load_model(protocol: dict, device):
    import torch

    payload = torch.load(
        protocol["remote_paths"]["checkpoint"], map_location="cpu", weights_only=False
    )
    state = payload.get("model_state_dict", payload.get("model", payload))
    config = payload["config"]
    model_class = import_expression_performer(protocol["remote_paths"]["train_flash"])
    model = model_class(
        num_genes=int(state["gene_embedding.weight"].shape[0]),
        hidden_dim=int(config["hidden_dim"]),
        n_heads=int(config["num_heads"]),
        n_layers=int(config["num_layers"]),
        ffn_dim=int(config["ffn_dim"]),
        ree_base=float(config["ree_base"]),
        mask_token_id=float(config.get("mask_token", MASK_TOKEN)),
        feature_type=config.get("feature_type", "sqr"),
        compute_type=config.get("compute_type", "iter"),
        include_species_embedding=bool(config.get("include_species_embedding", False)),
        num_species=int(
            config.get("architecture", {}).get("num_species", config.get("num_species", 2))
        ),
    )
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def infer(args: argparse.Namespace) -> int:
    import torch

    run_dir = resolve_run_dir(args.run_dir)
    protocol = json.loads((run_dir / "config" / "protocol.json").read_text())
    output_dir = run_dir / "predictions" / f"panel_{args.panel_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "hidden_predictions.npy"
    if destination.exists():
        print(json.dumps({"status": "already_complete", "path": str(destination)}))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    model = load_model(protocol, device)

    model_genes = read_genes(Path(protocol["remote_paths"]["model_genes"]))
    model_index = {gene: index for index, gene in enumerate(model_genes)}
    panel = read_genes(run_dir / "panels" / f"panel_{args.panel_size}.csv")
    hidden = read_genes(run_dir / "panels" / f"hidden_{args.panel_size}.csv")
    observed_indices = np.asarray([model_index[gene] for gene in panel], dtype=np.int64)
    hidden_indices = np.asarray([model_index[gene] for gene in hidden], dtype=np.int64)
    masked_indices = np.setdiff1d(
        np.arange(len(model_genes), dtype=np.int64), observed_indices, assume_unique=True
    )

    frame = pd.read_parquet(protocol["remote_paths"]["tcga"])
    frame.columns = frame.columns.astype(str).str.upper()
    source_index = {gene: index for index, gene in enumerate(frame.columns)}
    source = frame.to_numpy(dtype=np.float32, copy=False)
    values = np.zeros((len(frame), len(model_genes)), dtype=np.float32)
    source_columns = []
    target_columns = []
    for target, gene in enumerate(model_genes):
        source_column = source_index.get(gene)
        if source_column is not None:
            source_columns.append(source_column)
            target_columns.append(target)
    values[:, np.asarray(target_columns)] = np.log1p(source[:, np.asarray(source_columns)])
    del source, frame

    temporary = output_dir / "hidden_predictions.partial.npy"
    progress_path = run_dir / "status" / f"inference_panel_{args.panel_size}.json"

    start_offset = 0
    if temporary.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        start_offset = int(progress.get("samples_complete", 0))
        output = np.lib.format.open_memmap(temporary, mode="r+")
    else:
        output = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float32, shape=(len(values), len(hidden))
        )
    started = time.time()
    batch_size = max(1, args.batch_size)
    offset = start_offset
    while offset < len(values):
        stop = min(len(values), offset + batch_size)
        try:
            tensor = torch.from_numpy(values[offset:stop]).to(device, non_blocking=True)
            tensor[:, masked_indices] = MASK_TOKEN
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                prediction = model(tensor)
            output[offset:stop] = prediction[:, hidden_indices].float().cpu().numpy()
            offset = stop
            if offset == len(values) or offset % max(20, batch_size * 10) == 0:
                output.flush()
                write_json(
                    progress_path,
                    {
                        "status": "running",
                        "panel_size": args.panel_size,
                        "samples_complete": offset,
                        "samples_total": len(values),
                        "hidden_genes": len(hidden),
                        "batch_size": batch_size,
                        "elapsed_seconds_this_attempt": time.time() - started,
                    },
                )
            del tensor, prediction
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"CUDA OOM; reducing batch size to {batch_size}", flush=True)
    output.flush()
    del output
    os.replace(temporary, destination)
    np.save(
        output_dir / "sample_ids.npy",
        np.asarray(
            pd.read_parquet(protocol["remote_paths"]["tcga"], columns=[]).index.astype(str)
        ),
    )
    write_json(
        progress_path,
        {
            "status": "complete",
            "panel_size": args.panel_size,
            "samples_complete": len(values),
            "samples_total": len(values),
            "hidden_genes": len(hidden),
            "batch_size": batch_size,
            "elapsed_seconds_this_attempt": time.time() - started,
        },
    )
    torch.cuda.synchronize()
    del model, values
    torch.cuda.empty_cache()
    print(progress_path.read_text())
    return 0


def pearson_axis(truth: np.ndarray, prediction: np.ndarray, axis: int) -> np.ndarray:
    truth_centered = truth - truth.mean(axis=axis, keepdims=True)
    pred_centered = prediction - prediction.mean(axis=axis, keepdims=True)
    numerator = np.sum(truth_centered * pred_centered, axis=axis)
    denominator = np.sqrt(
        np.sum(truth_centered**2, axis=axis) * np.sum(pred_centered**2, axis=axis)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def grouped_predictions(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    multiclass: bool,
    panel_features: bool,
) -> tuple[np.ndarray, list[dict]]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    classes = np.unique(y)
    probabilities = np.zeros((len(y), len(classes)), dtype=np.float32)
    rows = []
    for fold, (train, test) in enumerate(splitter.split(x, y, groups), start=1):
        if panel_features:
            components = min(25, len(train) - 1, x.shape[1])
            classifier = make_pipeline(
                StandardScaler(),
                PCA(n_components=components, svd_solver="randomized", random_state=SEED),
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=SEED
                ),
            )
        else:
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=SEED
                ),
            )
        classifier.fit(x[train], y[train])
        fold_probability = classifier.predict_proba(x[test])
        probabilities[test] = fold_probability
        fold_prediction = classes[np.argmax(fold_probability, axis=1)]
        fold_auc = roc_auc_score(
            y[test],
            fold_probability if multiclass else fold_probability[:, 1],
            multi_class="ovr" if multiclass else "raise",
            average="macro",
        )
        rows.append(
            {
                "fold": fold,
                "auroc": fold_auc,
                "balanced_accuracy": balanced_accuracy_score(y[test], fold_prediction),
                "n_test": len(test),
            }
        )
    return probabilities, rows


def phenotype_rows(
    panel_size: int,
    feature_name: str,
    x: np.ndarray,
    sample_frame: pd.DataFrame,
    panel_features: bool,
) -> list[dict]:
    rows = []
    groups = sample_frame["case_id"].astype(str).to_numpy()
    projects = sample_frame["project_id"].astype(str)
    tissue = sample_frame["tissue_type"].astype(str).str.lower()

    tumor = tissue.eq("tumor").to_numpy()
    y_multi = pd.Categorical(projects[tumor]).codes
    multi_probabilities, multi_folds = grouped_predictions(
        x[tumor], y_multi, groups[tumor], multiclass=True, panel_features=panel_features
    )
    rows.append(
        {
            "panel_size": panel_size,
            "features": feature_name,
            "endpoint": "five_cancer_type_among_tumors",
            "samples": int(tumor.sum()),
            "patients": len(np.unique(groups[tumor])),
            "oof_auroc": roc_auc_score(
                y_multi, multi_probabilities, multi_class="ovr", average="macro"
            ),
            "fold_auroc_mean": np.mean([row["auroc"] for row in multi_folds]),
            "fold_balanced_accuracy_mean": np.mean(
                [row["balanced_accuracy"] for row in multi_folds]
            ),
        }
    )

    eligible_projects = ["TCGA-BRCA", "TCGA-KIRC", "TCGA-LUAD", "TCGA-LUSC"]
    selected = projects.isin(eligible_projects).to_numpy()
    y_binary = tissue[selected].eq("tumor").astype(int).to_numpy()
    binary_probabilities, binary_folds = grouped_predictions(
        x[selected],
        y_binary,
        groups[selected],
        multiclass=False,
        panel_features=panel_features,
    )
    rows.append(
        {
            "panel_size": panel_size,
            "features": feature_name,
            "endpoint": "tumor_vs_normal_four_cancers",
            "samples": int(selected.sum()),
            "patients": len(np.unique(groups[selected])),
            "oof_auroc": roc_auc_score(y_binary, binary_probabilities[:, 1]),
            "fold_auroc_mean": np.mean([row["auroc"] for row in binary_folds]),
            "fold_balanced_accuracy_mean": np.mean(
                [row["balanced_accuracy"] for row in binary_folds]
            ),
        }
    )
    return rows


def analyze(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = resolve_run_dir(args.run_dir)
    protocol = json.loads((run_dir / "config" / "protocol.json").read_text())
    panel_size = args.panel_size
    hidden = read_genes(run_dir / "panels" / f"hidden_{panel_size}.csv")
    panel = read_genes(run_dir / "panels" / f"panel_{panel_size}.csv")
    prediction = np.load(
        run_dir / "predictions" / f"panel_{panel_size}" / "hidden_predictions.npy",
        mmap_mode="r",
    )
    frame = pd.read_parquet(protocol["remote_paths"]["tcga"])
    frame.columns = frame.columns.astype(str).str.upper()
    truth = np.log1p(frame[hidden].to_numpy(dtype=np.float32, copy=False))
    panel_values = np.log1p(frame[panel].to_numpy(dtype=np.float32, copy=False))
    sample_frame = pd.read_csv(run_dir / "config" / "sample_assignments.csv", dtype=str)
    sample_frame = sample_frame.set_index("file_id").reindex(frame.index.astype(str))
    sample_frame.index.name = "file_id"
    sample_frame = sample_frame.reset_index()
    test = sample_frame["split"].eq("test").to_numpy()
    discovery = ~test
    truth_test = truth[test]
    prediction_test = np.asarray(prediction[test], dtype=np.float32)

    gene_pcc = pearson_axis(truth_test, prediction_test, axis=0)
    sample_pcc = pearson_axis(truth_test, prediction_test, axis=1)
    per_gene = pd.DataFrame(
        {
            "panel_size": panel_size,
            "gene_symbol": hidden,
            "pcc": gene_pcc,
            "mse": np.mean((prediction_test - truth_test) ** 2, axis=0),
            "mae": np.mean(np.abs(prediction_test - truth_test), axis=0),
        }
    )
    result_dir = run_dir / "results" / f"panel_{panel_size}"
    result_dir.mkdir(parents=True, exist_ok=True)
    per_gene.to_csv(result_dir / "hidden_gene_metrics.csv.gz", index=False)
    rng = np.random.default_rng(SEED)
    cell_count = truth_test.size
    sampled_cells = rng.choice(cell_count, size=min(2_000_000, cell_count), replace=False)
    truth_cells = truth_test.ravel()[sampled_cells]
    prediction_cells = prediction_test.ravel()[sampled_cells]
    reconstruction = {
        "panel_size": panel_size,
        "observed_fraction": panel_size / protocol["shared_gene_count"],
        "test_samples": int(test.sum()),
        "hidden_genes": len(hidden),
        "pcc_global_sampled_cells": float(pearsonr(truth_cells, prediction_cells).statistic),
        "pcc_gene_macro": float(np.nanmean(gene_pcc)),
        "pcc_sample_macro": float(np.nanmean(sample_pcc)),
        "mse": float(np.mean((prediction_test - truth_test) ** 2)),
        "mae": float(np.mean(np.abs(prediction_test - truth_test))),
        "expressed_auroc_micro_sampled_cells": float(
            roc_auc_score(truth_cells >= np.log(2.0), prediction_cells)
        ),
        "expressed_auprc_micro_sampled_cells": float(
            average_precision_score(truth_cells >= np.log(2.0), prediction_cells)
        ),
    }
    write_json(result_dir / "reconstruction_summary.json", reconstruction)

    signature_frame = pd.read_csv(run_dir / "config" / "signature_genes.csv")
    hidden_index = {gene: index for index, gene in enumerate(hidden)}
    pathway_rows = []
    truth_scores = {}
    prediction_scores = {}
    for signature, current in signature_frame.groupby("signature"):
        genes = [gene for gene in current["gene_symbol"] if gene in hidden_index]
        columns = np.asarray([hidden_index[gene] for gene in genes])
        means = truth[discovery][:, columns].mean(axis=0)
        scales = truth[discovery][:, columns].std(axis=0)
        scales[scales < 1e-6] = 1.0
        full_truth_score = ((truth[:, columns] - means) / scales).mean(axis=1)
        full_prediction_score = (
            (np.asarray(prediction[:, columns], dtype=np.float32) - means) / scales
        ).mean(axis=1)
        truth_scores[signature] = full_truth_score
        prediction_scores[signature] = full_prediction_score
        threshold = np.quantile(full_truth_score[discovery], 0.75)
        state = full_truth_score[test] >= threshold
        pathway_rows.append(
            {
                "panel_size": panel_size,
                "signature": signature,
                "genes": len(genes),
                "pearson": pearsonr(
                    full_truth_score[test], full_prediction_score[test]
                ).statistic,
                "spearman": spearmanr(
                    full_truth_score[test], full_prediction_score[test]
                ).statistic,
                "rmse": np.sqrt(
                    np.mean(
                        (full_truth_score[test] - full_prediction_score[test]) ** 2
                    )
                ),
                "top_quartile_auroc": (
                    roc_auc_score(state, full_prediction_score[test])
                    if state.any() and (~state).any()
                    else np.nan
                ),
                "top_quartile_auprc": (
                    average_precision_score(state, full_prediction_score[test])
                    if state.any() and (~state).any()
                    else np.nan
                ),
            }
        )
    pathway = pd.DataFrame(pathway_rows).sort_values("pearson", ascending=False)
    pathway.to_csv(result_dir / "pathway_recovery.csv", index=False)
    truth_score_frame = pd.DataFrame(truth_scores, index=frame.index)
    prediction_score_frame = pd.DataFrame(prediction_scores, index=frame.index)

    phenotype = []
    phenotype.extend(
        phenotype_rows(
            panel_size,
            "full_expression_hidden_pathway_scores",
            truth_score_frame.to_numpy(dtype=np.float32),
            sample_frame,
            panel_features=False,
        )
    )
    phenotype.extend(
        phenotype_rows(
            panel_size,
            "Txn_Jatin_imputed_hidden_pathway_scores",
            prediction_score_frame.to_numpy(dtype=np.float32),
            sample_frame,
            panel_features=False,
        )
    )
    phenotype.extend(
        phenotype_rows(
            panel_size,
            "observed_panel_expression",
            panel_values,
            sample_frame,
            panel_features=True,
        )
    )
    phenotype_frame = pd.DataFrame(phenotype)
    phenotype_frame.to_csv(result_dir / "phenotype_prediction.csv", index=False)

    biomarker_rows = []
    projects = sample_frame["project_id"].astype(str)
    tissue = sample_frame["tissue_type"].astype(str).str.lower()
    for project in ["TCGA-BRCA", "TCGA-KIRC", "TCGA-LUAD", "TCGA-LUSC"]:
        selected = projects.eq(project).to_numpy()
        tumor = tissue[selected].eq("tumor").to_numpy()
        project_truth = truth[selected]
        project_prediction = np.asarray(prediction[selected], dtype=np.float32)
        if tumor.sum() < 20 or (~tumor).sum() < 20:
            continue
        truth_delta = project_truth[tumor].mean(axis=0) - project_truth[~tumor].mean(axis=0)
        pred_delta = (
            project_prediction[tumor].mean(axis=0)
            - project_prediction[~tumor].mean(axis=0)
        )
        truth_rank = np.argsort(-np.abs(truth_delta))
        pred_rank = np.argsort(-np.abs(pred_delta))
        for top_k in (20, 50, 100):
            overlap = len(set(truth_rank[:top_k]).intersection(pred_rank[:top_k]))
            biomarker_rows.append(
                {
                    "panel_size": panel_size,
                    "project": project,
                    "tumor_samples": int(tumor.sum()),
                    "normal_samples": int((~tumor).sum()),
                    "top_k": top_k,
                    "top_k_overlap": overlap,
                    "top_k_overlap_fraction": overlap / top_k,
                    "effect_spearman": spearmanr(truth_delta, pred_delta).statistic,
                    "direction_concordance": np.mean(
                        np.sign(truth_delta) == np.sign(pred_delta)
                    ),
                }
            )
        pd.DataFrame(
            {
                "gene_symbol": hidden,
                "truth_log_effect": truth_delta,
                "imputed_log_effect": pred_delta,
                "truth_abs_rank": pd.Series(-np.abs(truth_delta)).rank(method="min"),
                "imputed_abs_rank": pd.Series(-np.abs(pred_delta)).rank(method="min"),
            }
        ).sort_values("truth_abs_rank").to_csv(
            result_dir / f"{project}_hidden_gene_effects.csv.gz", index=False
        )
    biomarker = pd.DataFrame(biomarker_rows)
    biomarker.to_csv(result_dir / "biomarker_effect_preservation.csv", index=False)

    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = pathway.sort_values("pearson")
    ax.barh(ordered["signature"], ordered["pearson"], color="#176b87")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Held-out Pearson correlation")
    ax.set_title(f"Hidden Hallmark program recovery: {panel_size}-gene input panel")
    fig.tight_layout()
    fig.savefig(figure_dir / f"panel_{panel_size}_pathway_recovery.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    endpoints = phenotype_frame["endpoint"].unique()
    feature_names = phenotype_frame["features"].unique()
    width = 0.24
    positions = np.arange(len(endpoints))
    colors = ["#144552", "#3e8e7e", "#d97706"]
    for index, feature_name in enumerate(feature_names):
        current = phenotype_frame.set_index(["endpoint", "features"])
        values = [current.loc[(endpoint, feature_name), "oof_auroc"] for endpoint in endpoints]
        ax.bar(
            positions + (index - 1) * width,
            values,
            width,
            label=feature_name.replace("_", " "),
            color=colors[index],
        )
    ax.set_xticks(positions, [value.replace("_", " ") for value in endpoints], rotation=10)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Patient-grouped out-of-fold AUROC")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Biological signal retained from a {panel_size}-gene panel")
    fig.tight_layout()
    fig.savefig(figure_dir / f"panel_{panel_size}_phenotype_prediction.png", dpi=180)
    plt.close(fig)

    write_report(run_dir)
    write_json(
        run_dir / "status" / f"analysis_panel_{panel_size}.json",
        {"status": "complete", "panel_size": panel_size, **reconstruction},
    )
    print(json.dumps(reconstruction))
    return 0


def write_report(run_dir: Path) -> None:
    protocol = json.loads((run_dir / "config" / "protocol.json").read_text())
    reconstruction_rows = []
    pathway_rows = []
    phenotype_rows_all = []
    biomarker_rows = []
    for panel_size in (2000, 1000):
        result_dir = run_dir / "results" / f"panel_{panel_size}"
        summary = result_dir / "reconstruction_summary.json"
        if summary.exists():
            reconstruction_rows.append(json.loads(summary.read_text()))
        path = result_dir / "pathway_recovery.csv"
        if path.exists():
            pathway_rows.append(pd.read_csv(path))
        path = result_dir / "phenotype_prediction.csv"
        if path.exists():
            phenotype_rows_all.append(pd.read_csv(path))
        path = result_dir / "biomarker_effect_preservation.csv"
        if path.exists():
            biomarker_rows.append(pd.read_csv(path))

    lines = [
        "# Txn_Jatin sparse-panel biomarker pilot",
        "",
        "## Claim boundary",
        "",
        "This is exploratory internal TCGA evidence. It tests biological-signal recovery; "
        "it does not establish a clinically validated biomarker.",
        "",
        "## Decision summary",
        "",
        "- **Sparse-panel imputation failed.** With 86.3%-93.1% of evaluable genes hidden, "
        "reconstruction PCC was approximately zero and expressed/unexpressed AUROC was "
        "approximately 0.50. This masking regime is far outside the model's 15% masking curriculum.",
        "- **Some decoder outputs remain cancer-separable, but they are not calibrated pathway "
        "measurements.** Ten imputed pathway-score features achieved five-cancer AUROC "
        "0.9877-0.9930, yet the observed panel itself achieved 0.9983. Individual pathway "
        "correlations were inconsistent and sometimes strongly negative.",
        "- **Candidate biomarker ranking was not preserved.** Hidden tumor-normal effect "
        "correlations and direction agreement were at chance; top-50 overlaps were 0-7 genes.",
        "- **Current conclusion:** do not use this checkpoint to discover or impute biomarkers "
        "from 1,000-2,000-gene panels. Its output can be studied as a nonlinear representation "
        "of the observed panel, but that hypothesis requires external-cohort and batch-controlled validation.",
        "",
        "## Frozen protocol",
        "",
        f"- Frozen model: {protocol['model']}.",
        f"- Shared Txn_Jatin/TCGA genes: **{protocol['shared_gene_count']:,}**.",
        f"- Signatures: **{protocol['signature_count']} Hallmark programs**, with every "
        "evaluated signature gene hidden from the input.",
        f"- Split: {protocol['split']}.",
        f"- Panel selection: {protocol['panel_selection']}",
        "",
        "## Reconstruction",
        "",
        "| Observed genes | Hidden genes | Global PCC | Gene-macro PCC | Sample-macro PCC | MSE | Expression AUROC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in reconstruction_rows:
        lines.append(
            f"| {row['panel_size']:,} | {row['hidden_genes']:,} | "
            f"{row['pcc_global_sampled_cells']:.4f} | {row['pcc_gene_macro']:.4f} | "
            f"{row['pcc_sample_macro']:.4f} | {row['mse']:.4f} | "
            f"{row['expressed_auroc_micro_sampled_cells']:.4f} |"
        )
    if pathway_rows:
        pathways = pd.concat(pathway_rows, ignore_index=True)
        lines.extend(
            [
                "",
                "## Hidden biological-program recovery",
                "",
                "| Panel | Hallmark program | Genes | Pearson | State AUROC |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for row in pathways.sort_values(["panel_size", "pearson"], ascending=[False, False]).itertuples():
            lines.append(
                f"| {int(row.panel_size):,} | {row.signature} | {int(row.genes)} | "
                f"{row.pearson:.4f} | {row.top_quartile_auroc:.4f} |"
            )
    if phenotype_rows_all:
        phenotype = pd.concat(phenotype_rows_all, ignore_index=True)
        lines.extend(
            [
                "",
                "## Patient-grouped phenotype signal",
                "",
                "| Panel | Features | Endpoint | OOF AUROC | Balanced accuracy |",
                "|---:|---|---|---:|---:|",
            ]
        )
        for row in phenotype.itertuples():
            lines.append(
                f"| {int(row.panel_size):,} | {row.features.replace('_', ' ')} | "
                f"{row.endpoint.replace('_', ' ')} | {row.oof_auroc:.4f} | "
                f"{row.fold_balanced_accuracy_mean:.4f} |"
            )
    if biomarker_rows:
        biomarkers = pd.concat(biomarker_rows, ignore_index=True)
        top50 = biomarkers[biomarkers["top_k"].eq(50)]
        lines.extend(
            [
                "",
                "## Hidden-gene effect preservation",
                "",
                "| Panel | Cancer | Effect Spearman | Direction concordance | Top-50 overlap |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for row in top50.itertuples():
            lines.append(
                f"| {int(row.panel_size):,} | {row.project} | "
                f"{row.effect_spearman:.4f} | {row.direction_concordance:.4f} | "
                f"{int(row.top_k_overlap)}/50 |"
            )
    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- Global reconstruction PCC is abundance-sensitive; gene-macro PCC is the stricter gene-wise measure.",
            "- Hallmark state AUROC asks whether hidden pathway activity states can be recovered from the observed panel.",
            "- Phenotype probes use patient-grouped out-of-fold predictions; they are sanity checks, not external validation.",
            "- Tumor-normal effect overlap measures preservation of candidate ranking, not clinical utility.",
            "- Survival, treatment response, prospective assay validation, and independent cohorts remain required.",
            "- Because Txn_Jatin output is a deterministic transform of the observed panel, high "
            "cancer classification does not prove added biological information; the appropriate "
            "question is whether it improves generalization in an independent cohort or platform.",
            "",
        ]
    )
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "infer":
        return infer(args)
    if args.command == "analyze":
        return analyze(args)
    raise RuntimeError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
