#!/usr/bin/env python3
"""Human GTEx to mouse Tabula Muris Senis tissue-transfer benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from sample_adapters import native_embedding


TISSUE_MAP_MOUSE = {
    "Brain": "Brain",
    "Heart": "Heart",
    "Kidney": "Kidney",
    "Liver": "Liver",
    "Lung": "Lung",
    "Limb_Muscle": "Muscle",
    "Pancreas": "Pancreas",
    "Skin": "Skin",
    "Spleen": "Spleen",
    "Small_Intestine": "Small Intestine",
}
TISSUE_MAP_GTEX = {
    "Brain": "Brain",
    "Heart": "Heart",
    "Kidney": "Kidney",
    "Liver": "Liver",
    "Lung": "Lung",
    "Muscle": "Muscle",
    "Pancreas": "Pancreas",
    "Skin": "Skin",
    "Spleen": "Spleen",
    "Small Intestine": "Small Intestine",
}
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def decode(values):
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values]
    )


def one_to_one_orthologs(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t")
    rows = []
    for _, group in data.groupby("DB Class Key"):
        human = group.loc[group["NCBI Taxon ID"].eq(9606), "Symbol"].dropna().unique()
        mouse = group.loc[group["NCBI Taxon ID"].eq(10090), "Symbol"].dropna().unique()
        if len(human) == 1 and len(mouse) == 1:
            rows.append({"human": human[0].upper(), "mouse": mouse[0].upper()})
    return pd.DataFrame(rows).drop_duplicates(["human", "mouse"])


def counts_to_log_tpm(counts, lengths):
    rate = counts / np.maximum(lengths[None, :], 1.0)
    denominator = rate.sum(axis=1, keepdims=True)
    return np.log1p(rate / np.maximum(denominator, 1e-12) * 1e6).astype(np.float32)


def load_human(gtex_path, annotations_path, tissues, maximum_per_tissue):
    annotations = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    ensembl_to_symbol = {}
    for symbol, item in annotations.items():
        records = item.get("ensembl", {})
        if isinstance(records, dict):
            records = [records]
        for record in records if isinstance(records, list) else []:
            gene = record.get("gene") if isinstance(record, dict) else None
            if gene:
                ensembl_to_symbol[str(gene).split(".")[0]] = symbol.upper()
    with h5py.File(gtex_path) as handle:
        genes = decode(handle["meta/genes"][:])
        labels_raw = decode(handle["meta/smts"][:])
        symbols = np.asarray(
            [
                ensembl_to_symbol.get(gene.split(".")[0], "")
                if gene.upper().startswith("ENSG")
                else gene.upper()
                for gene in genes
            ]
        )
        valid_genes = symbols != ""
        selected = []
        rng = np.random.default_rng(42)
        labels = np.asarray([TISSUE_MAP_GTEX.get(value, "") for value in labels_raw])
        for tissue in tissues:
            index = np.flatnonzero(labels == tissue)
            if len(index) > maximum_per_tissue:
                index = np.sort(rng.choice(index, maximum_per_tissue, replace=False))
            selected.extend(index.tolist())
        selected = np.asarray(sorted(selected))
        counts = handle["data/expression"][selected][:, valid_genes].astype(np.float32)
    return counts, symbols[valid_genes], labels[selected], selected


def load_mouse(matrix_path, metadata_path, orthologs, tissues):
    metadata = pd.read_csv(metadata_path)
    sample_to_tissue = {}
    for _, row in metadata.iterrows():
        source = str(row["source name"])
        base = source.rsplit("_", 1)[0]
        tissue = TISSUE_MAP_MOUSE.get(base)
        if tissue in tissues:
            sample_to_tissue[str(row["Sample name"])] = tissue
    matrix = pd.read_csv(matrix_path, index_col=0)
    matrix.columns = matrix.columns.str.replace(".gencode.vM19", "", regex=False)
    selected = [column for column in matrix.columns if column in sample_to_tissue]
    mouse_to_human = dict(zip(orthologs["mouse"], orthologs["human"]))
    mapped = pd.Index([mouse_to_human.get(str(gene).upper(), "") for gene in matrix.index])
    valid = mapped != ""
    values = matrix.loc[valid, selected].copy()
    values.index = mapped[valid]
    values = values.groupby(level=0).sum()
    return (
        values.T.to_numpy(dtype=np.float32),
        values.index.to_numpy(dtype=str),
        np.asarray([sample_to_tissue[column] for column in selected]),
        np.asarray(selected),
    )


def align(matrix, genes, shared):
    index = {gene: position for position, gene in enumerate(genes)}
    return matrix[:, [index[gene] for gene in shared]]


def tune_probe(X, y):
    candidates = []
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for c_value in C_GRID:
        scores = []
        for train, valid in folds.split(X, y):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value, max_iter=5000, class_weight="balanced",
                    multi_class="ovr", random_state=42,
                ),
            )
            model.fit(X[train], y[train])
            scores.append(f1_score(y[valid], model.predict(X[valid]), average="macro"))
        candidates.append((float(np.mean(scores)), c_value))
    best = sorted(candidates, key=lambda item: (-item[0], item[1]))[0][1]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=best, max_iter=5000, class_weight="balanced",
            multi_class="ovr", random_state=42,
        ),
    )
    model.fit(X, y)
    return model, best


def metrics(y, prediction, probability):
    binary = label_binarize(y, classes=np.arange(probability.shape[1]))
    return {
        "macro_f1": f1_score(y, prediction, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "macro_auroc": roc_auc_score(binary, probability, average="macro", multi_class="ovr"),
    }


def bootstrap(y, prediction, probability, replicates=2000):
    rng = np.random.default_rng(42)
    classes = np.unique(y)
    groups = [np.flatnonzero(y == value) for value in classes]
    rows = []
    for _ in range(replicates):
        index = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        rows.append(metrics(y[index], prediction[index], probability[index]))
    return {
        f"{metric}_{bound}": float(np.quantile([row[metric] for row in rows], quantile))
        for metric in ("macro_f1", "balanced_accuracy", "macro_auroc")
        for bound, quantile in (("ci_low", 0.025), ("ci_high", 0.975))
    }


def paired_bootstrap(y, left, right, replicates=2000):
    rng = np.random.default_rng(42)
    groups = [np.flatnonzero(y == value) for value in np.unique(y)]
    rows = []
    for _ in range(replicates):
        index = np.concatenate(
            [rng.choice(group, len(group), replace=True) for group in groups]
        )
        left_metrics = metrics(y[index], left[0][index], left[1][index])
        right_metrics = metrics(y[index], right[0][index], right[1][index])
        rows.append(
            {
                metric: left_metrics[metric] - right_metrics[metric]
                for metric in ("macro_f1", "balanced_accuracy", "macro_auroc")
            }
        )
    return {
        f"delta_{metric}": metrics(y, left[0], left[1])[metric]
        - metrics(y, right[0], right[1])[metric]
        for metric in ("macro_f1", "balanced_accuracy", "macro_auroc")
    } | {
        f"{metric}_{bound}": float(
            np.quantile([row[metric] for row in rows], quantile)
        )
        for metric in ("macro_f1", "balanced_accuracy", "macro_auroc")
        for bound, quantile in (("ci_low", 0.025), ("ci_high", 0.975))
    }


def espresso_embeddings(values, shared, runtime_path, config_path):
    runtime_parent = str(Path(runtime_path).resolve().parent)
    if runtime_parent not in sys.path:
        sys.path.insert(0, runtime_parent)
    spec = importlib.util.spec_from_file_location("espresso_runtime", runtime_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    runtime = module.ModelRuntime(
        Path(config_path),
        [{"id": "Txn_Jatin", "label": "ESPRESSO", "imputation_supported": True}],
    )
    runtime.ensure_loaded("Txn_Jatin")
    aligned = np.zeros((len(values), len(runtime.model_genes)), dtype=np.float32)
    shared_index = {gene: position for position, gene in enumerate(shared)}
    for model_position, gene in enumerate(runtime.model_genes):
        position = shared_index.get(gene)
        if position is not None:
            aligned[:, model_position] = values[:, position]
    embedding = runtime._embed_aligned(aligned)
    runtime.unload()
    return embedding


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtex", type=Path, required=True)
    parser.add_argument("--mouse-matrix", type=Path, required=True)
    parser.add_argument("--mouse-metadata", type=Path, required=True)
    parser.add_argument("--orthologs", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--gene-info", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--competitors", nargs="*", default=[])
    parser.add_argument("--human-per-tissue", type=int, default=250)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tissues = sorted(TISSUE_MAP_MOUSE.values())
    orthologs = one_to_one_orthologs(args.orthologs)
    human_counts, human_genes, human_labels, human_ids = load_human(
        args.gtex, args.annotations, tissues, args.human_per_tissue
    )
    mouse_counts, mouse_genes, mouse_labels, mouse_ids = load_mouse(
        args.mouse_matrix, args.mouse_metadata, orthologs, tissues
    )
    gene_info = pd.read_csv(args.gene_info)
    lengths = dict(zip(gene_info["gene_symbol"].str.upper(), gene_info["gene_length"]))
    shared = sorted(
        set(human_genes) & set(mouse_genes) & set(orthologs["human"]) & set(lengths)
    )
    length_vector = np.asarray([lengths[gene] for gene in shared], dtype=np.float32)
    human = counts_to_log_tpm(align(human_counts, human_genes, shared), length_vector)
    mouse = counts_to_log_tpm(align(mouse_counts, mouse_genes, shared), length_vector)

    encoder = LabelEncoder().fit(tissues)
    y_human = encoder.transform(human_labels)
    y_mouse = encoder.transform(mouse_labels)
    representations = {}
    raw_pca = make_pipeline(StandardScaler(), PCA(n_components=64, random_state=42))
    representations["Raw_ortholog_PCA64"] = (
        raw_pca.fit_transform(human),
        raw_pca.transform(mouse),
    )
    representations["ESPRESSO"] = (
        espresso_embeddings(human, shared, args.runtime, args.model_config),
        espresso_embeddings(mouse, shared, args.runtime, args.model_config),
    )
    for competitor in args.competitors:
        human_embedding, _ = native_embedding(
            pd.DataFrame(human, columns=shared),
            competitor,
            args.runtime,
            args.model_config,
        )
        mouse_embedding, _ = native_embedding(
            pd.DataFrame(mouse, columns=shared),
            competitor,
            args.runtime,
            args.model_config,
        )
        representations[competitor] = (human_embedding, mouse_embedding)
    rows = []
    prediction_rows = []
    method_outputs = {}
    for method, (human_X, mouse_X) in representations.items():
        probe, selected_c = tune_probe(human_X, y_human)
        prediction = probe.predict(mouse_X)
        probability = probe.predict_proba(mouse_X)
        method_outputs[method] = (prediction, probability)
        estimate = metrics(y_mouse, prediction, probability)
        interval = bootstrap(y_mouse, prediction, probability)
        rows.append(
            {
                "method": method,
                "human_samples": len(y_human),
                "mouse_samples": len(y_mouse),
                "tissues": len(tissues),
                "ortholog_genes": len(shared),
                "selected_C": selected_c,
                **estimate,
                **interval,
            }
        )
        prediction_rows.extend(
            {
                "method": method,
                "sample_id": sample,
                "true_tissue": encoder.inverse_transform([truth])[0],
                "predicted_tissue": encoder.inverse_transform([predicted])[0],
            }
            for sample, truth, predicted in zip(mouse_ids, y_mouse, prediction)
        )
        matrix = confusion_matrix(y_mouse, prediction, labels=np.arange(len(tissues)), normalize="true")
        pd.DataFrame(matrix, index=tissues, columns=tissues).to_csv(
            args.output / f"confusion_{method}.csv"
        )
        np.savez_compressed(
            args.output / f"representations_{method}.npz",
            human=human_X.astype(np.float32),
            mouse=mouse_X.astype(np.float32),
            human_labels=human_labels,
            mouse_labels=mouse_labels,
            human_ids=human_ids,
            mouse_ids=mouse_ids,
        )
    pd.DataFrame(rows).to_csv(args.output / "crossspecies_transfer.csv", index=False)
    paired = paired_bootstrap(
        y_mouse,
        method_outputs["ESPRESSO"],
        method_outputs["Raw_ortholog_PCA64"],
    )
    pd.DataFrame(
        [{"reference": "Raw_ortholog_PCA64", "method": "ESPRESSO", **paired}]
    ).to_csv(args.output / "paired_delta_vs_raw.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        args.output / "mouse_predictions.csv", index=False
    )
    pd.DataFrame({"human_symbol": shared}).to_csv(
        args.output / "shared_one_to_one_orthologs.csv", index=False
    )
    protocol = {
        "human": "GTEx recount2 matrix; maximum 250 samples per tissue, seed 42",
        "mouse": "GSE132040 Tabula Muris Senis bulk; all matching samples",
        "tissues": tissues,
        "orthologs": "MGI HOM_MouseHumanSequence; exactly one human and one mouse gene per class",
        "normalization": "counts to length-adjusted TPM, then log1p; same human gene lengths",
        "probe": "multinomial class-weighted logistic; C tuned by human-only 5-fold CV",
        "mouse_label_use": "evaluation only; no mouse labels used for preprocessing or tuning",
        "bootstrap": {"replicates": 2000, "seed": 42, "stratified_by_mouse_tissue": True},
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    try:
        import umap

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        for row, (method, (human_X, mouse_X)) in enumerate(representations.items()):
            combined = np.vstack([human_X, mouse_X])
            coordinates = umap.UMAP(
                n_neighbors=30, min_dist=0.2, metric="cosine", random_state=42
            ).fit_transform(combined)
            species = np.asarray(["Human"] * len(human_X) + ["Mouse"] * len(mouse_X))
            tissue_labels = np.concatenate([human_labels, mouse_labels])
            for species_name, marker in (("Human", "o"), ("Mouse", "^")):
                selected = species == species_name
                axes[row, 0].scatter(
                    coordinates[selected, 0], coordinates[selected, 1],
                    s=7, alpha=0.55, marker=marker, label=species_name,
                )
            axes[row, 0].set_title(f"{method}: species")
            axes[row, 0].legend(frameon=False)
            for tissue in tissues:
                selected = tissue_labels == tissue
                axes[row, 1].scatter(
                    coordinates[selected, 0], coordinates[selected, 1],
                    s=7, alpha=0.55, label=tissue,
                )
            axes[row, 1].set_title(f"{method}: tissue")
            if row == 0:
                axes[row, 1].legend(
                    frameon=False, fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1)
                )
            pd.DataFrame(
                {
                    "umap1": coordinates[:, 0],
                    "umap2": coordinates[:, 1],
                    "species": species,
                    "tissue": tissue_labels,
                    "method": method,
                }
            ).to_csv(args.output / f"umap_{method}.csv", index=False)
        for axis in axes.flat:
            axis.set_xticks([])
            axis.set_yticks([])
        fig.suptitle("Human GTEx and mouse Tabula Muris Senis co-embedding")
        fig.tight_layout()
        fig.savefig(args.output / "crossspecies_umap.png", dpi=300)
        fig.savefig(args.output / "crossspecies_umap.pdf")
        plt.close(fig)
    except Exception as error:
        (args.output / "umap_error.txt").write_text(str(error) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
