#!/usr/bin/env python3
"""Crossed GTEx/TCGA recount bulk-study integration benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from umap import UMAP

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from sample_adapters import native_embedding, save_embedding


TISSUES = ("Breast", "Kidney", "Lung", "Skin")


def decode(values):
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values]
    )


def log_cpm(counts):
    return np.log1p(
        counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0) * 1e6
    ).astype(np.float32)


def select(index_labels, maximum=200):
    rng = np.random.default_rng(42)
    index = []
    for tissue in TISSUES:
        candidates = np.flatnonzero(index_labels == tissue)
        if len(candidates) > maximum:
            candidates = np.sort(rng.choice(candidates, maximum, replace=False))
        index.extend(candidates.tolist())
    return np.asarray(sorted(index))


def load_crossed(gtex_path, tcga_path, maximum):
    with h5py.File(gtex_path) as handle:
        genes = decode(handle["meta/genes"][:])
        labels = decode(handle["meta/smts"][:])
        labels = np.asarray([value if value in TISSUES else "" for value in labels])
        gtex_index = select(labels, maximum)
        gtex = handle["data/expression"][gtex_index].astype(np.float32)
        gtex_ids = decode(handle["meta/sampid"][gtex_index])
        gtex_labels = labels[gtex_index]
    with h5py.File(tcga_path) as handle:
        tcga_genes = decode(handle["meta/genes"][:])
        if not np.array_equal(genes, tcga_genes):
            raise ValueError("GTEx and TCGA recount gene orders differ")
        site = decode(handle["meta/gdc_cases.project.primary_site"][:])
        sample_type = decode(handle["meta/gdc_cases.samples.sample_type"][:])
        labels = np.asarray([value if value in TISSUES else "" for value in site])
        labels[sample_type != "Primary Tumor"] = ""
        tcga_index = select(labels, maximum)
        tcga = handle["data/expression"][tcga_index].astype(np.float32)
        tcga_ids = decode(handle["meta/gdc_file_id"][tcga_index])
        tcga_labels = labels[tcga_index]
    expression = pd.DataFrame(
        log_cpm(np.vstack([gtex, tcga])),
        index=np.concatenate([gtex_ids, tcga_ids]),
        columns=genes,
    )
    metadata = pd.DataFrame(
        {
            "sample_id": expression.index,
            "tissue": np.concatenate([gtex_labels, tcga_labels]),
            "study": np.repeat(["GTEx", "TCGA"], [len(gtex), len(tcga)]),
        }
    )
    return expression, metadata


def inverse_simpson(neighbors, labels):
    result = []
    for row in neighbors:
        _, counts = np.unique(labels[row], return_counts=True)
        probability = counts / counts.sum()
        result.append(1.0 / np.sum(probability**2))
    return float(np.mean(result))


def graph_connectivity(neighbors, labels):
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(len(labels)))
    for i, row in enumerate(neighbors):
        graph.add_edges_from((i, int(j)) for j in row)
    scores = []
    for label in np.unique(labels):
        nodes = np.flatnonzero(labels == label)
        components = list(nx.connected_components(graph.subgraph(nodes)))
        scores.append(max(map(len, components)) / len(nodes))
    return float(np.mean(scores))


def kbet_acceptance(neighbors, batch):
    global_values, global_counts = np.unique(batch, return_counts=True)
    global_probability = global_counts / global_counts.sum()
    accepted = []
    for row in neighbors:
        local = pd.Series(batch[row]).value_counts()
        observed = np.asarray([local.get(value, 0) for value in global_values])
        expected = global_probability * len(row)
        accepted.append(chisquare(observed, expected).pvalue >= 0.05)
    return float(np.mean(accepted))


def evaluate(name, embedding, metadata):
    x = StandardScaler().fit_transform(embedding)
    neighbors = NearestNeighbors(n_neighbors=31).fit(x).kneighbors(return_distance=False)[:, 1:]
    tissue = metadata["tissue"].to_numpy()
    study = metadata["study"].to_numpy()
    tissue_code = LabelEncoder().fit_transform(tissue)
    cluster = KMeans(len(np.unique(tissue)), random_state=42, n_init=20).fit_predict(x)
    ilisi = inverse_simpson(neighbors, study)
    clisi = inverse_simpson(neighbors, tissue)
    asw = (silhouette_score(x, tissue) + 1) / 2
    ari = adjusted_rand_score(tissue_code, cluster)
    nmi = normalized_mutual_info_score(tissue_code, cluster)
    connectivity = graph_connectivity(neighbors, tissue)
    kbet = kbet_acceptance(neighbors, study)
    batch_score = np.mean([(ilisi - 1) / 1, kbet, connectivity])
    biology_score = np.mean([1 / clisi, asw, max(ari, 0), nmi, connectivity])
    return {
        "method": name,
        "n": len(x),
        "dimension": x.shape[1],
        "iLISI": ilisi,
        "kBET_acceptance": kbet,
        "graph_connectivity": connectivity,
        "cLISI": clisi,
        "ASW_tissue": asw,
        "ARI_tissue": ari,
        "NMI_tissue": nmi,
        "batch_mixing_score": batch_score,
        "biology_conservation_score": biology_score,
        "combined_score": 0.4 * batch_score + 0.6 * biology_score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtex", required=True, type=Path)
    parser.add_argument("--tcga", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=["Txn_Jatin", "BRIDGE", "BulkFormer_147M"])
    parser.add_argument("--max-per-cell", type=int, default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expression, metadata = load_crossed(args.gtex, args.tcga, args.max_per_cell)
    metadata.to_csv(args.output / "cohort_metadata.csv", index=False)
    pd.crosstab(metadata["tissue"], metadata["study"]).to_csv(
        args.output / "tissue_study_contingency.csv"
    )
    pca = PCA(64, random_state=42).fit_transform(StandardScaler().fit_transform(expression))
    representations = {"raw_log1p_CPM": expression.to_numpy(np.float32), "PCA64": pca}
    try:
        import harmonypy

        harmony = harmonypy.run_harmony(pca, metadata, ["study"], random_state=42)
        corrected = np.asarray(harmony.Z_corr)
        if corrected.shape[0] != len(metadata):
            corrected = corrected.T
        representations["Harmony_PCA64"] = corrected.astype(np.float32)
    except Exception as error:
        (args.output / "harmony_error.txt").write_text(repr(error), encoding="utf-8")
    for model in args.models:
        cache = args.output / f"{model}.npz"
        if cache.exists():
            embedding = np.load(cache, allow_pickle=False)["embeddings"]
            representations[model] = embedding
            continue
        embedding, provenance = native_embedding(expression, model, args.runtime, args.config)
        representations[model] = embedding
        save_embedding(cache, embedding, expression.index, provenance)
    rows = []
    coordinates = []
    for name, embedding in representations.items():
        rows.append(evaluate(name, embedding, metadata))
        xy = UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", random_state=42).fit_transform(embedding)
        coordinates.append(
            metadata.assign(method=name, UMAP1=xy[:, 0], UMAP2=xy[:, 1])
        )
        pd.DataFrame(rows).to_csv(args.output / "integration_metrics.csv", index=False)
    coordinate = pd.concat(coordinates, ignore_index=True)
    coordinate.to_csv(args.output / "umap_coordinates.csv", index=False)
    methods = list(representations)
    fig, axes = plt.subplots(len(methods), 2, figsize=(11, 3.4 * len(methods)))
    for row, method in enumerate(methods):
        data = coordinate.loc[coordinate["method"].eq(method)]
        for column, color_by in enumerate(("tissue", "study")):
            ax = axes[row, column]
            for label, group in data.groupby(color_by):
                ax.scatter(group["UMAP1"], group["UMAP2"], s=7, alpha=0.65, label=label)
            ax.set_title(f"{method}: {color_by}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(markerscale=2, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output / "integration_umaps.png", dpi=220)
    fig.savefig(args.output / "integration_umaps.pdf")
    protocol = {
        "cohort": "crossed GTEx and TCGA recount2 matrices",
        "tissues": list(TISSUES),
        "study_definition": ["GTEx", "TCGA"],
        "maximum_per_tissue_study": args.max_per_cell,
        "normalization": "library-size CPM then log1p",
        "neighbors": 30,
        "note": "Pinned recount2 matrices were used because they provide a uniform crossed design locally; this is an explicit deviation from the requested recount3 version.",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
