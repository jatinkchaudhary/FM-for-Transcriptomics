#!/usr/bin/env python3
"""Cold-gene DepMap common-essentiality benchmark on frozen gene embeddings."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.provenance import sha256
from shared.statistics import stratified_metric_bootstrap, stratified_paired_bootstrap


MODELS = (
    "Txn_Jatin",
    "Txn_Jatin_contextual",
    "BRIDGE",
    "ESM2_PCA512_prior",
    "scGPT",
    "Geneformer",
    "BulkFormer_37M",
    "BulkFormer_50M",
    "BulkFormer_93M",
    "BulkFormer_127M",
    "BulkFormer_147M",
)
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
ENTREZ_PATTERN = re.compile(r"\((\d+)\)\s*$")


def parse_entrez(value: str) -> int | None:
    match = ENTREZ_PATTERN.search(str(value))
    return int(match.group(1)) if match else None


def load_labels(effect_path: Path, essentials_path: Path, neutral_cutoff: float):
    effect = pd.read_csv(effect_path, index_col=0)
    entrez = np.asarray([parse_entrez(column) for column in effect.columns], dtype=object)
    valid = np.asarray([value is not None for value in entrez])
    effect = effect.iloc[:, valid]
    entrez = entrez[valid].astype(int)
    medians = np.nanmedian(effect.to_numpy(dtype=np.float32), axis=0)
    screened = pd.DataFrame({"entrez": entrez, "median_chronos": medians}).drop_duplicates(
        "entrez"
    )
    essentials = pd.read_csv(essentials_path)
    essential_ids = {
        value
        for value in (parse_entrez(item) for item in essentials.iloc[:, 0])
        if value is not None
    }
    screened["label"] = np.where(
        screened["entrez"].isin(essential_ids),
        1,
        np.where(screened["median_chronos"] >= neutral_cutoff, 0, -1),
    )
    return screened.loc[screened["label"].isin([0, 1])].copy(), {
        "screened_genes": len(screened),
        "common_essential_ids": len(essential_ids),
        "neutral_cutoff": neutral_cutoff,
        "positive_genes": int((screened["label"] == 1).sum()),
        "neutral_genes": int((screened["label"] == 0).sum()),
        "ambiguous_excluded": int((screened["label"] == -1).sum()),
    }


def load_embedding(root: Path, model: str):
    directory = root / model
    genes = np.loadtxt(directory / f"{model}_genelist.txt", dtype=np.int64)
    matrix = np.loadtxt(
        directory / f"{model}_emb.csv", delimiter=",", dtype=np.float32
    )
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if len(genes) != len(matrix):
        raise ValueError(f"{model}: gene and embedding lengths differ")
    finite = np.isfinite(matrix).all(axis=1)
    return genes[finite], matrix[finite], {
        "model": model,
        "genes": int(finite.sum()),
        "dimension": int(matrix.shape[1]),
        "embedding_sha256": sha256(directory / f"{model}_emb.csv"),
        "gene_list_sha256": sha256(directory / f"{model}_genelist.txt"),
    }


def nested_predictions(X, y, folds):
    predictions = np.full(len(y), np.nan, dtype=float)
    selected = []
    for outer_fold, (train, test) in enumerate(folds):
        candidates = []
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=420 + outer_fold)
        for c_value in C_GRID:
            scores = []
            for inner_train, valid in inner.split(X[train], y[train]):
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=42,
                    ),
                )
                model.fit(X[train][inner_train], y[train][inner_train])
                score = model.predict_proba(X[train][valid])[:, 1]
                scores.append(roc_auc_score(y[train][valid], score))
            candidates.append((float(np.mean(scores)), c_value))
        best_c = sorted(candidates, key=lambda item: (-item[0], item[1]))[0][1]
        selected.append(best_c)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=best_c,
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
                random_state=42,
            ),
        )
        model.fit(X[train], y[train])
        predictions[test] = model.predict_proba(X[test])[:, 1]
    if not np.isfinite(predictions).all():
        raise RuntimeError("out-of-fold predictions are incomplete")
    return predictions, selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--gene-effect", type=Path, required=True)
    parser.add_argument("--common-essentials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neutral-cutoff", type=float, default=-0.2)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    labels, label_summary = load_labels(
        args.gene_effect, args.common_essentials, args.neutral_cutoff
    )
    loaded = {}
    provenance = []
    shared = set(labels["entrez"])
    for model in MODELS:
        genes, matrix, item = load_embedding(args.embedding_root, model)
        loaded[model] = (genes, matrix)
        provenance.append(item)
        shared &= set(genes.tolist())
    shared = np.asarray(sorted(shared), dtype=np.int64)
    label_index = labels.set_index("entrez")
    y = label_index.loc[shared, "label"].to_numpy(dtype=int)
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(shared, y)
    )

    prediction_rows = []
    metric_rows = []
    model_scores = {}
    for model in MODELS:
        genes, matrix = loaded[model]
        index = {gene: position for position, gene in enumerate(genes)}
        X = matrix[[index[gene] for gene in shared]]
        scores, selected = nested_predictions(X, y, folds)
        model_scores[model] = scores
        metrics = stratified_metric_bootstrap(
            y, scores, replicates=args.bootstraps, seed=42
        )
        metric_rows.append(
            {
                "model": model,
                "genes": len(shared),
                "positives": int(y.sum()),
                "negatives": int((y == 0).sum()),
                **metrics,
                "selected_C_by_fold": "|".join(map(str, selected)),
            }
        )
        prediction_rows.extend(
            {
                "entrez": int(gene),
                "label": int(label),
                "model": model,
                "probability": float(score),
                "fold": next(
                    fold
                    for fold, (_, test) in enumerate(folds)
                    if position in set(test.tolist())
                ),
            }
            for position, (gene, label, score) in enumerate(zip(shared, y, scores))
        )
        print(
            f"{model}: AUROC={roc_auc_score(y, scores):.4f} "
            f"AUPRC={average_precision_score(y, scores):.4f}",
            flush=True,
        )

    deltas = []
    reference = model_scores["Txn_Jatin"]
    for model in MODELS:
        if model == "Txn_Jatin":
            continue
        deltas.append(
            {
                "reference": "Txn_Jatin",
                "comparator": model,
                **stratified_paired_bootstrap(
                    y,
                    reference,
                    model_scores[model],
                    replicates=args.bootstraps,
                    seed=42,
                ),
            }
        )

    pd.DataFrame(metric_rows).to_csv(args.output / "essentiality_metrics.csv", index=False)
    pd.DataFrame(deltas).to_csv(
        args.output / "paired_deltas_vs_espresso.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        args.output / "out_of_fold_predictions.csv", index=False
    )
    pd.DataFrame(provenance).to_csv(
        args.output / "representation_provenance.csv", index=False
    )
    protocol = {
        "task": "DepMap 24Q2 common-essential versus pan-neutral gene classification",
        "release_doi": "10.25452/figshare.plus.25880521.v1",
        "gene_effect_sha256": sha256(args.gene_effect),
        "common_essentials_sha256": sha256(args.common_essentials),
        "label_definition": (
            "positive: CRISPRInferredCommonEssentials; negative: screened gene with "
            f"median Chronos effect >= {args.neutral_cutoff}; intermediate genes excluded"
        ),
        "coverage": "strict intersection across every evaluated model",
        "outer_cv": "5-fold stratified gene-disjoint, shuffle seed 42",
        "inner_cv": "4-fold stratified; fixed C grid",
        "C_grid": C_GRID,
        "bootstrap": {"replicates": args.bootstraps, "seed": 42, "paired": True},
        "label_summary": label_summary,
        "strict_intersection_genes": len(shared),
        "strict_intersection_positives": int(y.sum()),
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
