#!/usr/bin/env python3
"""Compare Txn_Jatin with co-expression on held-out immune regulatory/SL edges."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


SEED = 42
EPOCHS = 35
BATCH_SIZE = 8192
LEARNING_RATE = 0.02
WEIGHT_DECAY = 1e-4
BOOTSTRAPS = 2000

IMMUNE_SIGNATURES = {
    "T_CELL_INFLAMED": ("CCL5", "CD27", "CD274", "CD8A", "CMKLR1", "CXCL9", "CXCR6", "HLA-DQA1", "HLA-DRB1", "IDO1", "LAG3", "NKG7", "PDCD1LG2", "PSMB10", "STAT1", "TIGIT"),
    "CYTOLYTIC": ("GZMA", "GZMB", "GZMH", "GNLY", "NKG7", "PRF1"),
    "ANTIGEN_PRESENTATION": ("B2M", "HLA-A", "HLA-B", "HLA-C", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1", "HLA-DRA", "HLA-DRB1", "TAP1", "TAP2"),
    "IFNG_AXIS": ("CXCL9", "CXCL10", "CXCL11", "GBP1", "IDO1", "IFNG", "IRF1", "JAK1", "JAK2", "STAT1"),
    "EXHAUSTION": ("CTLA4", "HAVCR2", "LAG3", "PDCD1", "TIGIT", "TOX"),
    "TREG": ("CCR8", "CTLA4", "FOXP3", "IL2RA", "IKZF2", "TNFRSF18"),
    "MYELOID_SUPPRESSION": ("ARG1", "CD163", "CSF1R", "IL10", "MRC1", "S100A8", "S100A9", "TGFB1"),
    "TGF_BETA": ("COL1A1", "COL1A2", "COL3A1", "SERPINE1", "SMAD3", "TGFB1", "TGFBR1", "TGFBR2"),
    "ANGIOGENESIS": ("ANGPT2", "ESM1", "FLT1", "KDR", "PECAM1", "VWF"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mapping(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    frame = pd.read_csv(path, dtype=str).dropna()
    frame["symbol"] = frame["symbol"].str.upper()
    frame["entrez"] = frame["entrez"].str.replace(r"\.0$", "", regex=True)
    frame = frame.drop_duplicates("symbol")
    symbol_to_entrez = dict(zip(frame["symbol"], frame["entrez"]))
    entrez_to_symbol = dict(zip(frame["entrez"], frame["symbol"]))
    return symbol_to_entrez, entrez_to_symbol


def load_embedding(folder: Path) -> tuple[list[str], np.ndarray]:
    genes = [line.strip() for line in next(folder.glob("*genelist.txt")).read_text().splitlines()]
    matrix = pd.read_csv(next(folder.glob("*_emb.csv")), header=None).to_numpy(np.float32)
    if len(genes) != len(matrix):
        raise ValueError("embedding rows and gene list differ")
    matrix = (matrix - matrix.mean(0)) / np.maximum(matrix.std(0), 1e-6)
    return genes, matrix


def make_coexpression_features(
    path: Path, symbol_to_entrez: dict[str, str]
) -> tuple[dict[str, int], np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        expression = payload["expression"].astype(np.float64)
        symbols = payload["genes"].astype(str)
    keep = [
        i for i, symbol in enumerate(symbols)
        if symbol.upper() in symbol_to_entrez and np.isfinite(expression[:, i]).sum() >= 30
    ]
    expression = expression[:, keep]
    genes = [symbol_to_entrez[symbols[i].upper()] for i in keep]
    ranks = np.empty_like(expression, dtype=np.float64)
    for column in range(expression.shape[1]):
        values = expression[:, column]
        finite = np.isfinite(values)
        replacement = np.nanmedian(values) if finite.any() else 0.0
        values = np.where(finite, values, replacement)
        ranks[:, column] = pd.Series(values).rank(method="average").to_numpy()
    ranks -= ranks.mean(0)
    ranks /= np.maximum(np.sqrt((ranks * ranks).sum(0)), 1e-12)
    return {gene: i for i, gene in enumerate(genes)}, ranks.astype(np.float32)


def pair_features(
    pairs: list[tuple[str, str]],
    embedding: np.ndarray,
    gene_index: dict[str, int],
    task: str,
) -> np.ndarray:
    left = embedding[[gene_index[a] for a, _ in pairs]]
    right = embedding[[gene_index[b] for _, b in pairs]]
    if task == "tf":
        return np.hstack([left, right]).astype(np.float32)
    return np.hstack([left * right, np.abs(left - right)]).astype(np.float32)


def correlation_features(
    pairs: list[tuple[str, str]],
    expression: np.ndarray,
    gene_index: dict[str, int],
) -> np.ndarray:
    left = expression[:, [gene_index[a] for a, _ in pairs]]
    right = expression[:, [gene_index[b] for _, b in pairs]]
    corr = np.sum(left * right, axis=0)
    return np.column_stack([corr, np.abs(corr)]).astype(np.float32)


class LinearEdgeModel(torch.nn.Module):
    def __init__(self, dimensions: int):
        super().__init__()
        self.linear = torch.nn.Linear(dimensions, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


def fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    model = LinearEdgeModel(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    positive_weight = float((train_y == 0).sum() / max((train_y == 1).sum(), 1))
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    order_rng = np.random.default_rng(seed)
    model.train()
    for _ in range(EPOCHS):
        order = order_rng.permutation(len(train_y))
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            x = torch.from_numpy(train_x[idx]).to(device)
            y = torch.from_numpy(train_y[idx].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(test_x), BATCH_SIZE):
            x = torch.from_numpy(test_x[start : start + BATCH_SIZE]).to(device)
            output.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(output)


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame["label"].nunique() < 2:
        return {"auroc": np.nan, "auprc": np.nan, "positives": int(frame["label"].sum()), "edges": len(frame)}
    return {
        "auroc": float(roc_auc_score(frame["label"], frame["score"])),
        "auprc": float(average_precision_score(frame["label"], frame["score"])),
        "positives": int(frame["label"].sum()),
        "edges": int(len(frame)),
    }


def paired_bootstrap(
    predictions: pd.DataFrame, task: str, scope: str, method: str, reference: str
) -> dict[str, float | str]:
    subset = predictions.query("task == @task and scope == @scope")
    wide = subset.pivot_table(
        index=["fold", "pair_index", "label"], columns="method", values="score"
    ).dropna(subset=[method, reference])
    y = wide.index.get_level_values("label").to_numpy()
    rng = np.random.default_rng(SEED)
    auc_delta, pr_delta = [], []
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(BOOTSTRAPS):
        idx = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        labels = y[idx]
        auc_delta.append(
            roc_auc_score(labels, wide[method].to_numpy()[idx])
            - roc_auc_score(labels, wide[reference].to_numpy()[idx])
        )
        pr_delta.append(
            average_precision_score(labels, wide[method].to_numpy()[idx])
            - average_precision_score(labels, wide[reference].to_numpy()[idx])
        )
    return {
        "task": task,
        "scope": scope,
        "method": method,
        "reference": reference,
        "delta_auroc": float(np.mean(auc_delta)),
        "auroc_ci_low": float(np.quantile(auc_delta, 0.025)),
        "auroc_ci_high": float(np.quantile(auc_delta, 0.975)),
        "delta_auprc": float(np.mean(pr_delta)),
        "auprc_ci_low": float(np.quantile(pr_delta, 0.025)),
        "auprc_ci_high": float(np.quantile(pr_delta, 0.975)),
        "bootstrap_replicates": BOOTSTRAPS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--coexpression", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbol_to_entrez, entrez_to_symbol = load_mapping(args.mapping)
    immune_symbols = sorted({gene for genes in IMMUNE_SIGNATURES.values() for gene in genes})
    immune_entrez = {symbol_to_entrez[g] for g in immune_symbols if g in symbol_to_entrez}

    embedding_genes, embedding = load_embedding(args.embedding_dir)
    emb_index = {gene: i for i, gene in enumerate(embedding_genes)}
    corr_index, corr_expression = make_coexpression_features(args.coexpression, symbol_to_entrez)
    eligible = set(emb_index) & set(corr_index)

    prediction_rows = []
    coverage_rows = []
    for task in ("tf", "sl"):
        with (args.split_dir / f"{task}_nested_cv_splits.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        pairs = [tuple(map(str, pair)) for pair in payload["pairs"]]
        labels = np.asarray(payload["labels"], dtype=np.int8)
        kept_master = [i for i, pair in enumerate(pairs) if pair[0] in eligible and pair[1] in eligible]
        kept_set = set(kept_master)
        coverage_rows.append({
            "task": task,
            "source_edges": len(pairs),
            "source_positives": int(labels.sum()),
            "eligible_edges": len(kept_master),
            "eligible_positives": int(labels[kept_master].sum()),
            "immune_edges": int(sum(bool(set(pairs[i]) & immune_entrez) for i in kept_master)),
            "immune_positives": int(sum(labels[i] for i in kept_master if set(pairs[i]) & immune_entrez)),
        })
        for fold, split in payload["cv_splits"].items():
            train_master = [i for i in split["train_idx"] if i in kept_set]
            test_master = [i for i in split["test_idx"] if i in kept_set]
            train_pairs = [pairs[i] for i in train_master]
            test_pairs = [pairs[i] for i in test_master]
            train_y, test_y = labels[train_master], labels[test_master]

            train_txn = pair_features(train_pairs, embedding, emb_index, task)
            test_txn = pair_features(test_pairs, embedding, emb_index, task)
            train_corr = correlation_features(train_pairs, corr_expression, corr_index)
            test_corr = correlation_features(test_pairs, corr_expression, corr_index)
            features = {
                "coexpression": (train_corr, test_corr),
                "Txn_Jatin": (train_txn, test_txn),
                "Txn_plus_coexpression": (
                    np.hstack([train_txn, train_corr]),
                    np.hstack([test_txn, test_corr]),
                ),
            }
            scores = {
                method: fit_predict(x_train, train_y, x_test, device, SEED + int(fold))
                for method, (x_train, x_test) in features.items()
            }
            for local_index, master_index in enumerate(test_master):
                g1, g2 = pairs[master_index]
                immune = bool({g1, g2} & immune_entrez)
                for method, values in scores.items():
                    prediction_rows.append({
                        "task": task,
                        "fold": int(fold),
                        "pair_index": master_index,
                        "gene1_entrez": g1,
                        "gene2_entrez": g2,
                        "gene1_symbol": entrez_to_symbol.get(g1, ""),
                        "gene2_symbol": entrez_to_symbol.get(g2, ""),
                        "label": int(test_y[local_index]),
                        "immune_edge": immune,
                        "scope": "immune" if immune else "nonimmune",
                        "method": method,
                        "score": float(values[local_index]),
                    })

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(args.output / "heldout_edge_predictions.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(args.output / "coverage.csv", index=False)

    summaries = []
    expanded = pd.concat([
        predictions,
        predictions.assign(scope="all"),
    ], ignore_index=True)
    for (task, scope, method), frame in expanded.groupby(["task", "scope", "method"]):
        summaries.append({"task": task, "scope": scope, "method": method, **metrics(frame)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / "summary_metrics.csv", index=False)

    bootstrap_rows = []
    for task in ("tf", "sl"):
        for scope in ("all", "immune"):
            check = expanded.query("task == @task and scope == @scope")
            if check["label"].nunique() < 2:
                continue
            for method in ("Txn_Jatin", "Txn_plus_coexpression"):
                bootstrap_rows.append(
                    paired_bootstrap(expanded, task, scope, method, "coexpression")
                )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(args.output / "paired_bootstrap_vs_coexpression.csv", index=False)

    predictions["percentile_rank"] = predictions.groupby(
        ["task", "fold", "method"]
    )["score"].rank(pct=True)
    wide = predictions.pivot_table(
        index=[
            "task", "fold", "pair_index", "gene1_entrez", "gene2_entrez",
            "gene1_symbol", "gene2_symbol", "label", "immune_edge"
        ],
        columns="method",
        values="score",
    ).reset_index()
    rank_wide = predictions.pivot_table(
        index=["task", "fold", "pair_index"],
        columns="method",
        values="percentile_rank",
    ).reset_index().rename(columns={
        "Txn_Jatin": "Txn_Jatin_percentile",
        "coexpression": "coexpression_percentile",
        "Txn_plus_coexpression": "Txn_plus_coexpression_percentile",
    })
    wide = wide.merge(rank_wide, on=["task", "fold", "pair_index"], validate="one_to_one")
    recovered = wide.query(
        "label == 1 and immune_edge and "
        "Txn_Jatin_percentile >= 0.90 and coexpression_percentile < 0.90"
    ).sort_values(["task", "Txn_Jatin"], ascending=[True, False])
    recovered.to_csv(args.output / "immune_edges_txn_recovers_coexpression_misses.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    palette = {"coexpression": "#718096", "Txn_Jatin": "#2b6cb0", "Txn_plus_coexpression": "#138a8a"}
    for ax, task in zip(axes, ("tf", "sl")):
        scope = "immune" if task == "tf" else "all"
        plot = summary.query("task == @task and scope == @scope")
        x = np.arange(len(plot["method"].unique()))
        methods = ["coexpression", "Txn_Jatin", "Txn_plus_coexpression"]
        values = [
            float(plot.loc[plot["method"] == method, "auroc"].iloc[0])
            if (plot["method"] == method).any() else np.nan
            for method in methods
        ]
        ax.bar(x, values, color=[palette[m] for m in methods])
        ax.set_xticks(x, ["Co-expression", "Txn_Jatin", "Combined"], rotation=15)
        ax.set_ylim(0.40, 0.85)
        ax.set_ylabel("Gene-disjoint held-out AUROC")
        ax.set_title(
            "Immune regulatory associations"
            if task == "tf"
            else "Synthetic-lethal edges (all eligible; immune positives = 1)"
        )
        ax.axhline(0.5, color="#a0aec0", linewidth=1, linestyle="--")
        for i, value in enumerate(values):
            if np.isfinite(value):
                ax.text(i, value + 0.015, f"{value:.3f}", ha="center", fontsize=10)
    fig.savefig(args.output / "immune_edge_recovery.png", dpi=300)
    fig.savefig(args.output / "immune_edge_recovery.pdf")
    plt.close(fig)

    protocol = {
        "seed": SEED,
        "github_commit": "d1320026a2a4ee033d49517f91e2d1c2ccc8df1e",
        "tasks": ["TF regulatory edges", "human synthetic-lethal edges"],
        "outer_splits": (
            "repository-provided five-fold gene-disjoint nested CV; test pairs have "
            "both endpoints in held-out genes and training pairs use only training genes"
        ),
        "immune_scope": "at least one endpoint in the nine predeclared immunotherapy signatures",
        "coexpression": "Spearman correlation over 239 independent pretreatment immunotherapy samples; response labels unused",
        "classifier": "class-weighted linear logistic head; fixed hyperparameters",
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "bootstrap_replicates": BOOTSTRAPS,
        "recovered_edge_definition": (
            "true immune positive in the Txn_Jatin top score decile within its outer "
            "fold but outside the co-expression top score decile"
        ),
        "device": str(device),
        "embedding_sha256": sha256(next(args.embedding_dir.glob("*_emb.csv"))),
        "split_sha256": {
            task: sha256(args.split_dir / f"{task}_nested_cv_splits.pkl")
            for task in ("tf", "sl")
        },
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(bootstrap.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
