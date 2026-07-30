#!/usr/bin/env python3
"""Evaluate training-cohort-only invariant transforms under outer LOCO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn

from run_provenance_not_response import (
    COHORTS,
    MODELS,
    SEED,
    bootstrap_summary,
    load_inputs,
    paired_deltas,
)


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weight):
        ctx.weight = weight
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.weight * gradient, None


class DANN(nn.Module):
    def __init__(self, inputs, domains):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(inputs, 128), nn.ReLU(), nn.Dropout(.15),
            nn.Linear(128, 32), nn.ReLU(),
        )
        self.response = nn.Linear(32, 1)
        self.domain = nn.Linear(32, domains)

    def forward(self, values, weight=.1):
        embedding = self.encoder(values)
        response = self.response(embedding).squeeze(1)
        domain = self.domain(GradientReverse.apply(embedding, weight))
        return embedding, response, domain


def project_out(train_x, test_x, train_groups):
    classifier = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=5000, multi_class="auto", random_state=SEED
    ).fit(train_x, train_groups)
    _, _, vt = np.linalg.svd(classifier.coef_, full_matrices=False)
    basis = vt[: min(len(np.unique(train_groups)) - 1, len(vt))].T
    return train_x - train_x @ basis @ basis.T, test_x - test_x @ basis @ basis.T


def domain_standardize(train_x, test_x, train_groups):
    """Diagonal CORAL domain generalization fitted without held-out-cohort data."""
    pooled_mean = train_x.mean(axis=0)
    pooled_sd = np.maximum(train_x.std(axis=0), 1e-6)
    transformed = np.zeros_like(train_x)
    for group in np.unique(train_groups):
        mask = train_groups == group
        mean = train_x[mask].mean(axis=0)
        sd = np.maximum(train_x[mask].std(axis=0), 1e-6)
        transformed[mask] = (train_x[mask] - mean) / sd * pooled_sd + pooled_mean
    # The unseen cohort receives only the pooled training transform.
    return transformed, test_x


def dann_embedding(train_x, test_x, y, train_groups, epochs=160):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = LabelEncoder().fit_transform(train_groups)
    model = DANN(train_x.shape[1], len(np.unique(labels))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    response_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(y == 0).sum() / max(1, (y == 1).sum())], device=device)
    )
    domain_loss = nn.CrossEntropyLoss()
    x_tensor = torch.tensor(train_x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    d_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        _, response, domain = model(x_tensor, .1)
        loss = response_loss(response, y_tensor) + domain_loss(domain, d_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_embedding = model.encoder(x_tensor).cpu().numpy()
        test_embedding = model.encoder(
            torch.tensor(test_x, dtype=torch.float32, device=device)
        ).cpu().numpy()
    return train_embedding, test_embedding


def calibrated_probability(train_x, test_x, y_train, groups_train):
    candidates = []
    for c_value in (.01, .1, 1., 10.):
        scores = []
        for group in np.unique(groups_train):
            valid = groups_train == group
            model = LogisticRegression(
                C=c_value, solver="liblinear", class_weight="balanced",
                max_iter=5000, random_state=SEED,
            ).fit(train_x[~valid], y_train[~valid])
            scores.append(roc_auc_score(y_train[valid], model.predict_proba(train_x[valid])[:, 1]))
        candidates.append((np.mean(scores), c_value))
    selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[0][1]
    model = LogisticRegression(
        C=selected, solver="liblinear", class_weight="balanced",
        max_iter=5000, random_state=SEED,
    ).fit(train_x, y_train)
    return model.predict_proba(test_x)[:, 1], selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--overlap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    genes, ids, metadata, raw, masks, completed, _ = load_inputs(args.frozen, args.clinical)
    strict = np.logical_and.reduce([
        np.isfinite(values).all(axis=0) for values in completed.values()
    ])
    common = np.logical_and.reduce(masks) & strict
    representations = {"Raw_common_measured": raw[:, common], **{
        f"{name}__completed": values[:, strict] for name, values in completed.items()
    }}
    y = metadata.label.to_numpy(int)
    groups = metadata.cohort.to_numpy(str)
    overlap = pd.read_csv(args.overlap, sep="\t")
    clean_ids = set(overlap.loc[
        ~overlap.present_in_archs4_pretraining_metadata.astype(bool), "evaluation_sample_id"
    ].astype(str))
    clean = np.asarray([sample_id in clean_ids for sample_id in ids])
    rows = []
    for name, matrix in representations.items():
        for held_out in COHORTS:
            test = groups == held_out
            train = ~test
            selected = np.argsort(np.var(matrix[train], axis=0))[-min(1000, matrix.shape[1]):]
            scaler = StandardScaler().fit(matrix[train][:, selected])
            train_x = scaler.transform(matrix[train][:, selected])
            test_x = scaler.transform(matrix[test][:, selected])
            pca = PCA(n_components=min(64, train.sum() - 1), random_state=SEED).fit(train_x)
            train_x, test_x = pca.transform(train_x), pca.transform(test_x)
            variants = {
                "unadapted": (train_x, test_x),
                "project_out": project_out(train_x, test_x, groups[train]),
                "training_domain_CORAL": domain_standardize(train_x, test_x, groups[train]),
                "DANN": dann_embedding(train_x, test_x, y[train], groups[train]),
            }
            for adaptation, (adapted_train, adapted_test) in variants.items():
                probability, selected_c = calibrated_probability(
                    adapted_train, adapted_test, y[train], groups[train]
                )
                for index, value in zip(np.flatnonzero(test), probability):
                    rows.append({
                        "representation": name, "adaptation": adaptation,
                        "method": f"{name}__{adaptation}", "held_out_cohort": held_out,
                        "sample_index": index, "sample_id": ids[index], "label": int(y[index]),
                        "probability": float(value), "selected_C": selected_c,
                        "pretraining_overlap_clean": bool(clean[index]),
                    })
            print(f"{name}: held out {held_out}", flush=True)
    predictions = pd.DataFrame(rows)
    predictions.to_csv(args.output / "invariance_loco_predictions.csv", index=False)
    summary = bootstrap_summary(predictions, args.bootstraps)
    summary.to_csv(args.output / "invariance_loco_summary.csv", index=False)
    unadapted = predictions[predictions.adaptation == "unadapted"]
    delta_rows = []
    for representation in representations:
        subset = predictions[predictions.representation == representation].copy()
        subset.loc[subset.adaptation == "unadapted", "method"] = "REFERENCE"
        for adaptation in ("project_out", "training_domain_CORAL", "DANN"):
            candidate = subset[
                subset.adaptation.isin(("unadapted", adaptation))
            ].copy()
            candidate.loc[candidate.adaptation == adaptation, "method"] = adaptation
            delta = paired_deltas(
                candidate, reference="REFERENCE", replicates=args.bootstraps
            )
            if len(delta):
                row = delta.iloc[0].to_dict()
                row["representation"] = representation
                delta_rows.append(row)
    pd.DataFrame(delta_rows).to_csv(args.output / "invariance_paired_deltas.csv", index=False)
    (args.output / "invariance_protocol.json").write_text(json.dumps({
        "outer_validation": "leave-one-cohort-out",
        "feature_selection_scaling_PCA": "fitted on outer training cohorts only",
        "project_out": "multinomial cohort-discriminant subspace learned on outer training cohorts",
        "CORAL": "diagonal domain-generalization alignment of each training cohort to pooled training moments; held-out cohort statistics unused",
        "DANN": "fixed 0.1 gradient-reversal weight, 160 epochs; cohort labels and response labels from outer training cohorts only",
        "held_out_cohort_used_for_fit": False,
        "seed": SEED,
    }, indent=2) + "\n")
    print(summary.query("scope == 'full'").sort_values("macro_auroc", ascending=False).head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
