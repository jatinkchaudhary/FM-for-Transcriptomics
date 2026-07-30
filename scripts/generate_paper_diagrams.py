#!/usr/bin/env python3
"""Generate publication-ready architecture and benchmark diagrams."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "ink": "#17212B",
    "muted": "#5E6B75",
    "line": "#A9B4BC",
    "data": "#0F7C82",
    "data_light": "#DDF2F1",
    "model": "#3569A8",
    "model_light": "#E4EDF8",
    "prior": "#A43A72",
    "prior_light": "#F5E3EC",
    "eval": "#C77A15",
    "eval_light": "#F9EBD5",
    "output": "#438B57",
    "output_light": "#E3F1E6",
    "neutral": "#F3F5F6",
    "white": "#FFFFFF",
}


def setup(figsize=(14, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, xy, width, height, title, body="", kind="neutral", fontsize=10, title_size=11):
    x, y = xy
    face = COLORS.get(f"{kind}_light", COLORS["neutral"])
    edge = COLORS.get(kind, COLORS["line"])
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=face, edgecolor=edge, linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2, y + height * 0.70, title,
        ha="center", va="center", fontsize=title_size, color=COLORS["ink"],
        fontweight="bold",
    )
    if body:
        ax.text(
            x + width / 2, y + height * 0.34, body,
            ha="center", va="center", fontsize=fontsize, color=COLORS["muted"],
            linespacing=1.25,
        )
    return patch


def arrow(ax, start, end, color=None, label=None, rad=0.0, fontsize=9):
    color = color or COLORS["line"]
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.5,
        color=color, connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.018, label, ha="center", va="bottom", fontsize=fontsize, color=color)


def title(ax, panel, heading, subtitle):
    ax.text(0.02, 0.955, panel, fontsize=16, fontweight="bold", color=COLORS["model"])
    ax.text(0.065, 0.955, heading, fontsize=18, fontweight="bold", color=COLORS["ink"])
    ax.text(0.065, 0.918, subtitle, fontsize=10.5, color=COLORS["muted"])


def footer(ax, text):
    ax.plot([0.02, 0.98], [0.055, 0.055], color=COLORS["line"], linewidth=0.8)
    ax.text(0.02, 0.025, text, fontsize=8.5, color=COLORS["muted"], va="center")


def save(fig, output: Path, stem: str):
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            kwargs["dpi"] = 300
        fig.savefig(output / f"{stem}.{extension}", **kwargs)
    plt.close(fig)


def bridge(output):
    fig, ax = setup()
    title(
        ax, "A", "BRIDGE: reconstruction-focused expression Transformer",
        "Continuous whole-transcriptome modeling with masked expression reconstruction",
    )
    box(ax, (0.03, 0.63), 0.14, 0.16, "Expression profile", "log1p-TPM\n15,165 genes", "data")
    box(ax, (0.21, 0.63), 0.14, 0.16, "Random masking", "15% genes\nmask token = -10", "eval")
    box(ax, (0.39, 0.72), 0.15, 0.13, "Gene identity", "learned E_g\n512 dimensions", "model")
    box(ax, (0.39, 0.54), 0.15, 0.13, "Expression encoding", "REE(x_g)\nsinusoidal rotation", "data")
    box(ax, (0.59, 0.60), 0.18, 0.20, "Expression Transformer", "12 blocks\n8 attention heads\nFFN 512 -> 2,048 -> 512", "model")
    box(ax, (0.82, 0.63), 0.14, 0.16, "Gene decoder", "linear projection\npredicted x_g", "output")
    arrow(ax, (0.17, 0.71), (0.21, 0.71), COLORS["data"])
    arrow(ax, (0.35, 0.71), (0.39, 0.78), COLORS["line"])
    arrow(ax, (0.35, 0.69), (0.39, 0.60), COLORS["line"])
    arrow(ax, (0.54, 0.78), (0.59, 0.72), COLORS["model"])
    arrow(ax, (0.54, 0.60), (0.59, 0.67), COLORS["data"])
    arrow(ax, (0.77, 0.70), (0.82, 0.70), COLORS["model"])

    ax.text(0.565, 0.865, r"$h_g^{(0)} = E_g + \mathrm{REE}(x_g)$", ha="center", fontsize=11, color=COLORS["ink"])
    box(ax, (0.23, 0.25), 0.25, 0.18, "Masked reconstruction loss", "MSE only on hidden genes\n" + r"$L_{\mathrm{rec}} = |M|^{-1}\sum_{g\in M}(\hat{x}_g-x_g)^2$", "eval")
    box(ax, (0.57, 0.25), 0.22, 0.18, "Optimization target", r"$L_{\mathrm{BRIDGE}} = L_{\mathrm{rec}}$" + "\nNo sequence or contrastive term", "output")
    arrow(ax, (0.89, 0.63), (0.47, 0.41), COLORS["output"], rad=-0.18, label="masked predictions")
    arrow(ax, (0.48, 0.34), (0.57, 0.34), COLORS["eval"])

    ax.text(0.03, 0.15, "What BRIDGE learns", fontsize=11, fontweight="bold", color=COLORS["ink"])
    ax.text(
        0.03, 0.105,
        "A gene representation optimized to reconstruct missing expression from the remaining transcriptome.",
        fontsize=10, color=COLORS["muted"],
    )
    footer(ax, "Shared backbone with Txn_Jatin; the principal distinction is the training objective and data preparation.")
    save(fig, output, "figure_01_bridge_architecture")


def txn(output):
    fig, ax = setup(figsize=(15, 9))
    title(
        ax, "B", "Txn_Jatin: biologically regularized expression Transformer",
        "BRIDGE backbone plus protein-sequence, contextual, and sample-structure objectives",
    )
    box(ax, (0.025, 0.64), 0.13, 0.15, "ARCHS4 profile", "counts -> log1p-TPM\n16,055 genes", "data")
    box(ax, (0.185, 0.64), 0.12, 0.15, "Mask 15%", "mask token -10\nwhole profile", "eval")
    box(ax, (0.34, 0.73), 0.14, 0.12, "Gene identity", "E_g initialized\nfrom ESM2 prior", "prior")
    box(ax, (0.34, 0.55), 0.14, 0.12, "Expression REE", "REE(x_g)\n512 dimensions", "data")
    box(ax, (0.52, 0.61), 0.17, 0.20, "Shared backbone", "12 Flash/SDPA blocks\n8 heads; hidden 512\nFFN 2,048", "model")
    box(ax, (0.73, 0.64), 0.12, 0.15, "Decoder", "per-gene\nexpression", "output")
    box(ax, (0.87, 0.64), 0.105, 0.15, "Context bank", "epoch-wise\nmean h_g", "prior")
    arrow(ax, (0.155, 0.715), (0.185, 0.715), COLORS["data"])
    arrow(ax, (0.305, 0.715), (0.34, 0.79), COLORS["line"])
    arrow(ax, (0.305, 0.69), (0.34, 0.61), COLORS["line"])
    arrow(ax, (0.48, 0.79), (0.52, 0.73), COLORS["prior"])
    arrow(ax, (0.48, 0.61), (0.52, 0.68), COLORS["data"])
    arrow(ax, (0.69, 0.71), (0.73, 0.71), COLORS["model"])
    arrow(ax, (0.69, 0.74), (0.87, 0.72), COLORS["prior"], rad=-0.06)
    ax.text(0.50, 0.86, r"$h_g^{(0)} = E_g + \mathrm{REE}(x_g)$", ha="center", fontsize=11, color=COLORS["ink"])

    losses = [
        (0.035, "Reconstruction", r"$L_{\mathrm{rec}}$" + "\nmasked-gene MSE", "eval"),
        (0.235, "Sequence prior", r"$0.02\,L_{\mathrm{gene}}$" + "\n" + r"$E_g \leftrightarrow \mathrm{ESM2}_g$", "prior"),
        (0.435, "Context alignment", r"$0.03\,L_{\mathrm{context}}$" + "\n" + r"$\bar{h}_g \leftrightarrow \mathrm{ESM2}_g$", "prior"),
        (0.635, "Sample contrastive", r"$0.01\,L_{\mathrm{sample}}$" + "\nprofile structure; T=0.1", "prior"),
    ]
    for x, name, body, kind in losses:
        box(ax, (x, 0.30), 0.165, 0.17, name, body, kind)
    box(ax, (0.835, 0.30), 0.14, 0.17, "Total objective", r"$L=L_{\mathrm{rec}}+0.02L_{\mathrm{gene}}$" + "\n" + r"$+0.03L_{\mathrm{context}}$" + "\n" + r"$+0.01L_{\mathrm{sample}}$", "output", fontsize=8.8)
    arrow(ax, (0.79, 0.64), (0.117, 0.47), COLORS["eval"], rad=0.25)
    arrow(ax, (0.41, 0.73), (0.317, 0.47), COLORS["prior"], rad=0.05)
    arrow(ax, (0.92, 0.64), (0.517, 0.47), COLORS["prior"], rad=0.10)
    arrow(ax, (0.60, 0.61), (0.717, 0.47), COLORS["prior"], rad=-0.05)
    for x in (0.2175, 0.4175, 0.6175):
        ax.text(x, 0.385, "+", ha="center", va="center", fontsize=16, color=COLORS["muted"], fontweight="bold")
    arrow(ax, (0.80, 0.385), (0.835, 0.385), COLORS["line"])

    ax.text(0.03, 0.20, "Architectural claim", fontsize=11, fontweight="bold", color=COLORS["ink"])
    ax.text(
        0.03, 0.155,
        "The backbone is BRIDGE-like. Novelty arises from prior initialization, multi-objective training, "
        "context accumulation, and harmonized human/mouse ARCHS4 coverage.",
        fontsize=10, color=COLORS["muted"],
    )
    footer(ax, "Final run: 46.05M parameters; 958,212 train + 20,000 validation profiles; 20 epochs; bf16 on H100.")
    save(fig, output, "figure_02_txn_jatin_architecture")


def contextual(output):
    fig, ax = setup(figsize=(15, 8.5))
    title(
        ax, "C", "Txn_Contextual: corpus-level contextual gene representation",
        "A derived frozen embedding matrix, not a separately optimized model",
    )
    box(ax, (0.03, 0.63), 0.15, 0.18, "Reference profiles", "20,000 ARCHS4\nvalidation samples\nlog1p-TPM", "data")
    box(ax, (0.225, 0.63), 0.16, 0.18, "Frozen Txn_Jatin", "gene identity + REE\n12 Transformer blocks\nno weight updates", "model")
    box(ax, (0.43, 0.63), 0.16, 0.18, "Context states", "H_s,g in R^512\none vector per\ngene and sample", "prior")
    box(ax, (0.635, 0.63), 0.16, 0.18, "Corpus aggregation", "Z_g = (1/N) sum_s H_s,g\nstreaming accumulation", "eval")
    box(ax, (0.84, 0.63), 0.13, 0.18, "Frozen matrix", "15,916 genes\nx 512 dimensions", "output")
    for start, end, color in (
        ((0.18, 0.72), (0.225, 0.72), COLORS["data"]),
        ((0.385, 0.72), (0.43, 0.72), COLORS["model"]),
        ((0.59, 0.72), (0.635, 0.72), COLORS["prior"]),
        ((0.795, 0.72), (0.84, 0.72), COLORS["eval"]),
    ):
        arrow(ax, start, end, color)

    box(ax, (0.08, 0.29), 0.20, 0.18, "Static Txn_Jatin", "one learned E_g per gene\nindependent of profile", "model")
    box(ax, (0.40, 0.29), 0.20, 0.18, "Contextual Txn", "average hidden state Z_g\ncaptures typical expression context", "prior")
    box(ax, (0.72, 0.29), 0.20, 0.18, "Sample-specific state", "H_s,g for one profile\nretains individual context", "data")
    arrow(ax, (0.28, 0.38), (0.40, 0.38), COLORS["line"])
    arrow(ax, (0.60, 0.38), (0.72, 0.38), COLORS["line"])

    ax.text(0.05, 0.19, "Representation boundary", fontsize=11, fontweight="bold", color=COLORS["ink"])
    ax.text(
        0.05, 0.135,
        "Txn_Contextual supports gene-level and gene-pair benchmarks. It is not an expression decoder and "
        "must not be assigned synthetic imputation scores.",
        fontsize=10, color=COLORS["muted"],
    )
    footer(ax, "For full-corpus dynamic experiments, the same construction was also run over 978,212 ARCHS4 profiles.")
    save(fig, output, "figure_03_txn_contextual_construction")


def experiment_pipeline(output):
    fig, ax = setup(figsize=(16, 10))
    title(
        ax, "D", "End-to-end experimental pipeline",
        "From harmonized pretraining to external reconstruction, biological probing, and clinical transfer",
    )
    stages = [
        ("1. Data and pretraining", "data", [
            ("ARCHS4", "978,212 profiles\nhuman + mouse"),
            ("Harmonization", "16,055 genes\ncounts -> log1p-TPM"),
            ("Txn training", "20 epochs\n4-objective loss"),
        ]),
        ("2. Representation audit", "model", [
            ("Epoch-7 comparison", "Txn vs BRIDGE"),
            ("Pinned GitHub", "gene + gene-pair"),
            ("Dynamic context", "static vs corpus mean"),
            ("Individual GO", "OSDR + TCGA"),
        ]),
        ("3. Reconstruction and adaptation", "eval", [
            ("Whole-gene mask", "TCGA + OSDR\n3 seeds"),
            ("OSDR PEFT", "unfreezing + LoRA"),
            ("Sparse panels", "1k / 2k genes"),
            ("External GTEx", "12 tissues\n15% gene mask"),
        ]),
        ("4. Downstream translation", "output", [
            ("Cancer RF", "2- and 5-cancer"),
            ("Immunotherapy", "4-cohort LOCO"),
            ("Gene atlas", "tissue/species/pathway"),
            ("LLM head", "evidence summary only"),
        ]),
    ]
    y_positions = [0.73, 0.54, 0.35, 0.16]
    for (stage, kind, items), y in zip(stages, y_positions):
        ax.text(0.025, y + 0.08, stage, fontsize=11.5, fontweight="bold", color=COLORS[kind])
        n = len(items)
        left, right, gap = 0.25, 0.97, 0.018
        width = (right - left - gap * (n - 1)) / n
        for index, (name, body) in enumerate(items):
            x = left + index * (width + gap)
            box(ax, (x, y), width, 0.13, name, body, kind, fontsize=8.7, title_size=10)
            if index < n - 1:
                arrow(ax, (x + width, y + 0.065), (x + width + gap, y + 0.065), COLORS["line"])
        if y != y_positions[-1]:
            arrow(ax, (0.58, y), (0.58, y - 0.055), COLORS["line"])

    ax.text(0.025, 0.085, "Validation principle", fontsize=10.5, fontweight="bold", color=COLORS["ink"])
    ax.text(
        0.18, 0.085,
        "fixed intersections | grouped or cohort-held-out splits | unsupported outputs = NaN | measured results only",
        fontsize=9.5, color=COLORS["muted"],
    )
    footer(ax, "Chronology follows the repository experiment order; exploratory and confirmatory analyses remain explicitly separated.")
    save(fig, output, "figure_04_experimental_pipeline")


def benchmark_map(output):
    fig, ax = setup(figsize=(16, 10))
    title(
        ax, "E", "Benchmark landscape and evidence hierarchy",
        "Tasks, eligible model outputs, validation units, and primary metrics",
    )
    columns = [
        ("Representation", 0.02, "model", [
            ("Alignment", "CKA, Procrustes,\nneighbor overlap"),
            ("Gene-level", "GO, OMIM, disease,\nHallmark/KEGG"),
            ("Gene-pair", "SL, POMBE, TF, NG\noperators + SVC"),
        ]),
        ("Reconstruction", 0.265, "data", [
            ("Cell masking", "random 15% genes\nMSE, PCC"),
            ("Whole-gene mask", "TCGA + OSDR\nAUROC, MSE, PCC"),
            ("External GTEx", "strict 2,117 genes\nMSE, MAE, PCC"),
        ]),
        ("Sample prediction", 0.51, "eval", [
            ("Cancer RF", "2 / 5 cancers\npatient-grouped CV"),
            ("OSDR", "flight vs ground\naccession grouping"),
            ("Immunotherapy", "4-cohort nested LOCO\nAUROC, AUPRC, Brier"),
        ]),
        ("Biological interpretation", 0.755, "output", [
            ("Sparse biomarker", "pathways + phenotype\n1k / 2k panels"),
            ("Atlas", "species, tissue,\ndisease evidence"),
            ("Pathway effects", "random effects,\nFDR, concordance"),
        ]),
    ]
    width = 0.225
    for heading, x, kind, items in columns:
        ax.text(x + width / 2, 0.84, heading, ha="center", fontsize=12, fontweight="bold", color=COLORS[kind])
        for index, (name, body) in enumerate(items):
            y = 0.64 - index * 0.19
            box(ax, (x, y), width, 0.145, name, body, kind, fontsize=8.8, title_size=10.2)
            if index < len(items) - 1:
                arrow(ax, (x + width / 2, y), (x + width / 2, y - 0.04), COLORS["line"])

    ax.text(0.04, 0.13, "Eligible outputs", fontsize=10.5, fontweight="bold", color=COLORS["ink"])
    legend = [
        ("Decoder models", "Txn_Jatin / OSDR-LoRA / BRIDGE\nBulkFormer family", "data"),
        ("Gene embeddings", "Txn static/contextual / BRIDGE / ESM2/3\nGeneformer / scGPT / BulkFormer", "model"),
        ("Language layer", "Structured evidence summarization\nnever used for prediction", "prior"),
    ]
    x = 0.16
    for heading, body, kind in legend:
        box(ax, (x, 0.075), 0.25, 0.105, heading, body, kind, fontsize=7.8, title_size=9.3)
        x += 0.27
    footer(ax, "Primary comparisons use identical genes/pairs/samples whenever possible; capability mismatches are reported, not imputed.")
    save(fig, output, "figure_05_benchmark_landscape")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bridge(args.output)
    txn(args.output)
    contextual(args.output)
    experiment_pipeline(args.output)
    benchmark_map(args.output)
    print(f"Wrote 15 figure files to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
