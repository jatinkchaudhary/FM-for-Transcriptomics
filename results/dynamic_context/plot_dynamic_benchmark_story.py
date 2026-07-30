from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
STATIC_TABLES = ROOT.parent / "benchmark" / "tables"
FIGURES.mkdir(exist_ok=True)

TXN = "#E64B5D"
BRIDGE = "#2864A5"
OTHER = "#C9CED6"
WINNER = "#F2B134"
POSITIVE = "#178F72"
NEGATIVE = "#D95B59"
RAW = "#202A44"
PCA = "#8B95A5"
GRID = "#DFE3E8"

TRACK_LABELS = {
    "gene-set": "Gene-set matching",
    "paired": "Paired genes",
    "single-gene-GO": "Single-gene GO",
    "single-gene-disease": "Single-gene disease",
}
TASK_LABELS = {
    ("TCGA-cancer-type", "stratified"): "TCGA\nstratified",
    ("TCGA-cancer-type", "group"): "TCGA\ngroup",
    ("OSDR-spaceflight", "stratified"): "OSDR\nstratified",
    ("OSDR-spaceflight", "group"): "OSDR\ngroup",
}


def style_axis(ax, axis="x"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


dynamic = pd.read_csv(TABLES / "leaderboard_gene_tracks.csv")
static = pd.read_csv(STATIC_TABLES / "leaderboard_gene_tracks.csv")
sample = pd.read_csv(TABLES / "sample_scores.csv")
cca = pd.read_csv(TABLES / "intermodel_cca.csv", index_col=0)
models = [c for c in dynamic.columns if c not in {"track", "variant"}]
full = dynamic[dynamic["variant"] == "full"].copy()
long_full = full.melt(
    id_vars=["track", "variant"],
    value_vars=models,
    var_name="model",
    value_name="score",
)


def plot_landscape(ax):
    tracks = list(TRACK_LABELS)
    ys = np.arange(len(tracks))[::-1]
    for y, track in zip(ys, tracks):
        subset = long_full[long_full["track"] == track]
        winner = subset.loc[subset["score"].idxmax()]
        others = subset[~subset["model"].isin(["Txn_Jatin", "BRIDGE"])]
        ax.scatter(
            others["score"], np.full(len(others), y), s=42, color=OTHER,
            edgecolor="white", linewidth=0.5, zorder=2
        )
        for model, color, size in [
            ("BRIDGE", BRIDGE, 82), ("Txn_Jatin", TXN, 105)
        ]:
            score = float(subset.loc[subset["model"] == model, "score"].iloc[0])
            ax.scatter(
                score, y, s=size, color=color, edgecolor="white",
                linewidth=0.8, zorder=4
            )
            rank = 1 + int((subset["score"] > score).sum())
            if model == "Txn_Jatin":
                ax.text(
                    0.535, y - 0.22, f"Txn rank {rank}/9",
                    color=TXN, fontsize=8, fontweight="bold"
                )
        ax.scatter(
            winner["score"], y, s=175, facecolors="none",
            edgecolors=WINNER, linewidth=2, zorder=5
        )
        ax.text(
            winner["score"] + 0.0015, y + 0.16,
            f'{winner["model"]}  {winner["score"]:.3f}',
            fontsize=8, color="#5B616B"
        )
    ax.set_yticks(ys, [TRACK_LABELS[t] for t in tracks])
    xmin = max(0.50, float(long_full["score"].min()) - 0.02)
    xmax = min(1.0, float(long_full["score"].max()) + 0.025)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Mean AUROC — full gene set")
    ax.set_title("1. Full-corpus dynamic embeddings versus all models",
                 loc="left", weight="bold")
    style_axis(ax)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=TXN, markersize=9, label="Txn_Jatin dynamic"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=BRIDGE, markersize=8, label="BRIDGE dynamic"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=OTHER, markersize=7, label="Static models"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="none", markeredgecolor=WINNER,
                   markeredgewidth=2, markersize=10, label="Track winner"),
        ],
        frameon=False, ncol=2, fontsize=8, loc="lower right"
    )


def plot_dynamic_gain(ax):
    rows = []
    for model in ("Txn_Jatin", "BRIDGE"):
        for track, label in TRACK_LABELS.items():
            dyn = float(
                dynamic[
                    (dynamic["track"] == track) & (dynamic["variant"] == "full")
                ][model].iloc[0]
            )
            sta = float(
                static[
                    (static["track"] == track) & (static["variant"] == "full")
                ][model].iloc[0]
            )
            rows.append(
                {"model": model, "track": label, "delta": 100 * (dyn - sta)}
            )
    frame = pd.DataFrame(rows)
    y = np.arange(len(TRACK_LABELS))
    width = 0.34
    for i, (model, color) in enumerate([("Txn_Jatin", TXN), ("BRIDGE", BRIDGE)]):
        part = frame[frame["model"] == model].set_index("track")
        values = [part.loc[label, "delta"] for label in TRACK_LABELS.values()]
        bars = ax.barh(y + (i - 0.5) * width, values, width, color=color,
                       label=model, zorder=3)
        for bar, value in zip(bars, values):
            ha = "left" if value >= 0 else "right"
            ax.text(
                value + (0.06 if value >= 0 else -0.06),
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.2f}", va="center", ha=ha, fontsize=7.5,
                color=color, fontweight="bold"
            )
    ax.axvline(0, color="#646B75", linewidth=1)
    ax.set_yticks(y, list(TRACK_LABELS.values()))
    ax.set_xlabel("Dynamic − static AUROC (percentage points)")
    ax.set_title("2. What dynamics changed for Txn_Jatin and BRIDGE",
                 loc="left", weight="bold")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        0.02,
        0.02,
        "Paired-track delta includes small negative-pair rerun variation "
        "(unchanged controls ≤0.43 points).",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#646B75",
    )


def plot_head_to_head(ax):
    rows = []
    for _, row in full.iterrows():
        rows.append(
            {
                "metric": TRACK_LABELS[row["track"]],
                "delta": 100 * (row["Txn_Jatin"] - row["BRIDGE"]),
            }
        )
    logistic = sample[
        (sample["probe"] == "Logistic")
        & sample["feature"].isin(["Txn_Jatin-emb", "BRIDGE-emb"])
    ]
    for (dataset, scheme), label in TASK_LABELS.items():
        part = logistic[
            (logistic["dataset"] == dataset) & (logistic["scheme"] == scheme)
        ]
        txn = float(part.loc[part["feature"] == "Txn_Jatin-emb", "f1_mean"].iloc[0])
        bridge = float(part.loc[part["feature"] == "BRIDGE-emb", "f1_mean"].iloc[0])
        rows.append({"metric": label.replace("\n", " "),
                     "delta": 100 * (txn - bridge)})
    frame = pd.DataFrame(rows).iloc[::-1]
    colors = [POSITIVE if value >= 0 else NEGATIVE for value in frame["delta"]]
    bars = ax.barh(frame["metric"], frame["delta"], color=colors, height=0.64)
    ax.axvline(0, color="#646B75", linewidth=1)
    for bar, value in zip(bars, frame["delta"]):
        ax.text(
            value + (0.07 if value >= 0 else -0.07),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}", va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8, fontweight="bold",
            color=POSITIVE if value >= 0 else NEGATIVE
        )
    ax.set_xlabel("Txn_Jatin − BRIDGE (percentage points)")
    ax.set_title("3. Dynamic Txn_Jatin versus dynamic BRIDGE",
                 loc="left", weight="bold")
    style_axis(ax)


def plot_similarity(ax):
    sims = cca.loc["Txn_Jatin"].drop("Txn_Jatin").sort_values()
    colors = [BRIDGE if model == "BRIDGE" else OTHER for model in sims.index]
    bars = ax.barh(sims.index, sims.values, color=colors, height=0.65)
    for bar, model, value in zip(bars, sims.index, sims.values):
        ax.text(
            value + 0.006, bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}", va="center", fontsize=8,
            color=BRIDGE if model == "BRIDGE" else "#505762",
            fontweight="bold" if model == "BRIDGE" else "normal"
        )
    ax.set_xlim(max(0, float(sims.min()) - 0.04),
                min(1, float(sims.max()) + 0.04))
    ax.set_xlabel("CCA similarity to dynamic Txn_Jatin")
    ax.set_title("4. Geometry after full-corpus contextualization",
                 loc="left", weight="bold")
    style_axis(ax)


def save_one(plotter, name, size):
    fig, ax = plt.subplots(figsize=size)
    plotter(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=240, bbox_inches="tight",
                facecolor="white")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    }
)

save_one(plot_landscape, "dynamic_story_01_landscape", (9.2, 4.5))
save_one(plot_dynamic_gain, "dynamic_story_02_gain_vs_static", (8.8, 4.8))
save_one(plot_head_to_head, "dynamic_story_03_txn_vs_bridge", (8.8, 4.8))
save_one(plot_similarity, "dynamic_story_04_similarity", (8.8, 4.8))

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
plot_landscape(axes[0, 0])
plot_dynamic_gain(axes[0, 1])
plot_head_to_head(axes[1, 0])
plot_similarity(axes[1, 1])
fig.suptitle(
    "Full-corpus dynamic embeddings: Txn_Jatin and BRIDGE versus static foundation models",
    fontsize=17, fontweight="bold", x=0.055, ha="left", y=0.995
)
fig.text(
    0.055, 0.962,
    "978,212 ARCHS4 context samples per dynamic model • nine-model benchmark",
    fontsize=10, color="#646B75"
)
fig.subplots_adjust(
    left=0.10, right=0.98, bottom=0.08, top=0.91,
    hspace=0.38, wspace=0.31
)
fig.savefig(FIGURES / "dynamic_story_dashboard.png", dpi=240,
            bbox_inches="tight", facecolor="white")
fig.savefig(FIGURES / "dynamic_story_dashboard.pdf",
            bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Wrote dynamic benchmark story figures to {FIGURES}")
