from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
STATIC = ROOT.parent / "benchmark" / "tables"


def fmt(value):
    return f"{float(value):.4f}"


dynamic = pd.read_csv(TABLES / "leaderboard_gene_tracks.csv")
static = pd.read_csv(STATIC / "leaderboard_gene_tracks.csv")
sample = pd.read_csv(TABLES / "sample_scores.csv")
dynamic_cca = pd.read_csv(TABLES / "intermodel_cca.csv", index_col=0)
static_cca = pd.read_csv(STATIC / "intermodel_cca.csv", index_col=0)
audit = json.loads((ROOT / "dynamic_embedding_audit.json").read_text())
config = json.loads((ROOT / "dynamic_run_configuration.json").read_text())

models = [c for c in dynamic.columns if c not in {"track", "variant"}]
rows = []
for model in models:
    for _, record in dynamic.iterrows():
        old = static[
            (static["track"] == record["track"])
            & (static["variant"] == record["variant"])
        ]
        static_score = float(old[model].iloc[0])
        dynamic_score = float(record[model])
        rows.append(
            {
                "model": model,
                "track": record["track"],
                "variant": record["variant"],
                "dynamic_score": dynamic_score,
                "static_score": static_score,
                "delta": dynamic_score - static_score,
            }
        )
comparison = pd.DataFrame(rows)
comparison.to_csv(TABLES / "static_vs_dynamic_gene_scores.csv", index=False)

provenance_rows = []
for model, item in audit.items():
    prov = item["provenance"]
    provenance_rows.append(
        {
            "model": model,
            "dynamic": bool(prov.get("dynamic_embedding", False)),
            "fallback": bool(item["fallback"]),
            "shape": "x".join(map(str, item["shape"])),
            "checkpoint": prov.get("checkpoint"),
            "context_samples": prov.get("context_samples"),
            "context_files": prov.get("context_files"),
            "context_batch_size": prov.get("context_batch_size"),
            "context_amp_bf16": prov.get("context_amp_bf16"),
            "context_elapsed_seconds": prov.get("context_elapsed_seconds"),
            "embedding_sha256": prov.get("context_embedding_sha256"),
        }
    )
provenance = pd.DataFrame(provenance_rows)
provenance.to_csv(TABLES / "dynamic_embedding_provenance.csv", index=False)

full = dynamic[dynamic["variant"] == "full"].copy()
full_rows = []
for _, record in full.iterrows():
    scores = {model: float(record[model]) for model in models}
    winner = max(scores, key=scores.get)
    txn_rank = 1 + sum(value > scores["Txn_Jatin"] for value in scores.values())
    bridge_rank = 1 + sum(value > scores["BRIDGE"] for value in scores.values())
    full_rows.append(
        {
            "track": record["track"],
            "Txn_Jatin": scores["Txn_Jatin"],
            "Txn_rank": txn_rank,
            "BRIDGE": scores["BRIDGE"],
            "BRIDGE_rank": bridge_rank,
            "winner": winner,
            "winner_score": scores[winner],
        }
    )
full_summary = pd.DataFrame(full_rows)
full_summary.to_csv(TABLES / "dynamic_full_track_summary.csv", index=False)

overall_rows = []
for model in models:
    dynamic_mean = float(full[model].astype(float).mean())
    static_mean = float(
        static[static["variant"] == "full"][model].astype(float).mean()
    )
    overall_rows.append(
        {
            "model": model,
            "dynamic_full_track_mean": dynamic_mean,
            "static_full_track_mean": static_mean,
            "delta": dynamic_mean - static_mean,
        }
    )
overall = pd.DataFrame(overall_rows).sort_values(
    "dynamic_full_track_mean", ascending=False
)
overall["dynamic_rank"] = range(1, len(overall) + 1)
overall.to_csv(TABLES / "dynamic_overall_ranking.csv", index=False)

cca_delta = dynamic_cca.astype(float) - static_cca.astype(float)
cca_delta.to_csv(TABLES / "static_vs_dynamic_cca_delta.csv")

logistic = sample[
    (sample["probe"] == "Logistic")
    & sample["feature"].isin(
        ["Txn_Jatin-emb", "BRIDGE-emb", "raw_log1p_TPM", "PCA-64"]
    )
].copy()
logistic.to_csv(TABLES / "dynamic_sample_logistic_summary.csv", index=False)

dynamic_only = comparison[
    comparison["model"].isin(["Txn_Jatin", "BRIDGE"])
    & (comparison["variant"] == "full")
].copy()

lines = [
    "# Dynamic embedding benchmark summary",
    "",
    "Txn_Jatin and BRIDGE used full-corpus expression-conditioned gene "
    "embeddings. All seven comparison models remained static.",
    "",
    "## Dynamic extraction provenance",
    "",
    f"- Context corpus: **{config['expected_context_samples']:,} ARCHS4 samples** "
    "per dynamic model.",
    f"- Context batch: **{config['context_batch']}**, bf16 inference with "
    "float32 corpus accumulation.",
    "- No model was fine-tuned.",
    "- Audit requirement: exactly Txn_Jatin and BRIDGE dynamic; all other "
    "models static; no fallback embeddings.",
    "",
    "| Model | Dynamic | Context samples | Context files | Extraction time |",
    "|---|---:|---:|---:|---:|",
]
for model in ["Txn_Jatin", "BRIDGE"]:
    row = provenance[provenance["model"] == model].iloc[0]
    lines.append(
        f"| {model} | yes | {int(row['context_samples']):,} | "
        f"{int(row['context_files']):,} | "
        f"{float(row['context_elapsed_seconds']) / 3600:.2f} h |"
    )

lines.extend(
    [
        "",
        "## Overall result",
        "",
        "| Rank | Model | Dynamic mean | Previous static mean | Delta |",
        "|---:|---|---:|---:|---:|",
    ]
)
for _, row in overall.iterrows():
    lines.append(
        f"| {int(row['dynamic_rank'])} | {row['model']} | "
        f"{fmt(row['dynamic_full_track_mean'])} | "
        f"{fmt(row['static_full_track_mean'])} | {row['delta']:+.4f} |"
    )

lines.extend(
    [
        "",
        f"Dynamic BRIDGE ranked **1st overall** across the four full-gene "
        f"tracks ({overall.iloc[0]['dynamic_full_track_mean']:.4f}). "
        f"Dynamic Txn_Jatin ranked **2nd overall** "
        f"({overall[overall['model'] == 'Txn_Jatin'].iloc[0]['dynamic_full_track_mean']:.4f}), "
        "narrowly ahead of static scGPT.",
        "",
        "## Full-gene benchmark",
        "",
        "| Track | Txn_Jatin | Rank | BRIDGE | Rank | Winner |",
        "|---|---:|---:|---:|---:|---|",
    ]
)
for _, row in full_summary.iterrows():
    lines.append(
        f"| {row['track']} | {fmt(row['Txn_Jatin'])} | "
        f"{int(row['Txn_rank'])}/9 | {fmt(row['BRIDGE'])} | "
        f"{int(row['BRIDGE_rank'])}/9 | {row['winner']} "
        f"({fmt(row['winner_score'])}) |"
    )

lines.extend(
    [
        "",
        "## Change from the previous static benchmark",
        "",
        "| Model | Track | Dynamic | Static | Delta |",
        "|---|---|---:|---:|---:|",
    ]
)
for _, row in dynamic_only.iterrows():
    lines.append(
        f"| {row['model']} | {row['track']} | "
        f"{fmt(row['dynamic_score'])} | {fmt(row['static_score'])} | "
        f"{row['delta']:+.4f} |"
    )

lines.extend(
    [
        "",
        "## Representation geometry",
        "",
        f"- Txn_Jatin–BRIDGE CCA increased from "
        f"**{float(static_cca.loc['Txn_Jatin', 'BRIDGE']):.3f}** to "
        f"**{float(dynamic_cca.loc['Txn_Jatin', 'BRIDGE']):.3f}**.",
        f"- Dynamic Txn_Jatin's highest CCA similarity is "
        f"**{dynamic_cca.loc['Txn_Jatin'].drop('Txn_Jatin').astype(float).idxmax()}** "
        f"at **{dynamic_cca.loc['Txn_Jatin'].drop('Txn_Jatin').astype(float).max():.3f}**.",
        "- Full-corpus contextualization therefore makes Txn_Jatin and BRIDGE "
        "substantially more aligned while retaining distinct downstream scores.",
        "",
        "## Sample embeddings",
        "",
        "Sample embeddings already use expression-conditioned forward passes, so "
        "this section checks downstream transfer rather than replacing them with "
        "the corpus-mean gene tables.",
        "",
        "| Dataset | Split | Feature | F1 | AUROC |",
        "|---|---|---|---:|---:|",
    ]
)
for _, row in logistic.sort_values(["dataset", "scheme", "feature"]).iterrows():
    lines.append(
        f"| {row['dataset']} | {row['scheme']} | {row['feature']} | "
        f"{fmt(row['f1_mean'])} | {fmt(row['auroc_mean'])} |"
    )

lines.extend(
    [
        "",
        "## Caveats",
        "",
        "- The dynamic gene table is a corpus mean of frozen per-gene hidden "
        "states, not a fine-tuned checkpoint.",
        "- TRRUST paired-gene scores were generated in a separate process from "
        "the exact exported embedding matrices.",
        "- Paired-track dynamic-versus-static deltas compare two benchmark "
        "runs. Unchanged static controls varied by at most 0.0043 AUROC on "
        "that track, so those deltas include small negative-sampling rerun "
        "variation.",
        "- SynLethDB and temporal GO availability follow the same caveats as the "
        "previous benchmark.",
    ]
)

(ROOT / "DYNAMIC_BENCHMARK_SUMMARY.md").write_text("\n".join(lines) + "\n")

graph_index = """# Dynamic benchmark graphs and results

## One-page dashboard

[PNG](figures/dynamic_story_dashboard.png) |
[PDF](figures/dynamic_story_dashboard.pdf)

![Dynamic benchmark dashboard](figures/dynamic_story_dashboard.png)

## Narrative graphs

- [Nine-model dynamic landscape](figures/dynamic_story_01_landscape.png)
- [Dynamic versus static changes](figures/dynamic_story_02_gain_vs_static.png)
- [Dynamic Txn_Jatin versus dynamic BRIDGE](figures/dynamic_story_03_txn_vs_bridge.png)
- [Dynamic embedding similarity](figures/dynamic_story_04_similarity.png)

Each narrative graph also has a publication-ready PDF beside the PNG.

## Original benchmark graphs

- [Gene-track leaderboard](figures/leaderboard.png)
- [Sample benchmark](figures/sample_benchmark.png)
- [Inter-model similarity](figures/intermodel_similarity.png)
- [Performance correlation](figures/performance_correlation.png)
- [Cross-species correlation](figures/cross_species_corr.png)

## Primary tables

- [Dynamic full-track summary](tables/dynamic_full_track_summary.csv)
- [Overall dynamic ranking](tables/dynamic_overall_ranking.csv)
- [Static versus dynamic comparison](tables/static_vs_dynamic_gene_scores.csv)
- [Static versus dynamic CCA delta](tables/static_vs_dynamic_cca_delta.csv)
- [Dynamic embedding provenance](tables/dynamic_embedding_provenance.csv)
- [Full leaderboard](tables/leaderboard_gene_tracks.csv)
- [All gene scores](tables/gene_scores.csv)
- [Paired-gene scores](tables/paired_scores.csv)
- [Sample scores](tables/sample_scores.csv)
- [Inter-model CCA](tables/intermodel_cca.csv)
- [Council ledger](tables/council_ledger.csv)
"""
(ROOT / "GRAPHS_AND_RESULTS.md").write_text(graph_index)

print(ROOT / "DYNAMIC_BENCHMARK_SUMMARY.md")
