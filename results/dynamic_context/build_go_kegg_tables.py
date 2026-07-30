"""Export the measured GO tables and document unavailable KEGG/dataset combinations."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"

MODELS = [
    "Txn_Jatin",
    "BRIDGE",
    "scGPT",
    "Geneformer",
    "BulkFormer_37M",
    "BulkFormer_50M",
    "BulkFormer_93M",
    "BulkFormer_127M",
    "BulkFormer_147M",
]


def markdown_table(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    rows = [headers, ["---"] * len(headers)]
    for values in df.itertuples(index=False, name=None):
        rows.append(["" if pd.isna(v) else str(v) for v in values])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main() -> None:
    gene = pd.read_csv(TABLES / "gene_scores.csv")
    sample = pd.read_csv(TABLES / "sample_scores.csv")
    provenance = pd.read_csv(TABLES / "dynamic_embedding_provenance.csv")
    provenance = provenance[provenance["model"].isin(MODELS)].copy()
    provenance["embedding_type"] = provenance["dynamic"].map(
        {True: "dynamic", False: "static"}
    )
    provenance = (
        provenance.set_index("model")
        .reindex(MODELS)
        .reset_index()
    )
    provenance[
        ["model", "embedding_type", "dynamic", "fallback", "shape", "checkpoint"]
    ].to_csv(TABLES / "hybrid_embedding_provenance.csv", index=False)

    go = gene[
        (gene["track"] == "single-gene-GO")
        & (gene["control"] == "real")
        & gene["model"].isin(MODELS)
    ].copy()
    full_long = go[go["variant"] == "full"][
        ["term", "n_pos", "model", "auroc", "auprc"]
    ].copy()
    full_long["term_AUROC_rank"] = full_long.groupby("term")["auroc"].rank(
        ascending=False, method="min"
    ).astype("Int64")
    full_long = full_long.sort_values(["term", "term_AUROC_rank", "model"])
    full_long.to_csv(
        TABLES / "go_term_scores_full_long_requested_models.csv",
        index=False,
        float_format="%.6f",
    )
    full_long.to_csv(
        TABLES / "go_term_scores_full_hybrid_embeddings.csv",
        index=False,
        float_format="%.6f",
    )

    summary_rows = []
    for variant in ("common", "full"):
        part = go[go["variant"] == variant]
        grouped = (
            part.groupby("model", as_index=False)
            .agg(
                GO_terms=("term", "nunique"),
                mean_AUROC=("auroc", "mean"),
                mean_AUPRC=("auprc", "mean"),
            )
            .set_index("model")
            .reindex(MODELS)
            .reset_index()
        )
        grouped.insert(1, "variant", variant)
        grouped["AUROC_rank"] = grouped["mean_AUROC"].rank(
            ascending=False, method="min"
        ).astype("Int64")
        summary_rows.append(grouped)

        for metric in ("auroc", "auprc"):
            wide = part.pivot(index="term", columns="model", values=metric)
            n_pos = part.groupby("term")["n_pos"].max()
            wide = wide.reindex(columns=MODELS)
            wide.insert(0, "n_positive_genes", n_pos)
            wide = wide.reset_index()
            wide.to_csv(
                TABLES / f"go_term_{metric}_{variant}_requested_models.csv",
                index=False,
                float_format="%.6f",
            )
            wide.to_csv(
                TABLES / f"go_term_{metric}_{variant}_hybrid_embeddings.csv",
                index=False,
                float_format="%.6f",
            )

    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(
        TABLES / "go_model_summary_requested.csv",
        index=False,
        float_format="%.6f",
    )
    summary.to_csv(
        TABLES / "go_model_summary_hybrid_embeddings.csv",
        index=False,
        float_format="%.6f",
    )

    tracks = pd.read_csv(TABLES / "leaderboard_gene_tracks.csv")
    full_tracks = tracks[tracks["variant"] == "full"].set_index("track")
    full_tracks = full_tracks[MODELS].T.reset_index().rename(columns={"index": "model"})
    full_tracks.insert(
        1,
        "embedding_type",
        full_tracks["model"].map(
            provenance.set_index("model")["embedding_type"].to_dict()
        ),
    )
    full_tracks.to_csv(
        TABLES / "hybrid_full_track_auroc.csv",
        index=False,
        float_format="%.6f",
    )

    static_tracks_path = ROOT.parent / "benchmark" / "tables" / "leaderboard_gene_tracks.csv"
    static_tracks = pd.read_csv(static_tracks_path)
    static_full = static_tracks[static_tracks["variant"] == "full"].set_index("track")
    static_full = static_full[MODELS].T.reset_index().rename(columns={"index": "model"})
    static_full.insert(1, "embedding_type", "static")
    static_full["four_track_mean"] = static_full[
        ["gene-set", "paired", "single-gene-GO", "single-gene-disease"]
    ].mean(axis=1)
    static_full.to_csv(
        TABLES / "static_full_track_auroc.csv",
        index=False,
        float_format="%.6f",
    )

    hybrid_numeric = full_tracks.set_index("model").drop(columns="embedding_type")
    static_numeric = static_full.set_index("model").drop(columns="embedding_type")
    hybrid_numeric["four_track_mean"] = hybrid_numeric.mean(axis=1)
    delta = hybrid_numeric - static_numeric
    delta.insert(
        0,
        "embedding_change",
        [
            "static -> dynamic" if model in {"Txn_Jatin", "BRIDGE"} else "static control"
            for model in delta.index
        ],
    )
    delta.reset_index().to_csv(
        TABLES / "hybrid_minus_static_full_track_delta.csv",
        index=False,
        float_format="%.6f",
    )

    # Only the two dynamic models have sample-level embeddings in this run.
    dynamic_sample = sample[
        sample["feature"].isin(["Txn_Jatin-emb", "BRIDGE-emb"])
        & (sample["probe"] == "Logistic")
    ][
        [
            "dataset",
            "feature",
            "scheme",
            "f1_mean",
            "f1_std",
            "auroc_mean",
            "auroc_std",
            "n",
            "n_classes",
        ]
    ].copy()
    dynamic_sample.insert(
        2, "model", dynamic_sample["feature"].str.replace("-emb", "", regex=False)
    )
    dynamic_sample = dynamic_sample.drop(columns="feature")
    dynamic_sample.to_csv(
        TABLES / "tcga_osdr_dynamic_sample_scores.csv",
        index=False,
        float_format="%.6f",
    )

    full = summary[summary["variant"] == "full"].copy()
    full["mean_AUROC"] = full["mean_AUROC"].map(lambda x: f"{x:.4f}")
    full["mean_AUPRC"] = full["mean_AUPRC"].map(lambda x: f"{x:.4f}")
    full["AUROC_rank"] = full["AUROC_rank"].astype(str)
    full = full[
        ["model", "GO_terms", "mean_AUROC", "mean_AUPRC", "AUROC_rank"]
    ].rename(
        columns={
            "model": "Model",
            "GO_terms": "GO terms",
            "mean_AUROC": "Mean AUROC",
            "mean_AUPRC": "Mean AUPRC",
            "AUROC_rank": "AUROC rank",
        }
    )

    common = summary[summary["variant"] == "common"].copy()
    common["mean_AUROC"] = common["mean_AUROC"].map(lambda x: f"{x:.4f}")
    common["mean_AUPRC"] = common["mean_AUPRC"].map(lambda x: f"{x:.4f}")
    common["AUROC_rank"] = common["AUROC_rank"].astype(str)
    common = common[
        ["model", "GO_terms", "mean_AUROC", "mean_AUPRC", "AUROC_rank"]
    ].rename(
        columns={
            "model": "Model",
            "GO_terms": "GO terms",
            "mean_AUROC": "Mean AUROC",
            "mean_AUPRC": "Mean AUPRC",
            "AUROC_rank": "AUROC rank",
        }
    )

    sample_md = dynamic_sample.copy()
    for col in ("f1_mean", "f1_std", "auroc_mean", "auroc_std"):
        sample_md[col] = sample_md[col].map(lambda x: f"{x:.4f}")
    sample_md = sample_md.rename(
        columns={
            "dataset": "Dataset",
            "scheme": "CV scheme",
            "model": "Model",
            "f1_mean": "F1 macro",
            "f1_std": "F1 SD",
            "auroc_mean": "AUROC",
            "auroc_std": "AUROC SD",
            "n": "Samples",
            "n_classes": "Classes",
        }
    )

    availability = pd.DataFrame(
        [
            ["GO term prediction", "Frozen gene embeddings", "Yes", "40 GO-BP terms; not TCGA/OSDR-specific"],
            ["KEGG term prediction", "Frozen gene embeddings", "No", "Hallmark, not KEGG, was used for gene-set matching"],
            ["GO terms within TCGA", "TCGA", "No", "TCGA task was five-class cancer-type prediction"],
            ["KEGG terms within TCGA", "TCGA", "No", "No pathway-label benchmark was run"],
            ["GO terms within OSDR", "OSDR", "No", "OSDR task was flight-vs-ground prediction"],
            ["KEGG terms within OSDR", "OSDR", "No", "No pathway-label benchmark was run"],
        ],
        columns=["Requested result", "Data/track", "Measured", "What exists"],
    )
    availability.to_csv(TABLES / "requested_result_availability.csv", index=False)

    provenance_md = provenance[
        ["model", "embedding_type", "shape", "fallback"]
    ].rename(
        columns={
            "model": "Model",
            "embedding_type": "Embedding",
            "shape": "Matrix shape",
            "fallback": "Fallback",
        }
    )

    tracks_md = full_tracks.rename(
        columns={
            "model": "Model",
            "embedding_type": "Embedding",
            "gene-set": "Hallmark gene-set",
            "paired": "Paired genes",
            "single-gene-GO": "GO",
            "single-gene-disease": "Disease",
        }
    )
    for col in ("Hallmark gene-set", "Paired genes", "GO", "Disease"):
        tracks_md[col] = tracks_md[col].map(lambda x: f"{x:.4f}")

    report = f"""# Hybrid dynamic/static embedding benchmark

## Embedding design

This is the requested hybrid comparison: Txn_Jatin and BRIDGE use dynamic,
full-corpus context-averaged embeddings; every other model uses a static gene
embedding matrix.

{markdown_table(provenance_md)}

No model used a fallback embedding.

## Full-coverage benchmark overview

All values below are mean AUROC.

{markdown_table(tracks_md)}

## What can be reported

The completed benchmark measured **40 individual GO Biological Process terms**
using frozen gene embeddings. These GO scores are not stratified by TCGA or
OSDR. The full-coverage aggregate results for the requested models are:

{markdown_table(full)}

The common-gene comparison, where every model is evaluated on the same shared
gene vocabulary, is:

{markdown_table(common)}

## Requested-result availability

{markdown_table(availability)}

The gene-set track used the **Hallmark** collection. Although a source-code
comment mentions "Hallmark/KEGG," the executed pipeline loaded only
`hallmark_gene_sets.json`; therefore no KEGG score is present and none is
reported as if it were measured.

## TCGA and OSDR sample-level results

Only the two dynamic models had sample embeddings evaluated in the completed
run. These are cancer-type and spaceflight classification results, not GO or
KEGG term scores:

{markdown_table(sample_md)}

## Detailed files

- `tables/hybrid_embedding_provenance.csv`: dynamic/static status and checkpoints
- `tables/hybrid_full_track_auroc.csv`: all four gene-level benchmark tracks
- `tables/go_term_auroc_full_hybrid_embeddings.csv`: all 40 GO terms, full coverage
- `tables/go_term_auprc_full_hybrid_embeddings.csv`: all 40 GO terms, full coverage
- `tables/go_term_auroc_common_hybrid_embeddings.csv`: all 40 GO terms, shared genes
- `tables/go_term_auprc_common_hybrid_embeddings.csv`: all 40 GO terms, shared genes
- `tables/go_term_scores_full_hybrid_embeddings.csv`: one row per GO term and model
- `tables/go_model_summary_hybrid_embeddings.csv`: aggregate GO scores and ranks
- `tables/tcga_osdr_dynamic_sample_scores.csv`: Txn_Jatin and BRIDGE sample scores
- `tables/requested_result_availability.csv`: explicit GO/KEGG and TCGA/OSDR availability

## Interpretation note

Txn_Jatin uses its one-epoch `epoch_00.pt` checkpoint and BRIDGE uses
`best_model.pt`. Both dynamic matrices were averaged from 978,212 full-corpus
ARCHS4 contexts. scGPT, Geneformer, and the five BulkFormer variants use static
gene embeddings. Missing evaluations are marked unavailable rather than zero.
"""
    appendix_path = ROOT / "ARCHITECTURE_AND_STATIC_APPENDIX.md"
    if appendix_path.exists():
        report = report.rstrip() + "\n\n" + appendix_path.read_text(
            encoding="utf-8"
        ).lstrip()
    (ROOT / "GO_KEGG_TCGA_OSDR_REPORT.md").write_text(report, encoding="utf-8")
    (ROOT / "HYBRID_DYNAMIC_STATIC_BENCHMARK.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
