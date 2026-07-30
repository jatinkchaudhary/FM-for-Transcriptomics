#!/usr/bin/env python3
"""Atlas-match all OSDR samples, render figures, and emit one LLM report per sample."""

from __future__ import annotations

import argparse
import importlib.util
import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def load_atlas_class(path: Path):
    spec = importlib.util.spec_from_file_location("atlas_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.AtlasRuntime


def save_figures(metrics: pd.DataFrame, matches: pd.DataFrame, output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(metrics["masked_pearson"].dropna(), bins=45, color="#147d92")
    axes[0].axvline(metrics["masked_pearson"].median(), color="#ca4b36", lw=2)
    axes[0].set(xlabel="Per-sample Pearson r", ylabel="Samples", title="15% masked-gene recovery")
    tissue = metrics.groupby("tissue")["masked_pearson"].median().sort_values().tail(20)
    axes[1].barh(tissue.index, tissue.values, color="#5669a8")
    axes[1].set(xlabel="Median Pearson r", title="Recovery by OSDR tissue (top 20)")
    fig.tight_layout()
    fig.savefig(output / "imputation_performance.png", dpi=220)
    plt.close(fig)

    top = matches["predicted_tissue"].value_counts().head(18)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(top.index[::-1], top.values[::-1], color="#2c8c6b")
    ax.set(xlabel="OSDR samples", title="Top atlas tissue assignments")
    fig.tight_layout()
    fig.savefig(output / "atlas_tissue_assignments.png", dpi=220)
    plt.close(fig)

    cross = pd.crosstab(matches["osdr_tissue"], matches["predicted_tissue"])
    cross = cross.loc[cross.sum(axis=1).nlargest(15).index, cross.sum().nlargest(15).index]
    fig, ax = plt.subplots(figsize=(13, 9))
    image = ax.imshow(np.log1p(cross.to_numpy()), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cross.columns)), cross.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(cross.index)), cross.index)
    ax.set(xlabel="Atlas prediction", ylabel="OSDR metadata", title="Tissue mapping matrix, log(1+n)")
    fig.colorbar(image, ax=ax, label="log(1 + samples)")
    fig.tight_layout()
    fig.savefig(output / "tissue_mapping_matrix.png", dpi=220)
    plt.close(fig)


def save_knowledge_graph(
    atlas, genes: list[str], values: np.ndarray, output: Path
) -> None:
    top_indices = np.argsort(values.mean(axis=0))[::-1][:40]
    top_genes = [genes[index] for index in top_indices]
    candidate = []
    selected = set(top_genes)
    for disease, members in atlas.disease_sets.items():
        overlap = sorted(selected & members)
        if len(overlap) >= 2:
            candidate.append((len(overlap), disease, overlap))
    candidate.sort(reverse=True)
    graph = nx.Graph()
    for gene in top_genes[:25]:
        graph.add_node(gene, kind="gene")
    for _, disease, overlap in candidate[:10]:
        label = disease[:45]
        graph.add_node(label, kind="disease")
        for gene in overlap:
            if gene in graph:
                graph.add_edge(gene, label)
    fig, ax = plt.subplots(figsize=(16, 11))
    position = nx.spring_layout(graph, seed=42, k=0.8)
    colors = ["#147d92" if graph.nodes[node]["kind"] == "gene" else "#ca4b36" for node in graph]
    sizes = [650 if graph.nodes[node]["kind"] == "gene" else 1100 for node in graph]
    nx.draw_networkx_edges(graph, position, alpha=0.28, width=1.2, ax=ax)
    nx.draw_networkx_nodes(graph, position, node_color=colors, node_size=sizes, ax=ax)
    nx.draw_networkx_labels(graph, position, font_size=7, font_color="#14202b", ax=ax)
    ax.set_title("Cohort gene-to-disease-set evidence graph (exploratory, not diagnostic)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "gene_disease_knowledge_graph.png", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sample-metrics", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--atlas-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--max-reports", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = np.load(args.matrix, allow_pickle=False)
    values = source["X"].astype(np.float32)
    genes = source["genes"].astype(str).tolist()
    metadata = pd.read_csv(args.metadata)
    masked = np.load(args.predictions, allow_pickle=False)
    values[:, masked["masked_indices"]] = masked["prediction"]
    config = json.loads(args.config.read_text())
    atlas_config = config["atlas"]
    atlas_config["ollama"]["enabled"] = False
    atlas_config["top_k"] = max(50, int(atlas_config.get("top_k", 8)))
    AtlasRuntime = load_atlas_class(args.atlas_runtime)
    atlas = AtlasRuntime(atlas_config)
    atlas._load()
    results, shared = atlas._match(genes, metadata["sample_id"].astype(str).tolist(), values.T)

    rows = []
    evidence_dir = args.output_dir / "evidence"
    report_dir = args.output_dir / "llm_reports"
    evidence_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    for index, result in enumerate(results):
        top = next(
            (
                match for match in result["matches"]
                if "unlabelled" not in match["tissue"].lower()
                and match["tissue"].lower() != "unknown"
            ),
            result["matches"][0],
        )
        species = result["species_evidence"][0]
        row = {
            "sample_index": index,
            "sample_id": result["sample"],
            "accession": metadata.iloc[index]["accession"],
            "condition": metadata.iloc[index]["condition"],
            "osdr_tissue": metadata.iloc[index]["tissue"],
            "predicted_species": species["label"],
            "species_weight": species["weight"],
            "predicted_tissue": top["tissue"],
            "top_similarity": top["similarity"],
            "top_reference": top["reference_id"],
        }
        rows.append(row)
        order = np.argsort(values[index])[::-1][:25]
        top_genes = [genes[i] for i in order]
        annotations = []
        for gene in top_genes:
            annotation = atlas.annotations.get(gene)
            if not annotation:
                continue
            annotations.append({
                "gene": gene,
                "name": annotation.get("name"),
                "summary": str(annotation.get("summary", ""))[:400],
                "type_of_gene": annotation.get("type_of_gene"),
            })
        evidence = {
            "sample_metadata": row,
            "atlas_matches": result["matches"][:5],
            "species_evidence": result["species_evidence"],
            "tissue_evidence": result["tissue_evidence"],
            "top_expressed_genes": top_genes,
            "gene_annotations": annotations[:12],
            "limitations": [
                "Similarity is not a calibrated probability.",
                "Disease associations are exploratory and are not diagnoses.",
                "The expression profile includes model-imputed values for 15% of genes.",
            ],
        }
        (evidence_dir / f"{index:04d}_{result['sample'][:80]}.json").write_text(
            json.dumps(evidence, indent=2) + "\n"
        )

    matches = pd.DataFrame(rows)
    matches.to_csv(args.output_dir / "atlas_matches.csv", index=False)
    metrics = pd.read_csv(args.sample_metrics)
    save_figures(metrics, matches, args.output_dir)
    save_knowledge_graph(atlas, genes, values, args.output_dir)
    summary = {
        "samples": len(results),
        "shared_genes": shared,
        "species_counts": matches["predicted_species"].value_counts().to_dict(),
        "top_predicted_tissues": matches["predicted_tissue"].value_counts().head(20).to_dict(),
    }
    (args.output_dir / "atlas_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.skip_llm:
        return 0
    ollama = config["atlas"]["ollama"]
    completed = {path.stem for path in report_dir.glob("*.md")}
    evidence_paths = sorted(evidence_dir.glob("*.json"))
    if args.max_reports:
        evidence_paths = evidence_paths[: args.max_reports]
    for count, path in enumerate(evidence_paths, 1):
        if path.stem in completed:
            continue
        evidence = json.loads(path.read_text())
        prompt = (
            "You are a transcriptomics evidence summarizer. Use only this JSON. "
            "Separate observations from hypotheses. Do not diagnose disease or invent "
            "probabilities. Return exactly these Markdown headings: ## Provenance, "
            "## Tissue and species, ## Genes, ## Biological hypotheses, "
            "## Cross-species implications, ## Limitations. Mention the sample_id "
            "verbatim under Provenance. The complete answer must be under 250 words; "
            "use one or two bullets per heading.\nEVIDENCE:\n"
            + json.dumps(evidence, separators=(",", ":"))
        )
        body = json.dumps({
            "model": ollama["model"], "prompt": prompt, "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.1, "seed": 42, "num_ctx": 8192, "num_predict": 750},
        }).encode()
        request = urllib.request.Request(
            ollama["url"], data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                text = json.loads(response.read()).get("response", "").strip()
            sample_id = str(evidence["sample_metadata"]["sample_id"])
            required = ("## Provenance", "## Tissue and species", "## Limitations")
            if sample_id not in text or not all(header in text for header in required):
                raise RuntimeError("LLM response failed grounding/structure validation")
            (report_dir / f"{path.stem}.md").write_text(text + "\n")
        except Exception as error:
            (report_dir / f"{path.stem}.error.json").write_text(
                json.dumps(
                    {"error": str(error), "raw_response": locals().get("text", "")},
                    indent=2,
                ) + "\n"
            )
        if count % 10 == 0 or count == len(evidence_paths):
            (args.output_dir / "llm_progress.json").write_text(json.dumps({
                "reports_complete": len(list(report_dir.glob("*.md"))),
                "reports_total": len(evidence_paths),
            }, indent=2) + "\n")
            print(f"LLM {count}/{len(evidence_paths)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
