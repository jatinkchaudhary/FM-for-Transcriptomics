#!/usr/bin/env python3
"""Build a production human/mouse bulk-expression atlas and annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def decode(values) -> np.ndarray:
    return np.asarray(
        [value.decode(errors="replace") if isinstance(value, bytes) else str(value) for value in values],
        dtype=str,
    )


def log1p_cpm(matrix: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(matrix, dtype=np.float32), 0)
    totals = values.sum(axis=1, keepdims=True)
    return np.log1p(values * (1_000_000.0 / np.maximum(totals, 1.0)))


def add_h5_references(
    path: Path,
    group_key: str,
    genes: list[str],
    source: str,
    species: str,
    representatives: int,
) -> tuple[list[np.ndarray], list[dict[str, str]], dict[str, int]]:
    vectors: list[np.ndarray] = []
    metadata: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    with h5py.File(path, "r") as handle:
        source_genes = decode(handle["meta/genes"][:])
        source_index = {gene.upper(): index for index, gene in enumerate(source_genes)}
        columns = np.asarray([source_index.get(gene, -1) for gene in genes])
        if np.any(columns < 0):
            missing = int(np.count_nonzero(columns < 0))
            print(f"{source}: {missing:,} atlas genes absent and filled with zero", flush=True)
        valid_target = np.flatnonzero(columns >= 0)
        valid_source = columns[valid_target]
        groups = decode(handle[f"meta/{group_key}"][:])
        sample_ids = decode(
            handle["meta/sampid"][:]
            if "meta/sampid" in handle
            else handle["meta/sampleid"][:]
        )
        source_expression = handle["data/expression"][:]
        expression = np.zeros((len(source_expression), len(genes)), dtype=np.float32)
        expression[:, valid_target] = source_expression[:, valid_source]
        del source_expression
        for group in sorted(set(groups)):
            indices = np.flatnonzero(groups == group)
            if not len(indices):
                continue
            counts[group] = len(indices)
            selected = indices[
                np.linspace(0, len(indices) - 1, min(representatives, len(indices)), dtype=int)
            ]
            normalized = log1p_cpm(expression[selected])
            # Build the centroid from up to 256 evenly spaced samples.
            centroid_indices = indices[
                np.linspace(0, len(indices) - 1, min(256, len(indices)), dtype=int)
            ]
            centroid_raw = np.mean(
                expression[centroid_indices], axis=0, keepdims=True
            )
            vectors.append(log1p_cpm(centroid_raw)[0])
            metadata.append(
                {
                    "reference_id": f"{source}:centroid:{group}",
                    "species": species,
                    "tissue": group,
                    "study": source,
                    "source": str(path),
                    "reference_type": "centroid",
                    "group_samples": str(len(indices)),
                }
            )
            for row, sample_index in zip(normalized, selected):
                vectors.append(row)
                metadata.append(
                    {
                        "reference_id": f"{source}:sample:{sample_ids[sample_index]}",
                        "species": species,
                        "tissue": group,
                        "study": source,
                        "source": str(path),
                        "reference_type": "representative_sample",
                        "group_samples": str(len(indices)),
                    }
                )
    return vectors, metadata, counts


def add_archs4_species(
    batch_dir: Path,
    genes: list[str],
    species: str,
    sample_limit: int,
) -> tuple[list[np.ndarray], list[dict[str, str]]]:
    vectors: list[np.ndarray] = []
    metadata: list[dict[str, str]] = []
    files = sorted(batch_dir.glob(f"{species}_batch_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {species} batch shards in {batch_dir}")
    shard_limit = min(64, len(files))
    files = [
        files[index]
        for index in np.linspace(0, len(files) - 1, shard_limit, dtype=int)
    ]
    per_file = max(1, int(np.ceil(sample_limit / len(files))))
    for path in files:
        frame = pd.read_parquet(path)
        aligned = frame.reindex(columns=genes, fill_value=0)
        take = np.linspace(0, len(aligned) - 1, min(per_file, len(aligned)), dtype=int)
        normalized = log1p_cpm(aligned.iloc[take].to_numpy(dtype=np.float32))
        for row, index in zip(normalized, aligned.index[take]):
            vectors.append(row)
            metadata.append(
                {
                    "reference_id": f"ARCHS4:{species}:{index}:{path.stem}",
                    "species": species,
                    "tissue": "unlabelled ARCHS4 bulk sample",
                    "study": "ARCHS4 prepared training corpus",
                    "source": str(path),
                    "reference_type": "species_reference",
                    "group_samples": "",
                }
            )
            if len(vectors) >= sample_limit:
                return vectors, metadata
        print(
            f"ARCHS4 {species}: {len(vectors):,}/{sample_limit:,} references",
            flush=True,
        )
    return vectors, metadata


def mygene_annotations(genes: list[str], output: Path) -> list[dict]:
    import requests

    fields = "symbol,name,summary,entrezgene,ensembl.gene,type_of_gene,go,pathway,homologene"
    results: list[dict] = []
    for start in range(0, len(genes), 500):
        chunk = genes[start : start + 500]
        response = requests.post(
            "https://mygene.info/v3/query",
            data={
                "q": ",".join(chunk),
                "scopes": "symbol",
                "fields": fields,
                "species": "human",
                "size": 1,
            },
            timeout=120,
        )
        response.raise_for_status()
        results.extend(response.json())
        print(f"MyGene annotations: {min(start + 500, len(genes)):,}/{len(genes):,}", flush=True)
    annotations = {
        row.get("query", "").upper(): {
            key: value
            for key, value in row.items()
            if key not in {"_id", "_score", "query"} and value is not None
        }
        for row in results
        if not row.get("notfound")
    }
    output.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    return results


def disease_sets(output: Path) -> int:
    libraries = ["DisGeNET", "OMIM_Disease", "Human_Phenotype_Ontology"]
    combined: dict[str, list[str]] = {}
    for library in libraries:
        query = urllib.parse.urlencode({"mode": "text", "libraryName": library})
        url = f"https://maayanlab.cloud/Enrichr/geneSetLibrary?{query}"
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                text = response.read().decode("utf-8")
        except Exception as error:
            print(f"warning: {library} unavailable: {error}", flush=True)
            continue
        for line in text.splitlines():
            cells = line.split("\t")
            if len(cells) >= 3:
                combined[f"{library} | {cells[0]}"] = sorted(
                    {gene.upper() for gene in cells[2:] if gene}
                )
    output.write_text(json.dumps(combined), encoding="utf-8")
    return len(combined)


def orthologs_from_mygene(rows: list[dict], output: Path) -> int:
    records = []
    for row in rows:
        homologene = row.get("homologene") or {}
        genes = homologene.get("genes", []) if isinstance(homologene, dict) else []
        mouse_ids = [str(item[1]) for item in genes if len(item) == 2 and int(item[0]) == 10090]
        for mouse_id in mouse_ids:
            records.append(
                {
                    "mouse_symbol": row.get("query", "").upper(),
                    "human_symbol": row.get("symbol", row.get("query", "")).upper(),
                    "orthology_type": "HomoloGene group",
                    "source": f"MyGene.info/HomoloGene mouse Entrez {mouse_id}",
                }
            )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mouse_symbol", "human_symbol", "orthology_type", "source"],
        )
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-genes", type=Path, required=True)
    parser.add_argument("--gtex", type=Path, required=True)
    parser.add_argument("--tcga", type=Path, required=True)
    parser.add_argument("--archs4-batches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representatives-per-group", type=int, default=24)
    parser.add_argument("--archs4-per-species", type=int, default=512)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    genes_frame = pd.read_csv(args.canonical_genes)
    column = next(
        (name for name in genes_frame.columns if name.lower() in {"gene", "symbol", "gene_symbol"}),
        genes_frame.columns[0],
    )
    genes = genes_frame[column].astype(str).str.upper().tolist()
    vectors: list[np.ndarray] = []
    metadata: list[dict[str, str]] = []
    source_counts = {}

    part, rows, counts = add_h5_references(
        args.gtex, "smtsd", genes, "GTEx recount2", "human",
        args.representatives_per_group,
    )
    vectors.extend(part)
    metadata.extend(rows)
    source_counts["GTEx"] = counts
    part, rows, counts = add_h5_references(
        args.tcga, "gdc_cases.project.project_id", genes, "TCGA recount2", "human",
        args.representatives_per_group,
    )
    vectors.extend(part)
    metadata.extend(rows)
    source_counts["TCGA"] = counts
    for species in ("human", "mouse"):
        part, rows = add_archs4_species(
            args.archs4_batches, genes, species, args.archs4_per_species
        )
        vectors.extend(part)
        metadata.extend(rows)
        source_counts[f"ARCHS4_{species}"] = len(rows)

    expression = np.asarray(vectors, dtype=np.float32)
    reference_ids = np.asarray([row["reference_id"] for row in metadata], dtype=str)
    np.savez_compressed(
        args.output / "atlas_expression.npz",
        genes=np.asarray(genes, dtype=str),
        reference_ids=reference_ids,
        expression=expression,
    )
    pd.DataFrame(metadata).to_csv(args.output / "reference_metadata.csv", index=False)
    annotation_rows = mygene_annotations(genes, args.output / "gene_annotations.json")
    disease_count = disease_sets(args.output / "disease_gene_sets.json")
    ortholog_count = orthologs_from_mygene(
        annotation_rows, args.output / "mouse_human_orthologs.csv"
    )
    manifest = {
        "schema_version": 1,
        "genes": len(genes),
        "references": len(metadata),
        "expression_shape": list(expression.shape),
        "sources": {
            "GTEx": str(args.gtex),
            "TCGA": str(args.tcga),
            "ARCHS4_batches": str(args.archs4_batches),
            "MyGene.info": "https://mygene.info/v3/query",
            "Enrichr": "https://maayanlab.cloud/Enrichr/",
        },
        "source_group_counts": source_counts,
        "disease_sets": disease_count,
        "ortholog_rows": ortholog_count,
        "normalization": "counts -> library-size CPM -> log1p",
        "representatives_per_group": args.representatives_per_group,
        "archs4_per_species": args.archs4_per_species,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (args.output / "SHA256SUMS.txt").open("w", encoding="utf-8") as handle:
        for path in sorted(args.output.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS.txt":
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                handle.write(f"{digest}  {path.name}\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
