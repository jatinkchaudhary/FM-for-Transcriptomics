#!/usr/bin/env python3
"""Reference-atlas matching and an evidence-grounded Ollama language head."""

from __future__ import annotations

import csv
import json
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


class AtlasUnavailableError(RuntimeError):
    """The configured atlas cannot support a requested analysis."""


def _read_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _softmax(values: np.ndarray, temperature: float = 0.08) -> np.ndarray:
    shifted = (values - np.max(values)) / temperature
    weights = np.exp(np.clip(shifted, -50, 50))
    return weights / max(float(weights.sum()), 1e-12)


class AtlasRuntime:
    """Match expression profiles to a locally versioned reference atlas.

    The NPZ contract is deliberately small: ``genes`` (G strings),
    ``reference_ids`` (R strings), and ``expression`` (R x G, log1p scale).
    Metadata and annotations remain inspectable CSV/JSON files.
    """

    def __init__(self, config: dict[str, Any] | None):
        self.config = config or {}
        self.loaded = False
        self.genes: list[str] = []
        self.gene_index: dict[str, int] = {}
        self.reference_ids: list[str] = []
        self.reference_expression: np.ndarray | None = None
        self.metadata: dict[str, dict[str, str]] = {}
        self.annotations: dict[str, dict[str, Any]] = {}
        self.disease_sets: dict[str, set[str]] = {}
        self.orthologs: dict[str, list[dict[str, str]]] = defaultdict(list)

    def status(self) -> dict[str, Any]:
        index = self.config.get("index")
        available = bool(index and Path(index).is_file())
        return {
            "configured": bool(index),
            "available": available,
            "loaded": self.loaded,
            "atlas_name": self.config.get("name", "Gene Atlas"),
            "atlas_version": self.config.get("version", "unversioned"),
            "reference_count": len(self.reference_ids) if self.loaded else None,
            "gene_count": len(self.genes) if self.loaded else None,
            "ollama": {
                "enabled": bool(self.config.get("ollama", {}).get("enabled", False)),
                "model": self.config.get("ollama", {}).get("model", "unknown"),
            },
        }

    def _load(self) -> None:
        if self.loaded:
            return
        index_path = self.config.get("index")
        if not index_path or not Path(index_path).is_file():
            raise AtlasUnavailableError(
                "No local atlas index is configured. Build one with "
                "scripts/build_atlas_index.py and set atlas.index in the model config."
            )
        payload = np.load(index_path, allow_pickle=False)
        self.genes = [str(value).upper() for value in payload["genes"].tolist()]
        self.reference_ids = [str(value) for value in payload["reference_ids"].tolist()]
        self.reference_expression = np.asarray(payload["expression"], dtype=np.float32)
        if self.reference_expression.shape != (len(self.reference_ids), len(self.genes)):
            raise AtlasUnavailableError("Atlas expression shape does not match genes/references")
        self.gene_index = {gene: index for index, gene in enumerate(self.genes)}

        metadata_path = self.config.get("metadata")
        if metadata_path and Path(metadata_path).is_file():
            rows = _read_table(Path(metadata_path))
            self.metadata = {row["reference_id"]: row for row in rows}
        annotation_path = self.config.get("gene_annotations")
        if annotation_path and Path(annotation_path).is_file():
            raw = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
            self.annotations = {str(key).upper(): value for key, value in raw.items()}
        disease_path = self.config.get("disease_gene_sets")
        if disease_path and Path(disease_path).is_file():
            raw = json.loads(Path(disease_path).read_text(encoding="utf-8"))
            self.disease_sets = {
                str(name): {str(gene).upper() for gene in genes}
                for name, genes in raw.items()
            }
        ortholog_path = self.config.get("orthologs")
        if ortholog_path and Path(ortholog_path).is_file():
            for row in _read_table(Path(ortholog_path)):
                self.orthologs[row["mouse_symbol"].upper()].append(row)
        self.loaded = True

    @staticmethod
    def _standardize_rows(matrix: np.ndarray) -> np.ndarray:
        centered = matrix - np.mean(matrix, axis=1, keepdims=True)
        scale = np.linalg.norm(centered, axis=1, keepdims=True)
        return centered / np.maximum(scale, 1e-8)

    def _match(
        self, genes: list[str], samples: list[str], expression: np.ndarray
    ) -> tuple[list[dict[str, Any]], int]:
        shared = [(i, self.gene_index[gene]) for i, gene in enumerate(genes) if gene in self.gene_index]
        full_profile_minimum = int(self.config.get("minimum_shared_genes", 100))
        sparse_minimum = int(self.config.get("sparse_minimum_shared_genes", 20))
        minimum = min(
            full_profile_minimum,
            max(sparse_minimum, int(np.ceil(len(genes) * 0.4))),
        )
        if len(shared) < minimum:
            raise AtlasUnavailableError(
                f"Only {len(shared):,} genes overlap the atlas; at least {minimum:,} are required."
            )
        query_indices, atlas_indices = zip(*shared)
        query = np.maximum(expression[list(query_indices), :].T, 0)
        if float(np.nanmax(query)) > 30:
            totals = query.sum(axis=1, keepdims=True)
            query = np.log1p(query * (1_000_000.0 / np.maximum(totals, 1.0)))
        reference = self.reference_expression[:, list(atlas_indices)]
        similarity = self._standardize_rows(query) @ self._standardize_rows(reference).T
        top_k = min(int(self.config.get("top_k", 8)), len(self.reference_ids))
        output = []
        for sample_index, sample in enumerate(samples):
            full_order = np.argsort(similarity[sample_index])[::-1]
            healthy_order = [
                index
                for index in full_order
                if "unlabelled"
                not in self.metadata.get(self.reference_ids[index], {})
                .get("tissue", "unknown")
                .lower()
                and self.metadata.get(self.reference_ids[index], {})
                .get("tissue", "unknown")
                .lower()
                != "unknown"
                and not self.metadata.get(self.reference_ids[index], {})
                .get("tissue", "")
                .upper()
                .startswith("TCGA-")
            ][:top_k]
            tumor_order = [
                index
                for index in full_order
                if self.metadata.get(self.reference_ids[index], {})
                .get("tissue", "")
                .upper()
                .startswith("TCGA-")
            ][: min(3, top_k)]
            labeled_order = healthy_order + tumor_order
            order = np.asarray(
                labeled_order if labeled_order else full_order[:top_k], dtype=int
            )
            matches = []
            for rank, reference_index in enumerate(order, 1):
                reference_id = self.reference_ids[reference_index]
                meta = self.metadata.get(reference_id, {})
                matches.append(
                    {
                        "rank": rank,
                        "reference_id": reference_id,
                        "similarity": round(float(similarity[sample_index, reference_index]), 6),
                        "species": meta.get("species", "unknown"),
                        "tissue": meta.get("tissue", "unknown"),
                        "study": meta.get("study", ""),
                        "source": meta.get("source", ""),
                        "reference_type": (
                            "tumor_cohort"
                            if meta.get("tissue", "").upper().startswith("TCGA-")
                            else "healthy_tissue"
                        ),
                    }
                )
            nearest_unlabelled = []
            for reference_index in full_order:
                reference_id = self.reference_ids[reference_index]
                meta = self.metadata.get(reference_id, {})
                if "unlabelled" not in meta.get("tissue", "").lower():
                    continue
                nearest_unlabelled.append(
                    {
                        "reference_id": reference_id,
                        "similarity": round(
                            float(similarity[sample_index, reference_index]), 6
                        ),
                        "species": meta.get("species", "unknown"),
                        "source": meta.get("source", ""),
                    }
                )
                if len(nearest_unlabelled) == 3:
                    break

            species_scores: dict[str, list[float]] = defaultdict(list)
            tissue_scores: dict[str, list[float]] = defaultdict(list)
            for match in matches:
                if match["reference_type"] == "healthy_tissue":
                    tissue_scores[match["tissue"]].append(match["similarity"])
            species_candidate_count = min(
                len(full_order),
                int(self.config.get("species_reference_candidates", 50)),
            )
            for reference_index in full_order[:species_candidate_count]:
                reference_id = self.reference_ids[reference_index]
                species = self.metadata.get(reference_id, {}).get("species", "unknown")
                if species.lower() in {"", "unknown"}:
                    continue
                species_scores[species].append(
                    float(similarity[sample_index, reference_index])
                )
            species_labels = list(species_scores)
            species_values = np.asarray([np.max(species_scores[key]) for key in species_labels])
            species_weights = _softmax(species_values)
            ranked_species = sorted(
                zip(species_labels, species_weights),
                key=lambda item: item[1],
                reverse=True,
            )
            species_margin = (
                float(ranked_species[0][1] - ranked_species[1][1])
                if len(ranked_species) > 1
                else 1.0
            )
            minimum_margin = float(self.config.get("species_call_minimum_margin", 0.1))
            species_call = (
                ranked_species[0][0]
                if ranked_species and species_margin >= minimum_margin
                else "indeterminate"
            )
            tissue_rank = sorted(
                ((key, float(np.mean(value))) for key, value in tissue_scores.items()),
                key=lambda item: item[1],
                reverse=True,
            )
            output.append(
                {
                    "sample": sample,
                    "matches": matches,
                    "nearest_unlabelled_references": nearest_unlabelled,
                    "primary_tissue_match": next(
                        (
                            match
                            for match in matches
                            if match["reference_type"] == "healthy_tissue"
                        ),
                        None,
                    ),
                    "nearest_tumor_cohort": next(
                        (
                            match
                            for match in matches
                            if match["reference_type"] == "tumor_cohort"
                        ),
                        None,
                    ),
                    "species_call": species_call,
                    "species_margin": round(species_margin, 6),
                    "species_evidence": [
                        {"label": label, "weight": round(float(weight), 6)}
                        for label, weight in ranked_species
                    ],
                    "tissue_evidence": [
                        {"label": label, "score": round(score, 6)}
                        for label, score in tissue_rank[:5]
                    ],
                }
            )
        return output, len(shared)

    def _gene_evidence(
        self, genes: list[str], expression: np.ndarray, atlas_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        mean_expression = np.mean(np.log1p(np.maximum(expression, 0)), axis=1)
        order = np.argsort(mean_expression)[::-1]
        top_genes = [genes[index] for index in order[: min(100, len(order))]]
        annotations = [
            {"gene": gene, **self.annotations[gene]}
            for gene in top_genes
            if gene in self.annotations
        ][:25]
        top_set = set(top_genes)
        diseases = []
        for disease, members in self.disease_sets.items():
            overlap = sorted(top_set & members)
            if overlap:
                diseases.append(
                    {
                        "disease": disease,
                        "overlap_genes": overlap,
                        "overlap_count": len(overlap),
                        "set_size": len(members),
                        "score": round(len(overlap) / np.sqrt(max(1, len(members))), 6),
                    }
                )
        diseases.sort(key=lambda row: row["score"], reverse=True)
        mappings = []
        if any(
            result.get("species_call", "").lower() == "mouse"
            for result in atlas_results
        ):
            for gene in top_genes:
                mappings.extend(self.orthologs.get(gene, []))
        return {
            "top_expressed_genes": top_genes[:25],
            "gene_annotations": annotations,
            "disease_associations": diseases[:15],
            "mouse_to_human_orthologs": mappings[:50],
        }

    def _language_head(self, evidence: dict[str, Any]) -> dict[str, Any]:
        config = self.config.get("ollama", {})
        if not config.get("enabled", False):
            return {"status": "disabled", "model": config.get("model")}
        language_evidence = dict(evidence)
        sample_limit = int(config.get("max_samples", 16))
        language_evidence["sample_results"] = [
            {**row, "matches": row.get("matches", [])[:5]}
            for row in evidence.get("sample_results", [])[:sample_limit]
        ]
        if len(evidence.get("sample_results", [])) > sample_limit:
            language_evidence["omitted_sample_count"] = (
                len(evidence["sample_results"]) - sample_limit
            )
        compact_annotations = []
        for row in evidence.get("gene_annotations", []):
            ensembl = row.get("ensembl")
            if isinstance(ensembl, list):
                ensembl = ensembl[0] if ensembl else None
            if isinstance(ensembl, dict):
                ensembl = ensembl.get("gene")
            compact_annotations.append(
                {
                    "gene": row.get("gene"),
                    "name": row.get("name"),
                    "summary": str(row.get("summary", ""))[:600],
                    "entrezgene": row.get("entrezgene"),
                    "ensembl_gene": ensembl,
                    "type_of_gene": row.get("type_of_gene"),
                }
            )
        language_evidence["gene_annotations"] = compact_annotations
        prompt = (
            "You are a transcriptomics evidence summarizer. Use only the JSON evidence. "
            "Separate observations from hypotheses. Never call disease associations a "
            "diagnosis or calibrated probability. State atlas coverage and ambiguity. "
            "Use species_call as the species conclusion; if it is indeterminate, say "
            "that species cannot be resolved from this panel. TCGA project matches are "
            "expression-reference similarities, not evidence that the sample has cancer. "
            "For mouse samples, describe only supplied ortholog mappings. There is no "
            "user question to answer and no evidence outside this JSON. Return concise "
            "sections: provenance, tissue, genes/pathways, disease associations, "
            "cross-species mapping, limitations.\nEVIDENCE:\n"
            + json.dumps(language_evidence, separators=(",", ":"))
        )
        body = json.dumps(
            {
                "model": config.get("model", "llama3.1:8b"),
                "prompt": prompt,
                "stream": False,
                "keep_alive": 0,
                "options": {
                    "temperature": 0.1,
                    "seed": 42,
                    "num_ctx": int(config.get("num_ctx", 16384)),
                    "num_predict": int(config.get("num_predict", 900)),
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            config.get("url", "http://127.0.0.1:11434/api/generate"),
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=float(config.get("timeout_seconds", 90))
            ) as response:
                result = json.loads(response.read())
            return {
                "status": "ok",
                "model": config.get("model", "llama3.1:8b"),
                "text": result.get("response", "").strip(),
                "prompt": prompt,
            }
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            return {
                "status": "unavailable",
                "model": config.get("model"),
                "error": str(error),
                "prompt": prompt,
            }

    def analyze(
        self, genes: list[str], samples: list[str], expression: np.ndarray
    ) -> dict[str, Any]:
        self._load()
        matches, shared = self._match(genes, samples, expression)
        gene_evidence = self._gene_evidence(genes, expression, matches)
        evidence = {
            "atlas": self.status(),
            "input": {"samples": len(samples), "genes": len(genes), "shared_genes": shared},
            "sample_results": matches,
            **gene_evidence,
            "limitations": [
                "Reference similarities are evidence scores, not calibrated class probabilities.",
                "Disease results are expression/marker associations, not diagnoses.",
                "Imputed values can amplify model and atlas biases.",
            ],
        }
        if shared < int(self.config.get("minimum_shared_genes", 100)):
            evidence["limitations"].insert(
                0,
                f"Sparse-panel analysis used only {shared} shared genes; tissue, species, "
                "and disease-set evidence is lower confidence than full-profile analysis.",
            )
        return {**evidence, "language_head": self._language_head(evidence)}
