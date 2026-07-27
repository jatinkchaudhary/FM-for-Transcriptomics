"""Input-derived sample retrieval, projection, and interpretation helpers."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _padded_coordinates(values: np.ndarray, dimensions: int = 3) -> np.ndarray:
    coordinates = np.asarray(values, dtype=np.float64)
    if coordinates.ndim != 2:
        raise ValueError("projection coordinates must be a matrix")
    if coordinates.shape[1] >= dimensions:
        return coordinates[:, :dimensions]
    padding = np.zeros(
        (coordinates.shape[0], dimensions - coordinates.shape[1]),
        dtype=np.float64,
    )
    return np.concatenate([coordinates, padding], axis=1)


def _coordinates_payload(
    coordinates_2d: np.ndarray,
    coordinates_3d: np.ndarray,
    *,
    status: str = "ok",
    parameters: dict[str, Any] | None = None,
    quality_2d: float | None = None,
    quality_3d: float | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "coordinates_2d": np.round(
            _padded_coordinates(coordinates_2d, 2), 6
        ).tolist(),
        "coordinates_3d": np.round(
            _padded_coordinates(coordinates_3d, 3), 6
        ).tolist(),
        "trustworthiness_2d": quality_2d,
        "trustworthiness_3d": quality_3d,
        "parameters": parameters or {},
    }


def _projection_quality(
    source: np.ndarray, coordinates: np.ndarray, neighbors: int
) -> float | None:
    if len(source) < 4:
        return None
    try:
        from sklearn.manifold import trustworthiness

        score = trustworthiness(
            source,
            coordinates,
            n_neighbors=min(neighbors, max(1, (len(source) - 1) // 2)),
            metric="cosine",
        )
        return round(float(score), 6)
    except Exception:
        return None


def build_projections(
    embeddings: np.ndarray,
    methods: Iterable[str] = ("pca", "umap", "tsne"),
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Build deterministic, request-scoped maps from sample embeddings."""

    from sklearn.decomposition import PCA

    source = np.asarray(embeddings, dtype=np.float32)
    sample_count, feature_count = source.shape
    requested = set(methods)
    warnings: list[str] = []
    projections: dict[str, dict[str, Any]] = {}
    components = max(1, min(3, sample_count, feature_count))
    pca = PCA(n_components=components, svd_solver="full")
    pca_coordinates = _padded_coordinates(pca.fit_transform(source), 3)
    explained = np.zeros(3, dtype=np.float64)
    explained[: len(pca.explained_variance_ratio_)] = pca.explained_variance_ratio_
    if "pca" in requested:
        projections["pca"] = _coordinates_payload(
            pca_coordinates[:, :2],
            pca_coordinates,
            parameters={
                "fit_scope": "current input",
                "components": components,
                "explained_variance_ratio": np.round(explained, 6).tolist(),
            },
            quality_2d=_projection_quality(source, pca_coordinates[:, :2], 5),
            quality_3d=_projection_quality(source, pca_coordinates, 5),
        )

    if "umap" in requested:
        try:
            import umap

            neighbors = min(15, max(2, sample_count - 1))
            if sample_count < 4:
                raise ValueError("UMAP requires at least four samples")
            umap_2d = umap.UMAP(
                n_components=2,
                n_neighbors=neighbors,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
                n_jobs=1,
            ).fit_transform(source)
            umap_3d = umap.UMAP(
                n_components=3,
                n_neighbors=neighbors,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
                n_jobs=1,
            ).fit_transform(source)
            projections["umap"] = _coordinates_payload(
                umap_2d,
                umap_3d,
                parameters={
                    "fit_scope": "current input",
                    "n_neighbors": neighbors,
                    "min_dist": 0.1,
                    "metric": "cosine",
                    "random_state": 42,
                },
                quality_2d=_projection_quality(source, umap_2d, 5),
                quality_3d=_projection_quality(source, umap_3d, 5),
            )
        except Exception as error:
            warnings.append(f"UMAP unavailable for this request: {error}")
            projections["umap"] = _coordinates_payload(
                pca_coordinates[:, :2],
                pca_coordinates,
                status="fallback_to_pca",
                parameters={"reason": str(error), "fit_scope": "current input"},
            )

    if "tsne" in requested:
        try:
            from sklearn.manifold import TSNE

            if sample_count < 4:
                raise ValueError("t-SNE requires at least four samples")
            perplexity = min(30.0, max(2.0, (sample_count - 1) / 3.0))
            tsne_2d = TSNE(
                n_components=2,
                perplexity=perplexity,
                metric="cosine",
                init="random",
                learning_rate="auto",
                max_iter=750,
                random_state=42,
            ).fit_transform(source)
            tsne_3d = TSNE(
                n_components=3,
                perplexity=perplexity,
                metric="cosine",
                init="random",
                learning_rate="auto",
                max_iter=750,
                random_state=42,
            ).fit_transform(source)
            projections["tsne"] = _coordinates_payload(
                tsne_2d,
                tsne_3d,
                parameters={
                    "fit_scope": "current input",
                    "perplexity": round(perplexity, 3),
                    "metric": "cosine",
                    "iterations": 750,
                    "random_state": 42,
                },
                quality_2d=_projection_quality(source, tsne_2d, 5),
                quality_3d=_projection_quality(source, tsne_3d, 5),
            )
        except Exception as error:
            warnings.append(f"t-SNE unavailable for this request: {error}")
            projections["tsne"] = _coordinates_payload(
                pca_coordinates[:, :2],
                pca_coordinates,
                status="fallback_to_pca",
                parameters={"reason": str(error), "fit_scope": "current input"},
            )

    return projections, warnings


def _sample_statistics(
    genes: list[str],
    samples: list[str],
    groups: list[str],
    expression: np.ndarray,
) -> list[dict[str, Any]]:
    statistics = []
    safe_expression = np.maximum(np.asarray(expression, dtype=np.float64), 0.0)
    for sample_index, sample in enumerate(samples):
        column = safe_expression[:, sample_index]
        order = np.argsort(column)[::-1][:8]
        total = float(column.sum())
        top_ten = float(np.sort(column)[-min(10, len(column)) :].sum())
        statistics.append(
            {
                "sample": sample,
                "group": groups[sample_index],
                "detected_genes": int(np.count_nonzero(column > 0)),
                "expression_sum": round(total, 6),
                "top10_concentration": round(top_ten / total, 6) if total else 0.0,
                "top_genes": [
                    {"gene": genes[index], "value": round(float(column[index]), 6)}
                    for index in order
                ],
            }
        )
    return statistics


def _group_metrics(similarity: np.ndarray, groups: list[str]) -> dict[str, Any]:
    labels = np.asarray(groups, dtype=object)
    named = np.asarray(
        [bool(value) and value.lower() not in {"unlabeled", "unknown", "-"} for value in groups]
    )
    unique = sorted(set(labels[named]))
    if len(unique) < 2:
        return {
            "groups": unique,
            "nearest_neighbor_group_agreement": None,
            "within_group_mean_cosine": None,
            "between_group_mean_cosine": None,
        }
    nearest = np.argmax(
        np.where(np.eye(len(similarity), dtype=bool), -np.inf, similarity),
        axis=1,
    )
    eligible = np.flatnonzero(named)
    agreement = np.mean(labels[eligible] == labels[nearest[eligible]])
    upper = np.triu_indices(len(similarity), 1)
    pair_valid = named[upper[0]] & named[upper[1]]
    same = labels[upper[0]] == labels[upper[1]]
    values = similarity[upper]
    within = values[pair_valid & same]
    between = values[pair_valid & ~same]
    return {
        "groups": unique,
        "nearest_neighbor_group_agreement": round(float(agreement), 6),
        "within_group_mean_cosine": (
            round(float(within.mean()), 6) if within.size else None
        ),
        "between_group_mean_cosine": (
            round(float(between.mean()), 6) if between.size else None
        ),
    }


def build_live_analysis(
    *,
    model: str,
    embedding_mode: str,
    genes: list[str],
    samples: list[str],
    groups: list[str],
    expression: np.ndarray,
    embeddings: np.ndarray,
    matched_genes: int,
    model_gene_count: int | None,
    methods: Iterable[str] = ("pca", "umap", "tsne"),
    initial_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Return only values derived from the current request."""

    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    projections, projection_warnings = build_projections(normalized, methods)
    group_metrics = _group_metrics(similarity, groups)
    statistics = _sample_statistics(genes, samples, groups, expression)
    nearest = []
    for sample_index, sample in enumerate(samples):
        order = np.argsort(similarity[sample_index])[::-1]
        order = order[order != sample_index][: min(20, len(samples) - 1)]
        nearest.append(
            {
                "sample": sample,
                "neighbors": [
                    {
                        "sample": samples[index],
                        "group": groups[index],
                        "cosine": round(float(similarity[sample_index, index]), 6),
                        "rank": rank + 1,
                    }
                    for rank, index in enumerate(order)
                ],
            }
        )
    upper = similarity[np.triu_indices(len(samples), 1)]
    mean_similarity = float(upper.mean()) if upper.size else 1.0
    median_similarity = float(np.median(upper)) if upper.size else 1.0
    coverage = (
        matched_genes / model_gene_count
        if model_gene_count
        else matched_genes / max(1, len(genes))
    )
    warnings = list(initial_warnings or []) + projection_warnings
    if model_gene_count and coverage < 0.5:
        warnings.append(
            f"Only {matched_genes:,}/{model_gene_count:,} model genes were supplied; "
            "sample embeddings may be dominated by the unobserved-gene baseline."
        )
    reading = [
        (
            f"The current input contains {len(samples):,} samples and {len(genes):,} "
            f"submitted genes. Results were recomputed in {embedding_mode} space."
        ),
        (
            f"Across all distinct sample pairs, mean cosine similarity is "
            f"{mean_similarity:.4f} and the median is {median_similarity:.4f}."
        ),
    ]
    if group_metrics["nearest_neighbor_group_agreement"] is not None:
        reading.append(
            "The nearest-neighbor group agreement is "
            f"{100 * group_metrics['nearest_neighbor_group_agreement']:.1f}% "
            "for samples with submitted group labels."
        )
    reading.append(
        "Retrieval ranks are exact within this uploaded cohort. They are not a search "
        "against the upstream 940,455-sample ARCHS4 index."
    )
    return {
        "source": "current_request",
        "scope": "uploaded cohort",
        "model": model,
        "embedding_mode": embedding_mode,
        "genes": len(genes),
        "samples": samples,
        "groups": groups,
        "sample_count": len(samples),
        "matched_genes": matched_genes,
        "model_gene_count": model_gene_count,
        "coverage_fraction": round(float(coverage), 8),
        "similarity": np.round(similarity, 6).tolist(),
        "nearest_neighbors": nearest,
        "projections": projections,
        "sample_statistics": statistics,
        "group_metrics": group_metrics,
        "pairwise_mean_cosine": round(mean_similarity, 6),
        "pairwise_median_cosine": round(median_similarity, 6),
        "reading": reading,
        "warnings": warnings,
        "reference_index": {
            "available": False,
            "scope": "current input only",
            "reason": (
                "The upstream BRIDGE ARCHS4 memmap is not distributed by its "
                "repository and cannot be mixed across model embedding spaces."
            ),
        },
    }
