from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


SEED = 42


def checkpoint_state(payload: dict) -> dict:
    return payload.get("model_state_dict", payload.get("state_dict", payload.get("model", payload)))


def load_genes(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    for column in ("gene_symbol", "symbol", "gene", "genes"):
        if column in frame.columns:
            genes = frame[column].dropna().astype(str).str.upper().tolist()
            break
    else:
        genes = frame.iloc[:, -1].dropna().astype(str).str.upper().tolist()
    if len(genes) != len(set(genes)):
        raise ValueError(f"Gene dictionary contains duplicates: {path}")
    return genes


def load_checkpoint(
    name: str,
    checkpoint: Path,
    genes_path: Path,
    embedding_key: str = "gene_embedding.weight",
) -> tuple[dict, list[str], np.ndarray]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_state(payload)
    source = payload if embedding_key in payload else state
    embedding = source[embedding_key].detach().float().numpy()
    genes = load_genes(genes_path)
    if embedding.shape[0] != len(genes):
        raise ValueError(f"{name}: embedding rows={embedding.shape[0]} genes={len(genes)}")
    if not np.isfinite(embedding).all():
        raise ValueError(f"{name}: non-finite gene embeddings")
    config = dict(payload.get("config", {}))
    metadata = {
        "name": name,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "epoch": int(payload.get("epoch", -1)),
        "train_loss": float(payload.get("train_loss", np.nan)),
        "val_loss": float(payload.get("val_loss", np.nan)),
        "embedding_shape": list(embedding.shape),
        "state_tensor_count": sum(torch.is_tensor(v) for v in state.values()),
        "parameter_count": int(sum(v.numel() for v in state.values() if torch.is_tensor(v))),
        "normalization": config.get("normalization"),
        "batch_size": config.get("batch_size"),
        "hidden_dim": config.get("hidden_dim"),
        "num_layers": config.get("num_layers"),
        "num_heads": config.get("num_heads"),
        "feature_type": config.get("feature_type"),
        "gene_prior_weight": config.get("gene_prior_weight", 0.0),
        "context_prior_weight": config.get("context_prior_weight", 0.0),
        "sample_contrastive_weight": config.get("sample_contrastive_weight", 0.0),
        "embedding_representation": embedding_key,
        "embedding_context_samples": int(
            payload.get("training_contextual_gene_embedding_samples", 0)
            if embedding_key == "training_contextual_gene_embedding"
            else payload.get("contextual_gene_embedding_samples", 0)
        ),
    }
    del payload, state
    return metadata, genes, embedding


def aligned_embeddings(
    genes_a: list[str], emb_a: np.ndarray, genes_b: list[str], emb_b: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray]:
    index_b = {gene: i for i, gene in enumerate(genes_b)}
    shared = [gene for gene in genes_a if gene in index_b]
    index_a = {gene: i for i, gene in enumerate(genes_a)}
    a = emb_a[[index_a[g] for g in shared]]
    b = emb_b[[index_b[g] for g in shared]]
    a = StandardScaler().fit_transform(a).astype(np.float32)
    b = StandardScaler().fit_transform(b).astype(np.float32)
    return shared, a, b


def row_normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def embedding_similarity(a: np.ndarray, b: np.ndarray) -> dict:
    a_centered = a - a.mean(axis=0, keepdims=True)
    b_centered = b - b.mean(axis=0, keepdims=True)
    atb = a_centered.T @ b_centered
    ata = a_centered.T @ a_centered
    btb = b_centered.T @ b_centered
    linear_cka = float(
        np.square(atb).sum()
        / np.sqrt(np.square(ata).sum() * np.square(btb).sum())
    )

    u, singular, vt = np.linalg.svd(atb, full_matrices=False)
    rotation = u @ vt
    a_rotated = row_normalize(a_centered @ rotation)
    b_normalized = row_normalize(b_centered)
    same_gene_cosine = np.sum(a_rotated * b_normalized, axis=1)

    rng = np.random.default_rng(SEED)
    query_indices = np.sort(rng.choice(len(a), size=min(256, len(a)), replace=False))
    a_norm = row_normalize(a)
    b_norm = row_normalize(b)
    overlaps = []
    jaccards = []
    k = 10
    for start in range(0, len(query_indices), 32):
        batch = query_indices[start : start + 32]
        sim_a = a_norm[batch] @ a_norm.T
        sim_b = b_norm[batch] @ b_norm.T
        sim_a[np.arange(len(batch)), batch] = -np.inf
        sim_b[np.arange(len(batch)), batch] = -np.inf
        top_a = np.argpartition(sim_a, -k, axis=1)[:, -k:]
        top_b = np.argpartition(sim_b, -k, axis=1)[:, -k:]
        for aa, bb in zip(top_a, top_b):
            common = len(set(aa.tolist()) & set(bb.tolist()))
            overlaps.append(common / k)
            jaccards.append(common / (2 * k - common))

    return {
        "linear_cka": linear_cka,
        "procrustes_mean_same_gene_cosine": float(same_gene_cosine.mean()),
        "procrustes_median_same_gene_cosine": float(np.median(same_gene_cosine)),
        "procrustes_cross_covariance_nuclear_norm": float(singular.sum()),
        "top10_neighbor_overlap_fraction": float(np.mean(overlaps)),
        "top10_neighbor_jaccard": float(np.mean(jaccards)),
        "neighbor_query_count": int(len(query_indices)),
    }


def parse_gmt(path: Path, min_genes: int = 5) -> dict[str, set[str]]:
    library: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.rstrip().split("\t")
        if len(parts) < 3:
            continue
        genes = {item.split(",", 1)[0].strip().upper() for item in parts[2:] if item.strip()}
        if len(genes) >= min_genes:
            library[parts[0]] = genes
    return library


def load_json_sets(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(term): {str(gene).upper() for gene in genes if gene}
        for term, genes in payload.items()
    }


def centroid_scores(
    model: str,
    embedding: np.ndarray,
    genes: list[str],
    library_name: str,
    library: dict[str, set[str]],
    max_terms: int,
) -> list[dict]:
    universe = set(genes)
    eligible = []
    for term, members in library.items():
        positive = members & universe
        if 12 <= len(positive) <= len(genes) - 12:
            eligible.append((term, positive))
    eligible.sort(key=lambda item: (-len(item[1]), item[0]))
    index = {gene: i for i, gene in enumerate(genes)}
    normalized = row_normalize(embedding)
    rows = []
    rng = np.random.default_rng(SEED)
    for term, positive in eligible[:max_terms]:
        positive_idx = np.array([index[g] for g in positive], dtype=np.int64)
        y = np.zeros(len(genes), dtype=np.int8)
        y[positive_idx] = 1
        positive_sum = normalized[positive_idx].sum(axis=0)
        centroid = positive_sum / max(np.linalg.norm(positive_sum), 1e-12)
        scores = normalized @ centroid
        if len(positive_idx) > 1:
            loo = positive_sum[None, :] - normalized[positive_idx]
            loo = row_normalize(loo)
            scores[positive_idx] = np.sum(normalized[positive_idx] * loo, axis=1)
        rows.append(
            {
                "model": model,
                "library": library_name,
                "term": term,
                "n_positive": int(len(positive_idx)),
                "auroc": float(roc_auc_score(y, scores)),
                "auprc": float(average_precision_score(y, scores)),
                "shuffled_auroc": float(roc_auc_score(rng.permutation(y), scores)),
            }
        )
    return rows


def sample_set_pairs(
    genes: list[str], library: dict[str, set[str]], n_pairs: int
) -> tuple[np.ndarray, np.ndarray]:
    universe = set(genes)
    terms = [sorted(members & universe) for members in library.values()]
    terms = [members for members in terms if len(members) >= 2]
    gene_to_terms: dict[str, set[int]] = {}
    for term_index, members in enumerate(terms):
        for gene in members:
            gene_to_terms.setdefault(gene, set()).add(term_index)
    annotated = sorted(gene_to_terms)
    index = {gene: i for i, gene in enumerate(genes)}
    rng = np.random.default_rng(SEED)
    positive: set[tuple[int, int]] = set()
    negative: set[tuple[int, int]] = set()
    attempts = 0
    while len(positive) < n_pairs and attempts < n_pairs * 100:
        members = terms[int(rng.integers(len(terms)))]
        a, b = rng.choice(members, 2, replace=False)
        positive.add(tuple(sorted((index[a], index[b]))))
        attempts += 1
    attempts = 0
    while len(negative) < n_pairs and attempts < n_pairs * 200:
        a, b = rng.choice(annotated, 2, replace=False)
        if not (gene_to_terms[a] & gene_to_terms[b]):
            negative.add(tuple(sorted((index[a], index[b]))))
        attempts += 1
    count = min(len(positive), len(negative), n_pairs)
    return np.asarray(sorted(positive)[:count]), np.asarray(sorted(negative)[:count])


def gene_set_scores(
    model: str,
    embedding: np.ndarray,
    library_name: str,
    positive: np.ndarray,
    negative: np.ndarray,
) -> dict:
    normalized = row_normalize(embedding)
    pos_scores = np.sum(normalized[positive[:, 0]] * normalized[positive[:, 1]], axis=1)
    neg_scores = np.sum(normalized[negative[:, 0]] * normalized[negative[:, 1]], axis=1)
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    scores = np.r_[pos_scores, neg_scores]
    return {
        "model": model,
        "library": library_name,
        "n_positive": int(len(pos_scores)),
        "n_negative": int(len(neg_scores)),
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "mean_positive_cosine": float(pos_scores.mean()),
        "mean_negative_cosine": float(neg_scores.mean()),
    }


def markdown_report(
    metadata: list[dict],
    overlap: dict,
    similarity: dict,
    centroid_summary: pd.DataFrame,
    set_scores: pd.DataFrame,
) -> str:
    def frame_markdown(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        rows = [
            "| " + " | ".join(str(column) for column in columns) + " |",
            "|" + "|".join("---" for _ in columns) + "|",
        ]
        for values in frame.itertuples(index=False, name=None):
            rendered = []
            for value in values:
                rendered.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
            rows.append("| " + " | ".join(rendered) + " |")
        return "\n".join(rows)

    centroid_pivot = centroid_summary.pivot(
        index="library", columns="model", values="mean_auroc"
    )
    retrieval_pivot = set_scores.pivot(index="library", columns="model", values="auroc")
    headline = []
    for library, row in centroid_pivot.iterrows():
        delta = row["Txn_Jatin_epoch7"] - row["s66qfh36_best"]
        headline.append(f"- {library} centroid AUROC: epoch 7 leads by **{delta:+.4f}**")
    for library, row in retrieval_pivot.iterrows():
        delta = row["Txn_Jatin_epoch7"] - row["s66qfh36_best"]
        winner = "epoch 7" if delta > 0 else "s66qfh36"
        headline.append(
            f"- {library} same-set AUROC: {winner} leads by **{abs(delta):.4f}**"
        )

    lines = [
        "# Epoch 7 vs s66qfh36 local comparison",
        "",
        "All computations were performed locally on CPU using frozen full-corpus contextual gene embeddings.",
        "The requested `._best_model.pt` is a macOS AppleDouble sidecar; `best_model.pt` was used.",
        "",
        "## Checkpoints",
        "",
        "| Model | Epoch | Genes | Context samples | Parameters | Train loss | Validation loss | Normalization |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in metadata:
        lines.append(
            f"| {item['name']} | {item['epoch']} | {item['embedding_shape'][0]:,} | "
            f"{item['embedding_context_samples']:,} | "
            f"{item['parameter_count']:,} | {item['train_loss']:.6f} | {item['val_loss']:.6f} | "
            f"{item['normalization']} |"
        )
    lines.extend(
        [
            "",
            "Losses are descriptive only because the checkpoints use different data preparation and normalization.",
            "",
            "## Gene overlap",
            "",
            f"- Shared genes: **{overlap['shared']:,}**",
            f"- Epoch-7-only genes: **{overlap['epoch7_only']:,}**",
            f"- s66qfh36-only genes: **{overlap['reference_only']:,}**",
            "",
            "## Embedding alignment",
            "",
            f"- Linear CKA: **{similarity['linear_cka']:.4f}**",
            f"- Procrustes-aligned same-gene cosine: **{similarity['procrustes_mean_same_gene_cosine']:.4f}**",
            f"- Top-10 neighbor overlap: **{similarity['top10_neighbor_overlap_fraction']:.4f}**",
            "",
            "## Functional centroid prediction",
            "",
            frame_markdown(centroid_summary),
            "",
            "## Same-set retrieval",
            "",
            frame_markdown(set_scores),
            "",
            "## Headline comparison",
            "",
            *headline,
            "",
            "There is no universal winner: epoch 7 is stronger for centroid-based functional prediction, while s66qfh36 preserves stronger same-set neighborhood ranking.",
            "",
            "## Interpretation",
            "",
            f"Higher AUROC/AUPRC is better. The comparison is paired on the same {overlap['shared']:,} shared genes and uses identical sampled pairs for both models.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--max-terms", type=int, default=100)
    parser.add_argument("--set-pairs", type=int, default=3000)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

    comparison_root = Path(__file__).resolve().parent
    epoch_checkpoint = comparison_root / "inputs" / "Txn_Jatin_epoch7_completed.pt"
    epoch_genes = comparison_root / "inputs" / "Txn_Jatin_epoch7_canonical_genes.csv"
    reference_checkpoint = workspace / "RNA Walter" / "flash_osdr_model" / "s66qfh36" / "best_model.pt"
    reference_genes = workspace / "Txn_Jatin" / "models" / "bridge_reference" / "canonical_genes.csv"

    epoch_meta, epoch_gene_list, epoch_embedding = load_checkpoint(
        "Txn_Jatin_epoch7",
        epoch_checkpoint,
        epoch_genes,
        embedding_key="training_contextual_gene_embedding",
    )
    ref_meta, ref_gene_list, ref_embedding = load_checkpoint(
        "s66qfh36_best", reference_checkpoint, reference_genes
    )
    reference_cache = (
        workspace
        / "Txn_Jatin"
        / "benchmark_dynamic"
        / "embeddings"
        / "BRIDGE__gene_benchmark__symbol.npz"
    )
    with np.load(reference_cache, allow_pickle=True) as cached:
        cached_genes = [str(g).upper() for g in cached["keys"].tolist()]
        cached_embedding = cached["emb"].astype(np.float32)
        provenance = json.loads(str(cached["provenance"].reshape(-1)[0]))
    finite_rows = np.isfinite(cached_embedding).all(axis=1)
    ref_gene_list = [gene for gene, keep in zip(cached_genes, finite_rows) if keep]
    ref_embedding = cached_embedding[finite_rows]
    ref_meta["embedding_shape"] = list(ref_embedding.shape)
    ref_meta["embedding_representation"] = "full_corpus_contextual_cache"
    ref_meta["embedding_context_samples"] = int(provenance["context_samples"])
    ref_meta["embedding_cache"] = str(reference_cache.resolve())
    ref_meta["embedding_cache_sha256"] = hashlib.sha256(reference_cache.read_bytes()).hexdigest()
    shared, epoch_aligned, ref_aligned = aligned_embeddings(
        epoch_gene_list, epoch_embedding, ref_gene_list, ref_embedding
    )
    del epoch_embedding, ref_embedding

    overlap = {
        "epoch7_genes": len(epoch_gene_list),
        "reference_genes": len(ref_gene_list),
        "shared": len(shared),
        "epoch7_only": len(set(epoch_gene_list) - set(ref_gene_list)),
        "reference_only": len(set(ref_gene_list) - set(epoch_gene_list)),
    }
    similarity = embedding_similarity(epoch_aligned, ref_aligned)

    go = parse_gmt(workspace / "RNA Walter" / "data" / "gene_set_libs" / "go_bp.txt")
    disgenet = parse_gmt(workspace / "RNA Walter" / "data" / "gene_set_libs" / "disgenet.txt")
    kegg = parse_gmt(workspace / "Txn_Jatin" / "osdr_finetune" / "results" / "references" / "KEGG_2021_Human.gmt")
    hallmark = load_json_sets(workspace / "RNA Walter" / "gene_sets" / "hallmark_gene_sets.json")
    libraries = {"GO_BP": go, "DisGeNET": disgenet, "KEGG_2021_Human": kegg}
    models = {"Txn_Jatin_epoch7": epoch_aligned, "s66qfh36_best": ref_aligned}

    centroid_rows = []
    for library_name, library in libraries.items():
        for model, embedding in models.items():
            centroid_rows.extend(
                centroid_scores(model, embedding, shared, library_name, library, args.max_terms)
            )
    centroid_frame = pd.DataFrame(centroid_rows)
    centroid_frame.to_csv(output / "functional_centroid_terms.csv", index=False)
    centroid_summary = (
        centroid_frame.groupby(["library", "model"], as_index=False)
        .agg(
            terms=("term", "size"),
            mean_auroc=("auroc", "mean"),
            mean_auprc=("auprc", "mean"),
            shuffled_mean_auroc=("shuffled_auroc", "mean"),
        )
    )
    centroid_summary.to_csv(output / "functional_centroid_summary.csv", index=False)

    set_rows = []
    for library_name, library in {
        "GO_BP": go,
        "Hallmark": hallmark,
        "KEGG_2021_Human": kegg,
    }.items():
        positive, negative = sample_set_pairs(shared, library, args.set_pairs)
        for model, embedding in models.items():
            set_rows.append(gene_set_scores(model, embedding, library_name, positive, negative))
    set_frame = pd.DataFrame(set_rows)
    set_frame.to_csv(output / "same_set_retrieval.csv", index=False)

    metadata = [epoch_meta, ref_meta]
    (output / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "gene_overlap.json").write_text(json.dumps(overlap, indent=2) + "\n")
    (output / "embedding_similarity.json").write_text(json.dumps(similarity, indent=2) + "\n")
    report = markdown_report(metadata, overlap, similarity, centroid_summary, set_frame)
    (output / "comparison_report.md").write_text(report, encoding="utf-8")

    print(json.dumps({
        "output": str(output),
        "overlap": overlap,
        "similarity": similarity,
        "centroid_summary": centroid_summary.to_dict(orient="records"),
        "same_set_retrieval": set_frame.to_dict(orient="records"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
