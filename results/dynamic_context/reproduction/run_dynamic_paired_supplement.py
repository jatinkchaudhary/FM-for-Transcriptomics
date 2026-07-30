from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = Path(os.environ["MYTHOS_OUTPUT_ROOT"]).resolve()
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
EMBEDDINGS = OUTPUT / "embeddings"
TRRUST = Path(
    os.environ.get(
        "TXN_TRRUST_TSV",
        ROOT / "RNA Walter" / "data" / "gene_set_libs" / "trrust.tsv",
    )
)


def load_embeddings():
    gene_emb = {}
    master = None
    provenance = {}
    for path in sorted(EMBEDDINGS.glob("*__gene_benchmark__symbol.npz")):
        model = path.name.split("__gene_benchmark__", 1)[0]
        with np.load(path, allow_pickle=True) as payload:
            keys = payload["keys"].astype(str).tolist()
            emb = payload["emb"].astype(np.float32)
            fallback = bool(payload["is_fallback"])
            prov_raw = payload.get("provenance")
            if prov_raw is not None and len(prov_raw):
                provenance[model] = json.loads(str(prov_raw[0]))
        if master is None:
            master = keys
        elif master != keys:
            raise RuntimeError(f"embedding key mismatch for {model}")
        gene_emb[model] = (emb, fallback)
    if len(gene_emb) != 9:
        raise RuntimeError(f"expected 9 exported models, found {sorted(gene_emb)}")
    return master, gene_emb, provenance


def load_pairs():
    if not TRRUST.exists():
        raise FileNotFoundError(TRRUST)
    pairs = []
    for line in TRRUST.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            pairs.append((fields[0].upper(), fields[1].upper()))
    return pairs


def main() -> int:
    os.environ.setdefault("MYTHOS_BENCH_ROOT", str(ROOT))
    os.environ.setdefault("MYTHOS_OUTPUT_ROOT", str(OUTPUT))
    os.environ.setdefault("MYTHOS_PROBE", "torch")
    os.environ.setdefault("MYTHOS_DEVICE", "cuda")
    os.environ.setdefault("MYTHOS_THREADS", "32")
    os.environ.setdefault("MYTHOS_CV_FOLDS", "5")
    os.environ.setdefault("MYTHOS_MAX_PAIRS", "200000")
    os.environ.setdefault("MPLBACKEND", "Agg")

    from mythos.common import Council
    from mythos import adapters as A
    from mythos import gene_bench as GB
    from mythos import leaderboard as LB

    master, gene_emb, provenance = load_embeddings()
    common_genes, _ = A.harmonised_gene_sets(gene_emb, master)
    pairs = load_pairs()
    council = Council()
    frames = []
    for variant in ("common", "full"):
        scores = GB.paired_genes(
            gene_emb,
            master,
            pairs,
            name="TRRUST-TF",
            variant=variant,
            common_genes=common_genes if variant == "common" else None,
            council=council,
        )
        if len(scores):
            scores["track"] = "paired"
            frames.append(scores)
    paired = pd.concat(frames, ignore_index=True, sort=False)
    paired.to_csv(TABLES / "paired_scores.csv", index=False)
    (TABLES / "trrust.tsv").write_text(TRRUST.read_text())

    gene_scores = pd.read_csv(TABLES / "gene_scores.csv")
    if "track" in gene_scores:
        gene_scores = gene_scores[gene_scores["track"] != "paired"]
    combined = pd.concat([gene_scores, paired], ignore_index=True, sort=False)
    combined.to_csv(TABLES / "gene_scores.csv", index=False)

    LB.leaderboard_table(combined)
    LB.plot_leaderboard(combined, out=FIGURES / "leaderboard.png")
    LB.performance_correlation(
        combined, council=council, out=FIGURES / "performance_correlation.png"
    )
    sample_path = TABLES / "sample_scores.csv"
    sample = pd.read_csv(sample_path) if sample_path.exists() else pd.DataFrame()
    LB.summary_table(combined, sample)

    summary = {
        "pairs_input": len(pairs),
        "common_genes": len(common_genes),
        "rows": len(paired),
        "models": sorted(gene_emb),
        "embedding_provenance": provenance,
        "mean_auroc": (
            paired.groupby(["variant", "model", "op"], as_index=False)["auroc"]
            .mean()
            .round(6)
            .to_dict(orient="records")
        ),
    }
    (TABLES / "paired_supplement_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

