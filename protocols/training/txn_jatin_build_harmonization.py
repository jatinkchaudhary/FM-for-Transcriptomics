from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


HUMAN_GTF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
    "release_49/gencode.v49.annotation.gtf.gz"
)
MOUSE_GTF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/"
    "release_M38/gencode.vM38.annotation.gtf.gz"
)
GENE_NAME_RE = re.compile(r'gene_name "([^"]+)"')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 1024:
        print(f"[download] reuse {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "10",
            "-C",
            "-",
            "-o",
            str(partial),
            url,
        ],
        check=True,
    )
    partial.replace(path)


def load_one_to_one_mouse_map(path: Path) -> dict[str, str]:
    hom = pd.read_csv(path, sep="\t")
    human_by_class: dict[object, set[str]] = defaultdict(set)
    mouse_by_class: dict[object, set[str]] = defaultdict(set)
    for _, row in hom.iterrows():
        key = row["DB Class Key"]
        organism = str(row["Common Organism Name"]).strip().lower()
        symbol = str(row["Symbol"]).strip()
        if not symbol or symbol.lower() == "nan":
            continue
        if organism == "human":
            human_by_class[key].add(symbol)
        elif organism == "mouse, laboratory":
            mouse_by_class[key].add(symbol)

    mouse_to_human = {}
    for key in human_by_class.keys() & mouse_by_class.keys():
        humans = human_by_class[key]
        mice = mouse_by_class[key]
        if len(humans) == 1 and len(mice) == 1:
            mouse_to_human[next(iter(mice)).upper()] = next(iter(humans)).upper()
    return mouse_to_human


def parse_exon_union_lengths(
    gtf_path: Path,
    canonical: set[str],
    symbol_map: dict[str, str] | None = None,
) -> dict[str, int]:
    intervals: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    with gzip.open(gtf_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "exon":
                continue
            match = GENE_NAME_RE.search(fields[8])
            if match is None:
                continue
            source_symbol = match.group(1).upper()
            symbol = symbol_map.get(source_symbol) if symbol_map is not None else source_symbol
            if symbol not in canonical:
                continue
            intervals[symbol].append((fields[0], int(fields[3]), int(fields[4])))

    lengths: dict[str, int] = {}
    for symbol, gene_intervals in intervals.items():
        total = 0
        current_chrom = None
        current_start = None
        current_end = None
        for chrom, start, end in sorted(gene_intervals):
            if chrom != current_chrom or current_end is None or start > current_end + 1:
                if current_end is not None:
                    total += current_end - current_start + 1
                current_chrom, current_start, current_end = chrom, start, end
            else:
                current_end = max(current_end, end)
        if current_end is not None:
            total += current_end - current_start + 1
        if total > 0:
            lengths[symbol] = total
    return lengths


def build_prior(
    genes: list[str],
    info_path: Path,
    esm_path: Path,
    hidden_dim: int,
    output_path: Path,
    seed: int,
) -> dict:
    info = pd.read_csv(info_path)
    esm = torch.load(esm_path, map_location="cpu", weights_only=False)
    if not isinstance(esm, torch.Tensor) or esm.ndim != 2:
        raise ValueError(f"expected a 2D ESM tensor, got {type(esm)}")
    if len(info) != esm.shape[0]:
        raise ValueError(f"gene info rows ({len(info)}) != ESM rows ({esm.shape[0]})")

    features = esm.float().numpy()
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-8)
    print(
        f"[prior] fitting randomized PCA: {features.shape[0]} x "
        f"{features.shape[1]} -> {hidden_dim}",
        flush=True,
    )
    pca = PCA(
        n_components=hidden_dim,
        svd_solver="randomized",
        random_state=seed,
        iterated_power=4,
    )
    reduced = pca.fit_transform(features).astype(np.float32)
    reduced /= np.maximum(np.linalg.norm(reduced, axis=1, keepdims=True), 1e-8)

    symbol_to_row = {
        str(symbol).upper(): i
        for i, symbol in enumerate(info["gene_symbol"].astype(str))
    }
    aligned = np.zeros((len(genes), hidden_dim), dtype=np.float32)
    covered = np.zeros(len(genes), dtype=bool)
    for i, gene in enumerate(genes):
        row = symbol_to_row.get(gene.upper())
        if row is not None:
            aligned[i] = reduced[row]
            covered[i] = True

    np.savez_compressed(
        output_path,
        embeddings=aligned,
        covered=covered,
        genes=np.asarray(genes, dtype=str),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "source_path": str(esm_path),
        "source_sha256": sha256_file(esm_path),
        "source_gene_info": str(info_path),
        "hidden_dim": hidden_dim,
        "covered_genes": int(covered.sum()),
        "total_genes": len(genes),
        "coverage": float(covered.mean()),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    try:
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Txn_Jatin v2 harmonized full-data view and ESM2 prior."
    )
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--esm-features", required=True)
    parser.add_argument("--bulkformer-gene-info", default=None)
    parser.add_argument("--human-gtf-url", default=HUMAN_GTF_URL)
    parser.add_argument("--mouse-gtf-url", default=MOUSE_GTF_URL)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    resource_dir = Path(args.resource_dir).resolve()
    marker = output_dir / "HARMONIZED_FULL_DATA.ok"
    if marker.exists() and not args.force:
        print(marker.read_text(), flush=True)
        return 0

    canonical_df = pd.read_csv(source_dir / "canonical_genes.csv")
    genes = canonical_df["gene_symbol"].astype(str).tolist()
    canonical = {gene.upper() for gene in genes}
    mouse_to_human = load_one_to_one_mouse_map(
        root / "RNA Walter" / "data" / "HOM_MouseHumanSequence.rpt"
    )

    human_gtf = resource_dir / "gencode.v49.annotation.gtf.gz"
    mouse_gtf = resource_dir / "gencode.vM38.annotation.gtf.gz"
    download(args.human_gtf_url, human_gtf)
    download(args.mouse_gtf_url, mouse_gtf)

    print("[lengths] parsing human exon unions", flush=True)
    human_lengths = parse_exon_union_lengths(human_gtf, canonical)
    print("[lengths] parsing mouse exon unions and mapping to human symbols", flush=True)
    mouse_lengths = parse_exon_union_lengths(mouse_gtf, canonical, mouse_to_human)

    bulkformer_info = Path(args.bulkformer_gene_info).resolve() if args.bulkformer_gene_info else (
        root / "external_models" / "BulkFormer" / "data" / "bulkformer_gene_info.csv"
    )
    info = pd.read_csv(bulkformer_info)
    fallback_human = {
        str(row.gene_symbol).upper(): int(row.gene_length)
        for row in info.itertuples(index=False)
        if pd.notna(row.gene_length) and int(row.gene_length) > 0
    }

    length_rows = []
    valid_genes = []
    fallback_counts = defaultdict(int)
    for gene in genes:
        key = gene.upper()
        human = human_lengths.get(key)
        mouse = mouse_lengths.get(key)
        human_source = "gencode_v49_exon_union"
        mouse_source = "gencode_vM38_exon_union"
        if not human:
            human = fallback_human.get(key)
            human_source = "bulkformer_gene_span_fallback"
            fallback_counts[human_source] += 1
        if not mouse and human:
            mouse = human
            mouse_source = "human_length_fallback"
            fallback_counts[mouse_source] += 1
        if not human or not mouse:
            continue
        valid_genes.append(gene)
        length_rows.append(
            {
                "gene_symbol": gene,
                "human_length_bp": int(human),
                "mouse_length_bp": int(mouse),
                "human_source": human_source,
                "mouse_source": mouse_source,
            }
        )

    if len(valid_genes) < 10000:
        raise RuntimeError(f"only {len(valid_genes)} genes have usable lengths")
    dropped = sorted(canonical - {gene.upper() for gene in valid_genes})

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    link_or_copy(source_dir / "batch_files", output_dir / "batch_files")
    for name in ("metadata.csv", "samples.json", "batch_manifest.json"):
        source = source_dir / name
        if source.exists():
            link_or_copy(source, output_dir / name)

    pd.DataFrame(
        {"token_id": np.arange(1, len(valid_genes) + 1), "gene_symbol": valid_genes}
    ).to_csv(output_dir / "canonical_genes.csv", index=False)
    (output_dir / "genes.json").write_text(json.dumps(valid_genes))
    lengths_df = pd.DataFrame(length_rows)
    lengths_df.to_csv(output_dir / "gene_lengths.csv", index=False)
    np.savez_compressed(
        output_dir / "gene_lengths.npz",
        genes=np.asarray(valid_genes, dtype=str),
        human=lengths_df["human_length_bp"].to_numpy(np.float32),
        mouse=lengths_df["mouse_length_bp"].to_numpy(np.float32),
    )

    prior = build_prior(
        valid_genes,
        bulkformer_info,
        Path(args.esm_features).resolve(),
        args.hidden_dim,
        output_dir / f"esm2_prior_{args.hidden_dim}d.npz",
        args.seed,
    )

    manifest = {
        "version": 2,
        "source_dir": str(source_dir),
        "sample_count": len(json.loads((source_dir / "samples.json").read_text())),
        "source_gene_count": len(genes),
        "gene_count": len(valid_genes),
        "dropped_genes": dropped,
        "normalization": {
            "name": "counts_to_log1p_tpm",
            "formula": "log1p((count/(exon_union_bp/1000))/sum(count/(exon_union_bp/1000))*1e6)",
            "human_gtf_url": args.human_gtf_url,
            "mouse_gtf_url": args.mouse_gtf_url,
            "human_gtf_sha256": sha256_file(human_gtf),
            "mouse_gtf_sha256": sha256_file(mouse_gtf),
            "length_fallback_counts": dict(fallback_counts),
        },
        "orthologs": {
            "mapping": "MGI one-to-one Mouse/Human homology classes",
            "path": str(root / "RNA Walter" / "data" / "HOM_MouseHumanSequence.rpt"),
        },
        "prior": prior,
    }
    (output_dir / "normalization_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    marker.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
