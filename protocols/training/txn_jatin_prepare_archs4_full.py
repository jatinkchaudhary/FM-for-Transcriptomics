from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ARCHS4_URLS = {
    "human": [
        "https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.6.h5",
        "https://s3.dev.maayanlab.cloud/releases/archs4/human_gene_v2.5.h5",
    ],
    "mouse": [
        "https://s3.dev.maayanlab.cloud/archs4/files/mouse_gene_v2.6.h5",
        "https://s3.dev.maayanlab.cloud/releases/archs4/mouse_gene_v2.5.h5",
    ],
}


def decode_array(values):
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8", errors="replace"))
        else:
            out.append(str(value))
    return np.asarray(out, dtype=object)


def run(cmd):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def write_count_parquet(df: pd.DataFrame, path: Path):
    compression_level = int(os.environ.get("TXN_PARQUET_COMPRESSION_LEVEL", "19"))
    df.to_parquet(path, compression="zstd", compression_level=compression_level)


def _head_content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as response:
        return int(response.headers["Content-Length"])


def _download_range(url: str, start: int, end: int, dest: Path, retries: int = 5):
    expected = end - start + 1
    if dest.exists() and dest.stat().st_size == expected:
        return dest
    if dest.exists():
        dest.unlink()
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            existing = tmp.stat().st_size if tmp.exists() else 0
            if existing > expected:
                tmp.unlink()
                existing = 0
            if existing == expected:
                tmp.replace(dest)
                return dest
            range_start = start + existing
            remaining = expected - existing
            timeout = int(os.environ.get("TXN_DOWNLOAD_TIMEOUT", "900"))
            req = urllib.request.Request(url, headers={"Range": f"bytes={range_start}-{end}"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                if status != 206:
                    raise IOError(f"server returned HTTP {status}, expected 206 Partial Content")
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) != remaining:
                    raise IOError(f"range content length {content_length} != expected {remaining}")
                with open(tmp, "ab") as out:
                    shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
            if tmp.stat().st_size != expected:
                raise IOError(f"part size {tmp.stat().st_size} != expected {expected}")
            tmp.replace(dest)
            return dest
        except Exception as exc:
            print(f"[download] range {start}-{end} attempt {attempt}/{retries} failed: {exc}", flush=True)
            if attempt == retries:
                raise
            time.sleep(min(60, 5 * attempt))
    return dest


def parallel_download(url: str, dest: Path, parts: int):
    size = _head_content_length(url)
    part_dir = dest.parent / (dest.name + ".parts")
    spans = []
    chunk = (size + parts - 1) // parts
    for i in range(parts):
        start = i * chunk
        if start >= size:
            break
        end = min(size - 1, start + chunk - 1)
        spans.append((i, start, end, part_dir / f"part_{i:04d}"))
    manifest = "\n".join(
        [url, str(size), str(parts)]
        + [f"{i},{start},{end}" for i, start, end, _ in spans]
    )
    manifest_path = part_dir / "manifest.txt"
    if part_dir.exists() and (not manifest_path.exists() or manifest_path.read_text() != manifest):
        print(f"[download] clearing stale range parts: {part_dir}", flush=True)
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest)

    print(f"[download] parallel ranges: {url} -> {dest} ({size / 1024**3:.1f} GiB, {len(spans)} parts)", flush=True)
    retries = int(os.environ.get("TXN_DOWNLOAD_RETRIES", "10"))
    with ThreadPoolExecutor(max_workers=max(1, parts)) as pool:
        futures = {
            pool.submit(_download_range, url, start, end, part, retries): (i, start, end, part)
            for i, start, end, part in spans
        }
        done = 0
        for future in as_completed(futures):
            i, start, end, part = futures[future]
            future.result()
            done += 1
            print(f"[download] part {i + 1}/{len(spans)} complete ({done}/{len(spans)})", flush=True)

    tmp_dest = dest.with_suffix(dest.suffix + ".partial")
    tmp_dest.unlink(missing_ok=True)
    with open(tmp_dest, "wb") as out:
        for i, _, _, part in spans:
            with open(part, "rb") as inp:
                shutil.copyfileobj(inp, out, length=16 * 1024 * 1024)
            part.unlink()
    if tmp_dest.stat().st_size != size:
        raise IOError(f"assembled file {tmp_dest.stat().st_size} != expected {size}")
    tmp_dest.replace(dest)
    try:
        manifest_path.unlink(missing_ok=True)
        part_dir.rmdir()
    except OSError:
        pass
    return dest


def download_archs4(raw_dir: Path, species: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{species}_gene_archs4.h5"
    if dest.exists() and dest.stat().st_size > 1024**3:
        print(f"[download] {dest} already exists ({dest.stat().st_size / 1024**3:.1f} GiB)", flush=True)
        return dest
    parallel_parts = int(os.environ.get("TXN_DOWNLOAD_PARTS", "12"))
    for url in ARCHS4_URLS[species]:
        tmp = dest.with_suffix(".h5.partial")
        try:
            if parallel_parts > 1:
                tmp.unlink(missing_ok=True)
                parallel_download(url, dest, parallel_parts)
            else:
                run(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "20",
                     "-C", "-", "-o", tmp, url])
                tmp.replace(dest)
            print(f"[download] wrote {dest}", flush=True)
            return dest
        except Exception as exc:
            print(f"[download] failed {url}: {exc}", flush=True)
    raise RuntimeError(f"could not download ARCHS4 {species} H5")


def load_hom_one_to_one(hom_path: Path):
    df = pd.read_csv(hom_path, sep="\t")
    required = {"DB Class Key", "Common Organism Name", "Symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{hom_path} missing columns: {sorted(missing)}")
    human_by_class = {}
    mouse_by_class = {}
    for _, row in df.iterrows():
        org = str(row["Common Organism Name"]).strip().lower()
        symbol = str(row["Symbol"]).strip()
        if not symbol or symbol.lower() == "nan":
            continue
        key = row["DB Class Key"]
        if org == "human":
            human_by_class.setdefault(key, set()).add(symbol)
        elif org == "mouse, laboratory":
            mouse_by_class.setdefault(key, set()).add(symbol)

    human_to_mouse = {}
    mouse_to_human = {}
    for key in sorted(set(human_by_class) & set(mouse_by_class)):
        humans = sorted(human_by_class[key])
        mice = sorted(mouse_by_class[key])
        if len(humans) != 1 or len(mice) != 1:
            continue
        h, m = humans[0], mice[0]
        human_to_mouse[h.upper()] = m
        mouse_to_human[m.upper()] = h
    return human_to_mouse, mouse_to_human


def h5_symbols(path: Path):
    with h5py.File(path, "r") as h5:
        return decode_array(h5["meta/genes/symbol"][:])


def build_source_map(symbols, canonical, species, human_to_mouse=None):
    by_upper = {}
    for i, symbol in enumerate(symbols):
        by_upper.setdefault(str(symbol).upper(), []).append(i)

    target_to_sources = []
    missing = []
    for gene in canonical:
        key = gene.upper()
        source_key = key
        if species == "mouse":
            source_key = str(human_to_mouse.get(key, "")).upper()
        rows = by_upper.get(source_key, [])
        if not rows:
            missing.append(gene)
        target_to_sources.append(rows)

    source_to_target = []
    for target_idx, rows in enumerate(target_to_sources):
        for row in rows:
            source_to_target.append((row, target_idx))
    source_to_target.sort(key=lambda x: x[0])
    source_rows = [row for row, _ in source_to_target]
    source_targets = [target for _, target in source_to_target]
    return source_rows, source_targets, missing


def single_cell_probability(h5):
    samples = h5.get("meta/samples")
    if samples is None:
        return None
    for key in ("singlecellprobability", "single_cell_probability", "sc_prob"):
        if key in samples:
            return np.asarray(samples[key][:])
    return None


def extract_species(
    h5_path: Path,
    out_batch_dir: Path,
    species: str,
    canonical,
    source_rows,
    source_targets,
    batch_size: int,
    qc_min_nonzero: int,
    remove_single_cell: bool,
):
    print(f"[{species}] extracting full ARCHS4 data from {h5_path}", flush=True)
    t0 = time.time()
    out_batch_dir.mkdir(parents=True, exist_ok=True)
    source_rows = np.asarray(source_rows, dtype=np.int64)
    source_targets = np.asarray(source_targets, dtype=np.int64)
    canonical = list(canonical)

    batch_files = []
    meta_rows = []
    gene_sum = np.zeros((len(canonical),), dtype=np.float64)
    total_kept = 0
    total_bulk = 0

    with h5py.File(h5_path, "r") as h5:
        expr = h5["data/expression"]
        accessions = decode_array(h5["meta/samples/geo_accession"][:])
        sc_prob = single_cell_probability(h5) if remove_single_cell else None
        n_samples = expr.shape[1]
        n_batches = (n_samples + batch_size - 1) // batch_size
        print(f"[{species}] H5 shape={expr.shape}, batches={n_batches}", flush=True)

        for batch_num, start in enumerate(range(0, n_samples, batch_size), start=1):
            end = min(start + batch_size, n_samples)
            raw = np.asarray(expr[source_rows, start:end], dtype=np.uint32)
            out = np.zeros((len(canonical), end - start), dtype=np.uint32)
            np.add.at(out, source_targets, raw)

            keep = np.ones((end - start,), dtype=bool)
            if sc_prob is not None:
                keep &= np.asarray(sc_prob[start:end]) < 0.5
            if qc_min_nonzero > 0:
                keep &= (out > 0).sum(axis=0) >= qc_min_nonzero
            if not keep.any():
                print(f"[{species}] batch {batch_num}/{n_batches}: 0 kept", flush=True)
                continue

            out = out[:, keep]
            kept_accessions = accessions[start:end][keep]
            batch_name = f"{species}_batch_{batch_num:05d}.parquet"
            batch_path = out_batch_dir / batch_name
            df = pd.DataFrame(out.T, index=kept_accessions, columns=canonical)
            df.index.name = "geo_accession"
            write_count_parquet(df, batch_path)

            gene_sum += out.sum(axis=1)
            batch_files.append(batch_path)
            meta_rows.extend({"geo_accession": str(s), "species": species} for s in kept_accessions)
            total_kept += int(out.shape[1])
            total_bulk += int(keep.size if sc_prob is None else (np.asarray(sc_prob[start:end]) < 0.5).sum())

            elapsed = time.time() - t0
            print(
                f"[{species}] batch {batch_num}/{n_batches}: kept={out.shape[1]:,}, "
                f"total={total_kept:,}, elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    return batch_files, pd.DataFrame(meta_rows), gene_sum, total_kept, total_bulk


def main():
    parser = argparse.ArgumentParser(description="Prepare full harmonized ARCHS4 shards for Txn_Jatin.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("TXN_PREP_BATCH_SIZE", "2000")))
    parser.add_argument("--qc-min-nonzero", type=int, default=int(os.environ.get("TXN_QC_MIN_NONZERO", "12000")))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sequential-download", action="store_true",
                        help="Keep at most one raw ARCHS4 H5 on disk at a time.")
    parser.add_argument("--keep-raw", action="store_true",
                        help="Do not delete raw H5 files in sequential mode.")
    parser.add_argument("--keep-single-cell", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    walter = root / "RNA Walter"
    raw_dir = walter / "data" / "archs4" / "raw"
    out_dir = Path(args.output_dir) if args.output_dir else walter / "data" / "archs4" / "train_txn_jatin_full"
    marker = out_dir / "PREPARED_FULL_DATA.ok"
    if marker.exists() and not args.force:
        print(f"[skip] full prepared data already exists: {out_dir}", flush=True)
        print(marker.read_text(), flush=True)
        return 0
    if out_dir.exists():
        shutil.rmtree(out_dir)
    batch_dir = out_dir / "batch_files"
    batch_dir.mkdir(parents=True, exist_ok=True)

    human_to_mouse, mouse_to_human = load_hom_one_to_one(walter / "data" / "HOM_MouseHumanSequence.rpt")

    if args.download and args.sequential_download:
        human_h5 = download_archs4(raw_dir, "human")
        human_symbols = h5_symbols(human_h5)
        human_present = {str(s).upper() for s in human_symbols}
        canonical = sorted(h for h in human_to_mouse if h in human_present)
        canonical_case = {str(s).upper(): str(s) for s in human_symbols}
        canonical = [canonical_case.get(g, g) for g in canonical]
        print(f"[genes] one-to-one human candidate genes: {len(canonical):,}", flush=True)
        if len(canonical) < 10000:
            raise RuntimeError(f"too few harmonized candidate genes: {len(canonical)}")

        h_rows, h_targets, h_missing = build_source_map(human_symbols, canonical, "human")
        print(f"[genes] human source rows: {len(h_rows):,}", flush=True)
        if h_missing:
            print(f"[genes] missing after human map: {len(h_missing)}", flush=True)

        h_files, h_meta, h_sum, h_kept, h_bulk = extract_species(
            human_h5, batch_dir, "human", canonical, h_rows, h_targets,
            args.batch_size, args.qc_min_nonzero, not args.keep_single_cell,
        )
        if not args.keep_raw:
            human_h5.unlink(missing_ok=True)
            human_h5.with_suffix(".h5.partial").unlink(missing_ok=True)
            print(f"[download] removed human H5 after processing: {human_h5}", flush=True)

        mouse_h5 = download_archs4(raw_dir, "mouse")
        mouse_symbols = h5_symbols(mouse_h5)
        m_rows, m_targets, m_missing = build_source_map(
            mouse_symbols, canonical, "mouse", human_to_mouse=human_to_mouse
        )
        print(f"[genes] mouse source rows: {len(m_rows):,}", flush=True)
        if m_missing:
            print(f"[genes] missing after mouse map: {len(m_missing)}", flush=True)

        m_files, m_meta, m_sum, m_kept, m_bulk = extract_species(
            mouse_h5, batch_dir, "mouse", canonical, m_rows, m_targets,
            args.batch_size, args.qc_min_nonzero, not args.keep_single_cell,
        )
        if not args.keep_raw:
            mouse_h5.unlink(missing_ok=True)
            mouse_h5.with_suffix(".h5.partial").unlink(missing_ok=True)
            print(f"[download] removed mouse H5 after processing: {mouse_h5}", flush=True)
    else:
        if args.download:
            human_h5 = download_archs4(raw_dir, "human")
            mouse_h5 = download_archs4(raw_dir, "mouse")
            human_symbols = h5_symbols(human_h5)
            mouse_symbols = h5_symbols(mouse_h5)
        else:
            human_h5 = next(raw_dir.glob("human*_gene*.h5"), raw_dir / "human_gene_archs4.h5")
            mouse_h5 = next(raw_dir.glob("mouse*_gene*.h5"), raw_dir / "mouse_gene_archs4.h5")
            human_symbols = h5_symbols(human_h5) if human_h5.exists() else None
            mouse_symbols = h5_symbols(mouse_h5) if mouse_h5.exists() else None
        if not human_h5.exists() or not mouse_h5.exists():
            raise FileNotFoundError(
                f"ARCHS4 H5 files missing in {raw_dir}. Rerun with --download. "
                f"human={human_h5.exists()} mouse={mouse_h5.exists()}"
            )

        human_present = {str(s).upper() for s in human_symbols}
        mouse_present = {str(s).upper() for s in mouse_symbols}
        canonical = sorted(
            h for h in human_to_mouse
            if h in human_present and str(human_to_mouse[h]).upper() in mouse_present
        )
        canonical_case = {str(s).upper(): str(s) for s in human_symbols}
        canonical = [canonical_case.get(g, g) for g in canonical]
        print(f"[genes] one-to-one harmonized genes present in both ARCHS4 H5s: {len(canonical):,}", flush=True)
        if len(canonical) < 10000:
            raise RuntimeError(f"too few harmonized genes: {len(canonical)}")

        h_rows, h_targets, h_missing = build_source_map(human_symbols, canonical, "human")
        m_rows, m_targets, m_missing = build_source_map(
            mouse_symbols, canonical, "mouse", human_to_mouse=human_to_mouse
        )
        print(f"[genes] source rows: human={len(h_rows):,}, mouse={len(m_rows):,}", flush=True)
        if h_missing or m_missing:
            print(f"[genes] missing after map: human={len(h_missing)}, mouse={len(m_missing)}", flush=True)

        m_files, m_meta, m_sum, m_kept, m_bulk = extract_species(
            mouse_h5, batch_dir, "mouse", canonical, m_rows, m_targets,
            args.batch_size, args.qc_min_nonzero, not args.keep_single_cell,
        )
        h_files, h_meta, h_sum, h_kept, h_bulk = extract_species(
            human_h5, batch_dir, "human", canonical, h_rows, h_targets,
            args.batch_size, args.qc_min_nonzero, not args.keep_single_cell,
        )
    final_mask = (h_sum > 0) & (m_sum > 0)
    final_genes = [g for g, keep in zip(canonical, final_mask) if keep]
    print(f"[genes] final shared nonzero genes: {len(final_genes):,}", flush=True)
    if len(final_genes) < 10000:
        raise RuntimeError(f"too few final genes after nonzero filter: {len(final_genes)}")

    all_files = h_files + m_files
    for i, file in enumerate(all_files, start=1):
        df = pd.read_parquet(file)
        df = df.reindex(columns=final_genes, fill_value=0).astype("uint32", copy=False)
        df.index.name = "geo_accession"
        write_count_parquet(df, file)
        if i % 25 == 0 or i == len(all_files):
            print(f"[finalize] rewrote {i}/{len(all_files)} parquets to final gene set", flush=True)

    meta = pd.concat([h_meta, m_meta], ignore_index=True)
    meta.to_csv(out_dir / "metadata.csv", index=False)
    pd.DataFrame({"token_id": range(1, len(final_genes) + 1), "gene_symbol": final_genes}).to_csv(
        out_dir / "canonical_genes.csv", index=False
    )
    (out_dir / "genes.json").write_text(json.dumps(final_genes))
    samples = [{"id": r.geo_accession, "species": r.species} for r in meta.itertuples(index=False)]
    (out_dir / "samples.json").write_text(json.dumps(samples))
    manifest = {}
    for file in all_files:
        df = pd.read_parquet(file, columns=[])
        manifest[file.name] = [str(x) for x in df.index.tolist()]
    (out_dir / "batch_manifest.json").write_text(json.dumps(manifest))

    stats = {
        "output_dir": str(out_dir),
        "batch_files": len(all_files),
        "genes": len(final_genes),
        "samples": len(meta),
        "human_samples": int((meta["species"] == "human").sum()),
        "mouse_samples": int((meta["species"] == "mouse").sum()),
        "human_bulk_seen": int(h_bulk),
        "mouse_bulk_seen": int(m_bulk),
        "qc_min_nonzero": args.qc_min_nonzero,
        "batch_size": args.batch_size,
    }
    marker.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
