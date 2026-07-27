#!/usr/bin/env python3
"""Extract resumable, sequence-only ESM3 gene embeddings.

The output follows the symbol-keyed NPZ contract used by the Txn_Jatin
benchmark suite. Protein sequences are taken from BulkFormer's pinned gene
metadata so ESM2 and ESM3 use the same gene-to-protein mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-info", type=Path, required=True)
    parser.add_argument("--master-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="esm3-sm-open-v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-residues", type=int, default=1022)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--flush-every", type=int, default=25)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_master(path: Path) -> list[str]:
    with np.load(path, allow_pickle=True) as payload:
        return [str(value).upper() for value in payload["keys"].tolist()]


def load_sequences(path: Path) -> dict[str, str]:
    table = pd.read_csv(path, dtype=str)
    required = {"gene_symbol", "Sequence"}
    if not required.issubset(table.columns):
        raise ValueError(f"{path} is missing columns {sorted(required - set(table))}")
    table = table.dropna(subset=["gene_symbol", "Sequence"]).copy()
    table["gene_symbol"] = table["gene_symbol"].str.upper()
    table["Sequence"] = (
        table["Sequence"].str.upper().str.replace(r"[^A-Z]", "", regex=True)
    )
    table = table[table["Sequence"].str.len() > 0]
    table = table.drop_duplicates("gene_symbol", keep="first")
    return dict(zip(table["gene_symbol"], table["Sequence"]))


def initialize_work(
    work_dir: Path,
    genes: list[str],
    dimensions: int,
) -> tuple[np.memmap, np.memmap]:
    work_dir.mkdir(parents=True, exist_ok=True)
    genes_path = work_dir / "genes.json"
    values_path = work_dir / "embeddings.npy"
    done_path = work_dir / "done.npy"
    if genes_path.exists():
        stored = json.loads(genes_path.read_text())
        if stored != genes:
            raise RuntimeError("ESM3 work directory uses a different master gene order")
        values = np.lib.format.open_memmap(values_path, mode="r+")
        done = np.lib.format.open_memmap(done_path, mode="r+")
        if values.shape != (len(genes), dimensions) or done.shape != (len(genes),):
            raise RuntimeError("ESM3 work arrays have incompatible shapes")
        return values, done

    genes_path.write_text(json.dumps(genes) + "\n")
    values = np.lib.format.open_memmap(
        values_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(genes), dimensions),
    )
    values[:] = np.nan
    done = np.lib.format.open_memmap(
        done_path,
        mode="w+",
        dtype=np.bool_,
        shape=(len(genes),),
    )
    done[:] = False
    values.flush()
    done.flush()
    return values, done


def make_batches(
    indices: list[int],
    sequences: list[str | None],
    max_residues: int,
    token_budget: int,
    max_batch_size: int,
) -> list[list[int]]:
    ordered = sorted(
        indices,
        key=lambda index: min(len(sequences[index] or ""), max_residues),
    )
    batches: list[list[int]] = []
    batch: list[int] = []
    longest = 0
    for index in ordered:
        length = min(len(sequences[index] or ""), max_residues) + 2
        proposed_longest = max(longest, length)
        proposed_size = len(batch) + 1
        if batch and (
            proposed_longest * proposed_size > token_budget
            or proposed_size > max_batch_size
        ):
            batches.append(batch)
            batch = []
            longest = 0
        batch.append(index)
        longest = max(longest, length)
    if batch:
        batches.append(batch)
    return batches


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    genes = load_master(args.master_npz)
    sequence_map = load_sequences(args.gene_info)
    sequences = [sequence_map.get(gene) for gene in genes]

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, LogitsConfig
    from esm.utils.sampling import _BatchedESMProteinTensor

    device = torch.device(args.device)
    print(f"[ESM3] loading {args.model} on {device}", flush=True)
    model = ESM3.from_pretrained(args.model, device=device).eval()
    dimensions = int(model.encoder.sequence_embed.embedding_dim)
    work_dir = args.output.with_suffix(args.output.suffix + ".work")
    values, done = initialize_work(work_dir, genes, dimensions)

    missing_sequence = np.asarray(
        [sequence is None for sequence in sequences], dtype=bool
    )
    done[missing_sequence] = True
    pending = [
        index
        for index in range(len(genes))
        if not bool(done[index]) and sequences[index] is not None
    ]
    batches = make_batches(
        pending,
        sequences,
        args.max_residues,
        args.token_budget,
        args.max_batch_size,
    )
    print(
        f"[ESM3] genes={len(genes):,} sequences={len(genes) - missing_sequence.sum():,} "
        f"pending={len(pending):,} batches={len(batches):,} dimensions={dimensions}",
        flush=True,
    )

    started = time.time()

    def try_embed_batch(indices: list[int]) -> bool:
        token_rows = None
        tokens = None
        batched = None
        output = None
        hidden = None
        pooled = None
        succeeded = False
        try:
            token_rows = []
            lengths = []
            for index in indices:
                sequence = (sequences[index] or "")[: args.max_residues]
                encoded = model.encode(ESMProtein(sequence=sequence))
                assert encoded.sequence is not None
                token_rows.append(encoded.sequence)
                lengths.append(int(encoded.sequence.numel()))
            padding_id = int(model.tokenizers.sequence.pad_token_id)
            tokens = pad_sequence(
                token_rows,
                batch_first=True,
                padding_value=padding_id,
            )
            batched = _BatchedESMProteinTensor(sequence=tokens)
            with torch.inference_mode():
                output = model.logits(
                    batched,
                    LogitsConfig(return_embeddings=True),
                )
            if output.embeddings is None:
                raise RuntimeError("ESM3 did not return embeddings")
            hidden = output.embeddings
            pooled = torch.stack(
                [
                    hidden[row, 1 : lengths[row] - 1].mean(dim=0)
                    for row in range(len(indices))
                ]
            )
            values[indices] = pooled.float().cpu().numpy()
            done[indices] = True
            succeeded = True
            return True
        except torch.OutOfMemoryError:
            return False
        finally:
            del token_rows, tokens, batched, output, hidden, pooled
            if not succeeded:
                torch.cuda.empty_cache()

    def embed_batch(indices: list[int]) -> None:
        queue = [indices]
        while queue:
            current = queue.pop()
            if try_embed_batch(current):
                continue
            if len(current) == 1:
                raise RuntimeError(
                    f"ESM3 cannot embed gene {genes[current[0]]} within H100 memory"
                )
            midpoint = len(current) // 2
            queue.append(current[midpoint:])
            queue.append(current[:midpoint])

    for batch_number, batch in enumerate(batches, start=1):
        embed_batch(batch)
        if batch_number % args.flush_every == 0 or batch_number == len(batches):
            values.flush()
            done.flush()
            elapsed = max(time.time() - started, 1e-6)
            completed = int(done.sum())
            rate = completed / elapsed
            print(
                f"[ESM3] batch={batch_number:,}/{len(batches):,} "
                f"complete={completed:,}/{len(genes):,} rate={rate:.2f} genes/s",
                flush=True,
            )

    covered = np.isfinite(values).all(axis=1)
    truncated = sum(
        sequence is not None and len(sequence) > args.max_residues
        for sequence in sequences
    )
    provenance = {
        "name": "ESM3",
        "role": "sequence-only protein-language-model control",
        "model": args.model,
        "pooling": "mean final-layer residue embeddings; BOS/EOS excluded",
        "sequence_source": str(args.gene_info.resolve()),
        "sequence_source_sha256": sha256(args.gene_info),
        "master_gene_source": str(args.master_npz.resolve()),
        "master_gene_source_sha256": sha256(args.master_npz),
        "max_residues": args.max_residues,
        "truncated_sequences": int(truncated),
        "covered_genes": int(covered.sum()),
        "total_genes": len(genes),
        "dimensions": dimensions,
        "device": str(device),
    }
    np.savez_compressed(
        args.output,
        keys=np.asarray(genes),
        emb=np.asarray(values),
        is_fallback=np.asarray(False),
        provenance=np.asarray([json.dumps(provenance, sort_keys=True)]),
        covered=covered,
    )
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
