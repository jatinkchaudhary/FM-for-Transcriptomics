#!/usr/bin/env python3
"""Build a deterministic inventory of published benchmark code and results."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "ARTIFACT_INVENTORY.csv"
INCLUDED_ROOTS = (ROOT / "protocols", ROOT / "results")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    files = sorted(
        path
        for base in INCLUDED_ROOTS
        for path in base.rglob("*")
        if path.is_file() and path != OUTPUT
    )
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "category", "bytes", "sha256"])
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            writer.writerow(
                [
                    relative,
                    relative.split("/", 1)[0],
                    path.stat().st_size,
                    sha256(path),
                ]
            )
    print(f"Wrote {len(files)} benchmark artifacts to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
