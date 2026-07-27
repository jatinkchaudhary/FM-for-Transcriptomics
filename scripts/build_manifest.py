#!/usr/bin/env python3
"""Write a deterministic file manifest and SHA-256 checksums."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
    )
    OUTPUT.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in files),
        encoding="ascii",
    )
    print(f"Wrote {len(files)} checksums to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
