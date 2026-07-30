#!/usr/bin/env python3
"""Refresh atlas counts and checksums after curated resource updates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("atlas", type=Path)
    args = parser.parse_args()
    manifest_path = args.atlas / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with (args.atlas / "mouse_human_orthologs.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        manifest["ortholog_rows"] = sum(1 for _ in csv.DictReader(handle))
    manifest["ortholog_source"] = (
        "Mouse Genome Informatics HOM_MouseHumanSequence.rpt"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checksum_path = args.atlas / "SHA256SUMS.txt"
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in sorted(args.atlas.iterdir()):
            if path.is_file() and path != checksum_path:
                handle.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
