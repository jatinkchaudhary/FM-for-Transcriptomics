#!/usr/bin/env python3
"""Validate the public bundle and guard against accidental weight inclusion."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {".pt", ".pth", ".ckpt", ".safetensors", ".h5", ".hdf5", ".parquet", ".npz"}


def main() -> int:
    forbidden = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in FORBIDDEN]
    if forbidden:
        raise RuntimeError("Forbidden model/data artifacts: " + ", ".join(map(str, forbidden)))
    registry = json.loads((ROOT / "app" / "data" / "results_registry.json").read_text(encoding="utf-8"))
    assert len(registry["models"]) >= 12
    assert len(registry["experiments"]) >= 8
    manifest = json.loads((ROOT / "test_data" / "random_50_gene_panels" / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["files"]) == 50
    for item in manifest["files"]:
        path = ROOT / "test_data" / "random_50_gene_panels" / item["file"]
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        assert len(rows) == 51, path
        assert len(rows[0]) == 9, path
    required = [
        ROOT / "README.md",
        ROOT / "app" / "backend" / "server.py",
        ROOT / "app" / "frontend" / "index.dc.html",
        ROOT / "results" / "github_protocol" / "all_results_long_numeric.csv",
        ROOT / "results" / "gtex_external" / "gtex_all_model_summary.csv",
        ROOT / "results" / "gtex_external" / "manifest.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Missing required files: " + ", ".join(map(str, missing)))
    print(
        json.dumps(
            {
                "status": "ok",
                "models": len(registry["models"]),
                "experiments": len(registry["experiments"]),
                "test_matrices": len(manifest["files"]),
                "forbidden_artifacts": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
