#!/usr/bin/env python3
"""Print compact assertions from a production atlas API response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.response.read_text(encoding="utf-8"))
    sample = payload["sample_results"][0]
    result = {
        "atlas": payload["atlas"],
        "input": payload["input"],
        "sample": sample["sample"],
        "best_match": sample["matches"][0],
        "species_evidence": sample["species_evidence"],
        "tissue_evidence": sample["tissue_evidence"],
        "gene_annotation_count": len(payload.get("gene_annotations", [])),
        "disease_association_count": len(payload.get("disease_associations", [])),
        "ortholog_count": len(payload.get("mouse_to_human_orthologs", [])),
        "language_head": {
            "status": payload.get("language_head", {}).get("status"),
            "model": payload.get("language_head", {}).get("model"),
            "text_characters": len(payload.get("language_head", {}).get("text", "")),
            "text_preview": payload.get("language_head", {}).get("text", "")[:800],
            "error": payload.get("language_head", {}).get("error"),
        },
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
