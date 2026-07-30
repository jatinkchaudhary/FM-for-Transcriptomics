#!/usr/bin/env python3
"""Convert the official MGI human/mouse homology report for atlas use."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(lambda: {"mouse": [], "human": []})
    with args.input.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            taxon = row["NCBI Taxon ID"]
            if taxon == "10090":
                groups[row["DB Class Key"]]["mouse"].append(row["Symbol"])
            elif taxon == "9606":
                groups[row["DB Class Key"]]["human"].append(row["Symbol"])
    records = []
    for group, members in groups.items():
        for mouse in members["mouse"]:
            for human in members["human"]:
                records.append(
                    {
                        "mouse_symbol": mouse.upper(),
                        "human_symbol": human.upper(),
                        "orthology_type": "MGI homology class",
                        "source": f"MGI HOM_MouseHumanSequence.rpt class {group}",
                    }
                )
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mouse_symbol", "human_symbol", "orthology_type", "source"],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records):,} mouse-human mappings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
