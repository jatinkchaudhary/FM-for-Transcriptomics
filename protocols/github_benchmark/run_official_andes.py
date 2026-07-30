#!/usr/bin/env python3
"""Run the pinned ANDES CLI, optionally enabling its documented distinct mode."""

from __future__ import annotations

import argparse
import functools
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--andes-root", type=Path, required=True)
    parser.add_argument("--emb", required=True)
    parser.add_argument("--genelist", required=True)
    parser.add_argument("--geneset1", required=True)
    parser.add_argument("--geneset2", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--processors", type=int, default=20)
    parser.add_argument("--distinct", action="store_true")
    args = parser.parse_args()

    source = args.andes_root / "src"
    sys.path.insert(0, str(source))
    import set_analysis_func  # type: ignore

    if args.distinct:
        original = set_analysis_func.andes
        set_analysis_func.andes = functools.partial(original, distinct=True)

    sys.argv = [
        str(source / "andes.py"),
        "--emb",
        args.emb,
        "--genelist",
        args.genelist,
        "--geneset1",
        args.geneset1,
        "--geneset2",
        args.geneset2,
        "--out",
        args.out,
        "-n",
        str(args.processors),
    ]
    runpy.run_path(str(source / "andes.py"), run_name="__main__")


if __name__ == "__main__":
    main()
