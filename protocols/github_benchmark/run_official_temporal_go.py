#!/usr/bin/env python3
"""Execution shim for the upstream temporal-GO script.

The pinned upstream file defines ``main`` but never calls it, ignores its split
arguments, and writes to absolute ``/results`` paths. This shim changes only
those execution details; the upstream classifier and evaluation code run as-is.
"""

from __future__ import annotations

import argparse
import builtins
import importlib.util
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-script", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--subfolder", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    for relative in ("fold", "holdout", "holdout_old"):
        (args.out_root / relative).mkdir(parents=True, exist_ok=True)

    original_open = builtins.open

    def redirected_open(file, *open_args, **open_kwargs):
        path = str(file)
        prefix = "/results/single_gene/go_general/"
        if path.startswith(prefix):
            path = str(args.out_root / path[len(prefix) :])
        return original_open(path, *open_args, **open_kwargs)

    spec = importlib.util.spec_from_file_location("upstream_temporal_go", args.upstream_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {args.upstream_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    os.chdir(args.split_dir)
    sys.argv = [
        str(args.upstream_script),
        "--subfolder",
        str(args.subfolder),
        "--cv-fold1-pkl",
        "go_cv_fold1_dict_all.pkl",
        "--cv-fold2-pkl",
        "go_cv_fold2_dict_all.pkl",
        "--cv-fold3-pkl",
        "go_cv_fold3_dict_all.pkl",
        "--holdout-pkl",
        "go_holdout_dict_all.pkl",
        "-d",
        str(args.out_root),
    ]
    builtins.open = redirected_open
    try:
        module.main()
    finally:
        builtins.open = original_open


if __name__ == "__main__":
    main()
