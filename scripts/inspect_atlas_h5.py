#!/usr/bin/env python3
"""Print HDF5 datasets and representative metadata values."""

from __future__ import annotations

import argparse

import h5py
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for raw_path in args.paths:
        print(f"=== {raw_path} ===")
        if raw_path.endswith(".parquet"):
            frame = pd.read_parquet(raw_path)
            print("shape", frame.shape)
            print("columns", list(frame.columns[:10]))
            print(frame.iloc[:2, :5].to_string())
            continue
        with h5py.File(raw_path, "r") as handle:
            def visit(name, item):
                if isinstance(item, h5py.Dataset):
                    print(name, item.shape, item.dtype)
                    if item.ndim == 1 and len(item):
                        values = item[: min(3, len(item))]
                        print("  example:", [value.decode(errors="replace") if isinstance(value, bytes) else str(value) for value in values])

            handle.visititems(visit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
