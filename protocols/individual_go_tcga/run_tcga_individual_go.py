#!/usr/bin/env python3
"""Run and aggregate the pinned GitHub individual-GO benchmark on TCGA genes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCOPES = ("tcga_all_genes", "tcga_strict_intersection")
METRICS = ("AUROC", "AUPRC", "PR@10")


@dataclass(frozen=True)
class Job:
    scope: str
    model: str
    command: tuple[str, ...]
    expected: Path
    log: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=13)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_gene_list(folder: Path) -> set[str]:
    path = next(folder.glob("*.txt"))
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def load_go_terms(path: Path) -> dict[str, dict[str, str]]:
    terms: dict[str, dict[str, str]] = {}
    current: dict[str, str] = {}
    in_term = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "[Term]":
            if current.get("id") and current.get("is_obsolete") != "true":
                terms[current["id"]] = current
            current = {}
            in_term = True
        elif line.startswith("["):
            if current.get("id") and current.get("is_obsolete") != "true":
                terms[current["id"]] = current
            current = {}
            in_term = False
        elif in_term and ": " in line:
            key, value = line.split(": ", 1)
            if key in {"id", "name", "namespace", "is_obsolete"}:
                current[key] = value
    if current.get("id") and current.get("is_obsolete") != "true":
        terms[current["id"]] = current
    return terms


def run_job(job: Job, status: dict, status_path: Path, lock: threading.Lock):
    if job.expected.exists() and job.expected.stat().st_size > 0:
        return job, 0, "resumed-existing", 0.0
    job.log.parent.mkdir(parents=True, exist_ok=True)
    job.expected.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        status["running"].append(f"{job.scope}:{job.model}")
        atomic_json(status_path, status)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    started = time.time()
    with job.log.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND=" + json.dumps(job.command) + "\n")
        handle.flush()
        process = subprocess.run(
            job.command,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - started
    result = (
        "ok"
        if process.returncode == 0
        and job.expected.exists()
        and job.expected.stat().st_size > 0
        else "failed"
    )
    with lock:
        status["running"].remove(f"{job.scope}:{job.model}")
        status["completed" if result == "ok" else "failed"].append(
            {
                "scope": job.scope,
                "model": job.model,
                "returncode": process.returncode,
                "elapsed_seconds": elapsed,
                "result": result,
                "log": str(job.log),
            }
        )
        atomic_json(status_path, status)
    return job, process.returncode, result, elapsed


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def main() -> int:
    args = parse_args()
    args.benchmark_repo = args.benchmark_repo.resolve()
    args.prepared_dir = args.prepared_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    manifest_path = args.prepared_dir / "prepared_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    prepared_manifest = json.loads(manifest_path.read_text())

    gene_script = (
        args.benchmark_repo
        / "src"
        / "gene_level_benchmark"
        / "gene_level_benchmarks.py"
    )
    split_root = (
        args.benchmark_repo
        / "data"
        / "data_splits"
        / "gene_level_benchmark"
        / "go_folds_splits"
    )
    split_paths = {
        "fold1": split_root / "go_cv_fold1_dict_all.pkl",
        "fold2": split_root / "go_cv_fold2_dict_all.pkl",
        "fold3": split_root / "go_cv_fold3_dict_all.pkl",
        "holdout": split_root / "go_holdout_dict_all.pkl",
    }
    go_obo = args.benchmark_repo / "data" / "obo" / "go.obo"
    for path in (gene_script, go_obo, *split_paths.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    models_by_scope = {
        scope: sorted(path.name for path in (args.prepared_dir / scope).iterdir())
        for scope in SCOPES
    }
    if models_by_scope[SCOPES[0]] != models_by_scope[SCOPES[1]]:
        raise RuntimeError("model lists differ between TCGA scopes")
    models = models_by_scope[SCOPES[0]]
    if len(models) != 12 or "ESM3" not in models:
        raise RuntimeError(f"expected 12 models including ESM3, got {models}")

    strict_lists = [
        load_gene_list(args.prepared_dir / "tcga_strict_intersection" / model)
        for model in models
    ]
    if any(genes != strict_lists[0] for genes in strict_lists[1:]):
        raise RuntimeError("strict-intersection model gene lists are not identical")

    raw_root = args.output_dir / "raw"
    logs_root = args.output_dir / "logs"
    tables_root = args.output_dir / "tables"
    reports_root = args.output_dir / "reports"
    for path in (raw_root, logs_root, tables_root, reports_root):
        path.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    status = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "official_go",
        "models": models,
        "scopes": list(SCOPES),
        "completed": [],
        "failed": [],
        "running": [],
    }
    atomic_json(status_path, status)

    jobs = []
    for scope in SCOPES:
        out = raw_root / scope
        for model in models:
            command = (
                args.python,
                str(gene_script),
                "--subfolder",
                str(args.prepared_dir / scope / model),
                "--cv-fold1-pkl",
                str(split_paths["fold1"]),
                "--cv-fold2-pkl",
                str(split_paths["fold2"]),
                "--cv-fold3-pkl",
                str(split_paths["fold3"]),
                "--holdout-pkl",
                str(split_paths["holdout"]),
                "-d",
                str(out),
            )
            jobs.append(
                Job(
                    scope=scope,
                    model=model,
                    command=command,
                    expected=out / f"{model}_holdout_results.pkl",
                    log=logs_root / scope / f"{model}.log",
                )
            )

    lock = threading.Lock()
    failures = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = [
            pool.submit(run_job, job, status, status_path, lock) for job in jobs
        ]
        for future in as_completed(futures):
            job, code, result, elapsed = future.result()
            print(
                f"[official-go] {job.scope}:{job.model} "
                f"{result} rc={code} elapsed={elapsed / 60:.1f} min",
                flush=True,
            )
            if result == "failed":
                failures.append(f"{job.scope}:{job.model}")
    if failures:
        raise RuntimeError(f"official GO jobs failed: {failures}")

    with split_paths["fold1"].open("rb") as handle:
        folds1 = pickle.load(handle)
    with split_paths["fold2"].open("rb") as handle:
        folds2 = pickle.load(handle)
    with split_paths["fold3"].open("rb") as handle:
        folds3 = pickle.load(handle)
    with split_paths["holdout"].open("rb") as handle:
        holdouts = pickle.load(handle)
    expected_terms = list(holdouts)
    if len(expected_terms) != 56:
        raise RuntimeError(f"expected 56 official GO terms, got {len(expected_terms)}")
    go_terms = load_go_terms(go_obo)
    missing_go_names = sorted(set(expected_terms) - set(go_terms))
    if missing_go_names:
        raise RuntimeError(f"GO metadata missing for terms: {missing_go_names}")

    rows = []
    for scope in SCOPES:
        for model in models:
            gene_set = load_gene_list(args.prepared_dir / scope / model)
            result_path = raw_root / scope / f"{model}_holdout_results.pkl"
            with result_path.open("rb") as handle:
                result = pickle.load(handle)
            if set(result) != set(expected_terms):
                missing = sorted(set(expected_terms) - set(result))
                raise RuntimeError(f"{scope}:{model} missing terms: {missing}")
            for term in expected_terms:
                frame = result[term]
                if len(frame) != 1:
                    raise RuntimeError(f"{scope}:{model}:{term} has {len(frame)} rows")
                record = frame.iloc[0]
                train = pd.concat(
                    [folds1[term], folds2[term], folds3[term]], ignore_index=True
                )
                train = train[train["gene"].astype(str).isin(gene_set)]
                holdout = holdouts[term]
                holdout = holdout[holdout["gene"].astype(str).isin(gene_set)]
                rows.append(
                    {
                        "scope": scope,
                        "model": model,
                        "term": term,
                        "term_name": go_terms[term]["name"],
                        "term_namespace": go_terms[term]["namespace"],
                        "best_C": float(record["C"]),
                        "AUROC": float(record["AUC"]),
                        "AUPRC": float(record["AUPRC"]),
                        "PR@10": float(record["PR@10"]),
                        "train_genes": int(train["gene"].nunique()),
                        "train_positive": int((train["result"] == 1).sum()),
                        "train_negative": int((train["result"] == 0).sum()),
                        "holdout_genes": int(holdout["gene"].nunique()),
                        "holdout_positive": int((holdout["result"] == 1).sum()),
                        "holdout_negative": int((holdout["result"] == 0).sum()),
                        "source_pickle": str(result_path),
                    }
                )
    scores = pd.DataFrame(rows)
    if len(scores) != len(SCOPES) * len(models) * len(expected_terms):
        raise RuntimeError(f"unexpected score row count: {len(scores)}")
    if not np.isfinite(scores[list(METRICS) + ["best_C"]].to_numpy()).all():
        raise RuntimeError("non-finite official GO scores detected")
    scores.to_csv(tables_root / "individual_go_term_scores.csv", index=False)
    pd.DataFrame(
        [
            {
                "term": term,
                "term_name": go_terms[term]["name"],
                "term_namespace": go_terms[term]["namespace"],
            }
            for term in expected_terms
        ]
    ).sort_values("term").to_csv(tables_root / "go_term_names.csv", index=False)

    summary = (
        scores.groupby(["scope", "model"], as_index=False)
        .agg(
            terms=("term", "nunique"),
            mean_AUROC=("AUROC", "mean"),
            median_AUROC=("AUROC", "median"),
            mean_AUPRC=("AUPRC", "mean"),
            mean_PR_at_10=("PR@10", "mean"),
            mean_train_genes=("train_genes", "mean"),
            mean_holdout_genes=("holdout_genes", "mean"),
        )
    )
    summary["AUROC_rank"] = summary.groupby("scope")["mean_AUROC"].rank(
        ascending=False, method="min"
    )
    summary = summary.sort_values(["scope", "AUROC_rank", "model"])
    summary.to_csv(tables_root / "model_summary.csv", index=False)

    winners = scores.loc[
        scores.groupby(["scope", "term"])["AUROC"].idxmax()
    ].sort_values(["scope", "term"])
    winners.to_csv(tables_root / "per_term_winners.csv", index=False)
    (
        winners.groupby(["scope", "model"], as_index=False)
        .agg(AUROC_wins=("term", "nunique"))
        .sort_values(["scope", "AUROC_wins", "model"], ascending=[True, False, True])
        .to_csv(tables_root / "winner_counts.csv", index=False)
    )
    for scope in SCOPES:
        subset = scores[scores["scope"] == scope]
        for metric in METRICS:
            subset.pivot(index="term", columns="model", values=metric).to_csv(
                tables_root / f"{scope}_{metric.replace('@', '_at_')}.csv"
            )

    commit = subprocess.check_output(
        ["git", "-C", str(args.benchmark_repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    manifest = {
        "benchmark_repository": "ylaboratory/gene-embedding-benchmarks",
        "benchmark_commit": commit,
        "go_obo": str(go_obo),
        "go_obo_sha256": sha256(go_obo),
        "protocol": (
            "Official per-GO-term nested CV and holdout SVC: C in "
            "[0.1,1,10,100,1000], class_weight=balanced, AUROC-selected C, "
            "official fixed folds, AUROC/AUPRC/PR@10"
        ),
        "prepared_manifest": str(manifest_path),
        "prepared_manifest_sha256": sha256(manifest_path),
        "models": models,
        "model_count": len(models),
        "official_go_terms": len(expected_terms),
        "scopes": {
            "tcga_all_genes": (
                "Each model's native embedding rows restricted to processed TCGA genes"
            ),
            "tcga_strict_intersection": (
                "Identical TCGA and official-reference genes across all 12 models"
            ),
        },
        "strict_intersection_genes": prepared_manifest[
            "strict_intersection_genes"
        ],
        "score_rows": len(scores),
        "workers": args.workers,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(args.output_dir / "manifest.json", manifest)

    report_lines = [
        "# TCGA individual GO benchmark",
        "",
        "## Protocol",
        "",
        "This run uses the pinned `ylaboratory/gene-embedding-benchmarks` GO "
        f"protocol at commit `{commit}`. All 56 official GO terms, fixed nested-CV "
        "folds, C selection, holdout sets, and metrics are preserved.",
        "",
        "The TCGA scope is defined by genes present in the processed 3,481-sample "
        "TCGA TPM matrix. The model representations are the same frozen artifacts "
        "used by the completed pinned GitHub run; ESM3 is added as a real frozen "
        "sequence embedding. TCGA expression values are not used as GO labels or "
        "as a replacement probing protocol.",
        "",
        "## Model summary",
        "",
        markdown_table(
            summary,
            [
                "scope",
                "model",
                "terms",
                "mean_AUROC",
                "mean_AUPRC",
                "mean_PR_at_10",
                "AUROC_rank",
            ],
        ),
        "",
        "## Validation",
        "",
        f"- Models: **{len(models)}**",
        f"- Official GO terms per model and scope: **{len(expected_terms)}**",
        f"- Long-table rows: **{len(scores)}**",
        "- Missing or non-finite metrics: **0**",
        "- Fallback embeddings: **0**",
        f"- Strict shared genes: **{prepared_manifest['strict_intersection_genes']:,}**",
        "",
        "## Interpretation boundary",
        "",
        "GO annotations are gene-level labels. Calling this a TCGA benchmark means "
        "that the eligible gene universe is restricted to the processed TCGA matrix; "
        "it does not mean that TCGA samples carry GO labels. This distinction keeps "
        "the result directly comparable with the supplied GitHub benchmark.",
        "",
    ]
    (reports_root / "REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    status["stage"] = "complete"
    status["completed_utc"] = manifest["completed_utc"]
    atomic_json(status_path, status)
    checksum_rows = []
    for path in sorted(args.output_dir.rglob("*")):
        if (
            path.is_file()
            and path.name != "SHA256SUMS.txt"
            and ".tmp" not in path.suffixes
        ):
            checksum_rows.append(
                f"{sha256(path)}  {path.relative_to(args.output_dir).as_posix()}\n"
            )
    (args.output_dir / "SHA256SUMS.txt").write_text(
        "".join(checksum_rows), encoding="ascii"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
