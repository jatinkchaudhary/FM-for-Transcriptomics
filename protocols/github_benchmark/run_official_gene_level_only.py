#!/usr/bin/env python3
"""Run only the pinned GitHub gene-level benchmarks for selected models."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    expected: Path
    log: Path
    cwd: Path | None = None


def average_pickle_metrics(path: Path) -> dict[str, float]:
    with path.open("rb") as handle:
        results = pickle.load(handle)
    frames = [
        frame for frame in results.values() if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    if not frames:
        raise ValueError(f"no metric frames in {path}")
    combined = pd.concat(frames, ignore_index=True)
    return {
        "AUROC": float(combined["AUC"].mean()),
        "AUPRC": float(combined["AUPRC"].mean()),
        "PR@10": float(combined["PR@10"].mean()),
        "terms": int(len(frames)),
    }


def write_summary(output: Path, models: list[str]) -> None:
    rows = []
    configs = [
        ("all_genes", "go", "GO full"),
        ("intersect", "go", "GO intersect"),
        ("all_genes", "omim", "OMIM full"),
        ("intersect", "omim", "OMIM intersect"),
    ]
    for scope, task, benchmark in configs:
        root = output / "gene_level" / scope / task
        for model in models:
            path = root / f"{model}_holdout_results.pkl"
            if path.exists():
                rows.append(
                    {
                        "benchmark": benchmark,
                        "scope": scope,
                        "task": task,
                        "model": model,
                        **average_pickle_metrics(path),
                        "source": str(path),
                    }
                )
    temporal = output / "gene_level" / "intersect" / "go_temporal" / "holdout"
    for model in models:
        path = temporal / f"{model}_go_post24_.pkl"
        if path.exists():
            rows.append(
                {
                    "benchmark": "Temporal GO",
                    "scope": "intersect",
                    "task": "go_temporal",
                    "model": model,
                    **average_pickle_metrics(path),
                    "source": str(path),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "gene_level_summary_long.csv", index=False)
    if not frame.empty:
        wide = frame.pivot_table(
            index=["benchmark", "scope", "task"],
            columns="model",
            values=["AUROC", "AUPRC", "PR@10"],
            aggfunc="first",
        )
        wide.to_csv(output / "gene_level_summary_wide.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helper-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / "logs"
    logs.mkdir(exist_ok=True)
    status_path = args.output / "status.json"
    lock = threading.Lock()
    status = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "initializing",
        "completed": [],
        "failed": [],
        "running": [],
        "models": args.models,
        "omitted": "gene-pair benchmarks SL/POMBE/TF/NG and ANDES gene-set jobs",
    }

    def save_status() -> None:
        status["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        temporary.replace(status_path)

    def run_job(job: Job) -> tuple[str, int, str]:
        if job.expected.exists() and job.expected.stat().st_size > 0:
            return job.name, 0, "resumed-existing"
        job.log.parent.mkdir(parents=True, exist_ok=True)
        job.expected.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            status["running"].append(job.name)
            save_status()
        started = time.time()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(args.helper_dir) + os.pathsep + env.get("PYTHONPATH", "")
        with job.log.open("w", encoding="utf-8") as handle:
            handle.write("COMMAND=" + json.dumps(job.command) + "\n")
            handle.flush()
            proc = subprocess.run(
                job.command,
                cwd=job.cwd,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        result = "ok" if proc.returncode == 0 and job.expected.exists() else "failed"
        with lock:
            status["running"].remove(job.name)
            record = {
                "name": job.name,
                "returncode": proc.returncode,
                "elapsed_sec": round(time.time() - started, 3),
                "expected": str(job.expected),
                "log": str(job.log),
            }
            status["completed" if result == "ok" else "failed"].append(record)
            save_status()
        return job.name, proc.returncode, result

    def run_stage(name: str, jobs: list[Job], parallel: int) -> None:
        with lock:
            status["stage"] = name
            status["stage_jobs"] = len(jobs)
            save_status()
        failures = []
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            futures = [pool.submit(run_job, job) for job in jobs]
            for future in as_completed(futures):
                job_name, code, result = future.result()
                print(f"[{name}] {job_name}: {result} rc={code}", flush=True)
                if result == "failed":
                    failures.append(job_name)
        if failures:
            raise RuntimeError(f"{name} failed jobs: {failures}")

    for scope in ("intersect", "all_genes"):
        for model in args.models:
            model_dir = args.embeddings / scope / model
            if not model_dir.exists():
                raise FileNotFoundError(f"missing prepared embedding dir: {model_dir}")

    gene_script = args.benchmark_repo / "src/gene_level_benchmark/gene_level_benchmarks.py"
    split_root = args.benchmark_repo / "data/data_splits/gene_level_benchmark"
    gene_jobs: list[Job] = []
    for scope in ("intersect", "all_genes"):
        for task, stem in (("go", "go"), ("omim", "omim")):
            split = split_root / f"{stem}_folds_splits"
            out = args.output / "gene_level" / scope / task
            for model in args.models:
                model_dir = args.embeddings / scope / model
                command = (
                    args.python,
                    str(gene_script),
                    "--subfolder",
                    str(model_dir),
                    "--cv-fold1-pkl",
                    str(split / f"{stem}_cv_fold1_dict_all.pkl"),
                    "--cv-fold2-pkl",
                    str(split / f"{stem}_cv_fold2_dict_all.pkl"),
                    "--cv-fold3-pkl",
                    str(split / f"{stem}_cv_fold3_dict_all.pkl"),
                    "--holdout-pkl",
                    str(split / f"{stem}_holdout_dict_all.pkl"),
                    "-d",
                    str(out),
                )
                gene_jobs.append(
                    Job(
                        f"gene-{scope}-{task}-{model}",
                        command,
                        out / f"{model}_holdout_results.pkl",
                        logs / "gene_level" / scope / task / f"{model}.log",
                    )
                )
    run_stage("gene_level", gene_jobs, min(args.parallel, len(gene_jobs)))

    temporal_split = split_root / "go_generalization_folds_splits"
    temporal_script = temporal_split / "run_go_generalization_benchmark.py"
    temporal_out = args.output / "gene_level" / "intersect" / "go_temporal"
    temporal_jobs = []
    for model in args.models:
        command = (
            args.python,
            str(args.helper_dir / "run_official_temporal_go.py"),
            "--upstream-script",
            str(temporal_script),
            "--split-dir",
            str(temporal_split),
            "--subfolder",
            str(args.embeddings / "intersect" / model),
            "--out-root",
            str(temporal_out),
        )
        temporal_jobs.append(
            Job(
                f"temporal-go-{model}",
                command,
                temporal_out / "holdout" / f"{model}_go_post24_.pkl",
                logs / "gene_level" / "intersect" / "go_temporal" / f"{model}.log",
            )
        )
    run_stage("temporal_go", temporal_jobs, min(args.parallel, len(temporal_jobs)))

    write_summary(args.output, args.models)
    with lock:
        status["stage"] = "complete"
        status["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_status()
    (args.output / "GENE_LEVEL_ONLY_COMPLETE").write_text(
        time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
