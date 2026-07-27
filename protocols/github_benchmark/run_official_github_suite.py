#!/usr/bin/env python3
"""Run the pinned ylaboratory benchmark suite over prepared model embeddings."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    expected: Path
    log: Path
    cwd: Path | None = None
    priority: float = 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--andes-repo", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helper-dir", type=Path, required=True)
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
    }

    def save_status() -> None:
        status["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        tmp.replace(status_path)

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
        with job.log.open("w") as handle:
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
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [pool.submit(run_job, job) for job in jobs]
            for future in as_completed(futures):
                job_name, code, result = future.result()
                print(f"[{name}] {job_name}: {result} rc={code}", flush=True)
                if result == "failed":
                    failures.append(job_name)
        if failures:
            raise RuntimeError(f"{name} failed jobs: {failures}")

    manifest_path = args.embeddings / "embedding_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    model_dimensions = {
        name: int(metadata.get("dimensions", 1))
        for name, metadata in manifest.get("models", {}).items()
    }
    models = sorted(
        (path.name for path in (args.embeddings / "intersect").iterdir() if path.is_dir()),
        key=lambda name: (model_dimensions.get(name, 1), name),
    )
    if not models:
        raise RuntimeError("no prepared model directories")

    gene_script = args.benchmark_repo / "src/gene_level_benchmark/gene_level_benchmarks.py"
    split_root = args.benchmark_repo / "data/data_splits/gene_level_benchmark"
    gene_jobs = []
    for scope in ("intersect", "all_genes"):
        for task, stem in (("go", "go"), ("omim", "omim")):
            split = split_root / f"{stem}_folds_splits"
            out = args.output / "gene_level" / scope / task
            for model in models:
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
                        priority=model_dimensions.get(model, 1),
                    )
                )
    gene_jobs.sort(key=lambda job: job.priority)
    run_stage("gene_level", gene_jobs, parallel=10)

    temporal_split = split_root / "go_generalization_folds_splits"
    temporal_script = temporal_split / "run_go_generalization_benchmark.py"
    temporal_jobs = []
    temporal_out = args.output / "gene_level" / "intersect" / "go_temporal"
    for model in models:
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
                priority=model_dimensions.get(model, 1),
            )
        )
    run_stage("temporal_go", temporal_jobs, parallel=10)

    pair_script = args.benchmark_repo / "src/gene_pair_benchmark/gene_pair_benchmarks.py"
    pair_split_root = args.benchmark_repo / "data/data_splits/gene_pair_benchmark"
    pair_jobs = []
    pair_counts = {"sl": 23_364, "pombe": 65_626, "tf": 110_000, "ng": 116_248}
    operation_factor = {"sum": 1.0, "product": 1.0, "concat": 2.0}
    scope_factor = {"intersect": 1.0, "all_genes": 1.15}
    for scope in ("intersect", "all_genes"):
        for dataset in ("sl", "pombe", "tf", "ng"):
            split_name = "pombe_nested_cv_splits.pkl" if dataset == "pombe" else f"{dataset}_nested_cv_splits.pkl"
            for operation in ("sum", "product", "concat"):
                out = args.output / "gene_pair" / scope / dataset / operation
                suffix = f"{dataset}_{operation}_{scope}"
                for model in models:
                    command = (
                        args.python,
                        str(pair_script),
                        "--subfolder",
                        str(args.embeddings / scope / model),
                        "--cv-pkl",
                        str(pair_split_root / split_name),
                        "--operation",
                        operation,
                        "--out-root",
                        str(out),
                        "--suffix",
                        suffix,
                    )
                    pair_jobs.append(
                        Job(
                            f"pair-{scope}-{dataset}-{operation}-{model}",
                            command,
                            out / f"{model}_{suffix}.csv",
                            logs / "gene_pair" / scope / dataset / operation / f"{model}.log",
                            priority=(
                                pair_counts[dataset]
                                * operation_factor[operation]
                                * scope_factor[scope]
                                * model_dimensions.get(model, 1)
                            ),
                        )
                    )
    pair_jobs.sort(key=lambda job: job.priority)
    run_stage("gene_pair", pair_jobs, parallel=8)

    gmt = args.benchmark_repo / "data/gmt"
    gene_set_tasks = {
        "disease_tissue": (gmt / "bto_specific.gmt", gmt / "omim_entrez.gmt", (False, True)),
        "kegg_go": (gmt / "KEGG_CPDB.gmt", gmt / "hsa_low_eval_BP_propagated.gmt", (False, True)),
    }
    andes_jobs = []
    for scope in ("intersect", "all_genes"):
        for task, (geneset1, geneset2, modes) in gene_set_tasks.items():
            for distinct in modes:
                mode = "no_overlap" if distinct else "overlap"
                out = args.output / "gene_set" / scope / task / mode
                for model in models:
                    model_dir = args.embeddings / scope / model
                    emb = next(model_dir.glob("*.csv"))
                    genes = next(model_dir.glob("*.txt"))
                    matrix = out / f"{model}.csv"
                    command = [
                        args.python,
                        str(args.helper_dir / "run_official_andes.py"),
                        "--andes-root",
                        str(args.andes_repo),
                        "--emb",
                        str(emb),
                        "--genelist",
                        str(genes),
                        "--geneset1",
                        str(geneset1),
                        "--geneset2",
                        str(geneset2),
                        "--out",
                        str(matrix),
                        "--processors",
                        "20",
                    ]
                    if distinct:
                        command.append("--distinct")
                    andes_jobs.append(
                        Job(
                            f"andes-{scope}-{task}-{mode}-{model}",
                            tuple(command),
                            matrix,
                            logs / "gene_set" / scope / task / mode / f"{model}.log",
                        )
                    )
    run_stage("gene_set", andes_jobs, parallel=1)

    with lock:
        status["stage"] = "complete"
        status["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_status()


if __name__ == "__main__":
    main()
