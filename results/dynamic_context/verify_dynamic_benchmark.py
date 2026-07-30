from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EXPECTED_MODELS = {
    "Txn_Jatin",
    "BRIDGE",
    "scGPT",
    "Geneformer",
    "BulkFormer_37M",
    "BulkFormer_50M",
    "BulkFormer_93M",
    "BulkFormer_127M",
    "BulkFormer_147M",
}
DYNAMIC_MODELS = {"Txn_Jatin", "BRIDGE"}
EXPECTED_SAMPLES = 978_212
EXPECTED_FILES = 863


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message, failures):
    if not condition:
        failures.append(message)


def main() -> int:
    failures = []

    for name in (
        "txn_jatin_dynamic_vs_all_benchmark.status",
        "txn_jatin_dynamic_paired_supplement.status",
    ):
        path = ROOT / name
        require(path.exists(), f"missing status: {name}", failures)
        if path.exists():
            require(path.read_text().strip() == "0",
                    f"nonzero status: {name}", failures)

    audit_path = ROOT / "dynamic_embedding_audit.json"
    require(audit_path.exists(), "missing dynamic embedding audit", failures)
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        require(set(audit) == EXPECTED_MODELS,
                f"audit models differ: {sorted(audit)}", failures)
        observed_dynamic = {
            model
            for model, item in audit.items()
            if item["provenance"].get("dynamic_embedding", False)
        }
        require(observed_dynamic == DYNAMIC_MODELS,
                f"dynamic model set differs: {observed_dynamic}", failures)
        for model, item in audit.items():
            require(not item["fallback"], f"{model} used fallback", failures)
            embedding = ROOT / "embeddings" / (
                f"{model}__gene_benchmark__symbol.npz"
            )
            require(embedding.exists(), f"missing embedding: {model}", failures)
            if model in DYNAMIC_MODELS:
                prov = item["provenance"]
                require(prov.get("context_samples") == EXPECTED_SAMPLES,
                        f"{model} sample count incorrect", failures)
                require(prov.get("context_files") == EXPECTED_FILES,
                        f"{model} file count incorrect", failures)
                require(bool(prov.get("context_embedding_sha256")),
                        f"{model} missing dynamic SHA", failures)
            else:
                require(
                    not item["provenance"].get("dynamic_embedding", False),
                    f"{model} unexpectedly dynamic",
                    failures,
                )
                require(
                    item["provenance"].get("context_samples") is None,
                    f"{model} unexpectedly has context samples",
                    failures,
                )
            if embedding.exists():
                with np.load(embedding, allow_pickle=True) as payload:
                    require(list(payload["emb"].shape) == item["shape"],
                            f"{model} embedding shape mismatch", failures)
                    require(not bool(payload["is_fallback"]),
                            f"{model} NPZ marked fallback", failures)

    leaderboard_path = ROOT / "tables" / "leaderboard_gene_tracks.csv"
    require(leaderboard_path.exists(), "missing leaderboard", failures)
    if leaderboard_path.exists():
        leaderboard = pd.read_csv(leaderboard_path)
        require(EXPECTED_MODELS.issubset(leaderboard.columns),
                "leaderboard lacks one or more models", failures)
        require(set(leaderboard["track"]) == {
            "gene-set", "paired", "single-gene-GO", "single-gene-disease"
        }, "leaderboard track set incorrect", failures)
        require(set(leaderboard["variant"]) == {"common", "full"},
                "leaderboard variants incorrect", failures)

    required_figures = {
        "cross_species_corr.png",
        "intermodel_similarity.png",
        "leaderboard.png",
        "performance_correlation.png",
        "sample_benchmark.png",
        "dynamic_story_dashboard.png",
        "dynamic_story_dashboard.pdf",
        "dynamic_story_01_landscape.png",
        "dynamic_story_01_landscape.pdf",
        "dynamic_story_02_gain_vs_static.png",
        "dynamic_story_02_gain_vs_static.pdf",
        "dynamic_story_03_txn_vs_bridge.png",
        "dynamic_story_03_txn_vs_bridge.pdf",
        "dynamic_story_04_similarity.png",
        "dynamic_story_04_similarity.pdf",
    }
    observed_figures = {
        path.name for path in (ROOT / "figures").glob("*") if path.is_file()
    }
    require(required_figures.issubset(observed_figures),
            f"missing figures: {sorted(required_figures - observed_figures)}",
            failures)

    remote_manifest = ROOT / "REMOTE_SHA256SUMS.txt"
    require(remote_manifest.exists(), "missing remote hash manifest", failures)
    if remote_manifest.exists():
        for line in remote_manifest.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            relative = relative.removeprefix("./")
            path = ROOT / "remote_original" / relative
            require(path.exists(), f"remote copy missing: {relative}", failures)
            if path.exists():
                require(sha256(path) == expected,
                        f"remote hash mismatch: {relative}", failures)

    local_manifest = ROOT / "SHA256SUMS.txt"
    if local_manifest.exists():
        for line in local_manifest.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            path = ROOT / relative
            require(path.exists(), f"local package file missing: {relative}",
                    failures)
            if path.exists():
                require(sha256(path) == expected,
                        f"local package hash mismatch: {relative}", failures)

    main_log = ROOT / "txn_jatin_dynamic_vs_all_benchmark.log"
    if main_log.exists():
        text = main_log.read_text(errors="replace")
        require("FAIL: 0" in text, "Council did not report zero failures", failures)
        require(
            "Txn_Jatin contextual gene embeddings  -- 978212 context samples "
            "across 863 files" in text,
            "Txn_Jatin full-context completion absent from log",
            failures,
        )
        require(
            "BRIDGE contextual gene embeddings  -- 978212 context samples "
            "across 863 files" in text,
            "BRIDGE full-context completion absent from log",
            failures,
        )
    else:
        failures.append("missing main benchmark log")

    if failures:
        print("\n".join(f"FAIL: {item}" for item in failures))
        return 1
    print(
        "Dynamic benchmark verified: exactly two full-corpus dynamic models, "
        "seven static comparison models, nine real embeddings, four tracks, "
        "all required figures, status 0, Council FAIL 0, and exact remote hashes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
