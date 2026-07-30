#!/usr/bin/env python3
"""Serve the Studio UI, measured results, and GPU imputation API."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import traceback
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import torch

from model_runtime import ModelRuntime, RequestError, UnsupportedModelError
from atlas_runtime import AtlasRuntime, AtlasUnavailableError


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parent
FRONTEND_ROOT = APP_ROOT / "frontend"
DATA_ROOT = APP_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"
TEST_ROOT = REPO_ROOT / "test_data"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class StudioHandler(SimpleHTTPRequestHandler):
    server_version = "TxnJatinStudio/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def _json(self, payload, status=HTTPStatus.OK) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        roots = (FRONTEND_ROOT.resolve(), RESULTS_ROOT.resolve(), TEST_ROOT.resolve())
        if not any(resolved == root or root in resolved.parents for root in roots):
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        body = resolved.read_bytes()
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        if route == "/api/health":
            runtime = self.server.runtime
            self._json(
                {
                    "status": "ok",
                    "cuda": torch.cuda.is_available(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "loaded_model": runtime.model_name,
                    "pid": os.getpid(),
                }
            )
            return
        if route == "/api/models":
            self._json({"models": self.server.models})
            return
        if route == "/api/atlas/status":
            self._json(self.server.atlas.status())
            return
        if route == "/api/experiments":
            self._json(self.server.results)
            return
        if route == "/api/osdr":
            root = self.server.osdr_root
            try:
                summary = load_json(root / "summary.json")
                atlas_summary = load_json(root / "atlas" / "atlas_summary.json")
                metrics = load_csv(root / "sample_metrics.csv")
                matches = {
                    row["sample_id"]: row
                    for row in load_csv(root / "atlas" / "atlas_matches.csv")
                }
            except FileNotFoundError:
                self._json(
                    {"error": "osdr_results_unavailable"},
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            search = query.get("search", [""])[0].strip().lower()
            if search:
                metrics = [
                    row
                    for row in metrics
                    if search
                    in " ".join(
                        str(row.get(key, ""))
                        for key in ("sample_id", "accession", "condition", "tissue")
                    ).lower()
                ]
            page = max(1, int(query.get("page", ["1"])[0]))
            page_size = min(200, max(10, int(query.get("page_size", ["50"])[0])))
            total = len(metrics)
            start = (page - 1) * page_size
            rows = []
            for row in metrics[start : start + page_size]:
                match = matches.get(row["sample_id"], {})
                rows.append(
                    {
                        **row,
                        **{
                            key: match.get(key)
                            for key in (
                                "sample_index",
                                "predicted_species",
                                "species_weight",
                                "predicted_tissue",
                                "top_similarity",
                                "top_reference",
                            )
                        },
                    }
                )
            report_dir = root / "atlas" / "llm_reports"
            valid = len(list(report_dir.glob("*.md"))) if report_dir.is_dir() else 0
            errors = (
                len(list(report_dir.glob("*.error.json"))) if report_dir.is_dir() else 0
            )
            self._json(
                {
                    "summary": summary,
                    "atlas_summary": atlas_summary,
                    "reports": {"valid": valid, "errors": errors, "total": summary["samples"]},
                    "pagination": {
                        "page": page,
                        "page_size": page_size,
                        "total": total,
                        "pages": max(1, (total + page_size - 1) // page_size),
                    },
                    "samples": rows,
                }
            )
            return
        if route == "/api/osdr/report":
            root = self.server.osdr_root / "atlas" / "llm_reports"
            raw_index = query.get("sample_index", [""])[0]
            try:
                sample_index = int(raw_index)
            except ValueError:
                self._json({"error": "invalid_sample_index"}, HTTPStatus.BAD_REQUEST)
                return
            matches = list(root.glob(f"{sample_index:04d}_*.md"))
            if not matches:
                errors = list(root.glob(f"{sample_index:04d}_*.error.json"))
                self._json(
                    {
                        "status": "failed_validation" if errors else "pending",
                        "sample_index": sample_index,
                    }
                )
                return
            self._json(
                {
                    "status": "complete",
                    "sample_index": sample_index,
                    "text": matches[0].read_text(encoding="utf-8"),
                }
            )
            return
        if route in {"/", "/index.html", "/index.dc.html"}:
            self._file(FRONTEND_ROOT / "index.dc.html")
            return
        if route == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if route.startswith("/results/"):
            self._file(RESULTS_ROOT / unquote(route.removeprefix("/results/")))
            return
        if route.startswith("/test-data/"):
            self._file(TEST_ROOT / unquote(route.removeprefix("/test-data/")))
            return
        if route.startswith("/osdr-assets/"):
            name = Path(unquote(route.removeprefix("/osdr-assets/"))).name
            if name not in {
                "imputation_performance.png",
                "atlas_tissue_assignments.png",
                "tissue_mapping_matrix.png",
                "gene_disease_knowledge_graph.png",
            }:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            path = self.server.osdr_root / "atlas" / name
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
            return
        self._file(FRONTEND_ROOT / unquote(route.lstrip("/")))

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route not in {"/api/impute", "/api/downstream", "/api/atlas"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 25 * 1024 * 1024:
                raise RequestError("request body must be between 1 byte and 25 MB")
            payload = json.loads(self.rfile.read(length))
            if route == "/api/impute":
                result = self.server.runtime.impute(payload)
            elif route == "/api/atlas":
                genes, samples, values, missing = ModelRuntime._validate_payload(
                    payload, require_missing=False, allow_negative=True
                )
                if missing.any():
                    raise RequestError(
                        "atlas analysis requires a completed matrix; run imputation first"
                    )
                if self.server.atlas.config.get("ollama", {}).get("enabled", False):
                    # gpt-oss:120b and the expression decoder cannot safely share
                    # one 80 GB GPU. Atlas matching is CPU-only, so release the
                    # decoder before the sequential language-head stage.
                    self.server.runtime.unload()
                result = self.server.atlas.analyze(genes, samples, values)
            else:
                result = self.server.runtime.analyze_downstream(payload)
            self._json(result)
        except UnsupportedModelError as error:
            self._json(
                {
                    "error": "imputation_unsupported",
                    "message": str(error),
                    "imputation_output": "NaN",
                },
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        except (RequestError, json.JSONDecodeError) as error:
            self._json({"error": "invalid_request", "message": str(error)}, HTTPStatus.BAD_REQUEST)
        except AtlasUnavailableError as error:
            self._json(
                {"error": "atlas_unavailable", "message": str(error)},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
        except Exception as error:
            traceback.print_exc()
            self._json(
                {"error": "inference_failed", "message": str(error)},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "config" / "model_paths.remote.json",
    )
    parser.add_argument(
        "--osdr-results-root",
        type=Path,
        default=Path(
            os.environ.get(
                "OSDR_RESULTS_ROOT",
                "/media/volume/AdditionalHeadroom/osdr_all_samples_20260729",
            )
        ),
    )
    args = parser.parse_args()
    results = load_json(DATA_ROOT / "results_registry.json")
    models = results["models"]
    runtime = ModelRuntime(args.model_config, models)
    model_config = load_json(args.model_config)
    atlas = AtlasRuntime(model_config.get("atlas"))
    server = ThreadingHTTPServer((args.host, args.port), StudioHandler)
    server.runtime = runtime
    server.atlas = atlas
    server.models = models
    server.results = results
    server.osdr_root = args.osdr_results_root
    print(
        json.dumps(
            {
                "status": "listening",
                "host": args.host,
                "port": args.port,
                "cuda": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        runtime.unload()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
