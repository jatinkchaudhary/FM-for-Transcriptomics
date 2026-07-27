#!/usr/bin/env python3
"""Serve the Studio UI, measured results, and GPU imputation API."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import traceback
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch

from model_runtime import ModelRuntime, RequestError, UnsupportedModelError


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parent
FRONTEND_ROOT = APP_ROOT / "frontend"
DATA_ROOT = APP_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"
TEST_ROOT = REPO_ROOT / "test_data"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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
        route = urlparse(self.path).path
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
        if route == "/api/experiments":
            self._json(self.server.results)
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
        self._file(FRONTEND_ROOT / unquote(route.lstrip("/")))

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route not in {"/api/impute", "/api/downstream"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 25 * 1024 * 1024:
                raise RequestError("request body must be between 1 byte and 25 MB")
            payload = json.loads(self.rfile.read(length))
            if route == "/api/impute":
                result = self.server.runtime.impute(payload)
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
    args = parser.parse_args()
    results = load_json(DATA_ROOT / "results_registry.json")
    models = results["models"]
    runtime = ModelRuntime(args.model_config, models)
    server = ThreadingHTTPServer((args.host, args.port), StudioHandler)
    server.runtime = runtime
    server.models = models
    server.results = results
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
