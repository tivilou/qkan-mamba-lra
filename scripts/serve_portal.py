#!/usr/bin/env python3
"""Serve a project research portal and explicitly declared static mounts."""

from __future__ import annotations

import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes allowed root: {path}") from exc
    return resolved


def parse_mount(value: str, project_root: Path) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("mount must use /url-prefix=relative-target")
    prefix, raw_target = value.split("=", 1)
    prefix = "/" + prefix.lstrip("/")
    if prefix == "/" or ".." in prefix.split("/"):
        raise ValueError("mount URL prefix must be a non-root path without '..'")
    target_value = Path(raw_target)
    if target_value.is_absolute():
        raise ValueError("mount target must be relative to --project-root")
    target = inside(project_root / target_value, project_root)
    if not target.exists() or not target.is_dir():
        raise ValueError(f"mount target must be an existing directory: {target}")
    return prefix.rstrip("/"), target


class PortalHandler(SimpleHTTPRequestHandler):
    portal_root: Path
    mounts: list[tuple[str, Path]]

    def _safe_path(self, root: Path, relative: str) -> str:
        if any(part == ".." for part in relative.replace("\\", "/").split("/")):
            raise ValueError("path traversal is not allowed")
        candidate = inside(root / relative.lstrip("/"), root)
        return os.fspath(candidate)

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/__health":
            self.send_health()
            return
        try:
            super().do_GET()
        except ValueError as exc:
            self.send_error(403, str(exc))

    def do_HEAD(self) -> None:
        if urlsplit(self.path).path == "/__health":
            body = self.health_body()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return
        try:
            super().do_HEAD()
        except ValueError as exc:
            self.send_error(403, str(exc))

    def health_body(self) -> bytes:
        return json.dumps(
            {"status": "ok", "service": "research-web-portal", "pid": os.getpid()},
            ensure_ascii=False,
        ).encode("utf-8")

    def send_health(self) -> None:
        body = self.health_body()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def translate_path(self, request_path: str) -> str:
        path = unquote(urlsplit(request_path).path)
        if path in {"", "/"}:
            return self._safe_path(self.portal_root, "index.html")
        for prefix, target in self.mounts:
            if path == prefix or path.startswith(prefix + "/"):
                return self._safe_path(target, path[len(prefix):])
        return self._safe_path(self.portal_root, path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve a loopback-only research web portal")
    parser.add_argument("--portal-root", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--mount", action="append", default=[], metavar="/prefix=relative-target")
    args = parser.parse_args()
    if args.host not in LOOPBACK_HOSTS:
        raise SystemExit("Portal only accepts loopback hosts")
    project_root = args.project_root.resolve()
    portal_root = inside(args.portal_root, project_root)
    if not portal_root.is_dir():
        raise SystemExit(f"portal root must be an existing directory: {portal_root}")
    if not (portal_root / "index.html").is_file():
        raise SystemExit(f"portal root is missing index.html: {portal_root}")
    try:
        mounts = [parse_mount(item, project_root) for item in args.mount]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    prefixes = [prefix for prefix, _ in mounts]
    if len(prefixes) != len(set(prefixes)):
        raise SystemExit("mount URL prefixes must be unique")
    PortalHandler.portal_root = portal_root
    PortalHandler.mounts = sorted(mounts, key=lambda item: len(item[0]), reverse=True)
    server = ThreadingHTTPServer((args.host, args.port), PortalHandler)
    print(f"Research portal: http://{args.host}:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
