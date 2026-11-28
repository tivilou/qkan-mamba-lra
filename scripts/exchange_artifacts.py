#!/usr/bin/env python3
"""Explicit, resumable artifact delivery. Configuration stays outside manifests."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
CHUNK = 1024 * 1024


class DeliveryError(ValueError):
    pass


def safe_path(value: str) -> str:
    parts = value.split("/")
    if not parts or any(not SEGMENT.fullmatch(p) or p in {".", ".."} for p in parts):
        raise DeliveryError("Expected a safe relative path without hidden or traversal segments")
    return "/".join(parts)


def local_file(root: Path, relative: str) -> Path:
    safe_path(relative)
    candidate = root
    for part in relative.split("/"):
        candidate = candidate / part
        if candidate.is_symlink():
            raise DeliveryError("Symlinks are not permitted in declared file paths")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise DeliveryError("Declared file is missing or outside the repository")
    return resolved


def digest_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def prepare(root: Path, experiment: str, run_id: str, files: list[str]) -> dict:
    if "/" in experiment or "/" in run_id:
        raise DeliveryError("experiment and run-id must each be one segment")
    target = safe_path(f"qkan-mamba/{experiment}/{run_id}")
    records, seen = [], set()
    for mapping in files:
        relative, sep, remote = mapping.partition("=")
        remote = safe_path(remote if sep else Path(relative).name)
        if remote in seen:
            raise DeliveryError("Duplicate remote destination")
        seen.add(remote)
        size, digest = digest_file(local_file(root, relative))
        records.append({"local_path": relative, "path": remote, "size_bytes": size,
                        "sha256": digest, "status": "pending"})
    if not records:
        raise DeliveryError("At least one explicitly declared file is required")
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip())
    return {"schema_version": "qkan-delivery.v1", "project": "qkan-mamba",
            "target_dir": target, "source_revision": revision, "source_dirty": dirty,
            "status": "prepared", "files": records}


class Transport:
    def __init__(self, url: str, token: str, ca_file: str | None = None,
                 allow_http: bool = False):
        parsed = urlsplit(url)
        if (parsed.scheme not in {"https", "http"} or not parsed.hostname or
                parsed.username or parsed.password or parsed.query or parsed.fragment or
                parsed.path not in {"", "/"}):
            raise DeliveryError("Exchange URL must be a bare HTTP(S) origin without credentials")
        if parsed.scheme == "http" and not (allow_http and parsed.hostname in {"127.0.0.1", "localhost", "::1"}):
            raise DeliveryError("HTTPS is required; HTTP is allowed only for explicit loopback tests")
        if not token or any(c in token for c in "\r\n"):
            raise DeliveryError("PROJECT_EXCHANGE_TOKEN is missing or invalid")
        self.parsed, self.token = parsed, token
        self.context = ssl.create_default_context(cafile=ca_file) if parsed.scheme == "https" else None

    def request(self, method: str, route: str, body: bytes | Path | None = None,
                size: int = 0) -> tuple[int, dict]:
        connection = (http.client.HTTPSConnection(self.parsed.hostname, self.parsed.port,
                      timeout=120, context=self.context) if self.context else
                      http.client.HTTPConnection(self.parsed.hostname, self.parsed.port, timeout=120))
        try:
            connection.putrequest(method, route)
            connection.putheader("Authorization", "Bearer " + self.token)
            if body is not None:
                connection.putheader("Content-Length", str(size))
                connection.putheader("Content-Type", "application/json" if isinstance(body, bytes) else "application/octet-stream")
            connection.endheaders()
            if isinstance(body, bytes):
                connection.send(body)
            elif isinstance(body, Path):
                with body.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(CHUNK), b""):
                        connection.send(chunk)
            response = connection.getresponse()
            raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise DeliveryError("Exchange response exceeded its bounded size")
            if response.status not in {200, 201}:
                raise DeliveryError(f"Exchange returned HTTP {response.status}")
            return response.status, json.loads(raw)
        finally:
            connection.close()


def validate_manifest(root: Path, manifest: dict) -> None:
    if manifest.get("schema_version") != "qkan-delivery.v1" or manifest.get("project") != "qkan-mamba":
        raise DeliveryError("Unsupported manifest identity")
    target = safe_path(manifest.get("target_dir", ""))
    if len(target.split("/")) != 3 or target.split("/")[0] != "qkan-mamba":
        raise DeliveryError("Manifest must use qkan-mamba/experiment/run-id")
    if not manifest.get("files"):
        raise DeliveryError("Manifest contains no files")
    seen = set()
    for record in manifest["files"]:
        remote = safe_path(record["path"])
        if remote in seen:
            raise DeliveryError("Duplicate remote destination")
        seen.add(remote)
        if digest_file(local_file(root, record["local_path"])) != (record["size_bytes"], record["sha256"]):
            raise DeliveryError("Declared source changed; prepare a new manifest and run identity")


def deliver(root: Path, path: Path, transport: Transport) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(root, manifest)
    manifest["status"] = "pending"
    manifest.pop("error", None)
    atomic_json(path, manifest)
    try:
        _, health = transport.request("GET", "/healthz")
        maximum = health.get("max_bytes")
        if health.get("status") != "ok" or not isinstance(maximum, int) or maximum <= 0:
            raise DeliveryError("Invalid service health response")
        if any(record["size_bytes"] > maximum for record in manifest["files"]):
            raise DeliveryError("A declared file exceeds the service limit; use an approved chunk manifest")
        body = json.dumps({"path": manifest["target_dir"]}).encode()
        transport.request("POST", "/api/directories", body, len(body))
        for record in manifest["files"]:
            remote = manifest["target_dir"] + "/" + record["path"]
            _, receipt = transport.request("PUT", "/upload/" + remote,
                        local_file(root, record["local_path"]), record["size_bytes"])
            receipt_path = safe_path(str(receipt.get("path", "")).replace("\\", "/"))
            if (receipt_path != remote or receipt.get("size_bytes") != record["size_bytes"] or
                    receipt.get("sha256") != record["sha256"]):
                raise DeliveryError("Upload receipt does not match path, size and SHA-256")
            record["status"] = "verified"
            record["receipt"] = {key: receipt[key] for key in ("path", "size_bytes", "sha256")}
            record["receipt"]["path"] = receipt_path
            atomic_json(path, manifest)
        manifest["status"] = "uploaded"
        atomic_json(path, manifest)
        return manifest
    except Exception:
        manifest["error"] = "Delivery incomplete; correct the connection or conflict and retry this manifest"
        atomic_json(path, manifest)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("--experiment", required=True)
    create.add_argument("--run-id", required=True)
    create.add_argument("--file", action="append", required=True)
    create.add_argument("--manifest", type=Path, required=True)
    upload = sub.add_parser("upload")
    upload.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise DeliveryError("Manifest output must remain inside the repository")
    if args.command == "prepare":
        if path.exists():
            raise DeliveryError("Refusing to overwrite an existing manifest")
        manifest = prepare(ROOT, args.experiment, args.run_id, args.file)
        atomic_json(path, manifest)
    else:
        manifest = deliver(ROOT, path, Transport(os.environ.get("PROJECT_EXCHANGE_URL", ""),
                           os.environ.get("PROJECT_EXCHANGE_TOKEN", ""),
                           os.environ.get("PROJECT_EXCHANGE_CA_FILE") or None))
    print(json.dumps({"status": manifest["status"], "files": len(manifest["files"])}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DeliveryError, OSError, ValueError, KeyError, http.client.HTTPException) as error:
        # Avoid raw URLs, credential-bearing headers, response bodies and tracebacks.
        print("Delivery failed (" + type(error).__name__ + "); check local configuration and pending manifest", file=sys.stderr)
        raise SystemExit(2)
