#!/usr/bin/env python3
"""Authenticated upload and read-only artifact exchange service."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import html
from http.cookies import CookieError, SimpleCookie
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import tempfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit


SERVICE_VERSION = "qore-upload-v4"
SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
UPLOAD_PREFIX = "/upload/"
FILE_PREFIX = "/files/"
SESSION_COOKIE = "qore_upload_session"
BUFFER_SIZE = 1024 * 1024
PREVIEW_BYTES = 256 * 1024
PREVIEW_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".text",
    ".txt",
    ".yaml",
    ".yml",
}
PUBLIC_ASSETS = {
    "/assets/app.js": "app.js",
    "/assets/lucide.min.js": "lucide.min.js",
    "/assets/styles.css": "styles.css",
}
CONTENT_TYPES = {
    "app.js": "text/javascript; charset=utf-8",
    "index.html": "text/html; charset=utf-8",
    "login.html": "text/html; charset=utf-8",
    "lucide.min.js": "text/javascript; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


class UploadServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 16

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        root: Path,
        web_root: Path,
        token: str,
        max_bytes: int,
        namespace: str,
    ) -> None:
        self.root = root.resolve()
        self.web_root = web_root.resolve()
        self.token = token
        self.max_bytes = max_bytes
        self.namespace = namespace
        self.session_value = hmac.new(
            token.encode("utf-8"),
            b"qore-upload-browser-session-v1",
            hashlib.sha256,
        ).hexdigest()
        super().__init__(server_address, handler_class)


class UploadHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "QoreUpload/4.0"

    @property
    def upload_server(self) -> UploadServer:
        return self.server  # type: ignore[return-value]

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        request_path = parsed.path
        if request_path == "/healthz":
            self._json(
                200,
                {
                    "status": "ok",
                    "service": SERVICE_VERSION,
                    "max_bytes": self.upload_server.max_bytes,
                },
            )
            return
        if request_path in PUBLIC_ASSETS:
            self._serve_web_file(PUBLIC_ASSETS[request_path])
            return
        if request_path == "/login":
            if self._browser_authorized():
                self._redirect("/")
            else:
                self._serve_login()
            return
        if request_path == "/":
            if not self._browser_authorized():
                self._redirect("/login")
                return
            self._serve_web_file("index.html")
            return
        if request_path == "/api/files":
            if not self._authorized():
                self._json(401, {"error": "unauthorized"})
                return
            requested = parse_qs(parsed.query, keep_blank_values=True).get(
                "path", [self.upload_server.namespace]
            )[0]
            self._serve_directory(requested)
            return
        if request_path == "/api/preview":
            if not self._authorized():
                self._json(401, {"error": "unauthorized"})
                return
            requested = parse_qs(parsed.query, keep_blank_values=True).get("path", [""])[0]
            self._serve_preview(requested)
            return
        if request_path.startswith(FILE_PREFIX):
            if not self._authorized():
                self._json(401, {"error": "unauthorized"})
                return
            self._serve_download(unquote(request_path[len(FILE_PREFIX) :]), send_body=True)
            return
        self._json(404, {"error": "not_found"})

    def do_HEAD(self) -> None:
        request_path = urlsplit(self.path).path
        if request_path.startswith(FILE_PREFIX):
            if not self._authorized():
                self._json(401, {"error": "unauthorized"}, send_body=False)
                return
            self._serve_download(
                unquote(request_path[len(FILE_PREFIX) :]),
                send_body=False,
            )
            return
        if not self._authorized():
            self._json(401, {"error": "unauthorized"}, send_body=False)
            return
        target = self._resolve_upload_target()
        if target is None or not target.is_file() or target.is_symlink():
            self._json(404, {"error": "not_found"}, send_body=False)
            return
        stat = target.stat()
        self.send_response(200)
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header("ETag", f'"sha256:{sha256_file(target)}"')
        self._common_headers()
        self.end_headers()

    def do_POST(self) -> None:
        request_path = urlsplit(self.path).path
        if request_path == "/login":
            self._handle_login()
            return
        if request_path == "/api/directories":
            if not self._authorized():
                self._json(401, {"error": "unauthorized"})
                return
            self._handle_create_directory()
            return
        if request_path == "/logout":
            self._redirect(
                "/login",
                cookie=f"{SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0",
            )
            return
        self._json(404, {"error": "not_found"})

    def do_PUT(self) -> None:
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        target = self._resolve_upload_target()
        if target is None:
            self._json(400, {"error": "invalid_upload_path"})
            return
        if self.headers.get("Transfer-Encoding"):
            self._json(411, {"error": "content_length_required"})
            return
        raw_length = self.headers.get("Content-Length")
        try:
            content_length = int(raw_length or "")
        except ValueError:
            self._json(411, {"error": "content_length_required"})
            return
        if content_length < 0 or content_length > self.upload_server.max_bytes:
            self._json(
                413,
                {"error": "file_too_large", "max_bytes": self.upload_server.max_bytes},
            )
            return

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            parent = target.parent.resolve()
            parent.relative_to(self.upload_server.root)
        except ValueError:
            self._json(400, {"error": "invalid_upload_path"})
            return
        except OSError:
            self._json(403, {"error": "upload_path_unavailable"})
            return
        try:
            is_symlink = target.is_symlink()
        except OSError:
            self._json(403, {"error": "upload_path_unavailable"})
            return
        if is_symlink:
            self._json(409, {"error": "symlink_target_rejected"})
            return

        digest = hashlib.sha256()
        received = 0
        temp_path: Path | None = None
        self.connection.settimeout(300)
        try:
            fd, raw_temp_path = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".uploading", dir=parent
            )
            temp_path = Path(raw_temp_path)
            with os.fdopen(fd, "wb") as destination:
                remaining = content_length
                while remaining:
                    block = self.rfile.read(min(BUFFER_SIZE, remaining))
                    if not block:
                        raise ConnectionError("upload ended before Content-Length bytes arrived")
                    destination.write(block)
                    digest.update(block)
                    received += len(block)
                    remaining -= len(block)
                destination.flush()
                os.fsync(destination.fileno())

            uploaded_hash = digest.hexdigest()
            if target.exists():
                existing_hash = sha256_file(target)
                if target.stat().st_size == received and hmac.compare_digest(
                    existing_hash, uploaded_hash
                ):
                    temp_path.unlink()
                    temp_path = None
                    self._json(
                        200,
                        self._receipt(target, received, uploaded_hash, "already_present"),
                    )
                    return
                temp_path.unlink()
                temp_path = None
                self._json(
                    409,
                    {
                        "error": "target_exists_with_different_content",
                        "path": str(target.relative_to(self.upload_server.root)),
                    },
                )
                return

            os.chmod(temp_path, 0o640)
            try:
                # Publish without ever overwriting a concurrent upload.
                os.link(temp_path, target)
            except FileExistsError:
                existing_hash = sha256_file(target)
                temp_path.unlink()
                temp_path = None
                if target.stat().st_size == received and hmac.compare_digest(
                    existing_hash, uploaded_hash
                ):
                    self._json(
                        200,
                        self._receipt(target, received, uploaded_hash, "already_present"),
                    )
                    return
                self._json(
                    409,
                    {
                        "error": "target_exists_with_different_content",
                        "path": str(target.relative_to(self.upload_server.root)),
                    },
                )
                return
            temp_path.unlink()
            temp_path = None
            receipt = self._receipt(target, received, uploaded_hash, "stored")
            self._write_receipt(target, receipt)
            self._json(201, receipt)
        except (ConnectionError, OSError, socket.timeout) as exc:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            self._json(400, {"error": "upload_failed", "detail": type(exc).__name__})

    def do_DELETE(self) -> None:
        self._json(405, {"error": "method_not_allowed"})

    def do_PATCH(self) -> None:
        self._json(405, {"error": "method_not_allowed"})

    def _authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.upload_server.token}"
        return hmac.compare_digest(supplied, expected) or self._browser_authorized()

    def _browser_authorized(self) -> bool:
        raw_cookie = self.headers.get("Cookie", "")
        if not raw_cookie:
            return False
        cookie = SimpleCookie()
        try:
            cookie.load(raw_cookie)
        except CookieError:
            return False
        supplied = cookie.get(SESSION_COOKIE)
        return supplied is not None and hmac.compare_digest(
            supplied.value,
            self.upload_server.session_value,
        )

    def _resolve_upload_target(self) -> Path | None:
        raw_path = unquote(urlsplit(self.path).path)
        if not raw_path.startswith(UPLOAD_PREFIX):
            return None
        return self._resolve_relative(raw_path[len(UPLOAD_PREFIX) :])

    def _resolve_relative(self, relative_text: str) -> Path | None:
        if not relative_text or len(relative_text) > 1024:
            return None
        segments = relative_text.split("/")
        if len(segments) < 2 or segments[0] != self.upload_server.namespace:
            return None
        if any(
            segment in {".", ".."}
            or segment.startswith(".")
            or not SAFE_SEGMENT.fullmatch(segment)
            for segment in segments
        ):
            return None
        if segments[-1].endswith(".upload.json"):
            return None
        target = self.upload_server.root.joinpath(*segments)
        try:
            target = target.resolve(strict=False)
            target.relative_to(self.upload_server.root)
        except ValueError:
            return None
        return target

    def _resolve_directory(self, relative_text: str) -> Path | None:
        if not relative_text or len(relative_text) > 1024:
            return None
        segments = relative_text.split("/")
        if not segments or segments[0] != self.upload_server.namespace:
            return None
        if any(
            segment in {".", ".."}
            or segment.startswith(".")
            or not SAFE_SEGMENT.fullmatch(segment)
            for segment in segments
        ):
            return None
        target = self.upload_server.root.joinpath(*segments)
        current = self.upload_server.root
        for segment in segments[1:]:
            current = current / segment
            try:
                if current.is_symlink():
                    return None
            except OSError:
                return None
        try:
            target = target.resolve(strict=False)
            target.relative_to(self.upload_server.root)
        except ValueError:
            return None
        return target

    def _handle_create_directory(self) -> None:
        if self.headers.get("Transfer-Encoding"):
            self._json(411, {"error": "content_length_required"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._json(411, {"error": "content_length_required"})
            return
        if content_length <= 0 or content_length > 8192:
            self._json(413, {"error": "directory_request_too_large"})
            return
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            self._json(400, {"error": "invalid_directory_request"})
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("path"), str):
            self._json(400, {"error": "directory_path_required"})
            return
        relative_text = payload["path"]
        if (
            relative_text == self.upload_server.namespace
            or len(relative_text.split("/")) < 2
        ):
            self._json(400, {"error": "directory_path_must_be_nested"})
            return
        target = self._resolve_directory(relative_text)
        if target is None:
            self._json(400, {"error": "invalid_directory_path"})
            return
        try:
            existed = target.exists()
            if target.is_symlink():
                self._json(409, {"error": "symlink_directory_rejected"})
                return
            target.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or not target.is_dir():
                self._json(409, {"error": "directory_target_rejected"})
                return
            relative = target.relative_to(self.upload_server.root).as_posix()
        except PermissionError:
            self._json(403, {"error": "directory_creation_forbidden"})
            return
        except (OSError, ValueError):
            self._json(403, {"error": "directory_creation_unavailable"})
            return
        self._json(
            200 if existed else 201,
            {
                "status": "already_present" if existed else "created",
                "service": SERVICE_VERSION,
                "path": relative,
            },
        )

    def _serve_directory(self, relative_text: str) -> None:
        target = self._resolve_directory(relative_text)
        if target is None:
            self._json(400, {"error": "invalid_directory_path"})
            return
        try:
            if target.is_symlink() or not target.is_dir():
                self._json(404, {"error": "directory_not_found"})
                return
            entries = list(os.scandir(target))
        except OSError:
            self._json(403, {"error": "directory_unavailable"})
            return

        directories: list[dict[str, Any]] = []
        files: list[dict[str, Any]] = []
        for entry in entries:
            if entry.name.startswith(".") or entry.name.endswith(".upload.json"):
                continue
            try:
                if entry.is_symlink():
                    continue
                stat = entry.stat(follow_symlinks=False)
                entry_target = Path(entry.path)
                entry_relative = entry_target.relative_to(self.upload_server.root)
                updated_at = datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat()
                if entry.is_dir(follow_symlinks=False):
                    directories.append(
                        {
                            "name": entry.name,
                            "path": entry_relative.as_posix(),
                            "updated_at": updated_at,
                        }
                    )
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
            except (OSError, ValueError):
                continue

            receipt_hash: str | None = None
            receipt_path = entry_target.with_name(f"{entry.name}.upload.json")
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                candidate_hash = receipt.get("sha256")
                if isinstance(candidate_hash, str) and re.fullmatch(
                    r"[0-9a-f]{64}", candidate_hash
                ):
                    receipt_hash = candidate_hash
            except (OSError, ValueError, TypeError):
                pass
            files.append(
                {
                    "name": entry.name,
                    "path": entry_relative.as_posix(),
                    "size_bytes": stat.st_size,
                    "sha256": receipt_hash,
                    "updated_at": updated_at,
                    "previewable": entry_target.suffix.lower() in PREVIEW_SUFFIXES,
                }
            )

        directories.sort(key=lambda item: item["name"].casefold())
        files.sort(key=lambda item: item["name"].casefold())
        relative = target.relative_to(self.upload_server.root)
        parent = (
            None
            if relative.as_posix() == self.upload_server.namespace
            else relative.parent.as_posix()
        )
        self._json(
            200,
            {
                "path": relative.as_posix(),
                "parent": parent,
                "directories": directories,
                "files": files,
            },
        )

    def _serve_preview(self, relative_text: str) -> None:
        target = self._resolve_relative(relative_text)
        if (
            target is None
            or target.suffix.lower() not in PREVIEW_SUFFIXES
            or not target.is_file()
            or target.is_symlink()
        ):
            self._json(404, {"error": "preview_not_available"})
            return
        try:
            size = target.stat().st_size
            with target.open("rb") as handle:
                content = handle.read(PREVIEW_BYTES + 1)
        except OSError:
            self._json(404, {"error": "preview_not_available"})
            return
        truncated = len(content) > PREVIEW_BYTES
        if truncated:
            content = content[:PREVIEW_BYTES]
        self._json(
            200,
            {
                "path": relative_text,
                "size_bytes": size,
                "truncated": truncated,
                "content": content.decode("utf-8", errors="replace"),
            },
        )

    def _serve_download(self, relative_text: str, *, send_body: bool) -> None:
        target = self._resolve_relative(relative_text)
        if target is None or not target.is_file() or target.is_symlink():
            self._json(404, {"error": "not_found"}, send_body=send_body)
            return
        try:
            stat = target.stat()
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            source = target.open("rb") if send_body else None
        except OSError:
            self._json(404, {"error": "not_found"}, send_body=send_body)
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(stat.st_size))
            self.send_header(
                "Content-Disposition",
                f"attachment; filename=\"{target.name}\"; filename*=UTF-8''{quote(target.name)}",
            )
            self._common_headers()
            self.end_headers()
            if source is not None:
                try:
                    shutil.copyfileobj(source, self.wfile, BUFFER_SIZE)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        finally:
            if source is not None:
                source.close()

    def _handle_login(self) -> None:
        if self.headers.get("Transfer-Encoding"):
            self._serve_login("Invalid login request.", status=400)
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._serve_login("Invalid login request.", status=400)
            return
        if content_length <= 0 or content_length > 4096:
            self._serve_login("Invalid login request.", status=400)
            return
        try:
            raw_body = self.rfile.read(content_length)
            form = parse_qs(raw_body.decode("utf-8"), keep_blank_values=True)
        except (UnicodeDecodeError, ValueError):
            self._serve_login("Invalid login request.", status=400)
            return
        supplied = form.get("token", [""])[0]
        if not hmac.compare_digest(supplied, self.upload_server.token):
            self._serve_login("Token is not valid.", status=401)
            return
        self._redirect(
            "/",
            cookie=(
                f"{SESSION_COOKIE}={self.upload_server.session_value}; "
                "Path=/; HttpOnly; SameSite=Strict; Max-Age=28800"
            ),
        )

    def _serve_login(self, error: str = "", *, status: int = 200) -> None:
        try:
            template = self._read_web_file("login.html").decode("utf-8")
        except (OSError, UnicodeDecodeError):
            self._json(500, {"error": "web_assets_unavailable"})
            return
        error_markup = (
            f'<p class="form-error" role="alert">{html.escape(error)}</p>' if error else ""
        )
        self._html(status, template.replace("{{ERROR}}", error_markup))

    def _serve_web_file(self, name: str) -> None:
        if name not in CONTENT_TYPES:
            self._json(404, {"error": "not_found"})
            return
        try:
            body = self._read_web_file(name)
        except OSError:
            self._json(500, {"error": "web_assets_unavailable"})
            return
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES[name])
        self.send_header("Content-Length", str(len(body)))
        self._common_headers()
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_web_file(self, name: str) -> bytes:
        target = self.upload_server.web_root / name
        target = target.resolve(strict=True)
        target.relative_to(self.upload_server.web_root)
        return target.read_bytes()

    def _receipt(
        self,
        target: Path,
        size: int,
        digest: str,
        status: str,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "service": SERVICE_VERSION,
            "path": str(target.relative_to(self.upload_server.root)),
            "size_bytes": size,
            "sha256": digest,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _write_receipt(target: Path, receipt: dict[str, Any]) -> None:
        receipt_path = target.with_name(f"{target.name}.upload.json")
        temp = receipt_path.with_name(f".{receipt_path.name}.{secrets.token_hex(6)}.tmp")
        temp.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temp, 0o640)
        os.replace(temp, receipt_path)

    def _redirect(self, location: str, *, cookie: str | None = None) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        if cookie is not None:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Length", "0")
        self._common_headers()
        self.end_headers()

    def _html(self, status: int, body_text: str) -> None:
        body = body_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._common_headers()
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _common_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'",
        )

    def _json(
        self,
        status: int,
        payload: dict[str, Any],
        *,
        send_body: bool = True,
    ) -> None:
        body = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        if status >= 400:
            # Early rejections may leave an unread request body.
            self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._common_headers()
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if send_body:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def log_message(self, fmt: str, *args: Any) -> None:
        # Authorization headers are never included by BaseHTTPRequestHandler's format.
        super().log_message(fmt, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--web-root", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=10 * 1024**3)
    parser.add_argument(
        "--namespace",
        default="five_ideas",
        help="top-level project namespace accepted by the exchange",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    web_root = args.web_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not web_root.is_dir():
        raise SystemExit(f"Web root does not exist: {web_root}")
    token = args.token_file.read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise SystemExit("Upload token must contain at least 32 characters")
    if args.max_bytes <= 0:
        raise SystemExit("--max-bytes must be positive")
    if not SAFE_SEGMENT.fullmatch(args.namespace) or args.namespace.startswith("."):
        raise SystemExit("--namespace must be one safe path segment")
    with UploadServer(
        (args.bind, args.port),
        UploadHandler,
        root=root,
        web_root=web_root,
        token=token,
        max_bytes=args.max_bytes,
        namespace=args.namespace,
    ) as server:
        print(
            f"{SERVICE_VERSION} listening on {args.bind}:{args.port}; root={root}",
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
