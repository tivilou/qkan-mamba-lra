#!/usr/bin/env python3
"""Fetch the author-linked MEGA archive with bounded size and resumable byte ranges."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

URL = "https://dl.fbaipublicfiles.com/mega/data/lra.zip"
EXPECTED_BYTES = 1447223076
ETAG = '"060665a1a941c74cfaae5254c65eab20-173"'


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    target, receipt = args.output.resolve(), args.receipt.resolve()
    if target == receipt or target.exists() or receipt.exists():
        parser.error("Use new distinct output and receipt paths; existing files are never overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < 10 * 1024**3:
        parser.error("Require 10 GiB free before staging this archive")
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > EXPECTED_BYTES or partial.is_symlink():
        parser.error("Invalid partial file")
    headers = {"User-Agent": "QKAN-Mamba-data-recovery/1.0", "Accept-Encoding": "identity"}
    if offset:
        headers.update({"Range": f"bytes={offset}-", "If-Range": ETAG})
    started = datetime.now(timezone.utc).isoformat()
    if offset < EXPECTED_BYTES:
        with urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=45) as response:
            if response.headers.get("ETag") != ETAG:
                raise ValueError("Source ETag changed; audit the new version before downloading")
            if (response.status != (206 if offset else 200) or
                    int(response.headers.get("Content-Length", -1)) != EXPECTED_BYTES - offset):
                raise ValueError("Server did not honor the bounded download/resume contract")
            if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Unexpected resumed byte range")
            with partial.open("ab" if offset else "xb") as handle:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    offset += len(block)
                    if offset > EXPECTED_BYTES:
                        raise ValueError("Download exceeds the frozen archive size")
                    handle.write(block)
    if partial.stat().st_size != EXPECTED_BYTES:
        raise ValueError("Incomplete archive; retry with the same paths")
    sha = digest(partial)
    partial.rename(target)
    record = {"schema_version": "qkan-source-download.v1", "url": URL,
              "version_identity": {"etag": ETAG, "size_bytes": EXPECTED_BYTES},
              "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
              "sha256": sha, "status": "downloaded_not_qualified",
              "warning": "Observed SHA-256 pins these bytes; no publisher SHA-256 was provided."}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
