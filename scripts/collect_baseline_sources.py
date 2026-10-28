#!/usr/bin/env python3
"""Bounded public primary-source collection; bodies remain in an ignored cache."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

REPOSITORIES = {
    "lra": "google-research/long-range-arena",
    "mega": "facebookresearch/mega",
    "s4": "state-spaces/s4",
    "s5": "lindermanlab/S5",
    "lru": "NicolasZucchet/minimal-LRU",
    "mamba": "state-spaces/mamba",
}
PAGES = {
    "imdb-use": "https://ai.stanford.edu/~amaas/data/sentiment/",
    "cifar-use": "https://www.cs.toronto.edu/~kriz/cifar.html",
    "acl-use": "https://aclanthology.org/info/copyright/",
    "lra-paper": "https://arxiv.org/abs/2011.04006",
    "s5-paper": "https://arxiv.org/abs/2208.04933",
    "lru-paper": "https://arxiv.org/abs/2303.06349",
    "mega-paper": "https://arxiv.org/abs/2209.10655",
    "mamba-paper": "https://arxiv.org/abs/2312.00752",
}


class Collector:
    def __init__(self, cache: Path):
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.last = {}
        self.blocked = set()
        self.records = []

    def get(self, name, url):
        origin = urllib.parse.urlsplit(url).netloc
        path = self.cache / (hashlib.sha256(url.encode()).hexdigest() + ".body")
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        record = {"id": name, "method": "GET", "url": url,
                  "retrieved_at": timestamp, "access_mode": "public_http",
                  "browser_escalations": 0}
        if path.exists():
            body = path.read_bytes()
            record.update(status="cache_hit", cache_state="existing_unvalidated",
                          observed_at=dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat())
        elif origin in self.blocked:
            record.update(status="blocked", failure_reason="origin_access_signal")
            self.records.append(record)
            return None
        else:
            time.sleep(max(0.0, 1.5 - (time.monotonic() - self.last.get(origin, 0))))
            request = urllib.request.Request(url, headers={"User-Agent": "QKAN-Mamba-research-source-audit/1.0"})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    body = response.read(4 * 1024 * 1024 + 1)
                    if len(body) > 4 * 1024 * 1024:
                        raise ValueError("source exceeds the 4 MiB bound")
                    record.update(status="fresh", http_status=response.status,
                                  final_url=response.url, cache_state="miss")
                path.write_bytes(body)
            except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as error:
                code = getattr(error, "code", None)
                record.update(status="blocked" if code in (401, 403, 429) else "failed",
                              http_status=code, failure_reason=str(error))
                if code in (401, 403, 429):
                    self.blocked.add(origin)
                self.records.append(record)
                return None
            finally:
                self.last[origin] = time.monotonic()
        record.update(sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body),
                      response_classification="public_primary_source")
        self.records.append(record)
        return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supplement", action="store_true", help="Bounded recent-paper search and missing dataset-source recovery")
    parser.add_argument("--recent-details", action="store_true", help="Follow six task-relevant primary papers and the next search page")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse to overwrite a source receipt")
    collector = Collector(args.cache)
    pins = {}
    repositories = {} if (args.supplement or args.recent_details) else REPOSITORIES
    pages = {
        "lra-recent-api": "https://export.arxiv.org/api/query?search_query=all%3A%22long%20range%20arena%22&start=0&max_results=30&sortBy=submittedDate&sortOrder=descending",
        "lra-recent-search": "https://arxiv.org/search/?query=%22long+range+arena%22&searchtype=all&abstracts=show&order=-announced_date_first&size=50",
        "acl-copyright-recovery": "https://aclanthology.org/faq/#what-is-the-copyright-for-materials-in-the-acl-anthology",
        "lra-listops-source": "https://raw.githubusercontent.com/google-research/long-range-arena/cd31e5c6b8e5bceabd28de2d2afb23f7ae5d36d8/lra_benchmarks/listops/data_processing.py",
        "lra-pathfinder-source": "https://raw.githubusercontent.com/google-research/long-range-arena/cd31e5c6b8e5bceabd28de2d2afb23f7ae5d36d8/lra_benchmarks/image/input_pipeline.py",
        "aan-author-source": "https://clair.eecs.umich.edu/aan/index.php",
    } if args.supplement else PAGES
    if args.recent_details:
        pages = {
            "lra-search-page2": "https://arxiv.org/search/?query=%22long+range+arena%22&searchtype=all&abstracts=show&order=-announced_date_first&size=50&start=50",
            "block-biased-mamba": "https://arxiv.org/html/2505.09022v1",
            "serpent": "https://arxiv.org/html/2501.11729v1",
            "s7": "https://arxiv.org/html/2410.03464v1",
            "locality-bias": "https://arxiv.org/html/2501.14850v1",
            "flare-plus-plus": "https://arxiv.org/html/2608.11519v1",
            "naju": "https://arxiv.org/html/2607.21000v1",
        }
    for name, repo in repositories.items():
        body = collector.get(name + "-commit", f"https://api.github.com/repos/{repo}/commits?per_page=1")
        if body is None:
            continue
        commit = json.loads(body)[0]["sha"]
        pins[name] = {"repository": f"https://github.com/{repo}", "commit": commit}
        collector.get(name + "-readme", f"https://raw.githubusercontent.com/{repo}/{commit}/README.md")
        if name == "mega":
            collector.get("mega-lra-protocol", f"https://raw.githubusercontent.com/{repo}/{commit}/examples/mega/README.lra.md")
        if name == "lra":
            collector.get("lra-license", f"https://raw.githubusercontent.com/{repo}/{commit}/LICENSE")
    for name, url in pages.items():
        collector.get(name, url)
    result = {"schema": "qkan-primary-source-receipt.v1", "pins": pins,
              "policy": {"requests_max": 7 if args.recent_details else (6 if args.supplement else 22), "retries": 0, "per_origin_concurrency": 1,
                         "minimum_delay_seconds": 1.5, "freshness": "retrieval time or explicitly unvalidated cache age",
                         "purpose": "baseline candidates, upstream protocol, dataset use terms",
                         "allowed_fields": ["public paper metadata", "README", "commit", "public use and license terms"]},
              "sources": collector.records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pins": pins, "sources": [{"id": r["id"], "status": r["status"]} for r in collector.records]}))


if __name__ == "__main__":
    main()
