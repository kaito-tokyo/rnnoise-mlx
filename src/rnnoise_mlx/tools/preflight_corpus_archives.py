"""Collect HTTP metadata for planned corpus archives without downloading them."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


USER_AGENT = "rnnoise-mlx-corpus-preflight/1"


def archive_urls(plan: dict[str, Any]) -> list[str]:
    sources = [*plan["official_sources"], *plan["regional_english_sources"]]
    urls = [url for source in sources for url in source["archives"]]
    return sorted(set(urls))


def inspect_url(
    url: str,
    *,
    timeout: float = 30,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with opener(request, timeout=timeout) as response:
            headers = response.headers
            content_length = headers.get("Content-Length")
            return {
                "url": url,
                "status": response.status,
                "final_url": response.geturl(),
                "content_length_bytes": int(content_length) if content_length is not None else None,
                "etag": headers.get("ETag"),
                "last_modified": headers.get("Last-Modified"),
                "content_type": headers.get("Content-Type"),
                "error": None,
            }
    except urllib.error.HTTPError as error:
        return {
            "url": url,
            "status": error.code,
            "final_url": error.geturl(),
            "content_length_bytes": None,
            "etag": None,
            "last_modified": None,
            "content_type": None,
            "error": str(error),
        }
    except (OSError, urllib.error.URLError) as error:
        return {
            "url": url,
            "status": None,
            "final_url": None,
            "content_length_bytes": None,
            "etag": None,
            "last_modified": None,
            "content_type": None,
            "error": str(error),
        }


def preflight(
    plan: dict[str, Any],
    *,
    workers: int = 8,
    timeout: float = 30,
    inspector: Callable[..., dict[str, Any]] = inspect_url,
) -> dict[str, Any]:
    urls = archive_urls(plan)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(inspector, url, timeout=timeout): url for url in urls}
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    results.sort(key=lambda item: item["url"])
    successful = [item for item in results if item["status"] is not None and 200 <= item["status"] < 400]
    sized = [item for item in successful if item["content_length_bytes"] is not None]
    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "request_method": "HEAD",
        "archive_count": len(results),
        "successful_count": len(successful),
        "sized_count": len(sized),
        "known_total_bytes": sum(item["content_length_bytes"] for item in sized),
        "archives": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    plan = json.loads(args.plan.read_text())
    result = preflight(plan, workers=args.workers, timeout=args.timeout)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("archive_count", "successful_count", "sized_count", "known_total_bytes")},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
