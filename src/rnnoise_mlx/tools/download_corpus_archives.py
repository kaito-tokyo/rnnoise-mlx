"""Download selected planned corpus archives with resumable partial files."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


def selected_archives(plan: dict[str, Any], source_ids: list[str]) -> list[dict[str, str]]:
    complement_sources = plan.get("preferred_complement", {}).get("sources", [])
    sources = {
        source["id"]: source
        for source in [*plan["official_sources"], *plan["regional_english_sources"], *complement_sources]
        if source.get("archives")
    }
    unknown = sorted(set(source_ids) - set(sources))
    if unknown:
        raise ValueError(f"unknown source ids: {', '.join(unknown)}")
    rows: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for source_id in source_ids:
        for url in sources[source_id]["archives"]:
            if url not in seen_urls:
                rows.append({"source_id": source_id, "url": url})
                seen_urls.add(url)
    return rows


def head(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "rnnoise-mlx-corpus-download/1"})
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as error:
        if error.code not in {403, 405}:
            raise
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "rnnoise-mlx-corpus-download/1", "Range": "bytes=0-0"},
        )
        response = urllib.request.urlopen(request, timeout=30)
    with response:
        content_range = response.headers.get("Content-Range")
        if content_range:
            length = content_range.rsplit("/", 1)[-1]
        else:
            length = response.headers.get("Content-Length")
        if length is None or length == "*":
            raise ValueError(f"missing total content length: {url}")
        return {
            "content_length_bytes": int(length),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "final_url": response.geturl(),
            "filename": Path(urlparse(response.geturl()).path).name,
        }


def run_curl(url: str, partial_path: Path) -> None:
    subprocess.run(
        [
            "curl",
            "--silent",
            "--show-error",
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "--output",
            str(partial_path),
            url,
        ],
        check=True,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    plan: dict[str, Any],
    source_ids: list[str],
    output_dir: Path,
    *,
    metadata_fetcher: Callable[[str], dict[str, Any]] = head,
    downloader: Callable[[str, Path], None] = run_curl,
) -> dict[str, Any]:
    rows = selected_archives(plan, source_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "download-manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"download manifest already exists: {manifest_path}")

    metadata = [(row, metadata_fetcher(row["url"])) for row in rows]
    required = 0
    for row, remote in metadata:
        filename = remote.get("filename") or Path(urlparse(row["url"]).path).name
        final_path = output_dir / filename
        partial_path = output_dir / f"{filename}.part"
        expected_bytes = remote["content_length_bytes"]
        if final_path.exists():
            actual_bytes = final_path.stat().st_size
            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"size mismatch for existing {filename}: expected {expected_bytes}, got {actual_bytes}"
                )
            if partial_path.exists():
                raise FileExistsError(f"both complete and partial archives exist: {filename}")
        elif partial_path.exists():
            partial_bytes = partial_path.stat().st_size
            if partial_bytes > expected_bytes:
                raise ValueError(
                    f"oversized partial for {filename}: expected at most {expected_bytes}, got {partial_bytes}"
                )
            required += expected_bytes - partial_bytes
        else:
            required += expected_bytes
    free = shutil.disk_usage(output_dir).free
    if free < required + 1024**3:
        raise OSError(f"insufficient free space: need archive bytes plus 1 GiB ({required})")

    records: list[dict[str, Any]] = []
    for row, remote in metadata:
        filename = remote.get("filename") or Path(urlparse(row["url"]).path).name
        final_path = output_dir / filename
        partial_path = output_dir / f"{filename}.part"
        if final_path.exists():
            actual_bytes = final_path.stat().st_size
        else:
            downloader(row["url"], partial_path)
            actual_bytes = partial_path.stat().st_size
            if actual_bytes != remote["content_length_bytes"]:
                raise ValueError(
                    f"size mismatch for {filename}: expected {remote['content_length_bytes']}, got {actual_bytes}"
                )
            partial_path.rename(final_path)
        records.append(
            {
                **row,
                **remote,
                "path": filename,
                "bytes": actual_bytes,
                "sha256": sha256(final_path),
            }
        )

    manifest = {
        "schema_version": 1,
        "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_ids": source_ids,
        "archives": records,
        "total_bytes": sum(record["bytes"] for record in records),
    }
    temporary = output_dir / "download-manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.rename(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source", action="append", required=True, dest="source_ids")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    manifest = download(plan, args.source_ids, args.output_dir)
    print(json.dumps({"archive_count": len(manifest["archives"]), "total_bytes": manifest["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
