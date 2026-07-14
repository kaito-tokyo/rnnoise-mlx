"""Download one Mozilla Data Collective dataset with resumable range requests."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


API_BASE = "https://mozilladatacollective.com/api"


def api_json(url: str, api_key: str, *, method: str = "GET") -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method=method,
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "rnnoise-mlx-corpus/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        if error.code == 403 and method == "POST":
            raise PermissionError(
                "MDC refused the download session; accept this dataset's terms in the web interface first"
            ) from error
        raise RuntimeError(f"MDC API {method} failed with HTTP {error.code}: {body}") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stream_download(url: str, partial: Path, expected_size: int, *, max_attempts: int = 5) -> None:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    for attempt in range(1, max_attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            raise ValueError(f"partial file exceeds expected size: {offset} > {expected_size}")
        if offset == expected_size:
            return
        headers = {"User-Agent": "rnnoise-mlx-corpus/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and response.status != 206:
                    raise RuntimeError("storage did not honor the resume Range request")
                with partial.open("ab" if offset else "wb") as target:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
        except OSError:
            if attempt == max_attempts:
                raise
        if partial.stat().st_size == expected_size:
            return
        if attempt < max_attempts:
            time.sleep(min(2 ** (attempt - 1), 8))
    actual_size = partial.stat().st_size
    raise RuntimeError(f"incomplete MDC download after {max_attempts} attempts: {actual_size} != {expected_size}")


def download(dataset_id: str, output_dir: Path, api_key: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    details = api_json(f"{API_BASE}/datasets/{dataset_id}", api_key)
    session = api_json(f"{API_BASE}/datasets/{dataset_id}/download", api_key, method="POST")
    filename = session["filename"]
    expected_size = int(session["sizeBytes"])
    partial = output_dir / f"{filename}.partial"
    final = output_dir / filename
    if final.exists():
        raise FileExistsError(f"final archive already exists: {final}")
    stream_download(session["downloadUrl"], partial, expected_size)
    local_sha256 = sha256(partial)
    reported_checksum = session.get("checksum")
    if reported_checksum and reported_checksum.startswith("sha256:"):
        expected_sha256 = reported_checksum.removeprefix("sha256:")
        if local_sha256 != expected_sha256:
            raise ValueError(f"MDC checksum mismatch: {local_sha256} != {expected_sha256}")
    partial.rename(final)
    manifest = {
        "schema_version": 1,
        "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "dataset_url": f"https://mozilladatacollective.com/datasets/{dataset_id}",
        "name": details.get("name"),
        "locale": details.get("locale"),
        "license": details.get("license"),
        "filename": filename,
        "bytes": final.stat().st_size,
        "api_checksum": reported_checksum,
        "sha256": local_sha256,
        "download_session_expires_at": session.get("expiresAt"),
        "note": "API key and presigned URL are intentionally not recorded.",
    }
    manifest_path = output_dir / f"{filename}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    api_key = os.environ.get("MDC_API_KEY")
    if not api_key:
        parser.error("MDC_API_KEY is not set")
    result = download(args.dataset_id, args.output_dir, api_key)
    print(json.dumps({key: result[key] for key in ("dataset_id", "filename", "bytes", "sha256")}, indent=2))


if __name__ == "__main__":
    main()
