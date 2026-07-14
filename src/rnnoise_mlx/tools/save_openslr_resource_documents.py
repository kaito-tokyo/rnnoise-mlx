"""Save source pages and small documents for explicit OpenSLR resource IDs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from rnnoise_mlx.tools.save_corpus_source_documents import document_links, fetch


def write_document(
    root: Path,
    relative_path: Path,
    url: str,
    fetcher: Callable[..., tuple[bytes, str, str | None]],
) -> dict[str, Any]:
    body, final_url, content_type = fetcher(url)
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {
        "url": url,
        "final_url": final_url,
        "path": str(relative_path),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "content_type": content_type,
    }


def save_resources(
    resource_ids: list[str],
    output_dir: Path,
    *,
    fetcher: Callable[..., tuple[bytes, str, str | None]] = fetch,
) -> dict[str, Any]:
    if not resource_ids or any(not value.isdigit() for value in resource_ids):
        raise ValueError("one or more numeric OpenSLR resource IDs are required")
    if len(resource_ids) != len(set(resource_ids)):
        raise ValueError("duplicate OpenSLR resource ID")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    partial = output_dir.with_name(output_dir.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"partial directory already exists: {partial}")
    partial.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for resource_id in sorted(resource_ids, key=int):
        page_url = f"https://www.openslr.org/{resource_id}/"
        page_body, final_url, content_type = fetcher(page_url)
        relative = Path(f"slr{resource_id}/dataset-page.html")
        path = partial / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(page_body)
        records.append(
            {
                "url": page_url,
                "final_url": final_url,
                "path": str(relative),
                "bytes": len(page_body),
                "sha256": hashlib.sha256(page_body).hexdigest(),
                "content_type": content_type,
            }
        )
        for url in document_links(page_url, page_body):
            name = Path(urlparse(url).path).name
            records.append(write_document(partial, Path(f"slr{resource_id}/{name}"), url, fetcher))
    manifest = {
        "schema_version": 1,
        "resource_ids": sorted(resource_ids, key=int),
        "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "documents": records,
        "note": "Archive-embedded LICENSE, NOTICE, and README files must be added after archive retrieval.",
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    partial.rename(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--resource", action="append", required=True, dest="resource_ids")
    args = parser.parse_args()
    manifest = save_resources(args.resource_ids, args.output_dir)
    print(json.dumps({"resource_ids": manifest["resource_ids"], "document_count": len(manifest["documents"])}, indent=2))


if __name__ == "__main__":
    main()
