"""Save small source and license documents for an acquisition stage."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse


MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
DOCUMENT_NAME_RE = re.compile(r"(?:license|readme|about|citation|metadata)", re.IGNORECASE)
RESOURCE_RE = re.compile(r"/resources/(\d+)/")


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self.texts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.hrefs.append(value)

    def handle_data(self, data: str) -> None:
        self.texts.append(data)


def stage_resource_ids(plan: dict[str, Any], stage_id: str) -> list[str]:
    stages = {stage["id"]: stage for stage in plan["acquisition_stages"]}
    if stage_id not in stages:
        raise ValueError(f"unknown stage: {stage_id}")
    source_by_id = {
        source["id"]: source
        for source in [*plan["official_sources"], *plan["regional_english_sources"]]
    }
    resource_ids: set[str] = set()
    for source_id in stages[stage_id]["source_ids"]:
        for url in source_by_id[source_id]["archives"]:
            match = RESOURCE_RE.search(urlparse(url).path)
            if match is None:
                raise ValueError(f"not an OpenSLR resource URL: {url}")
            resource_ids.add(match.group(1))
    return sorted(resource_ids, key=int)


def fetch(url: str, max_bytes: int = MAX_DOCUMENT_BYTES) -> tuple[bytes, str, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": "rnnoise-mlx-corpus-documents/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > max_bytes:
            raise ValueError(f"document is larger than {max_bytes} bytes: {url}")
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError(f"document exceeded {max_bytes} bytes: {url}")
        return body, response.geturl(), response.headers.get("Content-Type")


def document_links(page_url: str, body: bytes) -> list[str]:
    parser = Links()
    text = body.decode("utf-8", errors="replace")
    parser.feed(text)
    resource_id = page_url.rstrip("/").rsplit("/", 1)[-1]
    result: set[str] = set()
    for href in parser.hrefs:
        url = urljoin(page_url, href)
        parsed = urlparse(url)
        name = Path(parsed.path).name
        if f"/resources/{resource_id}/" in parsed.path and DOCUMENT_NAME_RE.search(name):
            result.add(f"https://www.openslr.org{parsed.path}")
    visible_text = " ".join(parser.texts)
    linked_names = {Path(urlparse(url).path).name.lower() for url in result}
    for family, pattern in (
        ("license", r"\bLICENSE(?:\.txt)?\b"),
        ("readme", r"\bREADME(?:\.txt)?\b"),
        ("about", r"\babout\.html\b"),
    ):
        if not any(name.startswith(family) for name in linked_names):
            match = re.search(pattern, visible_text)
            if match:
                result.add(f"https://www.openslr.org/resources/{resource_id}/{match.group(0)}")
    return sorted(result)


def _write_document(root: Path, relative_path: Path, url: str, fetcher: Callable[..., tuple[bytes, str, str | None]]) -> dict[str, Any]:
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


def save_documents(
    plan: dict[str, Any],
    stage_id: str,
    output_dir: Path,
    *,
    fetcher: Callable[..., tuple[bytes, str, str | None]] = fetch,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if partial_dir.exists():
        raise FileExistsError(f"partial output directory already exists: {partial_dir}")
    partial_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    upstream_url = plan["upstream"]["url"]
    records.append(_write_document(partial_dir, Path("rnnoise-datasets.txt"), upstream_url, fetcher))
    for resource_id in stage_resource_ids(plan, stage_id):
        page_url = f"https://www.openslr.org/{resource_id}/"
        page_body, final_url, content_type = fetcher(page_url)
        page_relative = Path(f"slr{resource_id}/dataset-page.html")
        page_path = partial_dir / page_relative
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_bytes(page_body)
        records.append(
            {
                "url": page_url,
                "final_url": final_url,
                "path": str(page_relative),
                "bytes": len(page_body),
                "sha256": hashlib.sha256(page_body).hexdigest(),
                "content_type": content_type,
            }
        )
        for url in document_links(page_url, page_body):
            name = Path(urlparse(url).path).name
            records.append(_write_document(partial_dir, Path(f"slr{resource_id}/{name}"), url, fetcher))
    manifest = {
        "schema_version": 1,
        "stage_id": stage_id,
        "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "documents": records,
    }
    (partial_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    partial_dir.rename(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("stage_id")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    manifest = save_documents(plan, args.stage_id, args.output_dir)
    print(json.dumps({"stage_id": manifest["stage_id"], "document_count": len(manifest["documents"])}, indent=2))


if __name__ == "__main__":
    main()
