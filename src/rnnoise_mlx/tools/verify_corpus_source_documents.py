"""Verify saved corpus source documents and their license review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, *, plan: dict[str, Any] | None = None, review: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    failures: list[str] = []
    for record in manifest["documents"]:
        path = root / record["path"]
        if not path.is_file():
            failures.append(f"missing: {record['path']}")
            continue
        if path.stat().st_size != record["bytes"]:
            failures.append(f"size: {record['path']}")
        if sha256(path) != record["sha256"]:
            failures.append(f"sha256: {record['path']}")

    if plan is not None:
        upstream_path = root / "rnnoise-datasets.txt"
        if sha256(upstream_path) != plan["upstream"]["sha256"]:
            failures.append("upstream plan SHA-256")

    if review is not None:
        if sha256(manifest_path) != review["evidence_manifest_sha256"]:
            failures.append("review manifest SHA-256")
        for resource in review["resources"]:
            path = root / resource["license_path"]
            if not path.is_file():
                failures.append(f"missing review license: {resource['license_path']}")
            elif sha256(path) != resource["license_sha256"]:
                failures.append(f"review license SHA-256: {resource['license_path']}")

    if failures:
        raise ValueError("; ".join(failures))
    return {
        "stage_id": manifest["stage_id"],
        "document_count": len(manifest["documents"]),
        "reviewed_resource_count": len(review["resources"]) if review is not None else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text()) if args.plan else None
    review = json.loads(args.review.read_text()) if args.review else None
    print(json.dumps(verify(args.root, plan=plan, review=review), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
