"""Extract only selected Common Voice MP3 clips and record their digests."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any


def extract(archive_path: Path, selection_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    partial = output_dir.with_name(output_dir.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"partial directory already exists: {partial}")
    selection = json.loads(selection_path.read_text())
    wanted = {record["path"]: record for record in selection["records"]}
    if len(wanted) != len(selection["records"]):
        raise ValueError("selection contains duplicate clip paths")
    partial.mkdir(parents=True)
    extracted: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            clip_name = Path(member.name).name
            record = wanted.get(member.name) or wanted.get(clip_name)
            if record is None:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read selected member: {member.name}")
            relative = Path(record["split"]) / clip_name
            destination = partial / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with stream, destination.open("wb") as target:
                while chunk := stream.read(1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            extracted.append(
                {
                    "path": str(relative),
                    "source_member": member.name,
                    "speaker_id": record["speaker_id"],
                    "split": record["split"],
                    "duration_seconds": record["duration_seconds"],
                    "bytes": size,
                    "sha256": digest.hexdigest(),
                }
            )
    extracted_sources = {record["source_member"] for record in extracted}
    extracted_basenames = {Path(source).name for source in extracted_sources}
    missing = sorted(
        path for path in wanted
        if path not in extracted_sources and Path(path).name not in extracted_basenames
    )
    if missing:
        shutil.rmtree(partial)
        raise ValueError(f"selected clips missing from archive: {len(missing)}")
    extracted.sort(key=lambda record: record["path"])
    manifest = {
        "schema_version": 1,
        "archive": archive_path.name,
        "selection_manifest": selection_path.name,
        "selection_manifest_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "clip_count": len(extracted),
        "total_bytes": sum(record["bytes"] for record in extracted),
        "records": extracted,
    }
    (partial / "extraction-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    partial.rename(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    manifest = extract(args.archive, args.selection, args.output_dir)
    print(json.dumps({"clip_count": manifest["clip_count"], "total_bytes": manifest["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
