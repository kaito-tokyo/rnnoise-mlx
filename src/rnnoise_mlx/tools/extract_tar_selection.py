"""Extract records from a JSON selection manifest into train/eval directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def extract(archive_path: Path, selection_path: Path, output: Path) -> dict[str, object]:
    selection = json.loads(selection_path.read_text())
    extracted = []
    selected = {record["path"]: record for record in selection["records"]}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            record = selected.get(member.name)
            if record is None:
                continue
            member_name = member.name
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot extract {member_name}")
            body = stream.read()
            destination = output / record["split"] / Path(member_name).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(body)
            extracted.append(
                {
                    "source_member": member_name,
                    "path": destination.relative_to(output).as_posix(),
                    "split": record["split"],
                    "speaker_id": record.get("speaker_id"),
                    "duration_seconds": record.get("duration_seconds"),
                    "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            )
    if len(extracted) != len(selected):
        found = {record["source_member"] for record in extracted}
        missing = sorted(set(selected) - found)
        raise ValueError(f"selected members missing from archive: {missing[:5]}")
    manifest = {
        "format_version": 1,
        "archive": str(archive_path.resolve()),
        "selection": str(selection_path.resolve()),
        "clip_count": len(extracted),
        "records": extracted,
    }
    (output / "extraction-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = extract(args.archive, args.selection, args.output)
    print(json.dumps({"clip_count": result["clip_count"]}, indent=2))


if __name__ == "__main__":
    main()
