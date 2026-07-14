"""Build deterministic priority and reference listening packs from an audio audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import tarfile
import zipfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


CRITICAL_FLAGS = {
    "decode_error",
    "hard_clipping",
    "large_dc_offset",
    "very_high_rms",
    "very_low_peak",
    "very_low_rms",
}


def stable_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{row['source_id']}:{row['archive']}:{row['path']}".encode()).hexdigest()


def round_robin(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    groups: dict[str, deque[dict[str, Any]]] = {}
    for speaker in sorted({str(row["speaker_id"]) for row in rows}):
        groups[speaker] = deque(sorted((row for row in rows if str(row["speaker_id"]) == speaker), key=stable_key))
    selected: list[dict[str, Any]] = []
    while len(selected) < count and groups:
        for speaker in list(groups):
            queue = groups[speaker]
            if queue:
                selected.append(queue.popleft())
                if len(selected) == count:
                    break
            if not queue:
                del groups[speaker]
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} rows available for requested count {count}")
    return selected


def select_source(rows: list[dict[str, Any]], count: int) -> dict[str, list[dict[str, Any]]]:
    valid = [row for row in rows if "error" not in row]
    priority_candidates = [row for row in valid if row["diagnostic_flags"]]
    reference_candidates = [row for row in valid if not row["diagnostic_flags"]]

    critical = [row for row in priority_candidates if CRITICAL_FLAGS & set(row["diagnostic_flags"])]
    critical.sort(key=stable_key)
    if len(critical) > count:
        critical = round_robin(critical, count)
    selected_ids = {(row["archive"], row["path"]) for row in critical}
    remaining = [row for row in priority_candidates if (row["archive"], row["path"]) not in selected_ids]
    remaining.sort(
        key=lambda row: (
            -(row["leading_low_rms_seconds"] + row["trailing_low_rms_seconds"]),
            stable_key(row),
        )
    )
    priority = [*critical, *remaining[: count - len(critical)]]
    if len(priority) < count:
        selected_ids = {(row["archive"], row["path"]) for row in priority}
        supplements = sorted(
            (row for row in valid if (row["archive"], row["path"]) not in selected_ids),
            key=stable_key,
        )
        priority.extend(supplements[: count - len(priority)])
    if len(priority) != count:
        raise ValueError(f"only {len(priority)} priority rows available for requested count {count}")
    priority_ids = {(row["archive"], row["path"]) for row in priority}
    reference = round_robin(
        [row for row in reference_candidates if (row["archive"], row["path"]) not in priority_ids], count
    )
    return {"priority": priority, "reference": reference}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_member(archive_path: Path, member_name: str, destination: Path) -> None:
    """Extract one exact member from a ZIP or tar archive."""
    lower_name = archive_path.name.lower()
    if lower_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open(member_name) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
        return
    if lower_name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as archive:
            member = archive.getmember(member_name)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"archive member is not a regular file: {member_name}")
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
        return
    raise ValueError(f"unsupported listening-pack archive: {archive_path.name}")


def extract_selected_members(
    archive_path: Path, items: list[tuple[str, str, dict[str, Any]]], partial: Path
) -> list[dict[str, Any]]:
    """Extract selected members while opening a potentially large archive once."""
    records: list[dict[str, Any]] = []
    lower_name = archive_path.name.lower()
    def copy_item(source: Any, item: tuple[str, str, dict[str, Any]]) -> None:
        source_id, group, row = item
        destination = partial / source_id / group / Path(row["path"]).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)
        records.append(
            {
                "source_id": source_id,
                "group": group,
                "archive": archive_path.name,
                "source_path": row["path"],
                "path": str(destination.relative_to(partial)),
                "speaker_id": row["speaker_id"],
                "duration_seconds": row["duration_seconds"],
                "diagnostic_flags": row["diagnostic_flags"],
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )

    if lower_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            for item in items:
                copy_item(archive.open(item[2]["path"]), item)
    elif lower_name.endswith((".tar.gz", ".tgz", ".tar")):
        wanted = {item[2]["path"]: item for item in items}
        with tarfile.open(archive_path, "r|*") as archive:
            for member in archive:
                item = wanted.pop(member.name, None)
                if item is None:
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"archive member is not a regular file: {member.name}")
                copy_item(source, item)
        if wanted:
            raise KeyError(f"archive members not found: {sorted(wanted)}")
    else:
        raise ValueError(f"unsupported listening-pack archive: {archive_path.name}")
    return records


def review_template(manifest: dict[str, Any]) -> str:
    header = "source_id\tgroup\tpath\tspeaker_id\tduration_seconds\tdiagnostic_flags\tdecision\treason\tnotes\n"
    lines = [header]
    for record in manifest["records"]:
        fields = (
            record["source_id"],
            record["group"],
            record["path"],
            str(record["speaker_id"]),
            f"{record['duration_seconds']:.6f}",
            ",".join(record["diagnostic_flags"]),
            "",
            "",
            "",
        )
        lines.append("\t".join(fields) + "\n")
    return "".join(lines)


def build(download_dir: Path, audit_path: Path, output_dir: Path, *, count: int = 100) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    partial = output_dir.with_name(output_dir.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"partial directory already exists: {partial}")
    partial.mkdir(parents=True)
    audit = json.loads(audit_path.read_text())
    selections: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for source_id in sorted(audit["source_summaries"]):
        selections[source_id] = select_source(
            [row for row in audit["clips"] if row["source_id"] == source_id], count
        )

    wanted: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    for source_id, groups in selections.items():
        for group, rows in groups.items():
            for row in rows:
                wanted[row["archive"]].append((source_id, group, row))

    records: list[dict[str, Any]] = []
    for archive_name, items in sorted(wanted.items()):
        archive_path = download_dir / archive_name
        records.extend(extract_selected_members(archive_path, items, partial))
    records.sort(key=lambda row: (row["source_id"], row["group"], row["path"]))
    manifest = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "samples_per_group_per_source": count,
        "selection_policy": {
            "priority": "include critical waveform flags, then largest edge-padding diagnostics; use stable-hash supplements only when fewer than the requested count are flagged",
            "reference": "round-robin speakers, stable SHA-256 order, no diagnostic flags",
            "note": "Priority is not synonymous with rejected; listening assigns the decision.",
        },
        "records": records,
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (partial / "review.tsv").write_text(review_template(manifest))
    partial.rename(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("download_dir", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")
    result = build(args.download_dir, args.audit, args.output_dir, count=args.count)
    counts: dict[str, int] = defaultdict(int)
    for record in result["records"]:
        counts[f"{record['source_id']}/{record['group']}"] += 1
    print(json.dumps(dict(sorted(counts.items())), indent=2))


if __name__ == "__main__":
    main()
