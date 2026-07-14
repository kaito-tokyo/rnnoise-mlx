"""Summarize validated Common Voice metadata without extracting the archive."""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import io
import json
import tarfile
from pathlib import Path
from typing import Any


csv.field_size_limit(10 * 1024 * 1024)


def unique_member(archive: tarfile.TarFile, suffix: str) -> tarfile.TarInfo:
    matches = [member for member in archive.getmembers() if member.isfile() and member.name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one {suffix} member, found {len(matches)}")
    return matches[0]


def rows(archive: tarfile.TarFile, member: tarfile.TarInfo) -> csv.DictReader:
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"cannot read archive member: {member.name}")
    return csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8", newline=""), delimiter="\t")


def value_summary(records: list[dict[str, str]], field: str, durations: dict[str, int]) -> list[dict[str, Any]]:
    counts: collections.Counter[str] = collections.Counter()
    seconds: collections.Counter[str] = collections.Counter()
    speakers: dict[str, set[str]] = collections.defaultdict(set)
    for record in records:
        value = record.get(field, "").strip()
        if not value:
            continue
        counts[value] += 1
        seconds[value] += durations.get(record["path"], 0) / 1000
        if record.get("client_id"):
            speakers[value].add(record["client_id"])
    return [
        {
            "value": value,
            "clip_count": counts[value],
            "duration_seconds": seconds[value],
            "speaker_count": len(speakers[value]),
        }
        for value in sorted(counts)
    ]


def inspect(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        duration_rows = list(rows(archive, unique_member(archive, "/clip_durations.tsv")))
        durations = {record["clip"]: int(record["duration[ms]"]) for record in duration_rows}
        validated = list(rows(archive, unique_member(archive, "/validated.tsv")))

    missing = sorted(record["path"] for record in validated if record["path"] not in durations)
    speaker_seconds: collections.Counter[str] = collections.Counter()
    for record in validated:
        if record.get("client_id"):
            speaker_seconds[record["client_id"]] += durations.get(record["path"], 0) / 1000
    locales = sorted({record.get("locale", "") for record in validated if record.get("locale")})
    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "archive": path.name,
        "locales": locales,
        "validated_clip_count": len(validated),
        "validated_duration_seconds": sum(durations.get(record["path"], 0) for record in validated) / 1000,
        "validated_speaker_count": len(speaker_seconds),
        "missing_duration_count": len(missing),
        "missing_duration_paths": missing,
        "speaker_duration_seconds": dict(sorted(speaker_seconds.items())),
        "metadata_summaries": {
            field: value_summary(validated, field, durations)
            for field in ("accents", "variant", "age", "gender")
        },
        "note": "Only validated metadata aggregates are recorded; sentence text is intentionally omitted.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = inspect(args.archive)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("locales", "validated_clip_count", "validated_duration_seconds", "validated_speaker_count", "missing_duration_count")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
