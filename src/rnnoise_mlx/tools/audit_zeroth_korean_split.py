"""Audit Zeroth Korean train/test split and AUDIO_INFO speaker metadata."""

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

from rnnoise_mlx.tools.inspect_corpus_archives import flac_info


def audit(archive_path: Path, output: Path) -> dict[str, Any]:
    speakers: dict[str, dict[str, str]] = {}
    split_seconds: collections.Counter[str] = collections.Counter()
    split_clips: collections.Counter[str] = collections.Counter()
    speaker_seconds: collections.defaultdict[str, float] = collections.defaultdict(float)
    speaker_clips: collections.Counter[str] = collections.Counter()
    failures: list[dict[str, str]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        info = archive.extractfile("AUDIO_INFO")
        if info is None:
            raise ValueError("AUDIO_INFO is missing")
        for row in csv.DictReader(io.TextIOWrapper(info, encoding="utf-8", newline=""), delimiter="|"):
            speakers[row["SPEAKERID"]] = {
                "name": row["NAME"],
                "sex": row["SEX"],
                "script_id": row["SCRIPTID"],
                "dataset": row["DATASET"],
            }
        for member in archive:
            if not member.isfile() or not member.name.endswith(".flac"):
                continue
            parts = Path(member.name).parts
            if len(parts) < 3:
                failures.append({"path": member.name, "error": "unexpected path"})
                continue
            dataset = parts[0]
            speaker_id = parts[-2]
            stream = archive.extractfile(member)
            if stream is None:
                failures.append({"path": member.name, "error": "cannot extract"})
                continue
            try:
                rate, channels, bits, samples = flac_info(stream)
                duration = samples / rate
            except ValueError as error:
                failures.append({"path": member.name, "error": str(error)})
                continue
            split_seconds[dataset] += duration
            split_clips[dataset] += 1
            speaker_seconds[speaker_id] += duration
            speaker_clips[speaker_id] += 1
    unknown_speakers = sorted(set(speaker_seconds) - set(speakers))
    split_speaker_ids = {
        dataset: sorted(speaker_id for speaker_id, row in speakers.items() if row["dataset"] == dataset)
        for dataset in sorted(set(row["dataset"] for row in speakers.values()))
    }
    result = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "archive": archive_path.name,
        "split_summaries": {
            dataset: {
                "clip_count": split_clips[dataset],
                "duration_seconds": split_seconds[dataset],
                "speaker_count": len(split_speaker_ids.get(dataset, [])),
                "speaker_ids": split_speaker_ids.get(dataset, []),
            }
            for dataset in sorted(split_seconds)
        },
        "speaker_metadata_count": len(speakers),
        "speaker_duration_seconds": dict(sorted(speaker_seconds.items())),
        "speaker_clip_counts": dict(sorted(speaker_clips.items())),
        "unknown_speaker_count": len(unknown_speakers),
        "unknown_speakers": unknown_speakers,
        "train_test_speaker_disjoint": not bool(
            set(split_speaker_ids.get("train_data_01", [])) & set(split_speaker_ids.get("test_data_01", []))
        ),
        "header_failure_count": len(failures),
        "header_failures": failures,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.archive, args.output)
    print(json.dumps(result["split_summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
