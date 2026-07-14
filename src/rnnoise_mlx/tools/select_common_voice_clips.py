"""Select speaker-disjoint Common Voice train/eval clips deterministically."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from rnnoise_mlx.tools.inspect_common_voice_archive import rows, unique_member


def stable_key(seed: int, label: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{label}:{value}".encode()).digest()


def load_validated(path: Path) -> tuple[list[dict[str, Any]], str]:
    with tarfile.open(path, "r:gz") as archive:
        durations = {
            record["clip"]: int(record["duration[ms]"]) / 1000
            for record in rows(archive, unique_member(archive, "/clip_durations.tsv"))
        }
        validated = list(rows(archive, unique_member(archive, "/validated.tsv")))
    records = []
    locales = set()
    for record in validated:
        client_id = record.get("client_id", "").strip()
        if not client_id:
            continue
        clip_path = record["path"]
        if clip_path not in durations:
            raise ValueError(f"validated clip is missing duration: {clip_path}")
        locales.add(record.get("locale", ""))
        records.append({"path": clip_path, "speaker_id": client_id, "duration_seconds": durations[clip_path]})
    if len(locales) != 1:
        raise ValueError(f"expected one locale, found {sorted(locales)}")
    return records, locales.pop()


def select_split(
    by_speaker: dict[str, list[dict[str, Any]]],
    speakers: list[str],
    target_seconds: float,
    speaker_cap_seconds: float,
    seed: int,
    label: str,
    *,
    round_robin: bool = False,
) -> tuple[list[dict[str, Any]], set[str]]:
    selected: list[dict[str, Any]] = []
    selected_speakers: set[str] = set()
    total = 0.0
    ordered_speakers = sorted(speakers, key=lambda value: stable_key(seed, f"{label}-speaker", value))
    ordered_records = {
        speaker_id: sorted(
            by_speaker[speaker_id], key=lambda record: stable_key(seed, f"{label}-clip", record["path"])
        )
        for speaker_id in ordered_speakers
    }
    if round_robin:
        speaker_totals: collections.Counter[str] = collections.Counter()
        indices: collections.Counter[str] = collections.Counter()
        while True:
            made_progress = False
            for speaker_id in ordered_speakers:
                speaker_records = ordered_records[speaker_id]
                while indices[speaker_id] < len(speaker_records):
                    record = speaker_records[indices[speaker_id]]
                    indices[speaker_id] += 1
                    if speaker_totals[speaker_id] + record["duration_seconds"] > speaker_cap_seconds:
                        continue
                    selected.append({**record, "split": label})
                    selected_speakers.add(speaker_id)
                    speaker_totals[speaker_id] += record["duration_seconds"]
                    total += record["duration_seconds"]
                    made_progress = True
                    if total >= target_seconds:
                        return selected, selected_speakers
                    break
            if not made_progress:
                break
        raise ValueError(f"{label} capacity is insufficient: selected {total:.3f} < target {target_seconds:.3f}")

    for speaker_id in ordered_speakers:
        speaker_total = 0.0
        speaker_records = ordered_records[speaker_id]
        for record in speaker_records:
            if speaker_total + record["duration_seconds"] > speaker_cap_seconds:
                continue
            selected.append({**record, "split": label})
            selected_speakers.add(speaker_id)
            speaker_total += record["duration_seconds"]
            total += record["duration_seconds"]
            if total >= target_seconds:
                return selected, selected_speakers
    raise ValueError(f"{label} capacity is insufficient: selected {total:.3f} < target {target_seconds:.3f}")


def select(
    archive_path: Path,
    *,
    train_target_seconds: float,
    eval_target_seconds: float,
    speaker_cap_seconds: float,
    seed: int,
) -> dict[str, Any]:
    records, locale = load_validated(archive_path)
    by_speaker: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        by_speaker[record["speaker_id"]].append(record)
    speakers = list(by_speaker)
    evaluation, eval_speakers = select_split(
        by_speaker, speakers, eval_target_seconds, speaker_cap_seconds, seed, "eval"
    )
    training, train_speakers = select_split(
        by_speaker,
        [speaker for speaker in speakers if speaker not in eval_speakers],
        train_target_seconds,
        speaker_cap_seconds,
        seed,
        "train",
        round_robin=True,
    )
    selected = sorted([*training, *evaluation], key=lambda record: (record["split"], record["path"]))
    return {
        "schema_version": 1,
        "archive": archive_path.name,
        "locale": locale,
        "seed": seed,
        "selection_policy": "stable SHA-256 speaker and clip order; whole clips; disjoint speakers; per-split speaker cap; training uses speaker round-robin",
        "speaker_cap_seconds": speaker_cap_seconds,
        "targets_seconds": {"train": train_target_seconds, "eval": eval_target_seconds},
        "selected_seconds": {
            "train": sum(record["duration_seconds"] for record in training),
            "eval": sum(record["duration_seconds"] for record in evaluation),
        },
        "selected_clip_counts": {"train": len(training), "eval": len(evaluation)},
        "selected_speaker_counts": {"train": len(train_speakers), "eval": len(eval_speakers)},
        "speaker_disjoint": not bool(train_speakers & eval_speakers),
        "records": selected,
        "note": "Sentence text and demographic fields are intentionally omitted.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-hours", type=float, required=True)
    parser.add_argument("--eval-minutes", type=float, default=25)
    parser.add_argument("--speaker-cap-minutes", type=float, default=5)
    parser.add_argument("--seed", type=int, default=141)
    args = parser.parse_args()
    result = select(
        args.archive,
        train_target_seconds=args.train_hours * 3600,
        eval_target_seconds=args.eval_minutes * 60,
        speaker_cap_seconds=args.speaker_cap_minutes * 60,
        seed=args.seed,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("selected_seconds", "selected_clip_counts", "selected_speaker_counts", "speaker_disjoint")}, indent=2))


if __name__ == "__main__":
    main()
