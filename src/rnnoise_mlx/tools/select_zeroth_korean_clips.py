"""Select a deterministic speaker-disjoint Zeroth Korean population."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from rnnoise_mlx.tools.inspect_corpus_archives import flac_info


def key(seed: int, label: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{label}:{value}".encode()).digest()


def load_records(archive_path: Path) -> list[dict[str, Any]]:
    records = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".flac"):
                continue
            parts = Path(member.name).parts
            if len(parts) < 3 or parts[0] not in {"train_data_01", "test_data_01"}:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot extract {member.name}")
            rate, channels, bits, samples = flac_info(stream)
            records.append(
                {
                    "path": member.name,
                    "dataset": parts[0],
                    "speaker_id": parts[-2],
                    "duration_seconds": samples / rate,
                    "sample_rate_hz": rate,
                    "channels": channels,
                    "bits_per_sample": bits,
                }
            )
    return records


def select(archive_path: Path, *, train_target_seconds: float, eval_target_seconds: float, speaker_cap_seconds: float, seed: int) -> dict[str, Any]:
    records = load_records(archive_path)
    by_speaker: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    split_by_speaker: dict[str, str] = {}
    for record in records:
        by_speaker[record["speaker_id"]].append(record)
        split_by_speaker[record["speaker_id"]] = record["dataset"]
    for speaker_id, rows in by_speaker.items():
        if any(row["dataset"] != rows[0]["dataset"] for row in rows):
            raise ValueError(f"speaker crosses train/test split: {speaker_id}")
    eval_speakers: set[str] = set()
    evaluation: list[dict[str, Any]] = []
    eval_total = 0.0
    for speaker_id in sorted((s for s, split in split_by_speaker.items() if split == "test_data_01"), key=lambda s: key(seed, "eval-speaker", s)):
        eval_speakers.add(speaker_id)
        for record in sorted(by_speaker[speaker_id], key=lambda r: key(seed, "eval-clip", r["path"])):
            evaluation.append({**record, "split": "eval"})
            eval_total += record["duration_seconds"]
        if eval_total >= eval_target_seconds:
            break
    if eval_total < eval_target_seconds:
        raise ValueError(f"test capacity is insufficient: {eval_total:.3f}")
    train_speakers = [s for s, split in split_by_speaker.items() if split == "train_data_01" and s not in eval_speakers]
    ordered = sorted(train_speakers, key=lambda s: key(seed, "train-speaker", s))
    rows_by_speaker = {s: sorted(by_speaker[s], key=lambda r: key(seed, "train-clip", r["path"])) for s in ordered}
    totals: collections.Counter[str] = collections.Counter()
    indices: collections.Counter[str] = collections.Counter()
    training: list[dict[str, Any]] = []
    train_total = 0.0
    while train_total < train_target_seconds:
        progress = False
        for speaker_id in ordered:
            rows = rows_by_speaker[speaker_id]
            while indices[speaker_id] < len(rows):
                record = rows[indices[speaker_id]]
                indices[speaker_id] += 1
                if totals[speaker_id] + record["duration_seconds"] > speaker_cap_seconds:
                    continue
                training.append({**record, "split": "train"})
                totals[speaker_id] += record["duration_seconds"]
                train_total += record["duration_seconds"]
                progress = True
                break
            if train_total >= train_target_seconds:
                break
        if not progress:
            raise ValueError(f"train capacity is insufficient: {train_total:.3f} < {train_target_seconds:.3f}")
    selected = sorted([*training, *evaluation], key=lambda r: (r["split"], r["path"]))
    return {
        "schema_version": 1,
        "archive": archive_path.name,
        "seed": seed,
        "speaker_cap_seconds": speaker_cap_seconds,
        "targets_seconds": {"train": train_target_seconds, "eval": eval_target_seconds},
        "selected_seconds": {"train": train_total, "eval": eval_total},
        "selected_clip_counts": {"train": len(training), "eval": len(evaluation)},
        "selected_speaker_counts": {"train": len({r["speaker_id"] for r in training}), "eval": len(eval_speakers)},
        "speaker_disjoint": not bool({r["speaker_id"] for r in training} & eval_speakers),
        "records": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-hours", type=float, default=6)
    parser.add_argument("--eval-minutes", type=float, default=25)
    parser.add_argument("--speaker-cap-minutes", type=float, default=5)
    parser.add_argument("--seed", type=int, default=144)
    args = parser.parse_args()
    result = select(args.archive, train_target_seconds=args.train_hours * 3600, eval_target_seconds=args.eval_minutes * 60, speaker_cap_seconds=args.speaker_cap_minutes * 60, seed=args.seed)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("selected_seconds", "selected_clip_counts", "selected_speaker_counts", "speaker_disjoint")}, indent=2))


if __name__ == "__main__":
    main()
