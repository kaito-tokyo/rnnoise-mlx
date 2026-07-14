"""Build the fixed 100-hour multilingual speech PCM used by the base run.

The mix is intentionally simple: 20 hours of existing LibriTTS-R PCM,
50 hours spread across the locally acquired upstream RNNoise languages, and
30 hours from the selected Japanese, Mandarin, Korean, Vietnamese, and Arabic
corpora.  Audio is converted to 48 kHz mono s16le and every contribution is
recorded in a JSON manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO, Iterable


RATE = 48_000
SAMPLE_BYTES = 2


def decode(source: bytes | Path) -> bytes:
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", "pipe:0" if isinstance(source, bytes) else str(source),
        "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(RATE), "pipe:1",
    ]
    result = subprocess.run(
        command,
        input=source if isinstance(source, bytes) else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace"))
    return result.stdout


def append_exact(target: BinaryIO, pcm: bytes, remaining_samples: int) -> int:
    samples = min(len(pcm) // SAMPLE_BYTES, remaining_samples)
    target.write(pcm[: samples * SAMPLE_BYTES])
    return samples


def append_paths(target: BinaryIO, paths: Iterable[Path], target_samples: int) -> tuple[int, int]:
    written = 0
    files = 0
    for path in paths:
        if written >= target_samples:
            break
        written += append_exact(target, decode(path), target_samples - written)
        files += 1
    if written != target_samples:
        raise ValueError(f"audio shortage: wanted {target_samples} samples, wrote {written}")
    return written, files


def archive_member(archive_path: Path, name: str) -> bytes:
    lower = archive_path.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            return archive.read(name)
    if lower.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as archive:
            stream = archive.extractfile(name)
            if stream is None:
                raise ValueError(f"missing archive member: {name}")
            return stream.read()
    raise ValueError(f"unsupported archive: {archive_path}")


def waterfill_targets(capacities: dict[str, int], total: int) -> dict[str, int]:
    remaining = total
    active = set(capacities)
    result = {key: 0 for key in capacities}
    while active:
        share = remaining // len(active)
        constrained = {key for key in active if capacities[key] <= share}
        if not constrained:
            ordered = sorted(active)
            for index, key in enumerate(ordered):
                result[key] = share + (1 if index < remaining % len(ordered) else 0)
            remaining = 0
            break
        for key in sorted(constrained):
            result[key] = capacities[key]
            remaining -= capacities[key]
            active.remove(key)
    if remaining or sum(result.values()) != total:
        raise ValueError("official corpus does not meet the requested duration")
    return result


def official_records(multilingual_root: Path) -> dict[str, list[tuple[Path, str, int]]]:
    records: dict[str, list[tuple[Path, str, int]]] = defaultdict(list)
    for audit_path in sorted((multilingual_root / "archives").glob("*/waveform-quality-audit.json")):
        audit = json.loads(audit_path.read_text())
        for row in audit.get("clips", []):
            if "error" in row or row["source_id"].startswith("eng_"):
                continue
            archive_path = audit_path.parent / row["archive"]
            samples = round(float(row["duration_seconds"]) * RATE)
            records[row["source_id"]].append((archive_path, row["path"], samples))
    for rows in records.values():
        rows.sort(key=lambda row: (row[0].name, row[1]))
    return dict(records)


def append_official(target: BinaryIO, root: Path, target_samples: int) -> tuple[int, dict[str, dict[str, int]]]:
    records = official_records(root)
    capacities = {key: sum(row[2] for row in rows) for key, rows in records.items()}
    targets = waterfill_targets(capacities, target_samples)
    summary: dict[str, dict[str, int]] = {}
    total = 0
    for source_id in sorted(records):
        selected: dict[Path, list[str]] = defaultdict(list)
        estimated = 0
        for archive_path, member, samples in records[source_id]:
            selected[archive_path].append(member)
            estimated += samples
            if estimated >= targets[source_id]:
                break
        written = 0
        files = 0
        for archive_path in sorted(selected):
            count, used = append_archive_names(
                target,
                archive_path,
                selected[archive_path],
                targets[source_id] - written,
                require_exact=False,
            )
            written += count
            files += used
            if written >= targets[source_id]:
                break
        if written != targets[source_id]:
            raise ValueError(f"{source_id} shortage: {written} != {targets[source_id]}")
        summary[source_id] = {"samples": written, "files": files}
        total += written
        print(
            json.dumps({"stage": "official", "source": source_id, "hours": written / RATE / 3600}),
            file=sys.stderr,
            flush=True,
        )
    return total, summary


def append_raw_windows(target: BinaryIO, pcm_path: Path, offsets_path: Path) -> int:
    offsets = [int(line) for line in offsets_path.read_text().splitlines() if line.strip()]
    window_samples = 20 * RATE
    with pcm_path.open("rb") as source:
        for offset in offsets:
            source.seek(offset * SAMPLE_BYTES)
            block = source.read(window_samples * SAMPLE_BYTES)
            if len(block) != window_samples * SAMPLE_BYTES:
                raise ValueError(f"short raw PCM window at {offset}")
            target.write(block)
    return len(offsets) * window_samples


def append_raw_prefix(target: BinaryIO, pcm_path: Path, target_samples: int) -> int:
    remaining = target_samples * SAMPLE_BYTES
    with pcm_path.open("rb") as stream:
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"raw PCM shortage: {pcm_path}")
            target.write(block)
            remaining -= len(block)
    return target_samples


def common_voice_members(archive_path: Path) -> list[str]:
    csv.field_size_limit(sys.maxsize)
    with tarfile.open(archive_path, "r:gz") as archive:
        validated = next(member for member in archive if member.name.endswith("/validated.tsv"))
        stream = archive.extractfile(validated)
        if stream is None:
            raise ValueError("validated.tsv is unreadable")
        rows = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"), delimiter="\t")
        names = {row["path"] for row in rows}
        prefix = validated.name.rsplit("/", 1)[0] + "/clips/"
        return [prefix + name for name in sorted(names)]


def append_archive_names(
    target: BinaryIO,
    archive_path: Path,
    names: Iterable[str],
    target_samples: int,
    *,
    require_exact: bool = True,
) -> tuple[int, int]:
    written = 0
    files = 0
    ordered = list(names)
    lower = archive_path.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            for name in ordered:
                if written >= target_samples:
                    break
                written += append_exact(target, decode(archive.read(name)), target_samples - written)
                files += 1
    elif lower.endswith((".tar.gz", ".tgz")):
        wanted = set(ordered)
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                if written >= target_samples:
                    break
                if not member.isfile() or member.name not in wanted:
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"unreadable archive member: {member.name}")
                written += append_exact(target, decode(stream.read()), target_samples - written)
                files += 1
    else:
        raise ValueError(f"unsupported archive: {archive_path}")
    if require_exact and written != target_samples:
        raise ValueError(f"archive shortage: wanted {target_samples}, wrote {written}")
    return written, files


def unique_fosd_names(archive_path: Path) -> list[str]:
    result: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".mp3"):
                continue
            normalized = name.rsplit(".mp3", 1)[0].split(".mp3 ", 1)[0] + ".mp3"
            result.setdefault(normalized, name)
    return [result[key] for key in sorted(result)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-prepared", type=Path, required=True)
    parser.add_argument("--multilingual-root", type=Path, required=True)
    parser.add_argument("--english-offsets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume-from-95h",
        action="store_true",
        help="append the Vietnamese 5-hour block to an already verified 95-hour prefix",
    )
    args = parser.parse_args()
    if args.resume_from_95h:
        if not args.output.is_dir():
            raise ValueError(f"resume output does not exist: {args.output}")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    train_path = args.output / "train_speech.pcm"
    manifest: dict[str, object] = {"sample_rate_hz": RATE, "sample_format": "s16le", "sources": {}}
    sources: dict[str, object] = manifest["sources"]  # type: ignore[assignment]

    prepared = args.multilingual_root / "prepared"
    mode = "ab" if args.resume_from_95h else "wb"
    if args.resume_from_95h:
        expected_prefix = 95 * 3600 * RATE * SAMPLE_BYTES
        if train_path.stat().st_size != expected_prefix:
            raise ValueError(
                f"resume prefix size mismatch: {train_path.stat().st_size} != {expected_prefix}"
            )
        sources["libritts_r"] = {"samples": 20 * 3600 * RATE, "hours": 20}
        records = official_records(args.multilingual_root)
        capacities = {key: sum(row[2] for row in rows) for key, rows in records.items()}
        targets = waterfill_targets(capacities, 50 * 3600 * RATE)
        sources["official_non_english"] = {
            "samples": 50 * 3600 * RATE,
            "hours": 50,
            "languages": {
                key: {"samples": samples} for key, samples in sorted(targets.items())
            },
        }
        for language, hours in (("jpn", 8), ("cmn", 6), ("kor", 6), ("ara", 5)):
            sources[language] = {"samples": hours * 3600 * RATE, "hours": hours}

    with train_path.open(mode) as target:
        if not args.resume_from_95h:
            samples = append_raw_windows(target, args.base_prepared / "train_speech.pcm", args.english_offsets)
            sources["libritts_r"] = {"samples": samples, "hours": samples / RATE / 3600}
            print(json.dumps({"stage": "english", "hours": 20}), file=sys.stderr, flush=True)

            samples, detail = append_official(target, args.multilingual_root, 50 * 3600 * RATE)
            sources["official_non_english"] = {"samples": samples, "hours": 50, "languages": detail}

            fixed = {
                "jpn": (prepared / "cv26-ja-selection" / "train", 8),
                "cmn": (prepared / "aishell3-selection" / "train-48k-mono.pcm", 6),
                "kor": (prepared / "zeroth-korean-selection" / "train", 6),
                "ara": (prepared / "cv26-ar-selection" / "train", 5),
            }
            for language, (path, hours) in fixed.items():
                wanted = hours * 3600 * RATE
                if path.suffix == ".pcm":
                    samples = append_raw_prefix(target, path, wanted)
                    files = 1
                else:
                    samples, files = append_paths(target, sorted(path.iterdir()), wanted)
                sources[language] = {"samples": samples, "hours": hours, "files": files}
                print(json.dumps({"stage": "complement", "source": language, "hours": hours}), file=sys.stderr, flush=True)

        cv_archive = next((args.multilingual_root / "api/common-voice-scripted-26.0/vi").glob("*.tar.gz"))
        cv_samples = round(5274.433 * RATE)
        cv_written, cv_files = append_archive_names(target, cv_archive, common_voice_members(cv_archive), cv_samples)
        fosd_archive = args.multilingual_root / "archives/fosd-v4/k9sxg2twv4-4.zip"
        fosd_samples = 5 * 3600 * RATE - cv_written
        fosd_written, fosd_files = append_archive_names(target, fosd_archive, unique_fosd_names(fosd_archive), fosd_samples)
        sources["vie"] = {
            "samples": cv_written + fosd_written,
            "hours": 5,
            "common_voice_files": cv_files,
            "fosd_files": fosd_files,
        }
        print(json.dumps({"stage": "complement", "source": "vie", "hours": 5}), file=sys.stderr, flush=True)

    expected = 100 * 3600 * RATE * SAMPLE_BYTES
    if train_path.stat().st_size != expected:
        raise ValueError(f"unexpected output size: {train_path.stat().st_size} != {expected}")
    manifest["total_samples"] = expected // SAMPLE_BYTES
    manifest["total_hours"] = 100
    manifest["train_pcm_bytes"] = expected
    manifest["train_pcm_sha256"] = sha256(train_path)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
