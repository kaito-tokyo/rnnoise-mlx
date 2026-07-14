"""Inspect WAV headers inside downloaded ZIP or tar.gz corpus archives."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import tarfile
import wave
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Iterator


SPEAKER_RE = re.compile(r"^([^_]+)_([^_]+)_")


def speaker_id(name: str, source_id: str | None = None) -> str | None:
    if source_id in {"aishell3", "zeroth_korean"}:
        parent = Path(name).parent.name
        return parent or None
    match = SPEAKER_RE.match(Path(name).name)
    return f"{match.group(1)}:{match.group(2)}" if match else None


def audio_streams(path: Path) -> Iterator[tuple[str, BinaryIO]]:
    lower = path.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.lower().endswith((".wav", ".flac")):
                    with archive.open(name) as stream:
                        yield name, stream
        return
    if lower.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if member.isfile() and member.name.lower().endswith((".wav", ".flac")):
                    stream = archive.extractfile(member)
                    if stream is not None:
                        with stream:
                            yield member.name, stream
        return
    raise ValueError(f"unsupported archive format: {path}")


def wav_streams(path: Path) -> Iterator[tuple[str, BinaryIO]]:
    for name, stream in audio_streams(path):
        if name.lower().endswith(".wav"):
            yield name, stream


def flac_info(stream: BinaryIO) -> tuple[int, int, int, int]:
    header = stream.read(42)
    if len(header) < 42 or header[:4] != b"fLaC" or header[4] & 0x7F != 0:
        raise ValueError("missing FLAC STREAMINFO header")
    block_length = int.from_bytes(header[5:8], "big")
    if block_length != 34:
        raise ValueError(f"unexpected FLAC STREAMINFO length: {block_length}")
    packed = int.from_bytes(header[18:26], "big")
    sample_rate = (packed >> 44) & 0xFFFFF
    channels = ((packed >> 41) & 0x7) + 1
    bits_per_sample = ((packed >> 36) & 0x1F) + 1
    total_samples = packed & 0xFFFFFFFFF
    if not sample_rate or not total_samples:
        raise ValueError("invalid FLAC STREAMINFO values")
    return sample_rate, channels, bits_per_sample, total_samples


def inspect_archive(path: Path, source_id: str | None = None) -> dict[str, Any]:
    total_seconds = 0.0
    formats: collections.Counter[tuple[int, int, int, str]] = collections.Counter()
    speakers: collections.Counter[str] = collections.Counter()
    failures: list[dict[str, str]] = []
    clip_count = 0
    for name, stream in audio_streams(path):
        try:
            if name.lower().endswith(".flac"):
                rate, channels, bits, frames = flac_info(stream)
                width = (bits + 7) // 8
                compression = "FLAC"
            else:
                with wave.open(stream, "rb") as wav:
                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    channels = wav.getnchannels()
                    width = wav.getsampwidth()
                    compression = wav.getcomptype()
        except (EOFError, ValueError, wave.Error) as error:
            failures.append({"path": name, "error": str(error)})
            continue
        clip_count += 1
        duration = frames / rate
        total_seconds += duration
        formats[(rate, channels, width, compression)] += 1
        identity = speaker_id(name, source_id)
        if identity is not None:
            speakers[identity] += duration
    return {
        "path": path.name,
        "clip_count": clip_count,
        "duration_seconds": total_seconds,
        "speaker_count": len(speakers),
        "speaker_duration_seconds": dict(sorted(speakers.items())),
        "formats": [
            {
                "sample_rate_hz": key[0],
                "channels": key[1],
                "sample_width_bytes": key[2],
                "compression": key[3],
                "clip_count": count,
            }
            for key, count in sorted(formats.items())
        ],
        "header_failures": failures,
    }


def inspect_download(download_dir: Path, output: Path) -> dict[str, Any]:
    download_manifest = json.loads((download_dir / "download-manifest.json").read_text())
    source_by_path = {record["path"]: record["source_id"] for record in download_manifest["archives"]}
    archives = []
    for record in download_manifest["archives"]:
        inspection = inspect_archive(download_dir / record["path"], record["source_id"])
        inspection["source_id"] = record["source_id"]
        inspection["archive_sha256"] = record["sha256"]
        archives.append(inspection)
    source_totals: dict[str, dict[str, Any]] = {}
    for source in sorted(set(source_by_path.values())):
        rows = [row for row in archives if row["source_id"] == source]
        speaker_keys = {speaker for row in rows for speaker in row["speaker_duration_seconds"]}
        source_totals[source] = {
            "archive_count": len(rows),
            "clip_count": sum(row["clip_count"] for row in rows),
            "duration_seconds": sum(row["duration_seconds"] for row in rows),
            "speaker_count": len(speaker_keys),
            "header_failure_count": sum(len(row["header_failures"]) for row in rows),
        }
    result = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "archives": archives,
        "source_totals": source_totals,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("download_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = inspect_download(args.download_dir, args.output)
    print(json.dumps(result["source_totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
