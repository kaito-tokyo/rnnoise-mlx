"""Decode extracted audio selections and audit waveform quality in parallel."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rnnoise_mlx.tools.audit_corpus_audio_quality import diagnostic_flags, measure_pcm16, quantiles


def inspect_clip(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    path = root / record["path"]
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        streams = json.loads(probe.stdout)["streams"]
        if len(streams) != 1:
            raise ValueError(f"expected one audio stream, found {len(streams)}")
        source = streams[0]
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "48000",
                "-ac",
                "1",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
        )
        samples = np.frombuffer(decoded.stdout, dtype="<i2")
        metrics = measure_pcm16(samples, 48000)
        return {
            **record,
            "source_codec": source["codec_name"],
            "source_sample_rate_hz": int(source["sample_rate"]),
            "source_channels": int(source["channels"]),
            **metrics,
            "diagnostic_flags": diagnostic_flags(metrics),
        }
    except (KeyError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        return {**record, "error": str(error), "duration_seconds": 0.0, "diagnostic_flags": ["decode_error"]}


def audit(
    root: Path,
    output: Path,
    *,
    workers: int = 8,
    inspector: Callable[[Path, dict[str, Any]], dict[str, Any]] = inspect_clip,
) -> dict[str, Any]:
    manifest_path = root / "extraction-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    records = manifest["records"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        clips = list(executor.map(lambda record: inspector(root, record), records))
    clips.sort(key=lambda record: record["path"])
    split_summaries: dict[str, Any] = {}
    for split in sorted({record["split"] for record in clips}):
        rows = [record for record in clips if record["split"] == split]
        valid = [record for record in rows if "error" not in record]
        flags: collections.Counter[str] = collections.Counter()
        formats: collections.Counter[tuple[str, int, int]] = collections.Counter()
        for record in rows:
            flags.update(record["diagnostic_flags"])
            if "error" not in record:
                formats[(record["source_codec"], record["source_sample_rate_hz"], record["source_channels"])] += 1
        split_summaries[split] = {
            "clip_count": len(rows),
            "decoded_clip_count": len(valid),
            "decode_error_count": len(rows) - len(valid),
            "duration_seconds": sum(record["duration_seconds"] for record in valid),
            "edge_trimmed_duration_seconds": sum(record["edge_trimmed_duration_seconds"] for record in valid),
            "speaker_count": len({record["speaker_id"] for record in valid}),
            "flagged_clip_count": sum(bool(record["diagnostic_flags"]) for record in rows),
            "flag_counts": dict(sorted(flags.items())),
            "source_formats": [
                {"codec": key[0], "sample_rate_hz": key[1], "channels": key[2], "clip_count": count}
                for key, count in sorted(formats.items())
            ],
            "quantiles": {
                field: quantiles(valid, field)
                for field in (
                    "peak_dbfs",
                    "rms_dbfs",
                    "hard_clipped_fraction",
                    "near_clipped_fraction",
                    "leading_low_rms_seconds",
                    "trailing_low_rms_seconds",
                )
            },
        }
    result = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "extraction_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "selection_manifest_sha256": manifest["selection_manifest_sha256"],
        "decode_format": "48000 Hz mono signed 16-bit PCM",
        "split_summaries": split_summaries,
        "clips": clips,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    result = audit(args.root, args.output, workers=args.workers)
    print(json.dumps(result["split_summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
