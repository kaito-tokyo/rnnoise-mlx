"""Measure waveform-quality diagnostics for WAV files inside corpus archives."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

from rnnoise_mlx.tools.inspect_corpus_archives import speaker_id, wav_streams


FRAME_SECONDS = 0.01
LOW_RMS_DBFS = -45.0
RETAINED_EDGE_PADDING_SECONDS = 0.3


def dbfs(value: float) -> float | None:
    return 20 * math.log10(value) if value > 0 else None


def edge_low_rms_seconds(samples: np.ndarray, rate: int, *, threshold_dbfs: float = LOW_RMS_DBFS) -> tuple[float, float]:
    frame_size = max(1, round(rate * FRAME_SECONDS))
    frame_count = len(samples) // frame_size
    if frame_count == 0:
        return 0.0, 0.0
    frames = samples[: frame_count * frame_size].reshape(frame_count, frame_size).astype(np.float32)
    rms = np.sqrt(np.mean(np.square(frames / 32768.0), axis=1))
    threshold = 10 ** (threshold_dbfs / 20)
    low = rms < threshold
    leading = 0
    for value in low:
        if not value:
            break
        leading += 1
    trailing = 0
    for value in low[::-1]:
        if not value:
            break
        trailing += 1
    return leading * FRAME_SECONDS, trailing * FRAME_SECONDS


def measure_pcm16(samples: np.ndarray, rate: int) -> dict[str, Any]:
    values = samples.astype(np.int32)
    absolute = np.abs(values)
    normalized = values.astype(np.float64) / 32768.0
    peak = float(absolute.max(initial=0)) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(normalized)))) if len(values) else 0.0
    leading, trailing = edge_low_rms_seconds(values, rate)
    duration = len(values) / rate
    edge_trimmed_duration = max(
        0.0,
        duration
        - max(0.0, leading - RETAINED_EDGE_PADDING_SECONDS)
        - max(0.0, trailing - RETAINED_EDGE_PADDING_SECONDS),
    )
    return {
        "sample_count": int(len(values)),
        "duration_seconds": duration,
        "edge_trimmed_duration_seconds": edge_trimmed_duration,
        "peak_dbfs": dbfs(peak),
        "rms_dbfs": dbfs(rms),
        "dc_offset_fraction": float(np.mean(normalized)) if len(values) else 0.0,
        "hard_clipped_sample_count": int(np.count_nonzero(absolute >= 32767)),
        "hard_clipped_fraction": float(np.mean(absolute >= 32767)) if len(values) else 0.0,
        "near_clipped_fraction": float(np.mean(absolute >= 32700)) if len(values) else 0.0,
        "low_level_sample_fraction": float(np.mean(absolute <= 104)) if len(values) else 0.0,
        "zero_sample_fraction": float(np.mean(values == 0)) if len(values) else 0.0,
        "leading_low_rms_seconds": leading,
        "trailing_low_rms_seconds": trailing,
    }


def diagnostic_flags(metrics: dict[str, Any]) -> list[str]:
    flags = []
    if metrics["hard_clipped_sample_count"]:
        flags.append("hard_clipping")
    if metrics["rms_dbfs"] is None or metrics["rms_dbfs"] < -35:
        flags.append("very_low_rms")
    if metrics["rms_dbfs"] is not None and metrics["rms_dbfs"] > -10:
        flags.append("very_high_rms")
    if abs(metrics["dc_offset_fraction"]) > 0.01:
        flags.append("large_dc_offset")
    if metrics["leading_low_rms_seconds"] > 1:
        flags.append("long_leading_low_rms")
    if metrics["trailing_low_rms_seconds"] > 1:
        flags.append("long_trailing_low_rms")
    if metrics["peak_dbfs"] is None or metrics["peak_dbfs"] < -20:
        flags.append("very_low_peak")
    return flags


def quantiles(rows: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    values = np.asarray([row[field] for row in rows if row[field] is not None], dtype=np.float64)
    if not len(values):
        return {key: None for key in ("min", "p01", "p05", "p50", "p95", "p99", "max")}
    points = np.quantile(values, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1])
    return dict(zip(("min", "p01", "p05", "p50", "p95", "p99", "max"), map(float, points), strict=True))


def audit(download_dir: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads((download_dir / "download-manifest.json").read_text())
    clips: list[dict[str, Any]] = []
    for archive_record in manifest["archives"]:
        archive_path = download_dir / archive_record["path"]
        for name, stream in wav_streams(archive_path):
            try:
                with wave.open(stream, "rb") as wav:
                    if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
                        raise ValueError("quality audit requires mono PCM16 WAV")
                    rate = wav.getframerate()
                    samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
                metrics = measure_pcm16(samples, rate)
                flags = diagnostic_flags(metrics)
                clips.append(
                    {
                        "source_id": archive_record["source_id"],
                        "archive": archive_record["path"],
                        "path": name,
                        "speaker_id": speaker_id(name),
                        "sample_rate_hz": rate,
                        **metrics,
                        "diagnostic_flags": flags,
                    }
                )
            except (EOFError, ValueError, wave.Error) as error:
                clips.append(
                    {
                        "source_id": archive_record["source_id"],
                        "archive": archive_record["path"],
                        "path": name,
                        "speaker_id": speaker_id(name),
                        "error": str(error),
                        "diagnostic_flags": ["decode_error"],
                        "duration_seconds": 0.0,
                    }
                )

    source_summaries: dict[str, Any] = {}
    for source_id in sorted({row["source_id"] for row in clips}):
        rows = [row for row in clips if row["source_id"] == source_id]
        valid = [row for row in rows if "error" not in row]
        reasons: collections.Counter[str] = collections.Counter()
        flagged_seconds: collections.Counter[str] = collections.Counter()
        for row in rows:
            for reason in row["diagnostic_flags"]:
                reasons[reason] += 1
                flagged_seconds[reason] += row["duration_seconds"]
        source_summaries[source_id] = {
            "clip_count": len(rows),
            "duration_seconds": sum(row["duration_seconds"] for row in rows),
            "edge_trimmed_duration_seconds": sum(
                row.get("edge_trimmed_duration_seconds", 0.0) for row in rows
            ),
            "speaker_count": len({row["speaker_id"] for row in valid}),
            "flagged_clip_count": sum(bool(row["diagnostic_flags"]) for row in rows),
            "flag_counts": dict(sorted(reasons.items())),
            "flagged_seconds_by_reason": dict(sorted(flagged_seconds.items())),
            "quantiles": {
                field: quantiles(valid, field)
                for field in (
                    "peak_dbfs",
                    "rms_dbfs",
                    "dc_offset_fraction",
                    "hard_clipped_fraction",
                    "near_clipped_fraction",
                    "low_level_sample_fraction",
                    "zero_sample_fraction",
                    "leading_low_rms_seconds",
                    "trailing_low_rms_seconds",
                )
            },
        }
    result = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "diagnostic_thresholds": {
            "hard_clipped_absolute_sample": 32767,
            "near_clipped_absolute_sample": 32700,
            "low_level_absolute_sample": 104,
            "edge_frame_seconds": FRAME_SECONDS,
            "edge_low_rms_dbfs": LOW_RMS_DBFS,
            "retained_edge_padding_seconds": RETAINED_EDGE_PADDING_SECONDS,
            "note": "Flags are audit cues, not automatic rejection decisions.",
        },
        "source_summaries": source_summaries,
        "clips": clips,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("download_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.download_dir, args.output)
    print(json.dumps(result["source_summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
