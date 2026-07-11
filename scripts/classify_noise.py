#!/usr/bin/env python3
"""Classify noise WAVs by stationarity and transient content.

The thresholds are intentionally exposed: use --calibrate to inspect corpus
quantiles before treating the labels as final.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Metrics:
    duration_s: float
    rms_dbfs: float
    rms_cv: float
    spectral_flux_median: float
    spectral_flux_p95: float
    spectral_flux_peak: float
    onset_density_hz: float
    crest_factor: float
    silence_ratio: float
    clipping_ratio: float
    stationarity_score: float
    transient_score: float


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels, width, rate, frames = (
            wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        )
        raw = wav.readframes(frames)
    if width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = b[:, 0].astype(np.int32) | b[:, 1].astype(np.int32) << 8 | b[:, 2].astype(np.int32) << 16
        values = (values ^ 0x800000) - 0x800000
        audio = values.astype(np.float32) / 8388608
    elif width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError(f"unsupported PCM width: {width}")
    return audio.reshape(-1, channels).mean(axis=1), rate


def analyze(audio: np.ndarray, rate: int, frame_ms: float = 20, hop_ms: float = 10) -> Metrics:
    frame = max(64, round(rate * frame_ms / 1000))
    hop = max(32, round(rate * hop_ms / 1000))
    if audio.size < frame:
        audio = np.pad(audio, (0, frame - audio.size))
    count = 1 + (audio.size - frame) // hop
    frames = np.lib.stride_tricks.sliding_window_view(audio, frame)[::hop][:count].copy()
    windowed = frames * np.hanning(frame).astype(np.float32)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    spectra = np.abs(np.fft.rfft(windowed, axis=1))
    spectra /= np.maximum(np.linalg.norm(spectra, axis=1, keepdims=True), 1e-12)
    flux = np.sqrt(np.mean(np.maximum(np.diff(spectra, axis=0), 0) ** 2, axis=1))
    flux = np.pad(flux, (1, 0))
    median_flux = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median_flux))) + 1e-9
    onset_threshold = median_flux + 6 * mad
    onset_density = float(np.count_nonzero(flux > onset_threshold) / max(audio.size / rate, 1e-9))
    rms_mean = float(np.mean(rms))
    rms_cv = float(np.std(rms) / max(rms_mean, 1e-9))
    flux_p95 = float(np.quantile(flux, 0.95))
    flux_peak = float(np.max(flux))
    crest = float(np.max(np.abs(audio)) / max(np.sqrt(np.mean(audio * audio)), 1e-9))
    silence = float(np.mean(rms < 10 ** (-60 / 20)))
    clipping = float(np.mean(np.abs(audio) >= 0.999))

    # Scores are rankings, not probabilities. Corpus-level calibration is expected.
    stationarity = float(1 / (1 + 2.5 * rms_cv + 80 * median_flux + 0.5 * silence))
    transient = float(np.log1p(8 * crest) * (120 * flux_p95 + 40 * flux_peak + onset_density))
    return Metrics(
        audio.size / rate, 20 * np.log10(max(rms_mean, 1e-12)), rms_cv,
        median_flux, flux_p95, flux_peak, onset_density, crest, silence, clipping,
        stationarity, transient,
    )


def label(metrics: Metrics, bg_threshold: float, fg_threshold: float, min_background_seconds: float) -> str:
    bg = metrics.duration_s >= min_background_seconds and metrics.stationarity_score >= bg_threshold
    fg = metrics.transient_score >= fg_threshold
    if metrics.clipping_ratio > 0.001:
        return "review"
    if bg and fg:
        return "mixed"
    if bg:
        return "background"
    if fg:
        return "foreground"
    return "review"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--background-threshold", type=float, default=0.23)
    parser.add_argument("--foreground-threshold", type=float, default=45.0)
    parser.add_argument("--min-background-seconds", type=float, default=10.0)
    parser.add_argument("--json", action="store_true", help="write JSON instead of CSV")
    parser.add_argument("--organize-dir", type=Path, help="create label folders containing symlinks")
    args = parser.parse_args()
    paths = sorted(args.input.rglob("*.wav"))
    rows = []
    for path in paths:
        try:
            metrics = analyze(*read_wav(path))
            row = {
                "path": str(path.relative_to(args.input)),
                "label": label(metrics, args.background_threshold, args.foreground_threshold, args.min_background_seconds),
                **asdict(metrics),
            }
        except Exception as error:
            row = {"path": str(path.relative_to(args.input)), "label": "error", "error": str(error)}
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.json:
        args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    else:
        fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in ("path", "label"), key))
        with args.output.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    if args.organize_dir:
        for row in rows:
            if row["label"] == "error":
                continue
            source = (args.input / row["path"]).resolve()
            label_dir = args.organize_dir / row["label"]
            label_dir.mkdir(parents=True, exist_ok=True)
            link_name = row["path"].replace("/", "__")
            link = label_dir / link_name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(os.path.relpath(source, label_dir.resolve()))
    print(f"classified {len(rows)} files -> {args.output}")


if __name__ == "__main__":
    main()
