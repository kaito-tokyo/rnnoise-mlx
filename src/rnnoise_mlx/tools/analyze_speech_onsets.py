"""Measure leading active-audio onset distributions for an audio corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def threshold_key(value: float) -> str:
    return f"{value:g}"


def decode(path: Path, sample_rate: int) -> np.ndarray:
    command = [
        "ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar",
        str(sample_rate), "-f", "f32le", "-",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    return np.frombuffer(result.stdout, dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def onsets(samples: np.ndarray, sample_rate: int, thresholds: list[float]) -> dict[str, float | None]:
    frame = round(sample_rate * 0.020)
    hop = round(sample_rate * 0.010)
    if len(samples) < frame:
        return {threshold_key(value): None for value in thresholds}
    starts = np.arange(0, len(samples) - frame + 1, hop)
    power = np.array([
        np.mean(np.square(samples[start:start + frame], dtype=np.float64))
        for start in starts
    ])
    dbfs = 10.0 * np.log10(np.maximum(power, 1e-12))
    output: dict[str, float | None] = {}
    for threshold in thresholds:
        active = dbfs >= threshold
        onset = None
        for index in range(max(0, len(active) - 2)):
            if active[index:index + 3].all():
                onset = float(starts[index] / sample_rate)
                break
        output[threshold_key(threshold)] = onset
    return output


def summarize(values: list[float | None]) -> dict[str, object]:
    detected = np.array([value for value in values if value is not None], dtype=np.float64)
    total = len(values)
    boundaries = [0.05, 0.10, 0.25, 0.50, 1.00]
    bin_edges = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, float("inf")]
    return {
        "clips": total,
        "detected": int(len(detected)),
        "undetected": total - int(len(detected)),
        "percentiles_seconds": {
            str(p): float(np.percentile(detected, p)) for p in [10, 25, 50, 75, 90, 95, 99]
        } if len(detected) else {},
        "cumulative": {
            f"lt_{boundary:.2f}s": {
                "count": int(np.sum(detected < boundary)),
                "percent_all": float(100.0 * np.sum(detected < boundary) / total),
            }
            for boundary in boundaries
        },
        "histogram": {
            f"{left:.2f}-{right:.2f}s" if np.isfinite(right) else f"ge_{left:.2f}s": {
                "count": int(np.sum((detected >= left) & (detected < right))),
                "percent_all": float(100.0 * np.sum((detected >= left) & (detected < right)) / total),
            }
            for left, right in zip(bin_edges, bin_edges[1:])
        },
        "ge_0.50s": {
            "count": int(np.sum(detected >= 0.50)),
            "percent_all": float(100.0 * np.sum(detected >= 0.50) / total),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[-35, -40, -45])
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--filter-threshold", type=float, default=-40)
    parser.add_argument("--minimum-onset-ms", type=float, default=250)
    parser.add_argument("--selection-manifest", type=Path)
    args = parser.parse_args()
    paths = sorted(path for path in args.root.rglob("*") if path.suffix.lower() in {".mp3", ".wav", ".flac"})

    selection = {}
    selection_sha256 = None
    if args.selection_manifest:
        selected = json.loads(args.selection_manifest.read_text())
        selection = {
            f"{record['split']}/{record['path']}": record for record in selected["records"]
        }
        selection_sha256 = sha256(args.selection_manifest)
        found = {path.relative_to(args.root).as_posix() for path in paths}
        if found != set(selection):
            missing = sorted(set(selection) - found)
            extra = sorted(found - set(selection))
            parser.error(f"selection/input mismatch: missing={len(missing)}, extra={len(extra)}")

    def measure(path: Path) -> dict[str, object]:
        samples = decode(path, args.sample_rate)
        relative = path.relative_to(args.root).as_posix()
        measured = onsets(samples, args.sample_rate, args.thresholds)
        onset = measured.get(threshold_key(args.filter_threshold))
        reasons = []
        if onset is None:
            reasons.append("speech_onset_not_detected")
        elif onset < args.minimum_onset_ms / 1000:
            reasons.append("leading_margin_below_250ms")
        metadata = selection.get(relative, {})
        return {
            "path": relative,
            "input_sha256": sha256(path),
            "duration_seconds": len(samples) / args.sample_rate,
            "speaker_id": metadata.get("speaker_id"),
            "split": metadata.get("split", Path(relative).parts[0]),
            "onsets_seconds": measured,
            "accepted": not reasons,
            "reasons": reasons,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(measure, paths))
    summary = {
        "method": {
            "sample_rate": args.sample_rate,
            "frame_ms": 20,
            "hop_ms": 10,
            "required_consecutive_active_frames": 3,
            "thresholds_dbfs": args.thresholds,
        },
        "root": str(args.root),
        "selection_manifest": str(args.selection_manifest.resolve()) if args.selection_manifest else None,
        "selection_manifest_sha256": selection_sha256,
        "filter": {
            "threshold_dbfs": args.filter_threshold,
            "minimum_onset_seconds": args.minimum_onset_ms / 1000,
        },
        "groups": {
            group: {
                str(threshold): summarize([
                    record["onsets_seconds"][threshold_key(threshold)]
                    for record in records
                    if group == "all" or Path(record["path"]).parts[0] == group
                ])
                for threshold in args.thresholds
            }
            for group in ["all", "train", "eval"]
        },
        "filter_summaries": {
            group: {
                "input_clips": len([r for r in records if group == "all" or r["split"] == group]),
                "accepted_clips": len([r for r in records if r["accepted"] and (group == "all" or r["split"] == group)]),
                "accepted_seconds": sum(r["duration_seconds"] for r in records if r["accepted"] and (group == "all" or r["split"] == group)),
                "accepted_speakers": len({r["speaker_id"] for r in records if r["accepted"] and r["speaker_id"] and (group == "all" or r["split"] == group)}),
            }
            for group in ["all", "train", "eval"]
        },
        "speaker_disjoint": not bool(
            {r["speaker_id"] for r in records if r["accepted"] and r["split"] == "train"}
            & {r["speaker_id"] for r in records if r["accepted"] and r["split"] == "eval"}
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "filter-manifest.json").write_text(
        json.dumps({"format_version": 1, "method": summary["method"], "filter": summary["filter"],
                    "selection_manifest": summary["selection_manifest"],
                    "selection_manifest_sha256": selection_sha256, "records": records},
                   indent=2, sort_keys=True) + "\n"
    )
    with (args.output / "onsets.csv").open("w", newline="") as handle:
        fields = ["path", "input_sha256", "split", "speaker_id", "duration_seconds",
                  "accepted", "reasons"] + [f"onset_{threshold:g}_dbfs" for threshold in args.thresholds]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {"path": record["path"], "input_sha256": record["input_sha256"],
                   "split": record["split"], "speaker_id": record["speaker_id"],
                   "duration_seconds": record["duration_seconds"], "accepted": record["accepted"],
                   "reasons": ",".join(record["reasons"])}
            row.update({
                f"onset_{threshold:g}_dbfs": record["onsets_seconds"][str(threshold)]
                for threshold in args.thresholds
            })
            writer.writerow(row)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
