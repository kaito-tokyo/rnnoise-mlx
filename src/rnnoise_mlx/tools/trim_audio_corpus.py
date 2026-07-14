"""Trim denoised audio to a fixed margin before measured speech onset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def trim_one(source_root: Path, output_root: Path, record: dict[str, object],
             threshold: float, margin_samples: int, sample_rate: int) -> dict[str, object]:
    relative = Path(str(record["path"])).with_suffix(".wav")
    source = source_root / relative
    output = output_root / relative
    onset = float(record["onsets_seconds"][f"{threshold:g}"])
    trim_samples = max(0, round(onset * sample_rate) - margin_samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.is_file():
        temporary = output.with_name(output.stem + ".partial.wav")
        temporary.unlink(missing_ok=True)
        try:
            subprocess.run([
                "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(source),
                "-af", f"atrim=start_sample={trim_samples},asetpts=PTS-STARTPTS",
                "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_f32le", str(temporary),
            ], check=True)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return {"input": relative.as_posix(), "input_sha256": sha256(source),
            "output": relative.as_posix(), "output_sha256": sha256(output),
            "onset_seconds": onset, "trim_samples": trim_samples,
            "retained_margin_samples": margin_samples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--filter-manifest", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=-40)
    parser.add_argument("--margin-ms", type=float, default=150)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        parser.error(f"output already exists: {args.output}")
    records = [r for r in json.loads(args.filter_manifest.read_text())["records"] if r["accepted"]]
    args.output.mkdir(parents=True, exist_ok=True)
    margin_samples = round(args.margin_ms * args.sample_rate / 1000)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda r: trim_one(args.input.resolve(), args.output.resolve(), r,
                                                   args.threshold, margin_samples, args.sample_rate), records))
    manifest = {"format_version": 1, "input_root": str(args.input.resolve()),
                "filter_manifest": str(args.filter_manifest.resolve()),
                "filter_manifest_sha256": sha256(args.filter_manifest),
                "sample_rate_hz": args.sample_rate, "margin_samples": margin_samples,
                "files": results}
    (args.output / "trim-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"trimmed_files": len(results), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
