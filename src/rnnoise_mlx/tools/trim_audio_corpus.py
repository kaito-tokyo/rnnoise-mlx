"""Trim denoised audio to a fixed margin before measured speech onset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import os
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
        temporary = output.with_name(f".{output.stem}.partial-{os.getpid()}.wav")
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
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    from .portable_storage import registered_volume_for_paths, volume_operation_guard

    guarded_paths = [input_root, output_root, args.filter_manifest.resolve()]
    guarded_paths.extend(
        (input_root / Path(str(record["path"])).with_suffix(".wav")).resolve()
        for record in records
    )
    portable_root = registered_volume_for_paths(guarded_paths)
    with volume_operation_guard(portable_root) if portable_root else nullcontext():
        output_root.mkdir(parents=True, exist_ok=True)
        margin_samples = round(args.margin_ms * args.sample_rate / 1000)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda r: trim_one(input_root, output_root, r,
                                                       args.threshold, margin_samples, args.sample_rate), records))
        manifest = {"format_version": 1, "input_root": str(input_root),
                    "filter_manifest": str(args.filter_manifest.resolve()),
                    "filter_manifest_sha256": sha256(args.filter_manifest),
                    "sample_rate_hz": args.sample_rate, "margin_samples": margin_samples,
                    "files": results}
        (output_root / "trim-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"trimmed_files": len(results), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
