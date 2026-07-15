"""Denoise a selected audio corpus and record an auditable file manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess


AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def clean_one(
    executable: Path, model: Path, source_root: Path, output_root: Path, source: Path
) -> dict[str, object]:
    relative = source.relative_to(source_root)
    output = (output_root / relative).with_suffix(".wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.is_file():
        # AVAudioFile selects the container from the final extension, so keep
        # `.wav` while still making the file visibly transactional.
        temporary = output.with_name(output.stem + ".partial.wav")
        temporary.unlink(missing_ok=True)
        try:
            subprocess.run(
                [str(executable), "--model", str(model), str(source), str(temporary)],
                check=True,
            )
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "input": relative.as_posix(),
        "input_sha256": sha256(source),
        "output": output.relative_to(output_root).as_posix(),
        "output_sha256": sha256(output),
        "output_bytes": output.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="compiled RNNoiseGraph.mlmodelc")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--filter-manifest", type=Path,
                        help="JSON manifest whose accepted records select input paths")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed WAV files in an existing output directory",
    )
    args = parser.parse_args()
    source_root = args.input.resolve()
    output_root = args.output.resolve()
    executable = args.executable.resolve()
    model = args.model.resolve()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not executable.is_file():
        parser.error(f"denoiser executable does not exist: {executable}")
    if not model.is_dir():
        parser.error(f"compiled model does not exist: {model}")
    if output_root.exists() and not args.resume:
        parser.error(f"output already exists: {output_root}")
    paths = sorted(
        path for path in source_root.rglob("*") if path.suffix.lower() in AUDIO_SUFFIXES
    )
    filter_manifest_sha256 = None
    if args.filter_manifest:
        filtered = json.loads(args.filter_manifest.read_text())
        accepted = {record["path"] for record in filtered["records"] if record["accepted"]}
        paths = [path for path in paths if path.relative_to(source_root).as_posix() in accepted]
        filter_manifest_sha256 = sha256(args.filter_manifest)
    if not paths:
        parser.error(f"no supported audio below {source_root}")
    output_root.mkdir(parents=True, exist_ok=args.resume)
    worker = lambda path: clean_one(executable, model, source_root, output_root, path)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(worker, paths))
    manifest = {
        "format_version": 1,
        "input_root": str(source_root),
        "filter_manifest": str(args.filter_manifest.resolve()) if args.filter_manifest else None,
        "filter_manifest_sha256": filter_manifest_sha256,
        "model": str(model),
        "model_files": {
            path.relative_to(model).as_posix(): sha256(path)
            for path in sorted(model.rglob("*")) if path.is_file()
        },
        "files": records,
    }
    (output_root / "cleaning-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"cleaned_files": len(records), "output": str(output_root)}))


if __name__ == "__main__":
    main()
