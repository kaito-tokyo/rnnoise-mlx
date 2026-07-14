"""Build an exact, deterministic weighted speech PCM from a JSON specification.

This tool deliberately handles speech only.  Background, foreground, and RIR
inputs are linked from an already prepared RNNoise dataset so stage-specific
speech populations can be compared without silently changing augmentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import BinaryIO


RATE = 48_000
SAMPLE_BYTES = 2
AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
AUGMENTATION_FILES = (
    "train_background.pcm",
    "eval_background.pcm",
    "train_foreground.pcm",
    "eval_foreground.pcm",
    "train_rir_list.txt",
    "eval_rir_list.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def exact_targets(total_samples: int, sources: list[dict[str, object]]) -> dict[str, int]:
    weights = [int(source["weight"]) for source in sources]
    if any(weight <= 0 for weight in weights):
        raise ValueError("source weights must be positive integers")
    denominator = sum(weights)
    floors = [total_samples * weight // denominator for weight in weights]
    remainder = total_samples - sum(floors)
    # Largest-remainder allocation, with source name as the stable tie-breaker.
    order = sorted(
        range(len(sources)),
        key=lambda index: (
            -(total_samples * weights[index] % denominator),
            str(sources[index]["name"]),
        ),
    )
    for index in order[:remainder]:
        floors[index] += 1
    return {str(source["name"]): floors[index] for index, source in enumerate(sources)}


def stable_audio_paths(root: Path, namespace: str) -> list[Path]:
    paths = [path for path in root.rglob("*") if path.suffix.lower() in AUDIO_SUFFIXES]
    return sorted(
        paths,
        key=lambda path: (
            hashlib.sha256(f"{namespace}:{path.relative_to(root).as_posix()}".encode()).digest(),
            path.as_posix(),
        ),
    )


def decode(path: Path) -> bytes:
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
            "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(RATE), "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed for {path}: {result.stderr.decode('utf-8', 'replace')}")
    return result.stdout


def write_pcm_prefix(output: BinaryIO, source: Path, samples: int) -> tuple[int, list[str]]:
    byte_count = samples * SAMPLE_BYTES
    with source.open("rb") as stream:
        remaining = byte_count
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"raw PCM shortage in {source}: need {remaining} more bytes")
            output.write(block)
            remaining -= len(block)
    return samples, [str(source.resolve())]


def write_audio_directory(
    output: BinaryIO, source: Path, samples: int, namespace: str
) -> tuple[int, list[str]]:
    written = 0
    used: list[str] = []
    for path in stable_audio_paths(source, namespace):
        if written >= samples:
            break
        pcm = decode(path)
        accepted = min(len(pcm) // SAMPLE_BYTES, samples - written)
        output.write(pcm[: accepted * SAMPLE_BYTES])
        written += accepted
        used.append(str(path.resolve()))
    if written != samples:
        raise ValueError(f"audio shortage in {source}: wrote {written}, need {samples}")
    return written, used


def render_split(
    specification: dict[str, object], split: str, destination: Path
) -> dict[str, object]:
    configured_sources = specification["sources"]
    assert isinstance(configured_sources, list)
    sources = []
    for configured in configured_sources:
        assert isinstance(configured, dict)
        weight = int(configured.get(f"{split}_weight", configured.get("weight", 0)))
        if weight < 0:
            raise ValueError(f"{configured['name']}: {split} weight must not be negative")
        if weight:
            sources.append({**configured, "weight": weight})
    if not sources:
        raise ValueError(f"no sources enabled for {split}")
    total_hours = float(specification[f"{split}_hours"])
    total_samples = round(total_hours * 3600 * RATE)
    targets = exact_targets(total_samples, sources)
    records: list[dict[str, object]] = []
    with destination.open("xb") as output:
        for source in sources:
            assert isinstance(source, dict)
            name = str(source["name"])
            path = Path(str(source[split])).expanduser().resolve()
            source_type = str(source.get(f"{split}_type", source.get("type", "audio-directory")))
            target = targets[name]
            if source_type == "pcm-s16le-48k-mono":
                written, used = write_pcm_prefix(output, path, target)
            elif source_type == "audio-directory":
                if not path.is_dir():
                    raise ValueError(f"audio directory does not exist: {path}")
                written, used = write_audio_directory(output, path, target, f"{split}:{name}")
            else:
                raise ValueError(f"unsupported source type for {name}: {source_type}")
            records.append(
                {
                    "name": name,
                    "weight": int(source["weight"]),
                    "source": str(path),
                    "source_type": source_type,
                    "samples": written,
                    "hours": written / RATE / 3600,
                    "files": used,
                }
            )
    return {
        "samples": total_samples,
        "hours": total_samples / RATE / 3600,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "sources": records,
    }


def link_augmentation(source: Path, output: Path) -> dict[str, str]:
    linked: dict[str, str] = {}
    for name in AUGMENTATION_FILES:
        origin = (source / name).resolve()
        if not origin.is_file():
            raise ValueError(f"missing augmentation input: {origin}")
        destination = output / name
        destination.symlink_to(origin)
        linked[name] = str(origin)
    return linked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specification", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--augmentation-prepared",
        type=Path,
        required=True,
        help="prepared dataset supplying unchanged noise, foreground, and RIR inputs",
    )
    args = parser.parse_args()
    specification = json.loads(args.specification.read_text())
    if specification.get("format_version") != 1:
        parser.error("unsupported specification format_version")
    names = [source["name"] for source in specification.get("sources", [])]
    if not names or len(names) != len(set(names)):
        parser.error("sources must have unique names")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        manifest = {
            "format_version": 1,
            "sample_rate_hz": RATE,
            "sample_format": "s16le",
            "specification": str(args.specification.resolve()),
            "splits": {
                split: render_split(specification, split, output / f"{split}_speech.pcm")
                for split in ("train", "eval")
            },
            "augmentation": link_augmentation(args.augmentation_prepared, output),
        }
        (output / "speech-mix-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    except Exception:
        # Never leave an apparently complete prepared directory behind.
        shutil.rmtree(output)
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
