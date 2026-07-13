"""Prepare deterministic RNNoise inputs from LibriTTS-R, MUSAN and RIRS_NOISES."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable


SEED = 141
AUDIO_SUFFIXES = {".wav", ".flac"}


def stable_score(path: Path, namespace: str) -> int:
    value = f"{SEED}:{namespace}:{path.as_posix()}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)


def relative_lines(paths: Iterable[Path], base: Path) -> str:
    return "".join(f"{path.relative_to(base).as_posix()}\n" for path in paths)


def write_manifest(path: Path, paths: list[Path], base: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(relative_lines(paths, base))


def stable_order(paths: list[Path], namespace: str, base: Path) -> list[Path]:
    return sorted(paths, key=lambda path: stable_score(path.relative_to(base), namespace))


def resolve_subset(root: Path, name: str) -> Path:
    variants = (name, name.replace("_", "-"))
    matches = [p for p in root.rglob("*") if p.is_dir() and p.name in variants]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {name} directory below {root}, found {len(matches)}")
    return matches[0]


def partition(
    paths: list[Path], namespace: str, eval_percent: int, base: Path | None = None
) -> tuple[list[Path], list[Path]]:
    train, evaluation = [], []
    for path in paths:
        identity = path.relative_to(base) if base is not None else path
        (evaluation if stable_score(identity, namespace) % 100 < eval_percent else train).append(path)
    return train, evaluation


def manifest_command(args: argparse.Namespace) -> None:
    corpus = args.corpus.resolve()
    output = args.output.resolve()
    train_speech = audio_files(resolve_subset(corpus, "train_clean_100"))
    if args.include_train_360:
        train_speech += audio_files(resolve_subset(corpus, "train_clean_360"))
    eval_speech = audio_files(resolve_subset(corpus, "dev_clean"))
    test_speech = audio_files(resolve_subset(corpus, "test_clean"))

    musan = resolve_subset(corpus, "musan")
    noise_files = audio_files(musan / "noise")
    music_files = audio_files(musan / "music")
    babble_files = audio_files(musan / "speech")
    train_noise, eval_noise = partition(noise_files + music_files, "background", 10, musan)
    train_fg, eval_fg = partition(babble_files, "foreground", 10, musan)

    rirs = resolve_subset(corpus, "RIRS_NOISES")
    # The archive also contains point-source noises. Only impulse-response trees
    # are valid inputs for dump_features -rir_list.
    # `real_rirs_isotropic_noises` mixes real RIRs and long noise recordings.
    # Use the unambiguous simulated RIR tree; point-source/isotropic noises are
    # already covered by MUSAN and must not be passed as impulse responses.
    rir_files = audio_files(rirs / "simulated_rirs")
    train_rir, eval_rir = partition(rir_files, "rir", 10, rirs)
    groups = {
        "train_speech": train_speech,
        "eval_speech": eval_speech,
        "test_speech": test_speech,
        "train_background": train_noise,
        "eval_background": eval_noise,
        "train_foreground": train_fg,
        "eval_foreground": eval_fg,
        "train_rir": train_rir,
        "eval_rir": eval_rir,
    }
    empty = [name for name, paths in groups.items() if not paths]
    if empty:
        raise SystemExit(f"empty dataset groups: {', '.join(empty)}")
    for name, paths in groups.items():
        write_manifest(output / f"{name}.txt", stable_order(paths, name, corpus), corpus)
    metadata = {
        "format_version": 1,
        "seed": SEED,
        "corpus_root": str(corpus),
        "eval_noise_percent": 10,
        "include_train_clean_360": args.include_train_360,
        "counts": {name: len(paths) for name, paths in groups.items()},
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def load_manifest(path: Path, corpus: Path, limit: int | None = None) -> list[Path]:
    paths = [corpus / line for line in path.read_text().splitlines() if line]
    if limit is not None:
        paths = paths[:limit]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} manifest inputs do not exist; first: {missing[0]}")
    return paths


def ffmpeg_bytes(source: Path, sample_format: str, codec: str) -> bytes:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(source), "-ar", "48000", "-ac", "1", "-f", sample_format, "-acodec", codec, "-"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout


def concatenate(
    manifest: Path,
    corpus: Path,
    output: Path,
    limit: int | None = None,
    workers: int = 1,
) -> None:
    paths = load_manifest(manifest, corpus, limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as destination:
        convert = lambda source: ffmpeg_bytes(source, "s16le", "pcm_s16le")
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ffmpeg") as executor:
            # executor.map yields in input order, preserving byte-for-byte determinism.
            converted = executor.map(convert, paths)
            for index, audio in enumerate(converted, 1):
                destination.write(audio)
                if "speech" in manifest.stem:
                    destination.write(bytes(9600))  # 100 ms at 48 kHz, mono s16le
                if index % 100 == 0:
                    print(f"{manifest.stem}: {index}/{len(paths)}", file=sys.stderr)


def render_command(args: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required")
    if args.workers < 1:
        raise SystemExit("workers must be at least 1")
    corpus = args.corpus.resolve()
    manifests = args.manifests.resolve()
    output = args.output.resolve()
    for split in ("train", "eval"):
        speech_limit = args.speech_limit if args.speech_limit is not None else args.limit
        noise_limit = args.noise_limit if args.noise_limit is not None else args.limit
        rir_limit = args.rir_limit if args.rir_limit is not None else args.limit
        concatenate(manifests / f"{split}_speech.txt", corpus, output / f"{split}_speech.pcm", speech_limit, args.workers)
        concatenate(manifests / f"{split}_background.txt", corpus, output / f"{split}_background.pcm", noise_limit, args.workers)
        concatenate(manifests / f"{split}_foreground.txt", corpus, output / f"{split}_foreground.pcm", noise_limit, args.workers)

        rir_dir = output / f"{split}_rirs"
        rir_dir.mkdir(parents=True, exist_ok=True)
        rir_paths = load_manifest(manifests / f"{split}_rir.txt", corpus, rir_limit)
        rir_list = []
        for index, source in enumerate(rir_paths):
            target = rir_dir / f"{index:06d}.f32"
            target.write_bytes(ffmpeg_bytes(source, "f32le", "pcm_f32le"))
            rir_list.append(str(target.resolve()))
        (output / f"{split}_rir_list.txt").write_text("\n".join(rir_list) + "\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    manifests = commands.add_parser("manifests", help="create deterministic train/eval manifests")
    manifests.add_argument("--corpus", type=Path, required=True)
    manifests.add_argument("--output", type=Path, required=True)
    manifests.add_argument("--include-train-360", action="store_true")
    manifests.set_defaults(func=manifest_command)
    render = commands.add_parser("render", help="convert manifest audio to RNNoise PCM and RIR files")
    render.add_argument("--corpus", type=Path, required=True)
    render.add_argument("--manifests", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument(
        "--limit",
        type=int,
        help="render only the first N files per group (pipeline validation only)",
    )
    render.add_argument("--speech-limit", type=int, help="maximum speech files per split")
    render.add_argument("--noise-limit", type=int, help="maximum background/foreground files per split")
    render.add_argument("--rir-limit", type=int, help="maximum RIR files per split")
    render.add_argument("--workers", type=int, default=8, help="parallel FFmpeg processes (default: 8)")
    render.set_defaults(func=render_command)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
