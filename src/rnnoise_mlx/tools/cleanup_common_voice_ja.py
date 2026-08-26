"""Clean selected Common Voice Japanese clips with SpeexDSP and trim them."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Protocol


SPEEX_PREPROCESS_SET_DENOISE = 0
SPEEX_PREPROCESS_SET_AGC = 2
SPEEX_PREPROCESS_SET_DEREVERB = 8
SPEEX_PREPROCESS_SET_NOISE_SUPPRESS = 18


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class Preprocessor(Protocol):
    def process(self, samples: bytes) -> bytes: ...

    def close(self) -> None: ...


class SpeexDSP:
    """Minimal ctypes wrapper around the SpeexDSP preprocessor API."""

    def __init__(self, library: Path):
        self.library_path = library
        self.lib = ctypes.CDLL(str(library))
        self.lib.speex_preprocess_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.speex_preprocess_state_init.restype = ctypes.c_void_p
        self.lib.speex_preprocess_state_destroy.argtypes = [ctypes.c_void_p]
        self.lib.speex_preprocess_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.speex_preprocess_ctl.restype = ctypes.c_int
        self.lib.speex_preprocess_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16)]
        self.lib.speex_preprocess_run.restype = ctypes.c_int

    def create(self, frame_size: int, sample_rate: int, noise_suppress_db: int) -> "SpeexPreprocessor":
        return SpeexPreprocessor(self.lib, frame_size, sample_rate, noise_suppress_db)


class SpeexPreprocessor:
    def __init__(self, lib: ctypes.CDLL, frame_size: int, sample_rate: int,
                 noise_suppress_db: int):
        self.lib = lib
        self.frame_size = frame_size
        self.state = lib.speex_preprocess_state_init(frame_size, sample_rate)
        if not self.state:
            raise RuntimeError("speex_preprocess_state_init failed")
        try:
            self._set(SPEEX_PREPROCESS_SET_DENOISE, 1)
            self._set(SPEEX_PREPROCESS_SET_NOISE_SUPPRESS, noise_suppress_db)
            self._set(SPEEX_PREPROCESS_SET_AGC, 0)
            # VAD defaults to off; its setter emits an upstream warning even for zero.
            self._set(SPEEX_PREPROCESS_SET_DEREVERB, 0)
        except Exception:
            self.close()
            raise

    def _set(self, request: int, value: int) -> None:
        parameter = ctypes.c_int(value)
        if self.lib.speex_preprocess_ctl(self.state, request, ctypes.byref(parameter)) != 0:
            raise RuntimeError(f"speex_preprocess_ctl failed for request {request}")

    def process(self, samples: bytes) -> bytes:
        if len(samples) != self.frame_size * 2:
            raise ValueError("SpeexDSP frame has the wrong byte length")
        frame_type = ctypes.c_int16 * self.frame_size
        frame = frame_type.from_buffer_copy(samples)
        self.lib.speex_preprocess_run(self.state, frame)
        return bytes(frame)

    def close(self) -> None:
        if self.state:
            self.lib.speex_preprocess_state_destroy(self.state)
            self.state = None


def resolve_library(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    discovered = ctypes.util.find_library("speexdsp")
    if discovered and "/" in discovered:
        candidates.append(Path(discovered))
    candidates.extend([
        Path("/opt/homebrew/lib/libspeexdsp.dylib"),
        Path("/usr/local/lib/libspeexdsp.dylib"),
        Path("/usr/lib/libspeexdsp.dylib"),
    ])
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "libspeexdsp was not found; install speexdsp or pass --speex-library"
    )


def require_internal_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == Path("/Volumes") or Path("/Volumes") in resolved.parents:
        from .portable_storage import DEFAULT_ROOT, load_volume_config

        configured = Path(os.environ.get("RNNOISE_MLX_STORAGE_ROOT", DEFAULT_ROOT)).resolve()
        if resolved != configured and configured not in resolved.parents:
            raise ValueError(
                "output below /Volumes is allowed only on the registered "
                f"rnnoise-mlx training volume: {resolved}"
            )
        load_volume_config(configured)
    return resolved


def decode_pcm(source: Path, sample_rate: int) -> bytes:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(source),
         "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-c:a", "pcm_s16le", "-"],
        check=True,
        stdout=subprocess.PIPE,
    )
    if len(result.stdout) % 2:
        raise RuntimeError(f"ffmpeg returned an incomplete PCM sample for {source}")
    return result.stdout


def denoise_pcm(pcm: bytes, preprocessor: Preprocessor, frame_size: int) -> bytes:
    frame_bytes = frame_size * 2
    original_size = len(pcm)
    padded_size = ((original_size + frame_bytes - 1) // frame_bytes) * frame_bytes
    padded = pcm + bytes(padded_size - original_size)
    output = bytearray()
    for offset in range(0, padded_size, frame_bytes):
        output.extend(preprocessor.process(padded[offset:offset + frame_bytes]))
    return bytes(output[:original_size])


def encode_wav(pcm: bytes, output: Path, sample_rate: int) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "s16le",
         "-ar", str(sample_rate), "-ac", "1", "-i", "-",
         "-c:a", "pcm_s16le", str(output)],
        input=pcm,
        check=True,
    )


def cleanup_one(source_root: Path, output_root: Path, record: dict[str, Any],
                processor_factory: Any, threshold: float, margin_samples: int,
                sample_rate: int, frame_size: int) -> dict[str, Any]:
    source_relative = Path(str(record["path"]))
    if source_relative.is_absolute() or ".." in source_relative.parts:
        raise ValueError(f"input path escapes the corpus root: {source_relative}")
    source = source_root / source_relative
    output_relative = source_relative.with_suffix(".wav")
    output = output_root / output_relative
    onset = float(record["onsets_seconds"][f"{threshold:g}"])
    trim_samples = max(0, round(onset * sample_rate) - margin_samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.is_file():
        temporary = output.with_name(output.stem + ".partial.wav")
        temporary.unlink(missing_ok=True)
        processor = processor_factory()
        try:
            pcm = decode_pcm(source, sample_rate)
            cleaned = denoise_pcm(pcm, processor, frame_size)
            encode_wav(cleaned[trim_samples * 2:], temporary, sample_rate)
            temporary.replace(output)
        finally:
            processor.close()
            temporary.unlink(missing_ok=True)
    return {
        "input": source_relative.as_posix(),
        "input_sha256": sha256(source),
        "output": output_relative.as_posix(),
        "output_sha256": sha256(output),
        "output_bytes": output.stat().st_size,
        "onset_seconds": onset,
        "trim_samples": trim_samples,
        "retained_margin_samples": margin_samples,
    }


def load_records(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    records = [record for record in manifest["records"] if record["accepted"]]
    if not records:
        raise ValueError("filter manifest contains no accepted clips")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="extracted Common Voice Japanese clips")
    parser.add_argument(
        "output",
        type=Path,
        help="internal-SSD or registered rnnoise-mlx training-volume output directory",
    )
    parser.add_argument("--filter-manifest", type=Path, required=True)
    parser.add_argument("--speex-library", type=Path)
    parser.add_argument("--noise-suppress-db", type=int, default=-12)
    parser.add_argument("--threshold", type=float, default=-40)
    parser.add_argument("--margin-ms", type=float, default=150)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--frame-ms", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source_root = args.input.resolve()
    try:
        output_root = require_internal_output(args.output)
    except ValueError as error:
        parser.error(str(error))
    filter_manifest = args.filter_manifest.resolve()
    if not source_root.is_dir():
        parser.error(f"input directory does not exist: {source_root}")
    if not filter_manifest.is_file():
        parser.error(f"filter manifest does not exist: {filter_manifest}")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.noise_suppress_db > 0:
        parser.error("--noise-suppress-db must be zero or negative")
    if args.sample_rate < 1 or args.frame_ms < 1:
        parser.error("--sample-rate and --frame-ms must be positive")
    if args.sample_rate * args.frame_ms % 1000:
        parser.error("--frame-ms must produce an integral sample count")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required")
    if output_root.exists() and not args.resume:
        parser.error(f"output already exists: {output_root}")

    try:
        library_path = resolve_library(args.speex_library)
        speex = SpeexDSP(library_path)
        records = load_records(filter_manifest)
    except (FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))

    missing = [str(record["path"]) for record in records
               if not (source_root / str(record["path"])).is_file()]
    if missing:
        parser.error(f"accepted input clips are missing: {len(missing)} (first: {missing[0]})")

    frame_size = args.sample_rate * args.frame_ms // 1000
    margin_samples = round(args.margin_ms * args.sample_rate / 1000)
    output_root.mkdir(parents=True, exist_ok=args.resume)
    factory = lambda: speex.create(frame_size, args.sample_rate, args.noise_suppress_db)
    worker = lambda record: cleanup_one(
        source_root, output_root, record, factory, args.threshold, margin_samples,
        args.sample_rate, frame_size,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(worker, records))

    manifest = {
        "format_version": 1,
        "tool": "speexdsp-preprocessor",
        "input_root": str(source_root),
        "filter_manifest": str(filter_manifest),
        "filter_manifest_sha256": sha256(filter_manifest),
        "speex_library": str(library_path),
        "speex_library_sha256": sha256(library_path),
        "sample_rate_hz": args.sample_rate,
        "frame_size_samples": frame_size,
        "frame_ms": args.frame_ms,
        "noise_suppress_db": args.noise_suppress_db,
        "denoise": True,
        "agc": False,
        "vad": False,
        "dereverb": False,
        "threshold_dbfs": args.threshold,
        "margin_samples": margin_samples,
        "files": results,
    }
    manifest_path = output_root / "cleanup-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cleaned_files": len(results), "output": str(output_root)}))


if __name__ == "__main__":
    main()
