"""Build a deterministic fixed-WAV evaluation set from a noise audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np


RATE = 48_000
SNRS = (-5, 0, 5, 10, 20)
SAMPLES = 5 * RATE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable(path: str) -> bytes:
    return hashlib.sha256(f"141:evaluation:{path}".encode()).digest()


def decode(path: Path, samples: int = SAMPLES) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
            "-ar", str(RATE), "-ac", "1", "-f", "f32le", "-acodec", "pcm_f32le", "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    audio = np.frombuffer(result.stdout, dtype="<f4").astype(np.float64)
    if not len(audio):
        raise ValueError(f"decoded empty audio: {path}")
    if len(audio) < samples:
        audio = np.tile(audio, (samples + len(audio) - 1) // len(audio))
    return audio[:samples]


def write_wav(path: Path, audio: np.ndarray) -> None:
    values = np.clip(audio, -1, 1)
    pcm = np.rint(values * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm.tobytes())


def output_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def decode_audited(record: dict[str, Any]) -> np.ndarray:
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"audited source checksum differs: {path}")
    return decode(path)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_rms = np.sqrt(np.mean(clean * clean) + 1e-20)
    noise_rms = np.sqrt(np.mean(noise * noise) + 1e-20)
    scaled_noise = noise * (clean_rms / (noise_rms * 10 ** (snr_db / 20)))
    mixture = clean + scaled_noise
    peak = max(
        np.max(np.abs(clean), initial=0),
        np.max(np.abs(scaled_noise), initial=0),
        np.max(np.abs(mixture), initial=0),
    )
    scale = min(1.0, 0.99 / peak) if peak else 1.0
    return clean * scale, scaled_noise * scale, mixture * scale


def evaluation_groups(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = report["records"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for category in (
        "dns_typing", "dns_door", "dns_squeak", "dns_dragging",
        "dns_copy-machine", "dns_human", "dns_fan",
    ):
        groups[category] = [row for row in records if row["accepted"] and row["split"] == "eval" and row["category"] == category]
    for source in ("mka-lenovo", "mka-msi", "mka-mac", "mka-messenger", "mka-zoom", "mka-hp"):
        groups[source] = [row for row in records if row["accepted"] and row["split"] == "eval" and row["source"] == source]
    for source in ("multi-pressure-highp", "multi-pressure-mediump", "multi-pressure-lowp"):
        groups[source] = [row for row in records if row["accepted"] and row["split"] == "eval" and row["source"] == source]
    groups["musan-rejected"] = [
        row for row in records
        if row["category"] == "musan_curated"
        and not row["accepted"]
        and "decode_error" not in row.get("exclusion_reasons", [])
    ]
    missing = [name for name, rows in groups.items() if not rows]
    if missing:
        raise ValueError(f"empty evaluation groups: {', '.join(missing)}")
    return {
        name: min(rows, key=lambda row: stable(row["relative_path"]))
        for name, rows in groups.items()
    }


def build(audit_path: Path, clean_path: Path, output: Path) -> dict[str, Any]:
    report = json.loads(audit_path.read_text())
    clean = decode(clean_path)
    groups = evaluation_groups(report)
    cases = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".noise-evaluation-", dir=output.parent
    ) as temporary_directory:
        staging = Path(temporary_directory)
        staged_files = []
        for group, record in sorted(groups.items()):
            noise = decode_audited(record)
            for snr in SNRS:
                clean_scaled, noise_scaled, mixture = mix_at_snr(clean, noise, snr)
                stem = f"{group}__snr-{snr:+d}"
                staged_paths = {
                    name: staging / f"{stem}__{name}.wav"
                    for name in ("clean", "noise", "mixture")
                }
                write_wav(staged_paths["clean"], clean_scaled)
                write_wav(staged_paths["noise"], noise_scaled)
                write_wav(staged_paths["mixture"], mixture)
                outputs = {}
                for name, staged in staged_paths.items():
                    target = output / staged.name
                    staged_files.append((staged, target))
                    outputs[name] = {**output_record(staged), "path": str(target.resolve())}
                cases.append({
                    "id": stem,
                    "group": group,
                    "snr_db": snr,
                    "source": record["relative_path"],
                    "source_identity": record["identity"],
                    "outputs": outputs,
                })
        staged_clean_only = staging / "clean-only.wav"
        peak = np.max(np.abs(clean), initial=0)
        write_wav(staged_clean_only, clean * min(1.0, 0.99 / peak) if peak else clean)
        clean_only = {
            **output_record(staged_clean_only),
            "path": str((output / "clean-only.wav").resolve()),
        }
        staged_files.append((staged_clean_only, output / "clean-only.wav"))
        manifest = {
            "format_version": 1,
            "kind": "rnnoise-fixed-noise-evaluation",
            "sample_rate_hz": RATE,
            "duration_seconds": SAMPLES / RATE,
            "snrs_db": list(SNRS),
            "audit": {
                "path": str(audit_path.resolve()),
                "sha256": sha256_file(audit_path),
            },
            "clean_source": str(clean_path.resolve()),
            "clean_only": clean_only,
            "cases": cases,
        }
        staged_manifest = staging / "manifest.json"
        staged_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        output.mkdir(parents=True, exist_ok=True)
        for staged, target in staged_files:
            os.replace(staged, target)
        os.replace(staged_manifest, output / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.audit.resolve(), args.clean.resolve(), args.output.resolve())
    print(json.dumps({"case_count": len(result["cases"]), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
