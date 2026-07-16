"""Build a deterministic fixed-WAV evaluation set from a noise audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np


RATE = 48_000
SNRS = (-5, 0, 5, 10, 20)
SAMPLES = 5 * RATE


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


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_rms = np.sqrt(np.mean(clean * clean) + 1e-20)
    noise_rms = np.sqrt(np.mean(noise * noise) + 1e-20)
    scaled_noise = noise * (clean_rms / (noise_rms * 10 ** (snr_db / 20)))
    mixture = clean + scaled_noise
    peak = np.max(np.abs(mixture), initial=0)
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
        if row["category"] == "musan_curated" and not row["accepted"]
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
    for group, record in sorted(groups.items()):
        noise = decode(Path(record["path"]))
        for snr in SNRS:
            clean_scaled, noise_scaled, mixture = mix_at_snr(clean, noise, snr)
            stem = f"{group}__snr-{snr:+d}"
            paths = {
                "clean": output / f"{stem}__clean.wav",
                "noise": output / f"{stem}__noise.wav",
                "mixture": output / f"{stem}__mixture.wav",
            }
            write_wav(paths["clean"], clean_scaled)
            write_wav(paths["noise"], noise_scaled)
            write_wav(paths["mixture"], mixture)
            cases.append({
                "id": stem,
                "group": group,
                "snr_db": snr,
                "source": record["relative_path"],
                "source_identity": record["identity"],
                "paths": {name: str(path.resolve()) for name, path in paths.items()},
            })
    clean_only = output / "clean-only.wav"
    peak = np.max(np.abs(clean), initial=0)
    write_wav(clean_only, clean * min(1.0, 0.99 / peak) if peak else clean)
    manifest = {
        "format_version": 1,
        "kind": "rnnoise-fixed-noise-evaluation",
        "sample_rate_hz": RATE,
        "duration_seconds": SAMPLES / RATE,
        "snrs_db": list(SNRS),
        "audit": str(audit_path.resolve()),
        "clean_source": str(clean_path.resolve()),
        "clean_only": str(clean_only.resolve()),
        "cases": cases,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
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
