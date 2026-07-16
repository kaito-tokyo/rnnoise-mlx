"""Publish and verify immutable RNNoise feature generations."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from .generate_features import BYTES_PER_SEQUENCE, RNG_ALGORITHM


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(manifest: dict) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _generation_id(feature_digest: str) -> str:
    return f"sha256:{feature_digest}"


def _generation_metadata(source_manifest: dict) -> dict:
    generator = source_manifest["generator"]
    return {
        "seed": generator["seed"],
        "sequence_start": generator["sequence_start"],
        "speech_offset_start": generator["speech_offset_start"],
        "disable_foreground": generator["disable_foreground"],
    }


def verify_generation(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    feature = path / "features.f32"
    required = {
        "format_version": 1,
        "kind": "rnnoise-mlx-feature-generation",
        "frames_per_sequence": 2000,
        "values_per_frame": 98,
        "rng_algorithm": RNG_ALGORITHM,
    }
    mismatches = [key for key, value in required.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError(f"feature manifest fields differ: {', '.join(mismatches)}")
    if manifest.get("output", {}).get("filename") != "features.f32":
        raise ValueError("feature manifest output filename differs")
    source_manifest = manifest.get("source_manifest")
    if not isinstance(source_manifest, dict) or manifest.get(
        "source_manifest_sha256"
    ) != _manifest_digest(source_manifest):
        raise ValueError("embedded source manifest checksum differs")
    expected_size = int(manifest["sequence_count"]) * BYTES_PER_SEQUENCE
    actual_size = feature.stat().st_size
    if actual_size != expected_size or manifest["output"].get("bytes") != actual_size:
        raise ValueError(f"feature size differs: {feature}")
    digest = sha256(feature)
    if digest != manifest["output"]["sha256"]:
        raise ValueError(f"feature checksum differs: {feature}")
    source_output = source_manifest.get("output", {})
    source_generator = source_manifest.get("generator", {})
    semantic_matches = (
        source_manifest.get("format_version") == 1
        and source_manifest.get("kind") == "rnnoise-training-features"
        and source_manifest.get("sequence_count") == manifest["sequence_count"]
        and source_manifest.get("frames_per_sequence") == manifest["frames_per_sequence"]
        and source_manifest.get("values_per_frame") == manifest["values_per_frame"]
        and source_generator.get("rng_algorithm") == manifest["rng_algorithm"]
        and source_output.get("bytes") == actual_size
        and source_output.get("sha256") == digest
        and manifest.get("generation_id") == _generation_id(digest)
        and manifest.get("generation") == _generation_metadata(source_manifest)
    )
    if not semantic_matches:
        raise ValueError("embedded source manifest does not match feature generation")
    sidecar = (path / "features.f32.sha256").read_text().split()[0]
    if sidecar != digest:
        raise ValueError(f"checksum sidecar differs: {feature}")
    return manifest


def _load_source_manifest(path: Path, source: Path, sequence_count: int) -> tuple[dict, str]:
    manifest = json.loads(path.read_text())
    source_digest = sha256(source)
    valid = (
        manifest.get("format_version") == 1
        and manifest.get("kind") == "rnnoise-training-features"
        and manifest.get("sequence_count") == sequence_count
        and manifest.get("frames_per_sequence") == 2000
        and manifest.get("values_per_frame") == 98
        and manifest.get("generator", {}).get("rng_algorithm") == RNG_ALGORITHM
        and manifest.get("output", {}).get("bytes") == source.stat().st_size
        and manifest.get("output", {}).get("sha256") == source_digest
    )
    if not valid:
        raise ValueError(f"source feature manifest does not match source: {path}")
    return manifest, _manifest_digest(manifest)


def _write_index(version_root: Path) -> None:
    with (version_root / ".index.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entries = []
        for manifest_path in sorted(version_root.glob("*/*/manifest.json")):
            manifest = json.loads(manifest_path.read_text())
            entries.append({
                "id": manifest["generation_id"],
                "path": str(manifest_path.parent.relative_to(version_root)),
                "sequence_count": manifest["sequence_count"],
                "sha256": manifest["output"]["sha256"],
            })
        temporary = version_root / f".index.json.tmp-{uuid.uuid4().hex}"
        temporary.write_text(json.dumps({"format_version": 1, "generations": entries}, indent=2) + "\n")
        os.replace(temporary, version_root / "index.json")


def publish(
    source: Path,
    source_manifest: Path,
    destination: Path,
    sequence_count: int,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    generated_manifest, generated_manifest_digest = _load_source_manifest(
        source_manifest, source, sequence_count
    )
    source_digest = generated_manifest["output"]["sha256"]
    generation_id = _generation_id(source_digest)
    metadata = _generation_metadata(generated_manifest)
    lock = destination.parent / f".{destination.name}.lock"
    with lock.open("a+b") as generation_lock:
        fcntl.flock(generation_lock, fcntl.LOCK_EX)
        temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
        try:
            if destination.exists():
                existing = verify_generation(destination)
                matches = (
                    existing["generation_id"] == generation_id
                    and existing["sequence_count"] == sequence_count
                    and existing["rng_algorithm"] == RNG_ALGORITHM
                    and existing["generation"] == metadata
                    and existing["output"]["sha256"] == source_digest
                    and existing["source_manifest_sha256"] == generated_manifest_digest
                )
                if not matches:
                    raise FileExistsError(
                        f"generation conditions differ from immutable destination: {destination}"
                    )
                _write_index(destination.parents[1])
                return destination
            temporary.mkdir()
            shutil.copyfile(source, temporary / "features.f32")
            digest = sha256(temporary / "features.f32")
            if digest != source_digest:
                raise ValueError("source feature changed while it was being published")
            manifest = {
                "format_version": 1,
                "kind": "rnnoise-mlx-feature-generation",
                "generation_id": generation_id,
                "sequence_count": sequence_count,
                "frames_per_sequence": 2000,
                "values_per_frame": 98,
                "rng_algorithm": RNG_ALGORITHM,
                "output": {"filename": "features.f32", "bytes": source.stat().st_size, "sha256": digest},
                "generation": metadata,
                "source_manifest_sha256": generated_manifest_digest,
                "source_manifest": generated_manifest,
            }
            (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            (temporary / "features.f32.sha256").write_text(f"{digest}  features.f32\n")
            verify_generation(temporary)
            os.replace(temporary, destination)
            _write_index(destination.parents[1])
            return destination
        finally:
            shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("generation", type=Path)
    add = sub.add_parser("publish")
    add.add_argument("source", type=Path)
    add.add_argument("destination", type=Path)
    add.add_argument(
        "--source-manifest",
        type=Path,
        help="generate_features manifest (default: SOURCE with .manifest.json suffix)",
    )
    add.add_argument("--sequence-count", type=int, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify_generation(args.generation), indent=2))
    else:
        source_manifest = args.source_manifest or args.source.with_suffix(".manifest.json")
        if not source_manifest.is_file():
            parser.error(f"source manifest does not exist: {source_manifest}")
        path = publish(args.source, source_manifest, args.destination, args.sequence_count)
        print(path)


if __name__ == "__main__":
    main()
