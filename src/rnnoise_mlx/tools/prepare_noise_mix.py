"""Audit and render the DNS/MKA/Multi-Pressure RNNoise noise mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SEED = 141
RATE = 48_000
SEQUENCE_SAMPLES = 2_000 * RATE // 100
DNS_FOREGROUND_WEIGHTS = {
    "dns_typing": 20,
    "dns_door": 15,
    "dns_squeak": 10,
    "dns_dragging": 5,
    "dns_copy-machine": 5,
    "dns_human": 5,
}
FOREGROUND_WEIGHTS = {**DNS_FOREGROUND_WEIGHTS, "mka": 25, "multi_pressure": 15}
BACKGROUND_WEIGHTS = {"dns_fan": 90, "musan_curated": 10}
MKA_COLLECTIONS = ("Lenovo", "MSI", "Mac", "Messenger", "Zoom", "hp")
MKA_SOURCES = tuple(f"mka-{collection.lower()}" for collection in MKA_COLLECTIONS)
PRESSURES = ("HighP", "MediumP", "LowP")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_score(value: str, namespace: str = "split") -> int:
    data = f"{SEED}:{namespace}:{value}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def split_for(identity: str) -> str:
    return "eval" if stable_score(identity) % 100 < 10 else "train"


def _native_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("expected uncompressed PCM16 WAV")
        rate = wav.getframerate()
        channels = wav.getnchannels()
        raw = wav.readframes(wav.getnframes())
    samples = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).astype(np.int32).mean(axis=1).astype("<i2")
    return samples, rate


def decode_48k(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
            "-ar", str(RATE), "-ac", "1", "-f", "s16le", "-acodec", "pcm_s16le", "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    return np.frombuffer(result.stdout, dtype="<i2")


def metrics(path: Path) -> dict[str, Any]:
    samples, rate = _native_pcm(path)
    if not len(samples):
        raise ValueError("empty audio")
    values = samples.astype(np.float64) / 32768.0
    frame_size = max(1, round(rate * 0.01))
    frame_count = len(values) // frame_size
    frames = values[: frame_count * frame_size].reshape(frame_count, frame_size)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-20)
    frame_db = 20 * np.log10(frame_rms)
    return {
        "duration_seconds": len(values) / rate,
        "sample_rate_hz": rate,
        "channels": 1,
        "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(values * values) + 1e-20)),),
        "peak_dbfs": float(20 * np.log10(np.max(np.abs(values), initial=1e-20))),
        "active_fraction": float(np.mean(frame_db > -45)) if len(frame_db) else 0.0,
        "frame_rms_spread_db": (
            float(np.quantile(frame_db, 0.95) - np.quantile(frame_db, 0.05))
            if len(frame_db) else math.inf
        ),
        "hard_clipped_fraction": float(np.mean(np.abs(samples.astype(np.int32)) >= 32767)),
        "dc_offset_fraction": float(np.mean(values)) if len(values) else 0.0,
    }


def _record(path: Path, corpus: Path, source: str, category: str, identity: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "relative_path": path.relative_to(corpus).as_posix(),
        "source": source,
        "category": category,
        "identity": identity,
        "split": split_for(identity),
        "sha256": sha256_file(path),
        "accepted": True,
        "exclusion_reasons": [],
    }
    try:
        record["metrics"] = metrics(path)
    except (EOFError, ValueError, wave.Error, OSError) as error:
        record["accepted"] = False
        record["exclusion_reasons"] = ["decode_error"]
        record["error"] = str(error)
    return record


def dns_records(corpus: Path) -> list[dict[str, Any]]:
    root = corpus / "DNS5"
    records = []
    for path in sorted(root.rglob("*.wav")):
        label = path.name.split("_", 1)[0]
        match = re.search(r"Freesound_validated_(\d+)", path.name)
        identity = f"dns:{match.group(1) if match else path.stem}"
        if label == "fan":
            category = "dns_fan"
        elif label in {"breath", "munching"}:
            category = "dns_human"
        else:
            category = f"dns_{label}"
        record = _record(path, corpus, "dns5-freesound", category, identity)
        if label == "mislabeled" or category not in BACKGROUND_WEIGHTS | FOREGROUND_WEIGHTS:
            record["accepted"] = False
            record["exclusion_reasons"].append("excluded_dns_class")
        records.append(record)
    return records


def mka_records(corpus: Path) -> list[dict[str, Any]]:
    root = corpus / "MKA" / "MKA datasets"
    records = []
    seen_hashes: set[str] = set()
    for collection in MKA_COLLECTIONS:
        for path in sorted((root / collection).rglob("*.wav")):
            key = path.parent.name
            take = re.search(r"(\d+)$", path.stem)
            identity = f"mka:{key}:{take.group(1) if take else path.stem}"
            record = _record(path, corpus, f"mka-{collection.lower()}", "mka", identity)
            if record["sha256"] in seen_hashes:
                record["accepted"] = False
                record["exclusion_reasons"].append("exact_duplicate")
            else:
                seen_hashes.add(record["sha256"])
            if record["accepted"]:
                quality = record["metrics"]
                if quality["hard_clipped_fraction"] > 0.0001:
                    record["accepted"] = False
                    record["exclusion_reasons"].append("excessive_clipping")
                if abs(quality["dc_offset_fraction"]) > 0.01:
                    record["accepted"] = False
                    record["exclusion_reasons"].append("large_dc_offset")
            records.append(record)
    return records


def multi_pressure_records(corpus: Path) -> list[dict[str, Any]]:
    root = corpus / "multi_pressure_keyboard" / "dataset"
    records = []
    for pressure in PRESSURES:
        for path in sorted((root / pressure).glob("*.wav")):
            match = re.match(r"Key-(.+)-[HML]$", path.stem)
            key = match.group(1) if match else path.stem
            records.append(
                _record(path, corpus, f"multi-pressure-{pressure.lower()}", "multi_pressure", f"mp:{key}")
            )
    return records


def musan_records(corpus: Path) -> list[dict[str, Any]]:
    root = corpus / "musan" / "noise" / "free-sound"
    annotation = root / "ANNOTATIONS"
    names = [line.strip() for line in annotation.read_text().splitlines()[1:] if line.strip()]
    candidates = []
    for name in names:
        path = root / f"{name}.wav"
        record = _record(path, corpus, "musan-annotated-background", "musan_curated", f"musan:{name}")
        if record["accepted"]:
            value = record["metrics"]
            failures = []
            if value["duration_seconds"] < 20:
                failures.append("duration_below_20s")
            if value["active_fraction"] < 0.9:
                failures.append("activity_below_90pct")
            if value["frame_rms_spread_db"] > 12:
                failures.append("rms_spread_above_12db")
            if value["hard_clipped_fraction"] > 0.0001:
                failures.append("excessive_clipping")
            if abs(value["dc_offset_fraction"]) > 0.01:
                failures.append("large_dc_offset")
            if failures:
                record["accepted"] = False
                record["exclusion_reasons"].extend(failures)
        candidates.append(record)
    return _apply_musan_cap(candidates)


def _apply_musan_cap(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passing = sorted(
        (row for row in candidates if row["accepted"]),
        key=lambda row: stable_score(row["identity"], "musan-selection"),
    )
    selected = {row["identity"] for row in passing[:8]}
    for record in candidates:
        if record["accepted"] and record["identity"] not in selected:
            record["accepted"] = False
            record["exclusion_reasons"].append("musan_eight_file_cap")
    return candidates


def stratify_splits(records: list[dict[str, Any]]) -> None:
    """Apply a deterministic 90/10 identity split within every accepted category."""
    by_category: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record["accepted"]:
            stratum = (
                record["source"]
                if record["category"] == "mka"
                else record["category"]
            )
            by_category[stratum].add(record["identity"])
    evaluation: dict[str, set[str]] = {}
    for category, identities in by_category.items():
        ordered = sorted(identities, key=lambda identity: stable_score(identity, f"split-{category}"))
        count = max(1, round(len(ordered) * 0.1))
        evaluation[category] = set(ordered[:count])
    for record in records:
        if record["accepted"]:
            stratum = (
                record["source"]
                if record["category"] == "mka"
                else record["category"]
            )
            record["split"] = (
                "eval" if record["identity"] in evaluation[stratum] else "train"
            )


def validate_splittable_categories(records: list[dict[str, Any]]) -> None:
    """Reject weighted categories that cannot contribute to both splits."""
    weighted = BACKGROUND_WEIGHTS.keys() | FOREGROUND_WEIGHTS.keys()
    identities: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record["accepted"] and record["category"] in weighted:
            identities[record["category"]].add(record["identity"])
    invalid = sorted(
        category for category in weighted if len(identities[category]) < 2
    )
    if invalid:
        raise ValueError(
            "weighted categories require at least two accepted identities: "
            + ", ".join(invalid)
        )
    missing_mka_sources = sorted(
        source
        for source in MKA_SOURCES
        if not any(
            record["accepted"] and record.get("source") == source
            for record in records
        )
    )
    if missing_mka_sources:
        raise ValueError(
            "MKA sources require at least one accepted identity: "
            + ", ".join(missing_mka_sources)
        )


def validate_render_splits(records: list[dict[str, Any]]) -> None:
    """Require every renderer input group before any output is written."""
    required_categories = BACKGROUND_WEIGHTS.keys() | FOREGROUND_WEIGHTS.keys()
    missing = []
    for split in ("train", "eval"):
        for category in required_categories:
            if not any(
                row["accepted"]
                and row["split"] == split
                and row["category"] == category
                for row in records
            ):
                missing.append(f"{split}:{category}")
        for pressure in PRESSURES:
            source = f"multi-pressure-{pressure.lower()}"
            if not any(
                row["accepted"]
                and row["split"] == split
                and row["source"] == source
                for row in records
            ):
                missing.append(f"{split}:{source}")
    if missing:
        raise ValueError("missing render groups: " + ", ".join(missing))


def audit(corpus: Path, output: Path) -> dict[str, Any]:
    records = dns_records(corpus) + mka_records(corpus) + multi_pressure_records(corpus) + musan_records(corpus)
    validate_splittable_categories(records)
    stratify_splits(records)
    validate_render_splits(records)
    accepted = [row for row in records if row["accepted"]]
    identities: dict[str, set[str]] = defaultdict(set)
    for row in accepted:
        identities[row["split"]].add(row["identity"])
    overlap = identities["train"] & identities["eval"]
    if overlap:
        raise RuntimeError(f"train/eval identity overlap: {sorted(overlap)[:3]}")
    summary = {
        "candidate_count": len(records),
        "accepted_count": len(accepted),
        "accepted_by_category": dict(sorted(Counter(row["category"] for row in accepted).items())),
        "excluded_by_reason": dict(sorted(Counter(reason for row in records for reason in row["exclusion_reasons"]).items())),
        "accepted_by_split": dict(sorted(Counter(row["split"] for row in accepted).items())),
    }
    result = {
        "format_version": 1,
        "kind": "rnnoise-mixed-noise-audit",
        "seed": SEED,
        "corpus_root": str(corpus.resolve()),
        "split_policy": "sha256-identity-90-10",
        "background_weights": BACKGROUND_WEIGHTS,
        "foreground_weights": FOREGROUND_WEIGHTS,
        "summary": summary,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for split in ("train", "eval"):
        manifest = output.with_name(f"{output.stem}-{split}.jsonl")
        rows = (row for row in accepted if row["split"] == split)
        manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return result


def _cycle_chunks(
    records: list[dict[str, Any]], sample_count: int
) -> Iterable[tuple[np.ndarray, str | None]]:
    if not records:
        raise ValueError("cannot render an empty category")
    written = 0
    index = 0
    while written < sample_count:
        record = records[index % len(records)]
        audio = decode_48k(Path(record["path"]))
        if not len(audio):
            raise ValueError(f"decoded empty audio: {record['path']}")
        take = min(len(audio), sample_count - written)
        if take:
            yield audio[:take], record["relative_path"]
        written += take
        index += 1


def _multi_pressure_chunks(
    records: list[dict[str, Any]], sample_count: int, split: str
) -> Iterable[tuple[np.ndarray, str | None]]:
    by_pressure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        pressure = row["source"].removeprefix("multi-pressure-")
        by_pressure[pressure].append(row)
    for pressure_records in by_pressure.values():
        pressure_records.sort(
            key=lambda row: stable_score(row["identity"], f"mp-{split}")
        )
    rng = random.Random(stable_score(split, "multi-pressure-silence"))
    written = 0
    indices = Counter()
    order = ("highp", "mediump", "lowp")
    while written < sample_count:
        pressure = order[sum(indices.values()) % len(order)]
        pressure_records = by_pressure[pressure]
        record = pressure_records[indices[pressure] % len(pressure_records)]
        indices[pressure] += 1
        audio = decode_48k(Path(record["path"]))
        if not len(audio):
            raise ValueError(f"decoded empty audio: {record['path']}")
        silence = np.zeros(round(RATE * rng.uniform(0.2, 1.2)), dtype="<i2")
        for piece, source in ((audio, record["relative_path"]), (silence, None)):
            take = min(len(piece), sample_count - written)
            if take:
                yield piece[:take], source
            written += take
            if written == sample_count:
                break


def _allocate(total: int, weights: dict[str, int]) -> dict[str, int]:
    items = list(weights.items())
    allocated = {name: total * weight // sum(weights.values()) for name, weight in items}
    allocated[items[-1][0]] += total - sum(allocated.values())
    return allocated


def _write_pcm(
    path: Path, chunks: Iterable[tuple[np.ndarray, str | None]]
) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    used_sources: list[str] = []
    seen_sources: set[str] = set()
    with path.open("wb") as destination:
        for chunk, source in chunks:
            destination.write(np.asarray(chunk, dtype="<i2").tobytes())
            if source is not None and source not in seen_sources:
                seen_sources.add(source)
                used_sources.append(source)
    return used_sources


def render(audit_path: Path, output: Path, train_hours: float, eval_hours: float) -> dict[str, Any]:
    for split, hours in (("train", train_hours), ("eval", eval_hours)):
        if round(hours * 3600 * RATE) < SEQUENCE_SAMPLES:
            raise ValueError(
                f"{split} render must contain at least one 20-second feature sequence"
            )
    report = json.loads(audit_path.read_text())
    accepted = [row for row in report["records"] if row["accepted"]]
    validate_render_splits(accepted)
    manifests: dict[str, Any] = {}
    for split, hours in (("train", train_hours), ("eval", eval_hours)):
        rows = [row for row in accepted if row["split"] == split]
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_category[row["category"]].append(row)
        for category in by_category:
            by_category[category].sort(key=lambda row: stable_score(row["identity"], f"render-{split}-{category}"))
        total_samples = round(hours * 3600 * RATE)
        split_manifest: dict[str, Any] = {"hours": hours, "files": {}}
        for kind, weights in (("background", BACKGROUND_WEIGHTS), ("foreground", FOREGROUND_WEIGHTS)):
            allocations = _allocate(total_samples, weights)

            def chunks():
                for category, count in allocations.items():
                    category_rows = by_category[category]
                    if category == "multi_pressure":
                        yield from _multi_pressure_chunks(category_rows, count, split)
                    else:
                        yield from _cycle_chunks(category_rows, count)
            target = output / f"{split}_{kind}.pcm"
            used_records = _write_pcm(target, chunks())
            split_manifest["files"][kind] = {
                "path": str(target.resolve()),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "allocations_samples": allocations,
                "source_files": used_records,
            }
        manifests[split] = split_manifest
    result = {
        "format_version": 1,
        "kind": "rnnoise-mixed-noise-render",
        "seed": SEED,
        "audit": {"path": str(audit_path.resolve()), "sha256": sha256_file(audit_path)},
        "decode_format": "pcm-s16le-48k-mono",
        "splits": manifests,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "noise-mix-manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--corpus", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    render_parser = commands.add_parser("render")
    render_parser.add_argument("--audit", type=Path, required=True)
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.add_argument("--train-hours", type=float, required=True)
    render_parser.add_argument("--eval-hours", type=float, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        audit(args.corpus.resolve(), args.output.resolve())
    else:
        if args.train_hours <= 0 or args.eval_hours <= 0:
            parser.error("render hours must be positive")
        render(args.audit.resolve(), args.output.resolve(), args.train_hours, args.eval_hours)


if __name__ == "__main__":
    main()
