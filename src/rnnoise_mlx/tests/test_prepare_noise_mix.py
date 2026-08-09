import hashlib
import wave
from pathlib import Path

import numpy as np

from rnnoise_mlx.tools.prepare_noise_mix import (
    PRESSURES,
    _apply_musan_cap,
    _allocate,
    _cycle_chunks,
    _multi_pressure_chunks,
    metrics,
    render,
    split_for,
    stratify_splits,
    validate_splittable_categories,
    validate_render_splits,
)


def write_wav(path: Path, samples: np.ndarray, rate: int = 48_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def test_split_is_deterministic_and_identity_based():
    assert split_for("dns:123") == split_for("dns:123")
    assert {split_for(f"identity:{index}") for index in range(100)} == {"train", "eval"}


def test_allocate_is_exact():
    allocation = _allocate(101, {"first": 90, "second": 10})
    assert allocation == {"first": 90, "second": 11}
    assert sum(allocation.values()) == 101


def test_stratify_splits_keeps_every_category_in_evaluation():
    records = [
        {"accepted": True, "category": "rare", "identity": "one", "split": "train"},
        *[
            {"accepted": True, "category": "common", "identity": f"id-{index}", "split": "train"}
            for index in range(20)
        ],
    ]
    stratify_splits(records)
    assert [row["split"] for row in records if row["category"] == "rare"] == ["eval"]
    assert sum(row["split"] == "eval" for row in records if row["category"] == "common") == 2


def test_stratify_splits_reserves_evaluation_identity_per_mka_source():
    records = [
        {
            "accepted": True,
            "category": "mka",
            "source": source,
            "identity": f"{source}-{index}",
            "split": "train",
        }
        for source in ("mka-lenovo", "mka-msi")
        for index in range(10)
    ]
    stratify_splits(records)
    assert {
        row["source"] for row in records if row["split"] == "eval"
    } == {"mka-lenovo", "mka-msi"}


def test_weighted_category_requires_two_identities():
    records = [
        {
            "accepted": True,
            "category": category,
            "identity": f"{category}-{index}",
        }
        for category in {
            "dns_typing", "dns_door", "dns_squeak", "dns_dragging",
            "dns_copy-machine", "dns_human", "mka", "multi_pressure",
            "dns_fan", "musan_curated",
        }
        for index in range(2)
    ]
    removed_category = records.pop()["category"]
    try:
        validate_splittable_categories(records)
    except ValueError as error:
        assert removed_category in str(error)
    else:
        raise AssertionError("unsplittable category was accepted")


def test_musan_cap_is_independent_of_provisional_split():
    records = [
        {
            "accepted": True,
            "identity": f"musan-{index}",
            "split": "eval",
            "exclusion_reasons": [],
        }
        for index in range(10)
    ]
    _apply_musan_cap(records)
    assert sum(row["accepted"] for row in records) == 8


def test_render_validation_requires_each_pressure_in_both_splits():
    records = []
    for split in ("train", "eval"):
        for category in {
            "dns_typing", "dns_door", "dns_squeak", "dns_dragging",
            "dns_copy-machine", "dns_human", "mka", "multi_pressure",
            "dns_fan", "musan_curated",
        }:
            records.append({
                "accepted": True,
                "split": split,
                "category": category,
                "source": "placeholder",
            })
        for pressure in PRESSURES:
            records.append({
                "accepted": True,
                "split": split,
                "category": "multi_pressure",
                "source": f"multi-pressure-{pressure.lower()}",
            })
    records = [
        row for row in records
        if not (
            row["split"] == "eval"
            and row["source"] == "multi-pressure-lowp"
        )
    ]
    try:
        validate_render_splits(records)
    except ValueError as error:
        assert "eval:multi-pressure-lowp" in str(error)
    else:
        raise AssertionError("missing pressure level was accepted")


def test_metrics_measure_stationary_pcm(tmp_path):
    samples = np.full(48_000, 1000, dtype="<i2")
    path = tmp_path / "steady.wav"
    write_wav(path, samples)
    result = metrics(path)
    assert result["duration_seconds"] == 1
    assert result["active_fraction"] == 1
    assert result["frame_rms_spread_db"] == 0
    assert result["hard_clipped_fraction"] == 0


def test_metrics_rejects_empty_pcm(tmp_path):
    path = tmp_path / "empty.wav"
    write_wav(path, np.empty(0, dtype="<i2"))
    try:
        metrics(path)
    except ValueError as error:
        assert "empty audio" in str(error)
    else:
        raise AssertionError("empty audio was accepted")


def test_multi_pressure_interleaves_pressure_and_silence(tmp_path, monkeypatch):
    records = []
    values = {"highp": 3000, "mediump": 2000, "lowp": 1000}
    for directory in PRESSURES:
        pressure = directory.lower()
        path = tmp_path / directory / "key.wav"
        write_wav(path, np.full(480, values[pressure], dtype="<i2"))
        records.append({
            "source": f"multi-pressure-{pressure}",
            "path": str(path),
            "relative_path": f"{directory}/key.wav",
            "identity": "key",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })

    monkeypatch.setattr(
        "rnnoise_mlx.tools.prepare_noise_mix.decode_48k",
        lambda path: np.frombuffer(wave.open(str(path), "rb").readframes(480), dtype="<i2"),
    )
    rendered = list(_multi_pressure_chunks(records, 200_000, "train"))
    stream = np.concatenate([chunk for chunk, _ in rendered])
    assert len(stream) == 200_000
    assert {1000, 2000, 3000}.issubset(set(np.unique(stream)))
    assert 0 in stream
    assert {source for _, source in rendered if source} == {
        f"{directory}/key.wav" for directory in PRESSURES
    }


def test_cycle_chunks_reports_only_consumed_sources(tmp_path, monkeypatch):
    records = [
        {
            "path": str(tmp_path / f"{index}.wav"),
            "relative_path": f"{index}.wav",
            "sha256": "unused",
        }
        for index in range(3)
    ]
    monkeypatch.setattr(
        "rnnoise_mlx.tools.prepare_noise_mix.decode_48k",
        lambda path: np.full(10, int(path.stem), dtype="<i2"),
    )
    monkeypatch.setattr(
        "rnnoise_mlx.tools.prepare_noise_mix.sha256_file",
        lambda path: "unused",
    )
    rendered = list(_cycle_chunks(records, 5))
    assert [source for _, source in rendered] == ["0.wav"]


def test_cycle_chunks_rejects_empty_decode(tmp_path, monkeypatch):
    record = {
        "path": str(tmp_path / "empty.wav"),
        "relative_path": "empty.wav",
        "sha256": "unused",
    }
    monkeypatch.setattr(
        "rnnoise_mlx.tools.prepare_noise_mix.decode_48k",
        lambda path: np.empty(0, dtype="<i2"),
    )
    monkeypatch.setattr(
        "rnnoise_mlx.tools.prepare_noise_mix.sha256_file",
        lambda path: "unused",
    )
    try:
        list(_cycle_chunks([record], 1))
    except ValueError as error:
        assert "decoded empty audio" in str(error)
    else:
        raise AssertionError("empty decode was accepted")


def test_cycle_chunks_rejects_source_changed_after_audit(tmp_path, monkeypatch):
    path = tmp_path / "source.wav"
    path.write_bytes(b"changed")
    record = {
        "path": str(path),
        "relative_path": "source.wav",
        "sha256": "old-digest",
    }
    monkeypatch.setattr(
        "rnnoise_mlx.tools.prepare_noise_mix.decode_48k",
        lambda path: np.ones(1, dtype="<i2"),
    )
    try:
        list(_cycle_chunks([record], 1))
    except ValueError as error:
        assert "checksum differs" in str(error)
    else:
        raise AssertionError("changed audited source was accepted")


def test_render_requires_one_complete_feature_sequence_per_split(tmp_path):
    try:
        render(tmp_path / "missing.json", tmp_path / "output", 1.0, 19 / 3600)
    except ValueError as error:
        assert "eval render" in str(error)
        assert "20-second" in str(error)
    else:
        raise AssertionError("short evaluation render was accepted")
