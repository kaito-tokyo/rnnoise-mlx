import wave
from pathlib import Path

import numpy as np

from rnnoise_mlx.tools.prepare_noise_mix import (
    PRESSURES,
    _allocate,
    _cycle_chunks,
    _multi_pressure_chunks,
    metrics,
    split_for,
    stratify_splits,
    validate_splittable_categories,
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


def test_metrics_measure_stationary_pcm(tmp_path):
    samples = np.full(48_000, 1000, dtype="<i2")
    path = tmp_path / "steady.wav"
    write_wav(path, samples)
    result = metrics(path)
    assert result["duration_seconds"] == 1
    assert result["active_fraction"] == 1
    assert result["frame_rms_spread_db"] == 0
    assert result["hard_clipped_fraction"] == 0


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
        }
        for index in range(3)
    ]
    monkeypatch.setattr(
        "rnnoise_mlx.tools.prepare_noise_mix.decode_48k",
        lambda path: np.full(10, int(path.stem), dtype="<i2"),
    )
    rendered = list(_cycle_chunks(records, 5))
    assert [source for _, source in rendered] == ["0.wav"]
