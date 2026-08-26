from pathlib import Path

import pytest

from rnnoise_mlx.tools.cleanup_common_voice_ja import (
    cleanup_one,
    denoise_pcm,
    require_internal_output,
)


class FakePreprocessor:
    def __init__(self):
        self.frames = []
        self.closed = False

    def process(self, samples: bytes) -> bytes:
        self.frames.append(samples)
        return bytes(value ^ 0xFF for value in samples)

    def close(self) -> None:
        self.closed = True


def test_denoise_pcm_pads_final_frame_and_restores_length():
    processor = FakePreprocessor()
    result = denoise_pcm(b"\x00\x01\x02\x03\x04\x05", processor, frame_size=2)
    assert result == b"\xff\xfe\xfd\xfc\xfb\xfa"
    assert len(processor.frames) == 2
    assert processor.frames[-1] == b"\x04\x05\x00\x00"


def test_cleanup_one_denoises_before_trimming_and_resumes(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source = source_root / "train" / "clip.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    processors = []
    encoded = []

    def factory():
        processor = FakePreprocessor()
        processors.append(processor)
        return processor

    monkeypatch.setattr(
        "rnnoise_mlx.tools.cleanup_common_voice_ja.decode_pcm",
        lambda path, rate: bytes(range(12)),
    )

    def fake_encode(pcm, output, sample_rate):
        encoded.append((pcm, sample_rate))
        output.write_bytes(b"wave")

    monkeypatch.setattr("rnnoise_mlx.tools.cleanup_common_voice_ja.encode_wav", fake_encode)
    record = {"path": "train/clip.mp3", "onsets_seconds": {"-40": 0.004}}
    first = cleanup_one(
        source_root, output_root, record, factory, -40, margin_samples=1,
        sample_rate=1000, frame_size=4,
    )
    second = cleanup_one(
        source_root, output_root, record, factory, -40, margin_samples=1,
        sample_rate=1000, frame_size=4,
    )

    assert first == second
    assert first["trim_samples"] == 3
    assert encoded == [(bytes(value ^ 0xFF for value in range(6, 12)), 1000)]
    assert len(processors) == 1
    assert processors[0].closed
    assert not list(output_root.rglob("*.partial.wav"))


def test_output_below_unregistered_volume_is_rejected():
    with pytest.raises(ValueError, match="registered"):
        require_internal_output(Path("/Volumes/doc-2026-05-26/output"))


def test_output_below_registered_training_volume_is_allowed(monkeypatch):
    monkeypatch.setenv("RNNOISE_MLX_STORAGE_ROOT", "/Volumes/rnnoise-mlx-train")
    monkeypatch.setattr(
        "rnnoise_mlx.tools.portable_storage.load_volume_config", lambda root: {}
    )

    result = require_internal_output(Path("/Volumes/rnnoise-mlx-train/datasets/prepared"))

    assert result == Path("/Volumes/rnnoise-mlx-train/datasets/prepared")


def test_cleanup_one_rejects_path_traversal(tmp_path: Path):
    with pytest.raises(ValueError, match="escapes the corpus root"):
        cleanup_one(
            tmp_path / "input", tmp_path / "output",
            {"path": "../clip.mp3", "onsets_seconds": {"-40": 0.25}},
            lambda: FakePreprocessor(), -40, 7_200, 48_000, 960,
        )
