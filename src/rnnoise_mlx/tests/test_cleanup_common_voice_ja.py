from pathlib import Path

import pytest

from rnnoise_mlx.tools.cleanup_common_voice_ja import (
    cleanup_contract,
    cleanup_one,
    denoise_pcm,
    require_internal_output,
    sha256,
    validate_input_digests,
    validate_records,
    validate_resume,
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
    record = {"path": "train/clip.mp3", "input_sha256": sha256(source), "onsets_seconds": {"-40": 0.004}}
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


def test_cleanup_rejects_dangling_output_symlink(tmp_path: Path):
    output = tmp_path / "output"
    output.symlink_to(tmp_path / "missing")

    with pytest.raises(ValueError, match="must not be a symlink"):
        require_internal_output(output)


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
            {"path": "../clip.mp3", "input_sha256": "unused", "onsets_seconds": {"-40": 0.25}},
            lambda: FakePreprocessor(), -40, 7_200, 48_000, 960,
        )


def test_validate_records_rejects_duplicate_wav_outputs_and_missing_onsets():
    duplicate = [
        {"path": "train/clip.mp3", "onsets_seconds": {"-40": 0.1}},
        {"path": "train/clip.flac", "onsets_seconds": {"-40": 0.2}},
    ]
    with pytest.raises(ValueError, match="same output"):
        validate_records(duplicate, -40)

    with pytest.raises(ValueError, match="same output"):
        validate_records(
            [
                {"path": "train/Clip.mp3", "onsets_seconds": {"-40": 0.1}},
                {"path": "train/clip.flac", "onsets_seconds": {"-40": 0.2}},
            ],
            -40,
        )

    with pytest.raises(ValueError, match="no onset"):
        validate_records([{"path": "train/clip.mp3", "onsets_seconds": {}}], -40)


def test_validate_input_digests_rejects_changed_source_before_workers(tmp_path: Path):
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"changed")

    with pytest.raises(ValueError, match="checksums differ"):
        validate_input_digests(tmp_path, [{"path": "clip.mp3", "input_sha256": "old"}])


def test_resume_rejects_changed_input_or_incomplete_output(tmp_path: Path):
    source_root = tmp_path / "input"
    source = source_root / "clip.mp3"
    source.parent.mkdir()
    source.write_bytes(b"source")
    output_root = tmp_path / "output"
    output_root.mkdir()
    filter_manifest = tmp_path / "filter.json"
    filter_manifest.write_text("{}")
    library = tmp_path / "libspeexdsp.dylib"
    library.write_bytes(b"library")
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_bytes(b"ffmpeg")
    record = {"path": "clip.mp3", "onsets_seconds": {"-40": 0}}
    contract = cleanup_contract(
        source_root, filter_manifest, library, ffmpeg, sample_rate=48_000, frame_size=960,
        frame_ms=20, noise_suppress_db=-12, threshold=-40, margin_samples=7200,
    )
    output = output_root / "clip.wav"
    output.write_bytes(b"wave")
    (output_root / "cleanup-manifest.json").write_text(__import__("json").dumps({
        **contract,
        "files": [{
            "input": "clip.mp3", "input_sha256": sha256(source),
            "output": "clip.wav", "output_sha256": sha256(output),
        }],
    }))

    validate_resume(output_root, contract, source_root, [record])
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="verification failed"):
        validate_resume(output_root, contract, source_root, [record])


def test_resume_rejects_symlinked_output(tmp_path: Path):
    source_root = tmp_path / "input"
    source = source_root / "clip.mp3"
    source.parent.mkdir()
    source.write_bytes(b"source")
    output_root = tmp_path / "output"
    output_root.mkdir()
    external = tmp_path / "external.wav"
    external.write_bytes(b"wave")
    (output_root / "clip.wav").symlink_to(external)
    filter_manifest = tmp_path / "filter.json"
    filter_manifest.write_text("{}")
    library = tmp_path / "libspeexdsp.dylib"
    library.write_bytes(b"library")
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_bytes(b"ffmpeg")
    contract = cleanup_contract(
        source_root, filter_manifest, library, ffmpeg, sample_rate=48_000, frame_size=960,
        frame_ms=20, noise_suppress_db=-12, threshold=-40, margin_samples=7200,
    )
    (output_root / "cleanup-manifest.json").write_text(__import__("json").dumps({
        **contract,
        "files": [{
            "input": "clip.mp3", "input_sha256": sha256(source),
            "output": "clip.wav", "output_sha256": sha256(external),
        }],
    }))

    with pytest.raises(ValueError, match="verification failed"):
        validate_resume(output_root, contract, source_root, [{"path": "clip.mp3"}])


def test_resume_accepts_verified_progress_from_an_interrupted_cleanup(tmp_path: Path):
    source = tmp_path / "input" / "clip.mp3"
    source.parent.mkdir()
    source.write_bytes(b"source")
    output_root = tmp_path / "output"
    output_root.mkdir()
    output = output_root / "clip.wav"
    output.write_bytes(b"wave")
    filter_manifest = tmp_path / "filter.json"
    filter_manifest.write_text("{}")
    library = tmp_path / "libspeexdsp.dylib"
    library.write_bytes(b"library")
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_bytes(b"ffmpeg")
    contract = cleanup_contract(
        source.parent, filter_manifest, library, ffmpeg, sample_rate=48_000, frame_size=960,
        frame_ms=20, noise_suppress_db=-12, threshold=-40, margin_samples=7200,
    )
    record = {"path": "clip.mp3", "onsets_seconds": {"-40": 0}}
    (output_root / "cleanup-progress.json").write_text(__import__("json").dumps({
        **contract,
        "files": [{
            "input": "clip.mp3", "input_sha256": sha256(source),
            "output": "clip.wav", "output_sha256": sha256(output),
        }],
    }))

    assert validate_resume(output_root, contract, source.parent, [record]) == {"clip.mp3"}
