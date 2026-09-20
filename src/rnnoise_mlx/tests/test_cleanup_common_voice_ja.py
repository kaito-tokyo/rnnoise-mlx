from pathlib import Path

import tempfile
import unittest
from unittest.mock import patch
import importlib.util

if importlib.util.find_spec("fcntl") is None:
    raise unittest.SkipTest("fcntl is not available on this platform")

from rnnoise_mlx.tools.cleanup_common_voice_ja import (
    cleanup_contract,
    cleanup_one,
    denoise_pcm,
    require_internal_output,
    sha256,
    validate_input_digests,
    validate_output_path,
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


class PatchHelper:
    def __init__(self):
        self._patches = []
    def setattr(self, target, name, value=None):
        if isinstance(target, str) and value is None:
            patcher = patch(target, name)
        else:
            patcher = patch.object(target, name, value)
        self._patches.append(patcher); patcher.start()
    def setenv(self, name, value):
        patcher = patch.dict("os.environ", {name: value})
        self._patches.append(patcher); patcher.start()
    def close(self):
        for patcher in reversed(self._patches): patcher.stop()

class CleanupCommonVoiceTests(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temporary_directory.name)
        self._patcher = PatchHelper()

    def tearDown(self):
        self._patcher.close(); self._temporary_directory.cleanup()

    def test_validate_output_path_rejects_symlinked_parent(self):
        output_root = self.tmp_path / "output"
        output_root.mkdir()
        external = self.tmp_path / "external"
        external.mkdir()
        (output_root / "speaker").symlink_to(external, target_is_directory=True)

        with self.assertRaisesRegex(ValueError, "parent must not be a symlink"):
            validate_output_path(output_root, Path("speaker/clip.wav"))


    def test_denoise_pcm_pads_final_frame_and_restores_length(self):
        processor = FakePreprocessor()
        result = denoise_pcm(b"\x00\x01\x02\x03\x04\x05", processor, frame_size=2)
        assert result == b"\xff\xfe\xfd\xfc\xfb\xfa"
        assert len(processor.frames) == 2
        assert processor.frames[-1] == b"\x04\x05\x00\x00"


    def test_cleanup_one_denoises_before_trimming_and_resumes(self):
        source_root = self.tmp_path / "input"
        output_root = self.tmp_path / "output"
        source = source_root / "train" / "clip.mp3"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"source")
        processors = []
        encoded = []

        def factory():
            processor = FakePreprocessor()
            processors.append(processor)
            return processor

        self._patcher.setattr(
            "rnnoise_mlx.tools.cleanup_common_voice_ja.decode_pcm",
            lambda path, rate: bytes(range(12)),
        )

        def fake_encode(pcm, output, sample_rate):
            encoded.append((pcm, sample_rate))
            output.write_bytes(b"wave")

        self._patcher.setattr("rnnoise_mlx.tools.cleanup_common_voice_ja.encode_wav", fake_encode)
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


    def test_output_below_unregistered_volume_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "registered"):
            require_internal_output(Path("/Volumes/doc-2026-05-26/output"))


    def test_cleanup_rejects_dangling_output_symlink(self):
        output = self.tmp_path / "output"
        output.symlink_to(self.tmp_path / "missing")

        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            require_internal_output(output)


    def test_output_below_registered_training_volume_is_allowed(self):
        self._patcher.setenv("RNNOISE_MLX_STORAGE_ROOT", "/Volumes/rnnoise-mlx-train")
        self._patcher.setattr(
            "rnnoise_mlx.tools.portable_storage.load_volume_config", lambda root: {}
        )

        result = require_internal_output(Path("/Volumes/rnnoise-mlx-train/datasets/prepared"))

        assert result == Path("/Volumes/rnnoise-mlx-train/datasets/prepared")


    def test_cleanup_one_rejects_path_traversal(self):
        with self.assertRaisesRegex(ValueError, "escapes the corpus root"):
            cleanup_one(
                self.tmp_path / "input", self.tmp_path / "output",
                {"path": "../clip.mp3", "input_sha256": "unused", "onsets_seconds": {"-40": 0.25}},
                lambda: FakePreprocessor(), -40, 7_200, 48_000, 960,
            )


    def test_validate_records_rejects_duplicate_wav_outputs_and_missing_onsets(self):
        duplicate = [
            {"path": "train/clip.mp3", "onsets_seconds": {"-40": 0.1}},
            {"path": "train/clip.flac", "onsets_seconds": {"-40": 0.2}},
        ]
        with self.assertRaisesRegex(ValueError, "same output"):
            validate_records(duplicate, -40)

        with self.assertRaisesRegex(ValueError, "same output"):
            validate_records(
                [
                    {"path": "train/Clip.mp3", "onsets_seconds": {"-40": 0.1}},
                    {"path": "train/clip.flac", "onsets_seconds": {"-40": 0.2}},
                ],
                -40,
            )

        with self.assertRaisesRegex(ValueError, "no onset"):
            validate_records([{"path": "train/clip.mp3", "onsets_seconds": {}}], -40)


    def test_validate_input_digests_rejects_changed_source_before_workers(self):
        source = self.tmp_path / "clip.mp3"
        source.write_bytes(b"changed")

        with self.assertRaisesRegex(ValueError, "checksums differ"):
            validate_input_digests(self.tmp_path, [{"path": "clip.mp3", "input_sha256": "old"}])


    def test_resume_rejects_changed_input_or_incomplete_output(self):
        source_root = self.tmp_path / "input"
        source = source_root / "clip.mp3"
        source.parent.mkdir()
        source.write_bytes(b"source")
        output_root = self.tmp_path / "output"
        output_root.mkdir()
        filter_manifest = self.tmp_path / "filter.json"
        filter_manifest.write_text("{}")
        library = self.tmp_path / "libspeexdsp.dylib"
        library.write_bytes(b"library")
        ffmpeg = self.tmp_path / "ffmpeg"
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
        with self.assertRaisesRegex(ValueError, "verification failed"):
            validate_resume(output_root, contract, source_root, [record])


    def test_resume_rejects_symlinked_output(self):
        source_root = self.tmp_path / "input"
        source = source_root / "clip.mp3"
        source.parent.mkdir()
        source.write_bytes(b"source")
        output_root = self.tmp_path / "output"
        output_root.mkdir()
        external = self.tmp_path / "external.wav"
        external.write_bytes(b"wave")
        (output_root / "clip.wav").symlink_to(external)
        filter_manifest = self.tmp_path / "filter.json"
        filter_manifest.write_text("{}")
        library = self.tmp_path / "libspeexdsp.dylib"
        library.write_bytes(b"library")
        ffmpeg = self.tmp_path / "ffmpeg"
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

        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            validate_resume(output_root, contract, source_root, [{"path": "clip.mp3"}])


    def test_resume_accepts_verified_progress_from_an_interrupted_cleanup(self):
        source = self.tmp_path / "input" / "clip.mp3"
        source.parent.mkdir()
        source.write_bytes(b"source")
        output_root = self.tmp_path / "output"
        output_root.mkdir()
        output = output_root / "clip.wav"
        output.write_bytes(b"wave")
        filter_manifest = self.tmp_path / "filter.json"
        filter_manifest.write_text("{}")
        library = self.tmp_path / "libspeexdsp.dylib"
        library.write_bytes(b"library")
        ffmpeg = self.tmp_path / "ffmpeg"
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

