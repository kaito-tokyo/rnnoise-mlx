import json
from pathlib import Path

from rnnoise_mlx.tools.audit_extracted_audio_quality import audit


def test_audit_aggregates_injected_clip_results(tmp_path: Path):
    root = tmp_path / "selected"
    root.mkdir()
    (root / "extraction-manifest.json").write_text(
        json.dumps(
            {
                "selection_manifest_sha256": "a" * 64,
                "records": [
                    {"path": "train/a.mp3", "split": "train", "speaker_id": "one"},
                    {"path": "eval/b.mp3", "split": "eval", "speaker_id": "two"},
                ],
            }
        )
    )

    def inspector(_root, record):
        metrics = {
            "duration_seconds": 1.0,
            "edge_trimmed_duration_seconds": 0.8,
            "peak_dbfs": -3.0,
            "rms_dbfs": -20.0,
            "hard_clipped_fraction": 0.0,
            "near_clipped_fraction": 0.0,
            "leading_low_rms_seconds": 0.1,
            "trailing_low_rms_seconds": 0.1,
        }
        return {
            **record,
            "source_codec": "mp3",
            "source_sample_rate_hz": 48000,
            "source_channels": 1,
            **metrics,
            "diagnostic_flags": [],
        }

    output = tmp_path / "audit.json"
    result = audit(root, output, workers=2, inspector=inspector)
    assert result["split_summaries"]["train"]["decoded_clip_count"] == 1
    assert result["split_summaries"]["eval"]["edge_trimmed_duration_seconds"] == 0.8
    assert json.loads(output.read_text())["decode_format"] == "48000 Hz mono signed 16-bit PCM"
