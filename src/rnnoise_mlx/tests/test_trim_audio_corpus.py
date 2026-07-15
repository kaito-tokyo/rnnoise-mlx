from pathlib import Path

from rnnoise_mlx.tools.trim_audio_corpus import trim_one


def test_trim_one_uses_onset_minus_margin_and_is_resumable(tmp_path: Path, monkeypatch):
    source_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source = source_root / "train/clip.wav"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"input")
    calls = []

    def fake_run(command, check):
        calls.append(command)
        Path(command[-1]).write_bytes(b"output")

    monkeypatch.setattr("rnnoise_mlx.tools.trim_audio_corpus.subprocess.run", fake_run)
    record = {"path": "train/clip.mp3", "onsets_seconds": {"-40": .25}}
    first = trim_one(source_root, output_root, record, -40, 7_200, 48_000)
    second = trim_one(source_root, output_root, record, -40, 7_200, 48_000)
    assert first == second
    assert first["trim_samples"] == 4_800
    assert any(value.startswith("atrim=start_sample=4800") for value in calls[0])
    assert len(calls) == 1
