from pathlib import Path

from rnnoise_mlx.tools.clean_audio_corpus import clean_one


def test_clean_one_is_atomic_and_resumable(tmp_path: Path, monkeypatch):
    executable = tmp_path / "denoiser"
    executable.write_text("binary")
    model = tmp_path / "model.mlmodelc"
    model.mkdir()
    source_root = tmp_path / "input"
    source_root.mkdir()
    source = source_root / "speaker" / "clip.mp3"
    source.parent.mkdir()
    source.write_bytes(b"input")
    output_root = tmp_path / "output"
    calls = []

    def fake_run(command, check):
        calls.append(command)
        Path(command[-1]).write_bytes(b"wave")

    monkeypatch.setattr("rnnoise_mlx.tools.clean_audio_corpus.subprocess.run", fake_run)
    first = clean_one(executable, model, source_root, output_root, source)
    second = clean_one(executable, model, source_root, output_root, source)
    assert first == second
    assert len(calls) == 1
    assert calls[0][-1].endswith(".partial.wav")
    assert (output_root / "speaker/clip.wav").read_bytes() == b"wave"
    assert not list(output_root.rglob("*.partial"))
