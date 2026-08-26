import io
from pathlib import Path

from rnnoise_mlx.tools.prepare_speech_mix import (
    exact_targets,
    resolve_config_path,
    render_split,
    stable_audio_paths,
    write_audio_directory,
    write_pcm_prefix,
)


def test_resolve_config_path_expands_environment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RNNOISE_MLX_STORAGE_ROOT", str(tmp_path))
    assert resolve_config_path("${RNNOISE_MLX_STORAGE_ROOT}/data") == tmp_path / "data"


def test_resolve_config_path_rejects_unset_variable(monkeypatch):
    monkeypatch.delenv("RNNOISE_MLX_MISSING", raising=False)
    import pytest

    with pytest.raises(ValueError, match="unset variable"):
        resolve_config_path("${RNNOISE_MLX_MISSING}/data")


def test_exact_targets_sum_exactly_and_are_order_independent():
    sources = [
        {"name": "jpn", "weight": 20},
        {"name": "cmn", "weight": 40},
        {"name": "kor", "weight": 40},
    ]
    forward = exact_targets(101, sources)
    reverse = exact_targets(101, list(reversed(sources)))
    assert sum(forward.values()) == 101
    assert forward == reverse
    assert forward == {"jpn": 20, "cmn": 41, "kor": 40}


def test_stable_audio_paths_is_repeatable(tmp_path: Path):
    (tmp_path / "b.wav").write_bytes(b"b")
    (tmp_path / "a.flac").write_bytes(b"a")
    (tmp_path / "ignored.txt").write_text("ignored")
    first = stable_audio_paths(tmp_path, "train:jpn")
    assert first == stable_audio_paths(tmp_path, "train:jpn")
    assert {path.name for path in first} == {"a.flac", "b.wav"}


def test_write_pcm_prefix_takes_exact_samples(tmp_path: Path):
    source = tmp_path / "source.pcm"
    source.write_bytes(bytes(range(20)))
    output = io.BytesIO()
    samples, used = write_pcm_prefix(output, source, 4)
    assert samples == 4
    assert output.getvalue() == bytes(range(8))
    assert used == [str(source.resolve())]


def test_render_split_uses_split_specific_weights(tmp_path: Path):
    first = tmp_path / "first.pcm"
    second = tmp_path / "second.pcm"
    first.write_bytes(b"\x01\x00" * 10)
    second.write_bytes(b"\x02\x00" * 10)
    specification = {
        "train_hours": 4 / 48_000 / 3600,
        "eval_hours": 4 / 48_000 / 3600,
        "sources": [
            {
                "name": "first",
                "train": str(first),
                "eval": str(first),
                "train_weight": 1,
                "eval_weight": 0,
                "type": "pcm-s16le-48k-mono",
            },
            {
                "name": "second",
                "train": str(second),
                "eval": str(second),
                "train_weight": 1,
                "eval_weight": 1,
                "type": "pcm-s16le-48k-mono",
            },
        ],
    }
    train = tmp_path / "train.pcm"
    evaluation = tmp_path / "eval.pcm"
    render_split(specification, "train", train)
    render_split(specification, "eval", evaluation)
    assert train.read_bytes() == b"\x01\x00" * 2 + b"\x02\x00" * 2
    assert evaluation.read_bytes() == b"\x02\x00" * 4


def test_write_audio_directory_repeats_only_when_enabled(tmp_path: Path, monkeypatch):
    source = tmp_path / "audio"
    source.mkdir()
    clip = source / "clip.wav"
    clip.write_bytes(b"placeholder")
    monkeypatch.setattr("rnnoise_mlx.tools.prepare_speech_mix.decode",
                        lambda path: b"\x01\x00" * 3)
    output = io.BytesIO()
    written, used, reuse = write_audio_directory(
        output, source, 8, "train:jpn", repeat_to_target=True
    )
    assert written == 8
    assert output.getvalue() == b"\x01\x00" * 8
    assert used == [str(clip.resolve())] * 3
    assert reuse["passes"] == 3
    assert reuse["file_use_counts"] == {str(clip.resolve()): 3}
    assert reuse["final_file_samples"] == 2
