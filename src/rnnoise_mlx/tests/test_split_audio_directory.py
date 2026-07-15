from pathlib import Path

import pytest

from rnnoise_mlx.tools.split_audio_directory import assignment, split


def test_assignment_is_deterministic_and_validates_fraction():
    assert assignment("a.flac", 0.1, 141) == assignment("a.flac", 0.1, 141)
    with pytest.raises(ValueError):
        assignment("a.flac", 0, 141)


def test_split_creates_symlinks_and_manifest(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(20):
        (source / f"{index}.flac").write_bytes(b"audio")
    output = tmp_path / "split"
    output.mkdir()
    manifest = split(source, output, 0.5, 141)
    assert sum(manifest["counts"].values()) == 20
    assert manifest["counts"]["train"] > 0
    assert manifest["counts"]["eval"] > 0
    assert len(list(output.rglob("*.flac"))) == 20
    assert all(path.is_symlink() for path in output.rglob("*.flac"))
    assert (output / "split-manifest.json").is_file()
