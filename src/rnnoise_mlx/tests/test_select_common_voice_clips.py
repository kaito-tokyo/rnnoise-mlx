import io
import tarfile
from pathlib import Path

import pytest

from rnnoise_mlx.tools.select_common_voice_clips import select


def add_text(archive: tarfile.TarFile, name: str, text: str) -> None:
    payload = text.encode()
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def fixture_archive(path: Path) -> None:
    with tarfile.open(path, "w:gz") as archive:
        add_text(
            archive,
            "cv/xx/clip_durations.tsv",
            "clip\tduration[ms]\n" + "".join(f"{speaker}-{clip}.mp3\t1000\n" for speaker in range(4) for clip in range(3)),
        )
        add_text(
            archive,
            "cv/xx/validated.tsv",
            "client_id\tpath\tsentence\tlocale\n"
            + "".join(f"s{speaker}\t{speaker}-{clip}.mp3\tsecret\txx\n" for speaker in range(4) for clip in range(3)),
        )


def test_select_is_deterministic_and_speaker_disjoint(tmp_path: Path):
    archive = tmp_path / "cv.tar.gz"
    fixture_archive(archive)
    first = select(archive, train_target_seconds=4, eval_target_seconds=2, speaker_cap_seconds=2, seed=7)
    second = select(archive, train_target_seconds=4, eval_target_seconds=2, speaker_cap_seconds=2, seed=7)
    assert first == second
    assert first["speaker_disjoint"] is True
    assert first["selected_seconds"] == {"train": 4.0, "eval": 2.0}
    assert first["selected_speaker_counts"]["train"] == 3
    assert set(row["speaker_id"] for row in first["records"] if row["split"] == "train").isdisjoint(
        row["speaker_id"] for row in first["records"] if row["split"] == "eval"
    )
    assert "secret" not in str(first)


def test_select_rejects_insufficient_capacity(tmp_path: Path):
    archive = tmp_path / "cv.tar.gz"
    fixture_archive(archive)
    with pytest.raises(ValueError, match="train capacity is insufficient"):
        select(archive, train_target_seconds=20, eval_target_seconds=2, speaker_cap_seconds=2, seed=7)
