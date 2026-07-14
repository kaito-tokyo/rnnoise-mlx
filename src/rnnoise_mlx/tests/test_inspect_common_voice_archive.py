import io
import tarfile
from pathlib import Path

from rnnoise_mlx.tools.inspect_common_voice_archive import inspect


def add_text(archive: tarfile.TarFile, name: str, text: str) -> None:
    payload = text.encode()
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def test_inspect_summarizes_validated_metadata_without_sentences(tmp_path: Path):
    archive_path = tmp_path / "cv.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        add_text(
            archive,
            "cv/xx/clip_durations.tsv",
            "clip\tduration[ms]\nfirst.mp3\t1000\nsecond.mp3\t2500\nother.mp3\t9000\n",
        )
        add_text(
            archive,
            "cv/xx/validated.tsv",
            "client_id\tpath\tsentence\tage\tgender\taccents\tvariant\tlocale\n"
            "speaker-a\tfirst.mp3\tsecret one\ttwenties\tfemale\tnorth\tformal\txx\n"
            "speaker-a\tsecond.mp3\tsecret two\ttwenties\tfemale\tnorth\tformal\txx\n",
        )

    result = inspect(archive_path)

    assert result["validated_clip_count"] == 2
    assert result["validated_duration_seconds"] == 3.5
    assert result["validated_speaker_count"] == 1
    assert result["speaker_duration_seconds"] == {"speaker-a": 3.5}
    assert result["metadata_summaries"]["accents"] == [
        {"value": "north", "clip_count": 2, "duration_seconds": 3.5, "speaker_count": 1}
    ]
    assert "secret" not in str(result)


def test_inspect_records_missing_duration(tmp_path: Path):
    archive_path = tmp_path / "cv.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        add_text(archive, "cv/xx/clip_durations.tsv", "clip\tduration[ms]\n")
        add_text(
            archive,
            "cv/xx/validated.tsv",
            "client_id\tpath\tlocale\nspk\tmissing.mp3\txx\n",
        )

    result = inspect(archive_path)
    assert result["missing_duration_count"] == 1
    assert result["missing_duration_paths"] == ["missing.mp3"]


def test_inspect_accepts_large_transcript_field_without_recording_it(tmp_path: Path):
    archive_path = tmp_path / "cv.tar.gz"
    sentence = "x" * 140_000
    with tarfile.open(archive_path, "w:gz") as archive:
        add_text(archive, "cv/xx/clip_durations.tsv", "clip\tduration[ms]\na.mp3\t1000\n")
        add_text(
            archive,
            "cv/xx/validated.tsv",
            f"client_id\tpath\tsentence\tlocale\nspk\ta.mp3\t{sentence}\txx\n",
        )

    result = inspect(archive_path)
    assert result["validated_clip_count"] == 1
    assert sentence not in str(result)
