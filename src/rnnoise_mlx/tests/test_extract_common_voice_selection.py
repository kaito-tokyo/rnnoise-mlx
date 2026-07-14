import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from rnnoise_mlx.tools.extract_common_voice_selection import extract


def test_extract_writes_only_selected_clips_and_digests(tmp_path: Path):
    archive_path = tmp_path / "cv.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, payload in (("a.mp3", b"aaa"), ("b.mp3", b"bbb"), ("unused.mp3", b"no")):
            member = tarfile.TarInfo(f"cv/xx/clips/{name}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "records": [
                    {"path": "a.mp3", "speaker_id": "one", "split": "train", "duration_seconds": 1.0},
                    {"path": "b.mp3", "speaker_id": "two", "split": "eval", "duration_seconds": 2.0},
                ]
            }
        )
    )

    output = tmp_path / "selected"
    manifest = extract(archive_path, selection_path, output)

    assert (output / "train/a.mp3").read_bytes() == b"aaa"
    assert (output / "eval/b.mp3").read_bytes() == b"bbb"
    assert not (output / "unused.mp3").exists()
    assert manifest["clip_count"] == 2
    assert manifest["records"][1]["sha256"] == hashlib.sha256(b"aaa").hexdigest()


def test_extract_removes_partial_output_when_clip_is_missing(tmp_path: Path):
    archive_path = tmp_path / "cv.tar.gz"
    with tarfile.open(archive_path, "w:gz"):
        pass
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {"records": [{"path": "missing.mp3", "speaker_id": "one", "split": "train", "duration_seconds": 1.0}]}
        )
    )
    output = tmp_path / "selected"

    with pytest.raises(ValueError, match="missing"):
        extract(archive_path, selection_path, output)
    assert not output.with_name("selected.partial").exists()
