import hashlib
import io
import tarfile

from rnnoise_mlx.tools.build_corpus_listening_audit import extract_member, review_template, round_robin, select_source


def row(path, speaker, flags=(), edge=0):
    return {
        "source_id": "eng",
        "archive": "a.zip",
        "path": path,
        "speaker_id": speaker,
        "diagnostic_flags": list(flags),
        "leading_low_rms_seconds": edge,
        "trailing_low_rms_seconds": 0,
    }


def test_round_robin_balances_speakers():
    rows = [row(f"a{i}", "a") for i in range(5)] + [row(f"b{i}", "b") for i in range(5)]
    selected = round_robin(rows, 4)
    assert {item["speaker_id"] for item in selected} == {"a", "b"}


def test_select_source_keeps_critical_flags_and_clean_reference():
    rows = [
        row("clip.wav", "a", ["hard_clipping"]),
        row("edge1.wav", "a", ["long_leading_low_rms"], 3),
        row("edge2.wav", "b", ["long_leading_low_rms"], 2),
        row("clean1.wav", "a"),
        row("clean2.wav", "b"),
    ]
    selected = select_source(rows, 2)
    assert "clip.wav" in {item["path"] for item in selected["priority"]}
    assert all(not item["diagnostic_flags"] for item in selected["reference"])


def test_review_template_leaves_decision_fields_empty():
    record = {
        "source_id": "eng",
        "group": "priority",
        "path": "eng/priority/a.wav",
        "speaker_id": "a:1",
        "duration_seconds": 1.25,
        "diagnostic_flags": ["hard_clipping"],
    }
    lines = review_template({"records": [record]}).splitlines()
    assert lines[0].endswith("decision\treason\tnotes")
    assert lines[1].endswith("hard_clipping\t\t\t")


def test_select_source_supplements_when_few_rows_are_flagged():
    rows = [row("flag.wav", "a", ["hard_clipping"])] + [
        row(f"clean{i}.wav", "a" if i % 2 else "b") for i in range(8)
    ]
    selected = select_source(rows, 4)
    assert len(selected["priority"]) == 4
    assert len(selected["reference"]) == 4
    assert not ({item["path"] for item in selected["priority"]} & {item["path"] for item in selected["reference"]})


def test_extract_member_supports_tar_gz(tmp_path):
    archive_path = tmp_path / "audio.tar.gz"
    payload = b"RIFF-test-wave"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("speaker/clip.wav")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    destination = tmp_path / "clip.wav"

    extract_member(archive_path, "speaker/clip.wav", destination)

    assert destination.read_bytes() == payload
