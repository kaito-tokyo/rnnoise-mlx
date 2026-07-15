import json
import tarfile
from pathlib import Path

from rnnoise_mlx.tools.extract_tar_selection import extract


def test_extract_writes_selected_member_and_manifest(tmp_path: Path):
    source = tmp_path / "clip.flac"
    source.write_bytes(b"flac")
    archive = tmp_path / "audio.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        target.add(source, arcname="test/speaker/clip.flac")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"records": [{
        "path": "test/speaker/clip.flac", "split": "eval", "speaker_id": "speaker"
    }]}))
    output = tmp_path / "output"
    output.mkdir()
    manifest = extract(archive, selection, output)
    assert (output / "eval/clip.flac").read_bytes() == b"flac"
    assert manifest["clip_count"] == 1
    assert (output / "extraction-manifest.json").is_file()
