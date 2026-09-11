import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from rnnoise_mlx.tools import mlflow_checkpoint as remote


class Client:
    def __init__(self, root):
        self.root = root
        self.fail_after = None
        self.uploads = 0
        self.corrupt = False

    def log_artifact(self, run_id, local, artifact_path):
        if self.fail_after is not None and self.uploads >= self.fail_after:
            raise OSError("interrupted upload")
        self.uploads += 1
        p = self.root / artifact_path / Path(local).name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, p)
        if self.corrupt and p.name == "optimizer.safetensors":
            p.write_bytes(b"corrupt")

    def download_artifacts(self, run_id, artifact, destination):
        src = self.root / artifact
        dst = Path(destination) / src.name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copyfile(src, dst)
        return str(dst)

    def list_artifacts(self, run_id, path):
        root = self.root / path
        if not root.exists():
            return []
        return [SimpleNamespace(path=p.relative_to(self.root).as_posix(), is_dir=p.is_dir()) for p in root.iterdir()]


def checkpoint(tmp_path, update=10):
    p = tmp_path / f"local-{update}"
    p.mkdir()
    for name in remote.FILES:
        (p / name).write_bytes(b"payload")
    (p / "trainer-state.json").write_text(json.dumps({"format_version": 1, "update": update}))
    manifest = {"format_version": 1, "files": {name: remote.sha256(p / name) for name in remote.FILES}}
    (p / "manifest.json").write_text(json.dumps(manifest))
    return p


def test_upload_download_roundtrip_and_unique_attempts(tmp_path):
    client = Client(tmp_path / "server")
    source = checkpoint(tmp_path)
    first = remote.upload_checkpoint(client, "run", source, 10)
    second = remote.upload_checkpoint(client, "run", source, 10)
    assert first != second
    destination = remote.download_checkpoint(client, "run", tmp_path / "restored")
    assert remote.verify_checkpoint(destination, 10) == remote.sha256(source / "manifest.json")
    assert (client.root / first / remote.COMPLETE).is_file()


def test_interrupted_upload_has_no_marker_and_is_not_selected(tmp_path):
    client = Client(tmp_path / "server")
    remote.upload_checkpoint(client, "run", checkpoint(tmp_path), 10)
    client.fail_after = client.uploads + 2
    with pytest.raises(OSError):
        remote.upload_checkpoint(client, "run", checkpoint(tmp_path, 20), 20)
    result = remote.download_checkpoint(client, "run", tmp_path / "restored")
    assert json.loads((result / "trainer-state.json").read_text())["update"] == 10
    with pytest.raises(FileNotFoundError):
        remote.download_checkpoint(client, "run", tmp_path / "missing", 20)
    assert not (tmp_path / "missing").exists()


def test_roundtrip_corruption_prevents_publication(tmp_path):
    client = Client(tmp_path / "server")
    client.corrupt = True
    with pytest.raises(ValueError, match="checksum"):
        remote.upload_checkpoint(client, "run", checkpoint(tmp_path), 10)
    assert not list(client.root.rglob(remote.COMPLETE))


def test_corrupt_committed_checkpoint_fails_closed(tmp_path):
    client = Client(tmp_path / "server")
    remote.upload_checkpoint(client, "run", checkpoint(tmp_path), 10)
    artifact = remote.upload_checkpoint(client, "run", checkpoint(tmp_path, 20), 20)
    (client.root / artifact / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        remote.download_checkpoint(client, "run", tmp_path / "restored")
    assert not (tmp_path / "restored").exists()
    assert not list(tmp_path.glob(".checkpoint-download-*"))


def test_existing_destination_is_not_overwritten(tmp_path):
    destination = tmp_path / "keep"
    destination.mkdir()
    (destination / "user-file").write_text("preserve")
    with pytest.raises(FileExistsError):
        remote.download_checkpoint(Client(tmp_path / "server"), "run", destination)
    assert (destination / "user-file").read_text() == "preserve"


def test_dangling_destination_symlink_is_not_overwritten(tmp_path):
    destination = tmp_path / "keep-link"
    destination.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        remote.download_checkpoint(Client(tmp_path / "server"), "run", destination)
    assert destination.is_symlink()


@pytest.mark.parametrize("uri", ["file:///tmp/mlruns", "sqlite:///db", "./mlruns"])
def test_direct_tracking_store_rejected(uri):
    with pytest.raises(ValueError, match="HTTP"):
        remote._client(uri)


def test_manifest_path_traversal_and_wrong_update_rejected(tmp_path):
    p = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="update"):
        remote.verify_checkpoint(p, 20)
    m = json.loads((p / "manifest.json").read_text())
    m["files"]["../outside"] = "x"
    (p / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="incomplete"):
        remote.verify_checkpoint(p, 10)


def test_marker_identity_is_checked(tmp_path):
    client = Client(tmp_path / "server")
    artifact = remote.upload_checkpoint(client, "run", checkpoint(tmp_path), 10)
    p = client.root / artifact / remote.COMPLETE
    marker = json.loads(p.read_text()); marker["run_id"] = "other"
    p.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="completion marker"):
        remote.download_checkpoint(client, "run", tmp_path / "restored")
