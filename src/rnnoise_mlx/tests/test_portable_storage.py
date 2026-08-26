import os
from pathlib import Path
import plistlib

import pytest

from rnnoise_mlx.tools import portable_storage


def _disk_info(uuid=portable_storage.DEFAULT_UUID, mount="/Volumes/rnnoise-mlx-train"):
    return {
        "VolumeUUID": uuid,
        "MountPoint": mount,
        "ReadOnly": False,
        "FilesystemType": "apfs",
    }


def test_preflight_accepts_registered_volume(monkeypatch):
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    monkeypatch.setattr(portable_storage, "disk_info", lambda root: _disk_info())

    result = portable_storage.preflight(portable_storage.DEFAULT_ROOT)

    assert result["VolumeUUID"] == portable_storage.DEFAULT_UUID


def test_preflight_rejects_wrong_uuid(monkeypatch):
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    monkeypatch.setattr(portable_storage, "disk_info", lambda root: _disk_info("wrong"))

    with pytest.raises(ValueError, match="UUID differs"):
        portable_storage.preflight(portable_storage.DEFAULT_ROOT)


def test_preflight_rejects_alternate_mount():
    with pytest.raises(ValueError, match="mounted exactly"):
        portable_storage.preflight(Path("/Volumes/rnnoise-mlx-train 1"))


def test_initialize_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(portable_storage, "DEFAULT_ROOT", tmp_path)
    monkeypatch.setattr(
        portable_storage,
        "preflight",
        lambda root, expected_uuid=portable_storage.DEFAULT_UUID: _disk_info(mount=str(tmp_path)),
    )

    first = portable_storage.initialize(tmp_path)
    second = portable_storage.initialize(tmp_path)

    assert first == second
    assert (tmp_path / "inventory" / "volume.json").is_file()
    assert (tmp_path / "experiments" / "active").is_dir()


def test_verify_copy_detects_matching_and_different_trees(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "file").write_bytes(b"same")
    (destination / "file").write_bytes(b"same")

    assert portable_storage.verify_copy(source, destination)["matched"]
    (destination / "file").write_bytes(b"different")
    with pytest.raises(ValueError, match="differs"):
        portable_storage.verify_copy(source, destination)


def test_sqlite_integrity_check(tmp_path):
    database = tmp_path / "mlflow.db"
    import sqlite3

    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE test (value TEXT)")
    connection.commit()
    connection.close()

    assert portable_storage.sqlite_integrity(database) == "ok"


def test_finalize_verified_copy_renames_and_registers(tmp_path, monkeypatch):
    root = tmp_path
    temporary = root / "datasets" / ".source.partial"
    destination = root / "datasets" / "source"
    temporary.mkdir(parents=True)
    (root / "inventory").mkdir()
    portable_storage._json_write(
        root / "inventory" / "datasets.json",
        {"format_version": 1, "datasets": []},
    )
    monkeypatch.setattr(portable_storage, "load_volume_config", lambda root: {})

    result = portable_storage.finalize_verified_copy(
        root,
        temporary,
        destination,
        name="source",
        source=tmp_path / "original",
        files=2,
        total_bytes=3,
    )

    assert destination.is_dir()
    assert result["verification"] == "rsync-checksum-dry-run"
    inventory = __import__("json").loads(
        (root / "inventory" / "datasets.json").read_text()
    )
    assert inventory["datasets"] == [result]


def test_disk_info_reads_plist(monkeypatch):
    payload = plistlib.dumps(_disk_info())
    result = type("Result", (), {"stdout": payload})()
    monkeypatch.setattr(portable_storage.subprocess, "run", lambda *args, **kwargs: result)

    assert portable_storage.disk_info(Path("/Volumes/rnnoise-mlx-train"))["VolumeUUID"] == portable_storage.DEFAULT_UUID


def test_display_result_compacts_file_records():
    result = portable_storage.display_result(
        {"matched": True, "summary": {"files": 1, "bytes": 2, "records": [{"path": "a"}]}}
    )

    assert result == {"matched": True, "summary": {"files": 1, "bytes": 2}}
