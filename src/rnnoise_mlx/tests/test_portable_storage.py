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


def test_registered_operations_reject_self_declared_uuid(tmp_path, monkeypatch):
    (tmp_path / "inventory").mkdir()
    portable_storage._json_write(
        tmp_path / "inventory" / "volume.json", {"format_version": 1, "volume_uuid": "other"}
    )
    monkeypatch.setattr(portable_storage, "DEFAULT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="unregistered"):
        portable_storage.load_volume_config(tmp_path)
    with pytest.raises(ValueError, match="cannot be overridden"):
        portable_storage.initialize(tmp_path, "other")


def test_preflight_cli_requires_initialized_volume(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["storage", "--root", str(tmp_path), "preflight"])

    with pytest.raises(FileNotFoundError, match="not initialized"):
        portable_storage.main()


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


def test_verify_copy_rejects_audit_record_inside_verified_tree(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "file").write_bytes(b"same")
    (destination / "file").write_bytes(b"same")

    with pytest.raises(ValueError, match="outside"):
        portable_storage.verify_copy(source, destination, source / "verification.json")


def test_copy_tree_recovers_record_after_post_rename_interruption(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "file").write_bytes(b"same")
    (destination / "file").write_bytes(b"same")
    record = tmp_path / "inventory" / "copy.json"

    result = portable_storage.copy_tree(source, destination, record)

    assert result["matched"]
    assert record.is_file()


def test_copy_tree_rejects_destination_below_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "file").write_bytes(b"same")

    with pytest.raises(ValueError, match="must not be inside"):
        portable_storage.copy_tree(source, source / "copy", tmp_path / "copy.json")


def test_copy_tree_preserves_directory_symlinks(tmp_path):
    source = tmp_path / "source"
    external = tmp_path / "external"
    destination = tmp_path / "destination"
    source.mkdir()
    external.mkdir()
    (external / "file").write_bytes(b"payload")
    (source / "linked").symlink_to(external, target_is_directory=True)

    result = portable_storage.copy_tree(source, destination, tmp_path / "record.json")

    assert result["matched"]
    assert (destination / "linked").is_symlink()


def test_copy_tree_cli_requires_destination_on_registered_volume(tmp_path, monkeypatch):
    root = tmp_path / "volume"
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "outside"
    monkeypatch.setattr(portable_storage, "load_volume_config", lambda root: {})
    monkeypatch.setattr(
        "sys.argv",
        [
            "storage", "--root", str(root), "copy-tree", str(source), str(destination),
            "--record", str(tmp_path / "record.json"),
        ],
    )

    with pytest.raises(ValueError, match="registered storage volume"):
        portable_storage.main()


def test_sqlite_integrity_check(tmp_path):
    database = tmp_path / "mlflow.db"
    import sqlite3

    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE test (value TEXT)")
    connection.commit()
    connection.close()

    assert portable_storage.sqlite_integrity(database) == "ok"


def test_running_pid_rejects_stale_or_unrelated_process(tmp_path, monkeypatch):
    root = tmp_path
    (root / "mlflow").mkdir()
    portable_storage._json_write(
        root / "mlflow" / "mlflow.pid", {"pid": 42, "database": "unused"}
    )
    result = type("Result", (), {"returncode": 0, "stdout": "python unrelated.py\n"})()
    monkeypatch.setattr(portable_storage.subprocess, "run", lambda *args, **kwargs: result)

    assert portable_storage._running_pid(root) is None
    assert not (root / "mlflow" / "mlflow.pid").exists()


def test_running_pid_accepts_matching_mlflow_server(tmp_path, monkeypatch):
    root = tmp_path
    database = root / "mlflow" / "mlflow.db"
    database.parent.mkdir()
    portable_storage._json_write(
        root / "mlflow" / "mlflow.pid", {"pid": 42, "database": str(database.resolve())}
    )
    result = type("Result", (), {
        "returncode": 0,
        "stdout": f"python -m mlflow server --backend-store-uri sqlite:///{database.resolve()}\n",
    })()
    monkeypatch.setattr(portable_storage.subprocess, "run", lambda *args, **kwargs: result)

    assert portable_storage._running_pid(root) == 42


def test_running_pid_requests_an_untruncated_command_line(tmp_path, monkeypatch):
    root = tmp_path
    database = root / "mlflow" / "mlflow.db"
    database.parent.mkdir()
    portable_storage._json_write(
        root / "mlflow" / "mlflow.pid", {"pid": 42, "database": str(database.resolve())}
    )
    seen = []
    result = type("Result", (), {
        "returncode": 0,
        "stdout": f"python -m mlflow server --backend-store-uri sqlite:///{database.resolve()}\n",
    })()
    monkeypatch.setattr(
        portable_storage.subprocess,
        "run",
        lambda args, **kwargs: seen.append(args) or result,
    )

    assert portable_storage._running_pid(root) == 42
    assert "-ww" in seen[0]


def test_mlflow_failure_cleanup_preserves_another_process_pid_record(tmp_path):
    (tmp_path / "mlflow").mkdir()
    pid_path = tmp_path / "mlflow" / "mlflow.pid"
    portable_storage._json_write(pid_path, {"pid": 43})

    portable_storage._remove_pid_if_owned(tmp_path, 42)

    assert __import__("json").loads(pid_path.read_text())["pid"] == 43


def test_start_mlflow_waits_for_failed_process_before_removing_pid(tmp_path, monkeypatch):
    root = tmp_path
    (root / "mlflow" / "logs").mkdir(parents=True)
    (root / "mlflow" / "artifacts").mkdir()
    (root / "runtime").mkdir()

    class Process:
        pid = 42

        def __init__(self):
            self.terminated = False
            self.waited = False

        def poll(self):
            return 0 if self.waited else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            assert self.terminated
            self.waited = True

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def setsockopt(self, *args):
            pass

        def bind(self, address):
            pass

    process = Process()
    monkeypatch.setattr(portable_storage, "load_volume_config", lambda root: {})
    monkeypatch.setattr(portable_storage, "_running_pid", lambda root: None)
    monkeypatch.setattr(portable_storage.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(portable_storage.os, "killpg", lambda pid, signal: process.terminate())
    monkeypatch.setattr(portable_storage.socket, "socket", lambda *args: Probe())

    with pytest.raises(TimeoutError):
        portable_storage.start_mlflow(root, timeout=0)

    assert process.waited
    assert not (root / "mlflow" / "mlflow.pid").exists()


def test_temporary_path_detection_does_not_match_ordinary_partial_names():
    assert portable_storage._is_temporary_path(Path("clip.partial.wav"))
    assert portable_storage._is_temporary_path(Path(".copy.partial-1"))
    assert portable_storage._is_temporary_path(Path(".record.tmp-1"))
    assert portable_storage._is_temporary_path(Path(".checkpoint-download.tmp-1"))
    assert portable_storage._is_temporary_path(Path("archive.part"))
    assert not portable_storage._is_temporary_path(Path("partial_speech.wav"))


def test_eject_check_scans_temporary_paths_outside_dataset_roots(tmp_path, monkeypatch):
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / ".copy.partial-1").mkdir()
    monkeypatch.setattr(portable_storage, "load_volume_config", lambda root: {"volume_uuid": "id", "minimum_free_bytes": 0})
    monkeypatch.setattr(portable_storage, "_running_pid", lambda root: None)

    with pytest.raises(RuntimeError, match="incomplete temporary"):
        portable_storage.eject_check(tmp_path)


def test_eject_check_acquires_mlflow_startup_lock(tmp_path, monkeypatch):
    calls = []
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / ".copy.partial-1").mkdir()
    monkeypatch.setattr(portable_storage, "load_volume_config", lambda root: {})
    monkeypatch.setattr(portable_storage, "_running_pid", lambda root: None)
    monkeypatch.setattr(portable_storage.fcntl, "flock", lambda stream, operation: calls.append(operation))

    with pytest.raises(RuntimeError, match="incomplete temporary"):
        portable_storage.eject_check(tmp_path)

    assert portable_storage.fcntl.LOCK_EX in calls


def test_eject_check_rejects_live_training_lock(tmp_path, monkeypatch):
    root = tmp_path
    lock = root / "experiments" / "active" / "trial" / ".rnnoise-training.lock"
    lock.parent.mkdir(parents=True)
    portable_storage._json_write(
        lock, {"pid": 42, "hostname": "host", "started_at": "start"}
    )
    (root / "inventory").mkdir()
    portable_storage._json_write(root / "inventory" / "volume.json", {"format_version": 1})
    monkeypatch.setattr(portable_storage, "load_volume_config", lambda root: {"volume_uuid": "id", "minimum_free_bytes": 0})
    monkeypatch.setattr(portable_storage, "_running_pid", lambda root: None)
    monkeypatch.setattr(portable_storage, "sqlite_integrity", lambda database: "ok")
    monkeypatch.setattr(portable_storage, "_training_lock_is_live", lambda metadata: True)

    with pytest.raises(RuntimeError, match="training is still running"):
        portable_storage.eject_check(root)


def test_finalize_verified_copy_rejects_symlinked_temporary_directory(tmp_path, monkeypatch):
    root = tmp_path / "volume"
    temporary = root / "datasets" / ".source.partial"
    destination = root / "datasets" / "source"
    external = tmp_path / "external"
    external.mkdir()
    temporary.parent.mkdir(parents=True)
    temporary.symlink_to(external, target_is_directory=True)
    (root / "inventory").mkdir()
    portable_storage._json_write(
        root / "inventory" / "datasets.json", {"format_version": 1, "datasets": []}
    )
    monkeypatch.setattr(portable_storage, "load_volume_config", lambda root: {})

    with pytest.raises(ValueError, match="real directory"):
        portable_storage.finalize_verified_copy(
            root,
            temporary,
            destination,
            name="source",
            source=tmp_path / "original",
            files=2,
            total_bytes=3,
        )


def test_eject_check_rejects_incomplete_checkpoint_manifest(tmp_path, monkeypatch):
    root = tmp_path
    checkpoint = root / "experiments" / "active" / "trial" / "checkpoints" / "update-1"
    checkpoint.mkdir(parents=True)
    portable_storage._json_write(
        checkpoint / "manifest.json", {"format_version": 1, "files": {}},
    )
    monkeypatch.setattr(portable_storage, "load_volume_config", lambda root: {"volume_uuid": "id", "minimum_free_bytes": 0})
    monkeypatch.setattr(portable_storage, "_running_pid", lambda root: None)

    with pytest.raises(RuntimeError, match="manifest is incomplete"):
        portable_storage.eject_check(root)


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
