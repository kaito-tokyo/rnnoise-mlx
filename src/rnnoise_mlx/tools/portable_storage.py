"""Manage the portable rnnoise-mlx training volume."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from urllib.request import urlopen
import uuid


DEFAULT_ROOT = Path("/Volumes/rnnoise-mlx-train")
DEFAULT_UUID = "A40C7B4F-BA11-4F0F-A307-308231058AC6"
FORMAT_VERSION = 1
DIRECTORIES = (
    "source",
    "datasets/sources",
    "datasets/prepared",
    "datasets/selections",
    "datasets/manifests",
    "datasets/legal",
    "features",
    "experiments/active",
    "experiments/stopped",
    "experiments/scratch",
    "mlflow/artifacts",
    "mlflow/logs",
    "runtime",
    "references/models",
    "references/evaluation-sets",
    "inventory/checksums",
)


def _json_write(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def disk_info(root: Path) -> dict[str, object]:
    result = subprocess.run(
        ["/usr/sbin/diskutil", "info", "-plist", str(root)],
        check=True,
        capture_output=True,
    )
    return plistlib.loads(result.stdout)


def preflight(root: Path, expected_uuid: str = DEFAULT_UUID) -> dict[str, object]:
    root = root.expanduser()
    if str(root) != str(DEFAULT_ROOT):
        raise ValueError(f"storage must be mounted exactly at {DEFAULT_ROOT}: {root}")
    if not root.is_dir():
        raise FileNotFoundError(f"storage is not mounted: {root}")
    if root.resolve() != DEFAULT_ROOT:
        raise ValueError(f"storage mount resolves to an unexpected path: {root.resolve()}")
    info = disk_info(root)
    actual_uuid = str(info.get("VolumeUUID", "")).upper()
    if actual_uuid != expected_uuid.upper():
        raise ValueError(
            f"storage volume UUID differs: expected {expected_uuid}, got {actual_uuid or 'unknown'}"
        )
    if info.get("MountPoint") != str(DEFAULT_ROOT):
        raise ValueError(f"storage mount point differs: {info.get('MountPoint')}")
    if info.get("ReadOnly"):
        raise ValueError("storage volume is read-only")
    if info.get("WritableVolume") is False:
        raise ValueError("storage volume is not writable")
    return info


def load_volume_config(root: Path) -> dict[str, object]:
    path = root / "inventory" / "volume.json"
    if not path.is_file():
        raise FileNotFoundError(f"portable storage is not initialized: {path}")
    config = json.loads(path.read_text())
    if str(config.get("volume_uuid", "")).upper() != DEFAULT_UUID:
        raise ValueError("portable storage configuration has an unregistered volume UUID")
    preflight(root)
    if config.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported portable storage format")
    return config


def _machine_identifiers() -> list[str]:
    identifiers = []
    for path in (Path("/etc/machine-id"), Path("/var/db/uuidtext")):
        try:
            identifiers.append(path.read_text().strip())
        except OSError:
            pass
    if not identifiers:
        identifiers.append(f"{os.getuid()}:{uuid.getnode():012x}")
    return identifiers


def machine_id() -> str:
    value = socket.gethostname().split(".", 1)[0].lower()
    normalized = re.sub(r"[^a-z0-9-]+", "-", value).strip("-")
    identifiers = _machine_identifiers()
    suffix = hashlib.sha256("\0".join(identifiers).encode()).hexdigest()[:12]
    return f"{normalized or 'unknown-machine'}-{suffix}"


def registered_volume_for_paths(paths: list[Path]) -> Path | None:
    """Validate and return the portable root when any path uses it."""
    root = Path(os.environ.get("RNNOISE_MLX_STORAGE_ROOT", DEFAULT_ROOT)).expanduser()
    resolved_root = root.resolve()
    if any(_is_within(path.expanduser().resolve(), resolved_root) for path in paths):
        load_volume_config(root)
        return resolved_root
    return None


@contextmanager
def volume_operation_guard(root: Path):
    """Exclude portable-volume writers while eject safety is being checked."""
    path = root / "runtime" / ".rnnoise-operation.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def initialize(root: Path, expected_uuid: str = DEFAULT_UUID) -> dict[str, object]:
    if expected_uuid.upper() != DEFAULT_UUID:
        raise ValueError("portable storage UUID is fixed and cannot be overridden")
    info = preflight(root, expected_uuid)
    for relative in DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "runtime" / machine_id()).mkdir(parents=True, exist_ok=True)
    config = {
        "format_version": FORMAT_VERSION,
        "volume_name": "rnnoise-mlx-train",
        "volume_uuid": expected_uuid,
        "mount_point": str(DEFAULT_ROOT),
        "encrypted": False,
        "filesystem": info.get("FilesystemType", "apfs"),
        "minimum_free_bytes": 180 * 1024**3,
    }
    config_path = root / "inventory" / "volume.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != config:
            raise FileExistsError(f"volume configuration differs: {config_path}")
    else:
        _json_write(config_path, config)
    datasets = root / "inventory" / "datasets.json"
    if not datasets.exists():
        _json_write(datasets, {"format_version": FORMAT_VERSION, "datasets": []})
    return config


def _pid_path(root: Path) -> Path:
    return root / "mlflow" / "mlflow.pid"


def _running_pid(root: Path) -> int | None:
    path = _pid_path(root)
    if not path.is_file():
        return None
    try:
        metadata = json.loads(path.read_text())
        if metadata == {"starting": True}:
            raise RuntimeError(f"MLflow startup state remains: {path}")
        pid = int(metadata["pid"])
        expected_database = str((root / "mlflow" / "mlflow.db").resolve())
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None
    try:
        result = subprocess.run(
            ["/bin/ps", "-ww", "-p", str(pid), "-o", "command="],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        result = None
    command = result.stdout.strip() if result is not None and result.returncode == 0 else ""
    if (
        "-m mlflow server" not in command
        or f"sqlite:///{expected_database}" not in command
    ):
        path.unlink(missing_ok=True)
        return None
    return pid


def _training_lock_is_live(metadata: object) -> bool:
    try:
        if not isinstance(metadata, dict) or metadata["hostname"] != socket.gethostname():
            return False
        pid = int(metadata["pid"])
        result = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "lstart="],
            check=False,
            text=True,
            capture_output=True,
        )
        return result.returncode == 0 and result.stdout.strip() == metadata["started_at"]
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _remove_pid_if_owned(root: Path, pid: int | None) -> None:
    path = _pid_path(root)
    try:
        metadata = json.loads(path.read_text())
        if metadata == {"starting": True} or (pid is not None and int(metadata["pid"]) == pid):
            path.unlink(missing_ok=True)
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass


def start_mlflow(root: Path, port: int = 5000, timeout: float = 30.0) -> int:
    load_volume_config(root)
    startup_lock = root / "runtime" / ".rnnoise-mlflow-start.lock"
    with startup_lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        return _start_mlflow_locked(root, port, timeout)


def _start_mlflow_locked(root: Path, port: int, timeout: float) -> int:
    running = _running_pid(root)
    if running is not None:
        raise RuntimeError(f"MLflow is already running with PID {running}")
    database = root / "mlflow" / "mlflow.db"
    artifacts = root / "mlflow" / "artifacts"
    log_path = root / "mlflow" / "logs" / "server.log"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(f"MLflow port is already in use: {port}") from error
    log = log_path.open("ab", buffering=0)
    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
        "--backend-store-uri",
        f"sqlite:///{database}",
        "--artifacts-destination",
        artifacts.as_uri(),
        "--serve-artifacts",
    ]
    process = None
    try:
        _json_write(_pid_path(root), {"starting": True})
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        _json_write(_pid_path(root), {"pid": process.pid, "database": str(database.resolve())})
        deadline = time.monotonic() + timeout
        health = f"http://127.0.0.1:{port}/health"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"MLflow exited with status {process.returncode}; see {log_path}")
            try:
                with urlopen(health, timeout=1) as response:
                    if response.status == 200:
                        return process.pid
            except OSError:
                time.sleep(0.25)
        raise TimeoutError(f"MLflow did not become healthy: {health}")
    except BaseException:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        _remove_pid_if_owned(root, process.pid if process is not None else None)
        raise
    finally:
        log.close()


def sqlite_integrity(database: Path) -> str:
    if not database.exists():
        return "not-created"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        result = connection.execute("PRAGMA integrity_check").fetchone()
    finally:
        connection.close()
    status = str(result[0]) if result else "missing-result"
    if status != "ok":
        raise RuntimeError(f"SQLite integrity check failed: {status}")
    return status


def stop_mlflow(root: Path, timeout: float = 30.0) -> str:
    load_volume_config(root)
    startup_lock = root / "runtime" / ".rnnoise-mlflow-start.lock"
    startup_lock.parent.mkdir(parents=True, exist_ok=True)
    with startup_lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        return _stop_mlflow_locked(root, timeout)


def _stop_mlflow_locked(root: Path, timeout: float) -> str:
    pid = _running_pid(root)
    if pid is not None:
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.25)
        else:
            raise TimeoutError(f"MLflow PID {pid} did not stop")
    _pid_path(root).unlink(missing_ok=True)
    return sqlite_integrity(root / "mlflow" / "mlflow.db")


def _tree_summary(path: Path, include_hashes: bool) -> dict[str, object]:
    files = sorted(item for item in path.rglob("*") if item.is_file() and not item.is_symlink())
    records = []
    total = 0
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path).as_posix()
        if item.is_symlink():
            records.append({"path": relative, "type": "symlink", "target": os.readlink(item)})
            continue
        if item.is_dir():
            records.append({"path": relative, "type": "directory"})
            continue
        if not item.is_file():
            continue
        size = item.stat().st_size
        total += size
        record: dict[str, object] = {"path": relative, "bytes": size}
        if include_hashes:
            digest = hashlib.sha256()
            with item.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            record["sha256"] = digest.hexdigest()
        records.append(record)
    return {"files": len(files), "bytes": total, "records": records}


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def verify_copy(source: Path, destination: Path, record: Path | None = None) -> dict[str, object]:
    if record is not None and (record.exists() or record.is_symlink()):
        raise FileExistsError(f"verification record already exists: {record}")
    paths = [source, destination]
    if record is not None:
        paths.append(record)
    portable_root = registered_volume_for_paths(paths)
    with volume_operation_guard(portable_root) if portable_root else nullcontext():
        return _verify_copy_locked(source, destination, record)


def _verify_copy_locked(source: Path, destination: Path, record: Path | None = None) -> dict[str, object]:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("source and destination must differ")
    if not source.is_dir() or not destination.is_dir():
        raise FileNotFoundError("source and destination must both be directories")
    if record is not None:
        record = record.resolve()
        if _is_within(record, source.resolve()) or _is_within(record, destination.resolve()):
            raise ValueError("verification record must be outside source and destination")
    source_summary = _tree_summary(source, True)
    destination_summary = _tree_summary(destination, True)
    matched = source_summary == destination_summary
    result = {
        "format_version": FORMAT_VERSION,
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "matched": matched,
        "summary": source_summary,
    }
    if record is not None:
        record.parent.mkdir(parents=True, exist_ok=True)
        _json_write(record, result)
    if not matched:
        raise ValueError("copied tree differs from source")
    return result


def copy_tree(source: Path, destination: Path, record: Path) -> dict[str, object]:
    if destination.is_symlink():
        raise FileExistsError(f"copy destination exists: {destination}")
    portable_root = registered_volume_for_paths([destination, record])
    with volume_operation_guard(portable_root) if portable_root else nullcontext():
        return _copy_tree_locked(source, destination, record)


def _copy_tree_locked(source: Path, destination: Path, record: Path) -> dict[str, object]:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("copy destination must not be inside the source")
    if record.exists() or record.is_symlink():
        raise FileExistsError(f"verification record already exists: {record}")
    record = record.resolve()
    if _is_within(record, source) or _is_within(record, destination):
        raise ValueError("verification record must be outside source and destination")
    if destination.exists():
        # A crash after rename but before _json_write leaves a verified
        # destination without its audit record. Recover only after hashing it.
        return _verify_copy_locked(source, destination, record)
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary destination already exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary, copy_function=shutil.copy2, symlinks=True)
    try:
        result = _verify_copy_locked(source, temporary)
        os.replace(temporary, destination)
        result["destination"] = str(destination.resolve())
        _json_write(record, result)
        return result
    except BaseException:
        raise


def finalize_verified_copy(
    root: Path,
    temporary: Path,
    destination: Path,
    *,
    name: str,
    source: Path,
    files: int,
    total_bytes: int,
) -> dict[str, object]:
    """Finalize a tree after an external rsync checksum dry-run succeeds."""
    load_volume_config(root)
    with volume_operation_guard(root):
        return _finalize_verified_copy_locked(
            root, temporary, destination, name=name, source=source,
            files=files, total_bytes=total_bytes,
        )


def _finalize_verified_copy_locked(
    root: Path,
    temporary: Path,
    destination: Path,
    *,
    name: str,
    source: Path,
    files: int,
    total_bytes: int,
) -> dict[str, object]:
    if temporary.parent != destination.parent:
        raise ValueError("temporary and destination must be siblings")
    if root not in destination.resolve().parents:
        raise ValueError("destination must be on the registered storage volume")
    inventory_path = root / "inventory" / "datasets.json"
    lock_path = root / "inventory" / "datasets.lock"
    lock_stream = lock_path.open("a+")
    fcntl.flock(lock_stream, fcntl.LOCK_EX)
    try:
        inventory = json.loads(inventory_path.read_text())
        if any(item.get("name") == name for item in inventory["datasets"]):
            raise ValueError(f"dataset is already registered: {name}")
        record = {"name": name, "source": str(source.resolve()), "destination": str(destination.resolve()), "files": files, "bytes": total_bytes, "verification": "rsync-checksum-dry-run"}
        marker = ".rnnoise-finalize-verified.json"
        if destination.exists():
            marker_path = destination / marker
            if not destination.is_dir() or not marker_path.is_file() or json.loads(marker_path.read_text()) != record:
                raise FileExistsError(f"destination already exists: {destination}")
            inventory["datasets"].append(record)
            _json_write(inventory_path, inventory)
            marker_path.unlink(missing_ok=True)
            return record
        resolved_temporary = temporary.resolve()
        if (
            temporary.is_symlink()
            or not temporary.is_dir()
            or not _is_within(resolved_temporary, root.resolve())
        ):
            raise ValueError(f"temporary copy must be a real directory on the storage volume: {temporary}")
        _json_write(temporary / marker, record)
        os.replace(temporary, destination)
        inventory["datasets"].append(record)
        _json_write(inventory_path, inventory)
        (destination / marker).unlink(missing_ok=True)
        return record
    finally:
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
        lock_stream.close()


def _is_temporary_path(path: Path) -> bool:
    name = path.name
    return (
        name.endswith((".partial", ".part"))
        or (name.startswith(".") and (".partial-" in name or ".tmp-" in name))
    )


def eject_check(root: Path) -> dict[str, object]:
    load_volume_config(root)
    startup_lock = root / "runtime" / ".rnnoise-mlflow-start.lock"
    startup_lock.parent.mkdir(parents=True, exist_ok=True)
    with startup_lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        training_guard = root / ".rnnoise-training.lock.guard"
        with training_guard.open("a+") as training_stream:
            fcntl.flock(training_stream, fcntl.LOCK_EX)
            with volume_operation_guard(root):
                return _eject_check_locked(root, root_training_guard_held=True)


def eject_volume(root: Path) -> dict[str, object]:
    """Verify and eject while excluding new portable-volume operations."""
    load_volume_config(root)
    startup_lock = root / "runtime" / ".rnnoise-mlflow-start.lock"
    startup_lock.parent.mkdir(parents=True, exist_ok=True)
    with startup_lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        training_guard = root / ".rnnoise-training.lock.guard"
        with training_guard.open("a+") as training_stream:
            fcntl.flock(training_stream, fcntl.LOCK_EX)
            with volume_operation_guard(root):
                result = _eject_check_locked(root, root_training_guard_held=True)
                subprocess.run(["/usr/sbin/diskutil", "eject", str(root)], check=True)
                return result


def _eject_check_locked(root: Path, *, root_training_guard_held: bool = False) -> dict[str, object]:
    config = load_volume_config(root)
    running = _running_pid(root)
    if running is not None:
        raise RuntimeError(f"MLflow is still running with PID {running}")
    partials = [
        str(path)
        for path in root.rglob("*")
        if _is_temporary_path(path)
    ]
    if partials:
        raise RuntimeError(f"incomplete temporary paths remain: {partials[:5]}")
    live_training = []
    for lock in root.rglob(".rnnoise-training.lock"):
        if root_training_guard_held and lock.parent == root:
            _inspect_training_lock(lock, live_training)
            continue
        guard = lock.with_name(f"{lock.name}.guard")
        with guard.open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            _inspect_training_lock(lock, live_training)
    if live_training:
        raise RuntimeError(f"training is still running: {live_training[:5]}")
    active_checkpoints = []
    for experiment in sorted((root / "experiments" / "active").iterdir()):
        if not experiment.is_dir():
            continue
        checkpoints = sorted((experiment / "checkpoints").glob("update-*"))
        if not checkpoints:
            raise RuntimeError(f"active experiment has no complete checkpoint: {experiment}")
        latest = checkpoints[-1]
        manifest_path = latest / "manifest.json"
        if manifest_path.is_symlink():
            raise RuntimeError(f"active checkpoint manifest is a symlink: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        required = {"model.safetensors", "optimizer.safetensors", "mlx-random-state.safetensors", "trainer-state.json"}
        files = manifest.get("files")
        if manifest.get("format_version") != 1 or not isinstance(files, dict) or set(files) != required:
            raise RuntimeError(f"active checkpoint manifest is incomplete: {latest}")
        corrupt = [
            name
            for name, expected in files.items()
            if (latest / name).is_symlink()
            or not (latest / name).is_file()
            or hashlib.sha256((latest / name).read_bytes()).hexdigest() != expected
        ]
        if corrupt:
            raise RuntimeError(f"active checkpoint differs from manifest: {latest}: {corrupt}")
        active_checkpoints.append(str(latest))
    database_status = sqlite_integrity(root / "mlflow" / "mlflow.db")
    free_bytes = shutil.disk_usage(root).free
    return {
        "volume_uuid": config["volume_uuid"],
        "mlflow": "stopped",
        "sqlite_integrity": database_status,
        "free_bytes": free_bytes,
        "minimum_free_bytes_met": free_bytes >= int(config["minimum_free_bytes"]),
        "active_checkpoints": active_checkpoints,
    }


def _inspect_training_lock(lock: Path, live_training: list[str]) -> None:
    try:
        metadata = json.loads(lock.read_text())
        pid = int(metadata["pid"])
        if not _training_lock_is_live(metadata):
            raise OSError
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        lock.unlink(missing_ok=True)
    else:
        live_training.append(f"{lock.parent} (PID {pid})")


def display_result(result: object) -> object:
    """Keep CLI output compact while detailed verification stays on disk."""
    if not isinstance(result, dict):
        return result
    if "VolumeUUID" in result:
        return {
            "volume_uuid": result.get("VolumeUUID"),
            "mount_point": result.get("MountPoint"),
            "filesystem": result.get("FilesystemType"),
            "writable": result.get("WritableVolume", not result.get("ReadOnly", False)),
        }
    summary = result.get("summary")
    if isinstance(summary, dict) and "records" in summary:
        return {
            **{key: value for key, value in result.items() if key != "summary"},
            "summary": {"files": summary["files"], "bytes": summary["bytes"]},
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("RNNOISE_MLX_STORAGE_ROOT", DEFAULT_ROOT)))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    subparsers.add_parser("machine-id")
    subparsers.add_parser("preflight")
    start = subparsers.add_parser("mlflow-start")
    start.add_argument("--port", type=int, default=5000)
    subparsers.add_parser("mlflow-stop")
    verify = subparsers.add_parser("verify-copy")
    verify.add_argument("source", type=Path)
    verify.add_argument("destination", type=Path)
    verify.add_argument("--record", type=Path)
    copy = subparsers.add_parser("copy-tree")
    copy.add_argument("source", type=Path)
    copy.add_argument("destination", type=Path)
    copy.add_argument("--record", type=Path, required=True)
    finalize = subparsers.add_parser("finalize-verified-copy")
    finalize.add_argument("temporary", type=Path)
    finalize.add_argument("destination", type=Path)
    finalize.add_argument("--name", required=True)
    finalize.add_argument("--source", type=Path, required=True)
    finalize.add_argument("--files", type=int, required=True)
    finalize.add_argument("--bytes", type=int, required=True)
    subparsers.add_parser("eject-check")
    subparsers.add_parser("eject")
    args = parser.parse_args()

    if args.command == "machine-id":
        print(machine_id())
        return
    if args.command == "init":
        result = initialize(args.root)
    elif args.command == "preflight":
        result = load_volume_config(args.root)
    elif args.command == "mlflow-start":
        result = {"pid": start_mlflow(args.root, args.port), "tracking_uri": f"http://127.0.0.1:{args.port}"}
    elif args.command == "mlflow-stop":
        result = {"sqlite_integrity": stop_mlflow(args.root)}
    elif args.command == "verify-copy":
        result = verify_copy(args.source, args.destination, args.record)
    elif args.command == "copy-tree":
        root = args.root.resolve()
        destination = args.destination.resolve()
        load_volume_config(root)
        if destination != root and root not in destination.parents:
            raise ValueError("copy destination must be on the registered storage volume")
        result = copy_tree(args.source, args.destination, args.record)
    elif args.command == "finalize-verified-copy":
        result = finalize_verified_copy(
            args.root,
            args.temporary,
            args.destination,
            name=args.name,
            source=args.source,
            files=args.files,
            total_bytes=args.bytes,
        )
    elif args.command == "eject-check":
        result = eject_check(args.root)
    else:
        result = eject_volume(args.root)
    print(json.dumps(display_result(result), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
