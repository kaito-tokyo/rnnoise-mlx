"""Manage the portable rnnoise-mlx training volume."""

from __future__ import annotations

import argparse
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
    preflight(root, str(config["volume_uuid"]))
    if config.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported portable storage format")
    return config


def machine_id() -> str:
    value = socket.gethostname().split(".", 1)[0].lower()
    normalized = re.sub(r"[^a-z0-9-]+", "-", value).strip("-")
    return normalized or "unknown-mac"


def initialize(root: Path, expected_uuid: str = DEFAULT_UUID) -> dict[str, object]:
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
    pid = int(path.read_text().strip())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        path.unlink(missing_ok=True)
        return None
    return pid


def start_mlflow(root: Path, port: int = 5000, timeout: float = 30.0) -> int:
    load_volume_config(root)
    running = _running_pid(root)
    if running is not None:
        raise RuntimeError(f"MLflow is already running with PID {running}")
    database = root / "mlflow" / "mlflow.db"
    artifacts = root / "mlflow" / "artifacts"
    log_path = root / "mlflow" / "logs" / "server.log"
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
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    _pid_path(root).write_text(f"{process.pid}\n")
    deadline = time.monotonic() + timeout
    health = f"http://127.0.0.1:{port}/health"
    try:
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
        if process.poll() is None:
            process.terminate()
        _pid_path(root).unlink(missing_ok=True)
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
    files = sorted(item for item in path.rglob("*") if item.is_file())
    records = []
    total = 0
    for item in files:
        size = item.stat().st_size
        total += size
        record: dict[str, object] = {"path": item.relative_to(path).as_posix(), "bytes": size}
        if include_hashes:
            digest = hashlib.sha256()
            with item.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            record["sha256"] = digest.hexdigest()
        records.append(record)
    return {"files": len(files), "bytes": total, "records": records}


def verify_copy(source: Path, destination: Path, record: Path | None = None) -> dict[str, object]:
    if not source.is_dir() or not destination.is_dir():
        raise FileNotFoundError("source and destination must both be directories")
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
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary destination already exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary, copy_function=shutil.copy2)
    try:
        result = verify_copy(source, temporary)
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
    if temporary.parent != destination.parent:
        raise ValueError("temporary and destination must be siblings")
    if root not in destination.resolve().parents:
        raise ValueError("destination must be on the registered storage volume")
    if not temporary.is_dir():
        raise FileNotFoundError(f"temporary copy does not exist: {temporary}")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    inventory_path = root / "inventory" / "datasets.json"
    inventory = json.loads(inventory_path.read_text())
    if any(item.get("name") == name for item in inventory["datasets"]):
        raise ValueError(f"dataset is already registered: {name}")
    os.replace(temporary, destination)
    record = {
        "name": name,
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "files": files,
        "bytes": total_bytes,
        "verification": "rsync-checksum-dry-run",
    }
    inventory["datasets"].append(record)
    _json_write(inventory_path, inventory)
    return record


def eject_check(root: Path) -> dict[str, object]:
    config = load_volume_config(root)
    running = _running_pid(root)
    if running is not None:
        raise RuntimeError(f"MLflow is still running with PID {running}")
    partials = [
        str(path)
        for base in (
            root / "datasets",
            root / "features",
            root / "experiments",
            root / "references",
        )
        for path in base.rglob("*")
        if path.name.startswith(".") and ("partial" in path.name or ".tmp-" in path.name)
    ]
    if partials:
        raise RuntimeError(f"incomplete temporary paths remain: {partials[:5]}")
    active_checkpoints = []
    for experiment in sorted((root / "experiments" / "active").iterdir()):
        if not experiment.is_dir():
            continue
        checkpoints = sorted((experiment / "checkpoints").glob("update-*"))
        if not checkpoints:
            raise RuntimeError(f"active experiment has no complete checkpoint: {experiment}")
        latest = checkpoints[-1]
        manifest = json.loads((latest / "manifest.json").read_text())
        corrupt = [
            name
            for name, expected in manifest.get("files", {}).items()
            if not (latest / name).is_file()
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
    args = parser.parse_args()

    if args.command == "init":
        result = initialize(args.root)
    elif args.command == "preflight":
        result = preflight(args.root)
    elif args.command == "mlflow-start":
        result = {"pid": start_mlflow(args.root, args.port), "tracking_uri": f"http://127.0.0.1:{args.port}"}
    elif args.command == "mlflow-stop":
        result = {"sqlite_integrity": stop_mlflow(args.root)}
    elif args.command == "verify-copy":
        result = verify_copy(args.source, args.destination, args.record)
    elif args.command == "copy-tree":
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
    else:
        result = eject_check(args.root)
    print(json.dumps(display_result(result), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
