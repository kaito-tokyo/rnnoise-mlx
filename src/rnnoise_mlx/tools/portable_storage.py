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
    if info.get("ReadOnlyVolume") or info.get("ReadOnlyMedia"):
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


def coordination_lock_path(root: Path, name: str) -> Path:
    """Use a per-user lock outside a volume that may be ejected."""
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    identity = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
    path = cache_root / "rnnoise-mlx" / "locks" / f"{identity}-{name}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"coordination lock must not be a symlink: {path}")
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)
    return path


@contextmanager
def volume_operation_guard(root: Path):
    """Exclude portable-volume writers while eject safety is being checked."""
    path = coordination_lock_path(root, "operation")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        load_volume_config(root)
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
    training_guard = coordination_lock_path(root, "training")
    with training_guard.open("a+") as training_stream:
        fcntl.flock(training_stream, fcntl.LOCK_EX)
        with volume_operation_guard(root):
            return _eject_check_locked(root, root_training_guard_held=True)


def eject_volume(root: Path) -> dict[str, object]:
    """Verify and eject while excluding new portable-volume operations."""
    load_volume_config(root)
    training_guard = coordination_lock_path(root, "training")
    with training_guard.open("a+") as training_stream:
        fcntl.flock(training_stream, fcntl.LOCK_EX)
        with volume_operation_guard(root):
            result = _eject_check_locked(root, root_training_guard_held=True)
            subprocess.run(["/usr/sbin/diskutil", "eject", str(root)], check=True)
            return result


def _eject_check_locked(root: Path, *, root_training_guard_held: bool = False) -> dict[str, object]:
    config = load_volume_config(root)
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
    active_root = root / "experiments" / "active"
    if active_root.is_symlink():
        raise RuntimeError(f"active experiment root is a symlink: {active_root}")
    for experiment in sorted(active_root.iterdir()):
        if experiment.is_symlink():
            raise RuntimeError(f"active experiment is a symlink: {experiment}")
        if not experiment.is_dir():
            continue
        checkpoint_root = experiment / "checkpoints"
        if checkpoint_root.is_symlink():
            raise RuntimeError(f"active checkpoint root is a symlink: {checkpoint_root}")
        checkpoints = sorted(checkpoint_root.glob("update-*"))
        if not checkpoints:
            raise RuntimeError(f"active experiment has no complete checkpoint: {experiment}")
        latest = checkpoints[-1]
        if latest.is_symlink():
            raise RuntimeError(f"active checkpoint is a symlink: {latest}")
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
    free_bytes = shutil.disk_usage(root).free
    return {
        "volume_uuid": config["volume_uuid"],
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
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
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
            "writable": not (result.get("ReadOnlyVolume") or result.get("ReadOnlyMedia")),
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
