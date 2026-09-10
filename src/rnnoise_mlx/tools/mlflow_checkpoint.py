"""Verified, committed checkpoint transfers through an HTTP MLflow server."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urlparse
from uuid import uuid4

from mlflow.tracking import MlflowClient

COMPLETE = "complete.json"
FILES = {"model.safetensors", "optimizer.safetensors", "mlx-random-state.safetensors", "trainer-state.json"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_checkpoint(path: Path, update: int) -> str:
    """Validate the complete allowlisted payload before it can be published."""
    manifest_path = path / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("checkpoint manifest must not be a symlink")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != 1 or set(manifest.get("files", {})) != FILES:
        raise ValueError("incomplete checkpoint manifest")
    for name, digest in manifest["files"].items():
        p = path / name
        if p.is_symlink() or not p.is_file() or sha256(p) != digest:
            raise ValueError(f"checkpoint checksum differs: {name}")
    state = json.loads((path / "trainer-state.json").read_text())
    if state.get("format_version") != 1 or state.get("update") != update:
        raise ValueError("checkpoint update differs")
    return sha256(manifest_path)


def upload_checkpoint(client, run_id: str, checkpoint: Path, update: int) -> str:
    """Commit only after remote payload round-trip verification; never overwrite."""
    digest = verify_checkpoint(checkpoint, update)
    artifact = f"checkpoints/update-{update:08d}/{uuid4().hex}"
    with tempfile.TemporaryDirectory(prefix="rnnoise-checkpoint-") as work:
        work = Path(work)
        for name in sorted(FILES | {"manifest.json"}):
            client.log_artifact(run_id, str(checkpoint / name), artifact_path=artifact)
        restored = Path(client.download_artifacts(run_id, artifact, str(work)))
        if verify_checkpoint(restored, update) != digest:
            raise ValueError("remote checkpoint manifest differs")
        marker = {"format_version": 1, "run_id": run_id, "update": update,
                  "manifest_sha256": digest, "committed_ns": time.time_ns()}
        marker_path = work / COMPLETE
        marker_path.write_text(json.dumps(marker, sort_keys=True) + "\n")
        client.log_artifact(run_id, str(marker_path), artifact_path=artifact)
        check_dir = work / "marker-check"
        check_dir.mkdir()
        remote_marker = Path(client.download_artifacts(run_id, f"{artifact}/{COMPLETE}", str(check_dir)))
        if json.loads(remote_marker.read_text()) != marker:
            raise ValueError("remote checkpoint completion marker differs")
    return artifact


def _client(uri: str):
    parsed = urlparse(uri)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("checkpoint tracking URI must be HTTP(S)")
    client = MlflowClient(tracking_uri=uri)
    client.search_experiments(max_results=1)
    return client


def download_checkpoint(client, run_id: str, destination: Path, update: int | None = None) -> Path:
    """Select a committed generation, validate, then atomically publish locally."""
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"checkpoint destination exists: {destination}")
    candidates = []
    with tempfile.TemporaryDirectory(prefix="rnnoise-markers-") as work:
        for generation in client.list_artifacts(run_id, "checkpoints"):
            match = re.fullmatch(r"checkpoints/update-(\d{8,})", generation.path)
            if not generation.is_dir or match is None:
                continue
            number = int(match[1])
            if update is not None and number != update:
                continue
            for attempt in client.list_artifacts(run_id, generation.path):
                if not attempt.is_dir or not re.fullmatch(re.escape(generation.path) + r"/[0-9a-f]{32}", attempt.path):
                    continue
                if f"{attempt.path}/{COMPLETE}" not in {a.path for a in client.list_artifacts(run_id, attempt.path)}:
                    continue
                with tempfile.TemporaryDirectory(dir=work) as marker_dir:
                    p = Path(client.download_artifacts(run_id, f"{attempt.path}/{COMPLETE}", marker_dir))
                    marker = json.loads(p.read_text())
                if marker.get("format_version") != 1 or marker.get("run_id") != run_id or marker.get("update") != number:
                    raise ValueError("invalid checkpoint completion marker")
                candidates.append((number, int(marker["committed_ns"]), attempt.path, marker))
    if not candidates:
        raise FileNotFoundError("no committed MLflow checkpoint found")
    number, _, artifact, marker = max(candidates, key=lambda item: item[:3])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".checkpoint-download-", dir=destination.parent) as work:
        payload = Path(client.download_artifacts(run_id, artifact, work))
        if verify_checkpoint(payload, number) != marker["manifest_sha256"]:
            raise ValueError("downloaded checkpoint manifest differs")
        if json.loads((payload / COMPLETE).read_text()) != marker:
            raise ValueError("downloaded completion marker differs")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"checkpoint destination exists: {destination}")
        payload.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--update", type=int, help="default: latest committed update")
    args = parser.parse_args()
    path = download_checkpoint(_client(args.tracking_uri), args.run_id, args.destination, args.update)
    print(json.dumps({"checkpoint": str(path), "run_id": args.run_id}))


if __name__ == "__main__":
    main()
