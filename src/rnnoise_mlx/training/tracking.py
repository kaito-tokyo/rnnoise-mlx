"""MLflow tracking for RNNoise-MLX training runs."""

from __future__ import annotations

import atexit
import json
from pathlib import Path
import subprocess
from typing import Any
from urllib.parse import urlparse

import mlflow
from mlflow.tracking import MlflowClient


MAX_PROVENANCE_ARTIFACT_BYTES = 16 * 1024 * 1024


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _flatten_metrics(prefix: str, values: dict[str, Any] | None) -> dict[str, float]:
    if values is None:
        return {}
    metrics = {}
    for key, value in values.items():
        if isinstance(value, bool):
            metrics[f"{prefix}_{key}"] = float(value)
        elif isinstance(value, (int, float)):
            metrics[f"{prefix}_{key}"] = float(value)
    return metrics


def initial_evaluation_from_run(run: Any) -> dict[str, Any] | None:
    """Reconstruct a legacy checkpoint's baseline from its MLflow run."""
    metrics = run.data.metrics
    prefix = "eval_initial_"
    values = {
        key.removeprefix(prefix): value
        for key, value in metrics.items()
        if key.startswith(prefix)
    }
    if not values:
        return None
    required = {
        "total_loss", "gain_loss", "vad_loss", "finite",
        "outputs_in_unit_interval", "batches",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(
            "MLflow run has incomplete initial evaluation metrics: "
            + ", ".join(missing)
        )
    return {
        "total_loss": float(values["total_loss"]),
        "gain_loss": float(values["gain_loss"]),
        "vad_loss": float(values["vad_loss"]),
        "finite": bool(values["finite"]),
        "outputs_in_unit_interval": bool(values["outputs_in_unit_interval"]),
        "batches": int(values["batches"]),
    }


def _git_metadata(root: Path) -> dict[str, str]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=root, text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_dirty": str(bool(git("status", "--porcelain"))).lower(),
    }


def validate_tracking_target(
    tracking_uri: str, experiment: str, run_id: str | None = None
):
    """Verify HTTP(S) server access and, when resuming, the run's experiment."""
    parsed_uri = urlparse(tracking_uri)
    if parsed_uri.scheme not in {"http", "https"} or not parsed_uri.netloc:
        raise ValueError(
            "MLflow tracking URI must be an HTTP(S) server; direct local "
            "file and SQLite tracking are forbidden"
        )
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    try:
        client.search_experiments(max_results=1)
    except Exception as error:
        raise ConnectionError(
            f"MLflow Tracking Server is unavailable at {tracking_uri}; "
            "training aborted without tracking-store fallback"
        ) from error
    if run_id is None:
        return None
    selected_experiment = client.get_experiment_by_name(experiment)
    if selected_experiment is None:
        raise ValueError(f"MLflow experiment does not exist: {experiment}")
    existing_run = client.get_run(run_id)
    if existing_run.info.experiment_id != selected_experiment.experiment_id:
        raise ValueError(
            f"MLflow run {run_id} does not belong to experiment {experiment}"
        )
    return existing_run


class MLflowTracker:
    """Own one HTTP(S)-served MLflow run and finalize it on success or failure."""

    def __init__(
        self,
        tracking_uri: str,
        experiment: str,
        run_name: str | None,
        run_id: str | None,
        output: Path,
        parameters: dict[str, Any],
        provenance_artifacts: list[Path] | None = None,
    ) -> None:
        # Fail before evaluation or training if the server cannot be reached.
        existing_run = validate_tracking_target(tracking_uri, experiment, run_id)
        tags = {
            "job_type": "training",
            **_git_metadata(Path(__file__).resolve().parents[3]),
        }
        if run_id is None:
            mlflow.set_experiment(experiment)
            self.run = mlflow.start_run(run_name=run_name, tags=tags)
        else:
            assert existing_run is not None
            self.run = mlflow.start_run(run_id=run_id)
            mlflow.set_tags({**tags, "resumed": "true"})
        self.closed = False
        normalized = _json_value(
            {key: value for key, value in parameters.items() if key != "mlflow_run_id"}
        )
        if run_id is None:
            mlflow.log_params(normalized)
        else:
            # MLflow parameters are immutable. Preserve the original invocation
            # and only fill parameters absent from an older run.
            existing_params = existing_run.data.params
            mlflow.log_params(
                {
                    key: value
                    for key, value in normalized.items()
                    if key not in existing_params
                }
            )
        atexit.register(self.fail_if_open)
        chapter = int(parameters.get("resume_update", 0))
        chapter_namespace = f"chapter-{chapter:08d}"
        run_config = output / "run-config.json"
        run_config.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
        mlflow.log_artifact(
            str(run_config), artifact_path=f"provenance/{chapter_namespace}"
        )
        self.log_provenance_artifacts(
            provenance_artifacts or [], namespace=chapter_namespace
        )

    @property
    def run_id(self) -> str:
        return self.run.info.run_id

    def log_evaluation(
        self, stage: str, evaluation: dict[str, Any] | None, step: int
    ) -> None:
        metrics = _flatten_metrics(f"eval_{stage}", evaluation)
        if metrics:
            mlflow.log_metrics(metrics, step=step)

    def log_training(
        self,
        update: int,
        epoch: int,
        loss: float,
        learning_rate: float,
        processed_frames: int,
    ) -> None:
        mlflow.log_metrics(
            {
                "train_loss": loss,
                "train_epoch": float(epoch),
                "train_learning_rate": learning_rate,
                "train_processed_frames": float(processed_frames),
            },
            step=update,
        )

    def log_checkpoint(self, checkpoint: Path, update: int) -> None:
        """Upload every complete checkpoint under an immutable update path."""
        mlflow.log_artifacts(
            str(checkpoint), artifact_path=f"checkpoints/{checkpoint.name}"
        )
        mlflow.log_metric("checkpoint_uploaded_update", float(update), step=update)

    def log_provenance_artifacts(
        self, artifacts: list[Path], namespace: str = "chapter-00000000"
    ) -> None:
        """Upload small manifests/configuration, never bulk feature or corpus data."""
        seen: set[Path] = set()
        for index, artifact in enumerate(artifacts):
            artifact = artifact.resolve()
            if artifact in seen:
                continue
            seen.add(artifact)
            if not artifact.is_file():
                raise ValueError(f"provenance artifact is not a file: {artifact}")
            size = artifact.stat().st_size
            if size > MAX_PROVENANCE_ARTIFACT_BYTES:
                raise ValueError(
                    f"provenance artifact exceeds 16 MiB: {artifact} ({size} bytes)"
                )
            mlflow.log_artifact(
                str(artifact),
                artifact_path=f"provenance/data/{namespace}/{index:03d}",
            )

    def complete(self, summary: dict[str, Any], output: Path) -> None:
        update = int(summary["updates"])
        self.log_evaluation("trained", summary.get("trained_evaluation"), update)
        self.log_evaluation("reloaded", summary.get("reloaded_evaluation"), update)
        mlflow.log_metrics(
            {
                "training_seconds": float(summary["training_seconds"]),
                "updates_per_second": float(summary["updates_per_second"]),
                "frames_per_second": float(summary["frames_per_second"]),
                "audio_seconds_per_second": float(summary["audio_seconds_per_second"]),
                "reload_matches": float(bool(summary["reload_matches"])),
            },
            step=update,
        )
        mlflow.log_artifact(str(output / "model.safetensors"), artifact_path="model")
        mlflow.log_artifact(str(output / "model-config.json"), artifact_path="model")
        mlflow.log_artifact(str(output / "training.json"), artifact_path="model")
        mlflow.end_run(status="FINISHED")
        self.closed = True

    def fail_if_open(self) -> None:
        if not self.closed:
            mlflow.end_run(status="FAILED")
            self.closed = True
