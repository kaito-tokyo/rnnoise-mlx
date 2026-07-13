"""MLflow tracking for RNNoise-MLX training runs."""

from __future__ import annotations

import atexit
from pathlib import Path
import subprocess
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


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


class MLflowTracker:
    """Own one remote MLflow run and finalize it on success or failure."""

    def __init__(
        self,
        tracking_uri: str,
        experiment: str,
        run_name: str,
        output: Path,
        parameters: dict[str, Any],
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        # Fail before evaluation or training if the server cannot be reached.
        MlflowClient().search_experiments(max_results=1)
        mlflow.set_experiment(experiment)
        self.run = mlflow.start_run(
            run_name=run_name,
            tags={
                "job_type": "training",
                **_git_metadata(Path(__file__).resolve().parents[3]),
            },
        )
        self.closed = False
        normalized = {
            key: str(value) if isinstance(value, Path) or value is None else value
            for key, value in parameters.items()
        }
        mlflow.log_params(normalized)
        atexit.register(self.fail_if_open)

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
        mlflow.log_artifact(str(output / "training.json"), artifact_path="model")
        mlflow.end_run(status="FINISHED")
        self.closed = True

    def fail_if_open(self) -> None:
        if not self.closed:
            mlflow.end_run(status="FAILED")
            self.closed = True
