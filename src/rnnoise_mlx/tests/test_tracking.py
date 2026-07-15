import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "rnnoise_mlx_tracking",
    Path(__file__).parents[1] / "training" / "tracking.py",
)
assert _SPEC is not None and _SPEC.loader is not None
tracking = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tracking)


def _mock_mlflow(monkeypatch, *, experiment_id="experiment-1"):
    calls = []
    existing = SimpleNamespace(
        info=SimpleNamespace(run_id="run-123", experiment_id=experiment_id),
        data=SimpleNamespace(params={"batch_size": "8"}),
    )
    client = SimpleNamespace(
        search_experiments=lambda max_results: [],
        get_experiment_by_name=lambda name: SimpleNamespace(
            experiment_id="experiment-1"
        ),
        get_run=lambda run_id: existing,
    )
    monkeypatch.setattr(tracking, "MlflowClient", lambda: client)
    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        tracking.mlflow,
        "start_run",
        lambda **kwargs: calls.append(("start_run", kwargs)) or existing,
    )
    monkeypatch.setattr(
        tracking.mlflow,
        "set_tags",
        lambda tags: calls.append(("set_tags", tags)),
    )
    monkeypatch.setattr(
        tracking.mlflow,
        "log_params",
        lambda params: calls.append(("log_params", params)),
    )
    monkeypatch.setattr(tracking.atexit, "register", lambda fn: None)
    return calls


def test_tracker_reuses_existing_run_and_retains_name(tmp_path, monkeypatch):
    calls = _mock_mlflow(monkeypatch)

    tracker = tracking.MLflowTracker(
        "http://mlflow.test",
        "rnnoise-mlx",
        None,
        "run-123",
        tmp_path,
        {
            "batch_size": 8,
            "resume_from": tmp_path / "checkpoint",
            "mlflow_run_id": "run-123",
        },
    )

    assert tracker.run_id == "run-123"
    assert ("start_run", {"run_id": "run-123"}) in calls
    assert ("log_params", {"resume_from": str(tmp_path / "checkpoint")}) in calls


def test_validate_tracking_target_does_not_start_or_update_run(monkeypatch):
    calls = _mock_mlflow(monkeypatch)

    existing = tracking.validate_tracking_target(
        "http://mlflow.test", "rnnoise-mlx", "run-123"
    )

    assert existing.info.run_id == "run-123"
    assert calls == []


def test_tracker_rejects_run_from_another_experiment(tmp_path, monkeypatch):
    _mock_mlflow(monkeypatch, experiment_id="experiment-2")

    with pytest.raises(ValueError, match="does not belong"):
        tracking.MLflowTracker(
            "http://mlflow.test",
            "rnnoise-mlx",
            None,
            "run-123",
            tmp_path,
            {},
        )
