import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from rnnoise_mlx.training.checkpoint import load_checkpoint, save_checkpoint
from rnnoise_mlx.training.model import ModelConfig, RNNoise
from rnnoise_mlx.training.tracking import MLflowTracker


def _updated_model_and_optimizer():
    mx.random.seed(11)
    config = ModelConfig()
    model = RNNoise(config)
    optimizer = optim.AdamW(learning_rate=1e-3, betas=(0.8, 0.98), eps=1e-8)

    def objective(model, features):
        gain, vad, _ = model(features)
        return mx.mean(gain) + mx.mean(vad)

    loss_and_grad = nn.value_and_grad(model, objective)
    loss, gradients = loss_and_grad(model, mx.random.normal((2, 10, 65)))
    optimizer.update(model, gradients)
    mx.eval(loss, model.state, optimizer.state)
    return config, model, optimizer


def _update(model, optimizer, features):
    def objective(model, batch):
        gain, vad, _ = model(batch)
        return mx.mean(gain) + mx.mean(vad)

    loss_and_grad = nn.value_and_grad(model, objective)
    loss, gradients = loss_and_grad(model, features)
    optimizer.update(model, gradients)
    mx.eval(loss, model.state, optimizer.state)


def _assert_tree_equal(left, right):
    left_flat = dict(tree_flatten(left))
    right_flat = dict(tree_flatten(right))
    assert left_flat.keys() == right_flat.keys()
    for key in left_flat:
        assert mx.array_equal(left_flat[key], right_flat[key]).item(), key


def test_complete_checkpoint_round_trip(tmp_path):
    config, model, optimizer = _updated_model_and_optimizer()
    parameters = {"features": "train.f32", "batch_size": 2, "sequence_length": 10}
    checkpoint = save_checkpoint(
        tmp_path,
        model,
        optimizer,
        config,
        update=32,
        next_epoch=2,
        next_batch=7,
        processed_frames=640,
        elapsed_seconds=12.5,
        history=[{"update": 32, "epoch": 1, "loss": 0.25}],
        training_config=parameters,
    )

    restored_model = RNNoise(config)
    restored_optimizer = optim.AdamW(
        learning_rate=1e-3, betas=(0.8, 0.98), eps=1e-8
    )
    state = load_checkpoint(
        checkpoint, restored_model, restored_optimizer, config, parameters
    )

    assert state["update"] == 32
    assert state["next_epoch"] == 2
    assert state["next_batch"] == 7
    assert {path.name for path in checkpoint.iterdir()} == {
        "manifest.json",
        "mlx-random-state.safetensors",
        "model.safetensors",
        "optimizer.safetensors",
        "trainer-state.json",
    }
    original = dict(tree_flatten(model.parameters()))
    restored = dict(tree_flatten(restored_model.parameters()))
    assert original.keys() == restored.keys()
    for key in original:
        assert mx.array_equal(original[key], restored[key]).item()
    original_optimizer = dict(tree_flatten(optimizer.state))
    restored_optimizer_state = dict(tree_flatten(restored_optimizer.state))
    assert original_optimizer.keys() == restored_optimizer_state.keys()
    for key in original_optimizer:
        assert mx.array_equal(
            original_optimizer[key], restored_optimizer_state[key]
        ).item()


def test_resumed_next_update_matches_uninterrupted_training(tmp_path):
    config, uninterrupted_model, uninterrupted_optimizer = (
        _updated_model_and_optimizer()
    )
    parameters = {"features": "train.f32", "batch_size": 2, "sequence_length": 10}
    checkpoint = save_checkpoint(
        tmp_path,
        uninterrupted_model,
        uninterrupted_optimizer,
        config,
        update=1,
        next_epoch=1,
        next_batch=1,
        processed_frames=20,
        elapsed_seconds=1.0,
        history=[{"update": 1, "epoch": 1, "loss": 0.5}],
        training_config=parameters,
    )

    uninterrupted_features = mx.random.normal((2, 10, 65))
    mx.eval(uninterrupted_features)
    _update(uninterrupted_model, uninterrupted_optimizer, uninterrupted_features)

    resumed_model = RNNoise(config)
    resumed_optimizer = optim.AdamW(
        learning_rate=1e-3, betas=(0.8, 0.98), eps=1e-8
    )
    state = load_checkpoint(
        checkpoint, resumed_model, resumed_optimizer, config, parameters
    )
    resumed_features = mx.random.normal((2, 10, 65))
    mx.eval(resumed_features)
    _update(resumed_model, resumed_optimizer, resumed_features)

    assert state["update"] == 1
    assert state["next_epoch"] == 1
    assert state["next_batch"] == 1
    assert mx.array_equal(uninterrupted_features, resumed_features).item()
    _assert_tree_equal(uninterrupted_model.parameters(), resumed_model.parameters())
    _assert_tree_equal(uninterrupted_optimizer.state, resumed_optimizer.state)


def test_checkpoint_rejects_incompatible_training_configuration(tmp_path):
    config, model, optimizer = _updated_model_and_optimizer()
    parameters = {"features": "train.f32", "batch_size": 2, "sequence_length": 10}
    checkpoint = save_checkpoint(
        tmp_path,
        model,
        optimizer,
        config,
        update=1,
        next_epoch=1,
        next_batch=1,
        processed_frames=20,
        elapsed_seconds=1.0,
        history=[],
        training_config=parameters,
    )
    incompatible = dict(parameters, batch_size=4)
    try:
        load_checkpoint(
            checkpoint,
            RNNoise(config),
            optim.AdamW(learning_rate=1e-3),
            config,
            incompatible,
        )
    except ValueError as error:
        assert "batch_size" in str(error)
    else:
        raise AssertionError("incompatible checkpoint was accepted")


def test_checkpoint_manifest_is_json_serializable_with_paths(tmp_path):
    config, model, optimizer = _updated_model_and_optimizer()
    checkpoint = save_checkpoint(
        tmp_path,
        model,
        optimizer,
        config,
        update=1,
        next_epoch=1,
        next_batch=1,
        processed_frames=20,
        elapsed_seconds=1.0,
        history=[],
        training_config={"features": tmp_path / "train.f32"},
    )
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert manifest["training_config"]["features"].endswith("train.f32")


def test_checkpoint_rejects_corrupt_tensor(tmp_path):
    config, model, optimizer = _updated_model_and_optimizer()
    parameters = {"features": "train.f32", "batch_size": 2, "sequence_length": 10}
    checkpoint = save_checkpoint(
        tmp_path,
        model,
        optimizer,
        config,
        update=1,
        next_epoch=1,
        next_batch=1,
        processed_frames=20,
        elapsed_seconds=1.0,
        history=[],
        training_config=parameters,
    )
    with (checkpoint / "optimizer.safetensors").open("ab") as output:
        output.write(b"corrupt")
    try:
        load_checkpoint(
            checkpoint,
            RNNoise(config),
            optim.AdamW(learning_rate=1e-3),
            config,
            parameters,
        )
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("corrupt checkpoint was accepted")


def test_mlflow_uploads_checkpoint_under_immutable_update_path(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "rnnoise_mlx.training.tracking.mlflow.log_artifacts",
        lambda local, artifact_path: calls.append((local, artifact_path)),
    )
    monkeypatch.setattr(
        "rnnoise_mlx.training.tracking.mlflow.log_metric",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    tracker = object.__new__(MLflowTracker)
    checkpoint = tmp_path / "update-00000500"
    checkpoint.mkdir()
    tracker.log_checkpoint(checkpoint, 500)
    assert calls[0] == (str(checkpoint), "checkpoints/update-00000500")
    assert calls[1] == (
        ("checkpoint_uploaded_update", 500.0),
        {"step": 500},
    )
