import hashlib
import json
import os
import importlib.util
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

if importlib.util.find_spec("mlx") is None:
    raise unittest.SkipTest("MLX is not installed")

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from rnnoise_mlx.training.checkpoint import load_checkpoint, save_checkpoint
from rnnoise_mlx.training.model import ModelConfig, RNNoise


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
    left_flat = dict(tree_flatten(left)); right_flat = dict(tree_flatten(right))
    assert left_flat.keys() == right_flat.keys()
    for key in left_flat:
        assert mx.array_equal(left_flat[key], right_flat[key]).item(), key


def _feature_parameters(tmp_path, content=b"features"):
    feature = tmp_path / "train.f32"
    feature.write_bytes(content)
    (tmp_path / "train.manifest.json").write_text(json.dumps({"output": {"filename": feature.name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}}) + "\n")
    return {"features": str(feature), "batch_size": 2, "sequence_length": 10}


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self._temporary_directory=tempfile.TemporaryDirectory(); self.tmp_path=Path(self._temporary_directory.name)
    def tearDown(self):
        self._temporary_directory.cleanup()

    def test_complete_checkpoint_round_trip(self):
        config, model, optimizer = _updated_model_and_optimizer()
        parameters = _feature_parameters(self.tmp_path)
        checkpoint = save_checkpoint(
            self.tmp_path,
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
            initial_evaluation={"loss": 0.75},
            feature_identity=hashlib.sha256(b"features").hexdigest(),
            evaluation_feature_identity=None,
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
        assert state["initial_evaluation"] == {"loss": 0.75}
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        assert manifest["feature_identity"] == hashlib.sha256(b"features").hexdigest()
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


    def test_resumed_next_update_matches_uninterrupted_training(self):
        config, uninterrupted_model, uninterrupted_optimizer = (
            _updated_model_and_optimizer()
        )
        parameters = _feature_parameters(self.tmp_path)
        checkpoint = save_checkpoint(
            self.tmp_path,
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


    def test_checkpoint_rejects_incompatible_training_configuration(self):
        config, model, optimizer = _updated_model_and_optimizer()
        parameters = {"features": "train.f32", "batch_size": 2, "sequence_length": 10}
        checkpoint = save_checkpoint(
            self.tmp_path,
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


    def test_checkpoint_allows_feature_and_tbptt_chapter_changes(self):
        config, model, optimizer = _updated_model_and_optimizer()
        parameters = {
            "features": "generation-0/train.f32",
            "batch_size": 2,
            "sequence_length": 10,
            "segmented_tbptt_length": 250,
            "segmented_tbptt_state": "carry",
        }
        checkpoint = save_checkpoint(
            self.tmp_path,
            model,
            optimizer,
            config,
            update=5_000,
            next_epoch=5,
            next_batch=0,
            processed_frames=100_000,
            elapsed_seconds=1.0,
            history=[],
            training_config=parameters,
        )
        changed_chapter = dict(
            parameters,
            features="generation-1/train.f32",
            segmented_tbptt_length=500,
        )

        state = load_checkpoint(
            checkpoint,
            RNNoise(config),
            optim.AdamW(learning_rate=1e-3),
            config,
            changed_chapter,
        )

        assert state["update"] == 5_000
        assert state["next_batch"] == 0


    def test_checkpoint_resets_nonzero_batch_for_new_features(self):
        config, model, optimizer = _updated_model_and_optimizer()
        parameters = {
            "features": "generation-0/train.f32",
            "batch_size": 2,
            "sequence_length": 10,
        }
        checkpoint = save_checkpoint(
            self.tmp_path,
            model,
            optimizer,
            config,
            update=5,
            next_epoch=2,
            next_batch=7,
            processed_frames=100,
            elapsed_seconds=1.0,
            history=[],
            training_config=parameters,
        )
        state = load_checkpoint(
            checkpoint,
            RNNoise(config),
            optim.AdamW(learning_rate=1e-3),
            config,
            dict(parameters, features="generation-1/train.f32"),
        )
        assert state["next_epoch"] == 2
        assert state["next_batch"] == 0


    def test_checkpoint_resets_batch_when_same_path_feature_content_changes(self):
        config, model, optimizer = _updated_model_and_optimizer()
        parameters = _feature_parameters(self.tmp_path, b"generation-zero")
        checkpoint = save_checkpoint(
            self.tmp_path / "checkpoints",
            model,
            optimizer,
            config,
            update=5,
            next_epoch=2,
            next_batch=7,
            processed_frames=100,
            elapsed_seconds=1.0,
            history=[],
            training_config=parameters,
        )
        _feature_parameters(self.tmp_path, b"generation-one")
        state = load_checkpoint(
            checkpoint,
            RNNoise(config),
            optim.AdamW(learning_rate=1e-3),
            config,
            parameters,
        )
        assert state["next_batch"] == 0


    def test_checkpoint_rejects_stale_manifest_for_same_size_feature(self):
        config, model, optimizer = _updated_model_and_optimizer()
        parameters = _feature_parameters(self.tmp_path, b"old-bytes")
        checkpoint = save_checkpoint(
            self.tmp_path / "checkpoints",
            model,
            optimizer,
            config,
            update=5,
            next_epoch=2,
            next_batch=7,
            processed_frames=100,
            elapsed_seconds=1.0,
            history=[],
            training_config=parameters,
        )
        Path(parameters["features"]).write_bytes(b"new-bytes")
        state = load_checkpoint(
            checkpoint,
            RNNoise(config),
            optim.AdamW(learning_rate=1e-3),
            config,
            parameters,
        )
        assert state["next_batch"] == 0


    def test_checkpoint_rejects_changed_evaluation_content_and_gamma(self):
        config, model, optimizer = _updated_model_and_optimizer()
        parameters = _feature_parameters(self.tmp_path)
        eval_dir = self.tmp_path / "eval"
        eval_dir.mkdir()
        eval_parameters = _feature_parameters(eval_dir, b"eval-zero")
        parameters["eval_features"] = eval_parameters["features"]
        parameters["gamma"] = 0.25
        checkpoint = save_checkpoint(
            self.tmp_path / "checkpoints",
            model,
            optimizer,
            config,
            update=5,
            next_epoch=2,
            next_batch=7,
            processed_frames=100,
            elapsed_seconds=1.0,
            history=[],
            training_config=parameters,
        )
        _feature_parameters(eval_dir, b"eval-one")
        try:
            load_checkpoint(
                checkpoint,
                RNNoise(config),
                optim.AdamW(learning_rate=1e-3),
                config,
                parameters,
            )
        except ValueError as error:
            assert "evaluation features differ" in str(error)
        else:
            raise AssertionError("changed evaluation content was accepted")

        changed_gamma = dict(parameters, gamma=0.5)
        try:
            load_checkpoint(
                checkpoint,
                RNNoise(config),
                optim.AdamW(learning_rate=1e-3),
                config,
                changed_gamma,
            )
        except ValueError as error:
            assert "gamma" in str(error)
        else:
            raise AssertionError("changed evaluation gamma was accepted")


    def test_legacy_checkpoint_without_feature_identity_resets_batch(self):
        config, model, optimizer = _updated_model_and_optimizer()
        parameters = _feature_parameters(self.tmp_path)
        checkpoint = save_checkpoint(
            self.tmp_path / "checkpoints",
            model,
            optimizer,
            config,
            update=5,
            next_epoch=2,
            next_batch=7,
            processed_frames=100,
            elapsed_seconds=1.0,
            history=[],
            training_config=parameters,
        )
        manifest_path = checkpoint / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        del manifest["feature_identity"]
        manifest_path.write_text(json.dumps(manifest) + "\n")
        state = load_checkpoint(
            checkpoint,
            RNNoise(config),
            optim.AdamW(learning_rate=1e-3),
            config,
            parameters,
        )
        assert state["next_batch"] == 0


    def test_legacy_checkpoint_without_evaluation_identity_is_rejected(self):
        config, model, optimizer = _updated_model_and_optimizer()
        parameters = _feature_parameters(self.tmp_path)
        eval_dir = self.tmp_path / "eval"
        eval_dir.mkdir()
        eval_parameters = _feature_parameters(eval_dir, b"eval")
        parameters["eval_features"] = eval_parameters["features"]
        checkpoint = save_checkpoint(
            self.tmp_path / "checkpoints",
            model,
            optimizer,
            config,
            update=5,
            next_epoch=2,
            next_batch=7,
            processed_frames=100,
            elapsed_seconds=1.0,
            history=[],
            training_config=parameters,
        )
        manifest_path = checkpoint / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        del manifest["evaluation_feature_identity"]
        manifest_path.write_text(json.dumps(manifest) + "\n")
        try:
            load_checkpoint(
                checkpoint,
                RNNoise(config),
                optim.AdamW(learning_rate=1e-3),
                config,
                parameters,
            )
        except ValueError as error:
            assert "evaluation features differ" in str(error)
        else:
            raise AssertionError("unverifiable legacy evaluation was accepted")


    def test_checkpoint_manifest_is_json_serializable_with_paths(self):
        config, model, optimizer = _updated_model_and_optimizer()
        checkpoint = save_checkpoint(
            self.tmp_path,
            model,
            optimizer,
            config,
            update=1,
            next_epoch=1,
            next_batch=1,
            processed_frames=20,
            elapsed_seconds=1.0,
            history=[],
            training_config={"features": self.tmp_path / "train.f32"},
        )
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        assert manifest["training_config"]["features"].endswith("train.f32")


    def test_checkpoint_rejects_corrupt_tensor(self):
        config, model, optimizer = _updated_model_and_optimizer()
        parameters = {"features": "train.f32", "batch_size": 2, "sequence_length": 10}
        checkpoint = save_checkpoint(
            self.tmp_path,
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
