"""PyTorch training, persistence, and backend-boundary checks."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import dataclass, field
from pathlib import Path

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("NumPy is not installed")
import numpy as np
try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("PyTorch is not installed")

from rnnoise_mlx.tools.rnnoise_weights import read_weights
from rnnoise_mlx.training_pytorch import (
    ModelConfig,
    RNNoiseTrainer,
    initial_state,
    train_update,
    RNNoise,
    TrainConfig,
)
from rnnoise_mlx.training_pytorch.train import rnnoise_loss, train
from rnnoise_mlx.training_pytorch.cli import parser
from rnnoise_mlx.training_pytorch.weights import load_weights, save_weights
from rnnoise_mlx.training_tools.data import FeatureDataset


@dataclass(frozen=True)
class TinyConfig(ModelConfig):
    cond_size: int = field(default=4, init=False)
    gru_size: int = field(default=8, init=False)


def inputs(batch=2, frames=6):
    return (
        torch.randn(batch, frames, 65),
        torch.rand(batch, frames, 32),
        torch.rand(batch, frames, 1),
    )


class PyTorchTrainingTests(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temporary_directory.name)
        self._previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def tearDown(self):
        torch.set_num_threads(self._previous_threads)
        self._temporary_directory.cleanup()

    def test_pytorch_import_does_not_load_mlx(self):
        code = """
    import importlib.abc, sys
    class NoMLX(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'mlx' or fullname.startswith('mlx.'):
                raise AssertionError('PyTorch attempted to import MLX')
    sys.meta_path.insert(0, NoMLX())
    import rnnoise_mlx
    assert 'torch' not in sys.modules
    import rnnoise_mlx.training_pytorch.train
    """
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )


    def test_model_accepts_variable_sequences_and_native_gru_state(self):
        for batch, frames in ((1, 1), (2, 3), (3, 9)):
            model = RNNoise(TinyConfig())
            features = torch.randn(batch, frames + 4, 65)
            gain, vad, state = model(features)
            assert gain.shape == (batch, frames, 32)
            assert vad.shape == (batch, frames, 1)
            assert all(s.shape == (1, batch, 8) for s in state)
            model(features, state)


    def test_model_does_not_truncate_recurrent_gradients(self):
        model = RNNoise(TinyConfig())
        initial = tuple(torch.randn(1, 2, 8, requires_grad=True) for _ in range(3))
        _, _, carry = model(torch.randn(2, 7, 65), initial)
        gain, vad, _ = model(torch.randn(2, 9, 65), carry)
        (gain.sum() + vad.sum()).backward()
        assert all(s.grad is not None and s.grad.abs().sum() > 0 for s in initial)


    def test_tbptt_accumulates_mean_gradient_and_steps_once(self):
        model = RNNoise(TinyConfig())
        settings = TrainConfig(2, 3)
        reference = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loop = RNNoiseTrainer(model, optimizer, settings)
        assert loop.train_config is settings
        batch = inputs()
        initial = tuple(s.requires_grad_() for s in loop.initial_state())
        calls = []
        optimizer.register_step_post_hook(lambda *args: calls.append(1))
        loss, state = train_update(model, optimizer, settings, *batch, state=initial)

        padded = torch.cat((batch[0].new_zeros(2, 4, 65), batch[0]), dim=1)
        first_gain, first_vad, carry = reference(
            padded[:, :7], tuple(s.detach() for s in initial)
        )
        second_gain, second_vad, _ = reference(
            padded[:, 3:10], tuple(s.detach() for s in carry)
        )
        first_loss = rnnoise_loss(
            first_gain, first_vad, batch[1][:, :3], batch[2][:, :3], gamma=settings.gamma
        )
        second_loss = rnnoise_loss(
            second_gain, second_vad, batch[1][:, 3:], batch[2][:, 3:], gamma=settings.gamma
        )
        expected = (first_loss + second_loss) / 2
        expected.backward()
        torch.testing.assert_close(loss, expected.detach())
        for actual, before in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, before.grad)
            torch.testing.assert_close(actual, before - 0.01 * before.grad)
        assert calls == [1]
        assert all(s.grad is None for s in initial)
        assert all(s.grad_fn is None and not s.requires_grad for s in state)
        loop.run_update(*batch, state=state)
        assert calls == [1, 1]


    def test_invalid_update_does_not_change_weights(self):
        for frames in (5, 0):
            model = RNNoise(TinyConfig())
            before = copy.deepcopy(model.state_dict())
            loop = RNNoiseTrainer(
                model, torch.optim.Adam(model.parameters()), TrainConfig(2, 3)
            )
            with self.assertRaises(ValueError):
                loop.run_update(*inputs(frames=frames))
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


    def test_loop_validates_training_configuration(self):
        for settings in (TrainConfig(0, 3), TrainConfig(2, 0), TrainConfig(2, 3, 0)):
            model = RNNoise(TinyConfig())
            with self.assertRaises(ValueError):
                RNNoiseTrainer(model, torch.optim.Adam(model.parameters()), settings)


    def test_exported_weights_preserve_sequence_outputs(self):
        config = TinyConfig()
        model = RNNoise(config).eval()
        path = self.tmp_path / "model.safetensors"
        save_weights(model, path)
        restored = RNNoise(config).eval()
        load_weights(restored, path)
        features = torch.randn(2, 11, 65)
        state = tuple(torch.randn(1, 2, config.gru_size) for _ in range(3))
        with torch.no_grad():
            expected_gain, expected_vad, expected_state = model(features, state)
            gain, vad, carry = restored(features, state)
        torch.testing.assert_close(gain, expected_gain)
        torch.testing.assert_close(vad, expected_vad)
        for actual, expected in zip(carry, expected_state):
            torch.testing.assert_close(actual, expected)


    def test_shared_config_and_mlx_batches_preserve_values(self):
        module = importlib.import_module("rnnoise_mlx.training.config")
        assert module.ModelConfig is ModelConfig
        assert module.TrainConfig is TrainConfig
        config = ModelConfig()
        assert (config.input_dim, config.cond_size, config.gru_size, config.output_dim) == (
            65,
            128,
            384,
            32,
        )
        with self.assertRaises(TypeError):
            ModelConfig(gru_size=256)
        adapter = importlib.import_module("rnnoise_mlx.training.data").FeatureDataset
        path = self.tmp_path / "features.npy"
        data = np.arange(4 * 7 * 98, dtype=np.float32).reshape(4, 7, 98)
        np.save(path, data)
        for chunk_length in (None, 5):
            expected = FeatureDataset(str(path), 7).batches(
                2, np.random.default_rng(0), chunk_length
            )
            actual = adapter(str(path), 7).batches(
                2, np.random.default_rng(0), chunk_length
            )
            for numpy_batch, mlx_batch in zip(expected, actual):
                for a, b in zip(numpy_batch, mlx_batch):
                    np.testing.assert_array_equal(a, np.array(b))


    def test_feature_formats_select_the_same_batches(self):
        data = np.random.default_rng(0).normal(size=(4, 6, 98)).astype(np.float32)
        np.save(self.tmp_path / "features.npy", data)
        data.tofile(self.tmp_path / "features.f32")
        a = FeatureDataset(str(self.tmp_path / "features.npy"), 6)
        b = FeatureDataset(str(self.tmp_path / "features.f32"), 6)
        assert isinstance(a.data, np.memmap)
        for aa, bb in zip(
            a.batches(2, np.random.default_rng(7)), b.batches(2, np.random.default_rng(7))
        ):
            for x, y in zip(aa, bb):
                np.testing.assert_array_equal(x, y)


    def test_training_resume_matches_uninterrupted_run(self):
        import rnnoise_mlx.training_pytorch.train as module

        monkeypatch.setattr(module, "ModelConfig", TinyConfig)
        data = np.random.default_rng(1).random((4, 6, 98), dtype=np.float32)
        features = self.tmp_path / "features.npy"
        np.save(features, data)

        def run(name, count, extra=()):
            args = parser().parse_args(
                [
                    str(features),
                    str(self.tmp_path / name),
                    "--device",
                    "cpu",
                    "--batch-size",
                    "2",
                    "--sequence-length",
                    "6",
                    "--segmented-tbptt-length",
                    "3",
                    "--max-updates",
                    str(count),
                    "--checkpoint-every",
                    "1",
                    *extra,
                ]
            )
            return train(args)

        full = run("full", 3)
        run("first", 1)
        resumed = run(
            "resumed",
            3,
            ["--resume-from", str(self.tmp_path / "first/update-00000001/checkpoint.pt")],
        )
        assert resumed["losses"] == full["losses"]
        legacy_path = self.tmp_path / "legacy.pt"
        legacy = torch.load(
            self.tmp_path / "first/update-00000001/checkpoint.pt", weights_only=True
        )
        legacy["format_version"] = 1
        legacy["signature"]["chunk"] = legacy["signature"].pop("training")
        legacy["model"] = {
            "frame_step." + key: value for key, value in legacy["model"].items()
        }
        legacy["carry"] = tuple(value.squeeze(0) for value in legacy["carry"])
        torch.save(legacy, legacy_path)
        migrated = run("migrated", 3, ["--resume-from", str(legacy_path)])
        assert migrated["losses"] == full["losses"]
        a = torch.load(self.tmp_path / "full/update-00000003/checkpoint.pt", weights_only=True)
        b = torch.load(
            self.tmp_path / "resumed/update-00000003/checkpoint.pt", weights_only=True
        )
        for name in a["model"]:
            torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
        fresh = run(
            "finetune", 1, ["--init-weights", str(self.tmp_path / "full/model.safetensors")]
        )
        assert fresh["updates"] == 1
        assert (
            json.loads((self.tmp_path / "finetune/training_summary.json").read_text())["updates"]
            == 1
        )
        with self.assertRaises(FileExistsError):
            run("full", 1)


    def test_chunk_loss_and_gradients_match_mlx(self):
        import mlx.core as mx

        # Initialize the existing compatibility package before training_cuda.
        importlib.import_module("rnnoise_mlx.training.config")
        from rnnoise_mlx.training_cuda.graph import RNNoiseChunk as MLXChunk

        config, settings = TinyConfig(), TrainConfig(2, 3)
        pytorch = RNNoise(config)
        path = self.tmp_path / "parity.safetensors"
        save_weights(pytorch, path)
        weights, _ = read_weights(path)
        mlx = MLXChunk(config, settings)
        mlx.load_weights(
            [
                ("frame_step." + name.replace("gain.", "dense_out."), mx.array(value))
                for name, value in weights.items()
            ]
        )
        features, gain, vad = inputs(frames=3)
        features = torch.cat((torch.zeros(2, 4, 65), features), dim=1)
        state = tuple(torch.randn(2, config.gru_size) for _ in range(3))
        predicted_gain, predicted_vad, carry = pytorch(
            features, tuple(s.unsqueeze(0) for s in state)
        )
        loss = rnnoise_loss(predicted_gain, predicted_vad, gain, vad, gamma=settings.gamma)
        loss.backward()
        (mlx_loss, mlx_carry), grads = mlx.value_and_grad(
            mx.array(features.numpy()),
            mx.array(gain.numpy()),
            mx.array(vad.numpy()),
            tuple(mx.array(s.numpy()) for s in state),
        )
        mx.eval(mlx_loss, mlx_carry, grads)
        np.testing.assert_allclose(
            loss.detach().numpy(), np.array(mlx_loss), rtol=2e-5, atol=2e-6
        )
        for x, y in zip(carry, mlx_carry):
            np.testing.assert_allclose(
                x.squeeze(0).detach().numpy(), np.array(y), rtol=2e-5, atol=2e-6
            )
        for i in range(1, 4):
            gru = getattr(pytorch, f"gru{i}")
            for torch_name, mlx_name in (
                ("weight_ih_l0", "Wx"),
                ("weight_hh_l0", "Wh"),
                ("bias_ih_l0", "b"),
            ):
                np.testing.assert_allclose(
                    getattr(gru, torch_name).grad.numpy(),
                    np.array(grads["frame_step"][f"gru{i}"][mlx_name]),
                    rtol=3e-4,
                    atol=3e-6,
                )

