from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from rnnoise_weights import read_bundle, write_bundle


def test_canonical_bundle_supports_official_dimensions(tmp_path: Path):
    cond_size, gru_size = 128, 384
    from rnnoise_weights import shapes

    weights = {
        name: (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / max(1, np.prod(shape))).astype(np.float32)
        for name, shape in shapes(cond_size, gru_size).items()
    }
    path = tmp_path / "official.bnns"
    write_bundle(path, weights, cond_size, gru_size)
    restored, config = read_bundle(path)
    assert config == {"input_dim": 65, "cond_size": 128, "gru_size": 384, "output_dim": 32}
    for name in weights:
        np.testing.assert_array_equal(restored[name], weights[name])


def test_official_pytorch_roundtrip_preserves_inference(tmp_path: Path):
    torch = pytest.importorskip("torch")
    upstream = Path(__file__).parents[1] / "data/rnnoise-upstream"
    checkpoint_path = upstream / "models/rnnoise10Ga_12.pth"
    if not checkpoint_path.exists():
        pytest.skip("official validation checkpoint is not prepared")

    sys.path.insert(0, str(upstream / "torch/rnnoise"))
    from rnnoise import RNNoise
    from convert_official_weights import export_official, import_official

    bundle, restored_path = tmp_path / "official.bnns", tmp_path / "restored.pth"
    import_official(checkpoint_path, bundle)
    export_official(bundle, restored_path)
    original = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = torch.load(restored_path, map_location="cpu", weights_only=False)
    torch.manual_seed(7)
    features = torch.randn(1, 20, 65)

    outputs = []
    for checkpoint in (original, restored):
        model = RNNoise(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            outputs.append(model(features)[:2])
    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=1e-6, rtol=1e-6)


def test_canonical_gru_equations_match_official_pytorch(tmp_path: Path):
    torch = pytest.importorskip("torch")
    upstream = Path(__file__).parents[1] / "data/rnnoise-upstream"
    checkpoint_path = upstream / "models/rnnoise10Ga_12.pth"
    if not checkpoint_path.exists():
        pytest.skip("official validation checkpoint is not prepared")
    sys.path.insert(0, str(upstream / "torch/rnnoise"))
    from rnnoise import RNNoise
    from convert_official_weights import import_official

    bundle = tmp_path / "official.bnns"
    import_official(checkpoint_path, bundle)
    weights, config = read_bundle(bundle)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = RNNoise(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    rng = np.random.default_rng(9)
    features = rng.normal(size=(24, 65)).astype(np.float32)
    with torch.no_grad():
        expected_gain, expected_vad, _ = model(torch.from_numpy(features[None]))

    conv1 = np.tanh(np.stack([
        weights["conv1.weight"].reshape(config["cond_size"], -1) @ features[i:i + 3].reshape(-1)
        + weights["conv1.bias"] for i in range(22)
    ]))
    conv2 = np.tanh(np.stack([
        weights["conv2.weight"].reshape(config["gru_size"], -1) @ conv1[i:i + 3].reshape(-1)
        + weights["conv2.bias"] for i in range(20)
    ]))
    states = [np.zeros(config["gru_size"], dtype=np.float32) for _ in range(3)]
    gains, vads = [], []
    for value in conv2:
        outputs = [value]
        for index, prefix in enumerate(("gru1", "gru2", "gru3")):
            x = weights[f"{prefix}.Wx"] @ value + weights[f"{prefix}.b"]
            h = weights[f"{prefix}.Wh"] @ states[index]
            size = config["gru_size"]
            reset = 1 / (1 + np.exp(-(x[:size] + h[:size])))
            update = 1 / (1 + np.exp(-(x[size:2 * size] + h[size:2 * size])))
            candidate = np.tanh(x[2 * size:] + reset * (h[2 * size:] + weights[f"{prefix}.bhn"]))
            states[index] = (1 - update) * candidate + update * states[index]
            value = states[index]
            outputs.append(value)
        joined = np.concatenate(outputs)
        gains.append(1 / (1 + np.exp(-(weights["gain.weight"] @ joined + weights["gain.bias"]))))
        vads.append(1 / (1 + np.exp(-(weights["vad.weight"] @ joined + weights["vad.bias"]))))
    np.testing.assert_allclose(gains, expected_gain[0].numpy(), atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(vads, expected_vad[0].numpy(), atol=2e-6, rtol=2e-6)
