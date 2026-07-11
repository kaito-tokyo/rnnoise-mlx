import importlib.util
from pathlib import Path
import sys


SPEC = importlib.util.spec_from_file_location(
    "split_noise_manifest", Path(__file__).parents[1] / "scripts/split_noise_manifest.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_split_is_deterministic_and_salt_sensitive():
    first = MODULE.split_for("free-sound/example.wav", 0.5, "a")
    assert first == MODULE.split_for("free-sound/example.wav", 0.5, "a")
    outcomes = {MODULE.split_for("free-sound/example.wav", 0.5, str(i)) for i in range(20)}
    assert outcomes == {"train", "eval"}
