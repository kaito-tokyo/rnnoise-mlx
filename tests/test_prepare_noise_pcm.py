import importlib.util
from pathlib import Path
import sys


SPEC = importlib.util.spec_from_file_location(
    "prepare_noise_pcm", Path(__file__).parents[1] / "scripts/prepare_noise_pcm.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_ffconcat_quote_escapes_apostrophes():
    assert MODULE.ffconcat_quote(Path("a'b.wav")) == "file 'a'\\''b.wav'"
