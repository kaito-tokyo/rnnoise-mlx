import json

import pytest

from rnnoise_mlx.tools.select_speech_offsets import generate, select_offsets
from rnnoise_mlx.tools.splitmix64 import SplitMix64, uniform_below


def test_splitmix64_known_vector_and_overflow():
    rng = SplitMix64(0)
    assert [rng.next_u64() for _ in range(3)] == [
        0xE220A8397B1DCDaf,
        0x6E789E6AA1B965F4,
        0x06C45D188009454F,
    ]
    assert SplitMix64(1 << 64).next_u64() == SplitMix64(0).next_u64()


def test_uniform_mapping_uses_exactly_one_draw():
    rng = SplitMix64(141)
    assert 0 <= uniform_below(rng, 17) < 17
    assert rng.calls == 1


def test_offsets_are_deterministic_in_range_and_one_draw_each():
    first = select_offsets(1000, 100, 20, 141)
    second = select_offsets(1000, 100, 20, 141)
    assert first == second
    assert all(0 <= offset <= 900 for offset in first)


def test_generate_writes_offsets_and_metadata(tmp_path):
    pcm = tmp_path / "speech.pcm"
    pcm.write_bytes(bytes(2000))
    output = tmp_path / "offsets.txt"
    generate(pcm, output, count=4, seed=141, sequence_samples=100)
    offsets = [int(value) for value in output.read_text().splitlines()]
    metadata = json.loads((tmp_path / "offsets.txt.json").read_text())
    assert offsets == select_offsets(1000, 100, 4, 141)
    assert metadata["algorithm"] == "splitmix64-multiply-high-offset-v1"
    assert metadata["rng_calls"] == 4
    assert metadata["offset_unit"] == "int16_sample"


@pytest.mark.parametrize(
    "args, message",
    [
        ((99, 100, 1, 0), "shorter"),
        ((100, 0, 1, 0), "sequence samples"),
        ((100, 100, 0, 0), "count"),
    ],
)
def test_invalid_selection(args, message):
    with pytest.raises(ValueError, match=message):
        select_offsets(*args)
