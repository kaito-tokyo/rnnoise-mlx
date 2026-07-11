from pathlib import Path

from training.scripts.prepare_base_dataset import partition, stable_order, stable_score


def test_stable_score_is_deterministic():
    path = Path("musan/noise/example.wav")
    assert stable_score(path, "background") == stable_score(path, "background")
    assert stable_score(path, "background") != stable_score(path, "foreground")


def test_partition_is_order_independent():
    paths = [Path(f"audio/{index}.wav") for index in range(100)]
    first = partition(paths, "noise", 10)
    reversed_result = partition(list(reversed(paths)), "noise", 10)
    assert set(first[0]) == set(reversed_result[0])
    assert set(first[1]) == set(reversed_result[1])
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[0]) | set(first[1]) == set(paths)


def test_stable_order_is_input_order_independent():
    base = Path("corpus")
    paths = [base / f"speaker/{index}.wav" for index in range(20)]
    assert stable_order(paths, "speech", base) == stable_order(list(reversed(paths)), "speech", base)
