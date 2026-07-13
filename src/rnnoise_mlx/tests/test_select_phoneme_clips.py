import json

import pytest

from rnnoise_mlx.tools.select_phoneme_clips import (
    SelectionError,
    SplitMix64,
    load_population,
    load_required_phones,
    select,
    shuffle,
    write_result,
)


def write_population(path, clips):
    path.write_text("".join(json.dumps(clip, ensure_ascii=False) + "\n" for clip in clips))


def population():
    return [
        {"path": "a.wav", "phones": ["s", "a"]},
        {"path": "b.wav", "phones": ["θ", "a"]},
        {"path": "c.wav", "phones": ["ɕ", "i"]},
        {"path": "d.wav", "phones": ["ʃ", "sː", "sʰ"]},
    ]


def test_splitmix64_known_vector_and_overflow():
    rng = SplitMix64(0)
    assert [rng.next_u64() for _ in range(3)] == [
        0xE220A8397B1DCDAF,
        0x6E789E6AA1B965F4,
        0x06C45D188009454F,
    ]
    assert SplitMix64(1 << 64).next_u64() == SplitMix64(0).next_u64()


def test_shuffle_uses_exactly_n_minus_one_calls_and_is_deterministic():
    first, calls = shuffle(population(), 141)
    second, second_calls = shuffle(population(), 141)
    assert first == second
    assert calls == second_calls == len(population()) - 1


def test_population_input_order_does_not_change_selection(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    write_population(first_path, population())
    write_population(second_path, list(reversed(population())))
    first = load_population(first_path)
    second = load_population(second_path)
    assert select(first, ["s", "θ"], 3, 10, 20) == select(second, ["s", "θ"], 3, 10, 20)


def test_rejects_whole_attempt_and_accepts_first_covering_seed():
    clips, metadata = select(population(), ["s", "θ", "ʃ"], 3, 0, 100)
    assert set(metadata["required_phones"]) <= {
        phone for clip in clips for phone in clip["phones"]
    }
    assert metadata["accepted_attempt"] == len(metadata["rejected_attempts"])
    assert all(item["missing_phones"] for item in metadata["rejected_attempts"])


def test_output_is_byte_for_byte_deterministic(tmp_path):
    clips, metadata = select(population(), ["s", "θ"], 3, 141, 20)
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_result(first, clips, metadata)
    write_result(second, clips, metadata)
    assert (first / "clips.jsonl").read_bytes() == (second / "clips.jsonl").read_bytes()
    assert (first / "selection.json").read_bytes() == (second / "selection.json").read_bytes()


def test_nfc_normalization_and_literal_phone_distinctions(tmp_path):
    source = tmp_path / "population.jsonl"
    write_population(source, [{"path": "x.wav", "phones": ["s", "θ", "ɕ", "ʃ", "sː", "sʰ", "a\u0303"]}])
    loaded = load_population(source)[0]["phones"]
    assert set(loaded) == {"s", "θ", "ɕ", "ʃ", "sː", "sʰ", "ã"}


@pytest.mark.parametrize(
    "clips, message",
    [
        ([], "population is empty"),
        ([{"path": "a.wav", "phones": []}], "phones must be a non-empty array"),
        ([{"path": "a.wav", "phones": ["s"]}, {"path": "a.wav", "phones": ["θ"]}], "duplicate population path"),
    ],
)
def test_invalid_population(tmp_path, clips, message):
    source = tmp_path / "population.jsonl"
    write_population(source, clips)
    with pytest.raises(SelectionError, match=message):
        load_population(source)


def test_invalid_json_and_required_phones(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text("not json\n")
    with pytest.raises(SelectionError, match="invalid JSON"):
        load_population(source)
    required = tmp_path / "required.json"
    required.write_text("{}")
    with pytest.raises(SelectionError, match="non-empty JSON array"):
        load_required_phones(required)


def test_impossible_constraints_and_attempt_limit():
    with pytest.raises(SelectionError, match="population lacks required phones: q"):
        select(population(), ["q"], 2, 0, 10)
    with pytest.raises(SelectionError, match="sample count"):
        select(population(), ["s"], 0, 0, 10)
    with pytest.raises(SelectionError, match="max attempts"):
        select(population(), ["s"], 1, 0, 0)
    with pytest.raises(SelectionError, match="no sample covered"):
        select(population(), ["s", "θ", "ɕ", "ʃ"], 1, 0, 3)
