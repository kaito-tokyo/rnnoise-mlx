import copy
import pytest

from rnnoise_mlx.tools.validate_corpus_acquisition_plan import validate


def load_plan():
    official = [
        {
            "id": f"language-{index}",
            "targets": {"E": 1, "B": 1, "C": 1},
            "archives": [f"https://example.com/language-{index}.tar.gz"],
        }
        for index in range(22)
    ]
    regional = [
        {
            "id": f"english-{index}",
            "targets": {"E": 1, "B": 1, "C": 1},
            "archives": [f"https://example.com/english-{index}.tar.gz"],
        }
        for index in range(7)
    ]
    return {
        "schema_version": 1,
        "upstream": {"sha256": "0" * 64, "excluded_archives": ["https://example.com/excluded.tar.gz"]},
        "designs": {
            key: {
                "total_hours": 40,
                "libritts_r_hours": 6,
                "official_regional_english_hours": 7,
                "official_non_english_hours": 22,
                "complement_hours": 5,
            }
            for key in ("E", "B", "C")
        },
        "official_sources": official,
        "regional_english_sources": regional,
        "acquisition_stages": [{"source_ids": [row["id"] for row in official + regional]}],
    }


def test_minimal_plan_is_valid():
    summary = validate(load_plan())
    assert summary == {
        "excluded_archive_count": 1,
        "official_language_count": 22,
        "planned_archive_count": 29,
        "regional_english_variety_count": 7,
    }


def test_rejects_allocation_drift():
    plan = copy.deepcopy(load_plan())
    plan["official_sources"][0]["targets"]["B"] += 1
    with pytest.raises(ValueError, match="official source targets plus reserve sum"):
        validate(plan)


def test_rejects_excluded_archive_in_plan():
    plan = copy.deepcopy(load_plan())
    excluded = plan["upstream"]["excluded_archives"][0]
    plan["official_sources"][0]["archives"].append(excluded)
    with pytest.raises(ValueError, match="excluded archive"):
        validate(plan)


def test_rejects_stage_coverage_drift():
    plan = copy.deepcopy(load_plan())
    plan["acquisition_stages"][0]["source_ids"].pop()
    with pytest.raises(ValueError, match="cover every official source"):
        validate(plan)
