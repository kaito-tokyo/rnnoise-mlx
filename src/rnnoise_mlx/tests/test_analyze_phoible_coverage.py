import csv

import pytest

from rnnoise_mlx.tools.analyze_phoible_coverage import analyze


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def fixture_cldf(root):
    write_csv(root / "languages.csv", [
        {"ID": "base", "Name": "Base", "ISO639P3code": "bas", "Glottocode": "base1234"},
        {"ID": "cand", "Name": "Candidate", "ISO639P3code": "can", "Glottocode": "cand1234"},
        {"ID": "alt", "Name": "Alternative", "ISO639P3code": "alt", "Glottocode": "alte1234"},
    ])
    write_csv(root / "contributions.csv", [
        {"ID": "1", "Name": "Base A", "Contributor_ID": "A", "Source": "a", "URL": "u"},
        {"ID": "2", "Name": "Base B", "Contributor_ID": "B", "Source": "b", "URL": "v"},
        {"ID": "3", "Name": "Candidate", "Contributor_ID": "C", "Source": "c", "URL": "w"},
        {"ID": "4", "Name": "Alternative", "Contributor_ID": "D", "Source": "d", "URL": "x"},
    ])
    write_csv(root / "parameters.csv", [
        {"ID": "s", "SegmentClass": "consonant"},
        {"ID": "theta", "SegmentClass": "consonant"},
        {"ID": "sh", "SegmentClass": "consonant"},
        {"ID": "longs", "SegmentClass": "consonant"},
        {"ID": "asp", "SegmentClass": "consonant"},
        {"ID": "vowel", "SegmentClass": "vowel"},
        {"ID": "tone", "SegmentClass": "tone"},
    ])
    write_csv(root / "values.csv", [
        {"Language_ID": "base", "Contribution_ID": "1", "Parameter_ID": "s", "Value": "s"},
        {"Language_ID": "base", "Contribution_ID": "1", "Parameter_ID": "theta", "Value": "θ"},
        {"Language_ID": "base", "Contribution_ID": "1", "Parameter_ID": "vowel", "Value": "a"},
        {"Language_ID": "base", "Contribution_ID": "2", "Parameter_ID": "s", "Value": "s"},
        {"Language_ID": "base", "Contribution_ID": "2", "Parameter_ID": "tone", "Value": "˥"},
        {"Language_ID": "cand", "Contribution_ID": "3", "Parameter_ID": "s", "Value": "s"},
        {"Language_ID": "cand", "Contribution_ID": "3", "Parameter_ID": "theta", "Value": "θ"},
        {"Language_ID": "cand", "Contribution_ID": "3", "Parameter_ID": "sh", "Value": "ɕ"},
        {"Language_ID": "cand", "Contribution_ID": "3", "Parameter_ID": "longs", "Value": "sː"},
        {"Language_ID": "cand", "Contribution_ID": "3", "Parameter_ID": "asp", "Value": "sʰ"},
        {"Language_ID": "cand", "Contribution_ID": "3", "Parameter_ID": "tone", "Value": "˩"},
        {"Language_ID": "cand", "Contribution_ID": "3", "Parameter_ID": "tone", "Value": "˥"},
        {"Language_ID": "alt", "Contribution_ID": "4", "Parameter_ID": "sh", "Value": "ɕ"},
        {"Language_ID": "alt", "Contribution_ID": "4", "Parameter_ID": "longs", "Value": "sː"},
        {"Language_ID": "alt", "Contribution_ID": "4", "Parameter_ID": "asp", "Value": "sʰ"},
        {"Language_ID": "alt", "Contribution_ID": "4", "Parameter_ID": "vowel", "Value": "i"},
        {"Language_ID": "alt", "Contribution_ID": "4", "Parameter_ID": "tone", "Value": "˩"},
    ])


def selection():
    return {
        "baseline": [
            {"label": "base", "mapping_status": "exact", "inventory_ids": ["1", "2"]},
            {"label": "missing", "mapping_status": "missing", "inventory_ids": []},
        ],
        "exclude_candidate_iso639_3": ["bas"],
        "exclude_candidate_inventory_ids": [],
        "cumulative_scenarios": [{"label": "one", "inventory_ids": ["3"]}],
    }


def test_literal_segments_uncertainty_tones_and_cumulative(tmp_path):
    fixture_cldf(tmp_path)
    result = analyze(tmp_path, selection())
    candidate = result["candidates"][0]

    assert result["baseline_definite"] == {"consonant": ["s"], "vowel": [], "tone": []}
    assert result["baseline_possible"] == {
        "consonant": ["s", "θ"], "vowel": ["a"], "tone": ["˥"]
    }
    assert candidate["definitely_novel"]["consonant"] == ["sʰ", "sː", "ɕ"]
    assert candidate["definitely_novel"]["tone"] == ["˩"]
    assert candidate["overlapping"]["consonant"] == ["s"]
    assert candidate["conditionally_novel"] == {"consonant": ["θ"], "vowel": [], "tone": ["˥"]}
    assert result["cumulative_scenarios"][0]["steps"][0]["new_at_step"]["tone"] == ["˩"]


def test_missing_language_is_visible_and_does_not_require_substitute(tmp_path):
    fixture_cldf(tmp_path)
    result = analyze(tmp_path, selection())
    missing = result["baseline"][1]
    assert missing["mapping_status"] == "missing"
    assert missing["inventories"] == []


def test_external_inventory_can_fill_phoible_gap_without_becoming_candidate(tmp_path):
    fixture_cldf(tmp_path)
    config = selection()
    config["baseline"][1] = {
        "label": "external",
        "mapping_status": "external-source-only",
        "inventory_ids": [],
        "external_inventories": [{
            "source": "grammar", "segments": {"consonant": ["ɕ"], "vowel": [], "tone": ["˩"]}
        }],
    }
    result = analyze(tmp_path, config)
    assert "ɕ" in result["baseline_definite"]["consonant"]
    assert "˩" in result["baseline_definite"]["tone"]
    assert result["candidates"][0]["definitely_novel"]["consonant"] == ["sʰ", "sː"]


def test_unknown_inventory_fails_instead_of_silently_becoming_empty(tmp_path):
    fixture_cldf(tmp_path)
    config = selection()
    config["baseline"][0]["inventory_ids"] = ["999"]
    with pytest.raises(ValueError, match="unknown or empty"):
        analyze(tmp_path, config)


def test_result_is_deterministic_and_scenario_math_matches_candidate(tmp_path):
    fixture_cldf(tmp_path)
    first = analyze(tmp_path, selection())
    second = analyze(tmp_path, selection())
    assert first == second
    candidate = first["candidates"][0]
    step = first["cumulative_scenarios"][0]["steps"][0]
    assert step["new_at_step"] == candidate["definitely_novel"]


def test_scenario_summary_and_literal_superset_alternatives(tmp_path):
    fixture_cldf(tmp_path)
    config = selection()
    config["cumulative_scenarios"].append(
        {"label": "superset", "inventory_ids": ["4"]}
    )
    result = analyze(tmp_path, config)
    selected, superset = result["cumulative_scenarios"]

    assert result["schema_version"] == 2
    assert selected["addition_counts"] == {"consonant": 3, "vowel": 0, "tone": 1}
    assert superset["addition_counts"] == {"consonant": 3, "vowel": 1, "tone": 1}
    assert selected["strictly_dominated_by"] == ["superset"]
    assert superset["strictly_dominated_by"] == []
    alternatives = selected["steps"][0]["covering_alternatives"]
    assert [item["inventory_id"] for item in alternatives] == ["4"]
    assert alternatives[0]["sources"] == ["d"]
    assert alternatives[0]["url"] == "x"
    assert alternatives[0]["additional_beyond_selected"] == {
        "consonant": [], "vowel": ["i"], "tone": []
    }


def test_alternative_requires_every_segment_without_similarity_folding(tmp_path):
    fixture_cldf(tmp_path)
    result = analyze(tmp_path, selection())
    alternatives = result["cumulative_scenarios"][0]["steps"][0][
        "covering_alternatives"
    ]
    assert alternatives[0]["new_at_step"]["consonant"] == ["sʰ", "sː", "ɕ"]
    assert "θ" not in alternatives[0]["new_at_step"]["consonant"]
