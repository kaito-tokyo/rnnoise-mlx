"""Analyze literal IPA coverage across source-specific PHOIBLE inventories.

This module intentionally performs no phonetic-feature or similarity folding.
Every distinct Unicode segment remains distinct. PHOIBLE must be supplied by
the caller; the licensed database is not vendored in this repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


SEGMENT_CLASSES = ("consonant", "vowel", "tone")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def load_cldf(root: Path) -> dict:
    languages = {row["ID"]: row for row in read_csv(root / "languages.csv")}
    contributions = {row["ID"]: row for row in read_csv(root / "contributions.csv")}
    parameters = {row["ID"]: row for row in read_csv(root / "parameters.csv")}
    segments: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {name: set() for name in SEGMENT_CLASSES}
    )
    language_ids: dict[str, str] = {}
    for row in read_csv(root / "values.csv"):
        inventory_id = row["Contribution_ID"]
        segment_class = parameters[row["Parameter_ID"]]["SegmentClass"].lower()
        if segment_class not in SEGMENT_CLASSES:
            raise ValueError(f"unknown segment class {segment_class!r}")
        segments[inventory_id][segment_class].add(row["Value"])
        language_ids[inventory_id] = row["Language_ID"]
    return {
        "languages": languages,
        "contributions": contributions,
        "segments": segments,
        "language_ids": language_ids,
    }


def sorted_classes(classes: dict[str, Iterable[str]]) -> dict[str, list[str]]:
    return {name: sorted(classes[name]) for name in SEGMENT_CLASSES}


def union_classes(items: Iterable[dict[str, Iterable[str]]]) -> dict[str, set[str]]:
    result = {name: set() for name in SEGMENT_CLASSES}
    for item in items:
        for name in SEGMENT_CLASSES:
            result[name].update(item[name])
    return result


def intersection_classes(items: list[dict[str, Iterable[str]]]) -> dict[str, set[str]]:
    if not items:
        return {name: set() for name in SEGMENT_CLASSES}
    return {
        name: set.intersection(*(set(item[name]) for item in items))
        for name in SEGMENT_CLASSES
    }


def subtract_classes(left, right) -> dict[str, set[str]]:
    return {name: set(left[name]) - set(right[name]) for name in SEGMENT_CLASSES}


def intersect_classes(left, right) -> dict[str, set[str]]:
    return {name: set(left[name]) & set(right[name]) for name in SEGMENT_CLASSES}


def includes_classes(left, right) -> bool:
    """Return whether every literal segment in right is present in left."""
    return all(set(right[name]) <= set(left[name]) for name in SEGMENT_CLASSES)


def strictly_includes_classes(left, right) -> bool:
    return includes_classes(left, right) and any(
        set(left[name]) != set(right[name]) for name in SEGMENT_CLASSES
    )


def class_counts(classes) -> dict[str, int]:
    return {name: len(classes[name]) for name in SEGMENT_CLASSES}


def inventory_record(inventory_id: str, data: dict) -> dict:
    if inventory_id not in data["contributions"] or inventory_id not in data["segments"]:
        raise ValueError(f"unknown or empty PHOIBLE inventory ID: {inventory_id}")
    contribution = data["contributions"][inventory_id]
    language = data["languages"][data["language_ids"][inventory_id]]
    return {
        "inventory_id": inventory_id,
        "inventory_name": contribution["Name"],
        "source_collection": contribution["Contributor_ID"],
        "sources": contribution["Source"].split(";"),
        "url": contribution["URL"],
        "language_name": language["Name"],
        "iso639_3": language["ISO639P3code"],
        "glottocode": language["Glottocode"],
        "segments": sorted_classes(data["segments"][inventory_id]),
    }


def baseline_coverage(entries: list[dict], data: dict) -> tuple[list[dict], dict, dict]:
    records = []
    language_definite = []
    language_possible = []
    for entry in entries:
        ids = [str(value) for value in entry.get("inventory_ids", [])]
        if not ids and not entry.get("external_inventories") and entry["mapping_status"] != "missing":
            raise ValueError(f"{entry['label']} has no inventory IDs but is not missing")
        inventories = [inventory_record(value, data) for value in ids]
        external_inventories = []
        for external in entry.get("external_inventories", []):
            classes_value = {
                name: list(external.get("segments", {}).get(name, []))
                for name in SEGMENT_CLASSES
            }
            external_inventories.append({**external, "segments": classes_value})
        classes = [record["segments"] for record in inventories + external_inventories]
        definite = intersection_classes(classes)
        possible = union_classes(classes)
        records.append(
            {
                **entry,
                "inventory_ids": ids,
                "inventories": inventories,
                "external_inventories": external_inventories,
                "definite_segments": sorted_classes(definite),
                "possible_segments": sorted_classes(possible),
            }
        )
        if classes:
            language_definite.append(definite)
            language_possible.append(possible)
    # A segment is definitely covered if any mapped baseline language has it in
    # every explicitly listed source inventory. Possible coverage is the union
    # of every source inventory. Missing languages contribute neither and stay
    # visible in records rather than being silently treated as empty languages.
    return records, union_classes(language_definite), union_classes(language_possible)


def classify_inventory(record: dict, definite: dict, possible: dict) -> dict:
    segments = record["segments"]
    return {
        **record,
        "definitely_novel": sorted_classes(subtract_classes(segments, possible)),
        "conditionally_novel": sorted_classes(
            intersect_classes(subtract_classes(segments, definite), possible)
        ),
        "overlapping": sorted_classes(intersect_classes(segments, definite)),
    }


def analyze(cldf: Path, selection: dict) -> dict:
    data = load_cldf(cldf)
    baseline, definite, possible = baseline_coverage(selection["baseline"], data)
    phoible_universe = union_classes(data["segments"].values())
    excluded_iso = set(selection.get("exclude_candidate_iso639_3", []))
    excluded_inventory_ids = set(map(str, selection.get("exclude_candidate_inventory_ids", [])))
    candidates = []
    for inventory_id in sorted(data["segments"], key=int):
        if inventory_id in excluded_inventory_ids:
            continue
        record = inventory_record(inventory_id, data)
        if record["iso639_3"] in excluded_iso:
            continue
        candidates.append(classify_inventory(record, definite, possible))

    cumulative = []
    for scenario in selection.get("cumulative_scenarios", []):
        covered = {name: set(possible[name]) for name in SEGMENT_CLASSES}
        steps = []
        inventory_ids = list(map(str, scenario["inventory_ids"]))
        for inventory_id in inventory_ids:
            record = inventory_record(inventory_id, data)
            added = subtract_classes(record["segments"], covered)
            alternatives = []
            if any(added[name] for name in SEGMENT_CLASSES):
                for alternative_id in sorted(data["segments"], key=int):
                    if (
                        alternative_id in inventory_ids
                        or alternative_id in excluded_inventory_ids
                    ):
                        continue
                    alternative = inventory_record(alternative_id, data)
                    if alternative["iso639_3"] in excluded_iso:
                        continue
                    alternative_added = subtract_classes(
                        alternative["segments"], covered
                    )
                    if includes_classes(alternative_added, added):
                        alternatives.append({
                            "inventory_id": alternative["inventory_id"],
                            "inventory_name": alternative["inventory_name"],
                            "language_name": alternative["language_name"],
                            "iso639_3": alternative["iso639_3"],
                            "glottocode": alternative["glottocode"],
                            "source_collection": alternative["source_collection"],
                            "sources": alternative["sources"],
                            "url": alternative["url"],
                            "new_at_step": sorted_classes(alternative_added),
                            "additional_beyond_selected": sorted_classes(
                                subtract_classes(alternative_added, added)
                            ),
                        })
            for name in SEGMENT_CLASSES:
                covered[name].update(record["segments"][name])
            steps.append({
                "inventory": record,
                "new_at_step": sorted_classes(added),
                "new_at_step_counts": class_counts(added),
                "covered_counts": class_counts(covered),
                "covering_alternatives": alternatives,
            })
        additions = subtract_classes(covered, possible)
        remaining = subtract_classes(phoible_universe, covered)
        cumulative.append({
            "label": scenario["label"],
            "inventory_ids": inventory_ids,
            "steps": steps,
            "additions_beyond_baseline_possible": sorted_classes(additions),
            "addition_counts": class_counts(additions),
            "remaining_phoible_segments": sorted_classes(remaining),
            "remaining_counts": class_counts(remaining),
        })

    for scenario in cumulative:
        additions = scenario["additions_beyond_baseline_possible"]
        scenario["strictly_dominated_by"] = [
            other["label"]
            for other in cumulative
            if other is not scenario
            and strictly_includes_classes(
                other["additions_beyond_baseline_possible"], additions
            )
        ]

    canonical = json.dumps(selection, ensure_ascii=False, sort_keys=True).encode()
    return {
        "schema_version": 2,
        "selection_sha256": hashlib.sha256(canonical).hexdigest(),
        "baseline": baseline,
        "baseline_definite": sorted_classes(definite),
        "baseline_possible": sorted_classes(possible),
        "baseline_counts": {
            "definite": {name: len(definite[name]) for name in SEGMENT_CLASSES},
            "possible": {name: len(possible[name]) for name in SEGMENT_CLASSES},
        },
        "phoible_universe_counts": {
            name: len(phoible_universe[name]) for name in SEGMENT_CLASSES
        },
        "candidates": candidates,
        "cumulative_scenarios": cumulative,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cldf", type=Path, help="directory containing PHOIBLE CLDF CSV files")
    parser.add_argument("selection", type=Path, help="baseline and scenario selection JSON")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    result = analyze(args.cldf, selection)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
