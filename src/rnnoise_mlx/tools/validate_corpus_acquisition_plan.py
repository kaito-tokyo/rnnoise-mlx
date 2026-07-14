"""Validate a corpus acquisition plan before downloading audio."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DESIGN_KEYS = {"E", "B", "C"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_source(source: dict[str, Any], seen_ids: set[str]) -> None:
    source_id = source.get("id")
    _require(isinstance(source_id, str) and source_id, "source id is required")
    _require(source_id not in seen_ids, f"duplicate source id: {source_id}")
    seen_ids.add(source_id)

    targets = source.get("targets")
    _require(isinstance(targets, dict), f"{source_id}: targets are required")
    _require(set(targets) == DESIGN_KEYS, f"{source_id}: targets must contain E, B, and C")
    for design, hours in targets.items():
        _require(isinstance(hours, (int, float)) and hours >= 0, f"{source_id}/{design}: invalid target")

    archives = source.get("archives")
    _require(isinstance(archives, list) and archives, f"{source_id}: at least one archive is required")
    for url in archives:
        parsed = urlparse(url)
        _require(parsed.scheme == "https" and bool(parsed.netloc), f"{source_id}: invalid HTTPS URL: {url}")


def validate(plan: dict[str, Any], upstream_urls: set[str] | None = None) -> dict[str, Any]:
    _require(plan.get("schema_version") == 1, "unsupported schema_version")
    upstream = plan.get("upstream")
    _require(isinstance(upstream, dict), "upstream is required")
    _require(SHA256_RE.fullmatch(str(upstream.get("sha256", ""))) is not None, "invalid upstream SHA-256")

    designs = plan.get("designs")
    _require(isinstance(designs, dict) and set(designs) == DESIGN_KEYS, "designs must be E, B, and C")
    for design_id, design in designs.items():
        parts = (
            design["libritts_r_hours"],
            design["official_regional_english_hours"],
            design["official_non_english_hours"],
            design["complement_hours"],
        )
        _require(abs(sum(parts) - design["total_hours"]) < 1e-8, f"design {design_id} does not sum to total_hours")

    official = plan.get("official_sources")
    regional = plan.get("regional_english_sources")
    _require(isinstance(official, list) and len(official) == 22, "official_sources must contain 22 non-English languages")
    _require(isinstance(regional, list) and len(regional) == 7, "regional_english_sources must contain 7 varieties")

    seen_ids: set[str] = set()
    for source in [*official, *regional]:
        _validate_source(source, seen_ids)

    for design_id, design in designs.items():
        official_total = sum(source["targets"][design_id] for source in official)
        regional_total = sum(source["targets"][design_id] for source in regional)
        reserve = design.get("official_non_english_unallocated_hours", 0)
        _require(
            abs(official_total + reserve - design["official_non_english_hours"]) < 1e-6,
            f"design {design_id}: official source targets plus reserve sum to {official_total + reserve}",
        )
        _require(
            abs(regional_total - design["official_regional_english_hours"]) < 1e-6,
            f"design {design_id}: regional English targets sum to {regional_total}",
        )

    planned_urls = {url for source in [*official, *regional] for url in source["archives"]}
    excluded_urls = set(upstream.get("excluded_archives", []))
    _require(not (planned_urls & excluded_urls), "an excluded archive is also planned")
    if upstream_urls is not None:
        _require(planned_urls | excluded_urls == upstream_urls, "planned plus excluded archives do not exactly match upstream")

    stages = plan.get("acquisition_stages")
    _require(isinstance(stages, list) and stages, "acquisition_stages are required")
    staged_ids = [source_id for stage in stages for source_id in stage.get("source_ids", [])]
    _require(len(staged_ids) == len(set(staged_ids)), "a source occurs in more than one acquisition stage")
    _require(set(staged_ids) == seen_ids, "acquisition stages must cover every official source exactly once")

    return {
        "official_language_count": len(official),
        "regional_english_variety_count": len(regional),
        "planned_archive_count": len(planned_urls),
        "excluded_archive_count": len(excluded_urls),
    }


def read_upstream_urls(path: Path) -> set[str]:
    return set(re.findall(r"https://[^\s]+?\.(?:zip|tar\.gz)", path.read_text()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--upstream-datasets", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    upstream_urls = read_upstream_urls(args.upstream_datasets) if args.upstream_datasets else None
    print(json.dumps(validate(plan, upstream_urls), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
