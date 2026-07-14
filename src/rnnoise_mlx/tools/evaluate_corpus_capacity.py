"""Evaluate retained-hour targets against measured, speaker-disjoint audio."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


def evaluate(plan: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    sources = {
        source["id"]: source
        for source in [*plan["official_sources"], *plan["regional_english_sources"]]
    }
    results: dict[str, Any] = {}
    for source_id in sorted(audit["source_summaries"]):
        source = sources[source_id]
        eval_speakers = set(source.get("eval_speakers", []))
        clips = [row for row in audit["clips"] if row["source_id"] == source_id and "error" not in row]
        train = [row for row in clips if row["speaker_id"] not in eval_speakers]
        evaluation = [row for row in clips if row["speaker_id"] in eval_speakers]
        uncapped_train_seconds = sum(row["edge_trimmed_duration_seconds"] for row in train)
        speaker_seconds: dict[str, float] = {}
        for row in train:
            speaker_seconds[row["speaker_id"]] = speaker_seconds.get(row["speaker_id"], 0) + row[
                "edge_trimmed_duration_seconds"
            ]
        cap = source.get("train_speaker_cap_seconds")
        measured_train_seconds = sum(min(seconds, cap) if cap is not None else seconds for seconds in speaker_seconds.values())
        measured_eval_seconds = sum(row["edge_trimmed_duration_seconds"] for row in evaluation)
        designs = {}
        for design, hours in source["targets"].items():
            target_seconds = hours * 3600
            designs[design] = {
                "target_seconds": target_seconds,
                "headroom_seconds": measured_train_seconds - target_seconds,
                "feasible_before_listening": measured_train_seconds >= target_seconds,
            }
        results[source_id] = {
            "eval_speakers": sorted(eval_speakers),
            "train_speaker_count": len({row["speaker_id"] for row in train}),
            "eval_speaker_count": len({row["speaker_id"] for row in evaluation}),
            "uncapped_edge_trimmed_train_seconds": uncapped_train_seconds,
            "edge_trimmed_train_seconds": measured_train_seconds,
            "train_speaker_cap_seconds": cap,
            "edge_trimmed_eval_seconds": measured_eval_seconds,
            "designs": designs,
        }
    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "note": "Feasible before listening is a capacity check, not final corpus acceptance.",
        "sources": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = evaluate(json.loads(args.plan.read_text()), json.loads(args.audit.read_text()))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["sources"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
