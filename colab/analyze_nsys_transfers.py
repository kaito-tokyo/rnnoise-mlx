"""Summarize Nsight Systems CUDA transfer events for an MLX run.

The report intentionally focuses on transfer frequency and granularity rather
than kernel throughput.  Phase timings are used to normalize transfer counts
per update and per observed ``mx.eval`` call; Nsight timestamps are not mixed
with Python's monotonic clock.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[index])


def summarize(values: list[int], *, small_threshold: int) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "total_bytes": 0,
            "mean_bytes": None,
            "median_bytes": None,
            "p95_bytes": None,
            "small_count": 0,
            "small_fraction": None,
        }
    return {
        "count": len(values),
        "total_bytes": sum(values),
        "mean_bytes": sum(values) / len(values),
        "median_bytes": percentile(values, 0.50),
        "p95_bytes": percentile(values, 0.95),
        "small_count": sum(value <= small_threshold for value in values),
        "small_fraction": sum(value <= small_threshold for value in values) / len(values),
    }


def summarize_gaps(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_ns": None, "median_ns": None, "p95_ns": None}
    return {
        "count": len(values),
        "mean_ns": sum(values) / len(values),
        "median_ns": percentile(values, 0.50),
        "p95_ns": percentile(values, 0.95),
        "min_ns": min(values),
        "max_ns": max(values),
    }


def size_histogram(values: list[int]) -> dict[str, int]:
    bins = {
        "0-4KiB": 0,
        "4-16KiB": 0,
        "16-64KiB": 0,
        "64-256KiB": 0,
        "256KiB-1MiB": 0,
        ">1MiB": 0,
    }
    for value in values:
        if value <= 4 * 1024:
            bins["0-4KiB"] += 1
        elif value <= 16 * 1024:
            bins["4-16KiB"] += 1
        elif value <= 64 * 1024:
            bins["16-64KiB"] += 1
        elif value <= 256 * 1024:
            bins["64-256KiB"] += 1
        elif value <= 1024 * 1024:
            bins["256KiB-1MiB"] += 1
        else:
            bins[">1MiB"] += 1
    return bins


def load_phase_counts(path: Path | None) -> dict[str, int | None]:
    if path is None:
        return {"updates": None, "mx_eval_calls": None, "chunks": None}
    rows = json.loads(path.read_text())
    return {
        "updates": len({row["update"] for row in rows if "update" in row}),
        "mx_eval_calls": sum(row["phase"] == "mx_eval" for row in rows),
        "chunks": sum(row["phase"] == "gradient_graph" for row in rows),
    }


def load_runtime_api_summary(connection: sqlite3.Connection) -> list[dict[str, int | str]]:
    """Summarize CUDA Runtime API calls when the report contains names."""
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if "CUPTI_ACTIVITY_KIND_RUNTIME" not in tables:
        return []
    name_column = None
    if "StringIds" in tables:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(StringIds)")
        }
        if {"id", "value"} <= columns:
            name_column = "value"
        elif {"id", "name"} <= columns:
            name_column = "name"
    if name_column is None:
        rows = connection.execute(
            "SELECT nameId, COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME "
            "GROUP BY nameId ORDER BY COUNT(*) DESC"
        ).fetchall()
        return [
            {"name_id": int(name_id), "count": int(count)}
            for name_id, count in rows
        ]
    rows = connection.execute(
        f"SELECT COALESCE(s.{name_column}, CAST(r.nameId AS TEXT)), COUNT(*) "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        f"LEFT JOIN StringIds s ON s.id = r.nameId "
        "GROUP BY r.nameId, s." + name_column + " ORDER BY COUNT(*) DESC"
    ).fetchall()
    return [{"name": str(name), "count": int(count)} for name, count in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="Nsight .sqlite report")
    parser.add_argument("--phase-timings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small-threshold", type=int, default=4096)
    args = parser.parse_args()

    with sqlite3.connect(args.report) as connection:
        enum_rows = connection.execute(
            "SELECT id, name FROM ENUM_CUDA_MEMCPY_OPER"
        ).fetchall()
        names = {identifier: name for identifier, name in enum_rows}
        events = connection.execute(
            """
            SELECT start, end, bytes, copyKind, streamId, correlationId
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE bytes >= 0
            ORDER BY start
            """
        ).fetchall()
        runtime_api_summary = load_runtime_api_summary(connection)

    phases = load_phase_counts(args.phase_timings)
    by_kind: dict[str, list[int]] = {}
    durations_by_kind: Counter[str] = Counter()
    for start, end, size, copy_kind, _stream, _correlation in events:
        kind = names.get(copy_kind, f"copyKind={copy_kind}")
        by_kind.setdefault(kind, []).append(int(size))
        durations_by_kind[kind] += max(0, int(end) - int(start))

    total_count = len(events)
    updates = phases["updates"]
    eval_calls = phases["mx_eval_calls"]
    kinds = {
        kind: {
            **summarize(values, small_threshold=args.small_threshold),
            "total_duration_ns": durations_by_kind[kind],
            "per_update_count": len(values) / updates if updates else None,
            "per_mx_eval_count": len(values) / eval_calls if eval_calls else None,
            "size_histogram": size_histogram(values),
        }
        for kind, values in sorted(by_kind.items())
    }

    # Events from different CUDA streams overlap.  Negative gaps therefore do
    # not represent a transfer stall; clamp them to zero for burst analysis.
    intervals = [
        max(0, int(current[0]) - int(previous[1]))
        for previous, current in zip(events, events[1:])
    ]
    report = {
        "report": str(args.report),
        "phase_timings": str(args.phase_timings) if args.phase_timings else None,
        "normalization": phases,
        "small_transfer_threshold_bytes": args.small_threshold,
        "transfer_count": total_count,
        "transfer_rate_per_update": total_count / updates if updates else None,
        "transfer_rate_per_mx_eval": total_count / eval_calls if eval_calls else None,
        "inter_transfer_gap_ns": summarize_gaps(intervals),
        "runtime_api_summary": runtime_api_summary,
        "classification": {
            "confirmed_code_origin": False,
            "candidate_boundary": "mx.eval and MLX/CUDA unified-memory synchronization",
            "status": "additional_investigation",
            "reason": "The trace has transfer/API events but no phase-correlated NVTX ranges.",
        },
        "by_kind": kinds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
