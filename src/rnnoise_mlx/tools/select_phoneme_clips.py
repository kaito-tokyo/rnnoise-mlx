"""Select a reproducible clip sample that covers every required phoneme."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import unicodedata

from .splitmix64 import MASK64, SplitMix64, uniform_below

ALGORITHM = "splitmix64-fisher-yates-v1"


class SelectionError(ValueError):
    """Raised when selection input or constraints are invalid."""


def normalize_phone(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise SelectionError(f"{context} must be a non-empty string")
    return unicodedata.normalize("NFC", value)


def normalize_clip(value: object, line_number: int) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SelectionError(f"population line {line_number} must be a JSON object")
    if set(value) != {"path", "phones"}:
        raise SelectionError(
            f"population line {line_number} must contain exactly path and phones"
        )
    path = value["path"]
    phones = value["phones"]
    if not isinstance(path, str) or not path:
        raise SelectionError(f"population line {line_number} path must be a non-empty string")
    if not isinstance(phones, list) or not phones:
        raise SelectionError(f"population line {line_number} phones must be a non-empty array")
    normalized = sorted(
        {normalize_phone(phone, f"population line {line_number} phone") for phone in phones}
    )
    return {"path": path, "phones": normalized}


def load_population(path: Path) -> list[dict[str, object]]:
    clips = []
    seen_paths = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise SelectionError(f"cannot read population: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise SelectionError(f"population line {line_number} is empty")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise SelectionError(f"invalid JSON on population line {line_number}: {error.msg}") from error
        clip = normalize_clip(value, line_number)
        if clip["path"] in seen_paths:
            raise SelectionError(f"duplicate population path: {clip['path']}")
        seen_paths.add(clip["path"])
        clips.append(clip)
    if not clips:
        raise SelectionError("population is empty")
    return sorted(clips, key=lambda clip: str(clip["path"]).encode("utf-8"))


def load_required_phones(path: Path) -> list[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SelectionError(f"cannot read required phones: {error}") from error
    if not isinstance(value, list) or not value:
        raise SelectionError("required phones must be a non-empty JSON array")
    return sorted({normalize_phone(phone, "required phone") for phone in value})


def shuffle(clips: list[dict[str, object]], seed: int) -> tuple[list[dict[str, object]], int]:
    result = list(clips)
    rng = SplitMix64(seed)
    for index in range(len(result) - 1, 0, -1):
        other = uniform_below(rng, index + 1)
        result[index], result[other] = result[other], result[index]
    return result, rng.calls


def covered_phones(clips: list[dict[str, object]]) -> set[str]:
    return {phone for clip in clips for phone in clip["phones"]}  # type: ignore[union-attr]


def select(
    population: list[dict[str, object]],
    required_phones: list[str],
    sample_count: int,
    base_seed: int,
    max_attempts: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not 1 <= sample_count <= len(population):
        raise SelectionError(
            f"sample count must be between 1 and population size {len(population)}"
        )
    if max_attempts < 1:
        raise SelectionError("max attempts must be at least 1")
    required = set(required_phones)
    unavailable = required - covered_phones(population)
    if unavailable:
        raise SelectionError(f"population lacks required phones: {', '.join(sorted(unavailable))}")

    rejected = []
    for attempt in range(max_attempts):
        seed = (base_seed + attempt) & MASK64
        ordered, calls = shuffle(population, seed)
        sample = ordered[:sample_count]
        covered = covered_phones(sample)
        missing = sorted(required - covered)
        if not missing:
            metadata: dict[str, object] = {
                "format_version": 1,
                "algorithm": ALGORITHM,
                "base_seed": base_seed & MASK64,
                "accepted_seed": seed,
                "accepted_attempt": attempt,
                "sample_count": sample_count,
                "population_count": len(population),
                "rng_calls_per_attempt": calls,
                "required_phones": sorted(required),
                "covered_phones": sorted(covered),
                "rejected_attempts": rejected,
            }
            return sample, metadata
        rejected.append({"attempt": attempt, "seed": seed, "missing_phones": missing})
    last_missing = rejected[-1]["missing_phones"]
    raise SelectionError(
        f"no sample covered all required phones in {max_attempts} attempts; "
        f"last missing: {', '.join(last_missing)}"
    )


def write_result(output: Path, clips: list[dict[str, object]], metadata: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    clips_text = "".join(
        json.dumps(clip, ensure_ascii=False, separators=(",", ":")) + "\n" for clip in clips
    )
    (output / "clips.jsonl").write_text(clips_text, encoding="utf-8")
    (output / "selection.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("population", type=Path)
    result.add_argument("required_phones", type=Path)
    result.add_argument("output", type=Path)
    result.add_argument("--sample-count", type=int, required=True)
    result.add_argument("--base-seed", type=int, required=True)
    result.add_argument("--max-attempts", type=int, default=1000)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        population = load_population(args.population)
        required = load_required_phones(args.required_phones)
        clips, metadata = select(
            population, required, args.sample_count, args.base_seed, args.max_attempts
        )
        write_result(args.output, clips, metadata)
    except SelectionError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
