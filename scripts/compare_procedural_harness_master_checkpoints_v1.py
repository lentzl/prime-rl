"""Compare frozen Harness Master checkpoint evaluations against untouched 27B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SPLITS = ("valid-baseline", "ood-baseline")


def _read_summary(candidate: Path, split: str) -> dict[str, Any]:
    path = candidate / split / "SUMMARY.json"
    if not path.is_file():
        raise ValueError(f"missing frozen evaluation summary: {path}")
    report = json.loads(path.read_text())
    if report.get("rescored") is not True:
        raise ValueError(f"summary was not rescored: {path}")
    if report.get("episodes") != 24:
        raise ValueError(f"summary must contain 24 episodes: {path}")
    if report.get("errors") != 0:
        raise ValueError(f"summary contains rollout errors: {path}")
    return report


def _candidate_report(candidate: Path) -> dict[str, Any]:
    splits = {split: _read_summary(candidate, split) for split in SPLITS}
    return {
        "label": candidate.name,
        "splits": {
            split: {
                "passed": report["harness"]["passed"],
                "episodes": report["harness"]["episodes"],
                "rate": report["harness"]["rate"],
                "by_family": report["by_family"],
                "diagnostic_means": report["diagnostic_means"],
            }
            for split, report in splits.items()
        },
        "hard_passes": sum(report["harness"]["passed"] for report in splits.values()),
        "episodes": sum(report["harness"]["episodes"] for report in splits.values()),
    }


def compare(root: Path, expected_steps: int) -> dict[str, Any]:
    labels = ["untouched", *(f"step-{step}" for step in range(1, expected_steps + 1))]
    candidates = [_candidate_report(root / label) for label in labels]
    untouched = candidates[0]
    base_splits = untouched["splits"]

    for candidate in candidates:
        if candidate["label"] == "untouched":
            candidate["screen_pass"] = False
            candidate["delta_hard_passes"] = 0
            candidate["split_deltas"] = {split: 0 for split in SPLITS}
            continue
        split_deltas = {
            split: candidate["splits"][split]["passed"] - base_splits[split]["passed"]
            for split in SPLITS
        }
        candidate["delta_hard_passes"] = candidate["hard_passes"] - untouched["hard_passes"]
        candidate["split_deltas"] = split_deltas
        candidate["screen_pass"] = (
            candidate["delta_hard_passes"] > 0
            and all(delta >= 0 for delta in split_deltas.values())
        )

    eligible = [candidate for candidate in candidates if candidate["screen_pass"]]
    recommended = max(
        eligible,
        key=lambda candidate: (
            candidate["hard_passes"],
            candidate["splits"]["ood-baseline"]["passed"],
            -int(candidate["label"].split("-")[1]),
        ),
        default=None,
    )
    return {
        "schema_version": 1,
        "screen": {
            "description": (
                "A checkpoint passes the first screen only when combined hard passes improve, "
                "neither frozen split regresses, and all 48 traces rescore without errors."
            ),
            "promotion_note": (
                "A screen pass selects a candidate for a larger replicated promotion evaluation; "
                "it does not by itself promote the model."
            ),
        },
        "untouched": "untouched",
        "recommended_for_replication": recommended["label"] if recommended else None,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_steps < 1:
        raise SystemExit("--expected-steps must be positive")
    rendered = json.dumps(compare(args.root, args.expected_steps), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
