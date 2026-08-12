#!/usr/bin/env python3
"""Summarize strict native-success supply by rollout resource family."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize(path: Path, *, minimum_families: int, minimum_multi_success_families: int) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for trace in record.get("traces") or [record]:
            by_family[trace["task"]["data"]["resource_family"]].append(trace)

    families: dict[str, dict[str, Any]] = {}
    for family, traces in sorted(by_family.items()):
        successes = [trace for trace in traces if float(trace["metrics"]["strict_success"]) == 1.0]
        families[family] = {
            "rollouts": len(traces),
            "strict_successes": len(successes),
            "strict_success_trace_ids": [trace["id"] for trace in successes],
        }

    families_with_success = sum(values["strict_successes"] >= 1 for values in families.values())
    families_with_multiple_successes = sum(values["strict_successes"] >= 2 for values in families.values())
    return {
        "families": families,
        "rollouts": sum(values["rollouts"] for values in families.values()),
        "strict_successes": sum(values["strict_successes"] for values in families.values()),
        "families_with_success": families_with_success,
        "families_with_multiple_successes": families_with_multiple_successes,
        "admission_pass": (
            families_with_success >= minimum_families
            and families_with_multiple_successes >= minimum_multi_success_families
        ),
        "criteria": {
            "minimum_families_with_success": minimum_families,
            "minimum_families_with_multiple_successes": minimum_multi_success_families,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--minimum-families", type=int, default=6)
    parser.add_argument("--minimum-multi-success-families", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(
        args.traces,
        minimum_families=args.minimum_families,
        minimum_multi_success_families=args.minimum_multi_success_families,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
