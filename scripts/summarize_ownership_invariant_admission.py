#!/usr/bin/env python3
"""Summarize native-sibling admission groups and the frozen Phase-A gate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize(path: Path) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for trace in record.get("traces") or [record]:
            task = trace["task"]["data"]
            groups[task["name"]].append(trace)

    rows: list[dict[str, Any]] = []
    success_families: set[str] = set()
    success_phrasings: set[int] = set()
    for name, traces in sorted(groups.items()):
        task = traces[0]["task"]["data"]
        successes = sum(trace.get("metrics", {}).get("strict_success", 0.0) == 1.0 for trace in traces)
        failures = len(traces) - successes
        if successes:
            success_families.add(task["resource_family"])
            success_phrasings.add(task["phrasing_variant"])
        rows.append(
            {
                "task": name,
                "resource_family": task["resource_family"],
                "phrasing_variant": task["phrasing_variant"],
                "rollouts": len(traces),
                "successes": successes,
                "failures": failures,
                "mixed": successes > 0 and failures > 0,
            }
        )

    mixed_groups = sum(row["mixed"] for row in rows)
    return {
        "traces": sum(row["rollouts"] for row in rows),
        "groups": len(rows),
        "strict_successes": sum(row["successes"] for row in rows),
        "mixed_groups": mixed_groups,
        "success_resource_families": sorted(success_families),
        "success_phrasings": sorted(success_phrasings),
        "phase_a_pass": (mixed_groups >= 4 and len(success_families) >= 3 and len(success_phrasings) >= 2),
        "criteria": {
            "minimum_mixed_groups": 4,
            "minimum_success_resource_families": 3,
            "minimum_success_phrasings": 2,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.traces), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
