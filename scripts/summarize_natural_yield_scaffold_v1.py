#!/usr/bin/env python3
"""Fail-closed summary for the training-only natural-yield scaffold ramp."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

if __package__:
    from .summarize_procedural_harness_master_v1 import _rescore, _score, _traces
else:
    from summarize_procedural_harness_master_v1 import _rescore, _score, _traces

SCHEMA = "prime-agent/natural-yield-scaffold-summary/v1"
SCAFFOLD_SCHEMA = "prime-agent/natural-yield-scaffold/v1"


def summarize(path: Path, *, rescore: bool = True) -> dict[str, Any]:
    rows = _traces(path)
    if rescore:
        rows = [_rescore(row) for row in rows]
    fired = []
    controls = []
    families: Counter[str] = Counter()
    for row in rows:
        data = row.get("task", {}).get("data", {})
        metadata = data.get("generation_metadata", {}) or {}
        graph = metadata.get("graph_variant")
        info = row.get("info", {}) or {}
        scaffold = info.get("natural_yield_scaffold")
        if isinstance(scaffold, dict) and scaffold.get("schema_version") == SCAFFOLD_SCHEMA:
            fired.append(row)
            families[str(metadata.get("semantic_family", "unknown"))] += 1
        elif graph == "child_plus_local_work_and_private_state":
            controls.append(row)

    hard = sum(_score(row, "harness_score", "rewards") == 1.0 for row in fired)
    native_yield = sum(
        _score(row, "forbidden_post_spawn_tool_before_child", "metrics") == 0.0
        for row in fired
    )
    control_premature = sum(
        _score(row, "premature_yield_before_local_work", "metrics") > 0.0
        for row in controls
    )
    return {
        "schema_version": SCHEMA,
        "episodes": len(rows),
        "errors": sum(bool(row.get("error") or row.get("errors")) for row in rows),
        "scaffolded": {
            "episodes": len(fired),
            "hard_successes": hard,
            "hard_rate": hard / len(fired) if fired else 0.0,
            "no_forbidden_post_spawn_tool": native_yield,
            "semantic_families": sorted(families),
            "semantic_family_counts": dict(sorted(families.items())),
        },
        "local_work_control": {
            "episodes": len(controls),
            "scaffold_fires": 0,
            "premature_yield_failures": control_premature,
        },
        "admission": {
            "y0_connected": (
                len(fired) >= 8
                and hard >= 4
                and native_yield >= max(4, len(fired) // 2)
                and len(families) >= 3
            ),
            "y1_harvest_ready": hard >= 16 and len(families) >= 6,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--no-rescore", action="store_true")
    args = parser.parse_args()
    print(json.dumps(summarize(args.path, rescore=not args.no_rescore), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
