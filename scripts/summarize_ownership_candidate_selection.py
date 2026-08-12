#!/usr/bin/env python3
"""Compare paired child-owned and coordinator-owned candidate screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    traces: dict[tuple[Any, ...], dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for trace in record.get("traces") or [record]:
            task = trace["task"]["data"]
            key = (
                task["resource_family"],
                task["phrasing_variant"],
                task["resource_path"],
                task["state_value"],
            )
            if key in traces:
                raise ValueError(f"duplicate task key in {path}: {key}")
            traces[key] = trace
    return traces


def _metric(trace: dict[str, Any], name: str) -> float:
    metrics = trace.get("metrics") or {}
    if name not in metrics:
        raise ValueError(f"trace {trace.get('id')} has no {name!r} metric")
    return float(metrics[name])


def _arm(base_path: Path, candidate_path: Path, ownership: str) -> dict[str, Any]:
    base = _load(base_path)
    candidate = _load(candidate_path)
    if base.keys() != candidate.keys():
        raise ValueError(f"{ownership} base and candidate task sets differ")

    rows: list[dict[str, Any]] = []
    for key in sorted(base):
        base_trace = base[key]
        candidate_trace = candidate[key]
        base_success = _metric(base_trace, "strict_success") == 1.0
        candidate_success = _metric(candidate_trace, "strict_success") == 1.0
        rows.append(
            {
                "resource_family": key[0],
                "phrasing_variant": key[1],
                "resource_path": key[2],
                "state_value": key[3],
                "base_trace_id": base_trace["id"],
                "candidate_trace_id": candidate_trace["id"],
                "base_strict_success": base_success,
                "candidate_strict_success": candidate_success,
                "strict_gain": not base_success and candidate_success,
                "strict_loss": base_success and not candidate_success,
                "base_parent_path_access": _metric(base_trace, "parent_path_access"),
                "candidate_parent_path_access": _metric(candidate_trace, "parent_path_access"),
                "base_spawn": _metric(base_trace, "one_spawn"),
                "candidate_spawn": _metric(candidate_trace, "one_spawn"),
            }
        )

    return {
        "ownership": ownership,
        "tasks": len(rows),
        "base_strict_successes": sum(row["base_strict_success"] for row in rows),
        "candidate_strict_successes": sum(row["candidate_strict_success"] for row in rows),
        "strict_gains": sum(row["strict_gain"] for row in rows),
        "strict_losses": sum(row["strict_loss"] for row in rows),
        "base_parent_path_accesses": sum(row["base_parent_path_access"] for row in rows),
        "candidate_parent_path_accesses": sum(row["candidate_parent_path_access"] for row in rows),
        "base_spawns": sum(row["base_spawn"] for row in rows),
        "candidate_spawns": sum(row["candidate_spawn"] for row in rows),
        "rows": rows,
    }


def summarize(
    child_base: Path,
    child_candidate: Path,
    direct_base: Path,
    direct_candidate: Path,
) -> dict[str, Any]:
    child = _arm(child_base, child_candidate, "child")
    direct = _arm(direct_base, direct_candidate, "coordinator")
    gains = child["strict_gains"] + direct["strict_gains"]
    losses = child["strict_losses"] + direct["strict_losses"]
    no_child_path_leakage = child["candidate_parent_path_accesses"] == 0
    no_direct_overdelegation = direct["candidate_spawns"] == 0
    return {
        "tasks": child["tasks"] + direct["tasks"],
        "strict_gains": gains,
        "strict_losses": losses,
        "candidate_child_path_accesses": child["candidate_parent_path_accesses"],
        "candidate_direct_spawns": direct["candidate_spawns"],
        "promotion_pass": gains >= 6 and losses <= 1 and no_child_path_leakage and no_direct_overdelegation,
        "criteria": {
            "minimum_strict_gains": 6,
            "maximum_strict_losses": 1,
            "maximum_candidate_child_path_accesses": 0,
            "maximum_candidate_direct_spawns": 0,
        },
        "child": child,
        "direct": direct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-base", type=Path, required=True)
    parser.add_argument("--child-candidate", type=Path, required=True)
    parser.add_argument("--direct-base", type=Path, required=True)
    parser.add_argument("--direct-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.child_base, args.child_candidate, args.direct_base, args.direct_candidate)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
