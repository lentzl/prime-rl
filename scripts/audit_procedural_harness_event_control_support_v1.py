#!/usr/bin/env python3
"""Validate that failed hard-gate traces contain trainable event-control signal."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def event_control_progress(metrics: dict[str, Any]) -> float:
    recorded = metrics.get("event_control_progress")
    if recorded is not None:
        return float(recorded)
    return (
        float(metrics.get("no_forbidden_atoms", 0.0))
        * float(metrics.get("required_atoms_fraction", 0.0))
        * float(metrics.get("ordering_fraction", 0.0))
        * float(metrics.get("cardinality_fraction", 0.0))
    )


def summarize_support(
    traces: list[dict[str, Any]],
    *,
    env_name: str,
    group_size: int,
) -> dict[str, Any]:
    selected = [trace for trace in traces if trace.get("info", {}).get("env_name") == env_name]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in selected:
        groups[str(trace.get("info", {}).get("group_id", ""))].append(trace)

    group_reports = []
    for group_id, members in groups.items():
        scores = [event_control_progress(trace.get("metrics", {})) for trace in members]
        group_reports.append(
            {
                "group_id": group_id,
                "episodes": len(members),
                "complete": len(members) == group_size,
                "informative": len(members) == group_size and len(set(scores)) > 1,
                "nonzero": sum(score > 0 for score in scores),
                "minimum": min(scores, default=0.0),
                "maximum": max(scores, default=0.0),
            }
        )

    return {
        "env_name": env_name,
        "episodes": len(selected),
        "errors": sum(trace.get("ok") is not True for trace in selected),
        "hard_successes": sum(
            float(trace.get("rewards", {}).get("harness_score", {}).get("score", 0.0)) > 0 for trace in selected
        ),
        "event_control_nonzero": sum(event_control_progress(trace.get("metrics", {})) > 0 for trace in selected),
        "complete_groups": sum(report["complete"] for report in group_reports),
        "informative_groups": sum(report["informative"] for report in group_reports),
        "groups": group_reports,
    }


def validate_support(
    report: dict[str, Any],
    *,
    min_episodes: int,
    min_informative_groups: int,
) -> None:
    if report["episodes"] < min_episodes:
        raise ValueError(f"event-control support requires at least {min_episodes} episodes; found {report['episodes']}")
    if report["errors"]:
        raise ValueError(f"event-control support contains {report['errors']} errored episodes")
    if report["hard_successes"]:
        raise ValueError(
            "event-control support must document a hard-disconnected cohort; "
            f"found {report['hard_successes']} hard successes"
        )
    if report["informative_groups"] < min_informative_groups:
        raise ValueError(
            f"event-control support requires {min_informative_groups} informative groups; "
            f"found {report['informative_groups']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path)
    parser.add_argument("--env-name", default="atomic-followup-target")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--min-episodes", type=int, default=64)
    parser.add_argument("--min-informative-groups", type=int, default=2)
    args = parser.parse_args()

    traces = [json.loads(line) for line in args.traces.read_text().splitlines() if line.strip()]
    report = summarize_support(traces, env_name=args.env_name, group_size=args.group_size)
    validate_support(
        report,
        min_episodes=args.min_episodes,
        min_informative_groups=args.min_informative_groups,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
