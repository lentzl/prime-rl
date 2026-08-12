"""Summarize saved Prime Agent mastery traces by capability family.

The eval writer stores one wire envelope per JSONL line. This script deliberately
uses only that saved evidence, so a completed battery can be audited without a live
environment server or model endpoint.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

METRICS = (
    "answer_accuracy",
    "protocol_aligned",
    "clean_protocol_aligned",
    "natural_followup_causal",
    "bidirectional_control",
    "post_fan_in_control",
    "coordinator_delegated_path_accesses",
    "roster_calls",
    "observation_calls",
    "failed_cells",
    "duplicate_cells",
    "strict_success",
    "first_decision_only",
    "state_retained",
    "state_precedes_spawn",
    "one_spawn",
    "retained_handle",
    "expected_child",
    "delegated_path",
    "parent_path_access",
    "direct_answer_accuracy",
)


def _trace_paths(paths: Iterable[Path]) -> list[Path]:
    resolved = []
    for path in paths:
        trace_path = path / "traces.jsonl" if path.is_dir() else path
        if not trace_path.is_file():
            raise SystemExit(f"no traces.jsonl found at {path}")
        resolved.append(trace_path)
    return resolved


def load_traces(paths: Iterable[Path]) -> list[dict[str, Any]]:
    traces = []
    for path in _trace_paths(paths):
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                envelope = json.loads(line)
                saved = envelope.get("traces")
                if not isinstance(saved, list):
                    raise SystemExit(f"{path}:{line_number}: missing traces list")
                traces.extend(saved)
    return traces


def _score(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict) and isinstance(value.get("score"), (int, float)):
        return float(value["score"])
    return None


def _family(trace: dict[str, Any]) -> str:
    data = trace.get("task", {}).get("data", {})
    family = data.get("family")
    if isinstance(family, str):
        return family
    name = data.get("name", "unknown")
    return str(name).split("-", 1)[0]


def _issues(trace: dict[str, Any]) -> list[str]:
    family = _family(trace)
    metrics = trace.get("metrics", {})
    issues = []
    if not trace.get("ok", True):
        issues.append("trace-error")
    checks = {
        "answer": _score(metrics.get("answer_accuracy")),
        "protocol": _score(metrics.get("protocol_aligned")),
        "clean": _score(metrics.get("clean_protocol_aligned")),
        "strict": _score(metrics.get("strict_success")),
    }
    for label, score in checks.items():
        if score is not None and score < 1.0:
            issues.append(label)
    if family in {"followup", "handshake"}:
        causal = _score(metrics.get("natural_followup_causal"))
        if causal is not None and causal < 1.0:
            issues.append("causal")
    counters = (
        ("path-access", "coordinator_delegated_path_accesses"),
        ("path-access", "parent_path_access"),
        ("roster", "roster_calls"),
        ("observe", "observation_calls"),
        ("failed-cell", "failed_cells"),
        ("duplicate-cell", "duplicate_cells"),
    )
    for label, metric in counters:
        value = _score(metrics.get(metric))
        if value is not None and value > 0.0:
            issues.append(f"{label}={value:g}")
    if not metrics:
        rewards = [_score(value) for value in trace.get("rewards", {}).values()]
        if rewards and sum(score for score in rewards if score is not None) < 1.0:
            issues.append("reward")
    return issues


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tasks = []
    for trace in traces:
        family = _family(trace)
        grouped[family].append(trace)
        rewards = [_score(value) for value in trace.get("rewards", {}).values()]
        tasks.append(
            {
                "name": trace.get("task", {}).get("data", {}).get("name", "unknown"),
                "family": family,
                "reward": sum(score for score in rewards if score is not None),
                "issues": _issues(trace),
                "stop_condition": trace.get("stop_condition"),
            }
        )

    families = {}
    for family, family_traces in sorted(grouped.items()):
        means = {}
        for metric in METRICS:
            values = [
                score for trace in family_traces if (score := _score(trace.get("metrics", {}).get(metric))) is not None
            ]
            if values:
                means[metric] = fmean(values)
        families[family] = {
            "count": len(family_traces),
            "clean_count": sum(not _issues(trace) for trace in family_traces),
            "means": means,
        }
    return {"trace_count": len(traces), "families": families, "tasks": tasks}


def _print(summary: dict[str, Any]) -> None:
    print(f"traces: {summary['trace_count']}")
    for family, data in summary["families"].items():
        fields = [f"n={data['count']}", f"clean={data['clean_count']}/{data['count']}"]
        fields.extend(f"{name}={value:.3f}" for name, value in data["means"].items())
        print(f"{family}: " + " ".join(fields))
    failures = [task for task in summary["tasks"] if task["issues"]]
    print(f"tasks with issues: {len(failures)}/{summary['trace_count']}")
    for task in failures:
        print(f"  {task['name']}: {', '.join(task['issues'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="run directories or traces.jsonl files")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    summary = summarize(load_traces(args.paths))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print(summary)


if __name__ == "__main__":
    main()
