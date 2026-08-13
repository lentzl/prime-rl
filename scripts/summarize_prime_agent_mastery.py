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
                if isinstance(saved, list):
                    traces.extend(saved)
                elif isinstance(envelope.get("task"), dict):
                    # Prime-RL's online evaluator stores one TraceRecord directly
                    # on each line; vf-eval exports wrap records in `traces`.
                    traces.append(envelope)
                else:
                    raise SystemExit(f"{path}:{line_number}: expected a trace record or traces list")
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
    name = data.get("name")
    if name:
        return str(name).split("-", 1)[0]
    return _environment(trace)


def _environment(trace: dict[str, Any]) -> str:
    env_name = (trace.get("info") or {}).get("env_name")
    return str(env_name) if env_name else "unknown"


def _trace_error_issue(trace: dict[str, Any]) -> str:
    messages = "\n".join(
        str(error.get("message", "")) for error in trace.get("errors", []) if isinstance(error, dict)
    ).lower()
    if "acp agent produced no visible reply" in messages:
        return "no-visible-reply"
    if "context length" in messages or "maximum context" in messages:
        return "context-length"
    return "trace-error"


def _issues(trace: dict[str, Any]) -> list[str]:
    family = _family(trace)
    metrics = trace.get("metrics", {})
    issues = []
    if not trace.get("ok", True):
        issues.append(_trace_error_issue(trace))
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
        control = _score(metrics.get("bidirectional_control"))
        if control is not None and control < 1.0:
            issues.append("bidirectional-control")
    if family in {"single", "parallel"}:
        control = _score(metrics.get("post_fan_in_control"))
        if control is not None and control < 1.0:
            issues.append("post-fan-in-control")
    counters = [
        ("path-access", "coordinator_delegated_path_accesses"),
        ("roster", "roster_calls"),
        ("observe", "observation_calls"),
        ("failed-cell", "failed_cells"),
        ("duplicate-cell", "duplicate_cells"),
    ]
    if family == "child":
        counters.append(("path-access", "parent_path_access"))
    for label, metric in counters:
        value = _score(metrics.get(metric))
        if value is not None and value > 0.0:
            issues.append(f"{label}={value:g}")
    if not metrics and trace.get("ok", True):
        rewards = [_score(value) for value in trace.get("rewards", {}).values()]
        if rewards and sum(score for score in rewards if score is not None) < 1.0:
            issues.append("reward")
    return issues


def summarize(traces: list[dict[str, Any]], *, include_tasks: bool = True) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tasks = []
    for trace in traces:
        family = _family(trace)
        task_data = trace.get("task", {}).get("data", {})
        grouped[family].append(trace)
        rewards = [_score(value) for value in trace.get("rewards", {}).values()]
        tasks.append(
            {
                "name": task_data.get("name") or _environment(trace),
                "family": family,
                "environment": _environment(trace),
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
    environments = {}
    for env_name in sorted({_environment(trace) for trace in traces}):
        env_traces = [trace for trace in traces if _environment(trace) == env_name]
        environments[env_name] = {
            "count": len(env_traces),
            "clean_count": sum(not _issues(trace) for trace in env_traces),
            "families": sorted({_family(trace) for trace in env_traces}),
        }
    policy_versions = sorted(
        {
            int(version)
            for trace in traces
            if isinstance((version := (trace.get("info") or {}).get("policy_version")), int)
            and not isinstance(version, bool)
        }
    )
    summary = {
        "trace_count": len(traces),
        "issue_count": sum(bool(task["issues"]) for task in tasks),
        "policy_versions": policy_versions,
        "environments": environments,
        "families": families,
    }
    if include_tasks:
        summary["tasks"] = tasks
    return summary


def summarize_by_policy_version(
    traces: list[dict[str, Any]], *, include_tasks: bool = True
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        version = (trace.get("info") or {}).get("policy_version")
        key = str(version) if isinstance(version, int) and not isinstance(version, bool) else "unknown"
        grouped[key].append(trace)
    return {
        key: summarize(grouped[key], include_tasks=include_tasks)
        for key in sorted(grouped, key=lambda value: (value == "unknown", int(value) if value != "unknown" else 0))
    }


def _print(summary: dict[str, Any]) -> None:
    print(f"traces: {summary['trace_count']}")
    if summary["policy_versions"]:
        print("policy versions: " + ", ".join(map(str, summary["policy_versions"])))
    for env_name, data in summary["environments"].items():
        print(f"environment {env_name}: n={data['count']} clean={data['clean_count']}/{data['count']}")
    for family, data in summary["families"].items():
        fields = [f"n={data['count']}", f"clean={data['clean_count']}/{data['count']}"]
        fields.extend(f"{name}={value:.3f}" for name, value in data["means"].items())
        print(f"{family}: " + " ".join(fields))
    failures = [task for task in summary.get("tasks", []) if task["issues"]]
    print(f"tasks with issues: {summary['issue_count']}/{summary['trace_count']}")
    for task in failures:
        print(f"  {task['name']}: {', '.join(task['issues'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="run directories or traces.jsonl files")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--by-policy-version",
        action="store_true",
        help="report each evaluated checkpoint separately instead of blending versions",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="omit per-task records while retaining aggregate issue counts",
    )
    args = parser.parse_args()
    traces = load_traces(args.paths)
    if args.by_policy_version:
        summaries = summarize_by_policy_version(traces, include_tasks=not args.summary_only)
        if args.json:
            print(json.dumps({"policy_versions": summaries}, indent=2, sort_keys=True))
        else:
            for version, summary in summaries.items():
                print(f"policy version {version}")
                _print(summary)
        return

    summary = summarize(traces, include_tasks=not args.summary_only)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print(summary)


if __name__ == "__main__":
    main()
