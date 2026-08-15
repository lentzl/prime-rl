"""Summarize saved Prime Agent mastery v2 traces by capability family."""

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
    "strict_success",
    "dense_reward",
    "state_retained",
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
    if trace.get("task", {}).get("type") == "OolongSynthTask":
        return "oolong"
    name = data.get("name")
    return str(name).split("-", 1)[0] if name else "unknown"


def _issues(trace: dict[str, Any]) -> list[str]:
    family = _family(trace)
    metrics = trace.get("metrics", {})
    issues = []
    if not trace.get("ok", True):
        errors = "\n".join(
            str(error.get("message", "")) for error in trace.get("errors", []) if isinstance(error, dict)
        ).lower()
        issue = "no-visible-reply" if "acp agent produced no visible reply" in errors else "trace-error"
        issues.append(issue)
    checks = {
        "answer": _score(metrics.get("answer_accuracy")),
        "protocol": _score(metrics.get("protocol_aligned")),
        "clean": _score(metrics.get("clean_protocol_aligned")),
        "strict": _score(metrics.get("strict_success")),
    }
    issues.extend(label for label, score in checks.items() if score is not None and score < 1.0)
    if family in {"followup", "handshake"}:
        for label, metric in (
            ("causal", "natural_followup_causal"),
            ("bidirectional-control", "bidirectional_control"),
        ):
            score = _score(metrics.get(metric))
            if score is not None and score < 1.0:
                issues.append(label)
    if family in {"single", "parallel"}:
        score = _score(metrics.get("post_fan_in_control"))
        if score is not None and score < 1.0:
            issues.append("post-fan-in-control")
    for label, metric in (
        ("path-access", "coordinator_delegated_path_accesses"),
        ("roster", "roster_calls"),
        ("observe", "observation_calls"),
        ("failed-cell", "failed_cells"),
        ("duplicate-cell", "duplicate_cells"),
    ):
        value = _score(metrics.get(metric))
        if value is not None and value > 0.0:
            issues.append(f"{label}={value:g}")
    if family == "child":
        value = _score(metrics.get("parent_path_access"))
        if value is not None and value > 0.0:
            issues.append(f"path-access={value:g}")
    if not metrics and trace.get("ok", True):
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
        data = trace.get("task", {}).get("data", {})
        rewards = [_score(value) for value in trace.get("rewards", {}).values()]
        tasks.append(
            {
                "name": data.get("name") or "unknown",
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
            "mean_reward": fmean(
                sum(score for value in trace.get("rewards", {}).values() if (score := _score(value)) is not None)
                for trace in family_traces
            ),
            "means": means,
        }
    return {
        "trace_count": len(traces),
        "issue_count": sum(bool(task["issues"]) for task in tasks),
        "families": families,
        "tasks": tasks,
    }


def require_valid_traces(traces: list[dict[str, Any]]) -> None:
    def is_behavioral_timeout(trace: dict[str, Any]) -> bool:
        errors = trace.get("errors")
        if trace.get("stop_condition") != "error" or not isinstance(errors, list) or not errors:
            return False
        return all(
            isinstance(error, dict)
            and error.get("type") == "HarnessError"
            and str(error.get("message", "")).startswith(
                "agent timeout: rollout exceeded its "
            )
            and str(error.get("message", "")).endswith(" budget")
            for error in errors
        )

    invalid = [
        str(trace.get("id") or trace.get("task", {}).get("data", {}).get("name"))
        for trace in traces
        if (trace.get("ok") is not True or trace.get("errors"))
        and not is_behavioral_timeout(trace)
    ]
    if invalid:
        preview = ", ".join(invalid[:8])
        suffix = "" if len(invalid) <= 8 else f" (+{len(invalid) - 8} more)"
        raise SystemExit(f"found {len(invalid)} invalid trace(s): {preview}{suffix}")


def _print(summary: dict[str, Any]) -> None:
    print(f"traces: {summary['trace_count']}")
    for family, data in summary["families"].items():
        fields = [
            f"n={data['count']}",
            f"clean={data['clean_count']}/{data['count']}",
            f"reward={data['mean_reward']:.3f}",
        ]
        fields.extend(f"{name}={value:.3f}" for name, value in data["means"].items())
        print(f"{family}: " + " ".join(fields))
    failures = [task for task in summary["tasks"] if task["issues"]]
    print(f"tasks with issues: {summary['issue_count']}/{summary['trace_count']}")
    for task in failures:
        print(f"  {task['name']}: {', '.join(task['issues'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--require-valid-traces", action="store_true")
    args = parser.parse_args()
    traces = load_traces(args.paths)
    if args.require_valid_traces:
        require_valid_traces(traces)
    summary = summarize(traces)
    if args.expected_count is not None and summary["trace_count"] != args.expected_count:
        raise SystemExit(f"expected {args.expected_count} traces, found {summary['trace_count']}")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print(summary)


if __name__ == "__main__":
    main()
