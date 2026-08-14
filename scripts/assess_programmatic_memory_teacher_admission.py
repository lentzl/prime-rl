#!/usr/bin/env python3
"""Assess the preregistered Programmatic Episodic Memory v2 SDFT teacher gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

ARMS = {
    "familiar_unconditioned": ("336-*", 50),
    "familiar_conditioned": ("337-*", 50),
    "ood_unconditioned": ("338-*", 16),
    "ood_conditioned": ("339-*", 16),
}
CORE_METRICS = (
    "strict_success",
    "answer_correct",
    "retrieval_decision",
    "grounded_answer",
    "valid_tool_behavior",
    "bounded_retrieval",
    "no_repeated_cell",
    "persistent_index_reuse",
    "stale_note_resolution",
    "context_reset_recovery",
    "current_turn_override",
)
DIAGNOSTIC_METRICS = ("expected_value_present",)


def task_identity(trace: dict) -> str:
    """Hash the frozen task payload while ignoring the intended hint intervention."""
    data = dict(trace.get("task", {}).get("data", {}))
    data.pop("system_prompt", None)
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_traces(path: Path) -> list[dict]:
    traces: list[dict] = []
    for trace_file in sorted(path.rglob("traces.jsonl")):
        for line in trace_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            traces.extend(record.get("traces", [record]))
    return traces


def expected_value_present(trace: dict) -> float:
    expected = [str(value).strip() for value in trace.get("task", {}).get("data", {}).get("expected_answers", [])]
    answers = []
    for node in trace.get("nodes", []):
        message = node.get("message", {})
        content = message.get("content")
        if node.get("sampled") and message.get("role") == "assistant" and isinstance(content, str) and content.strip():
            answers.append(content.strip())
    observed = answers[-len(expected) :] if expected else []
    if len(observed) != len(expected):
        return 0.0
    return float(
        all(
            re.search(rf"(?<!\w){re.escape(value)}(?!\w)", answer, re.IGNORECASE)
            for answer, value in zip(observed, expected)
        )
    )


def summarize(traces: list[dict]) -> dict:
    identities = Counter(task_identity(trace) for trace in traces)
    return {
        "count": len(traces),
        "clean_count": sum(bool(trace.get("ok")) for trace in traces),
        "task_identity_counts": dict(sorted(identities.items())),
        "means": {
            metric: mean(float(trace.get("metrics", {}).get(metric, 0.0)) for trace in traces)
            if traces
            else 0.0
            for metric in CORE_METRICS
        },
        "diagnostics": {
            "expected_value_present": mean(expected_value_present(trace) for trace in traces)
            if traces
            else 0.0
        },
    }


def relative_error_reduction(before: float, after: float) -> float:
    error = 1.0 - before
    if error <= 0.0:
        return 0.0
    return (after - before) / error


def early_rejection_reasons(arms: dict[str, dict]) -> list[str]:
    """Identify conditioned metrics that cannot recover before the arm is full."""
    reasons: list[str] = []
    thresholds = {"familiar": 0.90, "ood": 0.80}
    for split, threshold in thresholds.items():
        arm_name = f"{split}_conditioned"
        expected = ARMS[arm_name][1]
        arm = arms.get(arm_name, {"count": 0, "clean_count": 0, "means": {}})
        observed = min(int(arm["count"]), expected)
        remaining = expected - observed
        for metric in CORE_METRICS:
            score = float(arm["means"].get(metric, 0.0)) * observed
            maximum = (score + remaining) / expected
            if maximum < threshold:
                reasons.append(
                    f"{split}: conditioned {metric} can reach at most "
                    f"{maximum:.3f}, below {threshold:.2f}"
                )
    return reasons


def assess(arms: dict[str, dict]) -> dict:
    failures: list[str] = []
    for name, (_, expected) in ARMS.items():
        arm = arms.get(name, {"count": 0, "clean_count": 0, "means": {}})
        if arm["count"] != expected:
            failures.append(f"{name}: expected {expected} traces, found {arm['count']}")
        if arm["clean_count"] != arm["count"]:
            failures.append(f"{name}: contains errored traces")

    for split in ("familiar", "ood"):
        unconditioned = arms[f"{split}_unconditioned"].get("task_identity_counts")
        conditioned = arms[f"{split}_conditioned"].get("task_identity_counts")
        if (
            unconditioned is not None
            and conditioned is not None
            and unconditioned != conditioned
        ):
            failures.append(
                f"{split}: conditioned and unconditioned arms do not contain "
                "the same frozen task identities"
            )

    thresholds = {"familiar": 0.90, "ood": 0.80}
    for split, threshold in thresholds.items():
        conditioned = arms[f"{split}_conditioned"]["means"]
        unconditioned = arms[f"{split}_unconditioned"]["means"]
        for metric in CORE_METRICS:
            value = conditioned[metric]
            if value < threshold:
                failures.append(
                    f"{split}: conditioned {metric} {value:.3f} is below {threshold:.2f}"
                )
            if value + 0.05 < unconditioned[metric]:
                failures.append(
                    f"{split}: conditioned {metric} regressed by more than 0.05"
                )

    expected_by_split = {
        split: ARMS[f"{split}_conditioned"][1] for split in thresholds
    }
    total = sum(expected_by_split.values())
    conditioned_strict = sum(
        arms[f"{split}_conditioned"]["means"]["strict_success"]
        * expected_by_split[split]
        for split in thresholds
    ) / total
    unconditioned_strict = sum(
        arms[f"{split}_unconditioned"]["means"]["strict_success"]
        * expected_by_split[split]
        for split in thresholds
    ) / total
    absolute_gain = conditioned_strict - unconditioned_strict
    error_reduction = relative_error_reduction(unconditioned_strict, conditioned_strict)
    substantial_gain = absolute_gain >= 0.08 or (
        absolute_gain >= 0.04 and error_reduction >= 0.50
    )
    if not substantial_gain:
        failures.append(
            "conditioned strict behavior lacks a preregistered substantial gain "
            f"(absolute={absolute_gain:.3f}, error_reduction={error_reduction:.3f})"
        )

    early_rejections = early_rejection_reasons(arms)
    return {
        "schema_version": 1,
        "admission_pass": not failures,
        "admission_still_possible": not early_rejections,
        "early_rejection_reasons": early_rejections,
        "criteria": {
            "expected_traces": {name: expected for name, (_, expected) in ARMS.items()},
            "paired_task_identity": "exact multiset match within each split",
            "conditioned_core_minimum": thresholds,
            "maximum_core_regression": 0.05,
            "substantial_strict_gain": (
                "absolute >= 0.08 OR (absolute >= 0.04 AND relative error reduction >= 0.50)"
            ),
        },
        "comparison": {
            "unconditioned_strict": unconditioned_strict,
            "conditioned_strict": conditioned_strict,
            "absolute_gain": absolute_gain,
            "relative_error_reduction": error_reduction,
        },
        "arms": arms,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    summaries: dict[str, dict] = {}
    for name, (pattern, _) in ARMS.items():
        matches = sorted(args.root.glob(f"*/{pattern}"))
        if len(matches) > 1:
            raise ValueError(f"ambiguous {name} result directories: {matches}")
        summaries[name] = summarize(load_traces(matches[0])) if matches else summarize([])
    report = assess(summaries)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.require_pass and not report["admission_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
