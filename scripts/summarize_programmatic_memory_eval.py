#!/usr/bin/env python3
"""Summarize frozen Programmatic Episodic Memory evaluations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

CORE_METRICS = (
    "strict_success",
    "dense_reward",
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
    "efficient_calls",
)
FEEDBACK_SCHEMA = "programmatic-episodic-memory-v2/causal-feedback/v1"


def load_traces(root: Path) -> list[dict]:
    traces: list[dict] = []
    for path in sorted(root.rglob("traces.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            traces.extend(record.get("traces", [record]))
    return traces


def summarize(traces: list[dict]) -> dict:
    def strict_success(trace: dict) -> bool:
        return float(trace.get("metrics", {}).get("strict_success", 0.0)) >= 1.0

    failures = [trace for trace in traces if not strict_success(trace)]
    contracts = [trace.get("info", {}).get("feedback_contract") for trace in failures]
    typed = [contract for contract in contracts if isinstance(contract, dict)]
    success_contracts = sum(
        strict_success(trace) and isinstance(trace.get("info", {}).get("feedback_contract"), dict) for trace in traces
    )
    schemas = Counter(str(contract.get("schema_version")) for contract in typed)
    codes = Counter(str(contract.get("code")) for contract in typed)
    categories = Counter(str(contract.get("category")) for contract in typed)
    denominator = len(failures)

    return {
        "count": len(traces),
        "clean_count": sum(bool(trace.get("ok")) for trace in traces),
        "failure_count": denominator,
        "means": {
            metric: mean(float(trace.get("metrics", {}).get(metric, 0.0)) for trace in traces) if traces else 0.0
            for metric in CORE_METRICS
        },
        "typed_failures": {
            "count": len(typed),
            "coverage": len(typed) / denominator if denominator else 1.0,
            "untyped_count": denominator - len(typed),
            "success_contract_count": success_contracts,
            "schema_counts": dict(sorted(schemas.items())),
            "unexpected_schema_count": sum(count for schema, count in schemas.items() if schema != FEEDBACK_SCHEMA),
            "answer_free_violation_count": sum(contract.get("answer_free") is not True for contract in typed),
            "message_mismatch_count": sum(
                contract.get("message") != trace.get("info", {}).get("feedback")
                for trace, contract in zip(failures, contracts, strict=True)
                if isinstance(contract, dict)
            ),
            "code_counts": dict(sorted(codes.items())),
            "code_mass": {code: count / denominator if denominator else 0.0 for code, count in sorted(codes.items())},
            "category_counts": dict(sorted(categories.items())),
            "category_mass": {
                category: count / denominator if denominator else 0.0 for category, count in sorted(categories.items())
            },
        },
    }


def report(traces: list[dict]) -> dict:
    by_split: defaultdict[str, list[dict]] = defaultdict(list)
    by_family: defaultdict[str, list[dict]] = defaultdict(list)
    for trace in traces:
        data = trace.get("task", {}).get("data", {})
        by_split[str(data.get("split", "unknown"))].append(trace)
        by_family[str(data.get("family", "unknown"))].append(trace)
    return {
        "schema_version": 1,
        "feedback_schema": FEEDBACK_SCHEMA,
        "overall": summarize(traces),
        "by_split": {key: summarize(value) for key, value in sorted(by_split.items())},
        "by_family": {key: summarize(value) for key, value in sorted(by_family.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = report(load_traces(args.root))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
