#!/usr/bin/env python3
"""Compare completed Memory V2 tranche candidates against the frozen base."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_prime_agent_mastery import load_traces as load_mastery_traces
from scripts.summarize_prime_agent_mastery import summarize as summarize_mastery
from scripts.summarize_programmatic_memory_eval import load_traces as load_memory_traces
from scripts.summarize_programmatic_memory_eval import report as report_memory

DEFAULT_LABELS = ("base", "step-1", "step-2", "step-4", "step-8")
DEFAULT_MEMORY_COUNT = 396
DEFAULT_MASTERY_COUNT = 74


def _strict_success(trace: dict[str, Any]) -> bool:
    return float(trace.get("metrics", {}).get("strict_success", 0.0)) >= 1.0


def _memory_key(trace: dict[str, Any]) -> tuple[str, str, int, str]:
    data = trace.get("task", {}).get("data", {})
    return (
        str(data.get("split", "unknown")),
        str(data.get("family", "unknown")),
        int(data.get("idx", -1)),
        str(data.get("name", "unknown")),
    )


def _unique_index(
    traces: list[dict[str, Any]],
    *,
    label: str,
    key: Callable[[dict[str, Any]], tuple[str, str, int, str]],
) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    indexed: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for trace in traces:
        trace_key = key(trace)
        if trace_key in indexed:
            raise SystemExit(f"duplicate frozen task for {label}: {trace_key}")
        indexed[trace_key] = trace
    return indexed


def _paired_counts(
    reference: dict[tuple[str, str, int, str], dict[str, Any]],
    candidate: dict[tuple[str, str, int, str], dict[str, Any]],
) -> dict[str, Any]:
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        unexpected = sorted(candidate.keys() - reference.keys())
        raise SystemExit(
            "frozen memory task identities differ: "
            f"missing={missing[:3]} unexpected={unexpected[:3]}"
        )

    def counts(keys: list[tuple[str, str, int, str]]) -> dict[str, int]:
        result = {
            "count": len(keys),
            "reference_success": 0,
            "candidate_success": 0,
            "gain": 0,
            "loss": 0,
            "both_success": 0,
            "both_failure": 0,
        }
        for trace_key in keys:
            reference_success = _strict_success(reference[trace_key])
            candidate_success = _strict_success(candidate[trace_key])
            result["reference_success"] += int(reference_success)
            result["candidate_success"] += int(candidate_success)
            result["gain"] += int(not reference_success and candidate_success)
            result["loss"] += int(reference_success and not candidate_success)
            result["both_success"] += int(reference_success and candidate_success)
            result["both_failure"] += int(not reference_success and not candidate_success)
        result["net_gain"] = result["gain"] - result["loss"]
        return result

    keys = sorted(reference)
    by_split = {
        split: counts([trace_key for trace_key in keys if trace_key[0] == split])
        for split in sorted({trace_key[0] for trace_key in keys})
    }
    by_family = {
        family: counts([trace_key for trace_key in keys if trace_key[1] == family])
        for family in sorted({trace_key[1] for trace_key in keys})
    }
    return {"overall": counts(keys), "by_split": by_split, "by_family": by_family}


def _numeric_deltas(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    deltas = {}
    for key in sorted(reference.keys() | candidate.keys()):
        reference_value = reference.get(key, 0.0)
        candidate_value = candidate.get(key, 0.0)
        if not isinstance(reference_value, (int, float)) or not isinstance(candidate_value, (int, float)):
            continue
        delta = float(candidate_value) - float(reference_value)
        if not math.isfinite(delta):
            raise SystemExit(f"non-finite comparison value for {key}")
        deltas[key] = delta
    return deltas


def _memory_deltas(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result = {
        "overall_means": _numeric_deltas(reference["overall"]["means"], candidate["overall"]["means"]),
        "by_split": {},
        "by_family": {},
        "typed_failure_code_mass": _numeric_deltas(
            reference["overall"]["typed_failures"]["code_mass"],
            candidate["overall"]["typed_failures"]["code_mass"],
        ),
        "typed_failure_category_mass": _numeric_deltas(
            reference["overall"]["typed_failures"]["category_mass"],
            candidate["overall"]["typed_failures"]["category_mass"],
        ),
    }
    for group_name in ("by_split", "by_family"):
        reference_groups = reference[group_name]
        candidate_groups = candidate[group_name]
        if reference_groups.keys() != candidate_groups.keys():
            raise SystemExit(f"memory {group_name} groups differ from the frozen base")
        result[group_name] = {
            name: _numeric_deltas(reference_groups[name]["means"], candidate_groups[name]["means"])
            for name in sorted(reference_groups)
        }
    return result


def _mastery_deltas(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if reference["families"].keys() != candidate["families"].keys():
        raise SystemExit("mastery families differ from the frozen base")
    if reference["environments"].keys() != candidate["environments"].keys():
        raise SystemExit("mastery environments differ from the frozen base")

    families = {}
    for family in sorted(reference["families"]):
        reference_family = reference["families"][family]
        candidate_family = candidate["families"][family]
        if reference_family["count"] != candidate_family["count"]:
            raise SystemExit(f"mastery family count differs for {family}")
        families[family] = {
            "count": reference_family["count"],
            "clean_count_delta": candidate_family["clean_count"] - reference_family["clean_count"],
            "mean_deltas": _numeric_deltas(reference_family["means"], candidate_family["means"]),
        }

    environments = {}
    for environment in sorted(reference["environments"]):
        reference_environment = reference["environments"][environment]
        candidate_environment = candidate["environments"][environment]
        if reference_environment["count"] != candidate_environment["count"]:
            raise SystemExit(f"mastery environment count differs for {environment}")
        environments[environment] = {
            "count": reference_environment["count"],
            "clean_count_delta": (
                candidate_environment["clean_count"] - reference_environment["clean_count"]
            ),
        }

    return {
        "issue_count_delta": candidate["issue_count"] - reference["issue_count"],
        "families": families,
        "environments": environments,
    }


def _validate_typed_contracts(label: str, memory_summary: dict[str, Any]) -> list[str]:
    typed = memory_summary["overall"]["typed_failures"]
    checks = {
        "unexpected_schema_count": typed["unexpected_schema_count"],
        "answer_free_violation_count": typed["answer_free_violation_count"],
        "message_mismatch_count": typed["message_mismatch_count"],
        "success_contract_count": typed["success_contract_count"],
    }
    return [f"{label}:{name}={value}" for name, value in checks.items() if value]


def compare(
    root: Path,
    *,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    expected_memory_count: int = DEFAULT_MEMORY_COUNT,
    expected_mastery_count: int = DEFAULT_MASTERY_COUNT,
) -> dict[str, Any]:
    if not labels or labels[0] != "base" or len(set(labels)) != len(labels):
        raise SystemExit("labels must be unique and begin with base")

    memory_traces = {}
    memory_summaries = {}
    mastery_summaries = {}
    contract_violations = []
    for label in labels:
        model_root = root / label
        required = (
            model_root / "QUALIFICATION_COMPLETE",
            model_root / "memory-summary.json",
            model_root / "mastery-summary.json",
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise SystemExit(f"qualification is incomplete for {label}: {missing[0]}")

        traces = load_memory_traces(model_root / "memory")
        if len(traces) != expected_memory_count:
            raise SystemExit(
                f"unexpected memory trace count for {label}: {len(traces)} != {expected_memory_count}"
            )
        memory_traces[label] = _unique_index(traces, label=label, key=_memory_key)
        memory_summary = report_memory(traces)
        saved_memory_summary = json.loads((model_root / "memory-summary.json").read_text(encoding="utf-8"))
        if memory_summary != saved_memory_summary:
            raise SystemExit(f"saved memory summary does not match traces for {label}")
        memory_summaries[label] = memory_summary
        contract_violations.extend(_validate_typed_contracts(label, memory_summary))

        mastery_paths = sorted((model_root / "mastery").rglob("traces.jsonl"))
        mastery_traces = load_mastery_traces(mastery_paths)
        if len(mastery_traces) != expected_mastery_count:
            raise SystemExit(
                f"unexpected mastery trace count for {label}: {len(mastery_traces)} != {expected_mastery_count}"
            )
        mastery_summary = summarize_mastery(mastery_traces, include_tasks=False)
        saved_mastery_summary = json.loads((model_root / "mastery-summary.json").read_text(encoding="utf-8"))
        if mastery_summary != saved_mastery_summary:
            raise SystemExit(f"saved mastery summary does not match traces for {label}")
        mastery_summaries[label] = mastery_summary

    reference = labels[0]
    comparisons = {}
    for label in labels[1:]:
        comparisons[label] = {
            "memory_paired": _paired_counts(memory_traces[reference], memory_traces[label]),
            "memory_deltas": _memory_deltas(memory_summaries[reference], memory_summaries[label]),
            "mastery_deltas": _mastery_deltas(mastery_summaries[reference], mastery_summaries[label]),
        }

    return {
        "schema_version": 1,
        "reference": reference,
        "labels": list(labels),
        "expected_counts": {"memory": expected_memory_count, "mastery": expected_mastery_count},
        "typed_contract_violations": contract_violations,
        "comparisons": comparisons,
        "decision": "REVIEW_REQUIRED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--expected-memory-count", type=int, default=DEFAULT_MEMORY_COUNT)
    parser.add_argument("--expected-mastery-count", type=int, default=DEFAULT_MASTERY_COUNT)
    args = parser.parse_args()

    result = compare(
        args.root,
        labels=tuple(args.labels),
        expected_memory_count=args.expected_memory_count,
        expected_mastery_count=args.expected_mastery_count,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
