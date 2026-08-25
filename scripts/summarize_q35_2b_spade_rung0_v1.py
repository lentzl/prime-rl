#!/usr/bin/env python3
"""Summarize paired hint/no-hint Rung 0 returns and enforce its frozen gate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_q35_2b_spade_rung0_hints_v1 import (
    _leaks_protected_value,
    _protected_values,
)

HINT_HEADER = "[privileged strategy hint]"


def _load_traces(path: Path) -> list[dict[str, Any]]:
    traces = []
    with path.open() as handle:
        for line in handle:
            envelope = json.loads(line)
            traces.extend(envelope["traces"])
    return traces


def _arm_records(root: Path, prefix: str) -> tuple[dict[str, list[dict]], list[str]]:
    records: dict[str, list[dict]] = defaultdict(list)
    run_names = []
    for run_dir in sorted(root.glob(f"{prefix}-seed-*")):
        run_names.append(run_dir.name)
        for trace_path in sorted(run_dir.glob("*/traces.jsonl")):
            for trace in _load_traces(trace_path):
                key = trace["task"]["key"]
                task_data = trace["task"]["data"]
                records[key].append(
                    {
                        "run": run_dir.name,
                        "trace_id": trace["id"],
                        "family": task_data["family"],
                        "prompt": task_data["prompt"],
                        "score": trace["rewards"]["harness_score"]["score"],
                        "complete": trace.get("is_completed") is True,
                        "ok": trace.get("ok") is True and not trace.get("errors"),
                        "task_data": task_data,
                    }
                )
    return records, run_names


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--hints", type=Path, required=True)
    parser.add_argument("--expected-rollouts-per-arm", type=int, default=4)
    parser.add_argument("--minimum-regret", type=float, default=0.25)
    parser.add_argument("--minimum-positive-tasks", type=int, default=4)
    parser.add_argument("--minimum-positive-async-tasks", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    no_hint, no_hint_runs = _arm_records(args.root, "no-hint")
    hinted, hinted_runs = _arm_records(args.root, "hinted")
    hint_artifact = json.loads(args.hints.read_text())
    hints = hint_artifact["hints"]
    paired_keys = sorted(set(no_hint) & set(hinted))
    key_sets_match = set(no_hint) == set(hinted) == set(hints)

    task_records = []
    positive = []
    positive_async = []
    all_rollouts_valid = True
    leak_free = True
    for key in sorted(set(no_hint) | set(hinted) | set(hints)):
        plain_records = no_hint.get(key, [])
        hinted_records = hinted.get(key, [])
        count_valid = (
            len(plain_records) == args.expected_rollouts_per_arm
            and len(hinted_records) == args.expected_rollouts_per_arm
        )
        completion_valid = count_valid and all(
            record["complete"] and record["ok"]
            for record in plain_records + hinted_records
        )
        prompt_isolation_valid = count_valid and all(
            HINT_HEADER not in record["prompt"] for record in plain_records
        ) and all(HINT_HEADER in record["prompt"] for record in hinted_records)
        all_rollouts_valid &= completion_valid and prompt_isolation_valid
        if not plain_records or not hinted_records or key not in hints:
            task_records.append(
                {
                    "episode_id": key,
                    "paired": False,
                    "no_hint_rollouts": len(plain_records),
                    "hinted_rollouts": len(hinted_records),
                }
            )
            continue

        class Data:
            oracle = plain_records[0]["task_data"]["oracle"]

        class Task:
            data = Data()

        leaks = _leaks_protected_value(hints[key], _protected_values(Task()))
        leak_free &= not leaks
        no_hint_mean = _mean([record["score"] for record in plain_records])
        hinted_mean = _mean([record["score"] for record in hinted_records])
        regret = hinted_mean - no_hint_mean
        qualifies = (
            completion_valid
            and prompt_isolation_valid
            and not leaks
            and regret >= args.minimum_regret
            and no_hint_mean < hinted_mean
        )
        family = plain_records[0]["family"]
        if qualifies:
            positive.append(key)
            if family != "natural_direct_control":
                positive_async.append(key)
        task_records.append(
            {
                "episode_id": key,
                "family": family,
                "paired": True,
                "no_hint_rollouts": len(plain_records),
                "hinted_rollouts": len(hinted_records),
                "no_hint_mean_return": no_hint_mean,
                "hinted_mean_return": hinted_mean,
                "hint_regret": regret,
                "complete_error_free": completion_valid,
                "prompt_isolation_valid": prompt_isolation_valid,
                "leaked_values": leaks,
                "qualifies": qualifies,
            }
        )

    gate_checks = {
        "same_task_keys_between_arms_and_hints": key_sets_match,
        "all_counted_rollouts_complete_error_free_and_isolated": all_rollouts_valid,
        "all_hints_pass_independent_leak_scan": leak_free,
        "minimum_distinct_positive_regret_task_keys": (
            len(positive) >= args.minimum_positive_tasks
        ),
        "minimum_distinct_positive_regret_async_task_keys": (
            len(positive_async) >= args.minimum_positive_async_tasks
        ),
    }
    summary = {
        "schema_version": "qwen35-2b-spade-rung0-summary/v1",
        "gradient_updates": 0,
        "no_hint_runs": no_hint_runs,
        "hinted_runs": hinted_runs,
        "expected_rollouts_per_arm": args.expected_rollouts_per_arm,
        "paired_task_count": len(paired_keys),
        "positive_regret_task_keys": positive,
        "positive_regret_async_task_keys": positive_async,
        "gate_checks": gate_checks,
        "positive_frontier_gate_passed": all(gate_checks.values()),
        "tasks": task_records,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite summary: {args.output}")
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **gate_checks}))


if __name__ == "__main__":
    main()
