#!/usr/bin/env python3
"""Materialize P1 source task groups and prove disjointness from frozen P0 keys."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tomllib
from pathlib import Path

P0_SEED = 20270909
P0_OFFSET = 70000


def _taskset_types():
    from source_worker_first_call_v1.taskset import (
        SourceWorkerFirstCallConfig,
        SourceWorkerFirstCallTaskset,
    )

    return SourceWorkerFirstCallConfig, SourceWorkerFirstCallTaskset


def _bank(source_configs: list[dict[str, object]]) -> tuple[list[str], str, str]:
    config_type, taskset_type = _taskset_types()
    tasks = []
    for raw in source_configs:
        config = config_type(**{key: value for key, value in raw.items() if key != "id"})
        source_tasks = list(taskset_type(config).load())
        if len(source_tasks) != 32 or len({task.key for task in source_tasks}) != 32:
            raise ValueError("each P1 family source must materialize exactly 32 unique tasks")
        tasks.extend(source_tasks)
    rows = sorted(
        ((task.key, task.data.model_dump(mode="json", exclude_none=True)) for task in tasks),
        key=lambda row: row[0],
    )
    keys = [key for key, _ in rows]
    if len(rows) != 64 or len(set(keys)) != 64:
        raise ValueError("source task bank must contain 32 unique tasks per family")
    encoded = json.dumps(
        [{"key": key, "data": payload} for key, payload in rows],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    key_encoded = ("\n".join(keys) + "\n").encode()
    return keys, hashlib.sha256(encoded).hexdigest(), hashlib.sha256(key_encoded).hexdigest()


def materialize(path: Path) -> dict[str, object]:
    payload = tomllib.loads(path.read_text())
    sources = payload.get("orchestrator", {}).get("train", {}).get("source", [])
    source_configs = [source.get("env", {}).get("taskset", {}) for source in sources]
    p1_keys, p1_bank_sha, p1_key_sha = _bank(source_configs)
    p0_configs = copy.deepcopy(source_configs)
    for raw in p0_configs:
        raw["seed"] = P0_SEED
        raw["instance_offset"] = P0_OFFSET
    p0_keys, p0_bank_sha, p0_key_sha = _bank(p0_configs)
    if set(p1_keys) & set(p0_keys):
        raise ValueError("P1 full task bank overlaps the full P0 task bank")
    return {
        "tasks": len(p1_keys),
        "tasks_per_family": 32,
        "task_keys": p1_keys,
        "task_key_set_sha256": p1_key_sha,
        "task_bank_sha256": p1_bank_sha,
        "p0_tasks": len(p0_keys),
        "p0_task_keys": p0_keys,
        "p0_task_key_set_sha256": p0_key_sha,
        "p0_task_bank_sha256": p0_bank_sha,
        "p0_disjoint": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--expected-task-bank-sha256", required=True)
    parser.add_argument("--expected-task-key-set-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = materialize(args.config)
    if report["task_bank_sha256"] != args.expected_task_bank_sha256:
        raise SystemExit("P1 task-bank hash mismatch")
    if report["task_key_set_sha256"] != args.expected_task_key_set_sha256:
        raise SystemExit("P1 task-key-set hash mismatch")
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=False)
        with args.output.open("x") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
