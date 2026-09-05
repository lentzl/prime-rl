#!/usr/bin/env python3
"""Materialize the full P2 bank and prove disjointness from frozen P0 and P1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tomllib
from pathlib import Path

PRIOR_BANKS = {
    "P0": {
        "seed": 20270909,
        "offset": 70000,
        "task_bank_sha256": "5c1c553894af7cb5bbfca9faa72747778e9434ea03cb75030dc838aada97099a",
        "task_key_set_sha256": "9f314c1c808adb73b7d9036df62a7c15bf758c2f4af78d11fd20ba2d44a00aef",
    },
    "P1": {
        "seed": 20270917,
        "offset": 71000,
        "task_bank_sha256": "fd73d39273de3e10a89bffb881d59b4ae47e36e30b6d8f12644ad47ce8ef59de",
        "task_key_set_sha256": "b368b2964332d8f9834280956e3d592d0a6b8cc53fe4a97b3fa9bcdbbecd7130",
    },
}


def _taskset_types():
    from source_worker_first_call_v1.taskset import (
        SourceWorkerFirstCallConfig,
        SourceWorkerFirstCallTaskset,
    )

    return SourceWorkerFirstCallConfig, SourceWorkerFirstCallTaskset


def _bank(source_configs: list[dict[str, object]]) -> dict[str, object]:
    config_type, taskset_type = _taskset_types()
    rows = []
    source_counts: dict[str, int] = {}
    for raw in source_configs:
        source_id = str(raw.get("families"))
        config = config_type(**{key: value for key, value in raw.items() if key != "id"})
        tasks = list(taskset_type(config).load())
        if len(tasks) != 32 or len({task.key for task in tasks}) != 32:
            raise ValueError("each source family must materialize exactly 32 unique tasks")
        source_counts[source_id] = len(tasks)
        rows.extend(
            (task.key, task.data.model_dump(mode="json", exclude_none=True))
            for task in tasks
        )
    rows.sort(key=lambda row: row[0])
    keys = [key for key, _ in rows]
    if len(source_counts) != 2 or set(source_counts.values()) != {32}:
        raise ValueError("task bank must contain exactly two 32-task sources")
    if len(rows) != 64 or len(set(keys)) != 64:
        raise ValueError("task bank must contain 64 unique tasks")
    bank_bytes = json.dumps(
        [{"key": key, "data": data} for key, data in rows],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    key_bytes = ("\n".join(keys) + "\n").encode()
    return {
        "tasks": 64,
        "tasks_per_family": 32,
        "task_keys": keys,
        "task_bank_sha256": hashlib.sha256(bank_bytes).hexdigest(),
        "task_key_set_sha256": hashlib.sha256(key_bytes).hexdigest(),
    }


def _at_seed(
    source_configs: list[dict[str, object]], seed: int, offset: int
) -> dict[str, object]:
    configs = copy.deepcopy(source_configs)
    for raw in configs:
        raw["seed"] = seed
        raw["instance_offset"] = offset
    return _bank(configs)


def materialize(path: Path) -> dict[str, object]:
    payload = tomllib.loads(path.read_text())
    sources = payload.get("orchestrator", {}).get("train", {}).get("source", [])
    source_configs = [source.get("env", {}).get("taskset", {}) for source in sources]
    p2 = _bank(source_configs)
    prior = {
        label: _at_seed(source_configs, int(freeze["seed"]), int(freeze["offset"]))
        for label, freeze in PRIOR_BANKS.items()
    }
    for label, freeze in PRIOR_BANKS.items():
        if any(prior[label][key] != freeze[key] for key in ("task_bank_sha256", "task_key_set_sha256")):
            raise ValueError(f"frozen {label} task bank did not reproduce")
    key_sets = {"P2": set(p2["task_keys"]), **{label: set(bank["task_keys"]) for label, bank in prior.items()}}
    overlaps = {
        "P2_P0": sorted(key_sets["P2"] & key_sets["P0"]),
        "P2_P1": sorted(key_sets["P2"] & key_sets["P1"]),
        "P0_P1": sorted(key_sets["P0"] & key_sets["P1"]),
    }
    if any(overlaps.values()):
        raise ValueError("P0, P1, and P2 full task banks must be pairwise disjoint")
    return {
        **p2,
        "prior_banks": prior,
        "p0_tasks": prior["P0"]["tasks"],
        "p0_task_keys": prior["P0"]["task_keys"],
        "p0_task_bank_sha256": prior["P0"]["task_bank_sha256"],
        "p0_task_key_set_sha256": prior["P0"]["task_key_set_sha256"],
        "p1_tasks": prior["P1"]["tasks"],
        "p1_task_keys": prior["P1"]["task_keys"],
        "p1_task_bank_sha256": prior["P1"]["task_bank_sha256"],
        "p1_task_key_set_sha256": prior["P1"]["task_key_set_sha256"],
        "pairwise_disjoint": True,
        "overlaps": overlaps,
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
        raise SystemExit("P2 task-bank hash mismatch")
    if report["task_key_set_sha256"] != args.expected_task_key_set_sha256:
        raise SystemExit("P2 task-key-set hash mismatch")
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=False)
        with args.output.open("x") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
