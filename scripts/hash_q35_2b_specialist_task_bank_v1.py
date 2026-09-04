#!/usr/bin/env python3
"""Materialize and hash a frozen specialist task bank without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from subagent_communication_v1.taskset import (
    SubagentCommunicationConfig,
    SubagentCommunicationTaskset,
)


def task_bank(config_path: Path) -> tuple[list[Any], str]:
    document = tomllib.loads(config_path.read_text(encoding="utf-8"))
    taskset = document.get("env", {}).get("taskset")
    if not isinstance(taskset, dict) or taskset.get("id") != "subagent-communication-v1":
        raise ValueError("config does not select the specialist taskset")
    config_values = {key: value for key, value in taskset.items() if key != "id"}
    tasks = list(SubagentCommunicationTaskset(SubagentCommunicationConfig(**config_values)).load())
    payload = [task.data.model_dump(mode="json") for task in tasks]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return tasks, hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    tasks, digest = task_bank(args.config)
    if args.expected_sha256 is not None and digest != args.expected_sha256:
        raise ValueError(
            f"specialist task bank hash mismatch: expected {args.expected_sha256}, got {digest}"
        )
    print(
        json.dumps(
            {
                "config": str(args.config),
                "task_bank_sha256": digest,
                "tasks": len(tasks),
                "families": sorted({task.data.family for task in tasks}),
                "template_variants": sorted(
                    {task.data.template_variant for task in tasks}
                ),
                "first_task": tasks[0].data.name if tasks else None,
                "last_task": tasks[-1].data.name if tasks else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
