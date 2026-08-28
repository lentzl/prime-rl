#!/usr/bin/env python3
"""Build a bounded canonical child-action SFT leak from hard-success traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_prime_agent_role_sft_v1 import _wire_message, sha256_file

SCHEMA_VERSION = "qwen35-2b-child-action-booster-sft/v1"
CONTRACT_RECOVERY_SCHEMA_VERSION = "qwen35-2b-child-action-booster-sft/v3"
TRUNCATED_STOPS = {
    "max_turns",
    "max_input_tokens",
    "max_output_tokens",
    "max_total_tokens",
    "context_length",
    "generation_truncated",
}


def _score(value: Any) -> float:
    if not isinstance(value, dict):
        return 0.0
    score = value.get("score", value.get("value", 0.0))
    return float(score) if isinstance(score, int | float) else 0.0


def _path(nodes: list[dict[str, Any]], target: int) -> list[int]:
    result = []
    seen = set()
    current: int | None = target
    while current is not None:
        if current in seen or not 0 <= current < len(nodes):
            raise ValueError("child-action trace has invalid lineage")
        seen.add(current)
        result.append(current)
        parent = nodes[current].get("parent")
        if parent is not None and not isinstance(parent, int):
            raise ValueError("child-action trace has a non-integer parent")
        current = parent
    return list(reversed(result))


def _root(nodes: list[dict[str, Any]], index: int) -> int:
    path = _path(nodes, index)
    return path[0]


def _has_sampled_child_branch(trace: dict[str, Any]) -> bool:
    nodes = trace.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return False
    roots = [index for index, node in enumerate(nodes) if node.get("parent") is None]
    if len(roots) < 2:
        return False
    primary_root = roots[0]
    return any(
        node.get("sampled") is True
        and node.get("message", {}).get("role") == "assistant"
        and _root(nodes, index) != primary_root
        for index, node in enumerate(nodes)
    )


def _canonical_row(
    trace: dict[str, Any], *, source: Path, contract_recovery: bool = False
) -> dict[str, Any]:
    if trace.get("ok") is not True or trace.get("errors"):
        raise ValueError("child-action booster accepts structurally valid traces only")
    if not contract_recovery and (
        trace.get("stop_condition") in TRUNCATED_STOPS
        or _score((trace.get("rewards") or {}).get("harness_score")) != 1.0
    ):
        raise ValueError("child-action booster accepts hard-success traces only")
    task = trace.get("task", {}).get("data", {})
    children = task.get("oracle", {}).get("children")
    if not isinstance(children, list) or len(children) != 1:
        raise ValueError("child-action booster requires exactly one oracle child")
    child = children[0]
    expected = child.get("expected_result")
    if not isinstance(expected, int):
        raise ValueError("child-action booster requires an integer oracle result")
    nodes = trace.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("child-action trace has no message graph")
    roots = [index for index, node in enumerate(nodes) if node.get("parent") is None]
    if len(roots) < 2:
        raise ValueError("child-action trace has no non-root branch")
    primary_root = roots[0]
    child_sampled = [
        index
        for index, node in enumerate(nodes)
        if node.get("sampled") is True
        and _root(nodes, index) != primary_root
        and node.get("message", {}).get("role") == "assistant"
    ]
    if not child_sampled:
        raise ValueError("child-action trace has no sampled child response")
    prefix = _path(nodes, child_sampled[0])[:-1]
    if not prefix or _root(nodes, prefix[-1]) == primary_root:
        raise ValueError("child-action prefix crossed into the coordinator branch")
    tools = trace.get("tools") or []
    if not any(tool.get("name") == "ipython" for tool in tools if isinstance(tool, dict)):
        raise ValueError("child-action trace does not expose the IPython tool")
    code = f"await agent_message.send({str(expected)!r}, receiver_role='parent')"
    digest = hashlib.sha256(f"{trace['id']}:{code}".encode()).hexdigest()[:16]
    tool_call_id = f"child-action-{digest}"
    reasoning = (
        f"The private evidence gives {expected}. I will send that integer once through "
        "the required parent channel."
        if contract_recovery
        else (
            f"The private evidence gives {expected}. I must send that integer once to my "
            "parent and stop."
        )
    )
    target = {
        "role": "assistant",
        "content": "",
        "reasoning_content": reasoning,
        "tool_calls": [
            {
                "id": tool_call_id,
                "name": "ipython",
                "arguments": json.dumps({"code": code}, separators=(",", ":")),
            }
        ],
    }
    messages = [*[_wire_message(nodes[index]["message"]) for index in prefix], target]
    objective = "canonical_exact_parent_send_then_stop"
    if contract_recovery:
        messages.extend(
            [
                {
                    "role": "tool",
                    "content": "message queued",
                    "tool_call_id": tool_call_id,
                },
                {
                    "role": "assistant",
                    "content": "Done.",
                    "reasoning_content": "The result was delivered successfully, so I must stop.",
                    "tool_calls": [],
                },
            ]
        )
        objective = "canonical_exact_parent_send_ack_then_stop"
    task_key = task.get("episode_id") or task.get("name")
    if not isinstance(task_key, str):
        raise ValueError("child-action trace lacks a stable task key")
    return {
        "messages": messages,
        "tools": json.dumps(tools, sort_keys=True, separators=(",", ":")),
        "axis": "natural_n1a",
        "phase": "e0c29_evidence_available",
        "task_key": task_key,
        "trace_id": f"child-action-booster:{trace['id']}",
        "role": "child",
        "objective": objective,
        "expected_result": expected,
        "source_trace": str(source),
    }


def export(
    *,
    traces: list[Path],
    output_dir: Path,
    max_rows: int = 16,
    contract_recovery: bool = False,
) -> dict[str, Any]:
    row_cap = 32 if contract_recovery else 16
    if not 1 <= max_rows <= row_cap:
        raise ValueError(f"child-action booster row cap must be between one and {row_cap}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite child-action booster: {output_dir}")
    candidates: dict[str, dict[str, Any]] = {}
    source_records = []
    for path in traces:
        if not path.is_file():
            raise FileNotFoundError(path)
        source_records.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                envelope = json.loads(line)
                for trace in envelope.get("traces") or []:
                    eligible = trace.get("ok") is True and not trace.get("errors")
                    hard_success = (
                        trace.get("stop_condition") not in TRUNCATED_STOPS
                        and _score((trace.get("rewards") or {}).get("harness_score")) == 1.0
                    )
                    if (
                        eligible
                        and (contract_recovery or hard_success)
                        and _has_sampled_child_branch(trace)
                    ):
                        row = _canonical_row(
                            trace,
                            source=path,
                            contract_recovery=contract_recovery,
                        )
                        candidates[row["task_key"]] = row
    rows = list(candidates.values())[-max_rows:]
    if len(rows) < 2:
        raise ValueError("child-action booster needs at least two distinct hard successes")
    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": (
            CONTRACT_RECOVERY_SCHEMA_VERSION if contract_recovery else SCHEMA_VERSION
        ),
        "status": "complete",
        "role": "child",
        "objective": (
            "canonical_exact_parent_send_ack_then_stop"
            if contract_recovery
            else "canonical_exact_parent_send_then_stop"
        ),
        "rows": len(rows),
        "task_keys": [row["task_key"] for row in rows],
        "source_traces": source_records,
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=16)
    parser.add_argument(
        "--contract-recovery",
        action="store_true",
        help="include valid near-miss traces and train the post-send stop response",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                traces=[path.resolve() for path in args.traces],
                output_dir=args.output_dir.resolve(),
                max_rows=args.max_rows,
                contract_recovery=args.contract_recovery,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
