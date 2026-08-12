"""Audit an OPSD token filter against serialized Verifiers traces."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.types import content_text

from prime_rl.orchestrator.trajectories import iter_trainable_branches
from prime_rl.utils.utils import import_object


def _trace_payloads(line: str) -> list[dict[str, object]]:
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("trace JSONL rows must be objects")
    if "task" in payload and "agent" in payload:
        return [payload]
    traces = payload.get("traces")
    if not isinstance(traces, list) or not all(isinstance(trace, dict) for trace in traces):
        raise ValueError("expected a Verifiers trace or evaluation row containing a traces list")
    return traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path)
    parser.add_argument("--filter", required=True, dest="filter_path")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--demo-key", default=None)
    args = parser.parse_args()

    filter_fn = import_object(args.filter_path)
    reports: list[dict[str, object]] = []
    totals = {
        "selected_tokens": 0,
        "retained_tokens": 0,
        "teacher_selected_tokens": 0,
    }
    trace_index = 0
    for line in args.traces.read_text().splitlines():
        for trace_payload in _trace_payloads(line):
            trace = vf.Trace.model_validate(trace_payload)
            demonstrations = None
            if args.demo_key:
                task = trace_payload.get("task")
                if isinstance(task, Mapping):
                    task_data = task.get("data")
                    if isinstance(task_data, Mapping):
                        demonstrations = task_data.get(args.demo_key)
            trace_index += 1
            reports.append(
                _audit_trace(
                    trace_index - 1,
                    trace,
                    filter_fn,
                    args,
                    totals,
                    serialized_demonstrations=demonstrations,
                )
            )

    print(
        json.dumps(
            {"traces": len(reports), "seq_len": args.seq_len, **totals, "reports": reports},
            indent=2,
        )
    )


def _audit_trace(
    trace_index: int,
    trace: vf.Trace,
    filter_fn: Callable[[vf.Trace], list[list[bool]]],
    args: argparse.Namespace,
    totals: dict[str, int],
    serialized_demonstrations: object = None,
) -> dict[str, object]:
    branches = list(iter_trainable_branches(trace))
    masks = filter_fn(trace)
    if len(masks) != len(branches):
        raise ValueError(
            f"trace {trace_index}: filter returned {len(masks)} masks for {len(branches)} branches"
        )

    branch_reports: list[dict[str, object]] = []
    for branch_index, ((branch, trainable_mask), keep_mask) in enumerate(
        zip(branches, masks, strict=True)
    ):
        if len(keep_mask) != len(branch.token_ids):
            raise ValueError(
                f"trace {trace_index} branch {branch_index}: mask length {len(keep_mask)} "
                f"does not match {len(branch.token_ids)} tokens"
            )
        selected = [
            index
            for index, (trainable, keep) in enumerate(
                zip(trainable_mask, keep_mask, strict=True)
            )
            if trainable and keep
        ]
        retained = selected if args.seq_len is None else [index for index in selected if index < args.seq_len]
        teacher_selected = 0
        demonstrations = serialized_demonstrations
        if demonstrations is None and args.demo_key:
            demonstrations = trace.info.get(args.demo_key)
        if demonstrations is None and args.demo_key:
            demonstrations = getattr(trace.task.data, args.demo_key, None)
        branch_demo: str | Sequence[str | None] | None = None
        if isinstance(demonstrations, str):
            branch_demo = demonstrations
        elif isinstance(demonstrations, Mapping):
            question = next(
                (
                    content_text(node.message.content).strip()
                    for node in branch.nodes
                    if node.message.role == "user" and content_text(node.message.content).strip()
                ),
                None,
            )
            value = demonstrations.get(question, demonstrations.get("*"))
            if isinstance(value, str) or (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes))
                and all(isinstance(item, str) or item is None for item in value)
            ):
                branch_demo = value

        demo_index = 0
        node_start = 0
        for node in branch.nodes:
            node_end = node_start + len(node.token_ids)
            node_selected = [index for index in selected if node_start <= index < node_end]
            if node_selected:
                node_demo = (
                    branch_demo
                    if isinstance(branch_demo, str)
                    else branch_demo[demo_index]
                    if branch_demo is not None and demo_index < len(branch_demo)
                    else None
                )
                demo_index += 1
                if isinstance(node_demo, str):
                    teacher_selected += len(node_selected)
            node_start = node_end

        totals["selected_tokens"] += len(selected)
        totals["retained_tokens"] += len(retained)
        totals["teacher_selected_tokens"] += teacher_selected
        branch_reports.append(
            {
                "branch": branch_index,
                "tokens": len(branch.token_ids),
                "selected": len(selected),
                "retained": len(retained),
                "teacher_selected": teacher_selected,
                "first_selected": selected[0] if selected else None,
                "last_selected": selected[-1] if selected else None,
            }
        )
    return {"trace": trace_index, "branches": branch_reports}


if __name__ == "__main__":
    main()
