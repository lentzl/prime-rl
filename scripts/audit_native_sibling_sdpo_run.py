"""Audit native-sibling SDPO provenance and token locality from run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from verifiers.v1.types import content_text

from prime_rl.orchestrator.trajectories import iter_trainable_branches
from prime_rl.utils.utils import import_object


def _digest_ids(token_ids: list[int]) -> str:
    payload = ",".join(str(token_id) for token_id in token_ids).encode()
    return hashlib.sha256(payload).hexdigest()


def _demonstration(nodes: list[Any], *, remove_thinking: bool) -> tuple[str, bool]:
    turns: list[str] = []
    has_reasoning = False
    for node in nodes:
        message = node.message
        if message.role != "assistant":
            continue
        reasoning = getattr(message, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning.strip():
            has_reasoning = True
            turns.append(reasoning)
        for tool_call in getattr(message, "tool_calls", None) or []:
            turns.append(f"{tool_call.name}({tool_call.arguments})")
        text = content_text(message.content)
        if text.strip():
            turns.append(text)
    demonstration = "\n".join(turns)
    if remove_thinking:
        demonstration = re.sub(r"<think>.*?</think>\s*", "", demonstration, flags=re.DOTALL)
    return demonstration, has_reasoning


def _question(nodes: list[Any]) -> str:
    question = next(
        (
            content_text(node.message.content).strip()
            for node in nodes
            if node.message.role == "user" and content_text(node.message.content).strip()
        ),
        None,
    )
    if question is None:
        raise ValueError("trainable branch has no initial user question")
    return question


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        config = tomllib.load(file)
    algo = config["orchestrator"]["algo"]
    optim = config["trainer"]["optim"]
    expected = {
        "type": "sdpo",
        "dont_reprompt_on_self_success": True,
        "include_environment_feedback": False,
        "multi_turn_replay": False,
    }
    for key, value in expected.items():
        if algo.get(key) != value:
            raise ValueError(f"audit requires orchestrator.algo.{key}={value!r}")
    if float(optim["lr"]) != 0.0:
        raise ValueError("native-sibling replay audit requires trainer.optim.lr=0")
    return config


def _load_traces(path: Path) -> list[vf.Trace]:
    traces: list[vf.Trace] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"trace line {line_number} is not an object")
        traces.append(vf.Trace.model_validate(payload))
    return traces


def _load_exports(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for export_file in sorted(path.glob("rank_*.jsonl")):
        for line in export_file.read_text().splitlines():
            record = json.loads(line)
            record["export_file"] = export_file.name
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--token-exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    algo_config = config["orchestrator"]["algo"]
    threshold = float(algo_config["success_reward_threshold"])
    remove_thinking = bool(algo_config["remove_thinking_from_demonstration"])
    filter_fn = import_object(algo_config["filter"]["import_path"])
    groups: dict[str, list[vf.Trace]] = defaultdict(list)
    for trace in _load_traces(args.traces):
        group_id = trace.info.get("group_id")
        if not isinstance(group_id, str):
            raise ValueError(f"trace {trace.id} has no serialized group_id")
        groups[group_id].append(trace)

    branch_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_reports: list[dict[str, Any]] = []
    for group_id, traces in groups.items():
        successes = [trace for trace in traces if trace.reward >= threshold]
        success_demos: dict[str, dict[str, tuple[str, bool]]] = {}
        for success in successes:
            demonstrations: dict[str, tuple[str, bool]] = {}
            for branch, _ in iter_trainable_branches(success):
                question = _question(branch.nodes)
                if question in demonstrations:
                    raise ValueError(f"duplicate branch question in successful trace {success.id}")
                demonstrations[question] = _demonstration(
                    branch.nodes,
                    remove_thinking=remove_thinking,
                )
            success_demos[str(success.id)] = demonstrations

        students: list[dict[str, Any]] = []
        for trace in traces:
            teacher = next((candidate for candidate in successes if candidate is not trace), None)
            masks = filter_fn(trace)
            branches = list(iter_trainable_branches(trace))
            if len(masks) != len(branches):
                raise ValueError(f"filter/branch count mismatch in trace {trace.id}")
            branch_reports: list[dict[str, Any]] = []
            for branch_number, ((branch, trainable_mask), keep_mask) in enumerate(zip(branches, masks, strict=True)):
                selected = [
                    position
                    for position, (trainable, keep) in enumerate(zip(trainable_mask, keep_mask, strict=True))
                    if trainable and keep
                ]
                question = _question(branch.nodes)
                teacher_solution: str | None = None
                teacher_has_reasoning = False
                if teacher is not None and selected:
                    teacher_solution, teacher_has_reasoning = success_demos[str(teacher.id)][question]
                supervised = bool(selected and teacher_solution is not None)
                token_ids = list(branch.token_ids)
                token_hash = _digest_ids(token_ids)
                record = {
                    "trace_id": str(trace.id),
                    "group_id": group_id,
                    "task_idx": trace.task.data.idx,
                    "task_name": getattr(trace.task.data, "name", None),
                    "resource_family": getattr(trace.task.data, "resource_family", None),
                    "reward": trace.reward,
                    "branch": branch_number,
                    "token_hash": token_hash,
                    "selected_positions": selected if supervised else [],
                    "teacher_trace_id": str(teacher.id) if supervised else None,
                    "teacher_reward": teacher.reward if supervised else None,
                    "teacher_solution": teacher_solution,
                    "teacher_solution_sha256": (
                        hashlib.sha256(teacher_solution.encode()).hexdigest() if teacher_solution is not None else None
                    ),
                    "teacher_has_reasoning": teacher_has_reasoning if supervised else False,
                }
                branch_index[token_hash].append(record)
                branch_reports.append(record)
            students.append({"trace_id": str(trace.id), "branches": branch_reports})
        group_reports.append(
            {
                "group_id": group_id,
                "task_idx": traces[0].task.data.idx,
                "resource_family": getattr(traces[0].task.data, "resource_family", None),
                "rollouts": len(traces),
                "strict_successes": len(successes),
                "native_success_trace_ids": [str(trace.id) for trace in successes],
                "students": students,
            }
        )

    export_reports: list[dict[str, Any]] = []
    matched_branches: set[tuple[str, int]] = set()
    for export in _load_exports(args.token_exports):
        token_hash = _digest_ids(export["token_ids"])
        matches = branch_index.get(token_hash, [])
        if len(matches) != 1:
            raise ValueError(f"token export matched {len(matches)} trace branches ({token_hash})")
        match = matches[0]
        actual_positions = [index for index, weight in enumerate(export["sdpo_weights"]) if weight != 0.0]
        if actual_positions != match["selected_positions"]:
            raise ValueError(f"SDPO token locality mismatch for trace {match['trace_id']} branch {match['branch']}")
        selected_kl = [export["mismatch_kl"][index] for index in actual_positions]
        if not all(math.isfinite(value) for value in selected_kl):
            raise ValueError("non-finite mismatch KL on selected SDPO token")
        matched_branches.add((match["trace_id"], match["branch"]))
        export_reports.append(
            {
                "export_file": export["export_file"],
                "export_sequence_idx": export["export_sequence_idx"],
                "trace_id": match["trace_id"],
                "group_id": match["group_id"],
                "resource_family": match["resource_family"],
                "reward": match["reward"],
                "branch": match["branch"],
                "selected_tokens": len(actual_positions),
                "teacher_trace_id": match["teacher_trace_id"],
                "teacher_solution_sha256": match["teacher_solution_sha256"],
                "teacher_has_reasoning": match["teacher_has_reasoning"],
                "mean_mismatch_kl": sum(selected_kl) / len(selected_kl),
                "max_mismatch_kl": max(selected_kl),
            }
        )

    expected_branches = {
        (record["trace_id"], record["branch"])
        for records in branch_index.values()
        for record in records
        if record["selected_positions"]
    }
    if matched_branches != expected_branches:
        missing = sorted(expected_branches - matched_branches)
        unexpected = sorted(matched_branches - expected_branches)
        raise ValueError(f"token export coverage mismatch: missing={missing}, unexpected={unexpected}")

    no_success_groups = [group for group in group_reports if group["strict_successes"] == 0]
    replay_groups = [group for group in group_reports if group["strict_successes"] > 0]
    selected_kls = [report["mean_mismatch_kl"] for report in export_reports]
    report = {
        "config": str(args.config),
        "traces": sum(len(group) for group in groups.values()),
        "groups": len(groups),
        "replay_groups": len(replay_groups),
        "no_success_groups": len(no_success_groups),
        "exported_branches": len(export_reports),
        "selected_tokens": sum(report["selected_tokens"] for report in export_reports),
        "mean_sequence_mismatch_kl": (sum(selected_kls) / len(selected_kls) if selected_kls else None),
        "phase_b_structural_pass": bool(replay_groups and no_success_groups and export_reports),
        "group_reports": group_reports,
        "export_reports": export_reports,
    }
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
