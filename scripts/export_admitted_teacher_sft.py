#!/usr/bin/env python3
"""Export fully admitted multi-agent teacher traces as offline SFT data."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from verifiers.v1.dialects.chat import message_to_wire

QUESTION_PREFIX = "<Question>\n"
DEMONSTRATION_MARKER = "\nThis is an example for a response to the question:"
CONTRACT_MARKER = "\n\nDelegation contract:"
TOOL_FAILURE_MARKERS = ("Traceback (most recent call last)", " not found")
FORBIDDEN_COORDINATOR_CODE_MARKERS = (
    "agent_observe",
    "asyncio.sleep(",
    "conversation log",
    "list_agents(",
    "list_subagents(",
    "os.listdir(",
    "session_dir",
    "sessions/",
    "time.sleep(",
)
WAITING_CODE_MARKERS = (
    "checking for child",
    "still waiting",
    "wait for child",
    "waiting for child",
    "waiting for shard",
)


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tool_call_name(call: dict[str, Any]) -> str | None:
    function = call.get("function")
    if isinstance(function, dict):
        return function.get("name")
    return call.get("name")


def _tool_call_code(call: dict[str, Any]) -> str:
    function = call.get("function")
    arguments = function.get("arguments") if isinstance(function, dict) else call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return ""
    if not isinstance(arguments, dict):
        return ""
    code = arguments.get("code")
    return code if isinstance(code, str) else ""


def _declared_tool_names(raw: dict[str, Any]) -> set[str]:
    names = {_tool_call_name(tool) for tool in raw.get("tools") or []}
    names.discard(None)
    return names or {"ipython"}


def _forbidden_coordinator_code(code: str) -> bool:
    lowered = code.lower()
    return any(marker in lowered for marker in FORBIDDEN_COORDINATOR_CODE_MARKERS) or (
        "print(" in lowered and any(marker in lowered for marker in WAITING_CODE_MARKERS)
    )


def _effective_python_code(code: str) -> bool:
    """Return whether a cell performs more than a comment, pass, or string literal."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    return any(
        not isinstance(statement, ast.Pass)
        and not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant))
        for statement in tree.body
    )


def _is_child_message(message: dict[str, Any]) -> bool:
    return message.get("role") == "user" and _content_text(message.get("content")).lstrip().startswith("[from child:")


def _is_gate_retry(message: dict[str, Any]) -> bool:
    return message.get("role") == "user" and _content_text(message.get("content")).lstrip().startswith(
        "Autonomous quality gate failed"
    )


def _executable_branches(
    raw: dict[str, Any],
    *,
    allow_inert_coordinator_cells: bool = False,
) -> bool:
    trace = vf.Trace.model_validate(raw)
    declared_tools = _declared_tool_names(raw)
    for branch in trace.branches:
        messages = [message_to_wire(node.message) for node in branch.nodes]
        child_branch = _branch_role(messages) == "child"
        child_sent_message = False
        coordinator_spawned = False
        coordinator_waiting = False
        coordinator_received_child = False
        for node, message in zip(branch.nodes, messages, strict=True):
            content = _content_text(message.get("content"))
            if message.get("role") == "tool" and any(marker in content for marker in TOOL_FAILURE_MARKERS):
                return False
            if not child_branch and _is_child_message(message):
                coordinator_waiting = False
                coordinator_received_child = True
                continue
            if not child_branch and coordinator_received_child and _is_gate_retry(message):
                return False
            if not node.sampled or message.get("role") != "assistant":
                continue
            calls = message.get("tool_calls") or []
            if not child_branch and coordinator_waiting and calls:
                if not allow_inert_coordinator_cells or any(
                    _effective_python_code(_tool_call_code(call)) for call in calls
                ):
                    return False
            if child_sent_message and calls:
                return False
            for call in calls:
                if _tool_call_name(call) not in declared_tools:
                    return False
                code = _tool_call_code(call)
                if not child_branch:
                    if (
                        not _effective_python_code(code) and not allow_inert_coordinator_cells
                    ) or _forbidden_coordinator_code(code):
                        return False
                    if "rlm(" in code:
                        coordinator_spawned = True
                if child_branch and "agent_message.send" in code:
                    child_sent_message = True
            if not child_branch and coordinator_spawned and not calls and content.strip():
                coordinator_waiting = True
    return True


def _admitted(raw: dict[str, Any]) -> bool:
    metrics = raw.get("metrics") or {}
    return bool(
        raw.get("ok")
        and _executable_branches(raw)
        and metrics.get("causal_completed") == 1.0
        and metrics.get("causal_clean_completed") == 1.0
        and metrics.get("answer_accuracy") == 1.0
        and metrics.get("protocol_aligned") == 1.0
        and metrics.get("duplicate_cells") == 0.0
        and metrics.get("post_fan_in_failed_cells") == 0.0
        and metrics.get("post_fan_in_control_aligned") == 1.0
    )


def _inert_bootstrap_admitted(raw: dict[str, Any]) -> bool:
    """Admit strict traces whose only executable defect is an inert parent cell."""
    metrics = raw.get("metrics") or {}
    return bool(
        raw.get("ok")
        and _executable_branches(raw, allow_inert_coordinator_cells=True)
        and metrics.get("causal_completed") == 1.0
        and metrics.get("causal_clean_completed") == 1.0
        and metrics.get("answer_accuracy") == 1.0
        and metrics.get("protocol_aligned") == 1.0
        and metrics.get("duplicate_cells") == 0.0
        and metrics.get("post_fan_in_failed_cells") == 0.0
        and metrics.get("post_fan_in_control_aligned") == 1.0
    )


def _last_assistant_text(raw: dict[str, Any]) -> str:
    for node in reversed(raw.get("nodes") or []):
        message = node.get("message") or {}
        if message.get("role") == "assistant":
            return _content_text(message.get("content"))
    return ""


def _embedded_answer_matches(raw: dict[str, Any]) -> bool:
    answer = ((raw.get("task") or {}).get("data") or {}).get("answer")
    if not isinstance(answer, dict):
        return False

    text = _last_assistant_text(raw)
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if candidate == answer:
            return True
    return False


def _repairable_final(raw: dict[str, Any]) -> bool:
    metrics = raw.get("metrics") or {}
    return bool(
        raw.get("ok")
        and _executable_branches(raw)
        and metrics.get("causal_child_reply") == 1.0
        and metrics.get("causal_clean_child_reply") == 1.0
        and metrics.get("protocol_aligned") == 1.0
        and metrics.get("clean_protocol_aligned") == 1.0
        and metrics.get("duplicate_cells") == 0.0
        and metrics.get("failed_cells") == 0.0
        and metrics.get("post_fan_in_failed_cells") == 0.0
        and metrics.get("post_fan_in_control_aligned") == 1.0
        and _embedded_answer_matches(raw)
    )


def _strip_teacher_text(text: str) -> str:
    if not text.startswith(QUESTION_PREFIX):
        return text
    marker_index = text.find(DEMONSTRATION_MARKER)
    if marker_index < 0:
        return text
    question = text[len(QUESTION_PREFIX) : marker_index]
    contract_index = text.rfind(CONTRACT_MARKER)
    if contract_index > marker_index:
        question = f"{question}{text[contract_index:]}"
    return question


def _strip_teacher_demonstration(content: Any) -> Any:
    if isinstance(content, str):
        return _strip_teacher_text(content)
    if not isinstance(content, list):
        return content

    stripped = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
            part = {**part, "text": _strip_teacher_text(part["text"])}
        stripped.append(part)
    return stripped


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part["text"] for part in content if isinstance(part, dict) and isinstance(part.get("text"), str))
    return ""


def _normalize_text_content(content: Any) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError(f"unsupported message content type: {type(content).__name__}")

    text_parts = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
            raise ValueError("teacher SFT export only supports text content parts")
        text_parts.append(part["text"])
    return "".join(text_parts)


def _branch_role(messages: list[dict[str, Any]]) -> str:
    first_user = next(
        (message.get("content") for message in messages if message.get("role") == "user"),
        "",
    )
    return "child" if _content_text(first_user).lstrip().startswith("[task from parent]") else "coordinator"


def _drop_inert_coordinator_cells(messages: list[dict[str, Any]]) -> int:
    skipped_tool_call_ids: set[str] = set()
    retained = []
    removed = 0
    for message in messages:
        calls = message.get("tool_calls") or []
        if (
            message.get("role") == "assistant"
            and calls
            and all(not _effective_python_code(_tool_call_code(call)) for call in calls)
        ):
            skipped_tool_call_ids.update(call_id for call in calls if isinstance((call_id := call.get("id")), str))
            removed += 1
            continue
        if message.get("role") == "tool" and message.get("tool_call_id") in skipped_tool_call_ids:
            continue
        retained.append(message)
    messages[:] = retained
    return removed


def _trace_examples(
    raw: dict[str, Any],
    source: Path,
    *,
    canonicalize_final: bool = False,
    drop_inert_coordinator_cells: bool = False,
) -> list[dict[str, Any]]:
    trace = vf.Trace.model_validate(raw)
    tools = [tool.model_dump(mode="json", exclude_none=True) for tool in trace.tools]
    answer = ((raw.get("task") or {}).get("data") or {}).get("answer")
    examples = []
    for branch in trace.branches:
        if not any(node.sampled for node in branch.nodes):
            continue
        messages = [message_to_wire(node.message) for node in branch.nodes]
        for message in messages:
            if "content" in message:
                message["content"] = _normalize_text_content(message["content"])
        for message in messages:
            if message.get("role") == "user":
                message["content"] = _strip_teacher_demonstration(message.get("content"))
                break
        role = _branch_role(messages)
        inert_cells_removed = 0
        if drop_inert_coordinator_cells and role == "coordinator":
            inert_cells_removed = _drop_inert_coordinator_cells(messages)
        if canonicalize_final and role == "coordinator":
            final = next(
                (message for message in reversed(messages) if message.get("role") == "assistant"),
                None,
            )
            if final is None or final.get("tool_calls") or not isinstance(answer, dict):
                raise ValueError(f"cannot canonicalize coordinator final answer in trace {trace.id}")
            final["content"] = json.dumps(answer)
        examples.append(
            {
                "messages": messages,
                "tool_defs": tools,
                "metadata": {
                    "source": str(source),
                    "trace_id": trace.id,
                    "task": trace.task.data.name,
                    "branch_index": branch.index,
                    "role": role,
                    "final_answer_canonicalized": canonicalize_final and role == "coordinator",
                    "inert_coordinator_cells_removed": inert_cells_removed,
                },
            }
        )
    if not examples:
        raise ValueError(f"admitted trace {trace.id} has no sampled branches")
    return examples


def export(
    inputs: list[Path],
    output_dir: Path,
    min_traces: int,
    *,
    canonicalize_final_answer: bool = False,
    drop_inert_coordinator_cells: bool = False,
) -> dict[str, Any]:
    examples: list[dict[str, Any]] = []
    accepted_ids: list[str] = []
    source_hashes: dict[str, str] = {}
    seen_ids: set[str] = set()
    canonicalized_ids: list[str] = []
    inert_sanitized_ids: list[str] = []

    for path in inputs:
        source_hashes[str(path)] = _source_sha256(path)
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            episode = json.loads(line)
            for raw in episode.get("traces", []):
                admitted = _admitted(raw)
                repairable = canonicalize_final_answer and _repairable_final(raw)
                inert_bootstrap = drop_inert_coordinator_cells and not admitted and _inert_bootstrap_admitted(raw)
                if not admitted and not repairable and not inert_bootstrap:
                    continue
                trace_id = raw.get("id")
                if not isinstance(trace_id, str) or not trace_id:
                    raise ValueError(f"{path}:{line_number}: admitted trace has no id")
                if trace_id in seen_ids:
                    continue
                seen_ids.add(trace_id)
                accepted_ids.append(trace_id)
                if repairable and not admitted:
                    canonicalized_ids.append(trace_id)
                if inert_bootstrap:
                    inert_sanitized_ids.append(trace_id)
                examples.extend(
                    _trace_examples(
                        raw,
                        path,
                        canonicalize_final=repairable and not admitted,
                        drop_inert_coordinator_cells=inert_bootstrap,
                    )
                )

    if len(accepted_ids) < min_traces:
        raise ValueError(f"found {len(accepted_ids)} admitted teacher traces, need at least {min_traces}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "train.json"
    dataset_path.write_text("".join(json.dumps(example) + "\n" for example in examples))
    selection_modes = ["strict"]
    if canonicalize_final_answer:
        selection_modes.append("canonicalized_final")
    if drop_inert_coordinator_cells:
        selection_modes.append("inert_coordinator_bootstrap")
    manifest = {
        "schema_version": 1,
        "selection": {
            "mode": "+".join(selection_modes),
            "strict": {
                "sampled_tool_calls_declared": True,
                "tool_results_executable": True,
                "child_stops_tools_after_message_send": True,
                "coordinator_has_no_polling_cells": True,
                "causal_completed": 1.0,
                "causal_clean_completed": 1.0,
                "answer_accuracy": 1.0,
                "protocol_aligned": 1.0,
                "duplicate_cells": 0.0,
                "post_fan_in_failed_cells": 0.0,
                "post_fan_in_control_aligned": 1.0,
            },
            "canonicalized_final": {
                "sampled_tool_calls_declared": True,
                "tool_results_executable": True,
                "child_stops_tools_after_message_send": True,
                "coordinator_has_no_polling_cells": True,
                "causal_child_reply": 1.0,
                "causal_clean_child_reply": 1.0,
                "protocol_aligned": 1.0,
                "clean_protocol_aligned": 1.0,
                "duplicate_cells": 0.0,
                "failed_cells": 0.0,
                "post_fan_in_failed_cells": 0.0,
                "post_fan_in_control_aligned": 1.0,
                "embedded_answer_exact_match": 1.0,
                "edit": "replace only the final coordinator text with bare canonical JSON",
            }
            if canonicalize_final_answer
            else None,
            "inert_coordinator_bootstrap": {
                "scope": "temporary 9B teacher bootstrap only",
                "all_strict_metrics_required": True,
                "only_ast_inert_coordinator_cells_removed": True,
                "polling_and_post_child_gate_retries_rejected": True,
                "downstream_4b_requires_untouched_teacher_traces": True,
            }
            if drop_inert_coordinator_cells
            else None,
        },
        "source_sha256": source_hashes,
        "accepted_trace_ids": accepted_ids,
        "canonicalized_trace_ids": canonicalized_ids,
        "inert_sanitized_trace_ids": inert_sanitized_ids,
        "num_traces": len(accepted_ids),
        "num_branch_examples": len(examples),
        "dataset_sha256": _source_sha256(dataset_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-traces", type=int, default=1)
    parser.add_argument("--canonicalize-final-answer", action="store_true")
    parser.add_argument("--drop-inert-coordinator-cells", action="store_true")
    args = parser.parse_args()
    manifest = export(
        args.inputs,
        args.output_dir,
        args.min_traces,
        canonicalize_final_answer=args.canonicalize_final_answer,
        drop_inert_coordinator_cells=args.drop_inert_coordinator_cells,
    )
    print(
        f"wrote {manifest['num_branch_examples']} branches from "
        f"{manifest['num_traces']} admitted traces to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
