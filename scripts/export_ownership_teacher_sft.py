#!/usr/bin/env python3
"""Export strictly admitted native ownership decisions as first-response SFT data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from subagent_communication_v1.taskset import (
    OWNERSHIP_GUIDANCE,
    _ownership_transition_behavior,
)
from verifiers.v1.dialects.chat import message_to_wire


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _normalize_text_content(content: Any) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError(f"unsupported message content type: {type(content).__name__}")
    parts = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
            raise ValueError("ownership SFT export only supports text content parts")
        parts.append(part["text"])
    return "".join(parts)


def _strip_guidance(message: dict[str, Any]) -> bool:
    if message.get("role") != "system":
        return False
    content = message.get("content")
    if not isinstance(content, str) or OWNERSHIP_GUIDANCE not in content:
        return False
    message["content"] = content.replace(f"\n\n{OWNERSHIP_GUIDANCE}", "").replace(
        OWNERSHIP_GUIDANCE, ""
    )
    return True


def _tool_call_name(call: dict[str, Any]) -> str | None:
    function = call.get("function")
    if isinstance(function, dict):
        return function.get("name")
    return call.get("name")


def _ownership_behavior(raw: dict[str, Any], trace: vf.Trace) -> dict[str, float]:
    data = (raw.get("task") or {}).get("data") or {}
    return _ownership_transition_behavior(
        trace,
        data.get("family", ""),
        tuple(data.get("expected_children") or ()),
        dict(data.get("child_paths") or {}),
        data.get("followup_secret"),
    )


def _coordinator_first_response(trace: vf.Trace) -> list[dict[str, Any]] | None:
    for branch in trace.branches:
        messages = [message_to_wire(node.message) for node in branch.nodes]
        first_user = next(
            (_content_text(message.get("content")) for message in messages if message.get("role") == "user"),
            "",
        )
        if first_user.lstrip().startswith("[task from parent]"):
            continue

        retained: list[dict[str, Any]] = []
        sampled_assistant: dict[str, Any] | None = None
        for node, message in zip(branch.nodes, messages, strict=True):
            if "content" in message:
                message["content"] = _normalize_text_content(message["content"])
            retained.append(message)
            if node.sampled and message.get("role") == "assistant":
                sampled_assistant = message
                break
        if sampled_assistant is None:
            return None
        calls = sampled_assistant.get("tool_calls") or []
        if len(calls) != 1 or _tool_call_name(calls[0]) != "ipython":
            return None
        if not _content_text(sampled_assistant.get("reasoning_content")).strip():
            return None
        return retained
    return None


def _candidate(raw: dict[str, Any], source: Path, line_number: int) -> dict[str, Any] | None:
    if not raw.get("ok"):
        return None
    trace = vf.Trace.model_validate(raw)
    behavior = _ownership_behavior(raw, trace)
    if behavior["ownership_transition"] != 1.0:
        return None

    messages = _coordinator_first_response(trace)
    if messages is None:
        return None
    stripped = sum(_strip_guidance(message) for message in messages)
    if stripped != 1 or any(OWNERSHIP_GUIDANCE in _content_text(message.get("content")) for message in messages):
        raise ValueError(f"{source}:{line_number}: could not remove exactly one ownership guidance block")

    task = (raw.get("task") or {}).get("data") or {}
    tools = [tool.model_dump(mode="json", exclude_none=True) for tool in trace.tools]
    if not tools:
        raise ValueError(f"{source}:{line_number}: admitted trace {trace.id} has no tool definitions")
    return {
        "messages": messages,
        "tool_defs": tools,
        "metadata": {
            "source": str(source),
            "source_line": line_number,
            "trace_id": trace.id,
            "task": task.get("name"),
            "family": task.get("family"),
            "selection": "native_first_response_ownership_transition_v2",
            "guidance_removed": True,
            "thinking_required": True,
            "ownership_components": behavior,
        },
    }


def export(
    inputs: list[Path],
    output_dir: Path,
    min_traces: int,
    *,
    max_per_task: int = 2,
) -> dict[str, Any]:
    if max_per_task < 1:
        raise ValueError("max_per_task must be positive")

    candidates: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    seen_ids: set[str] = set()
    for path in inputs:
        source_hashes[str(path)] = _sha256(path)
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            episode = json.loads(line)
            for raw in episode.get("traces", []):
                trace_id = raw.get("id")
                if not isinstance(trace_id, str) or not trace_id or trace_id in seen_ids:
                    continue
                seen_ids.add(trace_id)
                candidate = _candidate(raw, path, line_number)
                if candidate is not None:
                    candidates.append(candidate)

    selected: list[dict[str, Any]] = []
    task_counts: defaultdict[str, int] = defaultdict(int)
    for candidate in candidates:
        task = candidate["metadata"]["task"]
        if not isinstance(task, str) or task_counts[task] >= max_per_task:
            continue
        selected.append(candidate)
        task_counts[task] += 1

    if len(selected) < min_traces:
        raise ValueError(f"found {len(selected)} admitted ownership traces, need at least {min_traces}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "train.json"
    dataset_path.write_text("".join(json.dumps(example) + "\n" for example in selected))
    manifest = {
        "schema_version": 1,
        "selection": {
            "mode": "native_first_response_ownership_transition_v2",
            "ownership_transition": 1.0,
            "first_sampled_response": True,
            "single_ipython_call": True,
            "thinking_required": True,
            "temporary_guidance_removed": True,
            "max_per_task": max_per_task,
        },
        "source_sha256": source_hashes,
        "accepted_before_task_cap": len(candidates),
        "accepted_trace_ids": [example["metadata"]["trace_id"] for example in selected],
        "tasks": dict(sorted(task_counts.items())),
        "num_traces": len(selected),
        "dataset_sha256": _sha256(dataset_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-traces", type=int, default=1)
    parser.add_argument("--max-per-task", type=int, default=2)
    args = parser.parse_args()
    manifest = export(
        args.inputs,
        args.output_dir,
        args.min_traces,
        max_per_task=args.max_per_task,
    )
    print(f"wrote {manifest['num_traces']} admitted first responses to {args.output_dir}")


if __name__ == "__main__":
    main()
