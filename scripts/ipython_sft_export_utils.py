"""Shared helpers for exporting persistent-IPython supervised traces."""

from __future__ import annotations

import json
from pathlib import Path

IPYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "ipython",
        "description": (
            "Execute Python scratchpad code in a persistent IPython kernel. "
            "Variables, imports, and loaded data persist across calls."
        ),
        "parameters": {
            "type": "object",
            "required": ["code"],
            "properties": {"code": {"type": "string"}},
        },
    },
}


def tool_call(call_id: str, code: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "ipython", "arguments": json.dumps({"code": code})},
    }


def system_prompt(trace_path: Path | None, default: str) -> str:
    if trace_path is None:
        return default
    for line in trace_path.read_text().splitlines():
        for trace in json.loads(line)["traces"]:
            for node in trace["nodes"]:
                message = node["message"]
                if message["role"] == "system" and isinstance(message.get("content"), str):
                    return message["content"]
    raise ValueError(f"no system message found in {trace_path}")


def replay_examples(path: Path | None, allowed_families: set[str]) -> list[dict]:
    if path is None:
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    unexpected = {row.get("metadata", {}).get("family") for row in rows} - allowed_families
    if unexpected:
        raise ValueError(f"unsupported replay families in {path}: {sorted(unexpected)}")
    return rows


__all__ = ["IPYTHON_TOOL", "replay_examples", "system_prompt", "tool_call"]
