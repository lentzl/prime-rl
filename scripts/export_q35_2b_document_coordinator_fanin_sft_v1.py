#!/usr/bin/env python3
"""Build a grounded coordinator spawn-and-fan-in SFT bootstrap."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import (
    STEMS,
    _wire_message,
    _worker_instruction,
    sha256_file,
)

SCHEMA_VERSION = "qwen35-2b-document-coordinator-fanin-sft/v1"
OBJECTIVE = "grounded_document_coordinator_spawn_partial_yield_fanin"
PHASES = ("spawn", "partial_fanin", "complete_fanin")


def canonical_spawn_action(root: str) -> tuple[str, str]:
    """Return a quoting-safe, answer-free three-child admission action."""
    assignments = {
        stem: _worker_instruction(f"{root}/{stem}.md") for stem in STEMS
    }
    declarations = "\n".join(
        f'{stem}_task = """{assignments[stem]}"""' for stem in STEMS
    )
    calls = "\n".join(
        f'{stem}_worker = await rlm({stem}_task, name="{stem}-document-worker")'
        for stem in STEMS
    )
    reasoning = (
        "I must admit all three named terminal workers without embedding quoted Python inside "
        "another quoted literal. I will store each contract in a triple-quoted variable, retain "
        "every returned handle, and then yield for explicit reports."
    )
    return reasoning, f"{declarations}\n{calls}"


def _assistant_tool_call(*, root: str, trace_id: str) -> tuple[dict[str, Any], str]:
    reasoning, code = canonical_spawn_action(root)
    digest = hashlib.sha256(f"{trace_id}:{code}".encode()).hexdigest()[:16]
    call_id = f"document-fanin-{digest}"
    return (
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": reasoning,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "ipython",
                        "arguments": json.dumps({"code": code}, separators=(",", ":")),
                    },
                }
            ],
        },
        call_id,
    )


def _stats(files: dict[str, str], root: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for stem in STEMS:
        text = files[f"{root}/{stem}.md"]
        result[stem] = {
            "words": len(text.split()),
            "h2": sum(line.startswith("## ") for line in text.splitlines()),
        }
    return result


def _child_message(stem: str, values: dict[str, int]) -> dict[str, Any]:
    payload = json.dumps(values, separators=(",", ":"))
    return {
        "role": "user",
        "content": (
            f"[from child:{stem}-document-worker]\n"
            "Agent-to-agent message received.\n"
            "Source: agent_message\n"
            f"From: {stem}-document-worker\n\n"
            f"{payload}"
        ),
    }


def _ack(stem: str, remaining: tuple[str, ...]) -> dict[str, Any]:
    waiting = ", ".join(f"{item}-document-worker" for item in remaining)
    return {
        "role": "assistant",
        "content": f"Stored {stem}-document-worker's report; waiting for {waiting}.",
        "reasoning_content": (
            "This is only a partial fan-in. I must preserve the delivered evidence, avoid "
            "polling or respawning, and yield until every required report arrives."
        ),
        "tool_calls": [],
    }


def _final_message(stats: dict[str, dict[str, int]]) -> dict[str, Any]:
    result: dict[str, int] = {}
    for stem in STEMS:
        result[f"{stem}_words"] = stats[stem]["words"]
        result[f"{stem}_h2"] = stats[stem]["h2"]
    result["total_words"] = sum(stats[stem]["words"] for stem in STEMS)
    result["total_h2"] = sum(stats[stem]["h2"] for stem in STEMS)
    return {
        "role": "assistant",
        "content": json.dumps(result, separators=(",", ":")),
        "reasoning_content": (
            "All three explicit reports are now present. I will aggregate only those reported "
            "values and return the exact requested JSON object without another tool call."
        ),
        "tool_calls": [],
    }


def _rows(trace: dict[str, Any], *, source: Path) -> list[dict[str, Any]]:
    task = trace.get("task", {}).get("data", {})
    if task.get("family") != "document_flat":
        raise ValueError("coordinator fan-in source must be a flat document trace")
    name = task.get("name")
    files = task.get("files")
    if not isinstance(name, str) or not isinstance(files, dict) or len(files) != 3:
        raise ValueError("coordinator fan-in trace lacks a stable task or three fixtures")
    roots = {str(Path(path).parent) for path in files}
    if len(roots) != 1:
        raise ValueError("coordinator fan-in fixtures do not share one directory")
    root = roots.pop()
    if set(files) != {f"{root}/{stem}.md" for stem in STEMS}:
        raise ValueError("coordinator fan-in fixtures do not match the canonical shards")
    nodes = trace.get("nodes")
    if not isinstance(nodes, list) or len(nodes) < 3:
        raise ValueError("coordinator fan-in trace lacks its root prompt prefix")
    if nodes[0].get("sampled") is not False or nodes[1].get("sampled") is not False:
        raise ValueError("coordinator fan-in trace prefix is not the canonical two messages")
    tools = trace.get("tools") or []
    if [tool.get("name") for tool in tools if isinstance(tool, dict)] != ["ipython"]:
        raise ValueError("coordinator fan-in trace must expose exactly the IPython tool")

    spawn, call_id = _assistant_tool_call(root=root, trace_id=str(trace.get("id")))
    receipt = {
        "role": "tool",
        "content": (
            "alpha_worker=RLMSpawnHandle(name='alpha-document-worker')\n"
            "beta_worker=RLMSpawnHandle(name='beta-document-worker')\n"
            "gamma_worker=RLMSpawnHandle(name='gamma-document-worker')"
        ),
        "tool_call_id": call_id,
    }
    stats = _stats(files, root)
    alpha = _child_message("alpha", stats["alpha"])
    beta = _child_message("beta", stats["beta"])
    gamma = _child_message("gamma", stats["gamma"])
    alpha_ack = _ack("alpha", ("beta", "gamma"))
    beta_ack = _ack("beta", ("gamma",))
    prefix = [_wire_message(nodes[0]["message"]), _wire_message(nodes[1]["message"])]
    phase_messages = {
        "spawn": [*prefix, copy.deepcopy(spawn)],
        "partial_fanin": [
            *prefix,
            copy.deepcopy(spawn),
            receipt,
            alpha,
            alpha_ack,
        ],
        "complete_fanin": [
            *prefix,
            copy.deepcopy(spawn),
            receipt,
            alpha,
            alpha_ack,
            beta,
            beta_ack,
            gamma,
            _final_message(stats),
        ],
    }
    return [
        {
            "messages": phase_messages[phase],
            "tools": json.dumps(tools, sort_keys=True, separators=(",", ":")),
            "task_key": f"{name}:{phase}",
            "trace_id": f"document-fanin:{trace.get('id')}:{phase}",
            "family": f"document_coordinator_{phase}",
            "role": "coordinator",
            "objective": OBJECTIVE,
            "source_trace": str(source),
        }
        for phase in PHASES
    ]


def export(*, traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite coordinator fan-in bootstrap: {output_dir}")
    rows: list[dict[str, Any]] = []
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
                    rows.extend(_rows(trace, source=path.resolve()))
    rows.sort(key=lambda row: row["task_key"])
    if len(rows) != 12 or len({row["task_key"] for row in rows}) != 12:
        raise ValueError("coordinator fan-in bootstrap must contain twelve unique rows")
    family_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in sorted({row["family"] for row in rows})
    }
    if set(family_counts.values()) != {4}:
        raise ValueError(f"coordinator fan-in bootstrap is not phase-balanced: {family_counts}")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "coordinator",
        "objective": OBJECTIVE,
        "rows": len(rows),
        "family_counts": family_counts,
        "task_keys": [row["task_key"] for row in rows],
        "source_traces": source_records,
        "answer_free": False,
        "grounded_on_fixture_contents": True,
        "leakage_mode": "environment_designer_grounded_v1",
        "tool_call_format": "openai_function_v1",
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
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                traces=[path.resolve() for path in args.traces],
                output_dir=args.output_dir.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
