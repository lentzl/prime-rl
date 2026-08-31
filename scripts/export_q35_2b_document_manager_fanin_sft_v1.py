#!/usr/bin/env python3
"""Build exact-context depth-one manager passive fan-in and parent-report SFT."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_coordinator_fanin_sft_v1 import (
    _ack,
    _child_message,
    _stats,
    canonical_spawn_action,
)
from export_q35_2b_document_decision_sft_v1 import STEMS, sha256_file
from export_q35_2b_document_manager_admission_sft_v1 import _manager_prefix

SCHEMA_VERSION = "qwen35-2b-document-manager-fanin-sft/v1"
OBJECTIVE = "depth_two_manager_passive_fanin_parent_report"
PHASES = ("post_admission_yield", "partial_fanin_yield", "complete_fanin_report")


def _spawn_action(*, root: str, trace_id: str) -> tuple[dict[str, Any], str]:
    reasoning, code = canonical_spawn_action(root)
    digest = hashlib.sha256(f"{trace_id}:{code}".encode()).hexdigest()[:16]
    call_id = f"document-manager-fanin-{digest}"
    return (
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I am the non-root document coordinator. " + reasoning,
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


def _spawn_receipt(call_id: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": (
            "alpha_worker=RLMSpawnHandle(name='alpha-document-worker')\n"
            "beta_worker=RLMSpawnHandle(name='beta-document-worker')\n"
            "gamma_worker=RLMSpawnHandle(name='gamma-document-worker')"
        ),
        "tool_call_id": call_id,
    }


def _passive_yield() -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "All three terminal children are admitted; waiting for their explicit reports.",
        "reasoning_content": (
            "All three admissions succeeded and their handles are retained. Child reports resume "
            "this session automatically, so I must make no polling, observation, or sleep call."
        ),
        "tool_calls": [],
    }


def _parent_report(stats: dict[str, dict[str, int]], *, trace_id: str) -> dict[str, Any]:
    result: dict[str, int] = {}
    for stem in STEMS:
        result[f"{stem}_words"] = stats[stem]["words"]
        result[f"{stem}_h2"] = stats[stem]["h2"]
    result["total_words"] = sum(stats[stem]["words"] for stem in STEMS)
    result["total_h2"] = sum(stats[stem]["h2"] for stem in STEMS)
    code = (
        "import json\n"
        f"parent_report = {result!r}\n"
        "await agent_message.send(json.dumps(parent_report, separators=(',', ':')), "
        "receiver_role='parent')"
    )
    digest = hashlib.sha256(f"{trace_id}:parent-report:{code}".encode()).hexdigest()[:16]
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": (
            "All three explicit reports are present. I will aggregate only their values, preserve "
            "the required string keys, serialize one JSON object, and send it once to my parent."
        ),
        "tool_calls": [
            {
                "id": f"document-manager-report-{digest}",
                "type": "function",
                "function": {
                    "name": "ipython",
                    "arguments": json.dumps({"code": code}, separators=(",", ":")),
                },
            }
        ],
    }


def _rows(trace: dict[str, Any], *, source: Path) -> list[dict[str, Any]]:
    task = trace.get("task", {}).get("data", {})
    if task.get("family") != "document_hierarchical" or not isinstance(task.get("name"), str):
        raise ValueError("manager fan-in source is not a hierarchical document task")
    files = task.get("files")
    if not isinstance(files, dict) or len(files) != 3:
        raise ValueError("manager fan-in source lacks three document fixtures")
    roots = {str(Path(path).parent) for path in files}
    if len(roots) != 1:
        raise ValueError("manager fan-in fixtures do not share one directory")
    root = roots.pop()
    if set(files) != {f"{root}/{stem}.md" for stem in STEMS}:
        raise ValueError("manager fan-in fixtures do not match canonical shards")
    tools = trace.get("tools") or []
    if [tool.get("name") for tool in tools if isinstance(tool, dict)] != ["ipython"]:
        raise ValueError("manager fan-in source must expose exactly the IPython tool")

    trace_id = str(trace.get("id"))
    prefix = _manager_prefix(trace)
    spawn, call_id = _spawn_action(root=root, trace_id=trace_id)
    receipt = _spawn_receipt(call_id)
    stats = _stats(files, root)
    alpha = _child_message("alpha", stats["alpha"])
    beta = _child_message("beta", stats["beta"])
    gamma = _child_message("gamma", stats["gamma"])
    alpha_ack = _ack("alpha", ("beta", "gamma"))
    beta_ack = _ack("beta", ("gamma",))
    messages = {
        "post_admission_yield": [
            *copy.deepcopy(prefix),
            copy.deepcopy(spawn),
            copy.deepcopy(receipt),
            _passive_yield(),
        ],
        "partial_fanin_yield": [
            *copy.deepcopy(prefix),
            copy.deepcopy(spawn),
            copy.deepcopy(receipt),
            copy.deepcopy(alpha),
            copy.deepcopy(alpha_ack),
        ],
        "complete_fanin_report": [
            *copy.deepcopy(prefix),
            copy.deepcopy(spawn),
            copy.deepcopy(receipt),
            copy.deepcopy(alpha),
            copy.deepcopy(alpha_ack),
            copy.deepcopy(beta),
            copy.deepcopy(beta_ack),
            copy.deepcopy(gamma),
            _parent_report(stats, trace_id=trace_id),
        ],
    }
    return [
        {
            "messages": messages[phase],
            "tools": json.dumps(tools, sort_keys=True, separators=(",", ":")),
            "task_key": f"{task['name']}:{phase}",
            "trace_id": f"document-manager-fanin:{trace_id}:{phase}",
            "family": f"document_manager_{phase}",
            "phase": phase,
            "role": "coordinator",
            "objective": OBJECTIVE,
            "source_trace": str(source),
        }
        for phase in PHASES
    ]


def export(*, traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite manager fan-in bootstrap: {output_dir}")
    rows: list[dict[str, Any]] = []
    sources = []
    for path in traces:
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved = path.resolve()
        sources.append({"path": str(resolved), "sha256": sha256_file(path)})
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    for trace in json.loads(line).get("traces") or []:
                        if trace.get("task", {}).get("data", {}).get("family") == "document_hierarchical":
                            rows.extend(_rows(trace, source=resolved))
    rows.sort(key=lambda row: row["task_key"])
    if len(rows) != 12 or len({row["task_key"] for row in rows}) != 12:
        raise ValueError("manager fan-in bootstrap requires twelve unique rows")
    family_counts = {
        family: sum(row["family"] == family for row in rows) for family in sorted({row["family"] for row in rows})
    }
    if set(family_counts.values()) != {4}:
        raise ValueError(f"manager fan-in bootstrap is not phase-balanced: {family_counts}")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "coordinator",
        "objective": OBJECTIVE,
        "rows": 12,
        "family_counts": family_counts,
        "task_keys": [row["task_key"] for row in rows],
        "source_traces": sources,
        "answer_free": False,
        "exact_live_depth_two_context": True,
        "root_policy_targeted": False,
        "child_policy_targeted": False,
        "manager_admission_targeted": False,
        "tool_call_format": "openai_function_v1",
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
