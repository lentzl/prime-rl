#!/usr/bin/env python3
"""Build answer-free, role-local SFT corpora for terminal specialists."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import _wire_message, sha256_file

SCHEMA_VERSION = "qwen35-2b-specialist-worker-sft/v1"
OBJECTIVE = "answer_free_terminal_specialist_file_compute_and_parent_report"
SPECIALISTS = ("table_analyst", "source_inspector")
FAMILIES = {
    "table_analyst": ("table_join", "table_reconcile"),
    "source_inspector": ("source_ast", "source_config"),
}


def _text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _load_traces(paths: list[Path]) -> list[dict[str, Any]]:
    traces = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                envelope = json.loads(line)
                values = envelope.get("traces") or []
                if not isinstance(values, list):
                    raise ValueError(f"{path}: traces must be a list")
                traces.extend(value for value in values if isinstance(value, dict))
    return traces


def _live_child_context(
    traces: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    tool_sets = []
    for trace in traces:
        nodes = trace.get("nodes") or []
        tools = trace.get("tools") or []
        for node in nodes:
            if node.get("parent") is not None or node.get("sampled") is not False:
                continue
            message = node.get("message") or {}
            text = _text(message)
            if "Recursive agent depth: 1" in text and "You are a child agent" in text:
                candidates.append(copy.deepcopy(message))
                tool_sets.append(copy.deepcopy(tools))
    if not candidates:
        raise ValueError("source traces contain no live terminal-child context")
    rendered_tools = {json.dumps(value, sort_keys=True, separators=(",", ":")) for value in tool_sets}
    if len(rendered_tools) != 1:
        raise ValueError("source traces expose inconsistent child tool contracts")
    tools = tool_sets[0]
    if [tool.get("name") for tool in tools if isinstance(tool, dict)] != ["ipython"]:
        raise ValueError("specialist corpus requires exactly the live IPython tool")
    return candidates[0], tools


def _assignment(expert_id: str, objective: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "[task from parent]\n\n"
                    "[selected terminal capability]\n"
                    f"expert_id={expert_id}\n"
                    "session_role=terminal_worker\n"
                    "is_root=false\n"
                    "has_parent=true\n"
                    "can_delegate=false\n"
                    "can_finalize_user=false\n"
                    "return_contract=exactly_one_parent_report\n"
                    f"{objective}"
                ),
            }
        ],
    }


def _family_target(family: str, root: str) -> tuple[str, str, str]:
    if family == "table_join":
        csv_path = f"{root}/transactions.csv"
        json_path = f"{root}/rates.json"
        objective = (
            f"Read {csv_path} and {json_path}. Join each transaction to its customer multiplier, "
            "keep only rows whose status is exactly `posted`, and sum units * unit_price * "
            "multiplier. Send exactly one compact JSON object with integer key `value` to "
            "receiver_role='parent', then stop."
        )
        code = f"""from pathlib import Path
import json
import pandas as pd
transactions = pd.read_csv(Path({csv_path!r}))
rates = json.loads(Path({json_path!r}).read_text())
transactions['multiplier'] = transactions['customer'].map(rates)
posted = transactions.loc[transactions['status'].eq('posted')]
value = int((posted['units'] * posted['unit_price'] * posted['multiplier']).sum())
await agent_message.send(json.dumps({{'value': value}}, separators=(',', ':')), receiver_role='parent')"""
        return objective, code, "Map multipliers by customer, filter posted rows, sum exactly, and report JSON."
    if family == "table_reconcile":
        csv_path = f"{root}/inventory.csv"
        json_path = f"{root}/corrections.json"
        objective = (
            f"Read {csv_path} and {json_path}. For every SKU compute opening + received - "
            "shipped + its JSON correction, then sum the reconciled quantities across all SKUs. "
            "Send exactly one compact JSON object with integer key `value` to "
            "receiver_role='parent', then stop."
        )
        code = f"""from pathlib import Path
import json
import pandas as pd
inventory = pd.read_csv(Path({csv_path!r}))
corrections = json.loads(Path({json_path!r}).read_text())
inventory['correction'] = inventory['sku'].map(corrections)
value = int((inventory['opening'] + inventory['received'] - inventory['shipped'] + inventory['correction']).sum())
await agent_message.send(json.dumps({{'value': value}}, separators=(',', ':')), receiver_role='parent')"""
        return objective, code, "Map corrections by SKU, reconcile every row, sum exactly, and report JSON."
    if family == "source_ast":
        alpha = f"{root}/alpha.py"
        beta = f"{root}/beta.py"
        objective = (
            f"Parse the complete Python files {alpha} and {beta} with ast. Across both files, "
            "count every FunctionDef (including helper functions), every AsyncFunctionDef, and "
            "every function node with at least one decorator. Compute 2 * FunctionDef + 3 * "
            "AsyncFunctionDef + decorated_function_nodes. Send exactly one compact JSON object "
            "with integer key `value` to receiver_role='parent', then stop."
        )
        code = f"""from pathlib import Path
import ast
import json
paths = [Path({alpha!r}), Path({beta!r})]
nodes = [node for path in paths for node in ast.walk(ast.parse(path.read_text()))]
sync_count = sum(isinstance(node, ast.FunctionDef) for node in nodes)
async_count = sum(isinstance(node, ast.AsyncFunctionDef) for node in nodes)
decorated_count = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and bool(node.decorator_list) for node in nodes)
value = int(2 * sync_count + 3 * async_count + decorated_count)
await agent_message.send(json.dumps({{'value': value}}, separators=(',', ':')), receiver_role='parent')"""
        return objective, code, "Parse both complete files, count the requested AST node types, and report JSON."
    if family == "source_config":
        toml_path = f"{root}/service.toml"
        env_path = f"{root}/features.env"
        objective = (
            f"Read {toml_path} with tomllib and {env_path} as KEY=value lines. Compute runtime "
            "workers * timeout_seconds + the number of feature values exactly equal to `true`. "
            "Send exactly one compact JSON object with integer key `value` to "
            "receiver_role='parent', then stop."
        )
        code = f"""from pathlib import Path
import json
import tomllib
config = tomllib.loads(Path({toml_path!r}).read_text())
features = dict(line.split('=', 1) for line in Path({env_path!r}).read_text().splitlines() if line)
enabled_count = sum(value == 'true' for value in features.values())
value = int(config['runtime']['workers'] * config['runtime']['timeout_seconds'] + enabled_count)
await agent_message.send(json.dumps({{'value': value}}, separators=(',', ':')), receiver_role='parent')"""
        return objective, code, "Read both configuration formats, count exact true values, and report JSON."
    raise ValueError(f"unknown specialist family: {family}")


def _rows(
    *,
    expert_id: str,
    runtime_message: dict[str, Any],
    tools: list[dict[str, Any]],
    instances_per_variant: int,
    instance_offset: int,
    source: str,
) -> list[dict[str, Any]]:
    rows = []
    for variant in range(4):
        for local_index in range(instances_per_variant):
            instance = instance_offset + local_index
            root = f"/workspace/specialist-worker/v{variant}-i{instance}"
            for family in FAMILIES[expert_id]:
                objective, code, reasoning = _family_target(family, root)
                digest = hashlib.sha256(f"{expert_id}:{family}:{variant}:{instance}:{code}".encode()).hexdigest()[:16]
                call_id = f"specialist-{digest}"
                rows.append(
                    {
                        "messages": [
                            _wire_message(copy.deepcopy(runtime_message)),
                            _wire_message(_assignment(expert_id, objective)),
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
                            {
                                "role": "tool",
                                "content": "message queued",
                                "tool_call_id": call_id,
                            },
                            {
                                "role": "assistant",
                                "content": "Done.",
                                "reasoning_content": ("The exact JSON report was delivered to my parent, so I stop."),
                                "tool_calls": [],
                            },
                        ],
                        "tools": json.dumps(tools, sort_keys=True, separators=(",", ":")),
                        "task_key": f"{expert_id}-{family}-v{variant}-i{instance}",
                        "trace_id": f"specialist-sft:{digest}",
                        "family": f"specialist_{family}",
                        "role": "child",
                        "expert_id": expert_id,
                        "objective": OBJECTIVE,
                        "source_trace": source,
                    }
                )
    return rows


def export(
    *,
    traces: list[Path],
    output_dir: Path,
    expert_id: str,
    instances_per_variant: int = 8,
    instance_offset: int = 36000,
) -> dict[str, Any]:
    if expert_id not in SPECIALISTS:
        raise ValueError(f"unsupported specialist: {expert_id}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite specialist corpus: {output_dir}")
    if instances_per_variant < 1:
        raise ValueError("instances_per_variant must be positive")
    source_records = [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in traces]
    runtime_message, tools = _live_child_context(_load_traces(traces))
    source = ",".join(record["path"] for record in source_records)
    rows = _rows(
        expert_id=expert_id,
        runtime_message=runtime_message,
        tools=tools,
        instances_per_variant=instances_per_variant,
        instance_offset=instance_offset,
        source=source,
    )
    expected_rows = 4 * instances_per_variant * len(FAMILIES[expert_id])
    if len(rows) != expected_rows or len({row["task_key"] for row in rows}) != expected_rows:
        raise ValueError("specialist corpus row cardinality is invalid")
    family_counts = {
        family: sum(row["family"] == f"specialist_{family}" for row in rows) for family in FAMILIES[expert_id]
    }
    if len(set(family_counts.values())) != 1:
        raise ValueError(f"specialist corpus is not family-balanced: {family_counts}")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "child",
        "expert_id": expert_id,
        "objective": OBJECTIVE,
        "rows": len(rows),
        "family_counts": family_counts,
        "training_template_variants": [0, 1, 2, 3],
        "instance_offset": instance_offset,
        "instances_per_variant": instances_per_variant,
        "heldout_template_variants_excluded": [4, 5],
        "source_traces": source_records,
        "answer_free": True,
        "model_authored_file_computation": True,
        "strict_json_parent_report": True,
        "tool_call_format": "openai_function_v1",
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expert-id", choices=SPECIALISTS, required=True)
    parser.add_argument("--instances-per-variant", type=int, default=8)
    parser.add_argument("--instance-offset", type=int, default=36000)
    args = parser.parse_args()
    manifest = export(
        traces=[path.resolve() for path in args.traces],
        output_dir=args.output_dir.resolve(),
        expert_id=args.expert_id,
        instances_per_variant=args.instances_per_variant,
        instance_offset=args.instance_offset,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
