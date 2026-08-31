#!/usr/bin/env python3
"""Build order-robust exact-context manager aggregation and parent-report SFT."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_coordinator_fanin_sft_v1 import (
    _ack,
    _child_message,
    _stats,
)
from export_q35_2b_document_decision_sft_v1 import STEMS, sha256_file
from export_q35_2b_document_manager_admission_sft_v1 import _manager_prefix
from export_q35_2b_document_manager_fanin_sft_v1 import (
    _parent_report,
    _spawn_action,
    _spawn_receipt,
)

SCHEMA_VERSION = "qwen35-2b-document-manager-aggregation-permuted-sft/v1"
OBJECTIVE = "depth_two_manager_order_robust_complete_fanin_parent_report"
ORDERS = tuple(itertools.permutations(STEMS))


def _rows(trace: dict[str, Any], *, source: Path) -> list[dict[str, Any]]:
    task = trace.get("task", {}).get("data", {})
    if task.get("family") != "document_hierarchical" or not isinstance(
        task.get("name"), str
    ):
        raise ValueError("manager aggregation source is not a hierarchical document task")
    files = task.get("files")
    if not isinstance(files, dict) or len(files) != 3:
        raise ValueError("manager aggregation source lacks three document fixtures")
    roots = {str(Path(path).parent) for path in files}
    if len(roots) != 1:
        raise ValueError("manager aggregation fixtures do not share one directory")
    root = roots.pop()
    if set(files) != {f"{root}/{stem}.md" for stem in STEMS}:
        raise ValueError("manager aggregation fixtures do not match canonical shards")
    tools = trace.get("tools") or []
    if [tool.get("name") for tool in tools if isinstance(tool, dict)] != [
        "ipython"
    ]:
        raise ValueError("manager aggregation source must expose exactly the IPython tool")

    trace_id = str(trace.get("id"))
    prefix = _manager_prefix(trace)
    spawn, call_id = _spawn_action(root=root, trace_id=trace_id)
    receipt = _spawn_receipt(call_id)
    stats = _stats(files, root)
    rows = []
    for order in ORDERS:
        order_name = "-".join(order)
        messages = [
            *copy.deepcopy(prefix),
            copy.deepcopy(spawn),
            copy.deepcopy(receipt),
        ]
        for index, stem in enumerate(order):
            messages.append(_child_message(stem, stats[stem]))
            remaining = order[index + 1 :]
            if remaining:
                messages.append(_ack(stem, remaining))
        messages.append(_parent_report(stats, trace_id=f"{trace_id}:{order_name}"))
        rows.append(
            {
                "messages": messages,
                "tools": json.dumps(tools, sort_keys=True, separators=(",", ":")),
                "task_key": f"{task['name']}:complete_fanin_report:{order_name}",
                "trace_id": f"document-manager-aggregation-permuted:{trace_id}:{order_name}",
                "family": f"document_manager_fanin_order_{order_name}",
                "phase": "complete_fanin_report",
                "arrival_order": order_name,
                "role": "coordinator",
                "objective": OBJECTIVE,
                "source_trace": str(source),
            }
        )
    return rows


def export(*, traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite permuted manager aggregation bootstrap: {output_dir}"
        )
    rows: list[dict[str, Any]] = []
    sources = []
    for path in traces:
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved = path.resolve()
        sources.append({"path": str(resolved), "sha256": sha256_file(path)})
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                for trace in json.loads(line).get("traces") or []:
                    if (
                        trace.get("task", {}).get("data", {}).get("family")
                        == "document_hierarchical"
                    ):
                        rows.extend(_rows(trace, source=resolved))
    rows.sort(key=lambda row: row["task_key"])
    if len(rows) != 24 or len({row["task_key"] for row in rows}) != 24:
        raise ValueError("permuted manager aggregation bootstrap requires 24 unique rows")
    family_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in sorted({row["family"] for row in rows})
    }
    if len(family_counts) != 6 or set(family_counts.values()) != {4}:
        raise ValueError(
            f"manager aggregation permutations are not balanced: {family_counts}"
        )

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "coordinator",
        "objective": OBJECTIVE,
        "rows": 24,
        "training_batch_size": 12,
        "family_counts": family_counts,
        "arrival_orders": ["-".join(order) for order in ORDERS],
        "task_keys": [row["task_key"] for row in rows],
        "source_traces": sources,
        "answer_free": False,
        "exact_live_depth_two_context": True,
        "root_policy_targeted": False,
        "child_policy_targeted": False,
        "manager_admission_targeted": False,
        "manager_aggregation_targeted": True,
        "arrival_order_balanced": True,
        "tool_call_format": "openai_function_v1",
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
