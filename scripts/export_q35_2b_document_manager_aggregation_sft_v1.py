#!/usr/bin/env python3
"""Build exact-context depth-one manager aggregation and parent-report SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import sha256_file
from export_q35_2b_document_manager_fanin_sft_v1 import _rows

SCHEMA_VERSION = "qwen35-2b-document-manager-aggregation-sft/v1"
OBJECTIVE = "depth_two_manager_complete_fanin_parent_report"
TARGET_PHASE = "complete_fanin_report"


def export(*, traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite manager aggregation bootstrap: {output_dir}"
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
                    task = trace.get("task", {}).get("data", {})
                    if task.get("family") != "document_hierarchical":
                        continue
                    row = next(
                        row
                        for row in _rows(trace, source=resolved)
                        if row["phase"] == TARGET_PHASE
                    )
                    row["objective"] = OBJECTIVE
                    row["family"] = "document_manager_complete_fanin_report"
                    row["trace_id"] = row["trace_id"].replace(
                        "document-manager-fanin:", "document-manager-aggregation:", 1
                    )
                    rows.append(row)
    rows.sort(key=lambda row: row["task_key"])
    if len(rows) != 4 or len({row["task_key"] for row in rows}) != 4:
        raise ValueError("manager aggregation bootstrap requires four unique rows")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "coordinator",
        "objective": OBJECTIVE,
        "rows": 4,
        "family_counts": {"document_manager_complete_fanin_report": 4},
        "task_keys": [row["task_key"] for row in rows],
        "source_traces": sources,
        "answer_free": False,
        "exact_live_depth_two_context": True,
        "root_policy_targeted": False,
        "child_policy_targeted": False,
        "manager_admission_targeted": False,
        "manager_aggregation_targeted": True,
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
