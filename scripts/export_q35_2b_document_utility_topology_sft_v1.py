#!/usr/bin/env python3
"""Build a balanced answer-free utility-topology coordinator bootstrap."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import _canonical_row, sha256_file

SCHEMA_VERSION = "qwen35-2b-document-utility-topology-sft/v1"
OBJECTIVE = "answer_free_root_topology_choice_from_ownership_and_resource_constraints"
FAMILY_TO_TOPOLOGY_FAMILY = {
    "document_utility_direct": "document_direct",
    "document_utility_flat": "document_flat",
    "document_utility_hierarchical": "document_hierarchical",
}


def _utility_row(trace: dict[str, Any], *, source: Path) -> dict[str, Any]:
    family = trace.get("task", {}).get("data", {}).get("family")
    topology_family = FAMILY_TO_TOPOLOGY_FAMILY.get(family)
    if topology_family is None:
        raise ValueError(f"trace is not a utility-topology task: {family!r}")
    canonical_trace = copy.deepcopy(trace)
    canonical_trace["task"]["data"]["family"] = topology_family
    row = _canonical_row(canonical_trace, source=source)
    row.update(
        {
            "family": family,
            "objective": OBJECTIVE,
            "trace_id": row["trace_id"].replace(
                "document-decision:", "document-utility-topology:", 1
            ),
        }
    )
    return row


def export(*, traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite document utility-topology bootstrap: {output_dir}"
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
                    family = trace.get("task", {}).get("data", {}).get("family")
                    if family in FAMILY_TO_TOPOLOGY_FAMILY:
                        rows.append(_utility_row(trace, source=resolved))
    rows.sort(key=lambda row: row["task_key"])
    if len(rows) != 6 or len({row["task_key"] for row in rows}) != 6:
        raise ValueError("document utility-topology bootstrap requires six unique rows")
    family_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in sorted(FAMILY_TO_TOPOLOGY_FAMILY)
    }
    if set(family_counts.values()) != {2}:
        raise ValueError(
            f"document utility-topology bootstrap is not balanced: {family_counts}"
        )

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "coordinator",
        "objective": OBJECTIVE,
        "rows": 6,
        "training_batch_size": 6,
        "family_counts": family_counts,
        "task_keys": [row["task_key"] for row in rows],
        "source_traces": sources,
        "answer_free": True,
        "topology_choice_targeted": True,
        "utility_constraints_targeted": True,
        "protocol_mechanics_targeted": False,
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
