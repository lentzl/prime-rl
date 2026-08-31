#!/usr/bin/env python3
"""Build a balanced answer-free worker-versus-manager topology bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import _canonical_row, sha256_file

SCHEMA_VERSION = "qwen35-2b-document-topology-contrast-sft/v1"
OBJECTIVE = "answer_free_root_worker_versus_manager_topology_choice"
FAMILIES = {"document_flat", "document_hierarchical"}


def export(*, traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite document topology bootstrap: {output_dir}"
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
                    if family not in FAMILIES:
                        continue
                    row = _canonical_row(trace, source=resolved)
                    row["objective"] = OBJECTIVE
                    row["trace_id"] = row["trace_id"].replace(
                        "document-decision:", "document-topology-contrast:", 1
                    )
                    rows.append(row)
    rows.sort(key=lambda row: row["task_key"])
    if len(rows) != 8 or len({row["task_key"] for row in rows}) != 8:
        raise ValueError("document topology bootstrap requires eight unique rows")
    family_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in sorted(FAMILIES)
    }
    if set(family_counts.values()) != {4}:
        raise ValueError(f"document topology bootstrap is not balanced: {family_counts}")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "coordinator",
        "objective": OBJECTIVE,
        "rows": 8,
        "training_batch_size": 8,
        "family_counts": family_counts,
        "task_keys": [row["task_key"] for row in rows],
        "source_traces": sources,
        "answer_free": True,
        "topology_choice_targeted": True,
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
