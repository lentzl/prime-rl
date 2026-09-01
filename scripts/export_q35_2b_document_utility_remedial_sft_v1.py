#!/usr/bin/env python3
"""Build an answer-free direct/hierarchical utility-topology remedial batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import sha256_file
from export_q35_2b_document_utility_topology_sft_v1 import _utility_row

SCHEMA_VERSION = "qwen35-2b-document-utility-remedial-sft/v1"
OBJECTIVE = "answer_free_root_direct_and_hierarchical_utility_remediation"
HIERARCHY_SCHEMA_VERSION = "qwen35-2b-document-hierarchy-remedial-sft/v1"
HIERARCHY_OBJECTIVE = "answer_free_root_hierarchical_utility_remediation"
FAMILIES = {"document_utility_direct", "document_utility_hierarchical"}
HIERARCHY_FAMILY = "document_utility_hierarchical"


def export(
    *,
    traces: list[Path],
    output_dir: Path,
    focus_family: str | None = None,
) -> dict[str, Any]:
    if focus_family not in {None, HIERARCHY_FAMILY}:
        raise ValueError(f"unsupported document utility remedial focus: {focus_family!r}")
    selected_families = {focus_family} if focus_family else FAMILIES
    schema_version = HIERARCHY_SCHEMA_VERSION if focus_family else SCHEMA_VERSION
    objective = HIERARCHY_OBJECTIVE if focus_family else OBJECTIVE
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite document utility remedial bootstrap: {output_dir}"
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
                    if family in selected_families:
                        row = _utility_row(trace, source=resolved)
                        row["objective"] = objective
                        rows.append(row)
    rows.sort(key=lambda row: row["task_key"])
    if len(rows) != 8 or len({row["task_key"] for row in rows}) != 8:
        raise ValueError("document utility remedial bootstrap requires eight unique rows")
    family_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in sorted(selected_families)
    }
    expected_per_family = 8 if focus_family else 4
    if set(family_counts.values()) != {expected_per_family}:
        raise ValueError(
            f"document utility remedial bootstrap is not balanced: {family_counts}"
        )

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": schema_version,
        "status": "complete",
        "role": "coordinator",
        "objective": objective,
        "rows": 8,
        "training_batch_size": 8,
        "family_counts": family_counts,
        "task_keys": [row["task_key"] for row in rows],
        "source_traces": sources,
        "answer_free": True,
        "topology_choice_targeted": True,
        "utility_constraints_targeted": True,
        "remedial_classes": sorted(selected_families),
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
    parser.add_argument("--focus-family", choices=[HIERARCHY_FAMILY])
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                traces=[path.resolve() for path in args.traces],
                output_dir=args.output_dir.resolve(),
                focus_family=args.focus_family,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
