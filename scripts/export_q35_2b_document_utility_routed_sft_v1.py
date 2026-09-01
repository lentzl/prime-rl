#!/usr/bin/env python3
"""Build utility-topology SFT rows aligned to the routed root prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset
from dual_policy_openai_proxy_v1 import ROOT_COORDINATOR_CONTRACT
from export_q35_2b_document_decision_sft_v1 import sha256_file
from export_q35_2b_document_utility_topology_sft_v1 import (
    FAMILY_TO_TOPOLOGY_FAMILY,
    _utility_row,
)

SCHEMA_VERSION = "qwen35-2b-document-utility-routed-sft/v1"
OBJECTIVE = (
    "answer_free_root_topology_choice_from_routed_ownership_and_resource_constraints"
)
ROWS = 24
PER_FAMILY = 8


def _sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    match = re.search(r"-v(?P<variant>\d+)-i(?P<instance>\d+)$", row["task_key"])
    if match is None:
        raise ValueError(f"invalid routed utility task key: {row['task_key']!r}")
    return int(match.group("instance")), int(match.group("variant")), row["family"]


def export(*, traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite routed document utility bootstrap: {output_dir}"
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
                    if family not in FAMILY_TO_TOPOLOGY_FAMILY:
                        continue
                    row = _utility_row(trace, source=resolved)
                    if any(
                        message.get("content") == ROOT_COORDINATOR_CONTRACT
                        for message in row["messages"]
                    ):
                        raise ValueError("source trace already contains the routed root contract")
                    row["messages"].insert(
                        0, {"role": "system", "content": ROOT_COORDINATOR_CONTRACT}
                    )
                    row["objective"] = OBJECTIVE
                    rows.append(row)
    rows.sort(key=_sort_key)
    if len(rows) != ROWS or len({row["task_key"] for row in rows}) != ROWS:
        raise ValueError("routed document utility bootstrap requires 24 unique rows")
    family_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in sorted(FAMILY_TO_TOPOLOGY_FAMILY)
    }
    if set(family_counts.values()) != {PER_FAMILY}:
        raise ValueError(
            f"routed document utility bootstrap is not balanced: {family_counts}"
        )
    for start in range(0, ROWS, 6):
        if {row["family"] for row in rows[start : start + 6]} != set(
            FAMILY_TO_TOPOLOGY_FAMILY
        ):
            raise ValueError("routed document utility ordering is not locally balanced")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "coordinator",
        "objective": OBJECTIVE,
        "rows": ROWS,
        "training_batch_size": 12,
        "family_counts": family_counts,
        "task_keys": [row["task_key"] for row in rows],
        "source_traces": sources,
        "answer_free": True,
        "topology_choice_targeted": True,
        "utility_constraints_targeted": True,
        "root_coordinator_contract_aligned": True,
        "root_coordinator_contract_sha256": hashlib.sha256(
            ROOT_COORDINATOR_CONTRACT.encode()
        ).hexdigest(),
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
