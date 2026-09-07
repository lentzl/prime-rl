#!/usr/bin/env python3
"""Build an interleaved SFT set for chapter summarization and gate repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import sha256_file
from export_q35_2b_document_summary_worker_repair_sft_v1 import (
    _messages as _repair_messages,
)
from export_q35_2b_document_summary_worker_sft_v1 import (
    EVALUATION_DOCUMENT_ID,
    TRAINING_CHAPTERS,
    _source_context,
)
from export_q35_2b_document_summary_worker_sft_v1 import (
    _messages as _summary_messages,
)

SCHEMA_VERSION = "qwen35-2b-document-summary-worker-mixed-sft/v1"
OBJECTIVE = "grounded_english_chapter_summary_and_gate_repair"


def _row(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chapter: dict[str, Any],
    source_path: Path,
    kind: str,
) -> dict[str, Any]:
    slug = chapter["slug"]
    return {
        "messages": messages,
        "tools": json.dumps(tools, sort_keys=True, separators=(",", ":")),
        "task_key": f"summary-worker-{kind}-{slug}",
        "trace_id": f"summary-worker-{kind}-authored:{slug}",
        "family": f"summary_{kind}_{chapter['family']}",
        "role": "child",
        "objective": OBJECTIVE,
        "source_trace": str(source_path),
    }


def export(*, traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite mixed summary SFT: {output_dir}")
    runtime_message, tools, source_path, source_trace_id = _source_context(traces)
    if len(TRAINING_CHAPTERS) != 12 or EVALUATION_DOCUMENT_ID in repr(TRAINING_CHAPTERS):
        raise ValueError("mixed summary training set overlaps or has the wrong size")

    rows = []
    for chapter in TRAINING_CHAPTERS:
        rows.extend(
            (
                _row(
                    messages=_summary_messages(runtime_message, chapter),
                    tools=tools,
                    chapter=chapter,
                    source_path=source_path,
                    kind="base",
                ),
                _row(
                    messages=_repair_messages(runtime_message, chapter),
                    tools=tools,
                    chapter=chapter,
                    source_path=source_path,
                    kind="repair",
                ),
            )
        )
    family_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in sorted({row["family"] for row in rows})
    }
    if (
        len(rows) != 24
        or set(family_counts.values()) != {4}
        or len({row["task_key"] for row in rows}) != 24
        or any(
            [row["task_key"].split("-")[2] for row in rows[offset : offset + 12]].count(
                "base"
            )
            != 6
            for offset in (0, 12)
        )
    ):
        raise ValueError("mixed summary set is not uniquely interleaved and balanced")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "child",
        "objective": OBJECTIVE,
        "rows": len(rows),
        "base_rows": 12,
        "repair_rows": 12,
        "batch_size": 12,
        "batch_composition": "six_base_six_repair_in_each_of_two_batches",
        "family_counts": family_counts,
        "task_keys": [row["task_key"] for row in rows],
        "source_traces": [
            {"path": str(source_path), "sha256": sha256_file(source_path)}
        ],
        "source_trace_id": source_trace_id,
        "answer_free": False,
        "authored_reference": True,
        "native_prime_agent_context": True,
        "on_policy_failure_context": True,
        "initial_bad_assistant_turns": 0,
        "evaluation_document_excluded": True,
        "evaluation_document_id": EVALUATION_DOCUMENT_ID,
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
