#!/usr/bin/env python3
"""Build a held-out-safe SFT set for repairing failed chapter-summary artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import _wire_message, sha256_file
from export_q35_2b_document_summary_worker_sft_v1 import (
    EVALUATION_DOCUMENT_ID,
    OUTPUT_PATH,
    TRAINING_CHAPTERS,
    _job,
    _source_context,
)

SCHEMA_VERSION = "qwen35-2b-document-summary-worker-repair-sft/v2"
OBJECTIVE = "grounded_english_chapter_summary_gate_surgical_repair"


def _repair_case(
    chapter: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, int]:
    job = _job(chapter)
    correct = chapter["report"]
    bad = json.loads(json.dumps(correct))
    merge_index = next(
        index
        for index, bullet in enumerate(bad["bullets"])
        if len(bullet["source_ids"]) > 1
    )
    missing_source_id = bad["bullets"][merge_index]["source_ids"].pop()
    return job, bad, missing_source_id, merge_index


def _messages(
    runtime_message: dict[str, Any], chapter: dict[str, Any]
) -> list[dict[str, Any]]:
    job, bad_report, missing_source_id, repair_index = _repair_case(chapter)
    correct_report = chapter["report"]
    slug = chapter["slug"]
    read_id = f"summary-repair-read-{hashlib.sha256(slug.encode()).hexdigest()[:16]}"
    write_id = f"summary-repair-write-{hashlib.sha256((slug + ':repair').encode()).hexdigest()[:16]}"
    read_code = (
        "import json\nfrom pathlib import Path\n"
        f"job_path = Path({job['path']!r})\n"
        f"output_path = Path({OUTPUT_PATH!r})\n"
        "job = json.loads(job_path.read_text(encoding='utf-8'))\n"
        "report = json.loads(output_path.read_text(encoding='utf-8'))\n"
        "{'job': job, 'report': report}"
    )
    write_code = (
        "before = json.loads(json.dumps(report['bullets']))\n"
        f"repair_index = {repair_index}\n"
        f"report['bullets'][repair_index] = {correct_report['bullets'][repair_index]!r}\n"
        "assert all(report['bullets'][i] == before[i] for i in range(3) if i != repair_index)\n"
        "expected_ids = {row['id'] for row in job['paragraphs']}\n"
        "ordered_ids = [source_id for bullet in report['bullets'] for source_id in bullet['source_ids']]\n"
        "assert len(report['bullets']) == 3 and set(ordered_ids) == expected_ids\n"
        "assert len(ordered_ids) == len(expected_ids)\n"
        "assert all(5 <= len(bullet['text'].split()) <= 45 for bullet in report['bullets'])\n"
        "written = output_path.write_text(json.dumps(report, indent=2) + '\\n', encoding='utf-8')\n"
        "written"
    )
    gate_message = (
        "Autonomous quality gate failed (attempt 1/3): `python "
        "/workspace/document-summary-v1/completion_gate.py` exited 1.\n\n"
        "Output:\ncompletion gate: write the exact grounded three-bullet chapter report at "
        f"{OUTPUT_PATH}. Diagnostic: AssertionError: missing paragraph coverage: "
        f"[{missing_source_id!r}]. Keep exactly three bullets; revise one bullet's text "
        "to summarize the missing paragraph together with its existing source, and cite both "
        "source IDs\n\n"
        "Continue working as the same terminal worker. There is no parent receiver: do not "
        "call agent_message. Do not edit the job or parse completion_gate.py. Read the original "
        "job and current report, update only the required worker-report.json in one corrective "
        "write, then stop after the gate passes."
    )
    return [
        _wire_message(runtime_message),
        {"role": "user", "content": gate_message},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": (
                "I am the terminal worker, so I will not message a parent, edit the input job, or "
                "parse the gate. I will inspect the original job and current artifact, then "
                "replace the three bullets in one atomic repair."
            ),
            "tool_calls": [
                {
                    "id": read_id,
                    "type": "function",
                    "function": {
                        "name": "ipython",
                        "arguments": json.dumps(
                            {"code": read_code}, separators=(",", ":")
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": repr({"job": job, "report": bad_report}),
            "tool_call_id": read_id,
        },
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": (
                "The gate identified one missing source in an otherwise complete report. I will "
                "change only the affected bullet, preserve the other two exactly, and stop."
            ),
            "tool_calls": [
                {
                    "id": write_id,
                    "type": "function",
                    "function": {
                        "name": "ipython",
                        "arguments": json.dumps(
                            {"code": write_code}, separators=(",", ":")
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": str(len(json.dumps(correct_report, indent=2)) + 1),
            "tool_call_id": write_id,
        },
        {
            "role": "assistant",
            "content": f"Repaired the grounded three-bullet summary at {OUTPUT_PATH}.",
            "reasoning_content": "The atomic repair write succeeded, so I must stop.",
            "tool_calls": [],
        },
    ]


def export(*, traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite summary repair SFT: {output_dir}")
    runtime_message, tools, source_path, source_trace_id = _source_context(traces)
    if len(TRAINING_CHAPTERS) != 12 or EVALUATION_DOCUMENT_ID in repr(TRAINING_CHAPTERS):
        raise ValueError("summary repair training set overlaps or has the wrong size")
    rows = [
        {
            "messages": _messages(runtime_message, chapter),
            "tools": json.dumps(tools, sort_keys=True, separators=(",", ":")),
            "task_key": f"summary-worker-repair-{chapter['slug']}",
            "trace_id": f"summary-worker-repair-authored:{chapter['slug']}",
            "family": f"summary_repair_{chapter['family']}",
            "role": "child",
            "objective": OBJECTIVE,
            "source_trace": str(source_path),
        }
        for chapter in TRAINING_CHAPTERS
    ]
    family_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in sorted({row["family"] for row in rows})
    }
    if set(family_counts.values()) != {4} or len({row["task_key"] for row in rows}) != 12:
        raise ValueError("summary repair training set is not balanced and unique")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "child",
        "objective": OBJECTIVE,
        "rows": len(rows),
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
        "repair_only": True,
        "surgical_single_bullet_repair": True,
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
