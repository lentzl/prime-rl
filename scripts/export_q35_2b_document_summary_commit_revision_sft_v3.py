#!/usr/bin/env python3
"""Build scaffold-aligned SFT rows from observed summary revision turns."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import _wire_message, sha256_file
from export_q35_2b_document_summary_live_revision_sft_v2 import (
    CHAPTER_ORDER,
    REPETITIONS_PER_CHAPTER,
    TARGETS,
    _conversation_sha256,
    _load_fixture_module,
    _validated_chapter_case,
)
from export_q35_2b_document_summary_text_revision_sft_v1 import (
    DEVELOPMENT_DOCUMENT_ID,
    _bullets,
    _word_count,
)

SCHEMA_VERSION = "qwen35-2b-document-summary-commit-revision-sft/v3"
OBJECTIVE = "grounded_english_chapter_summary_scaffold_aligned_commit_revision"


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            part.get("text", "") for part in value if isinstance(part, dict)
        )
    return ""


def _expected_feedback(*, fixture: Any, draft: str, word_budget: int) -> str:
    bullets = _bullets(draft)
    draft_words = _word_count(draft)
    base = fixture.TEXT_REVISION_FEEDBACK.format(
        draft_word_count=draft_words,
        bullet_count=len(bullets),
        word_budget=word_budget,
        reduction_needed=max(0, draft_words - word_budget),
    )
    suffix = "Return only the revised bullets."
    if not base.endswith(suffix):
        raise ValueError("revision feedback suffix differs")
    return (
        base[: -len(suffix)]
        + fixture.TEXT_REVISION_COMMIT_REQUIREMENT.format(
            bullet_count=len(bullets)
        )
        + suffix
    )


def _observed_cases(
    *,
    traces: list[Path],
    fixture: Any,
    chapters: dict[str, dict[str, Any]],
    expected_feedback_builder: Callable[..., str] = _expected_feedback,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    cases: dict[str, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    for path in traces:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                for trace in json.loads(line).get("traces") or []:
                    nodes = trace.get("nodes") or []
                    if (
                        trace.get("task", {}).get("type")
                        != "DocumentSummaryTextTask"
                        or len(nodes) != 5
                    ):
                        continue
                    raw_messages = [node.get("message", {}) for node in nodes]
                    task_prompt = _content_text(raw_messages[1].get("content"))
                    matched = [
                        chapter_id
                        for chapter_id, chapter in chapters.items()
                        if task_prompt.strip()
                        == fixture.render_text_summary_prompt(chapter).strip()
                    ]
                    if len(matched) != 1:
                        continue
                    chapter_id = matched[0]
                    if chapter_id in cases:
                        raise ValueError(f"duplicate observed revision: {chapter_id}")
                    draft = _content_text(raw_messages[2].get("content"))
                    if not draft.strip():
                        raise ValueError(f"observed draft is empty: {chapter_id}")
                    source_words = sum(
                        len(row["text"].split())
                        for row in chapters[chapter_id]["paragraphs"]
                    )
                    word_budget = int(source_words * 0.8)
                    feedback = _content_text(raw_messages[3].get("content"))
                    expected_feedback = expected_feedback_builder(
                        fixture=fixture, draft=draft, word_budget=word_budget
                    )
                    if feedback != expected_feedback:
                        raise ValueError(
                            f"observed scaffold feedback differs: {chapter_id}"
                        )
                    tools = trace.get("tools") or []
                    if [tool.get("name") for tool in tools] != ["ipython"]:
                        raise ValueError(f"observed tool surface differs: {chapter_id}")
                    messages = [
                        _wire_message(raw_messages[0]),
                        {"role": "user", "content": task_prompt},
                        {
                            "role": "assistant",
                            "content": draft,
                            "tool_calls": [],
                            "trainable": False,
                        },
                        {"role": "user", "content": feedback},
                        {
                            "role": "assistant",
                            "content": TARGETS[chapter_id],
                            "mask_generation_prompt": True,
                            "tool_calls": [],
                            "trainable": True,
                        },
                    ]
                    cases[chapter_id] = {
                        "messages": messages,
                        "raw_history": [
                            _wire_message(message) for message in raw_messages[:4]
                        ],
                        "tools": tools,
                        "trace_id": trace.get("id"),
                        "draft_words": _word_count(draft),
                        "word_budget": word_budget,
                        "raw_reasoning_present": bool(
                            raw_messages[2].get("reasoning_content")
                            or raw_messages[2].get("reasoning_details")
                        ),
                    }
                    evidence.append(
                        {
                            "chapter_id": chapter_id,
                            "path": str(path.resolve()),
                            "sha256": sha256_file(path),
                            "trace_id": trace.get("id"),
                        }
                    )
    if set(cases) != set(CHAPTER_ORDER):
        raise ValueError(
            f"expected observed revision traces for {CHAPTER_ORDER}, found {tuple(cases)}"
        )
    tools = {
        json.dumps(case["tools"], sort_keys=True, separators=(",", ":"))
        for case in cases.values()
    }
    if len(tools) != 1:
        raise ValueError("observed Prime Agent tool surface differs")
    return cases, sorted(evidence, key=lambda row: CHAPTER_ORDER.index(row["chapter_id"]))


def export(
    *,
    traces: list[Path],
    output_dir: Path,
    expected_feedback_builder: Callable[..., str] = _expected_feedback,
    schema_version: str = SCHEMA_VERSION,
    objective: str = OBJECTIVE,
    task_prefix: str = "summary-commit-revision",
    trace_prefix: str = "summary-commit-revision-observed",
    family_prefix: str = "summary_commit_revision",
    manifest_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite commit revision SFT: {output_dir}")
    fixture = _load_fixture_module()
    document, fact_groups = fixture.build_fixture()
    if document["document_id"] != DEVELOPMENT_DOCUMENT_ID:
        raise ValueError("summary commit revision development document differs")
    chapters = {chapter["id"]: chapter for chapter in document["chapters"]}
    observed, evidence = _observed_cases(
        traces=traces,
        fixture=fixture,
        chapters=chapters,
        expected_feedback_builder=expected_feedback_builder,
    )

    rows: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []
    conversation_hashes: dict[str, str] = {}
    for chapter_id in CHAPTER_ORDER:
        authored = _validated_chapter_case(
            chapter=chapters[chapter_id], groups=fact_groups[chapter_id]
        )
        observed_case = observed[chapter_id]
        messages = observed_case["messages"]
        if (
            observed_case["draft_words"] <= observed_case["word_budget"]
            or authored["target_words"] > observed_case["word_budget"]
        ):
            raise ValueError(f"invalid observed revision boundary: {chapter_id}")
        conversation_hashes[chapter_id] = _conversation_sha256(messages)
        serialized_tools = json.dumps(
            observed_case["tools"], sort_keys=True, separators=(",", ":")
        )
        for repetition in range(REPETITIONS_PER_CHAPTER):
            rows.append(
                {
                    "messages": messages,
                    "tools": serialized_tools,
                    "task_key": f"{task_prefix}-{chapter_id}-repeat-{repetition:02d}",
                    "trace_id": f"{trace_prefix}:{chapter_id}:{repetition:02d}",
                    "family": f"{family_prefix}_{chapter_id}",
                    "role": "child",
                    "objective": objective,
                    "source_trace": next(
                        row["path"] for row in evidence if row["chapter_id"] == chapter_id
                    ),
                }
            )
            case_records.append(
                {
                    "chapter_id": chapter_id,
                    "repetition": repetition,
                    "draft_words": observed_case["draft_words"],
                    "word_budget": observed_case["word_budget"],
                    "target_words": authored["target_words"],
                    "raw_reasoning_present": observed_case["raw_reasoning_present"],
                }
            )

    family_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in sorted({row["family"] for row in rows})
    }
    if (
        len(rows) != 12
        or set(family_counts.values()) != {REPETITIONS_PER_CHAPTER}
        or len({row["task_key"] for row in rows}) != 12
        or len(set(conversation_hashes.values())) != 3
    ):
        raise ValueError("summary commit revision set is not balanced and unique")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": schema_version,
        "status": "complete",
        "role": "child",
        "objective": objective,
        "rows": len(rows),
        "family_counts": family_counts,
        "task_keys": [row["task_key"] for row in rows],
        "case_records": case_records,
        "source_traces": evidence,
        "answer_free": False,
        "authored_reference": True,
        "observed_live_draft_context": True,
        "native_prime_agent_context": True,
        "prime_agent_tools_available": True,
        "live_revision_role_sequence": [
            "runtime_user",
            "task_user",
            "assistant_draft_context",
            "revision_feedback_user",
            "assistant_corrected_target",
        ],
        "assistant_messages_per_row": 2,
        "assistant_target_messages_per_row": 1,
        "context_assistant_messages_per_row": 1,
        "prior_assistant_draft_trainable": False,
        "prior_assistant_reasoning_present_in_training": False,
        "live_prior_assistant_reasoning_stripped_by_scaffold": True,
        "corrected_assistant_target_trainable": True,
        "generation_prompt_common_prefix_trainable": False,
        "renderer_enable_thinking": False,
        "completion_boundary_alignment": "renderer_common_prefix_v1",
        "live_transfer_check_required": True,
        "distinct_conversation_payloads": 3,
        "repetitions_per_chapter": REPETITIONS_PER_CHAPTER,
        "broad_skill_claim": False,
        "failed_reasoning_tokens_in_targets": False,
        "requires_renderer_boundary_audit_before_training": True,
        "word_counter": "markdown_bullet_regex_then_python_str_split_v1",
        "development_document_id": DEVELOPMENT_DOCUMENT_ID,
        "development_chapter_ids": list(CHAPTER_ORDER),
        "fresh_confirmation_documents_reserved": True,
        "tool_call_format": "openai_function_v1",
        "conversation_sha256_by_chapter": conversation_hashes,
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    if manifest_extra:
        overlap = set(manifest).intersection(manifest_extra)
        if overlap:
            raise ValueError(
                f"summary commit revision manifest extras collide: {sorted(overlap)}"
            )
        manifest.update(manifest_extra)
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
