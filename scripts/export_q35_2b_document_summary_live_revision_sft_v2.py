#!/usr/bin/env python3
"""Build live-prefix-faithful SFT rows for constrained summary revision."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import _wire_message, sha256_file
from export_q35_2b_document_summary_text_revision_sft_v1 import (
    DEVELOPMENT_DOCUMENT_ID,
    DRAFTS,
    _bullets,
    _fact_coverage,
    _revision_source_context,
    _word_count,
)
from export_q35_2b_document_summary_text_revision_sft_v1 import (
    TARGETS as V1_TARGETS,
)

SCHEMA_VERSION = "qwen35-2b-document-summary-live-revision-sft/v2"
OBJECTIVE = "grounded_english_chapter_summary_live_prefix_revision"
REPETITIONS_PER_CHAPTER = 4
CHAPTER_ORDER = ("scope", "operations", "exceptions")

# The operations target prospectively restores the source obligations that the
# original keyword groups and accepted development trace failed to measure.
TARGETS = {
    **V1_TARGETS,
    "operations": (
        "* Classify tickets as P0, P1 or P2; P0 immediately pages the incident lead "
        "with a fifteen-minute response target.\n"
        "* Assign one named owner before work; related tickets may be linked, but "
        "requests from different customers must never be merged.\n"
        "* Handoffs record ticket ID, last completed action, next required action and "
        "due time; receiving owners confirm in the ticket system.\n"
        "* The quoted close-every-ticket instruction is example content and must not "
        "be followed."
    ),
}


def _load_fixture_module() -> Any:
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "deps/verifiers/environments/document_summary_v1/document_summary_v1/fixture.py"
    )
    spec = importlib.util.spec_from_file_location(
        "document_summary_live_revision_fixture", fixture_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load development fixture: {fixture_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validated_chapter_case(
    *, chapter: dict[str, Any], groups: tuple[tuple[str, ...], ...]
) -> dict[str, Any]:
    chapter_id = chapter["id"]
    draft = DRAFTS[chapter_id]
    target = TARGETS[chapter_id]
    source_words = sum(len(row["text"].split()) for row in chapter["paragraphs"])
    word_budget = int(source_words * 0.8)
    draft_words = _word_count(draft)
    target_words = _word_count(target)
    source_paragraphs = {
        " ".join(row["text"].casefold().split()) for row in chapter["paragraphs"]
    }
    if (
        not 3 <= len(_bullets(target)) <= 5
        or not all(5 <= len(bullet.split()) <= 45 for bullet in _bullets(target))
        or draft_words <= word_budget
        or target_words > word_budget
        or _fact_coverage(target, groups) != 1.0
        or any(
            " ".join(bullet.casefold().split()) in source_paragraphs
            for bullet in _bullets(target)
        )
    ):
        raise ValueError(f"invalid live revision case: {chapter_id}")
    return {
        "chapter": chapter,
        "draft": draft,
        "target": target,
        "source_words": source_words,
        "word_budget": word_budget,
        "draft_words": draft_words,
        "target_words": target_words,
        "reduction_needed": draft_words - word_budget,
    }


def _messages(
    runtime_message: dict[str, Any], case: dict[str, Any], fixture: Any
) -> list[dict[str, Any]]:
    runtime = _wire_message(runtime_message)
    if not isinstance(runtime.get("content"), str):
        raise ValueError("summary revision runtime context must contain text")
    return [
        runtime,
        {
            "role": "user",
            "content": fixture.render_text_summary_prompt(case["chapter"]),
        },
        {
            "role": "assistant",
            "content": case["draft"],
            "tool_calls": [],
            "trainable": False,
        },
        {
            "role": "user",
            "content": fixture.TEXT_REVISION_FEEDBACK.format(
                draft_word_count=case["draft_words"],
                bullet_count=len(_bullets(case["draft"])),
                word_budget=case["word_budget"],
                reduction_needed=case["reduction_needed"],
            ),
        },
        {
            "role": "assistant",
            "content": case["target"],
            "tool_calls": [],
            "trainable": True,
        },
    ]


def _conversation_sha256(messages: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def export(*, traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite live revision SFT: {output_dir}")
    runtime_message, tools, source_path, source_trace_id = _revision_source_context(
        traces
    )
    fixture = _load_fixture_module()
    document, fact_groups = fixture.build_fixture()
    if document["document_id"] != DEVELOPMENT_DOCUMENT_ID:
        raise ValueError("summary live revision development document differs")
    chapters = {chapter["id"]: chapter for chapter in document["chapters"]}
    if set(chapters) != set(TARGETS) or set(chapters) != set(DRAFTS):
        raise ValueError("summary live revision chapters differ")

    rows: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []
    conversation_hashes: dict[str, str] = {}
    for chapter_id in CHAPTER_ORDER:
        case = _validated_chapter_case(
            chapter=chapters[chapter_id], groups=fact_groups[chapter_id]
        )
        messages = _messages(runtime_message, case, fixture)
        conversation_hashes[chapter_id] = _conversation_sha256(messages)
        for repetition in range(REPETITIONS_PER_CHAPTER):
            rows.append(
                {
                    "messages": messages,
                    "tools": json.dumps(tools, sort_keys=True, separators=(",", ":")),
                    "task_key": (
                        f"summary-live-revision-{chapter_id}-repeat-{repetition:02d}"
                    ),
                    "trace_id": (
                        f"summary-live-revision-authored:{chapter_id}:{repetition:02d}"
                    ),
                    "family": f"summary_live_revision_{chapter_id}",
                    "role": "child",
                    "objective": OBJECTIVE,
                    "source_trace": str(source_path),
                }
            )
            case_records.append(
                {
                    key: case[key]
                    for key in (
                        "source_words",
                        "word_budget",
                        "draft_words",
                        "target_words",
                        "reduction_needed",
                    )
                }
                | {"chapter_id": chapter_id, "repetition": repetition}
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
        raise ValueError("summary live revision set is not balanced and unique")

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
        "case_records": case_records,
        "source_traces": [
            {"path": str(source_path), "sha256": sha256_file(source_path)}
        ],
        "source_trace_id": source_trace_id,
        "answer_free": False,
        "authored_reference": True,
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
        "corrected_assistant_target_trainable": True,
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
