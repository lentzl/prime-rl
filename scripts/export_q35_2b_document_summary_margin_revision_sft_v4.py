#!/usr/bin/env python3
"""Build on-policy SFT rows from margin-scaffold summary revision turns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from export_q35_2b_document_decision_sft_v1 import sha256_file
from export_q35_2b_document_summary_commit_revision_sft_v3 import (
    _expected_feedback as _expected_commit_feedback,
)
from export_q35_2b_document_summary_commit_revision_sft_v3 import (
    _load_fixture_module,
)
from export_q35_2b_document_summary_commit_revision_sft_v3 import (
    _observed_cases as _observed_commit_cases,
)
from export_q35_2b_document_summary_commit_revision_sft_v3 import (
    export as export_commit_revision,
)

SCHEMA_VERSION = "qwen35-2b-document-summary-margin-revision-sft/v4"
OBJECTIVE = "grounded_english_chapter_summary_on_policy_margin_commit_revision"
SAFETY_MARGIN_WORDS = 3


def _expected_feedback(*, fixture: Any, draft: str, word_budget: int) -> str:
    feedback = _expected_commit_feedback(
        fixture=fixture, draft=draft, word_budget=word_budget
    )
    suffix = "Return only the revised bullets."
    if not feedback.endswith(suffix):
        raise ValueError("margin revision feedback suffix differs")
    return (
        feedback[: -len(suffix)]
        + fixture.TEXT_REVISION_SAFETY_MARGIN_REQUIREMENT.format(
            target_word_count=max(0, word_budget - SAFETY_MARGIN_WORDS)
        )
        + suffix
    )


def _observed_cases(
    *, traces: list[Path], fixture: Any, chapters: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    return _observed_commit_cases(
        traces=traces,
        fixture=fixture,
        chapters=chapters,
        expected_feedback_builder=_expected_feedback,
    )


def export(
    *, traces: list[Path], source_model: Path, output_dir: Path
) -> dict[str, Any]:
    source_model = source_model.resolve()
    source_weight = source_model / "model.safetensors"
    if not source_weight.is_file() or not (source_model / "STABLE").is_file():
        raise ValueError(f"on-policy summary source checkpoint is incomplete: {source_model}")
    fixture = _load_fixture_module()
    document, _ = fixture.build_fixture()
    feedback_targets = {
        chapter["id"]: max(
            0,
            int(
                sum(len(row["text"].split()) for row in chapter["paragraphs"])
                * 0.8
            )
            - SAFETY_MARGIN_WORDS,
        )
        for chapter in document["chapters"]
    }
    return export_commit_revision(
        traces=traces,
        output_dir=output_dir,
        expected_feedback_builder=_expected_feedback,
        schema_version=SCHEMA_VERSION,
        objective=OBJECTIVE,
        task_prefix="summary-margin-revision",
        trace_prefix="summary-margin-revision-observed",
        family_prefix="summary_margin_revision",
        manifest_extra={
            "feedback_contract": "commit_once_with_three_word_safety_margin_v1",
            "feedback_safety_margin_words": SAFETY_MARGIN_WORDS,
            "feedback_target_words_by_chapter": feedback_targets,
            "on_policy_source_model": str(source_model),
            "on_policy_source_model_sha256": sha256_file(source_weight),
            "observed_current_revision_outputs_trainable": False,
            "on_policy_development_failures_only": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", action="append", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                traces=[path.resolve() for path in args.traces],
                source_model=args.source_model,
                output_dir=args.output_dir.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
