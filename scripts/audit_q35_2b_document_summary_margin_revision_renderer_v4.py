#!/usr/bin/env python3
"""Audit the on-policy margin summary revision boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_q35_2b_document_summary_commit_revision_renderer_v3 import (
    audit as audit_commit_revision,
)
from export_q35_2b_document_summary_margin_revision_sft_v4 import (
    _observed_cases,
    export,
)

AUDIT_SCHEMA_VERSION = (
    "qwen35-2b-document-summary-margin-revision-renderer-audit/v4"
)


def audit(
    *, traces: list[Path], tokenizer_path: Path, dataset_dir: Path | None = None
) -> dict[str, Any]:
    def export_for_audit(
        *, traces: list[Path], output_dir: Path
    ) -> dict[str, Any]:
        return export(
            traces=traces,
            source_model=tokenizer_path,
            output_dir=output_dir,
        )

    result = audit_commit_revision(
        traces=traces,
        tokenizer_path=tokenizer_path,
        dataset_dir=dataset_dir,
        observed_cases_fn=_observed_cases,
        export_fn=export_for_audit,
        audit_schema_version=AUDIT_SCHEMA_VERSION,
        family_prefix="summary_margin_revision",
        temporary_prefix="summary-margin-revision-audit-",
    )
    if not all(
        chapter.get("wire_history_reasoning_stripped") is True
        and chapter.get("raw_history_token_equivalent") is True
        for chapter in result["chapters"]
    ):
        raise ValueError("margin revision traces do not expose the exact wire prefix")
    result["feedback_safety_margin_words"] = 3
    result["on_policy_margin_prefix_verified"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", action="append", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                traces=[path.resolve() for path in args.traces],
                tokenizer_path=args.tokenizer.resolve(),
                dataset_dir=args.dataset_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
