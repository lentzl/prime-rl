#!/usr/bin/env python3
"""Mix balanced live compute rows with admitted production-reporting rows."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dual_policy_openai_proxy_v1 import LEAF_REPORTER_CONTRACT

SCHEMA_VERSION = "q35-2b-child-consolidation-mix/v1"
SUPPORTED_FAMILIES = {
    "csv_total",
    "json_max",
    "json_sum",
    "log_error",
    "md_h2",
    "python_defs",
    "word_count",
}
EXPECTED_COLUMNS = [
    "messages",
    "tools",
    "axis",
    "phase",
    "task_key",
    "trace_id",
    "role",
    "objective",
    "expected_result",
    "resource_family",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def audit_answer_free_row(row: dict[str, Any]) -> None:
    """Fail closed unless one row computes from evidence and reports once."""

    if list(row) != EXPECTED_COLUMNS:
        raise ValueError("child consolidation row columns changed")
    if row.get("role") != "coordinator_nonroot":
        raise ValueError("child consolidation row has a non-child role")
    if row.get("resource_family") not in SUPPORTED_FAMILIES:
        raise ValueError("child consolidation row has an unsupported resource family")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 4:
        raise ValueError("child consolidation row lacks its live message context")
    if messages[0].get("role") != "system" or messages[0].get(
        "content"
    ) != LEAF_REPORTER_CONTRACT:
        raise ValueError("child consolidation row lacks the leaf reporter contract")
    assistants = [message for message in messages if message.get("role") == "assistant"]
    if len(assistants) != 1:
        raise ValueError("child consolidation row must contain one assistant action")
    tool_calls = assistants[0].get("tool_calls") or []
    if len(tool_calls) != 1 or tool_calls[0].get("name") != "ipython":
        raise ValueError("child consolidation row lacks one IPython action")
    try:
        arguments = json.loads(tool_calls[0]["arguments"])
        code = arguments["code"]
        tree = ast.parse(code)
    except (KeyError, TypeError, json.JSONDecodeError, SyntaxError) as exc:
        raise ValueError("child consolidation row has invalid IPython code") from exc
    if not isinstance(code, str):
        raise ValueError("child consolidation row has non-string IPython code")
    if not any(isinstance(node, ast.Name) and node.id == "INLINE_EVIDENCE" for node in ast.walk(tree)):
        raise ValueError("child consolidation target does not use INLINE_EVIDENCE")
    if any(
        isinstance(node, ast.Call) and _dotted_name(node.func) in {"open", "Path"}
        for node in ast.walk(tree)
    ):
        raise ValueError("child consolidation target reads a runtime path")
    sends = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and _dotted_name(node.value.func) == "agent_message.send"
    ]
    if len(sends) != 1:
        raise ValueError("child consolidation target must send exactly once")
    send = sends[0]
    keywords = {keyword.arg: keyword.value for keyword in send.keywords if keyword.arg}
    receiver = keywords.get("receiver_role")
    if not (
        isinstance(receiver, ast.Constant)
        and receiver.value == "parent"
        and len(send.args) == 1
        and isinstance(send.args[0], ast.Call)
        and _dotted_name(send.args[0].func) == "str"
        and len(send.args[0].args) == 1
        and isinstance(send.args[0].args[0], ast.Name)
        and send.args[0].args[0].id == "result"
    ):
        raise ValueError("child consolidation target does not report str(result) to parent")
    expected = str(row.get("expected_result"))
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else []
        if not any(isinstance(target, ast.Name) and target.id == "result" for target in targets):
            continue
        try:
            literal = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        if str(literal) == expected:
            raise ValueError("child consolidation target embeds its expected result")


def proportional_interleave(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Spread every source deterministically across the complete corpus."""

    scheduled = []
    for group_index, group in enumerate(groups):
        if not group:
            raise ValueError("child consolidation source is empty")
        scheduled.extend(
            (row_index / len(group), group_index, row_index, row)
            for row_index, row in enumerate(group)
        )
    scheduled.sort(key=lambda item: item[:3])
    return [item[3] for item in scheduled]


def load_corpus(path: Path, expected_objective: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import Dataset

    manifest_path = path / "MANIFEST.json"
    parquet_path = path / "train.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("objective") != expected_objective:
        raise ValueError(f"unexpected source corpus contract: {path}")
    if sha256_file(parquet_path) != (manifest.get("dataset") or {}).get("sha256"):
        raise ValueError(f"source corpus checksum mismatch: {path}")
    dataset = Dataset.from_parquet(str(parquet_path))
    if dataset.column_names != EXPECTED_COLUMNS:
        raise ValueError(f"source corpus columns changed: {path}")
    rows = [dict(row) for row in dataset]
    for row in rows:
        audit_answer_free_row(row)
    return rows, {
        "path": str(path),
        "manifest_sha256": sha256_file(manifest_path),
        "parquet_sha256": sha256_file(parquet_path),
        "rows": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced-live-corpus", type=Path, required=True)
    parser.add_argument("--production-reporting-corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite corpus: {args.output_dir}")

    balanced, balanced_source = load_corpus(
        args.balanced_live_corpus,
        "balanced_answer_free_compute_in_exact_live_child_context",
    )
    production, production_source = load_corpus(
        args.production_reporting_corpus,
        "scaffolded_compute_report_curriculum",
    )
    rows = proportional_interleave([balanced, production])
    family_counts = Counter(row["resource_family"] for row in rows)
    if set(family_counts) != SUPPORTED_FAMILIES:
        raise ValueError("mixed corpus does not cover every resource family")

    from datasets import Dataset

    args.output_dir.mkdir(parents=True)
    parquet_path = args.output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet_path))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "objective": "gentle_balanced_compute_with_production_reporting_retention",
        "row_count": len(rows),
        "unique_task_count": len({row["task_key"] for row in rows}),
        "resource_family_counts": dict(sorted(family_counts.items())),
        "context_contract": {
            "answer_free_targets": True,
            "inline_evidence_targets": True,
            "leaf_reporter_contract": True,
            "one_parent_send": True,
            "production_reporting_retention": True,
        },
        "sources": {
            "balanced_live_compute": balanced_source,
            "production_reporting": production_source,
        },
        "dataset": {"path": parquet_path.name, "sha256": sha256_file(parquet_path)},
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} audited mixed child consolidation rows")


if __name__ == "__main__":
    main()
