#!/usr/bin/env python3
"""Build balanced answer-free child compute SFT in the exact live proxy context."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_q35_2b_recursive_return_sft_v2 import (
    RECURSIVE_HEADER,
    SUPPORTED_FAMILIES,
    compute_resource,
    load_generator,
    select_balanced_episodes,
    sha256_file,
)
from dual_policy_openai_proxy_v1 import LEAF_REPORTER_CONTRACT, leaf_compute_report_code

SCHEMA_VERSION = "q35-2b-balanced-live-child-compute/v1"


def live_generic_prefix(template_row: dict[str, Any]) -> str:
    """Extract the invariant Prime Agent prompt before its dynamic child contract."""

    messages = template_row["messages"]
    if [message.get("role") for message in messages[:3]] != ["system", "user", "user"]:
        raise ValueError("template row does not have the live system/user/user prefix")
    if messages[0].get("content") != LEAF_REPORTER_CONTRACT:
        raise ValueError("template row does not use the current leaf reporter contract")
    marker = f"\n\n{RECURSIVE_HEADER}"
    generic, separator, _ = messages[1]["content"].partition(marker)
    if not separator or not generic.strip():
        raise ValueError("template row lacks a dynamic recursive child contract")
    return generic


def _recursive_contract(path: str, operation: str, content: str) -> str:
    return (
        f"{RECURSIVE_HEADER}\n"
        "session_role=coordinator\n"
        "is_root=false\n"
        "can_delegate=false\n"
        "can_finalize_user=false\n"
        "return_contract=exactly_one_parent_report\n"
        "You are the non-root coordinator for this bounded subproblem. Do not call rlm or "
        "create descendants. Compute from the inline evidence, send exactly one result to "
        "your parent with agent_message.send, then stop.\n"
        f"Evidence label: {path}\n"
        f"Required review: {operation}\n"
        "Evidence contents:\n"
        f"{content}"
    )


def _parent_task(child: dict[str, Any]) -> str:
    path = child["resource_path"]
    operation = child["operation"]
    return (
        f"[task from parent]\n\nYou are {child['name']}, a non-root coordinator. "
        f"Only this session owns {path}. You cannot finalize the user's request and must "
        "not create descendants. The private evidence is the inline card supplied in your "
        "system message; its evidence label is not a runtime path, so do not use Path, open, "
        f"or filesystem search. Process the supplied private evidence and {operation}. In "
        "your first IPython call send only the resulting integer as a string to "
        "receiver_role='parent'. Send exactly one report to your parent, then stop; do not "
        "return the result only as prose."
    )


def balanced_live_row(
    episode: dict[str, Any], *, generic_prefix: str, tools: str
) -> dict[str, Any]:
    oracle = episode["oracle"]
    child = oracle["children"][0]
    path = child["resource_path"]
    ownership = oracle["resource_ownership"][path]
    family = ownership["family"]
    operation = child["operation"]
    content = oracle["private_resources"][path]
    result = compute_resource(family, content, operation)
    if str(result) != str(child["expected_result"]):
        raise ValueError(f"computed result mismatch for {episode['episode_id']}")
    code = leaf_compute_report_code(operation)
    tree = ast.parse(code)
    integer_constants = {
        str(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    }
    if str(result) in integer_constants:
        raise ValueError("canonical compute target contains the episode answer")
    call_id = "balanced-live-child-" + hashlib.sha256(
        episode["episode_id"].encode()
    ).hexdigest()[:16]
    messages = [
        {"role": "system", "content": LEAF_REPORTER_CONTRACT},
        {
            "role": "user",
            "content": f"{generic_prefix}\n\n{_recursive_contract(path, operation, content)}",
        },
        {"role": "user", "content": _parent_task(child)},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "name": "ipython",
                    "arguments": json.dumps({"code": code}, separators=(",", ":")),
                }
            ],
        },
        {"role": "tool", "content": "message queued", "tool_call_id": call_id},
    ]
    return {
        "messages": messages,
        "tools": tools,
        "axis": "balanced_live_child_compute",
        "phase": "e0c4_recursive_coordinator_return",
        "task_key": episode["episode_id"],
        "trace_id": None,
        "role": "coordinator_nonroot",
        "objective": "compute_from_inline_evidence_then_one_parent_send",
        "expected_result": str(result),
        "resource_family": family,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--template-corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--examples-per-family", type=int, default=32)
    parser.add_argument("--master-seed", type=int, default=20260816)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite corpus: {args.output_dir}")
    if args.examples_per_family < 1:
        raise ValueError("examples per family must be positive")

    from datasets import Dataset

    template_manifest_path = args.template_corpus / "MANIFEST.json"
    template_parquet = args.template_corpus / "train.parquet"
    template_manifest = json.loads(template_manifest_path.read_text(encoding="utf-8"))
    if (
        template_manifest.get("status") != "complete"
        or sha256_file(template_parquet)
        != (template_manifest.get("dataset") or {}).get("sha256")
    ):
        raise ValueError("template corpus is incomplete or has a checksum mismatch")
    template = Dataset.from_parquet(str(template_parquet))[0]
    generic_prefix = live_generic_prefix(template)
    generator = load_generator(args.generator)
    episodes = select_balanced_episodes(
        generator,
        start_index=args.start_index,
        examples_per_family=args.examples_per_family,
        master_seed=args.master_seed,
    )
    rows = [
        balanced_live_row(
            episode, generic_prefix=generic_prefix, tools=template["tools"]
        )
        for episode in episodes
    ]
    family_counts = Counter(row["resource_family"] for row in rows)
    expected_counts = {family: args.examples_per_family for family in SUPPORTED_FAMILIES}
    if dict(family_counts) != expected_counts:
        raise ValueError("balanced generator did not cover every operation family exactly")

    args.output_dir.mkdir(parents=True)
    parquet = args.output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    operations = Counter(
        episode["oracle"]["children"][0]["operation"] for episode in episodes
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "objective": "balanced_answer_free_compute_in_exact_live_child_context",
        "row_count": len(rows),
        "unique_task_count": len({row["task_key"] for row in rows}),
        "resource_family_counts": dict(sorted(family_counts.items())),
        "operation_counts": dict(sorted(operations.items())),
        "task_index_range": {
            "requested_start": args.start_index,
            "selected_min": min(episode["index"] for episode in episodes),
            "selected_max": max(episode["index"] for episode in episodes),
        },
        "context_contract": {
            "roles": ["system", "user", "user", "assistant", "tool"],
            "leaf_reporter_contract": True,
            "exact_live_generic_prefix": True,
            "inline_evidence_target": True,
            "answer_free_target": True,
            "raw_evidence_visible": True,
            "replay_rows": 0,
        },
        "sources": {
            "generator": {
                "path": str(args.generator),
                "sha256": sha256_file(args.generator),
            },
            "template_corpus": {
                "path": str(args.template_corpus),
                "manifest_sha256": sha256_file(template_manifest_path),
                "parquet_sha256": sha256_file(template_parquet),
            },
        },
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} unique balanced live child compute rows")


if __name__ == "__main__":
    main()
