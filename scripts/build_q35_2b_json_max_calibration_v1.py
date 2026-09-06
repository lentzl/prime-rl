#!/usr/bin/env python3
"""Build matched JSON-max calibration splits and answer-free worker SFT corpora."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_q35_2b_balanced_live_child_compute_v1 import (
    live_generic_prefix,
    sha256_file,
)
from build_q35_2b_recursive_return_sft_v2 import compute_resource, load_generator
from dual_policy_openai_proxy_v1 import LEAF_REPORTER_CONTRACT, leaf_compute_report_code

SCHEMA_VERSION = "q35-2b-json-max-calibration/v1"
PRIVATE_EVIDENCE_HEADER = "[private evidence supplied to this reviewer]"


def _message(
    role: str,
    content: str,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_call_id: str | None = None,
) -> dict[str, Any]:
    return {
        "role": role,
        "content": content,
        "reasoning_content": None,
        "tool_calls": tool_calls,
        "tool_call_id": tool_call_id,
    }


def worker_row(
    episode: dict[str, Any], *, generic_prefix: str, tools: str
) -> dict[str, Any]:
    oracle = episode["oracle"]
    if episode["metadata"].get("curriculum_rung") != "json_max_two_shard":
        raise ValueError("worker rows require the composed JSON-max rung")
    if len(oracle["children"]) != 1 or len(oracle["private_resources"]) != 1:
        raise ValueError("composed JSON-max episode must have one private child shard")
    child = oracle["children"][0]
    path = child["resource_path"]
    operation = child["operation"]
    content = oracle["private_resources"][path]
    result = compute_resource("json_max", content, operation)
    if str(result) != str(child["expected_result"]):
        raise ValueError(f"computed result mismatch for {episode['episode_id']}")

    code = leaf_compute_report_code(operation)
    numeric_literals = {
        str(node.value)
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    }
    if str(result) in numeric_literals:
        raise ValueError("worker target contains the episode answer")

    child_prompt = (
        "[task from parent]\n\nReview your assigned private JSON evidence and return "
        "the largest JSON integer value to your parent. Use Python for the computation, "
        "send exactly one integer report, and stop after delivery.\n\n"
        f"{PRIVATE_EVIDENCE_HEADER}\n"
        "This evidence is available in this child context only; the coordinator does not "
        "have a workspace copy. The evidence label is an identifier, not a runtime file "
        "path; process the inline contents below directly.\n"
        f"Evidence label: {path}\n"
        f"Required review: {operation}\n"
        "Evidence contents:\n"
        f"{content}"
    )
    call_id = "json-max-worker-" + hashlib.sha256(
        episode["episode_id"].encode()
    ).hexdigest()[:16]
    return {
        "messages": [
            _message("system", LEAF_REPORTER_CONTRACT),
            _message("user", generic_prefix),
            _message("user", child_prompt),
            _message(
                "assistant",
                "",
                tool_calls=[
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "ipython",
                            "arguments": json.dumps(
                                {"code": code}, separators=(",", ":")
                            ),
                        },
                    }
                ],
            ),
            _message("tool", "message queued", tool_call_id=call_id),
        ],
        "tools": tools,
        "task_key": episode["episode_id"],
        "trace_id": None,
        "family": "json_max",
        "role": "child",
        "objective": "compute_json_max_from_inline_evidence_then_report_once",
        "source_trace": None,
    }


def _paired_episodes(
    generator: Any,
    *,
    split: str,
    start: int,
    count: int,
    master_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    direct_rows = []
    composed_rows = []
    for index in range(start, start + count):
        direct = generator.generate_curriculum_episode(
            "json_max_direct_raw", split, index, master_seed
        )
        composed = generator.generate_curriculum_episode(
            "json_max_two_shard", split, index, master_seed
        )
        generator.validate_row(direct)
        generator.validate_row(composed)
        if direct["seed"] != composed["seed"]:
            raise ValueError("paired JSON-max rows have different seeds")
        if direct["oracle"]["final_answer"] != composed["oracle"]["final_answer"]:
            raise ValueError("paired JSON-max rows have different answers")
        if direct["public"]["workspace_files"] != (
            composed["public"]["workspace_files"]
            | composed["oracle"]["private_resources"]
        ):
            raise ValueError("paired JSON-max rows do not share raw evidence")
        direct_rows.append(direct)
        composed_rows.append(composed)
    return direct_rows, composed_rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    data = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    path.write_bytes(data)
    return {"path": path.name, "rows": len(rows), "sha256": hashlib.sha256(data).hexdigest()}


def _token_counts(rows: list[dict[str, Any]], renderer: Any) -> dict[str, Any]:
    counts = []
    for row in rows:
        rendered = renderer.render(
            row["messages"],
            tools=json.loads(row["tools"]),
            add_generation_prompt=False,
        )
        counts.append(len(rendered.token_ids))
    return {
        "rows": len(counts),
        "minimum": min(counts),
        "maximum": max(counts),
        "mean": sum(counts) / len(counts),
    }


def _write_corpus(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    renderer: Any,
    unique_json_rows: int,
    retention_rows: int,
) -> dict[str, Any]:
    from datasets import Dataset

    path.mkdir()
    parquet = path / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    roundtrip = Dataset.from_parquet(str(parquet)).to_list()
    if roundtrip != rows:
        raise ValueError(f"parquet round trip differs for {path}")
    token_counts = _token_counts(roundtrip, renderer)
    if token_counts["maximum"] > 8192:
        raise ValueError(f"corpus exceeds 8192 tokens: {path}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "rows": len(rows),
        "unique_json_max_rows": unique_json_rows,
        "retention_rows": retention_rows,
        "retention_fraction": retention_rows / len(rows),
        "answer_free_worker_targets": True,
        "token_counts": token_counts,
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    manifest_path = path / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**manifest, "manifest_sha256": sha256_file(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--template-corpus", type=Path, required=True)
    parser.add_argument("--retention-corpus", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--master-seed", type=int, default=20260819)
    parser.add_argument("--tiny-start", type=int, default=4_100_000)
    parser.add_argument("--tiny-count", type=int, default=32)
    parser.add_argument("--train-start", type=int, default=4_101_000)
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--dev-start", type=int, default=0)
    parser.add_argument("--dev-count", type=int, default=128)
    parser.add_argument("--sealed-start", type=int, default=0)
    parser.add_argument("--sealed-count", type=int, default=128)
    parser.add_argument("--retention-repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"refusing to overwrite calibration artifact: {args.output_dir}")
    if min(
        args.tiny_count,
        args.train_count,
        args.dev_count,
        args.sealed_count,
        args.retention_repeats,
    ) < 1:
        raise ValueError("all counts must be positive")

    from datasets import Dataset
    from renderers import Qwen35Renderer, Qwen35RendererConfig
    from transformers import AutoTokenizer

    generator = load_generator(args.generator)
    template_manifest = json.loads(
        (args.template_corpus / "MANIFEST.json").read_text(encoding="utf-8")
    )
    template_parquet = args.template_corpus / "train.parquet"
    if sha256_file(template_parquet) != template_manifest["dataset"]["sha256"]:
        raise ValueError("template corpus hash differs")
    template = Dataset.from_parquet(str(template_parquet))[0]
    generic_prefix = live_generic_prefix(template)
    tools = template["tools"]

    retention_manifest = json.loads(
        (args.retention_corpus / "MANIFEST.json").read_text(encoding="utf-8")
    )
    retention_parquet = args.retention_corpus / "train.parquet"
    if (
        retention_manifest.get("answer_free") is not True
        or sha256_file(retention_parquet) != retention_manifest["dataset"]["sha256"]
    ):
        raise ValueError("retention corpus is not exact answer-free H176 data")
    retention_unique = Dataset.from_parquet(str(retention_parquet)).to_list()
    if len(retention_unique) != 12:
        raise ValueError("H176 retention corpus must contain its exact 12 rows")

    _, tiny_episodes = _paired_episodes(
        generator,
        split="train_gen",
        start=args.tiny_start,
        count=args.tiny_count,
        master_seed=args.master_seed,
    )
    _, train_episodes = _paired_episodes(
        generator,
        split="train_gen",
        start=args.train_start,
        count=args.train_count,
        master_seed=args.master_seed,
    )
    dev_direct, dev_composed = _paired_episodes(
        generator,
        split="valid_gen",
        start=args.dev_start,
        count=args.dev_count,
        master_seed=args.master_seed,
    )
    sealed_direct, sealed_composed = _paired_episodes(
        generator,
        split="ood_gen",
        start=args.sealed_start,
        count=args.sealed_count,
        master_seed=args.master_seed,
    )
    if {
        row["episode_id"] for row in tiny_episodes + train_episodes
    } & {row["episode_id"] for row in dev_composed + sealed_composed}:
        raise ValueError("train and evaluation task identities overlap")

    args.output_dir.mkdir(parents=True)
    episode_dir = args.output_dir / "episodes"
    episode_dir.mkdir()
    episode_files = {
        "tiny_fit": _write_jsonl(episode_dir / "tiny_fit.jsonl", tiny_episodes),
        "train": _write_jsonl(episode_dir / "train.jsonl", train_episodes),
        "dev_direct": _write_jsonl(episode_dir / "dev_direct.jsonl", dev_direct),
        "dev_composed": _write_jsonl(
            episode_dir / "dev_composed.jsonl", dev_composed
        ),
        "sealed_direct": _write_jsonl(
            episode_dir / "sealed_direct.jsonl", sealed_direct
        ),
        "sealed_composed": _write_jsonl(
            episode_dir / "sealed_composed.jsonl", sealed_composed
        ),
    }

    tiny_rows = [
        worker_row(row, generic_prefix=generic_prefix, tools=tools)
        for row in tiny_episodes
    ]
    new_train_rows = [
        worker_row(row, generic_prefix=generic_prefix, tools=tools)
        for row in train_episodes
    ]
    retention_rows = retention_unique * args.retention_repeats
    full_rows = []
    retention_index = 0
    for index, row in enumerate(new_train_rows, start=1):
        full_rows.append(row)
        if index % 9 == 0 and retention_index < len(retention_rows):
            full_rows.append(retention_rows[retention_index])
            retention_index += 1
    full_rows.extend(retention_rows[retention_index:])

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=False
    )
    renderer = Qwen35Renderer(
        tokenizer,
        Qwen35RendererConfig(enable_thinking=False),
    )
    corpora = {
        "tiny_fit": _write_corpus(
            args.output_dir / "tiny_fit",
            tiny_rows,
            renderer=renderer,
            unique_json_rows=len(tiny_rows),
            retention_rows=0,
        ),
        "train": _write_corpus(
            args.output_dir / "train",
            full_rows,
            renderer=renderer,
            unique_json_rows=len(new_train_rows),
            retention_rows=len(retention_rows),
        ),
    }
    top_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "objective": "matched_raw_direct_vs_two_shard_json_max_calibration",
        "master_seed": args.master_seed,
        "composition": "global_max=max(remote_max,local_max)",
        "splits": {
            "tiny_fit": {"split": "train_gen", "start": args.tiny_start, "count": args.tiny_count},
            "train": {"split": "train_gen", "start": args.train_start, "count": args.train_count},
            "dev": {"split": "valid_gen", "start": args.dev_start, "count_per_arm": args.dev_count},
            "sealed": {"split": "ood_gen", "start": args.sealed_start, "count_per_arm": args.sealed_count},
        },
        "case_counts": dict(
            sorted(
                Counter(
                    str(row["index"] % 5)
                    for row in tiny_episodes + train_episodes + dev_composed + sealed_composed
                ).items()
            )
        ),
        "episode_files": episode_files,
        "corpora": corpora,
        "sources": {
            "generator": {"path": str(args.generator), "sha256": sha256_file(args.generator)},
            "template_corpus": {"path": str(args.template_corpus), "sha256": sha256_file(template_parquet)},
            "retention_corpus": {"path": str(args.retention_corpus), "sha256": sha256_file(retention_parquet)},
            "tokenizer": str(args.tokenizer),
        },
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(top_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(top_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
