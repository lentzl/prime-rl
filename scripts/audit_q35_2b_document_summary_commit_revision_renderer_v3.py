#!/usr/bin/env python3
"""Audit the scaffold-aligned non-thinking summary revision boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from renderers.configs import Qwen35RendererConfig
from renderers.qwen35 import Qwen35Renderer
from transformers import AutoTokenizer

from export_q35_2b_document_summary_commit_revision_sft_v3 import (
    CHAPTER_ORDER,
    _load_fixture_module,
    _observed_cases,
    export,
)
from prime_rl.trainer.sft.data import SFTDataset, _drop_null_fields
from prime_rl.utils.chat_template import deserialize_tool_calls, normalize_messages


def _sha256_ids(token_ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode()
    ).hexdigest()


def _renderer_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = normalize_messages(messages, default_role="assistant")
    cleaned = [_drop_null_fields(message) for message in normalized]
    cleaned = deserialize_tool_calls(cleaned)
    return [
        {
            key: value
            for key, value in message.items()
            if key not in {"trainable", "mask_generation_prompt"}
        }
        for message in cleaned
    ]


def _renderer_tools(raw_tools: str) -> list[dict[str, Any]]:
    return [
        tool
        if tool.get("type") == "function" and "function" in tool
        else {
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters"),
                **(
                    {} if tool.get("strict") is None else {"strict": tool["strict"]}
                ),
            },
        }
        for tool in json.loads(raw_tools)
    ]


def _strip_history_reasoning(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in message.items()
            if key not in {"reasoning", "reasoning_content", "reasoning_details", "provider_state"}
        }
        if message.get("role") == "assistant"
        else message
        for message in messages
    ]


def _audit_row(
    *,
    row: dict[str, Any],
    observed: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    renderer = Qwen35Renderer(
        tokenizer, Qwen35RendererConfig(enable_thinking=False)
    )
    tools = _renderer_tools(row["tools"])
    messages = _renderer_messages(row["messages"])
    full = renderer.render(messages, tools=tools)
    generation_prompt = renderer.render(
        messages[:-1], tools=tools, add_generation_prompt=True
    )
    prompt_ids = list(generation_prompt.token_ids)
    full_ids = list(full.token_ids)
    shared_prefix = 0
    for prompt_id, full_id in zip(prompt_ids, full_ids, strict=False):
        if prompt_id != full_id:
            break
        shared_prefix += 1
    if shared_prefix != len(prompt_ids):
        raise ValueError("non-thinking generation prompt is not an exact full prefix")

    raw_history = _renderer_messages(observed["raw_history"])
    stripped_history = _strip_history_reasoning(raw_history)
    raw_prompt = renderer.render(
        raw_history, tools=tools, add_generation_prompt=True
    )
    stripped_prompt = renderer.render(
        stripped_history, tools=tools, add_generation_prompt=True
    )
    if stripped_prompt.token_ids != generation_prompt.token_ids:
        raise ValueError("reasoning-stripped live history differs from the SFT prefix")
    if observed["raw_reasoning_present"] and raw_prompt.token_ids == prompt_ids:
        raise ValueError("raw reasoning history unexpectedly equals the commit prefix")

    dataset = SFTDataset(
        Dataset.from_list([row]), renderer, shuffle=False, seq_len=16384
    )
    sample = next(iter(dataset))
    if sample["target_ids"] != full_ids[1:]:
        raise ValueError("SFT causal targets differ from the full rendered conversation")
    trainable_ids = [
        token_id
        for token_id, trainable in zip(
            sample["target_ids"], sample["loss_mask"], strict=True
        )
        if trainable
    ]
    expected_completion = [
        token_id
        for token_id, sampled in zip(
            full_ids[shared_prefix:], full.sampled_mask[shared_prefix:], strict=True
        )
        if sampled
    ]
    if trainable_ids != expected_completion:
        raise ValueError("loss mask does not select the direct commit completion")
    shifted_indices = full.message_indices[1:]
    if any(
        trainable
        for index, trainable in zip(
            shifted_indices, sample["loss_mask"], strict=True
        )
        if index == 2
    ):
        raise ValueError("prior assistant draft contributes to SFT loss")
    if not trainable_ids or len(sample["input_ids"]) >= 16384:
        raise ValueError("invalid commit completion length")
    return {
        "chapter_id": row["family"].removeprefix("summary_commit_revision_"),
        "enable_thinking": False,
        "full_tokens": len(full_ids),
        "generation_prompt_tokens": len(prompt_ids),
        "generation_prompt_shared_prefix_tokens": shared_prefix,
        "generation_prompt_is_full_prefix": True,
        "trainable_completion_tokens": len(trainable_ids),
        "generation_prompt_token_ids_sha256": _sha256_ids(prompt_ids),
        "trainable_token_ids_sha256": _sha256_ids(trainable_ids),
        "raw_history_token_equivalent": raw_prompt.token_ids == prompt_ids,
        "reasoning_stripped_history_token_equivalent": True,
        "prior_assistant_trainable_tokens": 0,
        "truncated": False,
        "completion_boundary_alignment": "renderer_common_prefix_v1",
        "decoded_trainable_prefix": tokenizer.decode(trainable_ids[:8]),
    }


def audit(
    *, traces: list[Path], tokenizer_path: Path, dataset_dir: Path | None = None
) -> dict[str, Any]:
    if torch.cuda.is_initialized():
        raise RuntimeError("renderer audit must begin before CUDA initialization")
    fixture = _load_fixture_module()
    document, _ = fixture.build_fixture()
    chapters = {chapter["id"]: chapter for chapter in document["chapters"]}
    observed, _ = _observed_cases(
        traces=traces, fixture=fixture, chapters=chapters
    )
    temporary = dataset_dir is None
    context = (
        tempfile.TemporaryDirectory(prefix="summary-commit-revision-audit-")
        if temporary
        else nullcontext(None)
    )
    with context as tmp:
        active_dataset_dir = (
            Path(tmp) / "dataset" if temporary else dataset_dir.resolve()
        )
        if temporary:
            manifest = export(traces=traces, output_dir=active_dataset_dir)
        else:
            manifest = json.loads(
                (active_dataset_dir / "MANIFEST.json").read_text(encoding="utf-8")
            )
        rows = Dataset.from_parquet(str(active_dataset_dir / "train.parquet"))
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, trust_remote_code=False
        )
        modes = []
        for chapter_id in CHAPTER_ORDER:
            row = next(
                row
                for row in rows
                if row["family"] == f"summary_commit_revision_{chapter_id}"
            )
            modes.append(
                _audit_row(
                    row=row, observed=observed[chapter_id], tokenizer=tokenizer
                )
            )
        manifest_sha256 = hashlib.sha256(
            (active_dataset_dir / "MANIFEST.json").read_bytes()
        ).hexdigest()
        parquet_sha256 = hashlib.sha256(
            (active_dataset_dir / "train.parquet").read_bytes()
        ).hexdigest()
    if torch.cuda.is_initialized():
        raise RuntimeError("renderer audit initialized CUDA")
    return {
        "schema_version": "qwen35-2b-document-summary-commit-revision-renderer-audit/v3",
        "status": "complete",
        "traces": [str(path.resolve()) for path in traces],
        "tokenizer": str(tokenizer_path.resolve()),
        "dataset_schema_version": manifest["schema_version"],
        "dataset_manifest_sha256": manifest_sha256,
        "dataset_parquet_sha256": parquet_sha256,
        "role_sequence": ["user", "user", "assistant", "user", "assistant"],
        "chapters": modes,
        "selected_enable_thinking": False,
        "historical_assistant_reasoning_stripped": True,
        "selection_reason": (
            "the live commit scaffold disables thinking and removes prior-turn reasoning; "
            "the resulting history is token-identical to each supervised SFT prefix"
        ),
        "cuda_initialized": False,
    }


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
                tokenizer_path=args.tokenizer,
                dataset_dir=args.dataset_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
