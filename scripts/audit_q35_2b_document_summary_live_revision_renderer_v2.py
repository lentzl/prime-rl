#!/usr/bin/env python3
"""Audit live revision history, Qwen rendering, and per-token SFT loss masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from export_q35_2b_document_summary_live_revision_sft_v2 import export
from renderers.configs import Qwen35RendererConfig
from renderers.qwen35 import Qwen35Renderer
from transformers import AutoTokenizer

from prime_rl.trainer.sft.data import SFTDataset, _drop_null_fields
from prime_rl.utils.chat_template import deserialize_tool_calls, normalize_messages


def _sha256_ids(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            part.get("text", "") for part in value if isinstance(part, dict)
        )
    return ""


def _trace_messages(path: Path) -> list[dict[str, Any]]:
    candidates: list[list[dict[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            for trace in json.loads(line).get("traces") or []:
                if trace.get("task", {}).get("type") != "DocumentSummaryTextTask":
                    continue
                nodes = trace.get("nodes") or []
                if len(nodes) == 5:
                    candidates.append([node["message"] for node in nodes])
    if len(candidates) != 1:
        raise ValueError(f"expected one five-message text trace, found {len(candidates)}")
    return candidates[0]


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


def _audit_mode(
    *,
    row: dict[str, Any],
    live_messages: list[dict[str, Any]],
    tokenizer: Any,
    enable_thinking: bool,
) -> dict[str, Any]:
    renderer = Qwen35Renderer(
        tokenizer, Qwen35RendererConfig(enable_thinking=enable_thinking)
    )
    tools = _renderer_tools(row["tools"])
    messages = _renderer_messages(row["messages"])
    full = renderer.render(messages, tools=tools)
    generation_prompt = renderer.render(
        messages[:-1], tools=tools, add_generation_prompt=True
    )
    prompt_length = len(generation_prompt.token_ids)
    shared_prefix = 0
    for prompt_id, full_id in zip(
        generation_prompt.token_ids, full.token_ids, strict=False
    ):
        if prompt_id != full_id:
            break
        shared_prefix += 1
    if shared_prefix == 0:
        raise ValueError("authored correction has no generation-prompt prefix")

    live_history = [
        messages[0],
        messages[1],
        {
            "role": "assistant",
            "content": _content_text(live_messages[2].get("content")),
            "reasoning_content": live_messages[2].get("reasoning_content"),
            "tool_calls": live_messages[2].get("tool_calls") or [],
        },
        messages[3],
    ]
    live_prompt = renderer.render(
        live_history, tools=tools, add_generation_prompt=True
    )
    live_history_token_equivalent = (
        live_prompt.token_ids == generation_prompt.token_ids
    )

    dataset = SFTDataset(
        Dataset.from_list([row]),
        renderer,
        shuffle=False,
        seq_len=16384,
    )
    sample = next(iter(dataset))
    if sample["target_ids"] != full.token_ids[1:]:
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
            full.token_ids[shared_prefix:],
            full.sampled_mask[shared_prefix:],
            strict=True,
        )
        if sampled
    ]
    if trainable_ids != expected_completion:
        raise ValueError("loss mask does not select the exact live completion suffix")

    shifted_indices = full.message_indices[1:]
    if any(
        trainable
        for index, trainable in zip(
            shifted_indices, sample["loss_mask"], strict=True
        )
        if index == 2
    ):
        raise ValueError("prior assistant draft contributes to SFT loss")
    if len(sample["input_ids"]) >= 16384:
        raise ValueError("live revision sample reached the context limit")
    return {
        "enable_thinking": enable_thinking,
        "full_tokens": len(full.token_ids),
        "generation_prompt_tokens": prompt_length,
        "generation_prompt_shared_prefix_tokens": shared_prefix,
        "generation_prompt_is_full_prefix": shared_prefix == prompt_length,
        "generation_prompt_divergent_tail_tokens": prompt_length - shared_prefix,
        "trainable_completion_tokens": len(trainable_ids),
        "full_token_ids_sha256": _sha256_ids(full.token_ids),
        "generation_prompt_token_ids_sha256": _sha256_ids(
            generation_prompt.token_ids
        ),
        "trainable_token_ids_sha256": _sha256_ids(trainable_ids),
        "live_history_token_equivalent": live_history_token_equivalent,
        "live_generation_prompt_token_ids_sha256": _sha256_ids(
            live_prompt.token_ids
        ),
        "prior_assistant_trainable_tokens": 0,
        "truncated": False,
        "completion_boundary_alignment": "renderer_common_prefix_v1",
        "decoded_trainable_prefix": tokenizer.decode(trainable_ids[:8]),
    }


def audit(*, trace: Path, tokenizer_path: Path) -> dict[str, Any]:
    if torch.cuda.is_initialized():
        raise RuntimeError("renderer audit must begin before CUDA initialization")
    live_messages = _trace_messages(trace)
    with tempfile.TemporaryDirectory(prefix="summary-live-revision-audit-") as tmp:
        dataset_dir = Path(tmp) / "dataset"
        manifest = export(traces=[trace], output_dir=dataset_dir)
        rows = Dataset.from_parquet(str(dataset_dir / "train.parquet"))
        row = next(
            row
            for row in rows
            if row["family"] == "summary_live_revision_exceptions"
        )
        messages = row["messages"]
        if _content_text(live_messages[1].get("content")).strip() != messages[1][
            "content"
        ].strip():
            raise ValueError("task prompt differs from the live trace")
        if _content_text(live_messages[2].get("content")).strip() != messages[2][
            "content"
        ].strip():
            raise ValueError("prior draft differs from the live trace")
        if _content_text(live_messages[3].get("content")) != messages[3]["content"]:
            raise ValueError("revision feedback differs from the live trace")

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, trust_remote_code=False
        )
        modes = [
            _audit_mode(
                row=row,
                live_messages=live_messages,
                tokenizer=tokenizer,
                enable_thinking=enable_thinking,
            )
            for enable_thinking in (False, True)
        ]
        selected_mode = next(mode for mode in modes if mode["enable_thinking"])
        if not selected_mode["live_history_token_equivalent"]:
            raise ValueError("thinking-enabled history differs from the live trace")
    if torch.cuda.is_initialized():
        raise RuntimeError("renderer audit initialized CUDA")
    return {
        "schema_version": "qwen35-2b-document-summary-live-revision-renderer-audit/v2",
        "status": "complete",
        "trace": str(trace.resolve()),
        "tokenizer": str(tokenizer_path.resolve()),
        "dataset_schema_version": manifest["schema_version"],
        "role_sequence": [message["role"] for message in messages],
        "live_task_prompt_equal": True,
        "live_prior_draft_equal": True,
        "live_revision_feedback_equal": True,
        "modes": modes,
        "selected_enable_thinking": True,
        "selection_reason": (
            "thinking-enabled rendering is token-equivalent to the live second-turn "
            "history and masks the generation-prefilled prefix"
        ),
        "cuda_initialized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(trace=args.trace, tokenizer_path=args.tokenizer),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
