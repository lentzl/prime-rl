#!/usr/bin/env python3
"""Audit rendered-token exposure for matched episodic-memory SFT and SDFT."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, median


def packed_step_budget(
    lengths: Sequence[int], *, epochs: int, seq_len: int, batch_size: int
) -> dict[str, int | float]:
    """Estimate Prime SFT's greedy packing budget without crossing a row boundary."""
    if not lengths:
        raise ValueError("at least one rendered example is required")
    if max(lengths) > seq_len:
        raise ValueError(
            f"rendered example length {max(lengths)} exceeds SFT seq_len {seq_len}"
        )

    packs = 0
    used = 0
    for length in list(lengths) * epochs:
        if used and used + length > seq_len:
            packs += 1
            used = 0
        used += length
        if used == seq_len:
            packs += 1
            used = 0
    if used:
        packs += 1

    steps = math.ceil(packs / batch_size)
    padded_tokens = steps * batch_size * seq_len
    rendered_tokens = sum(lengths) * epochs
    return {
        "epochs": epochs,
        "packs": packs,
        "steps": steps,
        "rendered_tokens": rendered_tokens,
        "padded_tokens": padded_tokens,
        "packing_utilization": rendered_tokens / padded_tokens,
    }


def percentile(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def normalize_tools(raw_tools: str) -> list[dict]:
    tools = json.loads(raw_tools)
    return [
        tool
        if tool.get("type") == "function" and "function" in tool
        else {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description"),
                "parameters": tool.get("parameters", {}),
            },
        }
        for tool in tools
    ]


def render_lengths(dataset_path: Path, tokenizer_name: str) -> tuple[list[int], list[int]]:
    from renderers import Qwen35RendererConfig
    from renderers.base import build_training_sample, create_renderer, load_tokenizer

    from prime_rl.utils.chat_template import deserialize_tool_calls, normalize_messages

    tokenizer = load_tokenizer(tokenizer_name)
    renderer = create_renderer(
        tokenizer,
        Qwen35RendererConfig(enable_thinking=True),
    )
    lengths: list[int] = []
    trainable_lengths: list[int] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        messages = deserialize_tool_calls(
            normalize_messages(json.loads(row["messages_json"]), default_role="assistant")
        )
        sample = build_training_sample(
            renderer,
            messages,
            tools=normalize_tools(row["tools"]),
            ensure_final_stop=True,
        )
        # Match the one-token causal shift in prime_rl.trainer.sft.data.SFTDataset.
        lengths.append(len(sample.token_ids) - 1)
        trainable_lengths.append(sum(sample.loss_mask[1:]))
    return lengths, trainable_lengths


def summarize_lengths(values: Sequence[int]) -> dict[str, int | float]:
    return {
        "total": sum(values),
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--sft-seq-len", type=int, default=4096)
    parser.add_argument("--sft-batch-size", type=int, default=6)
    parser.add_argument("--sdft-batch-size", type=int, default=24)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    lengths, trainable = render_lengths(args.dataset, args.tokenizer)
    sft = packed_step_budget(
        lengths,
        epochs=args.epochs,
        seq_len=args.sft_seq_len,
        batch_size=args.sft_batch_size,
    )
    report = {
        "dataset": str(args.dataset),
        "examples": len(lengths),
        "rendered_sequence_tokens": summarize_lengths(lengths),
        "sft_trainable_tokens": summarize_lengths(trainable),
        "sft": sft,
        "sdft": {
            "epochs": args.epochs,
            "episodes": len(lengths) * args.epochs,
            "batch_size": args.sdft_batch_size,
            "steps": math.ceil(len(lengths) * args.epochs / args.sdft_batch_size),
            "token_budget": "measure from admitted on-policy samples",
        },
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
