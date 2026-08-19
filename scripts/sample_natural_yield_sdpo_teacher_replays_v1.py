"""Compare unconditioned and feedback-conditioned continuations on SDPO states."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import httpx

from scripts.validate_prime_agent_sdpo_zero_lr_audit_v1 import _read_jsonl

EXPECTED_STATE_COUNT = 8


class TeacherSamplingFailure(ValueError):
    """The paired teacher sampling audit is incomplete or malformed."""


def _subsequence_starts(values: list[int], needle: list[int]) -> list[int]:
    return [
        start
        for start in range(len(values) - len(needle) + 1)
        if values[start : start + len(needle)] == needle
    ]


def _audit_states(run_dir: Path, expected_state_count: int) -> list[dict[str, Any]]:
    export_dir = run_dir / "token_exports" / "step_1"
    states = []
    for path in sorted(export_dir.glob("rank_*.jsonl")):
        for record in _read_jsonl(path):
            if not any(value is not None and value != 0 for value in record.get("sdpo_weights", [])):
                continue
            replays = record.get("sdpo_teacher_replays")
            if not isinstance(replays, list) or len(replays) != 1:
                raise TeacherSamplingFailure("each active export must expose one teacher replay")
            replay = replays[0]
            conditioned = replay.get("prefix_ids")
            completion = replay.get("completion_ids")
            student_positions = replay.get("student_positions")
            target_offsets = replay.get("target_offsets")
            token_ids = record.get("token_ids")
            values = (conditioned, completion, student_positions, target_offsets, token_ids)
            if not all(
                isinstance(value, list)
                and value
                and all(isinstance(token, int) for token in value)
                for value in values
            ):
                raise TeacherSamplingFailure("teacher replay has invalid token routing")
            starts = _subsequence_starts(token_ids, completion)
            if len(starts) != 1:
                raise TeacherSamplingFailure(
                    f"failed completion must occur exactly once in student tokens, found {starts}"
                )
            completion_start = starts[0]
            expected_positions = [completion_start + offset for offset in target_offsets]
            if student_positions != expected_positions:
                raise TeacherSamplingFailure("teacher target offsets do not align with student completion")
            unconditioned = token_ids[:completion_start]
            if not unconditioned:
                raise TeacherSamplingFailure("student post-spawn prefix is empty")
            states.append(
                {
                    "rank": record.get("rank"),
                    "export_sequence_idx": record.get("export_sequence_idx"),
                    "unconditioned_prefix_ids": unconditioned,
                    "conditioned_prefix_ids": conditioned,
                    "failed_completion_tokens": len(completion),
                }
            )
    if len(states) != expected_state_count:
        raise TeacherSamplingFailure(
            f"expected {expected_state_count} paired teacher states, found {len(states)}"
        )
    return states


def _visible_response(text: str) -> str:
    # Qwen reasoning may be embedded in raw completion text. Classification is
    # about the assistant response that Prime Agent would hand to its gate.
    for marker in ("</think>", "<|channel|>final<|message|>"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
    return text.replace("<|im_end|>", "").strip()


def _classify_completion(text: str, *, finish_reason: str | None) -> dict[str, Any]:
    lowered = text.lower()
    visible = _visible_response(text)
    visible_lower = visible.lower()
    tool_markers = ("<tool_call>", "<|tool_call|>")
    matched = [marker for marker in tool_markers if marker in lowered]
    premature_patterns = (
        r"^\s*\{",
        r"\bfinal\s+answer\b",
        r"\banswer\s+is\b",
        r"\bresult\s+is\b",
        r"```json",
    )
    waiting_patterns = (
        r"\bwait(?:ing)?\b",
        r"\bawait(?:ing)?\b",
        r"\bpending\b",
        r"\bonce\s+(?:the\s+)?(?:child|reviewer)",
        r"\b(?:child|reviewer).{0,80}\b(?:reply|report|result|message)\b",
        r"\bdelegat(?:ed|ion).{0,80}\b(?:reply|report|result|message)\b",
    )
    premature = any(re.search(pattern, visible_lower) for pattern in premature_patterns)
    waiting = any(re.search(pattern, visible_lower) for pattern in waiting_patterns)
    if matched:
        category = "forbidden_tool_action"
    elif finish_reason != "stop":
        category = "other_invalid_no_tool"
    elif premature:
        category = "premature_finalization"
    elif visible and waiting:
        category = "valid_passive_yield"
    else:
        category = "other_invalid_no_tool"
    return {
        "text": text,
        "visible_response": visible,
        "finish_reason": finish_reason,
        "category": category,
        "uses_tool_or_poll": category == "forbidden_tool_action",
        "matched_markers": matched,
        "complete_no_tool_yield": category == "valid_passive_yield",
    }


def _request_completions(
    client: httpx.Client,
    endpoint: str,
    *,
    model: str,
    prefix_ids: list[int],
    samples: int,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> list[dict[str, Any]]:
    response = client.post(
        endpoint,
        json={
            "model": model,
            "prompt": prefix_ids,
            "n": samples,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "skip_special_tokens": False,
        },
    )
    response.raise_for_status()
    choices = response.json().get("choices")
    if not isinstance(choices, list) or len(choices) != samples:
        raise TeacherSamplingFailure("sampling endpoint returned the wrong number of choices")
    completions = []
    for choice in choices:
        text = choice.get("text")
        finish_reason = choice.get("finish_reason")
        if not isinstance(text, str) or (
            finish_reason is not None and not isinstance(finish_reason, str)
        ):
            raise TeacherSamplingFailure("sampling endpoint returned a malformed completion")
        completions.append(_classify_completion(text, finish_reason=finish_reason))
    return completions


def _counts(completions: list[dict[str, Any]]) -> dict[str, int]:
    categories = (
        "valid_passive_yield",
        "forbidden_tool_action",
        "premature_finalization",
        "other_invalid_no_tool",
    )
    return {
        category: sum(completion["category"] == category for completion in completions)
        for category in categories
    }


def sample(
    run_dir: Path,
    *,
    base_url: str,
    model: str,
    samples_per_arm: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
    expected_state_count: int = EXPECTED_STATE_COUNT,
    seed: int = 20260819,
) -> dict[str, Any]:
    states = _audit_states(run_dir, expected_state_count)
    endpoint = base_url.rstrip("/") + "/completions"
    results = []
    with httpx.Client(timeout=timeout) as client:
        for index, state in enumerate(states):
            arms = {}
            for arm in ("unconditioned", "conditioned"):
                prefix_ids = state[f"{arm}_prefix_ids"]
                completions = _request_completions(
                    client,
                    endpoint,
                    model=model,
                    prefix_ids=prefix_ids,
                    samples=samples_per_arm,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed + index,
                )
                arms[arm] = {
                    "prefix_tokens": len(prefix_ids),
                    "counts": _counts(completions),
                    "completions": completions,
                }
            results.append(
                {
                    "state_index": index,
                    "rank": state["rank"],
                    "export_sequence_idx": state["export_sequence_idx"],
                    "failed_completion_tokens": state["failed_completion_tokens"],
                    "arms": arms,
                }
            )

    total_per_arm = len(states) * samples_per_arm
    arm_counts = {
        arm: {
            category: sum(result["arms"][arm]["counts"][category] for result in results)
            for category in results[0]["arms"][arm]["counts"]
        }
        for arm in ("unconditioned", "conditioned")
    }
    yield_rates = {
        arm: arm_counts[arm]["valid_passive_yield"] / total_per_arm
        for arm in arm_counts
    }
    states_with_distributed_gain = sum(
        result["arms"]["conditioned"]["counts"]["valid_passive_yield"] > 0
        and result["arms"]["conditioned"]["counts"]["forbidden_tool_action"]
        < result["arms"]["unconditioned"]["counts"]["forbidden_tool_action"]
        for result in results
    )
    absolute_yield_gain = yield_rates["conditioned"] - yield_rates["unconditioned"]
    admitted = (
        yield_rates["conditioned"] >= 0.30
        and absolute_yield_gain >= 0.20
        and states_with_distributed_gain >= len(states) / 2
    )
    return {
        "verdict": "pass",
        "mechanism": "natural-yield-paired-teacher-sampling-audit",
        "base_url": base_url,
        "model": model,
        "samples_per_arm_per_state": samples_per_arm,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "results": results,
        "summary": {
            "states": len(states),
            "total_samples_per_arm": total_per_arm,
            "arm_counts": arm_counts,
            "yield_rates": yield_rates,
            "conditioned_absolute_yield_gain": absolute_yield_gain,
            "states_with_conditioned_yield_and_forbidden_tool_reduction": states_with_distributed_gain,
            "required_distributed_gain_states": len(states) // 2,
            "behavioral_teacher_admitted": admitted,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples-per-arm", type=int, default=8)
    parser.add_argument("--expected-state-count", type=int, default=EXPECTED_STATE_COUNT)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.samples_per_arm < 8
        or args.expected_state_count < EXPECTED_STATE_COUNT
        or args.max_tokens <= 0
        or args.temperature <= 0
        or args.timeout <= 0
    ):
        parser.error("audit requires >=8 states, >=8 samples per arm, and positive decoding limits")
    report = sample(
        args.run_dir,
        base_url=args.base_url,
        model=args.model,
        samples_per_arm=args.samples_per_arm,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        expected_state_count=args.expected_state_count,
        seed=args.seed,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
