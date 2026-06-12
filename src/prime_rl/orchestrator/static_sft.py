from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import verifiers as vf

from prime_rl.configs.algorithm import DatasetConfig
from prime_rl.utils.chat_template import normalize_messages


def load_static_sft_rows(config: DatasetConfig, *, seed: int | None = None) -> list[dict]:
    if config.data_dir is not None:
        rows = _load_jsonl_rows(config.data_dir)
    else:
        from datasets import load_dataset

        assert config.name is not None
        dataset = load_dataset(config.name, config.subset, split=config.split)
        rows = [dict(row) for row in dataset]

    if config.max_examples is not None:
        rows = rows[: config.max_examples]

    examples: list[dict] = []
    for idx, row in enumerate(rows):
        example = dict(row)
        example.setdefault("example_id", idx)
        examples.append(example)
    return examples


def _load_jsonl_rows(data_dir: Path) -> list[dict]:
    data_dir = data_dir.expanduser()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"dataset.data_dir must be a directory: {data_dir}")
    files = sorted(data_dir.glob("*.jsonl"), key=lambda path: path.name)
    if not files:
        raise FileNotFoundError(f"dataset.data_dir contains no .jsonl files: {data_dir}")

    rows: list[dict] = []
    for path in files:
        with path.open() as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise TypeError(f"{path}:{line_num} must contain a JSON object")
                rows.append(row)
    return rows


def static_sft_rollout(row: dict, config: DatasetConfig) -> vf.RolloutOutput:
    messages = _row_messages(row, config)
    tools = _row_tools(row, config)
    trajectory, max_turns_reached = _trajectory_from_messages(messages, config.max_turns)

    return vf.RolloutOutput(
        example_id=row.get("example_id", -1),
        prompt=trajectory[0]["prompt"],
        trajectory=trajectory,
        sampling_args={"temperature": 1.0},
        error=None,
        completion=trajectory[-1]["completion"] if trajectory else None,
        reward=1.0,
        advantage=None,
        is_completed=True,
        is_truncated=max_turns_reached,
        timing=vf.RolloutTiming(),
        metrics={},
        stop_condition="max_turns_reached" if max_turns_reached else "replayed_messages",
        info=row.get("info", {}),
        token_usage={
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "final_input_tokens": 0.0,
            "final_output_tokens": 0.0,
        },
        tool_defs=tools,
    )


def _row_messages(row: dict, config: DatasetConfig) -> list[dict[str, Any]]:
    if config.messages_column in row and row[config.messages_column] is not None:
        return normalize_messages(_maybe_json(row[config.messages_column]), default_role="assistant")

    if config.prompt_column not in row or config.completion_column not in row:
        raise ValueError(
            "Static SFT rows must have either "
            f"'{config.messages_column}' or both '{config.prompt_column}' and '{config.completion_column}'."
        )

    prompt = normalize_messages(_maybe_json(row[config.prompt_column]), default_role="user")
    completion = normalize_messages(_maybe_json(row[config.completion_column]), default_role="assistant")
    return prompt + completion


def _row_tools(row: dict, config: DatasetConfig) -> list[dict[str, Any]]:
    raw = row.get(config.tools_column, row.get("tool_defs"))
    if raw is None:
        return []
    parsed = _maybe_json(raw)
    if not isinstance(parsed, list):
        raise TypeError(f"Static SFT tools must be a list, got {type(parsed).__name__}")
    return parsed


def _trajectory_from_messages(messages: list[dict[str, Any]], max_turns: int) -> tuple[list[vf.TrajectoryStep], bool]:
    assistant_indices = [idx for idx, message in enumerate(messages) if message.get("role") == "assistant"]
    max_turns_reached = max_turns > 0 and max_turns < len(assistant_indices)
    if max_turns > 0:
        assistant_indices = assistant_indices[:max_turns]

    steps: list[vf.TrajectoryStep] = []
    for turn, idx in enumerate(assistant_indices):
        message = messages[idx]
        prompt = [dict(m) for m in messages[:idx]]
        completion = [dict(message)]
        steps.append(
            vf.TrajectoryStep(
                prompt=prompt,
                completion=completion,
                response=None,
                tokens=None,
                reward=None,
                advantage=None,
                is_truncated=max_turns_reached and turn == len(assistant_indices) - 1,
                trajectory_id=str(len(steps)),
                extras={},
            )
        )
    if not steps:
        raise ValueError("Static SFT row contains no assistant messages to train on.")
    return steps, max_turns_reached


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, Iterable) and not isinstance(value, dict):
        return list(value)
    return value
