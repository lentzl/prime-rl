#!/usr/bin/env python3
"""Build a small execution-grounded JSON-max root-coordinator curriculum."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "q35-2b-json-max-coordinator-sft/v1"
OBJECTIVE = "root_json_max_direct_compute_and_one_child_lifecycle"
PHASES = ("direct_compute", "composed_spawn", "composed_wait", "composed_complete")
REQUIRED_SYSTEM_PROMPT = (
    "Coordinate through Prime Agent's persistent IPython kernel. Solve directly when the "
    "coordinator owns the work; delegate only explicitly child-owned resources. Preserve "
    "local state and child handles, spawn independent children before waiting, yield instead "
    "of polling, and treat visible child messages as the completion channel. Never inspect "
    "child-owned resources before an explicit failure and reclaim. Verify child results when "
    "coordinator-owned evidence exists. Return exactly the requested JSON."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_generator(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_json_max_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import procedural generator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _message(
    role: str,
    content: str,
    *,
    reasoning_content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_call_id: str | None = None,
) -> dict[str, Any]:
    return {
        "role": role,
        "content": content,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
        "tool_call_id": tool_call_id,
    }


def _tool_call(call_id: str, code: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "ipython",
            "arguments": json.dumps({"code": code}, separators=(",", ":")),
        },
    }


def _read_episodes(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or len({row["episode_id"] for row in rows}) != len(rows):
        raise ValueError(f"episode source is empty or duplicated: {path}")
    if any(row.get("split") != "train_gen" for row in rows):
        raise ValueError(f"coordinator SFT may consume TRAIN episodes only: {path}")
    return rows


def _wire_message(message: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in message.items() if value is not None}
    content = result.get("content")
    if isinstance(content, list):
        if not all(
            isinstance(part, dict) and part.get("type") == "text" for part in content
        ):
            raise ValueError("runtime source contains non-text content")
        result["content"] = "".join(str(part.get("text", "")) for part in content)
    return result


def load_runtime_surface(trace_path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Extract and audit the exact active root prompt and IPython tool schema."""
    with trace_path.open(encoding="utf-8") as handle:
        envelope = json.loads(next(line for line in handle if line.strip()))
    trace = (envelope.get("traces") or [None])[0]
    if not isinstance(trace, dict):
        raise ValueError("runtime trace lacks a trace")
    nodes = trace.get("nodes") or []
    roots = [
        node
        for node in nodes
        if node.get("parent") is None
        and node.get("sampled") is False
        and (node.get("message") or {}).get("role") == "user"
    ]
    if not roots:
        raise ValueError("runtime trace lacks its unsampled root instruction")
    runtime = _wire_message(roots[0]["message"])
    content = runtime.get("content")
    if not isinstance(content, str):
        raise ValueError("runtime root instruction is not text")
    required_fragments = (
        "Recursive agent depth: 0",
        "await rlm('sub-task')",
        "returns immediately after task admission",
        "Children reply explicitly with `await agent_message.send",
        "Spawn independent children in separate calls and end your turn instead of awaiting completion.",
        REQUIRED_SYSTEM_PROMPT,
    )
    missing = [fragment for fragment in required_fragments if fragment not in content]
    if missing:
        raise ValueError(f"active root rendering lacks required instructions: {missing}")
    if "receiver_role=\"root_coordinator\"" in content:
        raise ValueError("active root rendering contains a non-native receiver role")
    tools = trace.get("tools") or []
    if [tool.get("name") for tool in tools if isinstance(tool, dict)] != ["ipython"]:
        raise ValueError("active root surface must expose exactly IPython")
    task_system = ((trace.get("task") or {}).get("data") or {}).get("system_prompt")
    if task_system != REQUIRED_SYSTEM_PROMPT:
        raise ValueError("active task system prompt differs from the audited contract")
    return (
        runtime,
        json.dumps(tools, sort_keys=True, separators=(",", ":")),
        {
            "runtime_trace": str(trace_path.resolve()),
            "runtime_trace_sha256": sha256_file(trace_path),
            "required_fragments_present": True,
            "task_system_prompt_exact": True,
            "tool_names": ["ipython"],
            "native_spawn_api": "await rlm(task, name=child_name)",
            "child_completion_channel": "visible_agent_message",
            "formula": "global_max=max(remote_max,local_max)",
            "conflict_found": False,
        },
    )


def _paired_direct(generator: Any, composed: dict[str, Any], master_seed: int) -> dict[str, Any]:
    direct = generator.generate_curriculum_episode(
        "json_max_direct_raw",
        composed["split"],
        int(composed["index"]),
        master_seed,
    )
    generator.validate_row(direct)
    generator.validate_row(composed)
    if direct["seed"] != composed["seed"]:
        raise ValueError("paired direct/composed seeds differ")
    if direct["oracle"]["final_answer"] != composed["oracle"]["final_answer"]:
        raise ValueError("paired direct/composed answers differ")
    expected_files = (
        composed["public"]["workspace_files"] | composed["oracle"]["private_resources"]
    )
    if direct["public"]["workspace_files"] != expected_files:
        raise ValueError("paired direct surface does not expose the same two shards")
    return direct


def _owned_paths(episode: dict[str, Any]) -> tuple[str, str, str]:
    ownership = episode["oracle"]["resource_ownership"]
    local = [path for path, spec in ownership.items() if spec.get("owner") == "coordinator"]
    remote = [path for path, spec in ownership.items() if str(spec.get("owner", "")).startswith("child:")]
    children = episode["oracle"].get("children") or []
    if len(local) != 1 or len(remote) != 1 or len(children) != 1:
        raise ValueError("JSON-max coordinator row must have one local and one child shard")
    child = children[0]
    if child["resource_path"] != remote[0] or child["name"] not in ownership[remote[0]]["owner"]:
        raise ValueError("child identity and resource ownership differ")
    return remote[0], local[0], child["name"]


def _maxima(episode: dict[str, Any]) -> tuple[int, int, int]:
    remote_path, local_path, _ = _owned_paths(episode)
    remote = json.loads(episode["oracle"]["private_resources"][remote_path])
    local = json.loads(episode["public"]["workspace_files"][local_path])
    if not remote or not local or not all(type(value) is int for value in [*remote.values(), *local.values()]):
        raise ValueError("JSON-max shards must be nonempty integer objects")
    result = (max(remote.values()), max(local.values()))
    final = (result[0], result[1], max(result))
    expected = episode["oracle"]["final_answer"]
    if expected != {"remote_max": final[0], "local_max": final[1], "global_max": final[2]}:
        raise ValueError("computed JSON-max answer differs from oracle")
    return final


def _direct_paths_and_maxima(episode: dict[str, Any]) -> tuple[str, str, int, int, int]:
    files = episode["public"]["workspace_files"]
    remote = [path for path in files if path.endswith("/review.json")]
    local = [path for path in files if path.endswith("/coordinator.json")]
    if len(remote) != 1 or len(local) != 1 or len(files) != 2:
        raise ValueError("direct JSON-max row must expose review.json and coordinator.json")
    remote_values = json.loads(files[remote[0]])
    local_values = json.loads(files[local[0]])
    if not remote_values or not local_values:
        raise ValueError("direct JSON-max shards must be nonempty")
    maxima = (max(remote_values.values()), max(local_values.values()))
    final = (maxima[0], maxima[1], max(maxima))
    if episode["oracle"]["final_answer"] != {
        "remote_max": final[0],
        "local_max": final[1],
        "global_max": final[2],
    }:
        raise ValueError("computed direct JSON-max answer differs from oracle")
    return remote[0], local[0], *final


def direct_row(
    episode: dict[str, Any], *, runtime: dict[str, Any], tools: str
) -> dict[str, Any]:
    remote_path, local_path, remote_max, local_max, global_max = _direct_paths_and_maxima(
        episode
    )
    call_id = "json-max-direct-" + hashlib.sha256(episode["episode_id"].encode()).hexdigest()[:16]
    code = (
        "import json\n"
        "from pathlib import Path\n"
        f"remote_values = json.loads(Path({remote_path!r}).read_text())\n"
        f"local_values = json.loads(Path({local_path!r}).read_text())\n"
        "remote_max = max(remote_values.values())\n"
        "local_max = max(local_values.values())\n"
        "result = {'remote_max': remote_max, 'local_max': local_max, "
        "'global_max': max(remote_max, local_max)}\n"
        "result"
    )
    result = {"remote_max": remote_max, "local_max": local_max, "global_max": global_max}
    return {
        "messages": [
            copy.deepcopy(runtime),
            _message("user", episode["public"]["user_prompt"]),
            _message(
                "assistant",
                "",
                reasoning_content=(
                    "Both shards are coordinator-owned. I will parse each JSON object with Python, "
                    "take the maximum integer value from each, and compute the global maximum."
                ),
                tool_calls=[_tool_call(call_id, code)],
            ),
            _message("tool", repr(result), tool_call_id=call_id),
            _message(
                "assistant",
                json.dumps(result, separators=(",", ":")),
                reasoning_content=(
                    "The two shard maxima were computed from the owned files. I will return the "
                    "requested object with global_max equal to their larger value."
                ),
                tool_calls=[],
            ),
        ],
        "tools": tools,
        "task_key": f"{episode['episode_id']}:direct_compute",
        "trace_id": None,
        "family": "json_max_direct_raw",
        "phase": "direct_compute",
        "role": "coordinator_root",
        "objective": OBJECTIVE,
        "source_trace": None,
    }


def composed_rows(
    episode: dict[str, Any], *, runtime: dict[str, Any], tools: str
) -> list[dict[str, Any]]:
    remote_path, local_path, child_name = _owned_paths(episode)
    remote_max, local_max, global_max = _maxima(episode)
    call_id = "json-max-compose-" + hashlib.sha256(episode["episode_id"].encode()).hexdigest()[:16]
    child_task = (
        f"You are {child_name}, the only owner of the private JSON evidence shard identified as "
        f"{remote_path}. The host supplies that evidence in your child context. Compute the "
        "largest JSON integer value with Python, send exactly that integer to your parent with "
        "await agent_message.send(str(result), receiver_role='parent'), and then stop. Do not "
        "delegate or send more than one report."
    )
    code = (
        "import json\n"
        "from pathlib import Path\n"
        f"child_task = {child_task!r}\n"
        f"child_handle = await rlm(child_task, name={child_name!r})\n"
        f"local_values = json.loads(Path({local_path!r}).read_text())\n"
        "local_max = max(local_values.values())\n"
        "{'child': child_handle.name, 'local_max': local_max}"
    )
    spawn = _message(
        "assistant",
        "",
        reasoning_content=(
            f"Only {child_name} owns the review shard. I will spawn that child first, retain its "
            "handle, and compute the coordinator-owned shard while the review proceeds."
        ),
        tool_calls=[_tool_call(call_id, code)],
    )
    receipt = _message(
        "tool",
        repr({"child": child_name, "local_max": local_max}),
        tool_call_id=call_id,
    )
    wait = _message(
        "assistant",
        f"Retained {child_name}'s admission handle and computed local_max={local_max}; waiting for its explicit report.",
        reasoning_content=(
            "The child is admitted and local work is complete. I must yield without polling or "
            "respawning so the explicit child report can resume this session."
        ),
        tool_calls=[],
    )
    child_message = _message(
        "user",
        (
            f"[from child:{child_name}]\n"
            "Agent-to-agent message received.\n"
            "Source: agent_message\n"
            f"From: {child_name}\n\n"
            f"{remote_max}"
        ),
    )
    final_result = {
        "remote_max": remote_max,
        "local_max": local_max,
        "global_max": global_max,
    }
    final = _message(
        "assistant",
        json.dumps(final_result, separators=(",", ":")),
        reasoning_content=(
            "The explicit child report supplies remote_max and the retained local computation "
            "supplies local_max. I will take their larger value and finalize exactly once."
        ),
        tool_calls=[],
    )
    prefix = [copy.deepcopy(runtime), _message("user", episode["public"]["user_prompt"])]
    phase_messages = {
        "composed_spawn": [*prefix, copy.deepcopy(spawn)],
        "composed_wait": [*prefix, copy.deepcopy(spawn), receipt, wait],
        "composed_complete": [
            *prefix,
            copy.deepcopy(spawn),
            receipt,
            wait,
            child_message,
            final,
        ],
    }
    rows = []
    for phase, messages in phase_messages.items():
        rows.append(
            {
                "messages": messages,
                "tools": tools,
                "task_key": f"{episode['episode_id']}:{phase}",
                "trace_id": None,
                "family": "json_max_two_shard",
                "phase": phase,
                "role": "coordinator_root",
                "objective": OBJECTIVE,
                "source_trace": None,
            }
        )
    return rows


def validate_target_row(row: dict[str, Any]) -> None:
    messages = row["messages"]
    assistants = [message for message in messages if message["role"] == "assistant"]
    if not assistants or any(message.get("reasoning_content") is None for message in assistants):
        raise ValueError("each coordinator target must have grounded assistant reasoning")
    tool_calls = [call for message in assistants for call in (message.get("tool_calls") or [])]
    if row["phase"] == "direct_compute":
        if len(tool_calls) != 1:
            raise ValueError("direct target must have one tool call")
        code = json.loads(tool_calls[0]["function"]["arguments"])["code"]
        if "await rlm(" in code or code.count("Path(") != 2 or code.count("json.loads") != 2:
            raise ValueError("direct target does not parse exactly its two owned shards")
        if messages[-1]["role"] != "assistant" or "global_max" not in messages[-1]["content"]:
            raise ValueError("direct target lacks a final result")
        return
    if len(tool_calls) != 1:
        raise ValueError("composed target must have exactly one spawn/local-compute call")
    code = json.loads(tool_calls[0]["function"]["arguments"])["code"]
    if code.count("await rlm(") != 1 or code.count("agent_message.send") != 1:
        raise ValueError("composed target does not use the native one-child contract")
    if code.index("await rlm(") > code.index("Path("):
        raise ValueError("composed target does not spawn before local computation")
    if code.count("Path(") != 1 or "rlm.list_subagents" in code or "agent_observe" in code:
        raise ValueError("composed target crosses ownership or polls")
    if row["phase"] == "composed_spawn" and len(messages) != 3:
        raise ValueError("spawn target has trailing messages")
    if row["phase"] == "composed_wait" and [message["role"] for message in messages][-2:] != ["tool", "assistant"]:
        raise ValueError("wait target does not terminate at a passive yield")
    if row["phase"] == "composed_complete":
        if "[from child:" not in messages[-2]["content"] or messages[-1]["role"] != "assistant":
            raise ValueError("complete target lacks child-report integration")


def build_new_rows(
    episodes: list[dict[str, Any]],
    *,
    generator: Any,
    runtime: dict[str, Any],
    tools: str,
    master_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        direct = _paired_direct(generator, episode, master_seed)
        rows.append(direct_row(direct, runtime=runtime, tools=tools))
        rows.extend(composed_rows(episode, runtime=runtime, tools=tools))
    for row in rows:
        validate_target_row(row)
    if Counter(row["phase"] for row in rows) != Counter({phase: len(episodes) for phase in PHASES}):
        raise ValueError("coordinator curriculum is not phase-balanced")
    return rows


def _load_rehearsal(path: Path, repeats: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import Dataset

    manifest_path = path / "MANIFEST.json"
    parquet_path = path / "train.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(parquet_path) != manifest["dataset"]["sha256"]:
        raise ValueError(f"rehearsal corpus hash differs: {path}")
    unique = Dataset.from_parquet(str(parquet_path)).to_list()
    if not unique or any(row.get("role") != "coordinator" for row in unique):
        raise ValueError(f"rehearsal source is not coordinator-only: {path}")
    rows = [copy.deepcopy(row) for _ in range(repeats) for row in unique]
    return rows, {
        "path": str(path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "parquet_sha256": sha256_file(parquet_path),
        "unique_rows": len(unique),
        "repeats": repeats,
        "rows": len(rows),
        "role": "coordinator",
    }


def _interleave(new_rows: list[dict[str, Any]], rehearsal: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    rehearsal_index = 0
    for index, row in enumerate(new_rows, start=1):
        result.append(row)
        if index % 9 == 0 and rehearsal_index < len(rehearsal):
            result.append(rehearsal[rehearsal_index])
            rehearsal_index += 1
    result.extend(rehearsal[rehearsal_index:])
    return result


def _token_counts(rows: list[dict[str, Any]], renderer: Any) -> dict[str, Any]:
    total: list[int] = []
    masked: list[int] = []
    for row in rows:
        rendered = renderer.render(
            row["messages"],
            tools=json.loads(row["tools"]),
            add_generation_prompt=False,
        )
        total.append(len(rendered.token_ids))
        masked.append(sum(bool(value) for value in rendered.loss_mask))
    if not all(0 < masked_count < total_count <= 8192 for masked_count, total_count in zip(masked, total)):
        raise ValueError("coordinator corpus has invalid masked/total token counts")
    return {
        "rows": len(rows),
        "total_tokens": sum(total),
        "masked_assistant_tokens": sum(masked),
        "minimum_total_tokens": min(total),
        "maximum_total_tokens": max(total),
        "mean_total_tokens": sum(total) / len(total),
        "minimum_masked_assistant_tokens": min(masked),
        "maximum_masked_assistant_tokens": max(masked),
        "mean_masked_assistant_tokens": sum(masked) / len(masked),
    }


def _write_corpus(path: Path, rows: list[dict[str, Any]], renderer: Any) -> dict[str, Any]:
    from datasets import Dataset

    path.mkdir()
    parquet = path / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    roundtrip = Dataset.from_parquet(str(parquet)).to_list()
    if roundtrip != rows:
        raise ValueError(f"parquet round trip differs: {path}")
    tokens = _token_counts(roundtrip, renderer)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "objective": OBJECTIVE,
        "rows": len(rows),
        "role": "coordinator_root",
        "phase_counts": dict(sorted(Counter(row.get("phase", "rehearsal") for row in rows).items())),
        "token_counts": tokens,
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    manifest_path = path / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "manifest_sha256": sha256_file(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--runtime-trace", type=Path, required=True)
    parser.add_argument("--tiny-episodes", type=Path, required=True)
    parser.add_argument("--train-episodes", type=Path, required=True)
    parser.add_argument("--decision-rehearsal", type=Path, required=True)
    parser.add_argument("--fanin-rehearsal", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--master-seed", type=int, default=20260819)
    parser.add_argument("--tiny-pairs", type=int, default=8)
    parser.add_argument("--train-pairs", type=int, default=128)
    parser.add_argument("--decision-repeats", type=int, default=3)
    parser.add_argument("--fanin-repeats", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(f"refusing to overwrite coordinator artifact: {args.output_dir}")

    from renderers import Qwen35Renderer, Qwen35RendererConfig
    from transformers import AutoTokenizer

    runtime, tools, active_path_audit = load_runtime_surface(args.runtime_trace)
    generator = load_generator(args.generator)
    tiny_episodes = _read_episodes(args.tiny_episodes)
    train_episodes = _read_episodes(args.train_episodes)
    if args.tiny_pairs > len(tiny_episodes) or args.train_pairs > len(train_episodes):
        raise ValueError("requested coordinator pairs exceed available TRAIN episodes")
    selected_tiny = tiny_episodes[: args.tiny_pairs]
    selected_train = train_episodes[: args.train_pairs]
    if {row["episode_id"] for row in selected_tiny} & {row["episode_id"] for row in selected_train}:
        raise ValueError("tiny and expanded coordinator episodes overlap")

    tiny_rows = build_new_rows(
        selected_tiny,
        generator=generator,
        runtime=runtime,
        tools=tools,
        master_seed=args.master_seed,
    )
    train_new = build_new_rows(
        selected_train,
        generator=generator,
        runtime=runtime,
        tools=tools,
        master_seed=args.master_seed,
    )
    decision_rows, decision_source = _load_rehearsal(args.decision_rehearsal, args.decision_repeats)
    fanin_rows, fanin_source = _load_rehearsal(args.fanin_rehearsal, args.fanin_repeats)
    rehearsal = [*decision_rows, *fanin_rows]
    full_rows = _interleave(train_new, rehearsal)
    if len(tiny_rows) != 32 or len(train_new) != 512 or len(rehearsal) != 60 or len(full_rows) != 572:
        raise ValueError("coordinator curriculum differs from the bounded 32/512+60 design")
    serialized = json.dumps([*tiny_rows, *full_rows], sort_keys=True)
    if "sealed_" in serialized or "ood_gen" in serialized:
        raise ValueError("coordinator corpus references a sealed split")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=False
    )
    renderer = Qwen35Renderer(tokenizer, Qwen35RendererConfig(enable_thinking=False))
    args.output_dir.mkdir(parents=True)
    corpora = {
        "tiny_fit": _write_corpus(args.output_dir / "tiny_fit", tiny_rows, renderer),
        "train": _write_corpus(args.output_dir / "train", full_rows, renderer),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "objective": OBJECTIVE,
        "base_candidate": "e33",
        "fixed_child": "H176",
        "master_seed": args.master_seed,
        "active_path_audit": active_path_audit,
        "curriculum": {
            "tiny_pairs": len(selected_tiny),
            "tiny_rows": len(tiny_rows),
            "train_pairs": len(selected_train),
            "new_train_rows": len(train_new),
            "root_rehearsal_rows": len(rehearsal),
            "full_rows": len(full_rows),
            "phase_counts": dict(sorted(Counter(row["phase"] for row in train_new).items())),
            "intermediate_reports_are_training_inputs_not_autonomous_successes": True,
        },
        "data_boundary": {
            "train_only": True,
            "sealed_opened": False,
            "worker_return_corpus_used": False,
            "root_coordinator_rehearsal_only": True,
        },
        "sources": {
            "generator": {"path": str(args.generator.resolve()), "sha256": sha256_file(args.generator)},
            "tiny_episodes": {"path": str(args.tiny_episodes.resolve()), "sha256": sha256_file(args.tiny_episodes)},
            "train_episodes": {"path": str(args.train_episodes.resolve()), "sha256": sha256_file(args.train_episodes)},
            "decision_rehearsal": decision_source,
            "fanin_rehearsal": fanin_source,
            "tokenizer": str(args.tokenizer.resolve()),
        },
        "corpora": corpora,
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
