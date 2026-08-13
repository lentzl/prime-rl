#!/usr/bin/env python3
"""Generate a synthetic Prime Agent bootstrap for programmatic episodic memory.

The dataset teaches a model to treat a lossless append-only interaction history as
an external computational object. It must retrieve rather than guess when the
current request depends on distant prior events, prefer the append-only history
over stale derived notes, exploit persistent IPython state across requests, and
avoid unnecessary history access on self-contained controls.

This is intentionally a bootstrap corpus, not a claim of final memory mastery.
All examples are synthetic and structurally self-validated. No reasoning_content
is fabricated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    from datasets import Dataset
except Exception:  # pragma: no cover - JSONL remains usable without datasets.
    Dataset = None

DATASET_NAME = "programmatic-episodic-memory-v1"
DEFAULT_SEED = 20260813
HISTORY_PATH = "/workspace/history.log"
NOTES_PATH = "/workspace/notes.txt"

IPYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "ipython",
        "description": (
            "Execute Python in a persistent IPython kernel. Variables, imports, "
            "and derived indexes persist across calls. Files in /workspace may be "
            "read and written from Python."
        ),
        "parameters": {
            "type": "object",
            "required": ["code"],
            "properties": {"code": {"type": "string"}},
        },
    },
}

SYSTEM_PROMPT = """You are operating inside a persistent Prime Agent workspace.

The session's lossless append-only interaction history is available at
/workspace/history.log. It contains prior observations, actions, outcomes,
decisions, corrections, and other durable events. Treat this append-only history
as the source of truth for what actually happened.

Use programmatic retrieval when the current decision depends on earlier events:
search or parse the history with Python, return only the small relevant slice to
your active context, and compute over it when useful. Do not guess distant facts
from memory. Prefer the latest valid event when earlier entries were corrected or
superseded.

Other files in /workspace, including notes or summaries you wrote earlier, are
derived state and may be stale. They are useful indexes/caches, but when they
conflict with the append-only history, verify against history.log.

The IPython kernel persists across requests. Build compact derived indexes when
repeated lookups make that worthwhile, and reuse them instead of rereading the
whole history. Conversely, do not touch history.log when the current request is
self-contained and retrieval has no cognitive value.

Answer the user's actual request concisely after any necessary retrieval.
"""

@dataclass(frozen=True)
class Example:
    messages: list[dict]
    files: dict[str, str]
    metadata: dict


def tool_call(call_id: str, code: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "ipython", "arguments": json.dumps({"code": code})},
    }


def assistant_tool(call_id: str, code: str) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": [tool_call(call_id, code)]}


def tool_result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def kv_line(i: int, kind: str, **fields: object) -> str:
    encoded = " ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in fields.items())
    return f"[{i:04d}] TYPE={kind} {encoded}"


def insert_events(rng: random.Random, horizon: int, events: list[tuple[int, str]]) -> str:
    """Insert keyed event strings at fixed positions among synthetic distractors."""
    by_pos = {pos: text for pos, text in events}
    out = []
    for i in range(1, horizon + 1):
        if i in by_pos:
            out.append(by_pos[i])
        else:
            out.append(
                kv_line(
                    i,
                    "NOISE",
                    domain=rng.choice(["build", "research", "ops", "analysis"]),
                    action=rng.choice(["inspect", "cache", "measure", "review", "sync"]),
                    object=rng.choice(["alpha", "beta", "gamma", "queue", "dataset", "artifact"]),
                    token=f"d{i:04d}-{rng.randrange(1000,9999)}",
                )
            )
    return "\n".join(out) + "\n"


def wording(instance: int, explicit: str, natural: str, compacted: str | None = None) -> tuple[str, str]:
    """Cycle explicit, natural, and context-reset wording without changing the task."""
    mode = instance % (3 if compacted is not None else 2)
    if compacted is not None and mode == 2:
        return compacted, "context_reset"
    if mode == 0:
        return explicit, "explicit_history"
    return natural, "natural"


def base_row(messages: list[dict], files: dict[str, str], metadata: dict) -> dict:
    return {
        "messages_json": json.dumps(messages, ensure_ascii=False),
        "tools": json.dumps([IPYTHON_TOOL], ensure_ascii=False),
        "workspace_files_json": json.dumps(files, ensure_ascii=False),
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }


def latest_state_example(rng: random.Random, split: str, instance: int, horizon: int) -> Example:
    key = rng.choice(["deploy_region", "active_model", "release_channel", "primary_dataset"])
    values = {
        "deploy_region": ["eu-west", "us-east", "eu-central"],
        "active_model": ["model-a", "model-b", "model-c"],
        "release_channel": ["canary", "beta", "stable"],
        "primary_dataset": ["snapshot-17", "snapshot-21", "snapshot-24"],
    }[key]
    p1, p2, p3 = horizon // 6, horizon // 2, horizon - max(3, horizon // 8)
    history = insert_events(
        rng,
        horizon,
        [
            (p1, kv_line(p1, "STATE", key=key, value=values[0], status="active")),
            (p2, kv_line(p2, "STATE", key=key, value=values[1], status="active")),
            (p3, kv_line(p3, "STATE", key=key, value=values[2], status="active")),
        ],
    )
    q, instruction_level = wording(
        instance,
        f"Recover the current value of `{key}` from the session history. Return only the value.",
        f"What is the current `{key}`? Return only the value.",
        f"The active context was compacted. What is the current `{key}`? Return only the value.",
    )
    code = (
        "from pathlib import Path\n"
        f"key = {key!r}\n"
        "matches = []\n"
        f"with Path({HISTORY_PATH!r}).open() as f:\n"
        "    for line in f:\n"
        "        if 'TYPE=STATE' in line and f'key=\"{key}\"' in line and 'status=\"active\"' in line:\n"
        "            matches.append(line.strip())\n"
        "matches[-3:]"
    )
    output = repr([
        kv_line(p1, "STATE", key=key, value=values[0], status="active"),
        kv_line(p2, "STATE", key=key, value=values[1], status="active"),
        kv_line(p3, "STATE", key=key, value=values[2], status="active"),
    ])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
        assistant_tool("mem-latest-1", code),
        tool_result("mem-latest-1", output),
        {"role": "assistant", "content": values[2]},
    ]
    return Example(messages, {HISTORY_PATH: history}, {
        "dataset": DATASET_NAME, "split": split, "family": "latest_state",
        "instance": instance, "horizon_events": horizon, "history_format": "kv", "instruction_level": instruction_level,
        "requires_history": True, "retrieval_policy": "latest_valid", "expected_answer": values[2],
    })


def accepted_requirement_example(rng: random.Random, split: str, instance: int, horizon: int) -> Example:
    key = rng.choice(["api_version", "storage_backend", "serialization", "retry_policy"])
    candidates = {
        "api_version": ["v2", "v3", "v4"],
        "storage_backend": ["sqlite", "postgres", "duckdb"],
        "serialization": ["json", "msgpack", "parquet"],
        "retry_policy": ["fixed", "linear", "exponential"],
    }[key]
    p1, p2, p3 = horizon // 5, horizon // 2, horizon - 5
    history = insert_events(
        rng, horizon,
        [
            (p1, kv_line(p1, "REQUIREMENT", key=key, value=candidates[0], status="proposed")),
            (p2, kv_line(p2, "REQUIREMENT", key=key, value=candidates[1], status="accepted")),
            (p3, kv_line(p3, "REQUIREMENT", key=key, value=candidates[2], status="rejected")),
        ],
    )
    q, instruction_level = wording(
        instance,
        f"What accepted `{key}` should we follow now? Return only the accepted value.",
        f"Which `{key}` should we follow now? Return only the value.",
        f"After context compaction, continue with the accepted `{key}`. Return only its value.",
    )
    code = (
        "from pathlib import Path\n"
        f"key = {key!r}\n"
        "accepted = []\n"
        f"for line in Path({HISTORY_PATH!r}).open():\n"
        "    if 'TYPE=REQUIREMENT' in line and f'key=\"{key}\"' in line and 'status=\"accepted\"' in line:\n"
        "        accepted.append(line.strip())\n"
        "accepted[-1]"
    )
    output = repr(kv_line(p2, "REQUIREMENT", key=key, value=candidates[1], status="accepted"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
        assistant_tool("mem-req-1", code),
        tool_result("mem-req-1", output),
        {"role": "assistant", "content": candidates[1]},
    ]
    return Example(messages, {HISTORY_PATH: history}, {
        "dataset": DATASET_NAME, "split": split, "family": "accepted_requirement",
        "instance": instance, "horizon_events": horizon, "history_format": "kv", "instruction_level": instruction_level,
        "requires_history": True, "retrieval_policy": "accepted_not_latest_surface", "expected_answer": candidates[1],
    })


def successful_attempt_example(rng: random.Random, split: str, instance: int, horizon: int) -> Example:
    task = rng.choice(["parser_fix", "index_build", "migration", "latency_patch"])
    attempts = [f"{task}-a", f"{task}-b", f"{task}-c"]
    p1, p2, p3 = horizon // 4, horizon // 2, horizon - 7
    history = insert_events(
        rng, horizon,
        [
            (p1, kv_line(p1, "ATTEMPT", task=task, candidate=attempts[0], outcome="failed", score=0)),
            (p2, kv_line(p2, "ATTEMPT", task=task, candidate=attempts[1], outcome="succeeded", score=1)),
            (p3, kv_line(p3, "ATTEMPT", task=task, candidate=attempts[2], outcome="failed", score=0)),
        ],
    )
    q, instruction_level = wording(
        instance,
        f"Which `{task}` candidate actually succeeded earlier? Return only its candidate id.",
        f"Which `{task}` candidate should we preserve? Return only its id.",
        f"Resume after compaction: which `{task}` candidate should we preserve? Return only its id.",
    )
    code = (
        "from pathlib import Path\n"
        f"task = {task!r}\n"
        "successes = []\n"
        f"for line in Path({HISTORY_PATH!r}).open():\n"
        "    if 'TYPE=ATTEMPT' in line and f'task=\"{task}\"' in line and 'outcome=\"succeeded\"' in line:\n"
        "        successes.append(line.strip())\n"
        "successes"
    )
    output = repr([kv_line(p2, "ATTEMPT", task=task, candidate=attempts[1], outcome="succeeded", score=1)])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
        assistant_tool("mem-attempt-1", code),
        tool_result("mem-attempt-1", output),
        {"role": "assistant", "content": attempts[1]},
    ]
    return Example(messages, {HISTORY_PATH: history}, {
        "dataset": DATASET_NAME, "split": split, "family": "successful_attempt",
        "instance": instance, "horizon_events": horizon, "history_format": "kv", "instruction_level": instruction_level,
        "requires_history": True, "retrieval_policy": "outcome_filter", "expected_answer": attempts[1],
    })


def correction_aggregate_example(rng: random.Random, split: str, instance: int, horizon: int) -> Example:
    metric = rng.choice(["accepted_rows", "verified_items", "approved_cost", "completed_jobs"])
    base = rng.randint(30, 80)
    deltas = [rng.randint(5, 18), rng.randint(5, 18), rng.randint(5, 18)]
    correction = rng.randint(1, 6)
    p = [horizon // 6, horizon // 3, horizon // 2, horizon * 2 // 3, horizon - 4]
    history = insert_events(
        rng, horizon,
        [
            (p[0], kv_line(p[0], "METRIC_BASE", metric=metric, value=base)),
            (p[1], kv_line(p[1], "METRIC_DELTA", metric=metric, event="e1", delta=deltas[0], status="applied")),
            (p[2], kv_line(p[2], "METRIC_DELTA", metric=metric, event="e2", delta=deltas[1], status="applied")),
            (p[3], kv_line(p[3], "METRIC_DELTA", metric=metric, event="e3", delta=deltas[2], status="applied")),
            (p[4], kv_line(p[4], "METRIC_CORRECTION", metric=metric, event="e2", adjustment=-correction)),
        ],
    )
    expected = base + sum(deltas) - correction
    q, instruction_level = wording(
        instance,
        f"Reconstruct the current `{metric}` value from its base, deltas, and correction events. Return only the integer.",
        f"What is the current `{metric}` total? Return only the integer.",
        f"Context was compacted. Recover the current `{metric}` total and return only the integer.",
    )
    code = (
        "from pathlib import Path\nimport re\n"
        f"metric = {metric!r}\n"
        f"lines = [ln.strip() for ln in Path({HISTORY_PATH!r}).open() if f'metric=\"{{metric}}\"' in ln]\n"
        "base = int(re.search(r'value=(\\d+)', next(x for x in lines if 'TYPE=METRIC_BASE' in x)).group(1))\n"
        "deltas = [int(re.search(r'delta=(-?\\d+)', x).group(1)) for x in lines if 'TYPE=METRIC_DELTA' in x]\n"
        "adjustments = [int(re.search(r'adjustment=(-?\\d+)', x).group(1)) for x in lines if 'TYPE=METRIC_CORRECTION' in x]\n"
        "base + sum(deltas) + sum(adjustments)"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
        assistant_tool("mem-agg-1", code),
        tool_result("mem-agg-1", str(expected)),
        {"role": "assistant", "content": str(expected)},
    ]
    return Example(messages, {HISTORY_PATH: history}, {
        "dataset": DATASET_NAME, "split": split, "family": "correction_aggregate",
        "instance": instance, "horizon_events": horizon, "history_format": "kv", "instruction_level": instruction_level,
        "requires_history": True, "retrieval_policy": "retrieve_then_compute", "expected_answer": expected,
    })


def provenance_conflict_example(rng: random.Random, split: str, instance: int, horizon: int) -> Example:
    claim = rng.choice(["cache_is_safe", "feature_is_enabled", "dataset_is_complete", "latency_is_regressed"])
    sources = [f"S{rng.randint(100,999)}" for _ in range(3)]
    p1, p2, p3 = horizon // 5, horizon // 2, horizon - 6
    history = insert_events(
        rng, horizon,
        [
            (p1, kv_line(p1, "EVIDENCE", claim=claim, source=sources[0], verdict="supports")),
            (p2, kv_line(p2, "EVIDENCE", claim=claim, source=sources[1], verdict="supports")),
            (p3, kv_line(p3, "EVIDENCE", claim=claim, source=sources[2], verdict="contradicts")),
        ],
    )
    expected = f"contradicted by {sources[2]}"
    q, instruction_level = wording(
        instance,
        f"What is the latest evidential status of `{claim}` and which source establishes it? Answer `supported by S...` or `contradicted by S...`.",
        f"Where do we currently stand on `{claim}`? Give the latest verdict and source as `supported by S...` or `contradicted by S...`.",
        f"Resume the research state after compaction: what is the current verdict on `{claim}` and its source?",
    )
    code = (
        "from pathlib import Path\n"
        f"claim = {claim!r}\n"
        f"relevant = [ln.strip() for ln in Path({HISTORY_PATH!r}).open() if 'TYPE=EVIDENCE' in ln and f'claim=\"{{claim}}\"' in ln]\n"
        "relevant[-3:]"
    )
    output = repr([
        kv_line(p1, "EVIDENCE", claim=claim, source=sources[0], verdict="supports"),
        kv_line(p2, "EVIDENCE", claim=claim, source=sources[1], verdict="supports"),
        kv_line(p3, "EVIDENCE", claim=claim, source=sources[2], verdict="contradicts"),
    ])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
        assistant_tool("mem-prov-1", code),
        tool_result("mem-prov-1", output),
        {"role": "assistant", "content": expected},
    ]
    return Example(messages, {HISTORY_PATH: history}, {
        "dataset": DATASET_NAME, "split": split, "family": "provenance_conflict",
        "instance": instance, "horizon_events": horizon, "history_format": "kv", "instruction_level": instruction_level,
        "requires_history": True, "retrieval_policy": "latest_evidence_with_provenance", "expected_answer": expected,
    })


def checkpoint_resume_example(rng: random.Random, split: str, instance: int, horizon: int) -> Example:
    run = f"run-{rng.randint(200,999)}"
    steps = [4, 8, 12, 16]
    states = ["stable", "stable", "corrupt", "started"]
    p = [horizon // 6, horizon // 3, horizon // 2, horizon - 4]
    history = insert_events(
        rng, horizon,
        [(pos, kv_line(pos, "CHECKPOINT", run=run, step=step, state=state))
         for pos, step, state in zip(p, steps, states)],
    )
    q, instruction_level = wording(
        instance,
        f"We need to resume `{run}` after the later checkpoint corruption. What is the latest stable step? Return only the step number.",
        f"Which step should `{run}` resume from? Return only the step number.",
        f"Context was compacted during recovery. Which stable step should `{run}` resume from? Return only the number.",
    )
    code = (
        "from pathlib import Path\n"
        f"run = {run!r}\n"
        f"stable = [ln.strip() for ln in Path({HISTORY_PATH!r}).open() if 'TYPE=CHECKPOINT' in ln and f'run=\"{{run}}\"' in ln and 'state=\"stable\"' in ln]\n"
        "stable[-1]"
    )
    output = repr(kv_line(p[1], "CHECKPOINT", run=run, step=8, state="stable"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
        assistant_tool("mem-ckpt-1", code),
        tool_result("mem-ckpt-1", output),
        {"role": "assistant", "content": "8"},
    ]
    return Example(messages, {HISTORY_PATH: history}, {
        "dataset": DATASET_NAME, "split": split, "family": "checkpoint_resume",
        "instance": instance, "horizon_events": horizon, "history_format": "kv", "instruction_level": instruction_level,
        "requires_history": True, "retrieval_policy": "last_stable_before_failure", "expected_answer": "8",
    })


def stale_note_override_example(rng: random.Random, split: str, instance: int, horizon: int) -> Example:
    key = rng.choice(["owner", "target_branch", "deadline", "primary_endpoint"])
    old, new = {
        "owner": ("alice", "bob"),
        "target_branch": ("main", "release/v2"),
        "deadline": ("2026-09-01", "2026-09-08"),
        "primary_endpoint": ("api-a", "api-b"),
    }[key]
    p1, p2 = horizon // 3, horizon - 5
    history = insert_events(
        rng, horizon,
        [
            (p1, kv_line(p1, "DECISION", key=key, value=old, status="accepted")),
            (p2, kv_line(p2, "DECISION", key=key, value=new, status="accepted")),
        ],
    )
    notes = f"# working notes\n{key}: {old}\n# generated before the last decision change\n"
    q, instruction_level = wording(
        instance,
        f"The workspace note may be stale. What is the current accepted `{key}`? Return only the value.",
        f"What is the current accepted `{key}`? Return only the value.",
        f"After compaction, verify the current accepted `{key}`. Return only the value.",
    )
    code = (
        "from pathlib import Path\n"
        f"key = {key!r}\n"
        f"note = Path({NOTES_PATH!r}).read_text().strip()\n"
        f"history_matches = [ln.strip() for ln in Path({HISTORY_PATH!r}).open() if 'TYPE=DECISION' in ln and f'key=\"{{key}}\"' in ln and 'status=\"accepted\"' in ln]\n"
        "{'note': note, 'history_tail': history_matches[-2:]}"
    )
    output = repr({
        "note": notes.strip(),
        "history_tail": [
            kv_line(p1, "DECISION", key=key, value=old, status="accepted"),
            kv_line(p2, "DECISION", key=key, value=new, status="accepted"),
        ],
    })
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
        assistant_tool("mem-stale-1", code),
        tool_result("mem-stale-1", output),
        {"role": "assistant", "content": new},
    ]
    return Example(messages, {HISTORY_PATH: history, NOTES_PATH: notes}, {
        "dataset": DATASET_NAME, "split": split, "family": "stale_note_override",
        "instance": instance, "horizon_events": horizon, "history_format": "kv", "instruction_level": instruction_level,
        "requires_history": True, "retrieval_policy": "source_of_truth_over_derived_cache", "expected_answer": new,
    })


def repeated_lookup_index_example(rng: random.Random, split: str, instance: int, horizon: int) -> Example:
    keys = ["alpha", "beta", "gamma"]
    vals = {k: f"{k}-{rng.randint(10,99)}" for k in keys}
    positions = [horizon // 4, horizon // 2, horizon - 5]
    history = insert_events(
        rng, horizon,
        [(p, kv_line(p, "BINDING", key=k, value=vals[k], status="current")) for p, k in zip(positions, keys)],
    )
    first_key, second_key = rng.sample(keys, 2)
    code1 = (
        "from pathlib import Path\nimport re\n"
        "bindings = {}\n"
        f"for line in Path({HISTORY_PATH!r}).open():\n"
        "    if 'TYPE=BINDING' not in line or 'status=\"current\"' not in line:\n"
        "        continue\n"
        "    k = re.search(r'key=\"([^\"]+)\"', line).group(1)\n"
        "    v = re.search(r'value=\"([^\"]+)\"', line).group(1)\n"
        "    bindings[k] = v\n"
        "bindings"
    )
    q1, instruction_level = wording(
        instance,
        f"Recover the current binding for `{first_key}`.",
        f"What is `{first_key}` currently bound to?",
        f"Context was compacted before this request. What is `{first_key}` currently bound to?",
    )
    q2 = f"And `{second_key}`? Reuse useful state from the prior lookup."
    code2 = f"bindings[{second_key!r}]"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q1},
        assistant_tool("mem-index-1", code1),
        tool_result("mem-index-1", repr(vals)),
        {"role": "assistant", "content": vals[first_key]},
        {"role": "user", "content": q2},
        assistant_tool("mem-index-2", code2),
        tool_result("mem-index-2", repr(vals[second_key])),
        {"role": "assistant", "content": vals[second_key]},
    ]
    return Example(messages, {HISTORY_PATH: history}, {
        "dataset": DATASET_NAME, "split": split, "family": "repeated_lookup_index",
        "instance": instance, "horizon_events": horizon, "history_format": "kv", "instruction_level": instruction_level,
        "requires_history": True, "retrieval_policy": "build_then_reuse_derived_index",
        "expected_answer": [vals[first_key], vals[second_key]],
    })


def multi_key_join_example(rng: random.Random, split: str, instance: int, horizon: int) -> Example:
    service = rng.choice(["payments", "search", "training", "metrics"])
    version = f"v{rng.randint(3,9)}"
    region = rng.choice(["eu-central", "us-west", "ap-south"])
    p1, p2 = horizon // 3, horizon - 6
    history = insert_events(
        rng, horizon,
        [
            (p1, kv_line(p1, "DEPLOY", service=service, field="version", value=version, status="current")),
            (p2, kv_line(p2, "DEPLOY", service=service, field="region", value=region, status="current")),
        ],
    )
    expected = f"{service}@{version} in {region}"
    q, instruction_level = wording(
        instance,
        f"Reconstruct the current deployment identity for `{service}` as `<service>@<version> in <region>`.",
        f"What deployment of `{service}` is active? Answer `<service>@<version> in <region>`.",
        f"Resume after compaction: what deployment of `{service}` is active? Answer `<service>@<version> in <region>`.",
    )
    code = (
        "from pathlib import Path\nimport re\n"
        f"service = {service!r}\n"
        f"lines = [ln.strip() for ln in Path({HISTORY_PATH!r}).open() if 'TYPE=DEPLOY' in ln and f'service=\"{{service}}\"' in ln and 'status=\"current\"' in ln]\n"
        "fields = {}\n"
        "for ln in lines:\n"
        "    field = re.search(r'field=\"([^\"]+)\"', ln).group(1)\n"
        "    value = re.search(r'value=\"([^\"]+)\"', ln).group(1)\n"
        "    fields[field] = value\n"
        "fields"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
        assistant_tool("mem-join-1", code),
        tool_result("mem-join-1", repr({"version": version, "region": region})),
        {"role": "assistant", "content": expected},
    ]
    return Example(messages, {HISTORY_PATH: history}, {
        "dataset": DATASET_NAME, "split": split, "family": "multi_key_join",
        "instance": instance, "horizon_events": horizon, "history_format": "kv", "instruction_level": instruction_level,
        "requires_history": True, "retrieval_policy": "join_distant_events", "expected_answer": expected,
    })


def context_reset_resume_example(rng: random.Random, split: str, instance: int, horizon: int) -> Example:
    objective = rng.choice(["finish_migration", "publish_report", "repair_pipeline", "qualify_model"])
    actions = ["inspect current blockers", "validate the latest artifact", "run the frozen acceptance screen"]
    p1, p2, p3 = horizon // 5, horizon // 2, horizon - 5
    history = insert_events(
        rng, horizon,
        [
            (p1, kv_line(p1, "PLAN", objective=objective, next_action=actions[0], status="active")),
            (p2, kv_line(p2, "PLAN", objective=objective, next_action=actions[1], status="active")),
            (p3, kv_line(p3, "PLAN", objective=objective, next_action=actions[2], status="active")),
        ],
    )
    q, instruction_level = wording(
        instance,
        f"Use the persistent session history to resume `{objective}` after context reset. What is the latest active next action?",
        f"We lost the active context while working on `{objective}`. What should we do next?",
        f"The conversation context was compacted. Resume `{objective}` and return the next action only.",
    )
    code = (
        "from pathlib import Path\n"
        f"objective = {objective!r}\n"
        f"plans = [ln.strip() for ln in Path({HISTORY_PATH!r}).open() if 'TYPE=PLAN' in ln and f'objective=\"{{objective}}\"' in ln and 'status=\"active\"' in ln]\n"
        "plans[-3:]"
    )
    output = repr([
        kv_line(p1, "PLAN", objective=objective, next_action=actions[0], status="active"),
        kv_line(p2, "PLAN", objective=objective, next_action=actions[1], status="active"),
        kv_line(p3, "PLAN", objective=objective, next_action=actions[2], status="active"),
    ])
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":q},
        assistant_tool("mem-resume-1", code),
        tool_result("mem-resume-1", output),
        {"role":"assistant","content":actions[2]},
    ]
    return Example(messages, {HISTORY_PATH: history}, {
        "dataset": DATASET_NAME, "split": split, "family": "context_reset_resume",
        "instance": instance, "horizon_events": horizon, "history_format": "kv",
        "instruction_level": instruction_level, "requires_history": True,
        "retrieval_policy": "recover_after_context_loss", "expected_answer": actions[2],
    })


def constraint_update_example(rng: random.Random, split: str, instance: int, horizon: int) -> Example:
    constraint = rng.choice(["max_parallelism", "output_format", "allowed_region", "review_mode"])
    old, new = {
        "max_parallelism": ("4", "8"),
        "output_format": ("csv", "parquet"),
        "allowed_region": ("eu-west", "eu-central"),
        "review_mode": ("manual", "automatic"),
    }[constraint]
    p1, p2 = horizon // 4, horizon - 6
    history = insert_events(
        rng, horizon,
        [
            (p1, kv_line(p1, "USER_CONSTRAINT", key=constraint, value=old, status="active")),
            (p2, kv_line(p2, "USER_CONSTRAINT", key=constraint, value=new, status="active", supersedes=old)),
        ],
    )
    q, instruction_level = wording(
        instance,
        f"Recover the current user constraint `{constraint}`. Return only its value.",
        f"What `{constraint}` should we obey now? Return only the value.",
        f"After context compaction, continue under the user's current `{constraint}`. Return only the value.",
    )
    code = (
        "from pathlib import Path\n"
        f"key = {constraint!r}\n"
        f"rows = [ln.strip() for ln in Path({HISTORY_PATH!r}).open() if 'TYPE=USER_CONSTRAINT' in ln and f'key=\"{{key}}\"' in ln and 'status=\"active\"' in ln]\n"
        "rows[-2:]"
    )
    output = repr([
        kv_line(p1, "USER_CONSTRAINT", key=constraint, value=old, status="active"),
        kv_line(p2, "USER_CONSTRAINT", key=constraint, value=new, status="active", supersedes=old),
    ])
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":q},
        assistant_tool("mem-constraint-1", code),
        tool_result("mem-constraint-1", output),
        {"role":"assistant","content":new},
    ]
    return Example(messages, {HISTORY_PATH: history}, {
        "dataset": DATASET_NAME, "split": split, "family": "constraint_update",
        "instance": instance, "horizon_events": horizon, "history_format": "kv",
        "instruction_level": instruction_level, "requires_history": True,
        "retrieval_policy": "latest_user_constraint", "expected_answer": new,
    })


def direct_control_example(rng: random.Random, split: str, instance: int, horizon: int) -> Example:
    a, b = rng.randint(11, 50), rng.randint(11, 50)
    history = insert_events(rng, horizon, [])
    q = f"This request is self-contained: what is {a} + {b}? Return only the integer. Do not inspect prior history unless needed."
    expected = str(a + b)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
        {"role": "assistant", "content": expected},
    ]
    return Example(messages, {HISTORY_PATH: history}, {
        "dataset": DATASET_NAME, "split": split, "family": "direct_control",
        "instance": instance, "horizon_events": horizon, "history_format": "kv", "instruction_level": "natural",
        "requires_history": False, "retrieval_policy": "no_retrieval_when_self_contained", "expected_answer": int(expected),
    })


TRAIN_FAMILIES: list[Callable[[random.Random, str, int, int], Example]] = [
    latest_state_example,
    accepted_requirement_example,
    successful_attempt_example,
    correction_aggregate_example,
    provenance_conflict_example,
    checkpoint_resume_example,
    stale_note_override_example,
    repeated_lookup_index_example,
    multi_key_join_example,
    context_reset_resume_example,
    constraint_update_example,
    direct_control_example,
]


def ood_jsonl_latest_revision(rng: random.Random, split: str, instance: int, horizon: int) -> Example:
    topic = rng.choice(["schema", "policy", "experiment", "contract"])
    revisions = [rng.randint(1,3), rng.randint(4,6), rng.randint(7,9)]
    positions = [horizon // 5, horizon // 2, horizon - 3]
    events = []
    for i in range(1, horizon + 1):
        if i in positions:
            rev = revisions[positions.index(i)]
            events.append({"seq": i, "kind": "revision", "topic": topic, "revision": rev, "state": "adopted"})
        else:
            events.append({"seq": i, "kind": "noise", "topic": rng.choice(["a","b","c"]), "value": rng.randint(0,999)})
    history = "\n".join(json.dumps(e, separators=(",", ":")) for e in events) + "\n"
    expected = str(revisions[-1])
    q = f"The historical log is JSONL in this episode. What is the latest adopted revision for `{topic}`? Return only the revision number."
    code = (
        "from pathlib import Path\nimport json\n"
        f"topic = {topic!r}\n"
        f"rows = (json.loads(line) for line in Path({HISTORY_PATH!r}).open())\n"
        "matches = [r for r in rows if r.get('kind') == 'revision' and r.get('topic') == topic and r.get('state') == 'adopted']\n"
        "matches[-1]"
    )
    output = repr({"seq": positions[-1], "kind": "revision", "topic": topic, "revision": revisions[-1], "state": "adopted"})
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":q},
        assistant_tool("mem-ood-json-1", code),
        tool_result("mem-ood-json-1", output),
        {"role":"assistant","content":expected},
    ]
    return Example(messages, {HISTORY_PATH: history}, {
        "dataset": DATASET_NAME, "split": split, "family": "ood_jsonl_latest_revision",
        "instance": instance, "horizon_events": horizon, "history_format": "jsonl", "instruction_level": "natural_ood",
        "requires_history": True, "retrieval_policy": "format_generalization", "expected_answer": int(expected),
    })


def ood_temporal_window(rng: random.Random, split: str, instance: int, horizon: int) -> Example:
    tag = rng.choice(["latency", "accuracy", "throughput"])
    events = []
    selected = []
    for i in range(1, horizon + 1):
        if i % 17 == 0:
            val = rng.randint(10, 99)
            e = {"seq": i, "kind": "measurement", "tag": tag, "value": val}
            selected.append(e)
        else:
            e = {"seq": i, "kind": "noise", "tag": rng.choice(["x","y","z"]), "value": rng.randint(1,99)}
        events.append(e)
    history = "\n".join(json.dumps(e, separators=(",", ":")) for e in events) + "\n"
    tail = selected[-3:]
    expected = str(round(sum(e["value"] for e in tail) / len(tail), 2))
    q = f"Using the history, compute the mean of the last three `{tag}` measurements. Return only the number."
    code = (
        "from pathlib import Path\nimport json\n"
        f"tag = {tag!r}\n"
        f"vals = [r['value'] for r in (json.loads(x) for x in Path({HISTORY_PATH!r}).open()) if r.get('kind') == 'measurement' and r.get('tag') == tag]\n"
        "round(sum(vals[-3:]) / 3, 2)"
    )
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":q},
        assistant_tool("mem-ood-win-1", code),
        tool_result("mem-ood-win-1", expected),
        {"role":"assistant","content":expected},
    ]
    return Example(messages, {HISTORY_PATH: history}, {
        "dataset": DATASET_NAME, "split": split, "family": "ood_temporal_window",
        "instance": instance, "horizon_events": horizon, "history_format": "jsonl", "instruction_level": "natural_ood",
        "requires_history": True, "retrieval_policy": "temporal_window_compute", "expected_answer": float(expected),
    })


def ood_supersession_chain(rng: random.Random, split: str, instance: int, horizon: int) -> Example:
    object_id = f"obj-{rng.randint(100,999)}"
    p = [horizon // 5, horizon // 2, horizon - 4]
    vals = [f"state-{rng.randint(10,29)}", f"state-{rng.randint(30,59)}", f"state-{rng.randint(60,99)}"]
    events = []
    for i in range(1, horizon + 1):
        if i in p:
            idx = p.index(i)
            events.append({
                "seq": i, "kind": "transition", "object": object_id, "value": vals[idx],
                "supersedes": None if idx == 0 else vals[idx-1],
            })
        else:
            events.append({"seq":i,"kind":"noise","object":f"obj-{rng.randint(1,9)}","value":rng.randint(0,99)})
    history = "\n".join(json.dumps(e, separators=(",", ":")) for e in events) + "\n"
    q = f"Follow the supersession chain for `{object_id}` and return its current value."
    code = (
        "from pathlib import Path\nimport json\n"
        f"obj = {object_id!r}\n"
        f"events = [json.loads(x) for x in Path({HISTORY_PATH!r}).open()]\n"
        "chain = [e for e in events if e.get('kind') == 'transition' and e.get('object') == obj]\n"
        "chain"
    )
    output = repr([e for e in events if e.get("kind")=="transition" and e.get("object")==object_id])
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":q},
        assistant_tool("mem-ood-chain-1", code),
        tool_result("mem-ood-chain-1", output),
        {"role":"assistant","content":vals[-1]},
    ]
    return Example(messages, {HISTORY_PATH: history}, {
        "dataset": DATASET_NAME, "split": split, "family": "ood_supersession_chain",
        "instance": instance, "horizon_events": horizon, "history_format": "jsonl", "instruction_level": "natural_ood",
        "requires_history": True, "retrieval_policy": "supersession_chain", "expected_answer": vals[-1],
    })


OOD_FAMILIES = [ood_jsonl_latest_revision, ood_temporal_window, ood_supersession_chain]


def to_row(ex: Example) -> dict:
    return base_row(ex.messages, ex.files, ex.metadata)


def validate_row(row: dict) -> None:
    messages = json.loads(row["messages_json"])
    tools = json.loads(row["tools"])
    files = json.loads(row["workspace_files_json"])
    meta = json.loads(row["metadata_json"])
    assert tools == [IPYTHON_TOOL]
    assert HISTORY_PATH in files
    assert all("reasoning_content" not in m for m in messages)
    has_ipython = any(m.get("role") == "assistant" and m.get("tool_calls") for m in messages)
    if meta["requires_history"]:
        assert has_ipython, meta
        codes = []
        for m in messages:
            for call in m.get("tool_calls", []) or []:
                args = json.loads(call["function"]["arguments"])
                codes.append(args["code"])
        assert any(HISTORY_PATH in code for code in codes), meta
        if meta["family"] == "repeated_lookup_index":
            assert HISTORY_PATH in codes[0] and HISTORY_PATH not in codes[1]
    else:
        assert not has_ipython, meta
    history = files[HISTORY_PATH]
    active_text = json.dumps(messages, ensure_ascii=False)
    assert len(history) > 500
    assert history not in active_text


def generate(seed: int, train_per_family: int, heldout_per_family: int, ood_per_family: int) -> dict[str, list[dict]]:
    root = random.Random(seed)
    splits: dict[str, list[dict]] = {"train": [], "familiar_heldout": [], "semantic_ood": []}
    horizon_buckets = [32, 64, 128, 192]
    for family_index, family in enumerate(TRAIN_FAMILIES):
        for i in range(train_per_family):
            rng = random.Random(root.randrange(2**63))
            horizon = horizon_buckets[(i + family_index) % len(horizon_buckets)]
            splits["train"].append(to_row(family(rng, "train", i, horizon)))
        for i in range(heldout_per_family):
            rng = random.Random(root.randrange(2**63))
            horizon = [96, 160, 224][(i + family_index) % 3]
            splits["familiar_heldout"].append(to_row(family(rng, "familiar_heldout", i, horizon)))
    for family_index, family in enumerate(OOD_FAMILIES):
        for i in range(ood_per_family):
            rng = random.Random(root.randrange(2**63))
            horizon = [128, 256, 384][(i + family_index) % 3]
            splits["semantic_ood"].append(to_row(family(rng, "semantic_ood", i, horizon)))
    for rows in splits.values():
        for row in rows:
            validate_row(row)
    return splits


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dataset(output_dir: Path, splits: dict[str, list[dict]], seed: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    family_counts = {}
    for split, rows in splits.items():
        jsonl_path = output_dir / f"{split}.jsonl"
        jsonl_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        files[jsonl_path.name] = {"rows": len(rows), "sha256": sha256_text(jsonl_path)}
        if Dataset is not None:
            parquet_path = output_dir / f"{split}.parquet"
            Dataset.from_list(rows).to_parquet(str(parquet_path))
            files[parquet_path.name] = {"rows": len(rows), "sha256": sha256_text(parquet_path)}
        counts = {}
        for row in rows:
            fam = json.loads(row["metadata_json"])["family"]
            counts[fam] = counts.get(fam, 0) + 1
        family_counts[split] = dict(sorted(counts.items()))
    manifest = {
        "schema_version": 1,
        "dataset": DATASET_NAME,
        "seed": seed,
        "source_of_truth": HISTORY_PATH,
        "derived_state_policy": "workspace notes/indexes are caches; verify conflicts against history.log",
        "reasoning_policy": "no fabricated reasoning_content",
        "splits": {k: len(v) for k, v in splits.items()},
        "family_counts": family_counts,
        "files": files,
        "design": {
            "train_history_format": "tagged key-value append-only lines",
            "familiar_heldout": "same policy families, fresh instances and longer horizons",
            "semantic_ood": "unseen JSONL memory formats and unseen temporal/supersession operations",
            "horizon_events": "32-224 train/familiar; 128-384 OOD",
            "instruction_levels": "explicit-history, natural, and explicit context-reset variants; OOD is natural",
            "direct_controls": "self-contained tasks with no history lookup",
            "persistent_state": "repeated_lookup_index builds an index once and reuses it on the next request",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-per-family", type=int, default=6)
    parser.add_argument("--heldout-per-family", type=int, default=3)
    parser.add_argument("--ood-per-family", type=int, default=6)
    args = parser.parse_args()
    splits = generate(args.seed, args.train_per_family, args.heldout_per_family, args.ood_per_family)
    manifest = write_dataset(args.output_dir, splits, args.seed)
    print(json.dumps(manifest["splits"], sort_keys=True))
    print(f"wrote {DATASET_NAME} to {args.output_dir}")


if __name__ == "__main__":
    main()
