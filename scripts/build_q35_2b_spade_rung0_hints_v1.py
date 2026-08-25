#!/usr/bin/env python3
"""Generate and validate same-policy privileged hints for SPADE Rung 0."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from procedural_harness_master_v1.taskset import (
    ProceduralHarnessMasterConfig,
    ProceduralHarnessMasterTaskset,
)

SCHEMA_VERSION = "qwen35-2b-spade-rung0-hints/v1"
DEFAULT_AXES = (
    ("natural_direct_control", 4_007_000),
    ("natural_n1a", 4_004_000),
    ("natural_n1a_local", 4_006_000),
    ("natural_n1b", 4_005_000),
)
DESIGNER_SYSTEM_PROMPT = """You are the Environment Designer in a self-play experiment.
Write one operational 120-220 word strategy hint that covers every supplied hint requirement.
Use exact public child names and paths. For delegated work, include a valid first-cell Python skeleton
whose `await rlm("nonempty child instruction", name="exact-name")` call has both arguments, whose child
instruction contains `await agent_message.send(..., receiver_role='parent')`, and whose coordinator-local
work remains in the root cell. Then prescribe passive yielding without sleeps, polling, discovery, or
child-path inspection.
Never reveal or guess any exact answer value, private evidence value, reviewer result, or hidden
coordinator-state value. Never claim you executed the task. Return only the hint text."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _designer_view(task) -> dict[str, Any]:
    oracle = task.data.oracle
    children = [
        {
            key: child[key]
            for key in ("name", "resource_path", "operation", "message_contract")
            if key in child
        }
        for child in oracle.get("children", [])
    ]
    persistence = oracle.get("persistence_lease")
    persistence_contract = None
    if isinstance(persistence, dict):
        persistence_contract = {
            key: persistence[key]
            for key in ("path", "key")
            if key in persistence
        }
    return {
        "episode_id": task.key,
        "public_system_prompt": task.data.system_prompt,
        "public_user_prompt": task.data.prompt,
        "generation_metadata": task.data.generation_metadata,
        "privileged_contract": {
            "expected_route": oracle.get("expected_route"),
            "resource_ownership": oracle.get("resource_ownership", {}),
            "children": children,
            "coordinator_state_keys": sorted(oracle.get("coordinator_state", {})),
            "persistence_contract": persistence_contract,
            "fault_type": oracle.get("fault_plan", {}).get("type"),
            "trajectory_contract": oracle["trajectory_contract"],
            "final_answer_keys": list(oracle["final_answer"]),
        },
        "hint_requirements": _hint_requirements(task),
    }


def _hint_requirements(task) -> list[str]:
    oracle = task.data.oracle
    family = task.data.family
    if family == "natural_direct_control":
        return [
            "Solve at the coordinator without delegating or creating a child.",
            "Either a clean in-context calculation or one clean local IPython calculation is valid.",
            "Return the exact requested bare JSON keys without unrelated exploration.",
        ]

    children = oracle.get("children", [])
    child = children[0]
    requirements = [
        (
            "Show a first root IPython cell containing exactly one valid spawn of the form "
            f"`handle = await rlm(\"Review {child['resource_path']} and call await "
            "agent_message.send(..., receiver_role='parent') exactly once\", "
            f"name={child['name']!r})`."
        ),
        "Retain the returned handle variable without inspecting or waiting on it.",
        "The coordinator must not read or inspect the child-owned evidence path.",
        "Do not sleep, poll, discover/list agents, await the child, or spawn a replacement.",
        "After all coordinator-owned work is complete, stop tool use and yield by ending the turn briefly.",
        "Resume only from the delivered child message, then compute and return bare JSON.",
    ]
    for child in children:
        requirements.append(
            f"Use exact child name {child['name']!r} for child-owned path {child['resource_path']!r}."
        )
    for path, ownership in oracle.get("resource_ownership", {}).items():
        if ownership["owner"] == "coordinator":
            requirements.append(
                f"The coordinator must perform {ownership['operation']!r} on {path!r}."
            )
    if family == "natural_n1b":
        persistence = oracle["persistence_lease"]
        requirements.append(
            f"Capture key {persistence['key']!r} from {persistence['path']!r} before spawning and reuse it after the child report."
        )
    return requirements


def _hint_contract_gaps(task, hint: str) -> list[str]:
    text = hint.casefold()
    oracle = task.data.oracle
    if task.data.family == "natural_direct_control":
        gaps = []
        if "ipython" not in text:
            gaps.append("local_ipython_option")
        if not any(phrase in text for phrase in ("do not delegate", "no child", "without delegat")):
            gaps.append("delegation_abstention")
        return gaps

    checks = {
        "exactly_one_child": any(
            phrase in text
            for phrase in ("exactly one", "one child", "single child", "a single child")
        ),
        "rlm_spawn": "await rlm" in text,
        "rlm_task_argument": bool(re.search(r"await\s+rlm\(\s*['\"]\S", hint)),
        "named_spawn": "name=" in text,
        "retained_handle": any(
            verb in text for verb in ("retain", "keep", "hold")
        )
        and "handle" in text,
        "child_reply": "agent_message.send" in text and "parent" in text,
        "no_polling": bool(
            re.search(r"(?:do not|never|without)[^.]{0,80}\bpoll", text)
        ),
        "passive_yield": "yield" in text and "end" in text,
        "child_path_noninspection": bool(
            re.search(
                r"(?:do not|must not|never)[^.]{0,100}(?:read|inspect)[^.]{0,100}child",
                text,
            )
        ),
        "post_report_finalization": "json" in text
        and any(
            phrase in text
            for phrase in (
                "after the child",
                "after child",
                "once the child",
                "child report",
                "child message",
            )
        ),
    }
    for child in oracle.get("children", []):
        checks[f"child_name:{child['name']}"] = child["name"].casefold() in text
        checks[f"child_path:{child['resource_path']}"] = child["resource_path"].casefold() in text
    for path, ownership in oracle.get("resource_ownership", {}).items():
        if ownership["owner"] == "coordinator":
            checks[f"coordinator_path:{path}"] = path.casefold() in text
            rlm_call = re.search(r"await\s+rlm\((?P<body>.*?)\)", hint, re.DOTALL)
            checks[f"coordinator_work_not_delegated:{path}"] = (
                rlm_call is not None and path.casefold() not in rlm_call.group("body").casefold()
            )
    if task.data.family == "natural_n1b":
        persistence = oracle["persistence_lease"]
        checks["capture_before_spawn"] = "capture" in text and "before" in text
        checks["persistence_path"] = persistence["path"].casefold() in text
    return [name for name, present in checks.items() if not present]


def _contract_gap_feedback(task, gaps: list[str]) -> list[str]:
    child = next(iter(task.data.oracle.get("children", [])), None)
    explanations = {
        "exactly_one_child": "Use the literal words 'exactly one child'.",
        "rlm_spawn": "Include a literal `await rlm(...)` spawn.",
        "rlm_task_argument": (
            "Put a non-empty quoted child instruction as the first positional argument "
            "inside `await rlm(\"...\", name=...)`."
        ),
        "named_spawn": (
            f"Put the literal keyword argument `name={child['name']!r}` inside the rlm call."
            if child is not None
            else "Put the literal `name=` keyword argument inside the rlm call."
        ),
        "retained_handle": "Use the literal words 'retain the handle'.",
        "child_reply": (
            "Include literal `await agent_message.send(..., receiver_role='parent')` "
            "in the child instruction."
        ),
        "no_polling": "Use the literal clause 'Do not sleep, poll, or discover agents'.",
        "passive_yield": "Say to end the coordinator turn briefly to yield.",
        "child_path_noninspection": (
            "Use the literal clause 'The coordinator must not read or inspect the "
            "child-owned path'."
        ),
        "post_report_finalization": (
            "Say 'After the child report' before the final bare-JSON instruction."
        ),
        "capture_before_spawn": "Say to capture the value before the rlm spawn.",
        "persistence_path": "Include the exact coordinator persistence path.",
    }
    feedback = []
    for gap in gaps:
        if gap.startswith("child_name:"):
            feedback.append(f"Include exact child name {gap.removeprefix('child_name:')!r}.")
        elif gap.startswith("child_path:"):
            feedback.append(f"Include exact child path {gap.removeprefix('child_path:')!r}.")
        elif gap.startswith("coordinator_path:"):
            feedback.append(
                f"Include exact coordinator path {gap.removeprefix('coordinator_path:')!r}."
            )
        elif gap.startswith("coordinator_work_not_delegated:"):
            feedback.append(
                "Keep coordinator path "
                f"{gap.removeprefix('coordinator_work_not_delegated:')!r} outside the "
                "rlm child instruction and say the root coordinator performs that work."
            )
        else:
            feedback.append(explanations.get(gap, f"Cover requirement {gap!r}."))
    return feedback


def _protected_values(task) -> list[Any]:
    oracle = task.data.oracle
    values: list[Any] = []

    def collect(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float, str)):
            values.append(value)
            return
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
            return
        if isinstance(value, list):
            for item in value:
                collect(item)

    collect(oracle.get("final_answer", {}))
    collect(oracle.get("private_resources", {}))
    collect(oracle.get("coordinator_state", {}))
    for child in oracle.get("children", []):
        collect(child.get("expected_result"))
    persistence = oracle.get("persistence_lease")
    if isinstance(persistence, dict):
        collect(persistence.get("expected_value"))
    return values


def _leaks_protected_value(hint: str, protected_values: list[Any]) -> list[str]:
    leaks = []
    for value in protected_values:
        rendered = str(value).strip()
        if not rendered:
            continue
        if isinstance(value, (int, float)):
            pattern = rf"(?<![A-Za-z0-9_.-]){re.escape(rendered)}(?![A-Za-z0-9_.-])"
            if re.search(pattern, hint):
                leaks.append(rendered)
        elif len(rendered) >= 4 and rendered.casefold() in hint.casefold():
            leaks.append(rendered)
    return sorted(set(leaks))


def _chat_completion(
    base_url: str,
    model: str,
    designer_view: dict[str, Any],
    temperature: float,
    max_tokens: int,
    seed: int,
    timeout: float,
    revision_feedback: str | None,
) -> tuple[str | None, dict[str, Any]]:
    revision = ""
    if revision_feedback is not None:
        revision = f"\n\nRevision feedback from the validator:\n{revision_feedback}"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": DESIGNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Produce a nonleaking strategy hint for this executable environment:\n"
                    + json.dumps(designer_view, indent=2, ensure_ascii=False)
                    + revision
                ),
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer local-eval", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    message = payload["choices"][0]["message"]
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    audit = {
        "request_sha256": _sha256_text(_canonical_json(body)),
        "response_id": payload.get("id"),
        "finish_reason": payload["choices"][0].get("finish_reason"),
        "usage": payload.get("usage"),
        "reasoning_content_present": isinstance(reasoning, str) and bool(reasoning),
        "reasoning_content_sha256": (
            _sha256_text(reasoning) if isinstance(reasoning, str) and reasoning else None
        ),
    }
    if not isinstance(content, str) or not content.strip():
        return None, audit
    return content.strip(), audit


def _parse_axis(value: str) -> tuple[str, int]:
    name, separator, start = value.partition(":")
    if not separator or not name or not start.isdigit():
        raise argparse.ArgumentTypeError("axis must have the form NAME:START_INDEX")
    return name, int(start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--axis", action="append", type=_parse_axis)
    parser.add_argument("--tasks-per-axis", type=int, default=8)
    parser.add_argument("--master-seed", type=int, default=20260819)
    parser.add_argument("--sampling-seed", type=int, default=20260830)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite hint artifact: {args.output}")
    if args.tasks_per_axis < 1 or args.max_tokens < 1 or args.max_attempts < 1:
        raise ValueError("task, token, and attempt counts must be positive")
    if args.temperature <= 0 or args.sampling_seed < 0:
        raise ValueError("temperature must be positive and sampling seed non-negative")

    axes = tuple(args.axis or DEFAULT_AXES)
    hints: dict[str, str] = {}
    records = []

    def artifact(status: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "model": args.model,
            "model_revision": args.model_revision,
            "role": "environment_designer_hint_writer",
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "master_seed": args.master_seed,
            "sampling_seed": args.sampling_seed,
            "tasks_per_axis": args.tasks_per_axis,
            "axes": [{"name": name, "start_index": start} for name, start in axes],
            "designer_system_prompt_sha256": _sha256_text(DESIGNER_SYSTEM_PROMPT),
            "designer_view_policy": (
                "structural oracle metadata with protected values removed"
            ),
            "hints": hints,
            "records": records,
        }

    ordinal = 0
    for axis, start_index in axes:
        tasks = ProceduralHarnessMasterTaskset(
            ProceduralHarnessMasterConfig(
                count=args.tasks_per_axis,
                start_index=start_index,
                master_seed=args.master_seed,
                curriculum_rung=axis,
                private_payload_mode="finding_card",
            )
        ).load()
        for task in tasks:
            designer_view = _designer_view(task)
            protected = _protected_values(task)
            rejected = []
            accepted_hint = None
            accepted_audit = None
            revision_feedback = None
            for attempt in range(args.max_attempts):
                seed = args.sampling_seed + ordinal * args.max_attempts + attempt
                hint, audit = _chat_completion(
                    args.base_url,
                    args.model,
                    designer_view,
                    args.temperature,
                    args.max_tokens,
                    seed,
                    args.timeout,
                    revision_feedback,
                )
                if hint is None:
                    rejected.append(
                        {
                            "attempt": attempt + 1,
                            "seed": seed,
                            "rejection": "no_final_hint_text",
                            **audit,
                        }
                    )
                    revision_feedback = (
                        "The previous response had no final hint text. Return a complete hint."
                    )
                    continue
                leaks = _leaks_protected_value(hint, protected)
                if leaks:
                    rejected.append(
                        {
                            "attempt": attempt + 1,
                            "seed": seed,
                            "hint": hint,
                            "rejection": "protected_value_leak",
                            "leaked_values": leaks,
                            **audit,
                        }
                    )
                    revision_feedback = (
                        "The previous candidate leaked a protected value. Regenerate from "
                        "the structural contract without repeating any exact numeric or "
                        "private value."
                    )
                    continue
                gaps = _hint_contract_gaps(task, hint)
                if gaps:
                    rejected.append(
                        {
                            "attempt": attempt + 1,
                            "seed": seed,
                            "hint": hint,
                            "rejection": "contract_coverage_gap",
                            "contract_gaps": gaps,
                            **audit,
                        }
                    )
                    revision_feedback = (
                        f"Previous nonleaking candidate:\n{hint}\n\n"
                        "Missing validator requirements:\n- "
                        + "\n- ".join(_contract_gap_feedback(task, gaps))
                        + "\n"
                        "Replace the candidate and explicitly cover every missing item."
                    )
                    continue
                accepted_hint = hint
                accepted_audit = {"attempt": attempt + 1, "seed": seed, **audit}
                break
            if accepted_hint is None or accepted_audit is None:
                records.append(
                    {
                        "episode_id": task.key,
                        "axis": axis,
                        "start_index": start_index,
                        "designer_view_sha256": _sha256_text(
                            _canonical_json(designer_view)
                        ),
                        "accepted": None,
                        "rejected_candidates": rejected,
                    }
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(artifact("failed"), indent=2, ensure_ascii=False) + "\n"
                )
                raise RuntimeError(
                    f"designer exhausted nonleaking hint attempts for {task.key}"
                )
            hints[task.key] = accepted_hint
            records.append(
                {
                    "episode_id": task.key,
                    "axis": axis,
                    "start_index": start_index,
                    "designer_view_sha256": _sha256_text(
                        _canonical_json(designer_view)
                    ),
                    "accepted": accepted_audit,
                    "rejected_candidates": rejected,
                }
            )
            ordinal += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact("complete"), indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"output": str(args.output), "hint_count": len(hints)}))


if __name__ == "__main__":
    main()
