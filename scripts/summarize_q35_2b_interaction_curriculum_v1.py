#!/usr/bin/env python3
"""Fail-closed admission summary for the Qwen3.5-2B interaction curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

if __package__:
    from .summarize_procedural_harness_master_v1 import _rescore, _score, _traces
else:
    from summarize_procedural_harness_master_v1 import _rescore, _score, _traces

SCHEMA_VERSION = "qwen35-2b-interaction-curriculum-summary/v1"
CURRICULUM_SCHEMA_VERSION = "prime-agent/interaction-curriculum/v1"
YIELD_SCHEMA_VERSION = "prime-agent/natural-yield-scaffold/v1"
CHILD_VALUE_SEND_PATTERN = r"^await agent_message\.send\('[0-9]+', receiver_role='parent'\)$"
CHILD_VALUE_SEND_PATTERN_SHA256 = hashlib.sha256(CHILD_VALUE_SEND_PATTERN.encode()).hexdigest()
EXPECTED_EVENT_KINDS = {
    "e0_full_actions": ["root_retained_spawn", "child_typed_send"],
    "e0b_select_child_value": ["root_retained_spawn", "child_value_send"],
    "e0c_natural_child": ["root_retained_spawn", "child_natural_send"],
    "e0c2_natural_child_no_template": [
        "root_retained_spawn",
        "child_natural_send",
    ],
    "e0c25_inline_evidence": ["root_retained_spawn", "child_natural_send"],
    "e0c275_inline_location": ["root_retained_spawn", "child_natural_send"],
    "e0c28_inline_only": ["root_retained_spawn", "child_natural_send"],
    "e0c29_evidence_available": ["root_retained_spawn", "child_natural_send"],
    "e0c3_natural_child_minimal": [
        "root_retained_spawn",
        "child_natural_send",
    ],
    "e0d_guided_yield": ["root_retained_spawn", "child_value_send"],
    "e0d2_capped_yield": ["root_retained_spawn", "child_value_send"],
    "e0d2_capped_yield_exact_child": [
        "root_retained_spawn",
        "child_typed_send",
    ],
    "e0d3_uncapped_yield_exact_child": [
        "root_retained_spawn",
        "child_typed_send",
    ],
    "e0d3_uncapped_yield": ["root_retained_spawn", "child_value_send"],
    "e1_root_and_yield": ["root_retained_spawn"],
    "e2_yield_only": [],
}
HARD_METRICS = (
    "final_answer_exact",
    "all_required_atoms",
    "no_forbidden_atoms",
    "ordering_satisfied",
    "cardinality_exact",
)
TRUNCATED_STOPS = {
    "error",
    "max_turns",
    "max_input_tokens",
    "max_output_tokens",
    "max_total_tokens",
    "context_length",
}


def _root(nodes: list[dict[str, Any]], index: int) -> int:
    seen: set[int] = set()
    current = index
    while True:
        if current in seen:
            raise ValueError("trace message graph contains a cycle")
        seen.add(current)
        parent = nodes[current].get("parent")
        if parent is None:
            return current
        if not isinstance(parent, int) or not 0 <= parent < len(nodes):
            raise ValueError(f"trace node {current} has invalid parent {parent}")
        current = parent


def _tool_code(node: dict[str, Any]) -> str | None:
    message = node.get("message", {})
    calls = message.get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("name") != "ipython":
        return None
    try:
        arguments = json.loads(calls[0].get("arguments", ""))
    except (TypeError, json.JSONDecodeError):
        return None
    code = arguments.get("code") if isinstance(arguments, dict) else None
    return code if isinstance(code, str) else None


def _code_sha(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _event_hashes(record: dict[str, Any]) -> dict[str, str] | None:
    events = record.get("events")
    if not isinstance(events, list):
        return None
    result: dict[str, str] = {}
    for event in events:
        if not isinstance(event, dict):
            return None
        kind = event.get("kind")
        digest = event.get("code_sha256") or event.get("sampled_code_sha256")
        if not isinstance(kind, str) or not isinstance(digest, str) or kind in result:
            return None
        result[kind] = digest
    return result


def _native_yield_node(trace: dict[str, Any], spawn_index: int) -> int | None:
    nodes = trace.get("nodes")
    if not isinstance(nodes, list) or not 0 <= spawn_index < len(nodes):
        return None
    primary_root = _root(nodes, 0)
    first_incoming = next(
        (
            index
            for index in range(spawn_index + 1, len(nodes))
            if nodes[index].get("message", {}).get("role") == "user"
            and _message_text(nodes[index].get("message", {}).get("content", "")).lstrip().startswith("[from child:")
        ),
        len(nodes),
    )
    candidates = [
        index
        for index in range(spawn_index + 1, first_incoming)
        if nodes[index].get("sampled") is True
        and nodes[index].get("message", {}).get("role") == "assistant"
        and not nodes[index].get("message", {}).get("tool_calls")
        and _root(nodes, index) == primary_root
    ]
    return candidates[0] if len(candidates) == 1 else None


def _qualify(trace: dict[str, Any], phase: str) -> list[str]:
    if phase not in EXPECTED_EVENT_KINDS:
        raise ValueError(f"unsupported interaction curriculum phase: {phase}")
    expected_events = EXPECTED_EVENT_KINDS[phase]
    reasons: list[str] = []
    if trace.get("is_completed") is not True:
        reasons.append("incomplete")
    if trace.get("ok") is not True or trace.get("errors"):
        reasons.append("errored")
    if trace.get("stop_condition") in TRUNCATED_STOPS:
        reasons.append("truncated_or_error_stop")
    if _score(trace, "harness_score", "rewards") != 1.0:
        reasons.append("not_hard_success")
    metrics = trace.get("metrics") or {}
    if any(float(metrics.get(metric, 0.0)) != 1.0 for metric in HARD_METRICS):
        reasons.append("hard_metrics_incomplete")

    info = trace.get("info") or {}
    curriculum = info.get("interaction_curriculum")
    if expected_events and (
        not isinstance(curriculum, dict)
        or curriculum.get("schema_version") != CURRICULUM_SCHEMA_VERSION
        or curriculum.get("phase") != phase
    ):
        reasons.append(f"missing_{phase}_curriculum_audit")
        event_hashes = None
    elif not expected_events and curriculum is None:
        event_hashes = {}
    elif not isinstance(curriculum, dict):
        reasons.append(f"invalid_{phase}_event_audit")
        event_hashes = None
    else:
        event_hashes = _event_hashes(curriculum)
        if (
            curriculum.get("schema_version") != CURRICULUM_SCHEMA_VERSION
            or curriculum.get("phase") != phase
            or event_hashes is None
            or list(event_hashes) != expected_events
        ):
            reasons.append(f"invalid_{phase}_event_audit")
        if phase in {
            "e0b_select_child_value",
            "e0d_guided_yield",
            "e0d2_capped_yield",
            "e0d3_uncapped_yield",
        }:
            child_events = [
                event
                for event in curriculum.get("events") or []
                if isinstance(event, dict) and event.get("kind") == "child_value_send"
            ]
            if (
                len(child_events) != 1
                or child_events[0].get("mode") != "pattern_constrained_ipython_action"
                or child_events[0].get("code_pattern_sha256") != CHILD_VALUE_SEND_PATTERN_SHA256
                or not isinstance(child_events[0].get("sampled_code_sha256"), str)
            ):
                reasons.append("invalid_child_value_selection_audit")
        if phase in {
            "e0c_natural_child",
            "e0c2_natural_child_no_template",
            "e0c25_inline_evidence",
            "e0c275_inline_location",
            "e0c28_inline_only",
            "e0c29_evidence_available",
            "e0c3_natural_child_minimal",
        }:
            child_events = [
                event
                for event in curriculum.get("events") or []
                if isinstance(event, dict) and event.get("kind") == "child_natural_send"
            ]
            if (
                len(child_events) != 1
                or child_events[0].get("mode") != "unconstrained_ipython_action"
                or not isinstance(child_events[0].get("code_sha256"), str)
            ):
                reasons.append("invalid_natural_child_send_audit")

    nodes = trace.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        reasons.append("missing_nodes")
        return reasons
    primary_root = _root(nodes, 0)
    sampled_hashes: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        if node.get("sampled") is not True:
            continue
        code = _tool_code(node)
        if code is None:
            continue
        sampled_hashes.setdefault(_code_sha(code), []).append(index)
    if event_hashes is not None:
        for kind, digest in event_hashes.items():
            matches = sampled_hashes.get(digest, [])
            expected_root = primary_root if kind == "root_retained_spawn" else None
            lineage_valid = len(matches) == 1 and (
                _root(nodes, matches[0]) == expected_root
                if expected_root is not None
                else _root(nodes, matches[0]) != primary_root
            )
            if not lineage_valid:
                reasons.append(f"unmatched_{kind}_sample")

    scaffold = info.get("natural_yield_scaffold")
    if (
        not isinstance(scaffold, dict)
        or scaffold.get("schema_version") != YIELD_SCHEMA_VERSION
        or scaffold.get("fired") is not True
    ):
        reasons.append("missing_yield_scaffold_audit")
    else:
        spawn_index = scaffold.get("spawn_node_index")
        if not isinstance(spawn_index, int) or _native_yield_node(trace, spawn_index) is None:
            reasons.append("missing_native_yield_response")
        exact_yield = scaffold.get("exact_yield_guidance") is True
        exact_expected = phase in {
            "e0_full_actions",
            "e0b_select_child_value",
            "e0c_natural_child",
            "e0c2_natural_child_no_template",
            "e0c25_inline_evidence",
            "e0c275_inline_location",
            "e0c28_inline_only",
            "e0c29_evidence_available",
            "e0c3_natural_child_minimal",
        }
        if exact_yield != exact_expected:
            reasons.append("wrong_exact_yield_help_level")
        guided_yield = scaffold.get("guided_yield_instruction") is True
        if guided_yield != (phase == "e0d_guided_yield"):
            reasons.append("wrong_guided_yield_help_level")
        capped_yield = scaffold.get("capped_yield_decode") is True
        if capped_yield != (phase in {"e0d2_capped_yield", "e0d2_capped_yield_exact_child"}):
            reasons.append("wrong_capped_yield_help_level")
        if phase in {
            "e0d_guided_yield",
            "e0d2_capped_yield",
            "e0d2_capped_yield_exact_child",
        } and (
            scaffold.get("max_tokens") != 128
            or scaffold.get("decode_constraint") is not None
            or scaffold.get("response_sha256") is not None
        ):
            reasons.append("invalid_nonexact_yield_audit")
        if phase in {
            "e0d3_uncapped_yield_exact_child",
            "e0d3_uncapped_yield",
        } and (
            scaffold.get("max_tokens") is not None
            or scaffold.get("decode_constraint") is not None
            or scaffold.get("response_sha256") is not None
        ):
            reasons.append("invalid_uncapped_yield_audit")
    return reasons


def _qualify_e0(trace: dict[str, Any]) -> list[str]:
    """Backward-compatible E0 admission predicate used by the corpus exporter."""

    return _qualify(trace, "e0_full_actions")


def positive_prefix_audit(trace: dict[str, Any], phase: str) -> dict[str, dict[str, Any]]:
    """Return loss-safe role prefixes from an otherwise unsuccessful trajectory."""

    if phase not in EXPECTED_EVENT_KINDS or not EXPECTED_EVENT_KINDS[phase]:
        return {}
    if (
        trace.get("is_completed") is not True
        or trace.get("ok") is not True
        or trace.get("errors")
        or trace.get("stop_condition") in TRUNCATED_STOPS
    ):
        return {}
    metrics = trace.get("metrics") or {}
    if float(metrics.get("no_forbidden_atoms", 0.0)) != 1.0 or float(
        metrics.get("cardinality_exact", 0.0)
    ) != 1.0:
        return {}
    curriculum = (trace.get("info") or {}).get("interaction_curriculum")
    event_hashes = _event_hashes(curriculum) if isinstance(curriculum, dict) else None
    if (
        not isinstance(curriculum, dict)
        or curriculum.get("schema_version") != CURRICULUM_SCHEMA_VERSION
        or curriculum.get("phase") != phase
        or event_hashes is None
        or list(event_hashes) != EXPECTED_EVENT_KINDS[phase]
    ):
        return {}
    nodes = trace.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return {}
    primary_root = _root(nodes, 0)
    matched: dict[str, int] = {}
    for kind, digest in event_hashes.items():
        matches = [
            index
            for index, node in enumerate(nodes)
            if node.get("sampled") is True
            and (code := _tool_code(node)) is not None
            and _code_sha(code) == digest
        ]
        expected_primary = kind == "root_retained_spawn"
        if len(matches) != 1 or (_root(nodes, matches[0]) == primary_root) is not expected_primary:
            return {}
        matched[kind] = matches[0]

    result: dict[str, dict[str, Any]] = {
        "coordinator": {
            "target_node_index": matched["root_retained_spawn"],
            "atoms": ["root_retained_spawn"],
        }
    }
    child_kinds = [kind for kind in matched if kind != "root_retained_spawn"]
    if child_kinds:
        result["child"] = {
            "target_node_index": matched[child_kinds[0]],
            "atoms": [child_kinds[0]],
        }
    scaffold = (trace.get("info") or {}).get("natural_yield_scaffold") or {}
    spawn_index = scaffold.get("spawn_node_index")
    yield_index = _native_yield_node(trace, spawn_index) if isinstance(spawn_index, int) else None
    if (
        yield_index is not None
        and float(metrics.get("local_work_before_yield", 0.0)) == 1.0
        and float(metrics.get("forbidden_post_spawn_tool_before_child", 0.0)) == 0.0
    ):
        result["coordinator"] = {
            "target_node_index": yield_index,
            "atoms": ["root_retained_spawn", "passive_yield"],
        }
    return result


def summarize(
    path: Path | list[Path],
    *,
    phase: str = "e0_full_actions",
    rescore: bool = True,
) -> dict[str, Any]:
    paths = [path] if isinstance(path, Path) else path
    rows = [row for source in paths for row in _traces(source)]
    if rescore:
        rows = [_rescore(row) for row in rows]
    rejection_counts: Counter[str] = Counter()
    qualifying: list[dict[str, Any]] = []
    positive_prefixes: list[dict[str, Any]] = []
    for row in rows:
        reasons = _qualify(row, phase)
        if reasons:
            rejection_counts.update(reasons)
            audit = positive_prefix_audit(row, phase)
            if audit:
                task = row.get("task", {}).get("data", {})
                positive_prefixes.append(
                    {
                        "task_key": task.get("episode_id") or task.get("name"),
                        "trace_id": row.get("id"),
                        "roles": audit,
                    }
                )
            continue
        task = row.get("task", {}).get("data", {})
        qualifying.append(
            {
                "task_key": task.get("episode_id") or task.get("name"),
                "trace_id": row.get("id"),
            }
        )
    task_keys = [item["task_key"] for item in qualifying]
    distinct_task_keys = len(set(task_keys))
    gate_open = len(qualifying) >= 4 and distinct_task_keys >= 4
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "episodes": len(rows),
        "qualifying_trajectories": len(qualifying),
        "distinct_qualifying_task_keys": distinct_task_keys,
        "qualifying": qualifying,
        "positive_prefixes": positive_prefixes,
        "positive_prefix_rows_by_role": {
            role: sum(role in item["roles"] for item in positive_prefixes)
            for role in ("coordinator", "child")
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "gate": {
            "required_qualifying_trajectories": 4,
            "required_distinct_task_keys": 4,
            "acceptance_floor_relaxed": False,
            "gradient_gate_open": gate_open,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="+")
    parser.add_argument(
        "--phase",
        choices=tuple(EXPECTED_EVENT_KINDS),
        default="e0_full_actions",
    )
    parser.add_argument("--no-rescore", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(
        args.path,
        phase=args.phase,
        rescore=not args.no_rescore,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
