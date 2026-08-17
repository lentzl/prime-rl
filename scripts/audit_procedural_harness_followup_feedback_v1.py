"""Audit failure-local follow-up feedback before an SDPO weight update."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from procedural_harness_master_v1.followup_feedback import (
    FEEDBACK_SCHEMA_VERSION,
    feedback_contract_payload,
)
from procedural_harness_master_v1.taskset import (
    ProceduralHarnessMasterData,
    _contract_behavior,
    _followup_feedback_diagnostic,
    keep_followup_feedback_response,
)

from prime_rl.orchestrator.trajectories import iter_trainable_branches

EXPECTED_EPISODES_PER_ADMISSION = 8
MIN_ACTIVE_FEEDBACK = 2


class FeedbackAuditFailure(ValueError):
    """The admission run does not prove safe failure-local routing."""


def _rows(path: Path) -> list[dict[str, Any]]:
    trace_path = path / "traces.jsonl" if path.is_dir() else path
    if not trace_path.is_file():
        raise FeedbackAuditFailure(f"missing traces: {trace_path}")
    rows: list[dict[str, Any]] = []
    for line in trace_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        nested = record.get("traces")
        rows.extend(nested if isinstance(nested, list) else [record])
    if len(rows) != EXPECTED_EPISODES_PER_ADMISSION:
        raise FeedbackAuditFailure(
            "expected "
            f"{EXPECTED_EPISODES_PER_ADMISSION} admission traces, found {len(rows)}"
        )
    return rows


def _selected_nodes(trace: vf.Trace, masks: list[list[bool]]) -> list[int]:
    selected: list[int] = []
    trainable_branches = list(iter_trainable_branches(trace))
    if len(masks) != len(trainable_branches):
        raise FeedbackAuditFailure(
            "feedback masks do not align with trainable trace branches"
        )
    for (branch, _), mask in zip(trainable_branches, masks, strict=True):
        if len(mask) != len(branch.token_ids):
            raise FeedbackAuditFailure("feedback mask does not span its branch")
        offset = 0
        for node in branch.nodes:
            node_mask = mask[offset : offset + len(node.token_ids)]
            offset += len(node.token_ids)
            if any(node_mask):
                selected.append(next(i for i, item in enumerate(trace.nodes) if item is node))
    return selected


def audit(paths: list[Path]) -> dict[str, Any]:
    rows = [row for path in paths for row in _rows(path)]
    trace_ids = [row.get("id") for row in rows]
    if len(set(trace_ids)) != len(trace_ids):
        raise FeedbackAuditFailure("admission inputs contain duplicate traces")
    active = 0
    codes: Counter[str] = Counter()
    selected_tokens = 0
    tokenized_targets = 0
    for row in rows:
        trace = vf.Trace.model_validate(row)
        if trace.errors:
            raise FeedbackAuditFailure(f"trace {trace.id} contains runtime errors")
        data = ProceduralHarnessMasterData.model_validate(row["task"]["data"])
        if data.family != "atomic_followup":
            raise FeedbackAuditFailure(f"trace {trace.id} is not atomic_followup")
        if _contract_behavior(trace, data)["harness_score"] == 1.0:
            if trace.info.get("feedback") or trace.info.get("feedback_contract"):
                raise FeedbackAuditFailure(f"successful trace {trace.id} carries feedback")
            continue

        diagnostic = _followup_feedback_diagnostic(trace, data)
        if diagnostic is None:
            if trace.info.get("feedback") or trace.info.get("feedback_contract"):
                raise FeedbackAuditFailure(f"unattributable trace {trace.id} carries feedback")
            continue
        expected = feedback_contract_payload(diagnostic)
        feedback = trace.info.get("feedback")
        contract = trace.info.get("feedback_contract")
        if contract != expected or feedback != expected["message"]:
            raise FeedbackAuditFailure(f"trace {trace.id} has untrusted feedback metadata")
        if contract.get("schema_version") != FEEDBACK_SCHEMA_VERSION:
            raise FeedbackAuditFailure(f"trace {trace.id} has the wrong feedback schema")
        if any(str(value) in feedback for value in data.oracle["final_answer"].values()):
            raise FeedbackAuditFailure(f"trace {trace.id} feedback leaks an answer value")

        target = trace.nodes[diagnostic.target_node_index]
        if not isinstance(target.message, vf.AssistantMessage) or not target.sampled:
            raise FeedbackAuditFailure(f"trace {trace.id} target is not a sampled response")
        trace_has_tokens = any(branch.token_ids for branch in trace.branches)
        if trace_has_tokens:
            masks = keep_followup_feedback_response(trace)
            selected = _selected_nodes(trace, masks)
            if selected != [diagnostic.target_node_index]:
                raise FeedbackAuditFailure(
                    f"trace {trace.id} selected {selected}, "
                    f"expected {[diagnostic.target_node_index]}"
                )
            target_tokens = sum(keep for branch_mask in masks for keep in branch_mask)
            if target_tokens <= 0 or not any(target.mask):
                raise FeedbackAuditFailure(f"trace {trace.id} target is not trainable")
            selected_tokens += target_tokens
            tokenized_targets += 1
        active += 1
        codes[diagnostic.code.value] += 1

    if active < MIN_ACTIVE_FEEDBACK:
        raise FeedbackAuditFailure(
            f"need at least {MIN_ACTIVE_FEEDBACK} feedback-bearing traces, found {active}"
        )
    return {
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "admissions": len(paths),
        "episodes": len(rows),
        "active_feedback_traces": active,
        "selected_tokens": selected_tokens if tokenized_targets else None,
        "tokenized_target_traces": tokenized_targets,
        "structural_routing_verified": True,
        "routing_contract": "one-mask-per-trainable-branch",
        "codes": dict(codes),
        "answer_free": True,
        "failure_local": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.paths)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
