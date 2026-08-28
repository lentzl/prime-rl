#!/usr/bin/env python3
"""Append-only controller for a dual-policy dense Qwen3.5-2B SPADE loop."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVENT_SCHEMA_VERSION = "qwen35-2b-spade-dual-dense-event/v1"
STATUS_SCHEMA_VERSION = "qwen35-2b-spade-dual-dense-status/v1"
UPDATE_SCHEMA_VERSION = "qwen35-2b-spade-dual-dense-update/v1"
SUMMARY_SCHEMA_VERSION = "qwen35-2b-interaction-curriculum-summary/v1"
MINIMUM_QUALIFIERS = 4
PROMOTION_MINIMUM_QUALIFIERS = 4
ROLE_FOR_TRACK = {"child": "child", "yield": "coordinator"}
TRANSITION_KINDS = {
    "update_pair_recorded",
    "update_roles_recorded",
    "candidate_pair_selected",
    "candidate_roles_selected",
}
EVALUATION_ATTEMPT_KINDS = {"evaluation_recorded", "evaluation_aborted"}
PHASE_LADDERS = {
    "child": (
        "e0_full_actions",
        "e0c_natural_child",
        "e0c2_natural_child_no_template",
        "e0c25_inline_evidence",
        "e0c275_inline_location",
        "e0c28_inline_only",
        "e0c29_evidence_available",
        "e0c3_natural_child_minimal",
    ),
    "yield": (
        "e0d2_capped_yield_exact_child",
        "e0d3_uncapped_yield_exact_child",
        "e0d3_uncapped_yield",
    ),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _event_digest(event: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(event).encode()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"event log does not exist: {path}")
    events = []
    previous = None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            digest = event.pop("event_sha256", None)
            if event.get("schema_version") != EVENT_SCHEMA_VERSION:
                raise ValueError(f"{path}:{line_number} has an unsupported schema")
            if event.get("sequence") != len(events):
                raise ValueError(f"{path}:{line_number} has a broken sequence")
            if event.get("previous_event_sha256") != previous:
                raise ValueError(f"{path}:{line_number} has a broken hash chain")
            if digest != _event_digest(event):
                raise ValueError(f"{path}:{line_number} has an invalid digest")
            event["event_sha256"] = digest
            events.append(event)
            previous = digest
    if not events or events[0].get("kind") != "initialized":
        raise ValueError("event log must begin with initialized")
    return events


def _append_event(path: Path, payload: dict[str, Any], *, create: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x+" if create else "a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        existing = [json.loads(line) for line in handle if line.strip()]
        event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "sequence": len(existing),
            "previous_event_sha256": existing[-1]["event_sha256"] if existing else None,
            **payload,
        }
        event["event_sha256"] = _event_digest(event)
        handle.seek(0, 2)
        handle.write(_canonical_json(event) + "\n")
        handle.flush()


def _phase_index(track: str, phase: str) -> int:
    try:
        return PHASE_LADDERS[track].index(phase)
    except (KeyError, ValueError) as error:
        raise ValueError(f"phase {phase!r} is not on the {track!r} ladder") from error


def _cycle_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last_transition = max(
        (index for index, event in enumerate(events) if event["kind"] in TRANSITION_KINDS),
        default=0,
    )
    return events[last_transition + 1 :]


def _trainable_roles(events: list[dict[str, Any]]) -> tuple[str, ...]:
    roles = tuple(events[0].get("trainable_roles") or ("coordinator", "child"))
    if not roles or len(set(roles)) != len(roles) or set(roles) - {"coordinator", "child"}:
        raise ValueError("trainable_roles must be a non-empty unique subset of the two roles")
    return roles


def _trainable_tracks(events: list[dict[str, Any]]) -> tuple[str, ...]:
    roles = set(_trainable_roles(events))
    return tuple(track for track, role in ROLE_FOR_TRACK.items() if role in roles)


def _admission_policy(events: list[dict[str, Any]]) -> dict[str, Any]:
    changed = next(
        (event for event in reversed(events) if event["kind"] == "admission_policy_changed"),
        None,
    )
    if changed is None:
        return {
            "minimum_training_qualifiers": MINIMUM_QUALIFIERS,
            "minimum_promotion_qualifiers": PROMOTION_MINIMUM_QUALIFIERS,
            "failed_trajectory_rows_trainable": False,
            "aggressive_frontier": False,
        }
    return {
        "minimum_training_qualifiers": changed["minimum_training_qualifiers"],
        "minimum_promotion_qualifiers": changed["minimum_promotion_qualifiers"],
        "failed_trajectory_rows_trainable": changed.get("failed_trajectory_rows_trainable", False),
        "aggressive_frontier": changed.get("aggressive_frontier", False),
    }


def _candidate_pair(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    candidates = events[0]["candidates"]
    for event in events:
        if event["kind"] in {"update_pair_recorded", "update_roles_recorded"}:
            candidates = event["output_candidates"]
        elif event["kind"] in {"candidate_pair_selected", "candidate_roles_selected"}:
            candidates = event["selected_candidates"]
    return candidates


def _pair_signature(candidates: dict[str, dict[str, Any]]) -> tuple[str, str]:
    return tuple(candidates[role]["model_sha256"] for role in ("coordinator", "child"))


def _previous_viable_role_candidate(
    events: list[dict[str, Any]], *, role: str, current: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    rejected = {
        event["input_candidate_sha256"][role]
        for event in events
        if event["kind"] in {"candidate_pair_selected", "candidate_roles_selected"}
        and event["input_candidate_sha256"][role] != event["selected_candidates"][role]["model_sha256"]
    }
    history = [events[0]["candidates"]]
    history.extend(
        event[
            "output_candidates"
            if event["kind"] in {"update_pair_recorded", "update_roles_recorded"}
            else "selected_candidates"
        ]
        for event in events
        if event["kind"] in TRANSITION_KINDS
    )
    current_sha256 = current[role]["model_sha256"]
    for candidates in reversed(history[:-1]):
        candidate = candidates[role]
        if candidate["model_sha256"] != current_sha256 and candidate["model_sha256"] not in rejected:
            return candidate
    return None


def _previous_viable_candidate_pair(
    events: list[dict[str, Any]], *, current: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]] | None:
    previous = {
        role: _previous_viable_role_candidate(events, role=role, current=current) for role in ("coordinator", "child")
    }
    if any(candidate is None for candidate in previous.values()):
        return None
    return {role: candidate for role, candidate in previous.items() if candidate is not None}


def _cycle_targets(events: list[dict[str, Any]]) -> dict[str, str]:
    targets = dict(events[0]["initial_targets"])
    for event in events:
        if event["kind"] in {"update_pair_recorded", "update_roles_recorded"}:
            for track, source in event["training_sources"].items():
                hold_phase_for_designer_step = False
                if track == "yield":
                    replay = event["output_candidates"]["coordinator"].get("replay") or {}
                    designer = replay.get("environment_designer")
                    hold_phase_for_designer_step = isinstance(designer, dict) and (
                        designer.get("next_stage_index") != designer.get("trained_stage_index")
                    )
                targets[track] = (
                    source["phase"]
                    if hold_phase_for_designer_step
                    else PHASE_LADDERS[track][
                        min(
                            _phase_index(track, source["phase"]) + 1,
                            len(PHASE_LADDERS[track]) - 1,
                        )
                    ]
                )
        elif event["kind"] in {"candidate_pair_selected", "candidate_roles_selected"}:
            targets = dict(event["cycle_targets"])
    return targets


def _cycle_retention_floors(events: list[dict[str, Any]]) -> dict[str, str]:
    transition = next(
        (event for event in reversed(events) if event["kind"] in TRANSITION_KINDS),
        None,
    )
    if transition is None or transition["kind"] not in {"update_pair_recorded", "update_roles_recorded"}:
        return {}
    return {track: source["phase"] for track, source in transition["training_sources"].items()}


def _accepted_sources(cycle: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    accepted = {}
    for event in cycle:
        if event["kind"] != "evaluation_recorded" or not event["admission"]["gate_open"]:
            continue
        track = event["track"]
        current = accepted.get(track)
        if current is None or _phase_index(track, event["phase"]) > _phase_index(track, current["phase"]):
            accepted[track] = event
    return accepted


def _effective_cycle(cycle: list[dict[str, Any]]) -> list[dict[str, Any]]:
    invalidated = {event["bank_id"] for event in cycle if event["kind"] == "evaluation_invalidated"}
    return [
        event for event in cycle if event["kind"] != "evaluation_recorded" or event["bank"]["id"] not in invalidated
    ]


def _used_indices(events: list[dict[str, Any]]) -> set[int]:
    return {event["bank"]["start_index"] for event in events if event["kind"] in EVALUATION_ATTEMPT_KINDS}


def project(events: list[dict[str, Any]]) -> dict[str, Any]:
    initialized = events[0]
    candidates = _candidate_pair(events)
    cycle = _cycle_events(events)
    effective_cycle = _effective_cycle(cycle)
    targets = _cycle_targets(events)
    trainable_roles = _trainable_roles(events)
    trainable_tracks = _trainable_tracks(events)
    admission_policy = _admission_policy(events)
    retention_floors = _cycle_retention_floors(events)
    accepted = {
        track: source for track, source in _accepted_sources(effective_cycle).items() if track in trainable_tracks
    }
    latest = {event["track"]: event for event in effective_cycle if event["kind"] in EVALUATION_ATTEMPT_KINDS}
    attempts = {
        track: sum(event["kind"] in EVALUATION_ATTEMPT_KINDS and event["track"] == track for event in effective_cycle)
        for track in trainable_tracks
    }
    used = _used_indices(events)
    stride = initialized["bank_policy"]["index_stride"]
    cycle_base = max(used, default=initialized["bank_policy"]["next_start_index"] - stride) + stride

    floor_failures = {
        track: event
        for track, event in latest.items()
        if track not in accepted and event["phase"] == PHASE_LADDERS[track][0] and not event["admission"]["gate_open"]
    }
    retention_failures = {}
    retention_pending = {}
    for track, floor_phase in retention_floors.items():
        accepted_source = accepted.get(track)
        attempted = latest.get(track)
        floor_attempt = next(
            (
                event
                for event in reversed(effective_cycle)
                if event["kind"] in EVALUATION_ATTEMPT_KINDS
                and event["track"] == track
                and event["phase"] == floor_phase
            ),
            None,
        )
        if floor_attempt is not None and not floor_attempt["admission"]["gate_open"]:
            retention_failures[track] = floor_attempt
        elif (
            floor_attempt is None
            and accepted_source is not None
            and _phase_index(track, accepted_source["phase"]) >= _phase_index(track, targets[track])
        ):
            retention_pending[track] = floor_phase
        elif accepted_source is not None and _phase_index(track, accepted_source["phase"]) < _phase_index(
            track, floor_phase
        ):
            retention_failures[track] = accepted_source
        elif (
            accepted_source is None
            and attempted is not None
            and _phase_index(track, attempted["phase"]) <= _phase_index(track, floor_phase)
            and not attempted["admission"]["gate_open"]
        ):
            retention_failures[track] = attempted
    rejection_failures = {**floor_failures, **retention_failures}
    failed_roles = {ROLE_FOR_TRACK[track] for track in rejection_failures}
    rollback = dict(candidates)
    for role in failed_roles:
        previous = _previous_viable_role_candidate(events, role=role, current=candidates)
        if previous is not None:
            rollback[role] = previous
    rollback_changes_roles = {
        role for role in failed_roles if rollback[role]["model_sha256"] != candidates[role]["model_sha256"]
    }

    if (
        rejection_failures
        and rollback_changes_roles == failed_roles
        and not admission_policy["aggressive_frontier"]
    ):
        next_action = {
            "kind": "select_roles",
            "full_optimizer_steps_authorized": {"child": 0, "coordinator": 0},
            "selected_candidates": rollback,
            "rejected_roles": sorted(failed_roles),
            "cycle_targets": targets,
            "rejection_evidence": {
                track: {
                    "bank_id": event["bank"]["id"],
                    "phase": event["phase"],
                    "qualifying_trajectories": event["admission"]["qualifying_trajectories"],
                    "distinct_qualifying_task_keys": event["admission"]["distinct_task_keys"],
                    "required_retention_phase": retention_floors.get(track),
                    "evidence_sha256": event["artifacts"].get("summary_sha256") or event["artifacts"]["traces_sha256"],
                }
                for track, event in sorted(rejection_failures.items())
            },
        }
        status = "candidate_rejected"
    elif set(accepted) == set(trainable_tracks) and not retention_pending:
        authorized_steps = {role: int(role in trainable_roles) for role in ("child", "coordinator")}
        next_action: dict[str, Any] = {
            "kind": "train_pair" if len(trainable_roles) == 2 else "train_roles",
            "full_optimizer_steps_authorized": authorized_steps,
            "lora_updates_authorized": 0,
            "failed_trajectory_rows_trainable": admission_policy["failed_trajectory_rows_trainable"],
            "sources": {
                track: {
                    "phase": source["phase"],
                    "bank_id": source["bank"]["id"],
                    "qualifying_trajectories": source["admission"]["qualifying_trajectories"],
                    "distinct_qualifying_task_keys": source["admission"]["distinct_task_keys"],
                    "summary_sha256": source["artifacts"]["summary_sha256"],
                }
                for track, source in sorted(accepted.items())
            },
        }
        status = "training_authorized"
    else:
        arms = []
        for track_index, track in enumerate(trainable_tracks):
            if track in accepted and track not in retention_pending:
                continue
            attempted = latest.get(track)
            if track in retention_pending:
                phase = retention_pending[track]
                reason = "verify_exact_parent_retention_rung"
            elif attempted is None:
                phase = targets[track]
                reason = "evaluate_current_target"
            else:
                phase_index = _phase_index(track, attempted["phase"])
                phase = PHASE_LADDERS[track][max(0, phase_index - 1)]
                reason = "increase_environment_help_after_failed_admission"
            start_index = cycle_base + (attempts[track] * len(PHASE_LADDERS) + track_index) * stride
            if start_index in used:
                raise ValueError("derived next bank collides with a recorded start index")
            arms.append(
                {
                    "track": track,
                    "role_model": ROLE_FOR_TRACK[track],
                    "phase": phase,
                    "reason": reason,
                    "start_index": start_index,
                    "tasks": initialized["bank_policy"]["tasks_per_bank"],
                    "split": "train_gen",
                    "optimizer_updates_during_collection": 0,
                }
            )
        next_action = {
            "kind": "collect",
            "full_optimizer_steps_authorized": {"child": 0, "coordinator": 0},
            "arms": arms,
        }
        status = "collecting"

    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": status,
        "model_revision": initialized["model_revision"],
        "candidates": candidates,
        "invariants": initialized["invariants"],
        "admission_policy": admission_policy,
        "trainable_roles": list(trainable_roles),
        "cycle_targets": targets,
        "cycle_retention_floors": retention_floors,
        "accepted_sources": {
            track: {
                "phase": source["phase"],
                "bank_id": source["bank"]["id"],
                "qualifying_trajectories": source["admission"]["qualifying_trajectories"],
                "distinct_qualifying_task_keys": source["admission"]["distinct_task_keys"],
            }
            for track, source in sorted(accepted.items())
        },
        "event_count": len(events),
        "event_head_sha256": events[-1]["event_sha256"],
        "next_action": next_action,
    }


def _verified_dense_candidate(path: Path, sha256: str, *, label: str, model: str) -> dict[str, Any]:
    weight = path / "model.safetensors"
    if not label or not model:
        raise ValueError("dense candidate label and model name must be non-empty")
    if not path.is_absolute() or not (path / "STABLE").is_file() or not weight.is_file():
        raise ValueError(f"dense candidate is incomplete: {path}")
    if _sha256_file(weight) != sha256:
        raise ValueError(f"dense candidate SHA-256 mismatch: {path}")
    return {
        "label": label,
        "model": model,
        "model_path": str(path.resolve()),
        "model_sha256": sha256,
    }


def _validated_summary(path: Path, *, phase: str) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    gate = summary.get("gate") or {}
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION or summary.get("phase") != phase:
        raise ValueError("evaluation summary schema or phase mismatch")
    if (
        gate.get("required_qualifying_trajectories") != PROMOTION_MINIMUM_QUALIFIERS
        or gate.get("required_distinct_task_keys") != PROMOTION_MINIMUM_QUALIFIERS
        or gate.get("acceptance_floor_relaxed") is not False
    ):
        raise ValueError("evaluation summary changes the four-trajectory floor")
    qualifying = summary.get("qualifying_trajectories")
    distinct = summary.get("distinct_qualifying_task_keys")
    if not isinstance(qualifying, int) or not isinstance(distinct, int):
        raise ValueError("evaluation summary qualifier counts must be integers")
    computed = qualifying >= PROMOTION_MINIMUM_QUALIFIERS and distinct >= PROMOTION_MINIMUM_QUALIFIERS
    if gate.get("gradient_gate_open") is not computed:
        raise ValueError("evaluation summary gate is inconsistent")
    return summary


def _provider_error_count(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        for trace in episode.get("traces") or []:
            for call in trace.get("calls") or []:
                error = call.get("error")
                if isinstance(error, dict) and error.get("type") == "ProviderError":
                    count += 1
    return count


def _completed_trace_outcomes(path: Path, *, start_index: int) -> list[tuple[str, bool]]:
    outcomes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        episode = json.loads(line)
        traces = episode.get("traces") or []
        if episode.get("ok") is not True or episode.get("errors") or len(traces) != 1:
            raise ValueError(f"partial trace {line_number} is not a complete error-free episode")
        trace = traces[0]
        if trace.get("ok") is not True or trace.get("is_completed") is not True or trace.get("errors"):
            raise ValueError(f"partial trace {line_number} is not a complete error-free trajectory")
        task = (trace.get("task") or {}).get("data") or {}
        task_offset = task.get("idx")
        task_key = task.get("name")
        if (
            not isinstance(task_offset, int)
            or task_offset < 0
            or not isinstance(task_key, str)
            or f"{start_index + task_offset:08d}" not in task_key
        ):
            raise ValueError(f"partial trace {line_number} does not match the planned task bank")
        score = ((trace.get("rewards") or {}).get("harness_score") or {}).get("score")
        if score not in (0, 1):
            raise ValueError(f"partial trace {line_number} lacks a binary harness score")
        outcomes.append((task_key, score == 1))
    if len({task_key for task_key, _ in outcomes}) != len(outcomes):
        raise ValueError("partial traces contain duplicate task keys")
    return outcomes


def _init(args: argparse.Namespace) -> None:
    for track, phase in (("child", args.child_target), ("yield", args.yield_target)):
        _phase_index(track, phase)
    candidates = {
        "coordinator": _verified_dense_candidate(
            args.coordinator_path.resolve(),
            args.coordinator_sha256,
            label=args.coordinator_label,
            model=args.coordinator_model,
        ),
        "child": _verified_dense_candidate(
            args.child_path.resolve(),
            args.child_sha256,
            label=args.child_label,
            model=args.child_model,
        ),
    }
    payload = {
        "kind": "initialized",
        "recorded_at_utc": args.recorded_at or _now(),
        "model_revision": args.model_revision,
        "candidates": candidates,
        "trainable_roles": list(args.train_role or ("coordinator", "child")),
        "initial_targets": {"child": args.child_target, "yield": args.yield_target},
        "bank_policy": {
            "next_start_index": args.next_start_index,
            "index_stride": args.index_stride,
            "tasks_per_bank": args.tasks_per_bank,
        },
        "invariants": {
            "minimum_complete_qualifying_trajectories_per_source": MINIMUM_QUALIFIERS,
            "minimum_distinct_qualifying_task_keys_per_source": MINIMUM_QUALIFIERS,
            "acceptance_floor_relaxed": False,
            "failed_trajectory_rows_trainable": False,
            "maximum_full_optimizer_steps_per_role_per_cycle": 1,
            "lora_updates_authorized": 0,
            "heldout_training_allowed": False,
            "role_models_are_distinct_after_first_update": True,
        },
    }
    _append_event(args.events, payload, create=True)
    print(json.dumps(project(_load_events(args.events)), indent=2, sort_keys=True))


def _record_evaluation(args: argparse.Namespace) -> None:
    events = _load_events(args.events)
    status = project(events)
    action = status["next_action"]
    expected = next((arm for arm in action.get("arms", []) if arm["track"] == args.track), None)
    if (
        action["kind"] != "collect"
        or expected is None
        or expected["phase"] != args.phase
        or expected["start_index"] != args.start_index
    ):
        raise ValueError("evaluation does not match the controller's planned arm")
    summary = _validated_summary(args.summary, phase=args.phase)
    policy = _admission_policy(events)
    training_minimum = policy["minimum_training_qualifiers"]
    role = ROLE_FOR_TRACK[args.track]
    positive_prefix_rows = (summary.get("positive_prefix_rows_by_role") or {}).get(role, 0)
    if not isinstance(positive_prefix_rows, int) or positive_prefix_rows < 0:
        raise ValueError("evaluation summary has an invalid positive-prefix count")
    training_gate_open = (
        summary["qualifying_trajectories"] >= training_minimum
        and summary["distinct_qualifying_task_keys"] >= training_minimum
    ) or (policy["failed_trajectory_rows_trainable"] and positive_prefix_rows >= 1)
    versions = args.versions.read_text(encoding="utf-8")
    for role, candidate in status["candidates"].items():
        if f"{role}_model_sha256={candidate['model_sha256']}" not in versions:
            raise ValueError(f"VERSIONS.txt lacks the frozen {role} model SHA-256")
    if f"interaction_curriculum={args.phase}" not in versions:
        raise ValueError("VERSIONS.txt lacks the interaction curriculum phase")
    if not args.traces.is_file() or not args.bootstrap.is_file() or not args.routing_audit.is_file():
        raise FileNotFoundError("evaluation lacks traces, bootstrap, or routing audit")
    if _provider_error_count(args.traces):
        raise ValueError("evaluation traces contain provider errors")
    trace_count = sum(1 for line in args.traces.read_text(encoding="utf-8").splitlines() if line.strip())
    if summary.get("episodes") != expected["tasks"] or trace_count != expected["tasks"]:
        raise ValueError("evaluation artifacts do not contain the planned episode count")
    routing = [json.loads(line) for line in args.routing_audit.read_text().splitlines() if line.strip()]
    role_counts = {
        role: sum(
            event.get("schema_version") == "qwen35-2b-dual-policy-route/v1"
            and event.get("role") == role
            and event.get("upstream_model") == status["candidates"][role]["model_path"]
            and event.get("status") == 200
            for event in routing
        )
        for role in ("coordinator", "child")
    }
    if any(count < 1 for count in role_counts.values()):
        raise ValueError("routing audit does not prove both role models served requests")
    bank_id = args.bank_id or f"{args.track}-{args.phase}-{args.start_index}"
    if any(event["kind"] == "evaluation_recorded" and event["bank"]["id"] == bank_id for event in events):
        raise ValueError(f"bank id has already been recorded: {bank_id}")
    payload = {
        "kind": "evaluation_recorded",
        "recorded_at_utc": args.recorded_at or _now(),
        "candidate_model_sha256": {role: candidate["model_sha256"] for role, candidate in status["candidates"].items()},
        "track": args.track,
        "phase": args.phase,
        "bank": {
            "id": bank_id,
            "start_index": args.start_index,
            "tasks": summary["episodes"],
            "split": "train_gen",
        },
        "admission": {
            "qualifying_trajectories": summary["qualifying_trajectories"],
            "distinct_task_keys": summary["distinct_qualifying_task_keys"],
            "gate_open": training_gate_open,
            "training_minimum": training_minimum,
            "promotion_gate_open": summary["gate"]["gradient_gate_open"],
            "promotion_minimum": policy["minimum_promotion_qualifiers"],
            "acceptance_floor_relaxed": training_minimum < policy["minimum_promotion_qualifiers"],
            "positive_prefix_trajectories": positive_prefix_rows,
            "positive_prefix_rows_trainable": policy["failed_trajectory_rows_trainable"],
        },
        "routing": {
            "audit_path": str(args.routing_audit.resolve()),
            "audit_sha256": _sha256_file(args.routing_audit),
            "successful_requests_by_role": role_counts,
        },
        "artifacts": {
            "summary_path": str(args.summary.resolve()),
            "summary_sha256": _sha256_file(args.summary),
            "versions_path": str(args.versions.resolve()),
            "versions_sha256": _sha256_file(args.versions),
            "traces_path": str(args.traces.resolve()),
            "traces_sha256": _sha256_file(args.traces),
            "bootstrap_path": str(args.bootstrap.resolve()),
            "bootstrap_sha256": _sha256_file(args.bootstrap),
        },
    }
    _append_event(args.events, payload)
    print(json.dumps(project(_load_events(args.events)), indent=2, sort_keys=True))


def _invalidate_evaluation(args: argparse.Namespace) -> None:
    events = _load_events(args.events)
    matches = [
        event for event in events if event["kind"] == "evaluation_recorded" and event["bank"]["id"] == args.bank_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one recorded evaluation: {args.bank_id}")
    if any(event["kind"] == "evaluation_invalidated" and event["bank_id"] == args.bank_id for event in events):
        raise ValueError(f"evaluation is already invalidated: {args.bank_id}")
    target = matches[0]
    traces = Path(target["artifacts"]["traces_path"])
    provider_errors = _provider_error_count(traces)
    if provider_errors < 1:
        raise ValueError("evaluation invalidation requires a recorded provider error")
    payload = {
        "kind": "evaluation_invalidated",
        "recorded_at_utc": args.recorded_at or _now(),
        "bank_id": args.bank_id,
        "target_event_sha256": target["event_sha256"],
        "reason": "provider_error",
        "provider_error_count": provider_errors,
        "traces_path": str(traces.resolve()),
        "traces_sha256": _sha256_file(traces),
    }
    _append_event(args.events, payload)
    print(json.dumps(project(_load_events(args.events)), indent=2, sort_keys=True))


def _abort_evaluation(args: argparse.Namespace) -> None:
    events = _load_events(args.events)
    status = project(events)
    action = status["next_action"]
    expected = next((arm for arm in action.get("arms", []) if arm["track"] == args.track), None)
    if (
        action["kind"] != "collect"
        or expected is None
        or expected["phase"] != args.phase
        or expected["start_index"] != args.start_index
    ):
        raise ValueError("aborted evaluation does not match the controller's planned arm")
    for required in (args.traces, args.versions, args.bootstrap, args.routing_audit):
        if not required.is_file():
            raise FileNotFoundError(f"aborted evaluation lacks evidence: {required}")
    if f"interaction_curriculum={args.phase}" not in args.versions.read_text(encoding="utf-8"):
        raise ValueError("VERSIONS.txt lacks the interaction curriculum phase")
    if _provider_error_count(args.traces):
        raise ValueError("aborted evaluation traces contain provider errors")
    outcomes = _completed_trace_outcomes(args.traces, start_index=args.start_index)
    if not outcomes or len(outcomes) >= expected["tasks"]:
        raise ValueError("aborted evaluation must contain a non-empty partial task bank")
    qualifying_keys = {task_key for task_key, passed in outcomes if passed}
    remaining = expected["tasks"] - len(outcomes)
    maximum_qualifying = len(qualifying_keys) + remaining
    training_minimum = _admission_policy(events)["minimum_training_qualifiers"]
    if maximum_qualifying >= training_minimum:
        raise ValueError("partial task bank has not mathematically closed the admission gate")
    routing = [json.loads(line) for line in args.routing_audit.read_text().splitlines() if line.strip()]
    role_counts = {
        role: sum(
            event.get("schema_version") == "qwen35-2b-dual-policy-route/v1"
            and event.get("role") == role
            and event.get("upstream_model") == status["candidates"][role]["model_path"]
            and event.get("status") == 200
            for event in routing
        )
        for role in ("coordinator", "child")
    }
    if any(count < 1 for count in role_counts.values()):
        raise ValueError("routing audit does not prove both role models served requests")
    bank_id = args.bank_id or f"{args.track}-{args.phase}-{args.start_index}-aborted"
    if any(event["kind"] in EVALUATION_ATTEMPT_KINDS and event["bank"]["id"] == bank_id for event in events):
        raise ValueError(f"bank id has already been recorded: {bank_id}")
    payload = {
        "kind": "evaluation_aborted",
        "recorded_at_utc": args.recorded_at or _now(),
        "candidate_model_sha256": {role: candidate["model_sha256"] for role, candidate in status["candidates"].items()},
        "track": args.track,
        "phase": args.phase,
        "bank": {
            "id": bank_id,
            "start_index": args.start_index,
            "tasks": expected["tasks"],
            "completed_tasks": len(outcomes),
            "split": "train_gen",
        },
        "admission": {
            "qualifying_trajectories": len(qualifying_keys),
            "distinct_task_keys": len(qualifying_keys),
            "maximum_possible_qualifying_trajectories": maximum_qualifying,
            "gate_open": False,
            "gate_mathematically_closed": True,
            "training_minimum": training_minimum,
            "promotion_gate_open": False,
            "promotion_minimum": PROMOTION_MINIMUM_QUALIFIERS,
            "acceptance_floor_relaxed": training_minimum < PROMOTION_MINIMUM_QUALIFIERS,
        },
        "abort": {"reason": args.reason, "remaining_tasks": remaining},
        "routing": {
            "audit_path": str(args.routing_audit.resolve()),
            "audit_sha256": _sha256_file(args.routing_audit),
            "successful_requests_by_role": role_counts,
        },
        "artifacts": {
            "versions_path": str(args.versions.resolve()),
            "versions_sha256": _sha256_file(args.versions),
            "traces_path": str(args.traces.resolve()),
            "traces_sha256": _sha256_file(args.traces),
            "bootstrap_path": str(args.bootstrap.resolve()),
            "bootstrap_sha256": _sha256_file(args.bootstrap),
        },
    }
    _append_event(args.events, payload)
    print(json.dumps(project(_load_events(args.events)), indent=2, sort_keys=True))


def _verified_replay(value: dict[str, Any], *, role: str) -> dict[str, Any]:
    path = Path(value.get("path", ""))
    manifest = path / "MANIFEST.json"
    parquet = path / "train.parquet"
    if not path.is_absolute() or not manifest.is_file() or not parquet.is_file():
        raise ValueError(f"{role} replay corpus is incomplete")
    if (
        _sha256_file(manifest) != value.get("manifest_sha256")
        or _sha256_file(parquet) != value.get("train_parquet_sha256")
        or value.get("role") != role
        or not isinstance(value.get("rows"), int)
        or value["rows"] < 1
        or not isinstance(value.get("new_rows"), int)
        or not 1 <= value["new_rows"] <= PROMOTION_MINIMUM_QUALIFIERS
        or not isinstance(value.get("new_partial_rows", 0), int)
        or not 0 <= value.get("new_partial_rows", 0) <= 8
    ):
        raise ValueError(f"{role} replay corpus receipt is invalid")
    return value


def _set_admission_policy(args: argparse.Namespace) -> None:
    _load_events(args.events)
    if not 1 <= args.training_minimum <= args.promotion_minimum:
        raise ValueError("training admission must be between one and the promotion minimum")
    if args.promotion_minimum != PROMOTION_MINIMUM_QUALIFIERS:
        raise ValueError("canonical promotion remains fixed at four qualifiers")
    payload = {
        "kind": "admission_policy_changed",
        "recorded_at_utc": args.recorded_at or _now(),
        "minimum_training_qualifiers": args.training_minimum,
        "minimum_promotion_qualifiers": args.promotion_minimum,
        "failed_trajectory_rows_trainable": args.failed_trajectories_trainable,
        "aggressive_frontier": args.aggressive_frontier,
        "reason": args.reason,
    }
    _append_event(args.events, payload)
    print(json.dumps(project(_load_events(args.events)), indent=2, sort_keys=True))


def _record_update(args: argparse.Namespace) -> None:
    events = _load_events(args.events)
    status = project(events)
    action = status["next_action"]
    if action.get("kind") not in {"train_pair", "train_roles"}:
        raise ValueError("dense role training is not authorized")
    optimizer_steps = action.get("full_optimizer_steps_authorized")
    if optimizer_steps not in (
        {"child": 1, "coordinator": 1},
        {"child": 0, "coordinator": 1},
        {"child": 1, "coordinator": 0},
    ):
        raise ValueError("dense role training authorization is invalid")
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != UPDATE_SCHEMA_VERSION:
        raise ValueError("unsupported update receipt schema")
    inputs = {role: value["model_sha256"] for role, value in status["candidates"].items()}
    if (
        receipt.get("input_model_sha256") != inputs
        or receipt.get("optimizer_steps") != optimizer_steps
        or receipt.get("dense_model_updates") != optimizer_steps
        or receipt.get("lora_updates") != 0
        or not isinstance(receipt.get("failed_trajectory_rows"), int)
        or receipt["failed_trajectory_rows"] < 0
    ):
        raise ValueError("update receipt violates the authorized dense role update")
    expected_sources = {track: source["summary_sha256"] for track, source in action["sources"].items()}
    if receipt.get("source_summary_sha256") != expected_sources:
        raise ValueError("update receipt does not use exactly the admitted sources")
    outputs = dict(status["candidates"])
    updated_roles = tuple(role for role in ("coordinator", "child") if optimizer_steps[role] == 1)
    if set(receipt.get("outputs", {})) != set(updated_roles):
        raise ValueError("update receipt output roles do not match authorization")
    for role in updated_roles:
        output = receipt.get("outputs", {}).get(role, {})
        candidate = _verified_dense_candidate(
            Path(output.get("model_path", "")),
            output.get("model_sha256", ""),
            label=output.get("label", ""),
            model=output.get("model", ""),
        )
        if candidate["model_sha256"] == inputs[role]:
            raise ValueError(f"{role} dense update did not change model weights")
        candidate["replay"] = _verified_replay(output.get("replay", {}), role=role)
        outputs[role] = candidate
    failed_rows = sum(outputs[role]["replay"].get("new_partial_rows", 0) for role in updated_roles)
    if (
        receipt["failed_trajectory_rows"] != failed_rows
        or (failed_rows > 0 and not action.get("failed_trajectory_rows_trainable"))
    ):
        raise ValueError("update receipt has unauthorized positive-prefix rows")
    if outputs["child"]["model_sha256"] == outputs["coordinator"]["model_sha256"]:
        raise ValueError("role-specific dense updates produced identical models")
    payload = {
        "kind": "update_pair_recorded" if len(updated_roles) == 2 else "update_roles_recorded",
        "recorded_at_utc": args.recorded_at or _now(),
        "receipt_path": str(args.receipt.resolve()),
        "receipt_sha256": _sha256_file(args.receipt),
        "training_sources": action["sources"],
        "output_candidates": outputs,
        "updated_roles": list(updated_roles),
    }
    _append_event(args.events, payload)
    print(json.dumps(project(_load_events(args.events)), indent=2, sort_keys=True))


def _record_selection(args: argparse.Namespace) -> None:
    events = _load_events(args.events)
    status = project(events)
    action = status["next_action"]
    if action.get("kind") not in {"select_pair", "select_roles"} or action.get("full_optimizer_steps_authorized") != {
        "child": 0,
        "coordinator": 0,
    }:
        raise ValueError("candidate role selection is not authorized")
    selected = action["selected_candidates"]
    if selected["child"]["model_sha256"] == selected["coordinator"]["model_sha256"]:
        raise ValueError("candidate selection would collapse the two role models")
    payload = {
        "kind": "candidate_roles_selected",
        "recorded_at_utc": args.recorded_at or _now(),
        "input_candidate_sha256": {role: candidate["model_sha256"] for role, candidate in status["candidates"].items()},
        "selected_candidates": selected,
        "cycle_targets": action["cycle_targets"],
        "rejection_evidence": action["rejection_evidence"],
        "rejected_roles": action.get("rejected_roles", ["child", "coordinator"]),
        "optimizer_steps": {"child": 0, "coordinator": 0},
    }
    _append_event(args.events, payload)
    print(json.dumps(project(_load_events(args.events)), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--events", type=Path, required=True)
    init.add_argument("--model-revision", required=True)
    init.add_argument(
        "--train-role",
        action="append",
        choices=("coordinator", "child"),
        help="role to update; repeat for both (default: both)",
    )
    for role in ("coordinator", "child"):
        init.add_argument(f"--{role}-path", type=Path, required=True)
        init.add_argument(f"--{role}-sha256", required=True)
        init.add_argument(f"--{role}-label", required=True)
        init.add_argument(f"--{role}-model", required=True)
    init.add_argument("--child-target", default="e0c_natural_child")
    init.add_argument("--yield-target", default="e0d3_uncapped_yield_exact_child")
    init.add_argument("--next-start-index", type=int, default=4_020_000)
    init.add_argument("--index-stride", type=int, default=100)
    init.add_argument("--tasks-per-bank", type=int, default=6)
    init.add_argument("--recorded-at")
    init.set_defaults(func=_init)

    record_eval = commands.add_parser("record-eval")
    record_eval.add_argument("--events", type=Path, required=True)
    record_eval.add_argument("--track", choices=tuple(PHASE_LADDERS), required=True)
    record_eval.add_argument("--phase", required=True)
    record_eval.add_argument("--start-index", type=int, required=True)
    record_eval.add_argument("--bank-id")
    record_eval.add_argument("--summary", type=Path, required=True)
    record_eval.add_argument("--versions", type=Path, required=True)
    record_eval.add_argument("--traces", type=Path, required=True)
    record_eval.add_argument("--bootstrap", type=Path, required=True)
    record_eval.add_argument("--routing-audit", type=Path, required=True)
    record_eval.add_argument("--recorded-at")
    record_eval.set_defaults(func=_record_evaluation)

    update = commands.add_parser("record-update")
    update.add_argument("--events", type=Path, required=True)
    update.add_argument("--receipt", type=Path, required=True)
    update.add_argument("--recorded-at")
    update.set_defaults(func=_record_update)

    selection = commands.add_parser("record-selection")
    selection.add_argument("--events", type=Path, required=True)
    selection.add_argument("--recorded-at")
    selection.set_defaults(func=_record_selection)

    invalidate = commands.add_parser("invalidate-eval")
    invalidate.add_argument("--events", type=Path, required=True)
    invalidate.add_argument("--bank-id", required=True)
    invalidate.add_argument("--recorded-at")
    invalidate.set_defaults(func=_invalidate_evaluation)

    abort = commands.add_parser("abort-eval")
    abort.add_argument("--events", type=Path, required=True)
    abort.add_argument("--track", choices=tuple(PHASE_LADDERS), required=True)
    abort.add_argument("--phase", required=True)
    abort.add_argument("--start-index", type=int, required=True)
    abort.add_argument("--bank-id")
    abort.add_argument("--versions", type=Path, required=True)
    abort.add_argument("--traces", type=Path, required=True)
    abort.add_argument("--bootstrap", type=Path, required=True)
    abort.add_argument("--routing-audit", type=Path, required=True)
    abort.add_argument("--reason", required=True)
    abort.add_argument("--recorded-at")
    abort.set_defaults(func=_abort_evaluation)

    admission = commands.add_parser("set-admission-policy")
    admission.add_argument("--events", type=Path, required=True)
    admission.add_argument("--training-minimum", type=int, required=True)
    admission.add_argument(
        "--promotion-minimum",
        type=int,
        default=PROMOTION_MINIMUM_QUALIFIERS,
    )
    admission.add_argument("--reason", required=True)
    admission.add_argument("--failed-trajectories-trainable", action="store_true")
    admission.add_argument("--aggressive-frontier", action="store_true")
    admission.add_argument("--recorded-at")
    admission.set_defaults(func=_set_admission_policy)

    status = commands.add_parser("status")
    status.add_argument("--events", type=Path, required=True)
    status.set_defaults(
        func=lambda args: print(json.dumps(project(_load_events(args.events)), indent=2, sort_keys=True))
    )
    args = parser.parse_args()
    if hasattr(args, "tasks_per_bank") and args.tasks_per_bank < MINIMUM_QUALIFIERS:
        raise ValueError("a bank must contain at least four tasks")
    args.func(args)


if __name__ == "__main__":
    main()
