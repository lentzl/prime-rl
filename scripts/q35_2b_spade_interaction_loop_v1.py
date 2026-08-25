#!/usr/bin/env python3
"""Append-only controller for the Qwen3.5-2B SPADE interaction loop."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVENT_SCHEMA_VERSION = "qwen35-2b-spade-interaction-loop-event/v1"
STATUS_SCHEMA_VERSION = "qwen35-2b-spade-interaction-loop-status/v1"
SUMMARY_SCHEMA_VERSION = "qwen35-2b-interaction-curriculum-summary/v1"
UPDATE_SCHEMA_VERSION = "qwen35-2b-spade-interaction-update/v1"
MINIMUM_QUALIFIERS = 4
DEFAULT_TASKS_PER_BANK = 6
DEFAULT_INDEX_STRIDE = 100

# Ordered from strongest environment help to the deployment-side boundary.
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


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"event log does not exist: {path}")
    events: list[dict[str, Any]] = []
    previous = None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            digest = event.pop("event_sha256", None)
            if event.get("schema_version") != EVENT_SCHEMA_VERSION:
                raise ValueError(f"{path}:{line_number} has an unsupported event schema")
            if event.get("sequence") != len(events):
                raise ValueError(f"{path}:{line_number} has a broken sequence")
            if event.get("previous_event_sha256") != previous:
                raise ValueError(f"{path}:{line_number} has a broken hash chain")
            if digest != _event_digest(event):
                raise ValueError(f"{path}:{line_number} has an invalid event digest")
            event["event_sha256"] = digest
            events.append(event)
            previous = digest
    if not events or events[0].get("kind") != "initialized":
        raise ValueError("event log must begin with one initialized event")
    return events


def _append_event(path: Path, payload: dict[str, Any], *, create: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x+" if create else "a+"
    with path.open(mode, encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        existing = [json.loads(line) for line in handle if line.strip()]
        if create and existing:
            raise FileExistsError(f"refusing to overwrite event log: {path}")
        previous = existing[-1]["event_sha256"] if existing else None
        event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "sequence": len(existing),
            "previous_event_sha256": previous,
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
    last_update = max(
        (index for index, event in enumerate(events) if event["kind"] == "update_recorded"),
        default=0,
    )
    return events[last_update + 1 :]


def _candidate(events: list[dict[str, Any]]) -> dict[str, Any]:
    initialized = events[0]
    candidate = initialized["candidate"]
    for event in events:
        if event["kind"] == "update_recorded":
            candidate = event["output_candidate"]
    return candidate


def _cycle_targets(events: list[dict[str, Any]]) -> dict[str, str]:
    targets = dict(events[0]["initial_targets"])
    for event in events:
        if event["kind"] != "update_recorded":
            continue
        targets = {
            track: PHASE_LADDERS[track][
                min(_phase_index(track, source["phase"]) + 1, len(PHASE_LADDERS[track]) - 1)
            ]
            for track, source in event["training_sources"].items()
        }
    return targets


def _accepted_sources(cycle: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    accepted: dict[str, dict[str, Any]] = {}
    for event in cycle:
        if event["kind"] != "evaluation_recorded" or not event["admission"]["gate_open"]:
            continue
        track = event["track"]
        current = accepted.get(track)
        if current is None or _phase_index(track, event["phase"]) > _phase_index(track, current["phase"]):
            accepted[track] = event
    return accepted


def _latest_attempts(cycle: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in cycle:
        if event["kind"] == "evaluation_recorded":
            latest[event["track"]] = event
    return latest


def _used_start_indices(events: list[dict[str, Any]]) -> set[int]:
    return {
        event["bank"]["start_index"]
        for event in events
        if event["kind"] == "evaluation_recorded"
    }


def _cycle_bank_base(events: list[dict[str, Any]], initialized_base: int, stride: int) -> int:
    last_update = max(
        (index for index, event in enumerate(events) if event["kind"] == "update_recorded"),
        default=None,
    )
    if last_update is None:
        return initialized_base
    past_indices = _used_start_indices(events[: last_update + 1])
    return max(past_indices, default=initialized_base - stride) + stride


def project(events: list[dict[str, Any]]) -> dict[str, Any]:
    initialized = events[0]
    candidate = _candidate(events)
    cycle = _cycle_events(events)
    targets = _cycle_targets(events)
    accepted = _accepted_sources(cycle)
    latest = _latest_attempts(cycle)
    used_indices = _used_start_indices(events)
    stride = initialized["bank_policy"]["index_stride"]
    cycle_base = _cycle_bank_base(
        events,
        initialized["bank_policy"]["next_start_index"],
        stride,
    )
    track_count = len(PHASE_LADDERS)
    attempts = {
        track: [
            event
            for event in cycle
            if event["kind"] == "evaluation_recorded" and event["track"] == track
        ]
        for track in PHASE_LADDERS
    }

    if set(accepted) == set(PHASE_LADDERS):
        next_action: dict[str, Any] = {
            "kind": "train",
            "optimizer_steps_authorized": 1,
            "dense_base_updates_authorized": 0,
            "failed_trajectory_rows_trainable": False,
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
        for track_index, track in enumerate(PHASE_LADDERS):
            if track in accepted:
                continue
            attempted = latest.get(track)
            if attempted is None:
                phase = targets[track]
                reason = "evaluate_current_target"
            elif attempted["admission"]["gate_open"]:
                continue
            else:
                index = _phase_index(track, attempted["phase"])
                phase = PHASE_LADDERS[track][max(0, index - 1)]
                reason = "increase_environment_help_after_failed_admission"
            start_index = cycle_base + (
                len(attempts[track]) * track_count + track_index
            ) * stride
            if start_index in used_indices:
                raise ValueError("derived next bank collides with a recorded start index")
            arms.append(
                {
                    "track": track,
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
            "optimizer_steps_authorized": 0,
            "arms": arms,
        }
        status = "collecting"

    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": status,
        "candidate": candidate,
        "invariants": initialized["invariants"],
        "cycle_targets": targets,
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


def _validated_summary(path: Path, *, phase: str) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION or summary.get("phase") != phase:
        raise ValueError("evaluation summary schema or phase does not match")
    gate = summary.get("gate") or {}
    if (
        gate.get("required_qualifying_trajectories") != MINIMUM_QUALIFIERS
        or gate.get("required_distinct_task_keys") != MINIMUM_QUALIFIERS
        or gate.get("acceptance_floor_relaxed") is not False
    ):
        raise ValueError("evaluation summary relaxes or changes the four-trajectory floor")
    qualifying = summary.get("qualifying_trajectories")
    distinct = summary.get("distinct_qualifying_task_keys")
    if not isinstance(qualifying, int) or not isinstance(distinct, int):
        raise ValueError("evaluation summary has invalid admission counts")
    computed_gate = qualifying >= MINIMUM_QUALIFIERS and distinct >= MINIMUM_QUALIFIERS
    if gate.get("gradient_gate_open") is not computed_gate:
        raise ValueError("evaluation summary gate is inconsistent with its counts")
    return summary


def _require_planned_arm(status: dict[str, Any], *, track: str, phase: str, start_index: int) -> None:
    action = status["next_action"]
    if action["kind"] != "collect":
        raise ValueError("the controller is not currently collecting")
    expected = next((arm for arm in action["arms"] if arm["track"] == track), None)
    if expected is None or expected["phase"] != phase or expected["start_index"] != start_index:
        raise ValueError("evaluation does not match the controller's planned arm")


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _init(args: argparse.Namespace) -> None:
    for track, phase in (("child", args.child_target), ("yield", args.yield_target)):
        _phase_index(track, phase)
    candidate = {
        "label": args.candidate_label,
        "model": args.model,
        "model_revision": args.model_revision,
        "base_sha256": args.base_sha256,
        "adapter_sha256": args.adapter_sha256,
    }
    if args.adapter_path is not None:
        adapter_model = args.adapter_path / "adapter_model.safetensors"
        if not adapter_model.is_file() or _sha256_file(adapter_model) != args.adapter_sha256:
            raise ValueError("initial adapter path does not match its SHA-256")
        candidate["adapter_path"] = str(args.adapter_path.resolve())
    payload = {
        "kind": "initialized",
        "recorded_at_utc": args.recorded_at or _now(),
        "candidate": candidate,
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
            "maximum_optimizer_steps_per_cycle": 1,
            "dense_base_updates_authorized": 0,
            "heldout_training_allowed": False,
        },
    }
    _append_event(args.events, payload, create=True)
    print(json.dumps(project(_load_events(args.events)), indent=2, sort_keys=True))


def _record_evaluation(args: argparse.Namespace) -> None:
    events = _load_events(args.events)
    status = project(events)
    _phase_index(args.track, args.phase)
    _require_planned_arm(status, track=args.track, phase=args.phase, start_index=args.start_index)
    summary = _validated_summary(args.summary, phase=args.phase)
    versions_text = args.versions.read_text(encoding="utf-8")
    candidate = status["candidate"]
    required_versions = (
        f"model_revision={candidate['model_revision']}",
        f"interaction_curriculum={args.phase}",
        candidate["adapter_sha256"],
    )
    if any(expected not in versions_text for expected in required_versions):
        raise ValueError("VERSIONS.txt does not identify the frozen candidate and phase")
    if args.traces is not None and not args.traces.is_file():
        raise FileNotFoundError(f"trace artifact does not exist: {args.traces}")
    if args.bootstrap is not None and not args.bootstrap.is_file():
        raise FileNotFoundError(f"bootstrap artifact does not exist: {args.bootstrap}")
    bank_id = args.bank_id or f"{args.track}-{args.phase}-{args.start_index}"
    if any(
        event["kind"] == "evaluation_recorded" and event["bank"]["id"] == bank_id
        for event in events
    ):
        raise ValueError(f"bank id has already been recorded: {bank_id}")
    payload = {
        "kind": "evaluation_recorded",
        "recorded_at_utc": args.recorded_at or _now(),
        "candidate_adapter_sha256": candidate["adapter_sha256"],
        "track": args.track,
        "phase": args.phase,
        "bank": {
            "id": bank_id,
            "start_index": args.start_index,
            "tasks": summary.get("episodes"),
            "split": "train_gen",
        },
        "admission": {
            "qualifying_trajectories": summary["qualifying_trajectories"],
            "distinct_task_keys": summary["distinct_qualifying_task_keys"],
            "gate_open": summary["gate"]["gradient_gate_open"],
            "acceptance_floor_relaxed": False,
        },
        "artifacts": {
            "summary_path": str(args.summary.resolve()),
            "summary_sha256": _sha256_file(args.summary),
            "versions_path": str(args.versions.resolve()),
            "versions_sha256": _sha256_file(args.versions),
            "traces_path": str(args.traces.resolve()) if args.traces is not None else None,
            "traces_sha256": _sha256_file(args.traces) if args.traces is not None else None,
            "bootstrap_path": str(args.bootstrap.resolve()) if args.bootstrap is not None else None,
            "bootstrap_sha256": _sha256_file(args.bootstrap) if args.bootstrap is not None else None,
        },
    }
    _append_event(args.events, payload)
    print(json.dumps(project(_load_events(args.events)), indent=2, sort_keys=True))


def _record_update(args: argparse.Namespace) -> None:
    events = _load_events(args.events)
    status = project(events)
    action = status["next_action"]
    if action["kind"] != "train" or action["optimizer_steps_authorized"] != 1:
        raise ValueError("training is not authorized by the current event log")
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != UPDATE_SCHEMA_VERSION:
        raise ValueError("unsupported update receipt schema")
    candidate = status["candidate"]
    if (
        receipt.get("initial_adapter_sha256") != candidate["adapter_sha256"]
        or receipt.get("base_sha256_before") != candidate["base_sha256"]
        or receipt.get("base_sha256_after") != candidate["base_sha256"]
        or receipt.get("optimizer_steps") != 1
        or receipt.get("dense_base_updates") != 0
        or receipt.get("failed_trajectory_rows") != 0
    ):
        raise ValueError("update receipt violates the authorized bounded LoRA update")
    output_sha = receipt.get("output_adapter_sha256")
    if not isinstance(output_sha, str) or len(output_sha) != 64 or output_sha == candidate["adapter_sha256"]:
        raise ValueError("update receipt lacks a distinct output adapter SHA-256")
    output_model = receipt.get("output_model")
    if not isinstance(output_model, str) or not output_model.strip() or output_model == candidate["model"]:
        raise ValueError("update receipt lacks a distinct output model identifier")
    output_adapter_path_value = receipt.get("output_adapter_path")
    if not isinstance(output_adapter_path_value, str) or not output_adapter_path_value:
        raise ValueError("update receipt lacks an output adapter path")
    output_adapter_path = Path(output_adapter_path_value)
    output_adapter_model = output_adapter_path / "adapter_model.safetensors"
    if not output_adapter_path.is_absolute() or not output_adapter_model.is_file():
        raise ValueError("update receipt output adapter path is not an absolute complete adapter")
    if _sha256_file(output_adapter_model) != output_sha:
        raise ValueError("update receipt output adapter path does not match its SHA-256")
    expected_sources = {
        track: source["summary_sha256"] for track, source in action["sources"].items()
    }
    if receipt.get("source_summary_sha256") != expected_sources:
        raise ValueError("update receipt does not use exactly the admitted sources")
    payload = {
        "kind": "update_recorded",
        "recorded_at_utc": args.recorded_at or _now(),
        "receipt_path": str(args.receipt.resolve()),
        "receipt_sha256": _sha256_file(args.receipt),
        "training_sources": action["sources"],
        "output_candidate": {
            **candidate,
            "label": receipt["output_candidate_label"],
            "model": output_model,
            "adapter_path": str(output_adapter_path.resolve()),
            "adapter_sha256": output_sha,
        },
    }
    _append_event(args.events, payload)
    print(json.dumps(project(_load_events(args.events)), indent=2, sort_keys=True))


def _status(args: argparse.Namespace) -> None:
    print(json.dumps(project(_load_events(args.events)), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--events", type=Path, required=True)
    init.add_argument("--candidate-label", required=True)
    init.add_argument("--model", required=True)
    init.add_argument("--model-revision", required=True)
    init.add_argument("--base-sha256", required=True)
    init.add_argument("--adapter-sha256", required=True)
    init.add_argument("--adapter-path", type=Path)
    init.add_argument("--child-target", default="e0c28_inline_only")
    init.add_argument("--yield-target", default="e0d3_uncapped_yield_exact_child")
    init.add_argument("--next-start-index", type=int, default=4_008_300)
    init.add_argument("--index-stride", type=int, default=DEFAULT_INDEX_STRIDE)
    init.add_argument("--tasks-per-bank", type=int, default=DEFAULT_TASKS_PER_BANK)
    init.add_argument("--recorded-at")
    init.set_defaults(func=_init)

    record_eval = subparsers.add_parser("record-eval")
    record_eval.add_argument("--events", type=Path, required=True)
    record_eval.add_argument("--track", choices=tuple(PHASE_LADDERS), required=True)
    record_eval.add_argument("--phase", required=True)
    record_eval.add_argument("--start-index", type=int, required=True)
    record_eval.add_argument("--bank-id")
    record_eval.add_argument("--summary", type=Path, required=True)
    record_eval.add_argument("--versions", type=Path, required=True)
    record_eval.add_argument("--traces", type=Path)
    record_eval.add_argument("--bootstrap", type=Path)
    record_eval.add_argument("--recorded-at")
    record_eval.set_defaults(func=_record_evaluation)

    record_update = subparsers.add_parser("record-update")
    record_update.add_argument("--events", type=Path, required=True)
    record_update.add_argument("--receipt", type=Path, required=True)
    record_update.add_argument("--recorded-at")
    record_update.set_defaults(func=_record_update)

    status = subparsers.add_parser("status")
    status.add_argument("--events", type=Path, required=True)
    status.set_defaults(func=_status)

    args = parser.parse_args()
    if hasattr(args, "next_start_index") and args.next_start_index < 0:
        raise ValueError("next start index must be non-negative")
    if hasattr(args, "index_stride") and args.index_stride < 1:
        raise ValueError("bank stride must be positive")
    if hasattr(args, "tasks_per_bank") and args.tasks_per_bank < MINIMUM_QUALIFIERS:
        raise ValueError("bank stride must be positive and a bank must contain at least four tasks")
    args.func(args)


if __name__ == "__main__":
    main()
