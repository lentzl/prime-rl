#!/usr/bin/env python3
"""Durable open-ended role-alternating GRPO controller for Qwen3.5-2B."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVENT_SCHEMA = "qwen35-2b-role-grpo-autonomous-event/v1"
PROMOTION_MINIMUM = 4
GROUP_SIZE = 8
EVAL_TASKS = 6
EVAL_MAX_CONCURRENT = 2
EVAL_MAX_ADDRESS_SPACE_BYTES = 32 * 1024**3
# A coordinator update may need several complete GRPO groups before the
# zero-advantage filter yields eight trainable trajectories. Coordinator
# episodes are serialized to contain EnvServer RAM, so keep a three-hour outer
# guard while per-rollout timeouts continue to catch individual stalls.
TRAIN_TIMEOUT_SECONDS = 10800
EVAL_TIMEOUT_SECONDS = 1200
TRAIN_HEALTH_GRACE_SECONDS = 180
TRAIN_HEALTH_FAILURE_CHECKS = 3
TRAIN_HEALTH_POLL_SECONDS = 10
TRAIN_INFERENCE_PORTS = (8000, 8100, 8200)
CHILD_PHASES = (
    "e0_full_actions",
    "e0c_natural_child",
    "e0c2_natural_child_no_template",
    "e0c25_inline_evidence",
    "e0c275_inline_location",
    "e0c28_inline_only",
    "e0c29_evidence_available",
    "e0c3_natural_child_minimal",
)
COORDINATOR_PHASE = "e0d3_uncapped_yield_exact_child"
ROLE_SCOPE = {"child": "non_root", "coordinator": "root"}
RUNTIME_IMAGE_PREFIXES = ("rlm-prime-agent-runtime:", "python:3.11-slim")
LEAK_LADDER = (
    "action_scaffold",
    "child_contract_scaffold",
    "spawn_contract_scaffold",
    "ownership_scaffold",
    "strategy_hint",
)
COUPLED_CURRICULUM_POLICY = "coupled_v1"
SINGLE_AXIS_CURRICULUM_POLICY = "single_axis_v2"
ADMISSION_SCAFFOLD_PROFILES = (
    "custom",
    "tight_answer_free_child_reporting_v1",
)


def _admission_environment(profile: str) -> dict[str, str]:
    if profile == "custom":
        return {"DUAL_SCAFFOLD_PROFILE": "custom"}
    if profile != "tight_answer_free_child_reporting_v1":
        raise ValueError(f"unsupported admission scaffold profile: {profile}")
    return {
        "DUAL_SCAFFOLD_PROFILE": profile,
        "DUAL_LEAK_COORDINATOR_EXACT_ACTION": "0",
        "DUAL_LEAK_COORDINATOR_RETURN_ACTION": "0",
        "DUAL_TYPED_COORDINATOR_RETURN": "0",
        "DUAL_ROOT_COORDINATOR_CONTRACT": "1",
        "DUAL_LEAF_REPORTER_CONTRACT": "1",
        "DUAL_LEAF_INLINE_EVIDENCE": "1",
        "DUAL_LEAF_COMPUTE_REPORT_SCAFFOLD": "0",
        "DUAL_TYPED_CHILD_REPORT": "1",
        "DUAL_CHILD_AUTHORED_COMPUTE": "0",
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate(path: Path, label: str) -> dict[str, str]:
    weight = path / "model.safetensors"
    if not path.is_absolute() or not (path / "STABLE").is_file() or not weight.is_file():
        raise ValueError(f"incomplete dense checkpoint: {path}")
    return {"label": label, "model_path": str(path), "model_sha256": _sha256(weight)}


def _event_digest(event: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(event).encode()).hexdigest()


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    previous = None
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        event = json.loads(line)
        digest = event.pop("event_sha256", None)
        if (
            event.get("schema_version") != EVENT_SCHEMA
            or event.get("sequence") != len(events)
            or event.get("previous_event_sha256") != previous
            or digest != _event_digest(event)
        ):
            raise ValueError(f"invalid autonomous event chain at {path}:{line_number}")
        event["event_sha256"] = digest
        events.append(event)
        previous = digest
    return events


def append_event(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        existing = [json.loads(line) for line in handle if line.strip()]
        event = {
            "schema_version": EVENT_SCHEMA,
            "sequence": len(existing),
            "previous_event_sha256": existing[-1]["event_sha256"] if existing else None,
            "recorded_at_utc": _now(),
            **payload,
        }
        event["event_sha256"] = _event_digest(event)
        handle.seek(0, 2)
        handle.write(_canonical(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        return event


def _other(role: str) -> str:
    return "coordinator" if role == "child" else "child"


def _next_phase(role: str, phase: str) -> str:
    if role == "coordinator":
        return COORDINATOR_PHASE
    index = CHILD_PHASES.index(phase)
    return CHILD_PHASES[min(index + 1, len(CHILD_PHASES) - 1)]


def _default_leak_level(phase: str) -> str:
    return "solution_replay" if phase == "e0_full_actions" else LEAK_LADDER[0]


def _next_leak_level(leak_level: str) -> str:
    if leak_level == "solution_replay":
        return LEAK_LADDER[0]
    try:
        index = LEAK_LADDER.index(leak_level)
    except ValueError as error:
        raise ValueError(f"unsupported bootstrap leak level: {leak_level}") from error
    return LEAK_LADDER[min(index + 1, len(LEAK_LADDER) - 1)]


def _previous_phase(role: str, phase: str) -> str:
    if role == "coordinator":
        return COORDINATOR_PHASE
    index = CHILD_PHASES.index(phase)
    return CHILD_PHASES[max(index - 1, 0)]


def _previous_leak_level(leak_level: str) -> str:
    if leak_level == "solution_replay":
        return "solution_replay"
    try:
        index = LEAK_LADDER.index(leak_level)
    except ValueError as error:
        raise ValueError(f"unsupported bootstrap leak level: {leak_level}") from error
    return "solution_replay" if index == 0 else LEAK_LADDER[index - 1]


def _advance_single_axis_curriculum(role: str, phase: str, leak_level: str) -> tuple[str, str]:
    next_phase = _next_phase(role, phase)
    if next_phase != phase:
        return next_phase, leak_level
    return phase, _next_leak_level(leak_level)


def project(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events or events[0].get("kind") != "initialized":
        raise ValueError("autonomous state must begin with initialized")
    initial = events[0]
    state = {
        "frontier": copy.deepcopy(initial["frontier"]),
        "promoted": copy.deepcopy(initial["promoted"]),
        "initial": copy.deepcopy(initial["frontier"]),
        "phases": copy.deepcopy(initial["phases"]),
        "leak_levels": copy.deepcopy(
            initial.get(
                "leak_levels",
                {role: _default_leak_level(initial["phases"][role]) for role in ("coordinator", "child")},
            )
        ),
        "next_role": initial["next_role"],
        "next_cycle": initial["next_cycle"],
        "pending_eval": None,
        "curriculum_policy": initial.get("curriculum_policy", COUPLED_CURRICULUM_POLICY),
        "admission_scaffold_profile": initial.get("admission_scaffold_profile", "custom"),
    }
    for event in events[1:]:
        kind = event["kind"]
        if kind == "train_completed":
            state["frontier"][event["role"]] = copy.deepcopy(event["output_candidate"])
            state["pending_eval"] = {
                "cycle": event["cycle"],
                "role": event["role"],
                "phase": event["phase"],
                "bootstrap_leak_level": event.get("bootstrap_leak_level", _default_leak_level(event["phase"])),
            }
        elif kind in {"train_failed", "train_no_update"}:
            # A clean no-update is learning evidence and hands control to the
            # other role. An infrastructure failure is not policy evidence;
            # retry the same role under the next immutable cycle label.
            state["next_role"] = event["role"] if kind == "train_failed" else _other(event["role"])
            state["next_cycle"] = event["cycle"] + 1
            state["pending_eval"] = None
        elif kind in {"evaluation_completed", "evaluation_failed"}:
            if kind == "evaluation_completed" and event["admitted"]:
                role = event["role"]
                state["promoted"][role] = copy.deepcopy(state["frontier"][role])
                # Only explicitly versioned evaluations advance the leak ladder.
                # This prevents a controller upgrade from retroactively counting
                # historical admissions collected under different router semantics.
                evaluated_leak = event.get("bootstrap_leak_level")
                if evaluated_leak is not None:
                    if evaluated_leak != state["leak_levels"][role]:
                        raise ValueError("admission leak level does not match the projected curriculum")
                if state["curriculum_policy"] == COUPLED_CURRICULUM_POLICY:
                    state["phases"][role] = _next_phase(role, event["phase"])
                    if evaluated_leak is not None:
                        state["leak_levels"][role] = _next_leak_level(evaluated_leak)
                elif state["curriculum_policy"] == SINGLE_AXIS_CURRICULUM_POLICY:
                    next_phase, next_leak = _advance_single_axis_curriculum(
                        role,
                        event["phase"],
                        state["leak_levels"][role],
                    )
                    state["phases"][role] = next_phase
                    if evaluated_leak is not None:
                        state["leak_levels"][role] = next_leak
                else:
                    raise ValueError(f"unsupported curriculum policy: {state['curriculum_policy']}")
            state["next_role"] = _other(event["role"])
            state["next_cycle"] = event["cycle"] + 1
            state["pending_eval"] = None
        elif kind == "frontier_recovered":
            state["frontier"] = copy.deepcopy(event["frontier"])
            state["next_role"] = event["next_role"]
            state["next_cycle"] = event["next_cycle"]
            state["pending_eval"] = None
        elif kind == "leak_level_changed":
            role = event["role"]
            if event["from_leak_level"] != state["leak_levels"][role]:
                raise ValueError("leak-level migration does not match projected state")
            if event["to_leak_level"] != _next_leak_level(event["from_leak_level"]):
                raise ValueError("leak-level migration must move exactly one ladder step")
            state["leak_levels"][role] = event["to_leak_level"]
        elif kind == "curriculum_policy_changed":
            if event["from_policy"] != state["curriculum_policy"]:
                raise ValueError("curriculum-policy migration does not match projected state")
            if event["from_policy"] != COUPLED_CURRICULUM_POLICY or event["to_policy"] != SINGLE_AXIS_CURRICULUM_POLICY:
                raise ValueError("unsupported curriculum-policy migration")
            state["curriculum_policy"] = event["to_policy"]
        elif kind == "curriculum_recovered":
            role = event["role"]
            current_phase = state["phases"][role]
            current_leak = state["leak_levels"][role]
            if event["from_phase"] != current_phase or event["from_leak_level"] != current_leak:
                raise ValueError("curriculum recovery does not match projected state")
            to_phase = event["to_phase"]
            to_leak = event["to_leak_level"]
            phase_rollback = (
                to_phase == _previous_phase(role, current_phase)
                and to_phase != current_phase
                and to_leak == current_leak
            )
            leak_rollback = (
                to_phase == current_phase and to_leak == _previous_leak_level(current_leak) and to_leak != current_leak
            )
            if not (phase_rollback or leak_rollback):
                raise ValueError("curriculum recovery must roll back exactly one axis by one step")
            state["phases"][role] = to_phase
            state["leak_levels"][role] = to_leak
    return state


def _cycle_events(events: list[dict[str, Any]], cycle: int) -> list[dict[str, Any]]:
    return [event for event in events if event.get("cycle") == cycle]


def _process_active(label: str) -> bool:
    result = subprocess.run(["pgrep", "-f", label], capture_output=True, text=True)
    return result.returncode == 0 and bool(result.stdout.strip())


def _gpus_idle() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _runtime_containers() -> dict[str, dict[str, str]]:
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.ID}}\t{{.Image}}\t{{.Names}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    containers = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        container_id, image, name = line.split("\t", 2)
        containers[container_id] = {
            "container_id": container_id,
            "image": image,
            "name": name,
        }
    return containers


def _created_runtime_containers(before: set[str], after: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    return [
        after[container_id]
        for container_id in sorted(after.keys() - before)
        if after[container_id]["image"].startswith(RUNTIME_IMAGE_PREFIXES)
    ]


def _ports_listening(ports: tuple[int, ...]) -> bool:
    for port in ports:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                pass
        except OSError:
            return False
    return True


def _stop_process_group(process: subprocess.Popen[Any]) -> int:
    os.killpg(process.pid, signal.SIGINT)
    try:
        return process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            return process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            return process.wait()


def _run_bounded(
    command: list[str],
    *,
    env: dict[str, str],
    log: Path,
    timeout_seconds: int,
    health_ports: tuple[int, ...] = (),
    health_grace_seconds: int = 0,
    health_terminal: Path | None = None,
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started = time.monotonic()
        deadline = started + timeout_seconds
        unhealthy_checks = 0
        while process.poll() is None:
            now = time.monotonic()
            if now >= deadline:
                handle.write("autonomous guard: action timeout; stopping process group\n")
                handle.flush()
                return _stop_process_group(process)
            if (
                health_ports
                and now >= started + health_grace_seconds
                and not (health_terminal is not None and health_terminal.is_file())
            ):
                unhealthy_checks = unhealthy_checks + 1 if not _ports_listening(health_ports) else 0
                if unhealthy_checks >= TRAIN_HEALTH_FAILURE_CHECKS:
                    handle.write(
                        "autonomous guard: inference ports unavailable after startup; "
                        "stopping failed action without treating samples as learning evidence\n"
                    )
                    handle.flush()
                    return _stop_process_group(process)
            time.sleep(
                min(
                    TRAIN_HEALTH_POLL_SECONDS,
                    max(0.1, deadline - time.monotonic()),
                )
            )
        return process.returncode


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_train_receipt(
    receipt_path: Path,
    *,
    role: str,
    phase: str,
    bootstrap_leak_level: str,
    source: dict[str, str],
) -> dict[str, str]:
    receipt = _read_json(receipt_path)
    output = receipt.get("output")
    if (
        receipt.get("schema_version") != "qwen35-2b-role-grpo-update/v1"
        or receipt.get("role") != role
        or receipt.get("phase") != phase
        or receipt.get("bootstrap_leak_level") != bootstrap_leak_level
        or receipt.get("sampled_session_scope") != ROLE_SCOPE[role]
        or receipt.get("optimizer_updates") != 1
        or receipt.get("promotion_minimum") != PROMOTION_MINIMUM
        or receipt.get("env_server_max_concurrent") != (1 if role == "coordinator" else 2)
        or receipt.get("coordinator_action_leak") is not (role == "child")
        or receipt.get("first_action_sampling")
        != ("prompted_native_spawn" if role == "coordinator" else "masked_frozen_anchor")
        or receipt.get("child_prompt_action_leak") is not True
        or receipt.get("child_router_action_leak") is not (role == "coordinator")
        or receipt.get("child_action_sampling")
        != ("synthetic_exact_send" if role == "coordinator" else "prompted_native_send")
        or receipt.get("reward_mode") != ("event_control" if role == "coordinator" else "child_action")
        or receipt.get("task_bank", {}).get("group_size") != GROUP_SIZE
        or receipt.get("source", {}).get("model_sha256") != source["model_sha256"]
        or not isinstance(output, dict)
        or output.get("model_sha256") == source["model_sha256"]
    ):
        raise ValueError(f"invalid role-GRPO update receipt: {receipt_path}")
    path = Path(output["path"])
    candidate = _candidate(path, receipt_path.stem.removesuffix("-receipt"))
    if candidate["model_sha256"] != output["model_sha256"]:
        raise ValueError("role-GRPO output hash does not match its receipt")
    return candidate


def _qualifying_evaluation(
    run_dir: Path,
    frontier: dict[str, dict[str, str]],
    admission_scaffold_profile: str | None = None,
) -> dict[str, Any]:
    summary = _read_json(run_dir / "SUMMARY.json")
    traces_path = run_dir / "natural_n1a" / "traces.jsonl"
    envelopes = [json.loads(line) for line in traces_path.read_text().splitlines() if line.strip()]
    traces = []
    for envelope in envelopes:
        nested = envelope.get("traces")
        if envelope.get("ok") is not True or envelope.get("errors") or not isinstance(nested, list):
            raise ValueError(f"admission envelope contains an error: {run_dir}")
        traces.extend(nested)
    qualifying = []
    for trace in traces:
        reward = trace.get("rewards", {}).get("harness_score", {}).get("score")
        task = trace.get("task", {})
        task_key = task.get("key") if isinstance(task, dict) else None
        if trace.get("is_completed") is True and not trace.get("errors") and reward == 1 and isinstance(task_key, str):
            qualifying.append(task_key)
    audit_path = run_dir / "ROUTING_AUDIT.jsonl"
    audit = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
    role_counts = {
        role: sum(event.get("role") == role and event.get("status") == 200 for event in audit)
        for role in ("coordinator", "child")
    }
    role_failure_counts = {
        role: sum(event.get("role") == role and event.get("status") != 200 for event in audit)
        for role in ("coordinator", "child")
    }
    versions = {}
    for line in (run_dir / "VERSIONS.txt").read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            versions[key] = value
    if (
        summary.get("episodes") != EVAL_TASKS
        or summary.get("errors") != 0
        or len(envelopes) != EVAL_TASKS
        or len(traces) != EVAL_TASKS
        or len(set(qualifying)) != len(qualifying)
        or min(role_counts.values()) < 1
        or versions.get("coordinator_model_sha256") != frontier["coordinator"]["model_sha256"]
        or versions.get("child_model_sha256") != frontier["child"]["model_sha256"]
        or (
            admission_scaffold_profile is not None
            and versions.get("dual_policy_scaffold_profile") != admission_scaffold_profile
        )
    ):
        raise ValueError(f"incomplete or routing-invalid admission evaluation: {run_dir}")
    return {
        "episodes": len(traces),
        "qualifying": len(qualifying),
        "distinct_qualifying": len(set(qualifying)),
        "admitted": len(set(qualifying)) >= PROMOTION_MINIMUM,
        "promotion_minimum": PROMOTION_MINIMUM,
        "role_route_counts": role_counts,
        "role_route_failure_counts": role_failure_counts,
        "admission_scaffold_profile": admission_scaffold_profile,
    }


class Controller:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = args.repo.resolve()
        self.state_dir = args.state_dir.resolve()
        self.events_path = self.state_dir / "events.jsonl"
        self.logs = self.state_dir / "logs"
        self.lock_path = self.state_dir / "runner.lock"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup_action_containers(
        self,
        *,
        cycle: int,
        role: str,
        label: str,
        baseline: list[str] | None,
    ) -> None:
        if baseline is None:
            return
        created = _created_runtime_containers(set(baseline), _runtime_containers())
        if not created:
            return
        subprocess.run(
            ["docker", "rm", "--force", *(item["container_id"] for item in created)],
            check=True,
            capture_output=True,
            text=True,
        )
        append_event(
            self.events_path,
            {
                "kind": "runtime_containers_pruned",
                "cycle": cycle,
                "role": role,
                "run_name": label,
                "containers": created,
                "recoverable": False,
            },
        )

    def _initialize(self) -> None:
        if self.events_path.exists():
            state = project(load_events(self.events_path))
            if state["admission_scaffold_profile"] != self.args.admission_scaffold_profile:
                raise ValueError("controller admission scaffold does not match durable state")
            return
        frontier = {
            "coordinator": _candidate(self.args.coordinator_model.resolve(), self.args.coordinator_label),
            "child": _candidate(self.args.child_model.resolve(), self.args.child_label),
        }
        if self.args.initial_admission_run is None:
            raise ValueError("a fresh autonomous state requires --initial-admission-run")
        admission_run = self.args.initial_admission_run.resolve()
        initial_admission = _qualifying_evaluation(admission_run, frontier, self.args.admission_scaffold_profile)
        if not initial_admission["admitted"]:
            raise ValueError("initial pair does not satisfy the four-trajectory promotion floor")
        append_event(
            self.events_path,
            {
                "kind": "initialized",
                "frontier": frontier,
                "promoted": copy.deepcopy(frontier),
                "phases": {
                    "coordinator": COORDINATOR_PHASE,
                    "child": self.args.child_phase,
                },
                "leak_levels": {
                    "coordinator": LEAK_LADDER[0],
                    "child": _default_leak_level(self.args.child_phase),
                },
                "next_role": self.args.next_role,
                "next_cycle": self.args.start_cycle,
                "promotion_minimum": PROMOTION_MINIMUM,
                "full_dense_only": True,
                "admission_scaffold_profile": self.args.admission_scaffold_profile,
                "initial_admission": {
                    "run_dir": str(admission_run),
                    **initial_admission,
                },
            },
        )

    def _names(self, cycle: int, role: str, phase: str) -> tuple[str, int, str, int]:
        tag = phase.replace("_", "-")
        train_start = self.args.start_index + cycle * 100
        train_label = f"grpo-auto-{cycle:06d}-{role}-{tag}-{train_start}"
        eval_start = train_start + 50
        eval_label = f"{train_label}-admission-{eval_start}-n{EVAL_TASKS}"
        return train_label, train_start, eval_label, eval_start

    def _train_paths(self, label: str) -> tuple[Path, Path]:
        base = self.args.experiment_dir.resolve()
        return base / f"{label}-receipt.json", base / f"{label}-attempt.json"

    def _record_train_terminal(
        self,
        *,
        cycle: int,
        role: str,
        phase: str,
        bootstrap_leak_level: str,
        label: str,
        source: dict[str, str],
    ) -> bool:
        receipt_path, attempt_path = self._train_paths(label)
        if receipt_path.is_file():
            output = _validate_train_receipt(
                receipt_path,
                role=role,
                phase=phase,
                bootstrap_leak_level=bootstrap_leak_level,
                source=source,
            )
            append_event(
                self.events_path,
                {
                    "kind": "train_completed",
                    "cycle": cycle,
                    "role": role,
                    "phase": phase,
                    "bootstrap_leak_level": bootstrap_leak_level,
                    "run_name": label,
                    "receipt_path": str(receipt_path),
                    "output_candidate": output,
                },
            )
            return True
        if attempt_path.is_file():
            attempt = _read_json(attempt_path)
            status = attempt.get("status")
            kind = "train_no_update" if status == "no_update" else "train_failed"
            append_event(
                self.events_path,
                {
                    "kind": kind,
                    "cycle": cycle,
                    "role": role,
                    "phase": phase,
                    "bootstrap_leak_level": bootstrap_leak_level,
                    "run_name": label,
                    "attempt_path": str(attempt_path),
                    "attempt_status": status,
                    "attempt_stage": attempt.get("stage"),
                },
            )
            return True
        return False

    def _start_train(self, state: dict[str, Any]) -> None:
        cycle, role = state["next_cycle"], state["next_role"]
        phase = state["phases"][role]
        bootstrap_leak_level = state["leak_levels"][role]
        label, start, _, _ = self._names(cycle, role, phase)
        source = state["frontier"][role]
        anchor = state["frontier"][_other(role)]
        # Validate both immutable inputs before recording train_started.  A
        # missing recovery frontier is infrastructure damage, not a completed
        # training attempt, and must never turn into a rapid cycle-number loop.
        for checkpoint_role, checkpoint in ((role, source), (_other(role), anchor)):
            path = Path(checkpoint["model_path"])
            if not path.is_absolute() or not (path / "STABLE").is_file() or not (path / "model.safetensors").is_file():
                raise RuntimeError(f"{checkpoint_role} frontier checkpoint is unavailable: {path}")
        runtime_container_baseline = sorted(_runtime_containers())
        append_event(
            self.events_path,
            {
                "kind": "train_started",
                "cycle": cycle,
                "role": role,
                "phase": phase,
                "bootstrap_leak_level": bootstrap_leak_level,
                "run_name": label,
                "task_bank": {"start_index": start, "count": GROUP_SIZE},
                "source": copy.deepcopy(source),
                "anchor": copy.deepcopy(anchor),
                "runtime_container_baseline": runtime_container_baseline,
            },
        )
        env = os.environ.copy()
        env["UV_BIN"] = str(self.args.uv_bin)
        env["Q35_2B_ROLE_GRPO_BOOTSTRAP_LEAK_LEVEL"] = bootstrap_leak_level
        command = [
            "bash",
            str(self.repo / "scripts/run_q35_2b_role_grpo_v1.sh"),
            role,
            source["model_path"],
            anchor["model_path"],
            label,
            phase,
            str(start),
            str(GROUP_SIZE),
        ]
        _run_bounded(
            command,
            env=env,
            log=self.logs / f"{label}.log",
            timeout_seconds=self.args.train_timeout,
            health_ports=TRAIN_INFERENCE_PORTS,
            health_grace_seconds=TRAIN_HEALTH_GRACE_SECONDS,
            health_terminal=self.args.output_root.resolve() / label / "weights/step_1/STABLE",
        )
        self._cleanup_action_containers(
            cycle=cycle,
            role=role,
            label=label,
            baseline=runtime_container_baseline,
        )
        if not self._record_train_terminal(
            cycle=cycle,
            role=role,
            phase=phase,
            bootstrap_leak_level=bootstrap_leak_level,
            label=label,
            source=source,
        ):
            append_event(
                self.events_path,
                {
                    "kind": "train_failed",
                    "cycle": cycle,
                    "role": role,
                    "phase": phase,
                    "bootstrap_leak_level": bootstrap_leak_level,
                    "run_name": label,
                    "attempt_status": "missing_terminal_receipt",
                },
            )
            raise RuntimeError(f"training ended without a terminal receipt: {label}")

    def _bootstrap(self, *, label: str, leak: str, start: int) -> Path:
        path = self.args.artifact_root.resolve() / f"{label}-{leak}-bootstrap.json"
        if path.exists():
            payload = _read_json(path)
            if (
                payload.get("leak_level") != leak
                or payload.get("tasks_per_axis") != EVAL_TASKS
                or payload.get("axes") != [{"name": "natural_n1a", "start_index": start}]
            ):
                raise ValueError(f"existing admission bootstrap does not match: {path}")
            return path
        subprocess.run(
            [
                str(self.args.uv_bin),
                "run",
                "--frozen",
                "--no-sync",
                str(self.repo / "scripts/build_q35_2b_environment_bootstrap_context_v1.py"),
                "--output",
                str(path),
                "--axis",
                f"natural_n1a:{start}",
                "--tasks-per-axis",
                str(EVAL_TASKS),
                "--master-seed",
                str(self.args.master_seed),
                "--leak-level",
                leak,
            ],
            check=True,
        )
        return path

    def _record_eval_terminal(
        self,
        *,
        cycle: int,
        role: str,
        phase: str,
        bootstrap_leak_level: str,
        label: str,
        frontier: dict[str, dict[str, str]],
    ) -> bool:
        run_dir = self.args.result_root.resolve() / label
        if not (run_dir / "SUMMARY.json").is_file() or not (run_dir / "VERSIONS.txt").is_file():
            return False
        try:
            evidence = _qualifying_evaluation(
                run_dir,
                frontier,
                getattr(self.args, "admission_scaffold_profile", None),
            )
        except ValueError as error:
            # Admission is deliberately fail-closed: an errored envelope cannot
            # promote a checkpoint. Record invalid completed evidence as a
            # terminal failure so the controller advances instead of crashing
            # and retrying the same immutable result forever.
            append_event(
                self.events_path,
                {
                    "kind": "evaluation_failed",
                    "cycle": cycle,
                    "role": role,
                    "phase": phase,
                    "bootstrap_leak_level": bootstrap_leak_level,
                    "run_name": label,
                    "run_dir": str(run_dir),
                    "failure_type": "invalid_admission_evidence",
                    "error": str(error),
                },
            )
            return True
        append_event(
            self.events_path,
            {
                "kind": "evaluation_completed",
                "cycle": cycle,
                "role": role,
                "phase": phase,
                "bootstrap_leak_level": bootstrap_leak_level,
                "run_name": label,
                "run_dir": str(run_dir),
                **evidence,
            },
        )
        return True

    def _start_evaluation(self, state: dict[str, Any]) -> None:
        pending = state["pending_eval"]
        cycle, role, phase = pending["cycle"], pending["role"], pending["phase"]
        bootstrap_leak_level = pending["bootstrap_leak_level"]
        _, _, label, start = self._names(cycle, role, phase)
        bootstrap = self._bootstrap(label=label, leak=bootstrap_leak_level, start=start)
        runtime_container_baseline = sorted(_runtime_containers())
        append_event(
            self.events_path,
            {
                "kind": "evaluation_started",
                "cycle": cycle,
                "role": role,
                "phase": phase,
                "bootstrap_leak_level": bootstrap_leak_level,
                "run_name": label,
                "task_bank": {"start_index": start, "count": EVAL_TASKS},
                "frontier": copy.deepcopy(state["frontier"]),
                "bootstrap_path": str(bootstrap),
                "runtime_container_baseline": runtime_container_baseline,
            },
        )
        env = os.environ.copy()
        admission_profile = state["admission_scaffold_profile"]
        env.update(
            {
                "UV_BIN": str(self.args.uv_bin),
                "QWEN38_QUALIFICATION_OUTPUT_ROOT": str(self.args.result_root.resolve()),
                "QWEN38_QUALIFICATION_AXES": "natural_n1a",
                "QWEN38_QUALIFICATION_NUM_TASKS": str(EVAL_TASKS),
                "QWEN38_QUALIFICATION_NUM_ROLLOUTS": "1",
                "QWEN38_QUALIFICATION_MAX_CONCURRENT": str(EVAL_MAX_CONCURRENT),
                "QWEN38_QUALIFICATION_EVAL_MAX_ADDRESS_SPACE_BYTES": str(EVAL_MAX_ADDRESS_SPACE_BYTES),
                "QWEN38_QUALIFICATION_START_INDEX": str(start),
                "QWEN38_QUALIFICATION_MASTER_SEED": str(self.args.master_seed),
                "QUALIFICATION_REASONING_EFFORT": "high",
                "QUALIFICATION_SAMPLING_SEED": str(self.args.sampling_seed + cycle),
                "QUALIFICATION_SAMPLING_TEMPERATURE": "1.0",
                "QUALIFICATION_PRIVILEGED_BOOTSTRAP_PATH": str(bootstrap),
                "PROCEDURAL_INTERACTION_CURRICULUM": (
                    "e0c4_recursive_coordinator_return"
                    if admission_profile == "tight_answer_free_child_reporting_v1"
                    else phase
                ),
                "DUAL_EXTERNAL_MODEL": f"q35-grpo-auto-{cycle:06d}",
                **_admission_environment(admission_profile),
            }
        )
        command = [
            "bash",
            str(self.repo / "scripts/run_q35_2b_dual_policy_mastery_v1.sh"),
            state["frontier"]["coordinator"]["model_path"],
            state["frontier"]["child"]["model_path"],
            label,
            "local",
        ]
        return_code = _run_bounded(
            command,
            env=env,
            log=self.logs / f"{label}.log",
            timeout_seconds=self.args.eval_timeout,
        )
        self._cleanup_action_containers(
            cycle=cycle,
            role=role,
            label=label,
            baseline=runtime_container_baseline,
        )
        if not self._record_eval_terminal(
            cycle=cycle,
            role=role,
            phase=phase,
            bootstrap_leak_level=bootstrap_leak_level,
            label=label,
            frontier=state["frontier"],
        ):
            append_event(
                self.events_path,
                {
                    "kind": "evaluation_failed",
                    "cycle": cycle,
                    "role": role,
                    "phase": phase,
                    "bootstrap_leak_level": bootstrap_leak_level,
                    "run_name": label,
                    "return_code": return_code,
                },
            )

    def _reconcile(self, events: list[dict[str, Any]], state: dict[str, Any]) -> bool:
        cycle = state["pending_eval"]["cycle"] if state["pending_eval"] else state["next_cycle"]
        current = _cycle_events(events, cycle)
        train_started = next((event for event in current if event["kind"] == "train_started"), None)
        train_terminal = next(
            (event for event in current if event["kind"] in {"train_completed", "train_failed", "train_no_update"}),
            None,
        )
        if train_started is not None and train_terminal is None:
            label = train_started["run_name"]
            if _process_active(label):
                time.sleep(self.args.poll_seconds)
                return True
            self._cleanup_action_containers(
                cycle=cycle,
                role=train_started["role"],
                label=label,
                baseline=train_started.get("runtime_container_baseline"),
            )
            if not self._record_train_terminal(
                cycle=cycle,
                role=train_started["role"],
                phase=train_started["phase"],
                bootstrap_leak_level=train_started.get(
                    "bootstrap_leak_level",
                    _default_leak_level(train_started["phase"]),
                ),
                label=label,
                source=train_started["source"],
            ):
                append_event(
                    self.events_path,
                    {
                        "kind": "train_failed",
                        "cycle": cycle,
                        "role": train_started["role"],
                        "phase": train_started["phase"],
                        "bootstrap_leak_level": train_started.get(
                            "bootstrap_leak_level",
                            _default_leak_level(train_started["phase"]),
                        ),
                        "run_name": label,
                        "attempt_status": "interrupted_without_terminal_receipt",
                    },
                )
                raise RuntimeError(f"interrupted training has no terminal receipt: {label}")
            return True
        eval_started = next((event for event in current if event["kind"] == "evaluation_started"), None)
        eval_terminal = next(
            (event for event in current if event["kind"] in {"evaluation_completed", "evaluation_failed"}),
            None,
        )
        if eval_started is not None and eval_terminal is None:
            label = eval_started["run_name"]
            if _process_active(label):
                time.sleep(self.args.poll_seconds)
                return True
            self._cleanup_action_containers(
                cycle=cycle,
                role=eval_started["role"],
                label=label,
                baseline=eval_started.get("runtime_container_baseline"),
            )
            if not self._record_eval_terminal(
                cycle=cycle,
                role=eval_started["role"],
                phase=eval_started["phase"],
                bootstrap_leak_level=eval_started.get(
                    "bootstrap_leak_level",
                    _default_leak_level(eval_started["phase"]),
                ),
                label=label,
                frontier=eval_started["frontier"],
            ):
                append_event(
                    self.events_path,
                    {
                        "kind": "evaluation_failed",
                        "cycle": cycle,
                        "role": eval_started["role"],
                        "phase": eval_started["phase"],
                        "bootstrap_leak_level": eval_started.get(
                            "bootstrap_leak_level",
                            _default_leak_level(eval_started["phase"]),
                        ),
                        "run_name": label,
                        "return_code": None,
                    },
                )
            return True
        return False

    def _prune(self, events: list[dict[str, Any]], state: dict[str, Any]) -> None:
        free = shutil.disk_usage(self.args.output_root).free
        if free >= self.args.prune_below_gib * 1024**3:
            return
        protected = {
            candidate["model_path"]
            for group in (state["frontier"], state["promoted"], state["initial"])
            for candidate in group.values()
        }
        completed = [event for event in events if event["kind"] == "train_completed"]
        for event in completed[:-2]:
            candidate = event["output_candidate"]
            checkpoint = Path(candidate["model_path"])
            if str(checkpoint) in protected or not checkpoint.exists():
                continue
            if not checkpoint.is_relative_to(self.args.output_root.resolve()):
                raise ValueError(f"refusing to prune checkpoint outside autonomous output root: {checkpoint}")
            run_dir = checkpoint.parent.parent
            if not run_dir.name.startswith("grpo-auto-"):
                raise ValueError(f"refusing to prune non-autonomous checkpoint: {checkpoint}")
            before = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
            shutil.rmtree(checkpoint)
            append_event(
                self.events_path,
                {
                    "kind": "checkpoint_pruned",
                    "cycle": event["cycle"],
                    "role": event["role"],
                    "model_path": str(checkpoint),
                    "model_sha256": candidate["model_sha256"],
                    "bytes_removed": before,
                    "recoverable": False,
                },
            )
            if shutil.disk_usage(self.args.output_root).free >= self.args.prune_below_gib * 1024**3:
                break

    def run(self) -> None:
        with self.lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("another role-GRPO autonomous controller holds the lock") from error
            self._initialize()
            while not self.args.stop_file.exists():
                events = load_events(self.events_path)
                state = project(events)
                if self.args.max_cycles and state["next_cycle"] >= self.args.start_cycle + self.args.max_cycles:
                    return
                if self._reconcile(events, state):
                    continue
                if not _gpus_idle():
                    time.sleep(self.args.poll_seconds)
                    continue
                self._prune(events, state)
                events = load_events(self.events_path)
                state = project(events)
                if state["pending_eval"] is not None:
                    self._start_evaluation(state)
                else:
                    self._start_train(state)
                time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--coordinator-model", type=Path, required=True)
    parser.add_argument("--coordinator-label", default="C26")
    parser.add_argument("--child-model", type=Path, required=True)
    parser.add_argument("--child-label", default="KGRPO1")
    parser.add_argument("--initial-admission-run", type=Path)
    parser.add_argument("--child-phase", choices=CHILD_PHASES, default="e0c_natural_child")
    parser.add_argument("--next-role", choices=("coordinator", "child"), default="coordinator")
    parser.add_argument("--start-cycle", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=9_400_000)
    parser.add_argument("--master-seed", type=int, default=20260824)
    parser.add_argument("--sampling-seed", type=int, default=20260825)
    parser.add_argument(
        "--admission-scaffold-profile",
        choices=ADMISSION_SCAFFOLD_PROFILES,
        default="custom",
    )
    parser.add_argument("--uv-bin", type=Path, default=Path("/home/ubuntu/.local/bin/uv"))
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("/home/ubuntu/rlm/prime-rl/experiments/qwen35-2b-self-bootstrap-dual-dense-v1/grpo-runs"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/home/ubuntu/rlm/artifacts/q35-2b-self-bootstrap-dual-dense-grpo-v1"),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("/home/ubuntu/rlm/results/q35-2b-self-bootstrap-dual-dense-grpo-v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/ubuntu/rlm/outputs/q35-2b-self-bootstrap-dual-dense-grpo-v1"),
    )
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--train-timeout", type=int, default=TRAIN_TIMEOUT_SECONDS)
    parser.add_argument("--eval-timeout", type=int, default=EVAL_TIMEOUT_SECONDS)
    parser.add_argument("--prune-below-gib", type=int, default=40)
    parser.add_argument("--max-cycles", type=int, default=0)
    args = parser.parse_args()
    if (
        min(
            args.start_cycle,
            args.start_index,
            args.poll_seconds,
            args.train_timeout,
            args.eval_timeout,
            args.prune_below_gib,
        )
        < 1
    ):
        raise ValueError("numeric controller arguments must be positive")
    Controller(args).run()


if __name__ == "__main__":
    main()
