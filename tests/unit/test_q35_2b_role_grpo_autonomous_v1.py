import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_q35_2b_role_grpo_autonomous_v1.py"
MASTERY_SCRIPT = ROOT / "scripts/run_q35_2b_dual_policy_mastery_v1.sh"
RECOVERY_SCRIPT = ROOT / "scripts/recover_q35_2b_autonomous_frontier_v1.py"
SPEC = importlib.util.spec_from_file_location("role_grpo_auto", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _candidate(label: str) -> dict[str, str]:
    return {
        "label": label,
        "model_path": f"/models/{label}",
        "model_sha256": label.lower() * 8,
    }


def _initialized() -> dict:
    return {
        "kind": "initialized",
        "frontier": {"coordinator": _candidate("C26"), "child": _candidate("K1")},
        "promoted": {"coordinator": _candidate("C26"), "child": _candidate("K1")},
        "phases": {
            "coordinator": MODULE.COORDINATOR_PHASE,
            "child": "e0c_natural_child",
        },
        "next_role": "coordinator",
        "next_cycle": 1,
    }


def test_project_advances_training_frontier_before_promotion() -> None:
    output = _candidate("C27")
    state = MODULE.project(
        [
            _initialized(),
            {
                "kind": "train_completed",
                "cycle": 1,
                "role": "coordinator",
                "phase": MODULE.COORDINATOR_PHASE,
                "output_candidate": output,
            },
        ]
    )
    assert state["frontier"]["coordinator"] == output
    assert state["promoted"]["coordinator"]["label"] == "C26"
    assert state["pending_eval"]["cycle"] == 1


def test_project_advances_explicit_leak_level_one_step_only_after_admission() -> None:
    output = _candidate("C27")
    trained = {
        "kind": "train_completed",
        "cycle": 1,
        "role": "coordinator",
        "phase": MODULE.COORDINATOR_PHASE,
        "bootstrap_leak_level": "action_scaffold",
        "output_candidate": output,
    }
    rejected = MODULE.project(
        [
            _initialized(),
            trained,
            {
                "kind": "evaluation_completed",
                "cycle": 1,
                "role": "coordinator",
                "phase": MODULE.COORDINATOR_PHASE,
                "bootstrap_leak_level": "action_scaffold",
                "admitted": False,
            },
        ]
    )
    assert rejected["leak_levels"]["coordinator"] == "action_scaffold"

    admitted = MODULE.project(
        [
            _initialized(),
            trained,
            {
                "kind": "evaluation_completed",
                "cycle": 1,
                "role": "coordinator",
                "phase": MODULE.COORDINATOR_PHASE,
                "bootstrap_leak_level": "action_scaffold",
                "admitted": True,
            },
        ]
    )
    assert admitted["leak_levels"]["coordinator"] == "child_contract_scaffold"


def test_project_does_not_retroactively_count_legacy_admissions() -> None:
    state = MODULE.project(
        [
            _initialized(),
            {
                "kind": "evaluation_completed",
                "cycle": 0,
                "role": "coordinator",
                "phase": MODULE.COORDINATOR_PHASE,
                "admitted": True,
            },
        ]
    )
    assert state["leak_levels"]["coordinator"] == "action_scaffold"


def test_project_accepts_audited_one_step_leak_migration() -> None:
    state = MODULE.project(
        [
            _initialized(),
            {
                "kind": "leak_level_changed",
                "role": "coordinator",
                "from_leak_level": "action_scaffold",
                "to_leak_level": "child_contract_scaffold",
                "evidence_sequence": 284,
            },
        ]
    )
    assert state["leak_levels"]["coordinator"] == "child_contract_scaffold"


def test_project_accepts_explicit_forensic_frontier_recovery() -> None:
    coordinator = _candidate("C57")
    child = _candidate("K56")
    state = MODULE.project(
        [
            _initialized(),
            {
                "kind": "frontier_recovered",
                "frontier": {"coordinator": coordinator, "child": child},
                "next_role": "coordinator",
                "next_cycle": 59,
            },
        ]
    )

    assert state["frontier"] == {"coordinator": coordinator, "child": child}
    assert state["next_role"] == "coordinator"
    assert state["next_cycle"] == 59
    assert state["pending_eval"] is None


def test_dual_policy_mastery_waits_on_eval_without_wait_n_pid_race() -> None:
    launcher = MASTERY_SCRIPT.read_text()

    assert 'wait "$eval_pid"' in launcher
    assert 'kill -TERM "$eval_pid"' in launcher
    assert "wait -n -p completed_pid" not in launcher


def test_recovery_tool_forks_valid_prefix_without_overwriting_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_events = source / "events.jsonl"
    MODULE.append_event(source_events, _initialized())
    MODULE.append_event(
        source_events,
        {
            "kind": "evaluation_completed",
            "cycle": 0,
            "role": "child",
            "phase": "e0c_natural_child",
            "admitted": False,
        },
    )
    source_before = source_events.read_bytes()
    models = {}
    for role in ("coordinator", "child"):
        path = tmp_path / role
        path.mkdir()
        (path / "STABLE").touch()
        weight = f"{role}-weights".encode()
        (path / "model.safetensors").write_bytes(weight)
        models[role] = (path, hashlib.sha256(weight).hexdigest())
    target = tmp_path / "target"

    subprocess.run(
        [
            sys.executable,
            str(RECOVERY_SCRIPT),
            "--source-state-dir",
            str(source),
            "--target-state-dir",
            str(target),
            "--through-sequence",
            "1",
            "--coordinator-model",
            str(models["coordinator"][0]),
            "--coordinator-label",
            "C57",
            "--coordinator-sha256",
            models["coordinator"][1],
            "--child-model",
            str(models["child"][0]),
            "--child-label",
            "K56",
            "--child-sha256",
            models["child"][1],
            "--next-cycle",
            "59",
            "--next-role",
            "coordinator",
            "--recovery-reason",
            "test",
            "--discarded-cycle",
            "58",
            "--discarded-model-sha256",
            "lost",
            "--discarded-evaluation-run",
            "/results/c58",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert source_events.read_bytes() == source_before
    recovered = MODULE.load_events(target / "events.jsonl")
    assert len(recovered) == 3
    assert recovered[-1]["kind"] == "frontier_recovered"
    assert recovered[-1]["duplicate_evaluation_launched"] is False
    state = MODULE.project(recovered)
    assert state["next_cycle"] == 59
    assert state["next_role"] == "coordinator"


def test_admission_keeps_six_examples_but_caps_runtime_pressure() -> None:
    assert MODULE.EVAL_TASKS == 6
    assert MODULE.EVAL_MAX_CONCURRENT == 2
    assert MODULE.EVAL_MAX_ADDRESS_SPACE_BYTES == 32 * 1024**3
    assert MODULE.TRAIN_TIMEOUT_SECONDS == 10800
    assert MODULE.EVAL_TIMEOUT_SECONDS == 1200


def test_project_promotes_only_after_four_qualifiers() -> None:
    output = _candidate("K2")
    prefix = [
        _initialized(),
        {
            "kind": "train_completed",
            "cycle": 1,
            "role": "child",
            "phase": "e0c_natural_child",
            "output_candidate": output,
        },
    ]
    rejected = MODULE.project(
        [
            *prefix,
            {
                "kind": "evaluation_completed",
                "cycle": 1,
                "role": "child",
                "phase": "e0c_natural_child",
                "admitted": False,
            },
        ]
    )
    assert rejected["promoted"]["child"]["label"] == "K1"
    assert rejected["phases"]["child"] == "e0c_natural_child"

    admitted = MODULE.project(
        [
            *prefix,
            {
                "kind": "evaluation_completed",
                "cycle": 1,
                "role": "child",
                "phase": "e0c_natural_child",
                "admitted": True,
                "distinct_qualifying": 4,
                "promotion_minimum": 4,
            },
        ]
    )
    assert admitted["promoted"]["child"] == output
    assert admitted["phases"]["child"] == "e0c2_natural_child_no_template"


def test_event_log_is_hash_chained_and_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    MODULE.append_event(path, _initialized())
    MODULE.append_event(
        path,
        {
            "kind": "train_no_update",
            "cycle": 1,
            "role": "coordinator",
            "phase": MODULE.COORDINATOR_PHASE,
        },
    )
    assert len(MODULE.load_events(path)) == 2
    path.write_text(path.read_text().replace("train_no_update", "train_failed"))
    try:
        MODULE.load_events(path)
    except ValueError as error:
        assert "invalid autonomous event chain" in str(error)
    else:
        raise AssertionError("tampered autonomous event chain was accepted")


def test_runtime_cleanup_selects_only_new_prime_agent_containers() -> None:
    after = {
        "old": {"container_id": "old", "image": "rlm-prime-agent-runtime:v1", "name": "old"},
        "new": {"container_id": "new", "image": "rlm-prime-agent-runtime:v1", "name": "episode"},
        "stock": {"container_id": "stock", "image": "python:3.11-slim", "name": "trace"},
        "db": {"container_id": "db", "image": "postgres:17", "name": "database"},
    }

    assert MODULE._created_runtime_containers({"old"}, after) == [
        after["new"],
        after["stock"],
    ]


def test_qualifying_evaluation_requires_distinct_hard_successes_and_both_routes(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    axis = run / "natural_n1a"
    axis.mkdir(parents=True)
    frontier = {"coordinator": _candidate("C26"), "child": _candidate("K2")}
    (run / "SUMMARY.json").write_text(json.dumps({"episodes": 6, "errors": 0}))
    (run / "VERSIONS.txt").write_text(
        f"coordinator_model_sha256={frontier['coordinator']['model_sha256']}\n"
        f"child_model_sha256={frontier['child']['model_sha256']}\n"
    )
    with (axis / "traces.jsonl").open("w") as handle:
        for index in range(6):
            trace = {
                "is_completed": True,
                "errors": [],
                "task": {"key": f"task-{index}"},
                "rewards": {"harness_score": {"score": 1}},
            }
            handle.write(json.dumps({"ok": True, "errors": [], "traces": [trace]}) + "\n")
    with (run / "ROUTING_AUDIT.jsonl").open("w") as handle:
        for role in ("coordinator", "child"):
            handle.write(json.dumps({"role": role, "status": 200}) + "\n")
        handle.write(json.dumps({"role": "child", "status": 400}) + "\n")

    result = MODULE._qualifying_evaluation(run, frontier)
    assert result["admitted"] is True
    assert result["distinct_qualifying"] == 6
    assert result["promotion_minimum"] == 4
    assert result["role_route_failure_counts"] == {"coordinator": 0, "child": 1}
