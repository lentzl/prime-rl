import importlib.util
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "sync_q35_2b_hf_promotions_v1.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("sync_q35_2b_hf_promotions_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _candidate(tmp_path: Path, role: str) -> dict[str, str]:
    checkpoint = tmp_path / role
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(role.encode())
    (checkpoint / "STABLE").touch()
    return {
        "label": role.upper(),
        "model_path": str(checkpoint),
        "model_sha256": MODULE.controller._sha256(weights),
    }


def _args(tmp_path: Path) -> Namespace:
    secret = tmp_path / "token.json"
    secret.write_text(json.dumps({"token": "secret"}))
    os.chmod(secret, 0o600)
    return Namespace(
        events_file=tmp_path / "events.jsonl",
        publication_events=tmp_path / "publications.jsonl",
        secret_file=secret,
        coordinator_repo="owner/coordinator",
        child_repo="owner/child",
    )


def test_reconcile_publishes_initial_promotions_once(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    candidates = {role: _candidate(tmp_path, role) for role in MODULE.ROLES}
    MODULE.controller.append_event(
        args.events_file,
        {
            "kind": "initialized",
            "frontier": candidates,
            "promoted": candidates,
            "phases": {"coordinator": "e0d3_uncapped_yield_exact_child", "child": "e0c3_natural_child_minimal"},
            "leak_levels": {"coordinator": "action_scaffold", "child": "action_scaffold"},
            "next_role": "coordinator",
            "next_cycle": 1,
            "promotion_minimum": 4,
            "initial_admission": {
                "admitted": True,
                "qualifying": 6,
                "distinct_qualifying": 6,
                "run_dir": "/result",
            },
        },
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        MODULE.publisher,
        "publish_checkpoint",
        lambda **kwargs: calls.append(kwargs) or f"rev-{len(calls)}",
    )
    monkeypatch.setattr(
        MODULE.publisher,
        "remote_checkpoint_sha256",
        lambda **kwargs: candidates["coordinator"]["model_sha256"]
        if kwargs["repo_id"] == "owner/coordinator"
        else candidates["child"]["model_sha256"],
    )

    assert MODULE.reconcile_once(args) == 2
    assert MODULE.reconcile_once(args) == 0
    assert [call["repo_id"] for call in calls] == [
        "owner/coordinator",
        "owner/child",
    ]
    publications = MODULE.load_publications(args.publication_events)
    assert [event["kind"] for event in publications] == [
        "publication_started",
        "publication_completed",
        "publication_started",
        "publication_completed",
    ]


def test_failed_publication_is_recorded_and_retryable(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    candidates = {role: _candidate(tmp_path, role) for role in MODULE.ROLES}
    MODULE.controller.append_event(
        args.events_file,
        {
            "kind": "initialized",
            "frontier": candidates,
            "promoted": candidates,
            "phases": {"coordinator": "e0d3_uncapped_yield_exact_child", "child": "e0c3_natural_child_minimal"},
            "leak_levels": {"coordinator": "action_scaffold", "child": "action_scaffold"},
            "next_role": "coordinator",
            "next_cycle": 1,
            "promotion_minimum": 4,
            "initial_admission": {"admitted": True, "qualifying": 6, "distinct_qualifying": 6},
        },
    )
    attempts = 0

    def publish(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("secret transient failure")
        return f"rev-{attempts}"

    monkeypatch.setattr(MODULE.publisher, "publish_checkpoint", publish)
    monkeypatch.setattr(
        MODULE.publisher,
        "remote_checkpoint_sha256",
        lambda **kwargs: candidates["coordinator"]["model_sha256"]
        if kwargs["repo_id"] == "owner/coordinator"
        else candidates["child"]["model_sha256"],
    )

    assert MODULE.reconcile_once(args) == 1
    assert MODULE.reconcile_once(args) == 1
    publications = MODULE.load_publications(args.publication_events)
    assert publications[1]["kind"] == "publication_failed"
    assert "secret" not in publications[1]["error"]


def test_completed_publication_repairs_remote_head_drift(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    candidates = {role: _candidate(tmp_path, role) for role in MODULE.ROLES}
    MODULE.controller.append_event(
        args.events_file,
        {
            "kind": "initialized",
            "frontier": candidates,
            "promoted": candidates,
            "phases": {"coordinator": "e0d3_uncapped_yield_exact_child", "child": "e0c3_natural_child_minimal"},
            "leak_levels": {"coordinator": "action_scaffold", "child": "action_scaffold"},
            "next_role": "coordinator",
            "next_cycle": 1,
            "promotion_minimum": 4,
            "initial_admission": {"admitted": True, "qualifying": 6, "distinct_qualifying": 6},
        },
    )
    remote = {
        "owner/coordinator": candidates["coordinator"]["model_sha256"],
        "owner/child": candidates["child"]["model_sha256"],
    }
    calls: list[str] = []

    def publish(**kwargs):
        calls.append(kwargs["repo_id"])
        remote[kwargs["repo_id"]] = MODULE.controller._sha256(kwargs["checkpoint_dir"] / "model.safetensors")
        return f"rev-{len(calls)}"

    monkeypatch.setattr(MODULE.publisher, "publish_checkpoint", publish)
    monkeypatch.setattr(
        MODULE.publisher,
        "remote_checkpoint_sha256",
        lambda **kwargs: remote[kwargs["repo_id"]],
    )

    assert MODULE.reconcile_once(args) == 2
    remote["owner/coordinator"] = "stale-sha"
    assert MODULE.reconcile_once(args) == 1
    assert calls == ["owner/coordinator", "owner/child", "owner/coordinator"]
    publications = MODULE.load_publications(args.publication_events)
    repaired = publications[-2]
    assert repaired["kind"] == "publication_started"
    assert repaired["reconciliation_reason"] == "remote_head_missing_or_mismatched"
    assert repaired["observed_remote_model_sha256"] == "stale-sha"
