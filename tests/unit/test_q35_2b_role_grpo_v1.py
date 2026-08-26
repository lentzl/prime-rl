import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/validate_q35_2b_role_grpo_v1.py"
SPEC = importlib.util.spec_from_file_location("validate_q35_2b_role_grpo_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _resolved(
    tmp_path: Path, role: str, phase: str | None = None
) -> tuple[Path, Path, Path, Path]:
    template = ROOT / "experiments/qwen35-2b-self-bootstrap-dual-dense-v1/role-shaped-grpo.toml"
    model = tmp_path / "model"
    model.mkdir()
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    bootstrap = tmp_path / "bootstrap.json"
    bootstrap.write_text(
        json.dumps(
            {
                "schema_version": "qwen35-2b-environment-bootstrap-context/v1",
                "status": "complete",
                "split": "train_gen",
                "master_seed": 20260824,
                "private_payload_mode": "finding_card",
                "leak_level": "action_scaffold",
                "tasks_per_axis": 64,
                "axes": [{"name": "natural_n1a", "start_index": 9100000}],
                "contexts": {f"task-{index}": "context" for index in range(64)},
            }
        )
    )
    scope = {"coordinator": "root", "child": "non_root"}[role]
    source = template.read_text()
    source = source.replace('name = "role-grpo-shadow"', 'name = "test-run"', 1)
    source = source.replace('dir = "role-grpo-shadow"', 'dir = "test-run"', 1)
    source = source.replace('name = "Qwen/Qwen3.5-2B"', f'name = "{model}"', 1)
    source = source.replace('sampled_session_scope = "root"', f'sampled_session_scope = "{scope}"')
    source = source.replace('policy_role = "coordinator"', f'policy_role = "{role}"', 1)
    source = source.replace(
        "leak_coordinator_exact_action = false",
        f"leak_coordinator_exact_action = {'true' if role == 'child' else 'false'}",
        1,
    )
    source = source.replace(
        "leak_child_exact_action = false",
        "leak_child_exact_action = true",
        2,
    )
    source = source.replace(
        "strip_child_tool_choice = false",
        f"strip_child_tool_choice = {'true' if role == 'child' else 'false'}",
        1,
    )
    source = source.replace(
        "strip_coordinator_tool_choice = false",
        "strip_coordinator_tool_choice = false",
        1,
    )
    source = source.replace(
        "enable_thinking = false",
        f"enable_thinking = {'true' if role == 'child' else 'false'}",
        1,
    )
    source = source.replace(
        "max_concurrent = 1",
        f"max_concurrent = {1 if role == 'coordinator' else 2}",
        1,
    )
    if phase == "e0_full_actions":
        source = source.replace("max_completion_tokens = 2048", "max_completion_tokens = 1024", 1)
        source = source.replace("max_turns = 16", "max_turns = 8", 1)
        source = source.replace("max_output_tokens = 16384", "max_output_tokens = 8192", 1)
        source = source.replace("max_total_tokens = 65536", "max_total_tokens = 32768", 1)
        source = source.replace("autonomous_max_turns = 16", "autonomous_max_turns = 8", 1)
        source = source.replace("autonomous_max_tokens = 65536", "autonomous_max_tokens = 32768", 1)
    source = source.replace(
        'anchor_model = "/tmp/q35-2b-role-grpo-anchor"',
        f'anchor_model = "{anchor}"',
        1,
    )
    source = source.replace('privileged_bootstrap_path = "/tmp/q35-2b-role-grpo-bootstrap.json"', f'privileged_bootstrap_path = "{bootstrap}"')
    config = tmp_path / f"{role}.toml"
    config.write_text(source)
    return config, model, anchor, bootstrap


def test_role_grpo_template_audits_both_session_scopes(tmp_path: Path) -> None:
    for role in ("coordinator", "child"):
        role_dir = tmp_path / role
        role_dir.mkdir()
        config, model, anchor, bootstrap = _resolved(role_dir, role)
        report = MODULE.audit(
            config,
            role=role,
            model_path=model,
            anchor_model_path=anchor,
            run_name="test-run",
            bootstrap_path=bootstrap,
            phase=("e0c3_natural_child_minimal" if role == "child" else "e0d3_uncapped_yield_exact_child"),
            start_index=9100000,
            task_count=64,
        )
        assert report["scope"] == {"coordinator": "root", "child": "non_root"}[role]
        assert report["group_size"] == 8
        assert report["max_inflight_episodes"] == 8
        assert report["max_concurrent_agents"] == 2
        assert report["env_server_max_concurrent"] == (
            1 if role == "coordinator" else 2
        )
        assert report["full_dense"] is True
        assert report["coordinator_action_leak"] is (role == "child")
        assert report["child_action_leak"] is True
        assert report["child_action_sampling"] == "synthetic_exact_send"
        assert report["first_action_sampling"] == (
            "prompted_native_spawn" if role == "coordinator" else "masked_frozen_anchor"
        )
        assert report["child_tool_choice_stripped"] is (role == "child")
        assert report["coordinator_tool_choice_stripped"] is False
        assert report["enable_thinking"] is (role == "child")
        assert report["promotion_minimum"] == 4


def test_role_grpo_audits_bounded_exact_child_rung(tmp_path: Path) -> None:
    config, model, anchor, bootstrap = _resolved(tmp_path, "child", "e0_full_actions")
    payload = json.loads(bootstrap.read_text())
    payload["leak_level"] = "solution_replay"
    bootstrap.write_text(json.dumps(payload))

    report = MODULE.audit(
        config,
        role="child",
        model_path=model,
        anchor_model_path=anchor,
        run_name="test-run",
        bootstrap_path=bootstrap,
        phase="e0_full_actions",
        start_index=9100000,
        task_count=64,
    )
    assert report["early_rung_bounded"] is True
    assert report["bootstrap_leak_level"] == "solution_replay"


def test_role_grpo_audits_tapered_child_contract_scaffold(tmp_path: Path) -> None:
    config, model, anchor, bootstrap = _resolved(tmp_path, "coordinator")
    payload = json.loads(bootstrap.read_text())
    payload["leak_level"] = "child_contract_scaffold"
    bootstrap.write_text(json.dumps(payload))

    report = MODULE.audit(
        config,
        role="coordinator",
        model_path=model,
        anchor_model_path=anchor,
        run_name="test-run",
        bootstrap_path=bootstrap,
        phase="e0d3_uncapped_yield_exact_child",
        start_index=9100000,
        task_count=64,
        bootstrap_leak_level="child_contract_scaffold",
    )
    assert report["bootstrap_leak_level"] == "child_contract_scaffold"


def test_coordinator_role_grpo_rejects_unscaffolded_frozen_child(tmp_path: Path) -> None:
    config, model, anchor, bootstrap = _resolved(tmp_path, "coordinator")
    config.write_text(
        config.read_text().replace(
            "leak_child_exact_action = true",
            "leak_child_exact_action = false",
            1,
        )
    )

    try:
        MODULE.audit(
            config,
            role="coordinator",
            model_path=model,
            anchor_model_path=anchor,
            run_name="test-run",
            bootstrap_path=bootstrap,
            phase="e0d3_uncapped_yield_exact_child",
            start_index=9100000,
            task_count=64,
        )
    except MODULE.AuditFailure as error:
        assert "partial causal protocol progress" in str(error)
    else:
        raise AssertionError("unscaffolded frozen child passed the coordinator GRPO audit")


def test_coordinator_role_grpo_rejects_forwarded_child_send(tmp_path: Path) -> None:
    config, model, anchor, bootstrap = _resolved(tmp_path, "coordinator")
    source = config.read_text()
    first = source.index("leak_child_exact_action = true")
    second = source.index("leak_child_exact_action = true", first + 1)
    config.write_text(
        source[:second]
        + source[second:].replace(
            "leak_child_exact_action = true",
            "leak_child_exact_action = false",
            1,
        )
    )

    try:
        MODULE.audit(
            config,
            role="coordinator",
            model_path=model,
            anchor_model_path=anchor,
            run_name="test-run",
            bootstrap_path=bootstrap,
            phase="e0d3_uncapped_yield_exact_child",
            start_index=9100000,
            task_count=64,
        )
    except MODULE.AuditFailure as error:
        assert "synthesize the private child send" in str(error)
    else:
        raise AssertionError("forwarded child send passed the coordinator GRPO audit")


def test_coordinator_role_grpo_rejects_synthetic_root_spawn(tmp_path: Path) -> None:
    config, model, anchor, bootstrap = _resolved(tmp_path, "coordinator")
    config.write_text(
        config.read_text().replace(
            "leak_coordinator_exact_action = false",
            "leak_coordinator_exact_action = true",
            1,
        )
    )

    try:
        MODULE.audit(
            config,
            role="coordinator",
            model_path=model,
            anchor_model_path=anchor,
            run_name="test-run",
            bootstrap_path=bootstrap,
            phase="e0d3_uncapped_yield_exact_child",
            start_index=9100000,
            task_count=64,
        )
    except MODULE.AuditFailure as error:
        assert "tapered curriculum" in str(error)
    else:
        raise AssertionError("synthetic root spawn passed the tapered coordinator audit")


def test_child_role_grpo_rejects_unscaffolded_frozen_coordinator(tmp_path: Path) -> None:
    config, model, anchor, bootstrap = _resolved(tmp_path, "child")
    config.write_text(
        config.read_text().replace(
            "leak_coordinator_exact_action = true",
            "leak_coordinator_exact_action = false",
            1,
        )
    )

    try:
        MODULE.audit(
            config,
            role="child",
            model_path=model,
            anchor_model_path=anchor,
            run_name="test-run",
            bootstrap_path=bootstrap,
            phase="e0c3_natural_child_minimal",
            start_index=9100000,
            task_count=64,
        )
    except MODULE.AuditFailure as error:
        assert "tapered curriculum" in str(error)
    else:
        raise AssertionError("unscaffolded frozen coordinator passed the child audit")


def test_role_grpo_audit_rejects_child_reasoning_mode_mismatch(tmp_path: Path) -> None:
    config, model, anchor, bootstrap = _resolved(tmp_path, "child")
    config.write_text(config.read_text().replace("enable_thinking = true", "enable_thinking = false", 1))

    try:
        MODULE.audit(
            config,
            role="child",
            model_path=model,
            anchor_model_path=anchor,
            run_name="test-run",
            bootstrap_path=bootstrap,
            phase="e0d3_uncapped_yield_exact_child",
            start_index=9100000,
            task_count=64,
        )
    except MODULE.AuditFailure as error:
        assert "preserve reasoning" in str(error)
    else:
        raise AssertionError("child reasoning-mode mismatch passed the role-GRPO audit")


def test_role_grpo_audit_rejects_percentage_only_kv_cache_sizing(tmp_path: Path) -> None:
    config, model, anchor, bootstrap = _resolved(tmp_path, "coordinator")
    config.write_text(
        config.read_text().replace(
            "anchor_kv_cache_memory_bytes = 4294967296\n", "", 1
        )
    )

    try:
        MODULE.audit(
            config,
            role="coordinator",
            model_path=model,
            anchor_model_path=anchor,
            run_name="test-run",
            bootstrap_path=bootstrap,
            phase="e0d3_uncapped_yield_exact_child",
            start_index=9100000,
            task_count=64,
        )
    except MODULE.AuditFailure as error:
        assert "explicit four-GiB KV caches" in str(error)
    else:
        raise AssertionError("percentage-only same-GPU KV-cache sizing passed the role-GRPO audit")


def test_role_grpo_audit_rejects_unsafe_environment_concurrency(tmp_path: Path) -> None:
    config, model, anchor, bootstrap = _resolved(tmp_path, "coordinator")
    config.write_text(
        config.read_text()
        .replace("max_concurrent_agents = 2", "max_concurrent_agents = 8", 1)
    )

    try:
        MODULE.audit(
            config,
            role="coordinator",
            model_path=model,
            anchor_model_path=anchor,
            run_name="test-run",
            bootstrap_path=bootstrap,
            phase="e0d3_uncapped_yield_exact_child",
            start_index=9100000,
            task_count=64,
        )
    except MODULE.AuditFailure as error:
        assert "environment concurrency" in str(error)
    else:
        raise AssertionError("unsafe Prime Agent environment concurrency passed the audit")


def test_role_grpo_audit_rejects_unbounded_env_server_episodes(tmp_path: Path) -> None:
    config, model, anchor, bootstrap = _resolved(tmp_path, "coordinator")
    config.write_text(
        config.read_text().replace("max_concurrent = 1\n", "", 1)
    )

    try:
        MODULE.audit(
            config,
            role="coordinator",
            model_path=model,
            anchor_model_path=anchor,
            run_name="test-run",
            bootstrap_path=bootstrap,
            phase="e0d3_uncapped_yield_exact_child",
            start_index=9100000,
            task_count=64,
        )
    except MODULE.AuditFailure as error:
        assert "EnvServer" in str(error)
    else:
        raise AssertionError("unbounded EnvServer episode concurrency passed the audit")


def test_role_grpo_audit_rejects_role_mixing(tmp_path: Path) -> None:
    config, model, anchor, bootstrap = _resolved(tmp_path, "coordinator")
    source = config.read_text().replace('sampled_session_scope = "root"', 'sampled_session_scope = "all"')
    config.write_text(source)

    try:
        MODULE.audit(
            config,
            role="coordinator",
            model_path=model,
            anchor_model_path=anchor,
            run_name="test-run",
            bootstrap_path=bootstrap,
            phase="e0d3_uncapped_yield_exact_child",
            start_index=9100000,
            task_count=64,
        )
    except MODULE.AuditFailure as error:
        assert "scope mismatch" in str(error)
    else:
        raise AssertionError("role-mixed GRPO config passed its audit")


def test_role_grpo_audit_rejects_unreachable_family_filter(tmp_path: Path) -> None:
    config, model, anchor, bootstrap = _resolved(tmp_path, "coordinator")
    source = config.read_text().replace(
        'curriculum_rung = "natural_n1a"',
        'families = ["natural_n1a"]',
    )
    config.write_text(source)

    try:
        MODULE.audit(
            config,
            role="coordinator",
            model_path=model,
            anchor_model_path=anchor,
            run_name="test-run",
            bootstrap_path=bootstrap,
            phase="e0d3_uncapped_yield_exact_child",
            start_index=9100000,
            task_count=64,
        )
    except MODULE.AuditFailure as error:
        assert "reachable natural_n1a curriculum rung" in str(error)
    else:
        raise AssertionError("unreachable natural_n1a family filter passed its audit")


def test_role_grpo_audit_rejects_bootstrap_seed_drift(tmp_path: Path) -> None:
    config, model, anchor, bootstrap = _resolved(tmp_path, "coordinator")
    payload = json.loads(bootstrap.read_text())
    payload["master_seed"] = 20260819
    bootstrap.write_text(json.dumps(payload))

    try:
        MODULE.audit(
            config,
            role="coordinator",
            model_path=model,
            anchor_model_path=anchor,
            run_name="test-run",
            bootstrap_path=bootstrap,
            phase="e0d3_uncapped_yield_exact_child",
            start_index=9100000,
            task_count=64,
        )
    except MODULE.AuditFailure as error:
        assert "generator coordinates" in str(error)
    else:
        raise AssertionError("bootstrap seed drift passed the role-GRPO audit")


def test_role_grpo_audit_rejects_live_evidence_mode_drift(tmp_path: Path) -> None:
    config, model, anchor, bootstrap = _resolved(tmp_path, "coordinator")
    config.write_text(
        config.read_text().replace(
            'private_payload_mode = "finding_card"',
            'private_payload_mode = "raw_resource"',
        )
    )

    try:
        MODULE.audit(
            config,
            role="coordinator",
            model_path=model,
            anchor_model_path=anchor,
            run_name="test-run",
            bootstrap_path=bootstrap,
            phase="e0d3_uncapped_yield_exact_child",
            start_index=9100000,
            task_count=64,
        )
    except MODULE.AuditFailure as error:
        assert "finding-card bootstrap" in str(error)
    else:
        raise AssertionError("live evidence mode drift passed the role-GRPO audit")
