from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REWARD_PATH = REPO / "patches/verifiers/environments/source_worker_first_call_v1/source_worker_first_call_v1/reward.py"
SPEC = importlib.util.spec_from_file_location("source_worker_first_call_reward", REWARD_PATH)
assert SPEC is not None and SPEC.loader is not None
REWARD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REWARD
SPEC.loader.exec_module(REWARD)
CellEvidence = REWARD.CellEvidence
score_first_call = REWARD.score_first_call
is_designated_source_inspector_task = REWARD.is_designated_source_inspector_task


AST_PATHS = ("/workspace/sample/alpha.py", "/workspace/sample/beta.py")
CONFIG_PATHS = ("/workspace/sample/service.toml", "/workspace/sample/features.env")
LAUNCHER_PATH = REPO / "scripts/run_q35_2b_source_first_call_grpo_s6_v1.sh"
VALIDATOR_PATH = REPO / "scripts/validate_q35_2b_source_first_call_grpo_s6_v1.py"
PROXY_PATH = REPO / "scripts/dual_policy_openai_proxy_v1.py"
P0_VALIDATOR_PATH = REPO / "scripts/validate_q35_2b_source_route_parity_p0_v1.py"
P0_LAUNCHER_PATH = REPO / "scripts/run_q35_2b_source_route_parity_p0_v1.sh"
EXPERIMENT = REPO / "experiments/qwen35-2b-document-recursion-zero-update-v1"
P0_CONFIG_PATH = EXPERIMENT / "specialist-source-route-parity-p0-lr0.toml"
P1_CONFIG_PATH = EXPERIMENT / "specialist-source-route-parity-p1-lr0.toml"
P1_VALIDATOR_PATH = REPO / "scripts/validate_q35_2b_source_route_parity_p1_v1.py"
P1_LAUNCHER_PATH = REPO / "scripts/run_q35_2b_source_route_parity_p1_v1.sh"
P1_HASHER_PATH = REPO / "scripts/hash_q35_2b_source_route_parity_p1_tasks_v1.py"
P2_CONFIG_PATH = EXPERIMENT / "specialist-source-route-parity-p2-lr0.toml"
P2_VALIDATOR_PATH = REPO / "scripts/validate_q35_2b_source_route_parity_p2_v1.py"
P2_LAUNCHER_PATH = REPO / "scripts/run_q35_2b_source_route_parity_p2_v1.sh"
P2_HASHER_PATH = REPO / "scripts/hash_q35_2b_source_route_parity_p2_tasks_v1.py"
P2R_CONFIG_PATH = EXPERIMENT / "specialist-source-route-parity-p2r-lr0.toml"
P2R_VALIDATOR_PATH = REPO / "scripts/validate_q35_2b_source_route_parity_p2r_v1.py"
P2R_LAUNCHER_PATH = REPO / "scripts/run_q35_2b_source_route_parity_p2r_v1.sh"
P2_FAILED_START_PATH = EXPERIMENT / "specialist-source-route-parity-p2-failed-start-v1.json"
P1_FORENSIC_PATH = REPO / "scripts/recover_q35_2b_source_route_parity_p1_v1.py"


def _validator_module():
    spec = importlib.util.spec_from_file_location("source_first_call_validator", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _proxy_module():
    spec = importlib.util.spec_from_file_location("source_first_call_proxy", PROXY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _p0_validator_module():
    spec = importlib.util.spec_from_file_location(
        "source_route_parity_p0_validator", P0_VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _p1_hasher_module():
    spec = importlib.util.spec_from_file_location("p1_task_hasher_test", P1_HASHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _p2_validator_module():
    spec = importlib.util.spec_from_file_location(
        "source_route_parity_p2_validator", P2_VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _p2r_validator_module():
    spec = importlib.util.spec_from_file_location(
        "source_route_parity_p2r_validator", P2R_VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _p2_hasher_module():
    spec = importlib.util.spec_from_file_location("p2_task_hasher_test", P2_HASHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_atomic_ast_loop_receives_full_reward() -> None:
    code = """import ast, json
from pathlib import Path
paths = ('/workspace/sample/alpha.py', '/workspace/sample/beta.py')
trees = [ast.parse(Path(path).read_text()) for path in paths]
value = sum(isinstance(node, ast.FunctionDef) for tree in trees for node in ast.walk(tree))
await agent_message.send(json.dumps({'value': value}, separators=(',', ':')), receiver_role='parent')
"""
    result = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(CellEvidence(code, "Agent message sent: agentmsg_123"),),
        delivered_bodies=('{"value":30}',),
    )
    assert result.score == pytest.approx(1.0)
    assert result.atomic_compact_parent_send == 1


def test_unused_path_literals_cannot_earn_file_api_credit() -> None:
    code = """import ast
from pathlib import Path
claimed = ('/workspace/sample/alpha.py', '/workspace/sample/beta.py')
tree = ast.parse(Path('/workspace/sample/other.py').read_text())
"""
    result = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(CellEvidence(code, "ok"),),
        delivered_bodies=(),
    )
    assert result.exception_free_first_call == 0
    assert result.correct_file_api == 0
    assert result.score == pytest.approx(0.0)


def test_two_hop_loop_proves_each_path_is_read() -> None:
    code = """import ast
from pathlib import Path
paths = ('/workspace/sample/alpha.py', '/workspace/sample/beta.py')
trees = []
for path in paths:
    trees.append(ast.parse(Path(path).read_text()))
value = sum(isinstance(node, ast.FunctionDef) for tree in trees for node in ast.walk(tree))
"""
    result = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(CellEvidence(code, "ok"),),
        delivered_bodies=(),
    )
    assert result.correct_file_api == 1
    assert result.score == pytest.approx(0.3)


@pytest.mark.parametrize(
    "toml_expression",
    (
        "tomllib.loads(Path('/workspace/sample/service.toml').read_text())",
        "tomllib.load(open('/workspace/sample/service.toml', 'rb'))",
    ),
)
def test_config_accepts_text_and_binary_tomllib_reads(toml_expression: str) -> None:
    code = f"""import tomllib
from pathlib import Path
config = {toml_expression}
features = dict(line.split('=', 1) for line in Path('/workspace/sample/features.env').read_text().splitlines())
value = config['runtime']['workers'] * config['runtime']['timeout_seconds'] + sum(value == 'true' for value in features.values())
"""
    result = score_first_call(
        family="specialist_source_config",
        required_paths=CONFIG_PATHS,
        expected_value=30,
        cells=(CellEvidence(code, "ok"),),
        delivered_bodies=(),
    )
    assert result.correct_file_api == 1
    assert result.score == pytest.approx(0.3)


def test_later_exact_recovery_cannot_outrank_correct_first_call() -> None:
    correct_first_call = """import ast
from pathlib import Path
trees = [ast.parse(Path(path).read_text()) for path in ('/workspace/sample/alpha.py', '/workspace/sample/beta.py')]
value = sum(isinstance(node, ast.FunctionDef) for tree in trees for node in ast.walk(tree))
"""
    baseline = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(CellEvidence(correct_first_call, "ok"),),
        delivered_bodies=(),
    )
    recovered = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(
            CellEvidence("value = 29", "29"),
            CellEvidence(
                "await agent_message.send('{\"value\":30}', receiver_role='parent')",
                "Agent message sent: agentmsg_456",
            ),
        ),
        delivered_bodies=('{"value":30}',),
    )
    assert baseline.score == pytest.approx(0.3)
    assert recovered.exact_oracle_value == 1
    assert recovered.atomic_compact_parent_send == 0
    assert recovered.score == pytest.approx(-0.01)
    assert recovered.score < baseline.score


def test_extra_send_prevents_atomic_credit_and_is_penalized() -> None:
    first = """import ast
from pathlib import Path
trees = [ast.parse(Path(path).read_text()) for path in ('/workspace/sample/alpha.py', '/workspace/sample/beta.py')]
value = sum(isinstance(node, ast.FunctionDef) for tree in trees for node in ast.walk(tree))
await agent_message.send('{"value":30}', receiver_role='parent')
"""
    second = "await agent_message.send('{\"value\":30}', receiver_role='parent')"
    result = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(
            CellEvidence(first, "Agent message sent: agentmsg_1"),
            CellEvidence(second, "Agent message sent: agentmsg_2"),
        ),
        delivered_bodies=('{"value":30}',),
    )
    assert result.atomic_compact_parent_send == 0
    assert result.extra_sends == 1
    assert result.retries == 1
    assert result.score == pytest.approx(0.26)


@pytest.mark.parametrize(
    "code",
    (
        "",
        "rlm.list_subagents()",
        "await agent_observe('child')",
        "print('conversation log')",
        "goal = 'inspect source eventually'",
    ),
)
def test_silence_and_control_plane_inspection_earn_no_positive_credit(code: str) -> None:
    cells = () if not code else (CellEvidence(code, "ok"),)
    result = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=cells,
        delivered_bodies=(),
    )
    assert result.exception_free_first_call == 0
    assert result.correct_file_api == 0
    assert result.score == pytest.approx(0.0)


def test_reads_without_family_computation_earn_no_positive_credit() -> None:
    code = """from pathlib import Path
texts = [Path(path).read_text() for path in ('/workspace/sample/alpha.py', '/workspace/sample/beta.py')]
"""
    result = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(CellEvidence(code, "ok"),),
        delivered_bodies=(),
    )
    assert result.exception_free_first_call == 0
    assert result.score == pytest.approx(0.0)


def test_partial_correct_source_work_outranks_silence() -> None:
    code = """import ast
from pathlib import Path
trees = [ast.parse(Path(path).read_text()) for path in ('/workspace/sample/alpha.py', '/workspace/sample/beta.py')]
value = sum(isinstance(node, ast.FunctionDef) for tree in trees for node in ast.walk(tree))
"""
    partial = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(CellEvidence(code, "ok"),),
        delivered_bodies=(),
    )
    silence = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(),
        delivered_bodies=(),
    )
    assert partial.exception_free_first_call == 1
    assert partial.correct_file_api == 1
    assert partial.score == pytest.approx(0.3)
    assert partial.score > silence.score


def test_only_typed_source_inspector_assignment_is_designated() -> None:
    typed = """[task from parent]
[selected terminal capability]
expert_id=source_inspector
session_role=terminal_worker
Read both files.
"""
    assert is_designated_source_inspector_task(typed)
    assert not is_designated_source_inspector_task("[task from parent]\nInspect files")
    assert not is_designated_source_inspector_task(
        typed.replace("expert_id=source_inspector", "expert_id=table_analyst")
    )


def _healthy_lr0_metrics() -> list[dict[str, float]]:
    return [
        {
            "entropy/all/mean": 1.0,
            "entropy/all/max": 8.0,
            "unmasked_mismatch_kl/mean": 0.1,
            "mismatch_kl/all/max": 20.0,
            "is_masked/mean": 0.01,
        }
    ]


def _lr0_traces(*, all_max_turns: bool) -> list[dict[str, object]]:
    traces = []
    for family in ("specialist_source_ast", "specialist_source_config"):
        for index in range(8):
            traces.append(
                {
                    "task": {"data": {"family": family}},
                    "stop_condition": (
                        "max_turns" if all_max_turns or index else "agent_completed"
                    ),
                }
            )
    return traces


def test_prospective_lr0_health_requires_uncensored_trace_per_family() -> None:
    validator = _validator_module()
    with pytest.raises(validator.AuditFailure, match="100% max-turn censored"):
        validator._validate_prospective_lr0_health(
            _healthy_lr0_metrics(), _lr0_traces(all_max_turns=True)
        )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("unmasked_mismatch_kl/mean", 0.3, "unmasked trainer/inference mismatch"),
        ("mismatch_kl/all/max", 900.0, "mismatch-KL outliers"),
        ("is_masked/mean", 0.06, "masks too much"),
    ),
)
def test_prospective_lr0_health_rejects_pathological_mismatch(
    key: str, value: float, message: str
) -> None:
    validator = _validator_module()
    metrics = _healthy_lr0_metrics()
    metrics[0][key] = value
    with pytest.raises(validator.AuditFailure, match=message):
        validator._validate_prospective_lr0_health(
            metrics, _lr0_traces(all_max_turns=False)
        )


def test_prospective_lr0_health_accepts_finite_aligned_uncensored_control() -> None:
    validator = _validator_module()
    report = validator._validate_prospective_lr0_health(
        _healthy_lr0_metrics(), _lr0_traces(all_max_turns=False)
    )
    assert report["unmasked_mismatch_to_entropy_ratio"] == pytest.approx(0.1)
    assert all(
        counts["agent_completed"] == 1
        for counts in report["stop_conditions"].values()
    )


def test_p0_observes_pathological_lr0_values_without_applying_thresholds() -> None:
    validator = _validator_module()
    metrics = _healthy_lr0_metrics()
    metrics[0].update(
        {
            "entropy/all/std": 0.0,
            "unmasked_mismatch_kl/mean": 999.0,
            "unmasked_mismatch_kl/max": 9999.0,
            "mismatch_kl/all/mean": 1000.0,
            "mismatch_kl/all/std": 500.0,
            "mismatch_kl/all/max": 10000.0,
            "masked_mismatch_kl/mean": 2000.0,
            "masked_mismatch_kl/max": 20000.0,
            "is_masked/max": 1.0,
            "is_masked_low/mean": 0.5,
            "is_masked_low/max": 1.0,
            "is_masked_high/mean": 0.5,
            "is_masked_high/max": 1.0,
            "kl_ent_ratio/mean": 999.0,
        }
    )
    for metric in ("entropy", "mismatch_kl"):
        for source in ("source-worker-ast-s6", "source-worker-config-s6"):
            for statistic, value in (("mean", 1.0), ("std", 2.0), ("max", 3.0)):
                metrics[0][f"{metric}/{source}/{statistic}"] = value
    report = validator._observe_lr0_health(
        metrics, _lr0_traces(all_max_turns=True)
    )

    assert report["thresholds_evaluated"] is False
    assert report["thresholds_frozen"] is False
    assert report["max_turn_fraction"] == {
        "specialist_source_ast": 1.0,
        "specialist_source_config": 1.0,
    }
    assert report["metrics"]["unmasked_mismatch_kl/mean"] == 999.0
    assert report["metrics"]["entropy/source-worker-ast-s6/std"] == 2.0
    assert report["metrics"]["mismatch_kl/source-worker-config-s6/max"] == 3.0


def test_p0_config_is_unique_old_seed_lr0_observational_calibration() -> None:
    validator = _p0_validator_module()
    report = validator.validate_config(
        Path(
            "experiments/qwen35-2b-document-recursion-zero-update-v1/"
            "specialist-source-route-parity-p0-lr0.toml"
        )
    )
    config = P0_CONFIG_PATH.read_text()

    assert report["learning_rate"] == 0.0
    assert report["checkpoint_enabled"] is False
    assert report["observational_only"] is True
    assert report["thresholds_evaluated"] is False
    assert report["optimizer_update_authorized"] is False
    assert config.count("seed = 20270909") == 2
    assert config.count("instance_offset = 70000") == 2
    assert "[trainer.ckpt]" not in config
    assert "[orchestrator.ckpt]" not in config
    assert "q35-2b-source-route-parity-p0-v1" in config


def test_p0_records_zero_reward_variance_without_rejecting_it() -> None:
    validator = _p0_validator_module()
    traces = [
        {
            "id": f"trace-{index}",
            "task": {
                "key": f"task-{index // 8}",
                "data": {
                    "family": (
                        "specialist_source_ast"
                        if index < 8
                        else "specialist_source_config"
                    )
                },
            },
            "rewards": {"source_worker_first_call": {"score": 0.0}},
        }
        for index in range(16)
    ]

    report = validator._reward_observations(traces)

    assert report["overall"]["population_variance"] == 0.0
    assert report["overall"]["unique_values"] == [0.0]
    assert report["variance_is_observational"] is True


def test_p0_launcher_is_write_once_lr0_only_and_rehashes_protected_models() -> None:
    launcher = P0_LAUNCHER_PATH.read_text()
    validator = P0_VALIDATOR_PATH.read_text()

    assert "specialist-source-route-parity-p0-lr0.toml" in launcher
    assert "specialist-source-competence-s6-first-call-grpo-step1.toml" not in launcher
    assert "run_q35_2b_specialist_competence_eval_v1.sh" not in launcher
    assert "refusing duplicate or partial P0 output/result root" in launcher
    assert "exactly 2x RTX A6000" in launcher
    assert "--runtime" in launcher
    assert launcher.count("rehash_protected_models") == 4
    assert "--output \"$calibration_result\"" in launcher
    assert '"optimizer_steps_executed": 1' in validator
    assert '"learning_rate": 0.0' in validator
    assert '"persisted_optimizer_state": False' in validator
    assert '"model_weight_identity_verified_pre_and_post": True' in validator
    assert 'for source in EXPECTED_SOURCE_FAMILIES' in (
        REPO / "scripts/validate_q35_2b_source_first_call_grpo_s6_v1.py"
    ).read_text()


def test_launcher_uses_source_paths_without_dependency_overlay() -> None:
    launcher = LAUNCHER_PATH.read_text()
    assert 's6_pythonpath="$root/src:$root/packages/prime-rl-configs/src:' in launcher
    assert 'export PYTHONPATH="$s6_pythonpath${PYTHONPATH:+:$PYTHONPATH}"' in launcher
    assert "--with-editable" not in launcher


def test_launcher_checks_s5_exactness_at_task_schema_location() -> None:
    launcher = LAUNCHER_PATH.read_text()
    assert 'treatment_tasks = summary.get("treatment", {}).get("tasks", [])' in launcher
    assert 'task.get("answer_accuracy") != 0' in launcher
    assert 'get("exact_answers")' not in launcher


def test_launcher_uses_nested_run_roots_and_fail_closed_audit_resume() -> None:
    launcher = LAUNCHER_PATH.read_text()
    assert "audit_run=$audit_output/source-first-call-s6-zero-lr-audit" in launcher
    assert "update_run=$update_output/source-first-call-s6-step1" in launcher
    assert "candidate=$update_run/weights/step_1" in launcher
    assert 'run_s6_python "$validator" "$audit_run"' in launcher
    assert 'run_s6_python "$validator" "$update_run" --runtime --stage update' in launcher
    assert "resume_after_audit=${S6_RESUME_AFTER_AUDIT:-false}" in launcher
    assert "S6 resume requires exactly one unvalidated completed audit" in launcher


def test_generate_routes_exact_specialist_child_without_rewriting_it(tmp_path: Path) -> None:
    proxy_module = _proxy_module()
    marker = [77, 78]
    assignment = {
        "worker_name": "task-worker",
        "objective": "Inspect both files and send one compact JSON report.",
        "paths": [
            "/workspace/specialist-worker/train/a.py",
            "/workspace/specialist-worker/train/b.py",
        ],
    }
    coordinator_prompt = "\n".join(
        (
            proxy_module.SPECIALIST_WORKER_ROUTING_HEADER,
            proxy_module.LOCAL_COGNITION_FACTS_HEADER,
            "owns_required_evidence=false",
            "remaining_work_requires_decomposition=false",
            "terminal_shards_ready=true",
            "[capability registry]",
            json.dumps({"expert_id": "generic_worker"}),
            json.dumps({"expert_id": "source_inspector"}),
            proxy_module.SPECIALIST_ASSIGNMENT_HEADER,
            json.dumps(assignment, separators=(",", ":")),
        )
    )

    class Tokenizer:
        def decode(self, token_ids, *, skip_special_tokens):
            assert marker[0] not in token_ids, "child request entered coordinator rewrite"
            assert skip_special_tokens is False
            return coordinator_prompt

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            assert "expert_id=source_inspector" in text
            return [10, 11]

        def convert_tokens_to_ids(self, token):
            assert token == "<|im_end|>"
            return 12

    upstream_body = json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "token_ids": [88],
                    "logprobs": {
                        "content": [{"token": "token_id:88", "logprob": -0.2}]
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    ).encode()

    class Upstream:
        status = 200
        headers = {"content-type": "application/json"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def read(self):
            return upstream_body

    class Client:
        def __init__(self):
            self.calls = []

        def post(self, url, *, data, headers):
            self.calls.append((url, json.loads(data), dict(headers)))
            return Upstream()

    class Request:
        def __init__(self, token_ids, session, client_session=None):
            self.headers = {"x-session-id": session}
            if client_session is not None:
                self.headers["x-client-session-id"] = client_session
            self._body = json.dumps(
                {"model": "external", "token_ids": token_ids}
            ).encode()

        async def read(self):
            return self._body

    proxy = proxy_module.DualPolicyProxy(
        coordinator_url="http://coordinator/v1",
        coordinator_model="e33",
        child_url="http://child/v1",
        child_model="S5",
        external_model="external",
        audit_log=tmp_path / "routing.jsonl",
        private_evidence_token_ids=[99],
        specialist_child_token_ids=marker,
        tokenizer=Tokenizer(),
        specialist_routes={"source_inspector": ("http://child/v1", "S5")},
        specialist_fixed_expert="source_inspector",
        specialist_force_fixed_action=True,
        specialist_worker_routing=True,
        require_client_session_id=True,
    )
    client = Client()
    proxy.client = client

    forced = asyncio.run(
        proxy.generate(Request([1, 2, 3], "rollout-root", "root-branch"))
    )
    child = asyncio.run(
        proxy.generate(Request([4, *marker, 5], "rollout-child", "child-branch"))
    )

    assert json.loads(forced.body)["choices"][0]["token_ids"] == [10, 11, 12]
    assert child.body == upstream_body
    assert len(client.calls) == 1
    assert client.calls[0][0] == "http://child/inference/v1/generate"
    assert client.calls[0][1]["model"] == "S5"
    events = [
        json.loads(line)
        for line in (tmp_path / "routing.jsonl").read_text().splitlines()
    ]
    assert events[0]["role"] == "coordinator"
    assert events[0]["upstream_model"] == "e33"
    assert events[0]["session_sha256"] == hashlib.sha256(b"rollout-root").hexdigest()
    assert events[0]["branch_session_sha256"] == hashlib.sha256(
        b"root-branch"
    ).hexdigest()
    assert events[1]["role"] == "child"
    assert events[1]["expert_id"] == "source_inspector"
    assert events[1]["upstream_model"] == "S5"
    assert events[1]["route_evidence"] == "exact_specialist_child_prefix"
    assert events[1]["session_sha256"] != events[1]["branch_session_sha256"]
    with pytest.raises(ValueError, match="lacks x-client-session-id"):
        asyncio.run(proxy.generate(Request([1, 2, 3], "missing-branch")))


def test_runtime_validator_rejects_non_source_reward_leakage() -> None:
    validator = VALIDATOR_PATH.read_text()
    assert "if name != EXPECTED_REWARD" in validator
    assert 'payload.get("score") not in (0, 0.0)' in validator
    assert "has non-S6 reward leakage" in validator


@pytest.mark.parametrize(
    ("filename", "stage"),
    (
        ("specialist-source-competence-s6-first-call-grpo-zero-lr.toml", "audit"),
        ("specialist-source-competence-s6-first-call-grpo-step1.toml", "update"),
    ),
)
def test_s6_resolved_selection_is_one_eight_way_group_per_family(filename: str, stage: str) -> None:
    validator = _validator_module()
    report = validator.validate_config(EXPERIMENT / filename, stage)

    assert report["family_sources"] == {
        "source-worker-ast-s6": "specialist_source_ast",
        "source-worker-config-s6": "specialist_source_config",
    }
    assert report["batch_source_minimums"] == {
        "source-worker-ast-s6": 8,
        "source-worker-config-s6": 8,
    }
    assert report["batch_size"] == 16
    assert report["group_size"] == 8


def test_s6_rejects_duplicate_family_sources_before_launch(tmp_path: Path) -> None:
    validator = _validator_module()
    source = EXPERIMENT / "specialist-source-competence-s6-first-call-grpo-zero-lr.toml"
    broken = source.read_text().replace(
        'families = ["specialist_source_config"]',
        'families = ["specialist_source_ast"]',
    )
    path = tmp_path / "duplicate-family.toml"
    path.write_text(broken)

    with pytest.raises(validator.AuditFailure, match="isolated reward taskset"):
        validator.validate_config(path, "audit")


def test_runtime_validator_matches_forced_routes_to_every_trace_and_group() -> None:
    validator = _validator_module()
    traces = []
    audits = []
    for group in range(2):
        code = (
            "task_worker = await rlm("
            f"'[selected terminal capability]\\nexpert_id=source_inspector\\ngroup={group}', "
            'name="task-worker")'
        )
        action_sha = hashlib.sha256(code.encode()).hexdigest()
        for rollout in range(8):
            client_session_id = f"session-{group}-{rollout}"
            traces.append(
                {
                    "task": {"key": f"group-{group}"},
                    "nodes": [
                        {
                            "sampled": True,
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "name": "ipython",
                                        "arguments": json.dumps({"code": code}),
                                    }
                                ],
                            },
                        }
                    ],
                    "calls": [{"node": 0, "client_session_id": client_session_id}],
                }
            )
            audits.append(
                {
                    "schema_version": "qwen35-2b-dual-policy-route/v1",
                    "mode": validator.FORCED_ROUTE_MODE,
                    "endpoint": "/inference/v1/generate",
                    "role": "coordinator",
                    "expert_id": "source_inspector",
                    "status": 200,
                    "session_sha256": hashlib.sha256(client_session_id.encode()).hexdigest(),
                    "action_sha256": action_sha,
                }
            )

    traces[0]["nodes"].append(
        {
            "sampled": False,
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "ipython",
                        "arguments": json.dumps(
                            {
                                "code": (
                                    "task_worker = await rlm("
                                    "'[selected terminal capability]\\n"
                                    "expert_id=source_inspector\\ngroup=0', "
                                    'name="task-worker")'
                                )
                            }
                        ),
                    }
                ],
            },
        }
    )

    report = validator._validate_forced_assignment_routes(traces, audits)
    assert report["events"] == 16
    assert report["matched_effective_events"] == 16
    assert report["extra_filtered_events"] == 0
    assert report["sessions"] == 16
    assert report["groups"] == 2
    assert sorted(report["effective_action_sha256_counts"].values()) == [8, 8]

    extra_session = "filtered-raw-session"
    report = validator._validate_forced_assignment_routes(
        traces,
        [
            *audits,
            {
                **audits[0],
                "session_sha256": hashlib.sha256(extra_session.encode()).hexdigest(),
            },
        ],
    )
    assert report["events"] == 17
    assert report["extra_filtered_events"] == 1

    with pytest.raises(validator.AuditFailure, match="more than once in a session"):
        validator._validate_forced_assignment_routes(traces, [*audits, audits[0]])

    with pytest.raises(
        validator.AuditFailure,
        match="lack forced-assignment route action multiplicity",
    ):
        validator._validate_forced_assignment_routes(traces, audits[:-1])


def test_runtime_validator_rejects_child_or_unmatched_forced_route() -> None:
    validator = _validator_module()
    code = (
        "task_worker = await rlm('[selected terminal capability]\\nexpert_id=source_inspector', name=\"task-worker\")"
    )
    action_sha = hashlib.sha256(code.encode()).hexdigest()
    traces = []
    for index in range(16):
        traces.append(
            {
                "task": {"key": f"group-{index // 8}"},
                "nodes": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "ipython",
                                        "arguments": json.dumps({"code": code}),
                                    }
                                }
                            ],
                        }
                    }
                ],
                "calls": [{"node": 0, "client_session_id": f"session-{index}"}],
            }
        )
    audits = [
        {
            "schema_version": "qwen35-2b-dual-policy-route/v1",
            "mode": validator.FORCED_ROUTE_MODE,
            "endpoint": "/inference/v1/generate",
            "role": "child" if index == 0 else "coordinator",
            "expert_id": "source_inspector",
            "status": 200,
            "session_sha256": hashlib.sha256(f"session-{index}".encode()).hexdigest(),
            "action_sha256": action_sha,
        }
        for index in range(16)
    ]
    with pytest.raises(validator.AuditFailure, match="invalid forced-assignment"):
        validator._validate_forced_assignment_routes(traces, audits)


def _effective_route_fixture(validator):
    traces = []
    audits = []
    for index in range(16):
        session_sha = hashlib.sha256(f"rollout-{index}".encode()).hexdigest()
        traces.append(
            {
                "nodes": [
                    {
                        "parent": None,
                        "message": {"role": "user", "content": "root system"},
                    },
                    {
                        "parent": 0,
                        "message": {
                            "role": "user",
                            "content": "[specialist worker routing contract]",
                        },
                    },
                    {
                        "parent": 1,
                        "message": {"role": "assistant", "content": None},
                    },
                    {
                        "parent": None,
                        "message": {"role": "user", "content": "child system"},
                    },
                    {
                        "parent": 3,
                        "message": {
                            "role": "user",
                            "content": validator.SPECIALIST_CHILD_PREFIX
                            + "\nis_root=false\nassigned paths",
                        },
                    },
                    {
                        "parent": 4,
                        "message": {"role": "assistant", "content": None},
                    },
                ],
                "calls": [
                    {"node": 2, "client_session_id": f"root-{index}"},
                    {"node": 5, "client_session_id": f"child-{index}"},
                ],
            }
        )
        audits.extend(
            (
                {
                    "schema_version": validator.ROUTE_SCHEMA,
                    "endpoint": "/inference/v1/generate",
                    "status": 200,
                    "role": "coordinator",
                    "mode": validator.FORCED_ROUTE_MODE,
                    "session_sha256": session_sha,
                    "upstream_model": str(validator.E33_PATH),
                    "expert_id": "source_inspector",
                    "route_evidence": "coordinator_without_specialist_child_prefix",
                },
                {
                    "schema_version": validator.ROUTE_SCHEMA,
                    "endpoint": "/inference/v1/generate",
                    "status": 200,
                    "role": "child",
                    "mode": "forwarded_without_tool_choice",
                    "session_sha256": session_sha,
                    "upstream_model": str(validator.S5_PATH),
                    "expert_id": "source_inspector",
                    "route_evidence": "exact_specialist_child_prefix",
                },
            )
        )
    return traces, audits


def _p0_terminal_route_fixture(validator):
    traces = []
    audits = []
    sequence = 0
    for index in range(16):
        session_sha = hashlib.sha256(f"p0-rollout-{index}".encode()).hexdigest()
        nodes = [
            {
                "parent": None,
                "message": {
                    "role": "user",
                    "content": (
                        f"/tmp/vf-prime-agent-runs/{session_sha[:16]}/conversation.json\n"
                        "[specialist worker routing contract]"
                    ),
                },
            },
            {
                "parent": None,
                "message": {"role": "user", "content": "child system"},
            },
            {
                "parent": 1,
                "message": {
                    "role": "user",
                    "content": validator.SPECIALIST_CHILD_PREFIX
                    + "\nis_root=false\nassigned paths",
                },
            },
        ]
        coordinator_count = 3 + int(index < 7)
        child_count = 4 + int(index < 9)
        calls = [
            {"node": 0, "error": None, "client_session_id": f"p0-{index}"}
            for _ in range(coordinator_count)
        ] + [
            {"node": 2, "error": None, "client_session_id": f"p0-{index}"}
            for _ in range(child_count)
        ]
        stop_condition = "max_turns"
        residual_role = None
        residual_error = None
        if index == 0:
            stop_condition = "agent_completed"
            residual_role = "child"
            residual_error = {
                "type": "ClientConnectionResetError",
                "message": "Cannot write to closing transport",
            }
        elif index in {1, 12}:
            residual_role = "coordinator"
        elif index == 15:
            residual_role = "child"
        if residual_role is not None:
            calls.append(
                {
                    "node": None,
                    "error": residual_error,
                    "finish_reason": "stop" if index == 0 else "tool_calls",
                    "time": {"start": float(index), "end": float(index + 1)},
                }
            )
        traces.append(
            {
                "id": f"p0-trace-{index}",
                "nodes": nodes,
                "calls": calls,
                "stop_condition": stop_condition,
            }
        )

        roles = ["coordinator"] * coordinator_count + ["child"] * child_count
        if residual_role is not None:
            roles.append(residual_role)
        if index == 8:
            roles.append("child")
        for role_index, role in enumerate(roles):
            forced = role == "coordinator" and role_index == 0
            audits.append(
                {
                    "schema_version": validator.ROUTE_SCHEMA,
                    "endpoint": "/inference/v1/generate",
                    "status": 200,
                    "role": role,
                    "mode": validator.FORCED_ROUTE_MODE
                    if forced
                    else "forwarded_without_tool_choice",
                    "session_sha256": session_sha,
                    "upstream_model": str(
                        validator.E33_PATH if role == "coordinator" else validator.S5_PATH
                    ),
                    "expert_id": "source_inspector" if role == "child" or forced else None,
                    "route_evidence": (
                        "coordinator_without_specialist_child_prefix"
                        if role == "coordinator"
                        else "exact_specialist_child_prefix"
                    ),
                    "sequence": sequence,
                    "request_sha256": hashlib.sha256(
                        f"p0-request-{sequence}".encode()
                    ).hexdigest(),
                    "latency_ms": 36_410.419 if index == 8 and role_index == len(roles) - 1 else 1.0,
                }
            )
            sequence += 1
    return traces, audits


def test_runtime_validator_reconciles_every_child_and_coordinator_route() -> None:
    validator = _validator_module()
    traces, audits = _effective_route_fixture(validator)

    report = validator._validate_effective_call_routes(traces, audits)

    assert report["expected_calls"] == {"coordinator": 16, "child": 16}
    assert report["observed_routes"] == report["expected_calls"]
    assert report["forced_assignments"] == 16
    assert report["rollout_sessions"] == 16


def test_runtime_validator_requires_raw_and_effective_trace_identity() -> None:
    validator = _validator_module()
    effective = [{"id": f"trace-{index}"} for index in range(16)]
    report = validator._validate_raw_equals_effective(
        list(reversed(effective)), effective
    )
    assert report == {"raw": 16, "effective": 16, "identical": True}

    with pytest.raises(validator.AuditFailure, match="identical raw and effective"):
        validator._validate_raw_equals_effective(
            [*effective, {"id": "filtered-extra"}], effective
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("role", "coordinator", "not frozen e33"),
        ("upstream_model", "/wrong/model", "exact source_inspector/S5"),
        ("expert_id", None, "exact source_inspector/S5"),
        ("route_evidence", None, "exact source_inspector/S5"),
    ),
)
def test_runtime_validator_rejects_misrouted_effective_child_calls(
    field: str, value: object, message: str
) -> None:
    validator = _validator_module()
    traces, audits = _effective_route_fixture(validator)
    audits[1][field] = value

    with pytest.raises(validator.AuditFailure, match=message):
        validator._validate_effective_call_routes(traces, audits)


def test_runtime_validator_rejects_non_disclosed_or_unattached_child_calls() -> None:
    validator = _validator_module()
    traces, audits = _effective_route_fixture(validator)
    traces[0]["nodes"][4]["message"]["content"] = (
        "[task from parent]\n\nexpert_id=source_inspector"
    )
    with pytest.raises(validator.AuditFailure, match="not the exact source_inspector child"):
        validator._validate_effective_call_routes(traces, audits)

    traces, audits = _effective_route_fixture(validator)
    traces[0]["calls"][1]["node"] = None
    with pytest.raises(validator.AuditFailure, match="is unattached"):
        validator._validate_effective_call_routes(traces, audits)


def test_p0_forensic_recovery_inventories_terminal_artifacts_without_weakening_strict() -> None:
    validator = _validator_module()
    traces, audits = _p0_terminal_route_fixture(validator)

    report = validator._validate_p0_terminal_route_artifacts(traces, audits)

    assert report["scope"] == "retained_P0_postflight_only_not_prospective_admission"
    assert report["attached_successful_calls"] == {"coordinator": 55, "child": 73}
    assert report["attached_successful_call_total"] == 128
    assert report["terminal_trace_call_categories"] == {
        "delivery_reset_after_agent_completed": 1,
        "successful_call_unattached_at_max_turns": 3,
    }
    assert len(report["audit_only_successful_events"]) == 1
    assert report["audit_only_successful_events"][0]["trace_index"] == 8
    assert report["route_events"] == 133
    assert report["route_events_by_role"] == {"coordinator": 57, "child": 76}
    assert report["forced_route_events_by_role"] == {"coordinator": 16}
    assert report["natural_route_events_by_role"] == {
        "coordinator": 41,
        "child": 76,
    }
    with pytest.raises(validator.AuditFailure, match="is unattached"):
        validator._validate_effective_call_routes(traces, audits)


def test_p0_forensic_recovery_rejects_nonterminal_null_call_or_route_mutation() -> None:
    validator = _validator_module()
    traces, audits = _p0_terminal_route_fixture(validator)
    traces[1]["calls"][-1]["error"] = {"type": "UnexpectedFailure"}
    with pytest.raises(validator.AuditFailure, match="exact terminal/cancelled"):
        validator._validate_p0_terminal_route_artifacts(traces, audits)

    traces, audits = _p0_terminal_route_fixture(validator)
    audits[-1]["upstream_model"] = "/wrong/model"
    with pytest.raises(validator.AuditFailure, match="exact source_inspector/S5"):
        validator._validate_p0_terminal_route_artifacts(traces, audits)


def test_p0_recovery_receipt_is_hash_locked_posthoc_and_nonadmitting() -> None:
    source = P0_VALIDATOR_PATH.read_text()

    assert "P0_EXECUTION_REVISION = \"35f9cd4667132902a98ed41b494ca44253635022\"" in source
    assert "P0_RETAINED_ARTIFACT_HASHES" in source
    assert "P0_RETAINED_ROUTING_AUDIT_SHA256" in source
    assert '"verdict": "posthoc_terminal_artifacts_reconciled"' in source
    assert '"launcher_postflight_passed": False' in source
    assert '"p0_mechanism_admitted": False' in source
    assert '"calibration_measurements_recovered": True' in source
    assert '"prospective_exact_multiplicity_pass": False' in source
    assert '"next_step_authorized": False' in source


def test_p0_recovery_receipt_uses_distinct_audit_revision_and_forensic_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _p0_validator_module()
    run_dir = Path("/immutable/p0/run")
    route_path = Path("/immutable/p0/routing-audit.jsonl")

    def fake_digest(path: Path) -> str:
        if path == validator.P0_CONFIG:
            return validator.P0_CONFIG_SHA256
        if path == route_path:
            return validator.P0_RETAINED_ROUTING_AUDIT_SHA256
        return validator.P0_RETAINED_ARTIFACT_HASHES[str(path.relative_to(run_dir))]

    captured: dict[str, object] = {}

    def fake_validate_runtime(*args, **kwargs):
        captured.update(kwargs)
        return {
            "effective_call_routes": {
                "scope": "retained_P0_postflight_only_not_prospective_admission"
            },
            "prospective_lr0_health": {"metrics": {}},
            "child_branches": 1,
            "child_trainable_tokens": 2,
            "exported_rl_tokens": 2,
            "raw_coordinator_sampled_tokens": 0,
            "coordinator_exported_trainable_tokens": 0,
            "gradient_norm": 1.0,
            "routing": {},
            "trace_sets": {"raw": 16, "effective": 16, "identical": True},
        }

    monkeypatch.setattr(validator, "_digest", fake_digest)
    monkeypatch.setattr(validator.BASE, "validate_runtime", fake_validate_runtime)
    monkeypatch.setattr(validator.BASE, "_read_jsonl", lambda path: [])
    monkeypatch.setattr(validator, "_reward_observations", lambda records: {})
    report = validator.recover_retained_runtime(
        run_dir,
        route_path,
        execution_revision=validator.P0_EXECUTION_REVISION,
        audit_revision="a" * 40,
        verifiers_revision="b" * 40,
        config_sha256=validator.P0_CONFIG_SHA256,
    )

    assert captured["calibration_only"] is True
    assert captured["p0_terminal_artifact_recovery"] is True
    assert captured["routing_audit_read_path"] == route_path
    assert report["execution_revision"] == validator.P0_EXECUTION_REVISION
    assert report["audit_revision"] == "a" * 40
    assert report["interpretation"]["launcher_postflight_passed"] is False
    assert report["interpretation"]["p0_mechanism_admitted"] is False
    assert report["interpretation"]["calibration_measurements_recovered"] is True


def test_p1_config_launcher_and_task_bank_are_fresh_and_nonupdating() -> None:
    config = P1_CONFIG_PATH.read_text()
    launcher = P1_LAUNCHER_PATH.read_text()
    hasher = P1_HASHER_PATH.read_text()

    assert config.count("seed = 20270917") == 2
    assert config.count("instance_offset = 71000") == 2
    assert "q35-2b-source-route-parity-p1-v1" in config
    assert "[trainer.ckpt]" not in config and "[orchestrator.ckpt]" not in config
    assert "run_q35_2b_specialist_competence_eval" not in launcher
    assert 'test "$(sha256sum "$sampling_contract"' in launcher
    assert "rehash_protected_models" in launcher
    assert "--write-preflight-model-hashes" in launcher
    assert "--preflight-model-hashes" in launcher
    assert launcher.index('if [[ "$dry_run" == true ]]') < launcher.index(
        "--write-preflight-model-hashes"
    )
    assert "postflight_recomputed" in P1_VALIDATOR_PATH.read_text()
    assert "export_derived_family_likelihood" in P1_VALIDATOR_PATH.read_text()
    assert "abs(family_mismatch_mean) / family_entropy_mean > 0.01" in P1_VALIDATOR_PATH.read_text()
    assert "max(all_active_abs_mismatch) > 0.35" in P1_VALIDATOR_PATH.read_text()
    assert "preflight_model_hashes != MODEL_PREFLIGHT" in P1_VALIDATOR_PATH.read_text()
    assert "len(source_tasks) != 32" in hasher
    assert "len(rows) != 64" in hasher
    assert "set(p1_keys) & set(p0_keys)" in hasher


def test_p1_task_hasher_requires_32_unique_tasks_per_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hasher = _p1_hasher_module()

    class Data:
        def __init__(self, key: str):
            self.key = key

        def model_dump(self, **kwargs):
            return {"key": self.key}

    class Task:
        def __init__(self, key: str):
            self.key = key
            self.data = Data(key)

    class Config:
        def __init__(self, **values):
            self.values = values

    class Taskset:
        def __init__(self, config):
            self.config = config

        def load(self):
            count = self.config.values["count"]
            prefix = self.config.values["prefix"]
            return [Task(f"{prefix}-{index}") for index in range(count)]

    monkeypatch.setattr(hasher, "_taskset_types", lambda: (Config, Taskset))
    with pytest.raises(ValueError, match="32 unique tasks"):
        hasher._bank([{"prefix": "ast", "count": 31}, {"prefix": "config", "count": 32}])
    keys, _, _ = hasher._bank(
        [{"prefix": "ast", "count": 32}, {"prefix": "config", "count": 32}]
    )
    assert len(keys) == 64


def test_p1_timed_route_join_is_bijective_and_rejects_audit_surplus() -> None:
    validator = _validator_module()
    traces, audits = _effective_route_fixture(validator)
    sequence = 0
    for trace_index, trace in enumerate(traces):
        session = audits[trace_index * 2]["session_sha256"]
        trace["id"] = f"p1-{trace_index}"
        trace["nodes"][0]["message"]["content"] = (
            f"/tmp/vf-prime-agent-runs/{session[:16]}/conversation.json"
        )
        trace["calls"][0].update({"time": {"start": 10.0, "end": 11.0}, "error": None})
        trace["calls"][1].update({"time": {"start": 12.0, "end": 13.0}, "error": None})
        trace["stop_condition"] = "agent_completed"
        for event_index, event in enumerate(audits[trace_index * 2 : trace_index * 2 + 2]):
            start = 10.2 if event_index == 0 else 12.2
            event.update(
                {
                    "sequence": sequence,
                    "request_sha256": hashlib.sha256(f"p1-{sequence}".encode()).hexdigest(),
                    "request_start_unix_s": start,
                    "request_end_unix_s": start + 0.5,
                }
            )
            sequence += 1

    report = validator._validate_p1_timed_call_routes(traces, audits)
    assert report["timing_session_bijection"] is True
    assert report["audit_only_events"] == 0
    assert report["terminal_residues"] == []

    surplus = {**audits[-1], "sequence": sequence, "request_sha256": "f" * 64}
    with pytest.raises(validator.AuditFailure, match="timed route join|audit-only"):
        validator._validate_p1_timed_call_routes(traces, [*audits, surplus])


def test_proxy_audit_records_joinable_wall_clock_interval() -> None:
    source = PROXY_PATH.read_text()
    assert '"request_start_unix_s": request_start_unix_s' in source
    assert '"request_end_unix_s": request_end_unix_s' in source
    assert "request_end_unix_s - latency_ms / 1000.0" in source


def _p2_trace(
    *, trace_id: str, family: str, source: str, task_key: str, group_id: str, reward: float
) -> dict:
    return {
        "id": trace_id,
        "ok": True,
        "errors": [],
        "task": {"key": task_key, "data": {"family": family}},
        "info": {"env_name": source, "group_id": group_id},
        "rewards": {"source_worker_first_call": {"score": reward}},
        "nodes": [{"sampled": True, "advantages": None}],
    }


def test_p2_partition_accepts_only_complete_zero_advantage_groups() -> None:
    validator = _validator_module()
    config = [
        _p2_trace(
            trace_id=f"config-{index}",
            family="specialist_source_config",
            source="source-worker-config-s6",
            task_key="config-key",
            group_id="config-group",
            reward=-0.79 + index / 10,
        )
        for index in range(8)
    ]
    ast = [
        _p2_trace(
            trace_id=f"ast-{index}",
            family="specialist_source_ast",
            source="source-worker-ast-s6",
            task_key="ast-key",
            group_id="ast-group",
            reward=-0.79 + index / 10,
        )
        for index in range(8)
    ]
    rejected = [
        _p2_trace(
            trace_id=f"rejected-{index}",
            family="specialist_source_ast",
            source="source-worker-ast-s6",
            task_key="rejected-key",
            group_id="rejected-group",
            reward=-0.79,
        )
        for index in range(8)
    ]
    metrics = [
        {
            "pre_filters/all/dropped_rate": 1 / 3,
            "pre_filters/all/zero_advantage/rate": 1 / 3,
        }
    ]

    report = validator._validate_p2_raw_effective_partition(
        [*config, *ast, *rejected], [*config, *ast], metrics
    )

    assert report["raw"] == 24
    assert report["effective"] == 16
    assert report["attempted_groups_by_family"] == {
        "specialist_source_config": 1,
        "specialist_source_ast": 2,
    }
    assert len(report["rejected_zero_advantage_groups"]) == 1
    rejected[0]["rewards"]["source_worker_first_call"]["score"] = -0.78
    with pytest.raises(validator.AuditFailure, match="nondegenerate"):
        validator._validate_p2_raw_effective_partition(
            [*config, *ast, *rejected], [*config, *ast], metrics
        )


def test_p2_forced_routes_cover_all_raw_resampled_groups() -> None:
    validator = _validator_module()
    traces = []
    audits = []
    group_specs = (
        ("specialist_source_ast", "source-worker-ast-s6", "ast-a"),
        ("specialist_source_ast", "source-worker-ast-s6", "ast-b"),
        ("specialist_source_config", "source-worker-config-s6", "config-a"),
    )
    for family, source, group_id in group_specs:
        code = (
            "task_worker = await rlm("
            "'[selected terminal capability]\\n"
            f"expert_id=source_inspector\\ngroup={group_id}', "
            'name="task-worker")'
        )
        action_sha = hashlib.sha256(code.encode()).hexdigest()
        for rollout in range(8):
            session_id = f"rollout-{group_id}-{rollout}"
            trace = _p2_trace(
                trace_id=session_id,
                family=family,
                source=source,
                task_key=f"task-{group_id}",
                group_id=group_id,
                reward=-0.79,
            )
            trace["nodes"] = [
                {
                    "sampled": True,
                    "advantages": None,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "name": "ipython",
                                "arguments": json.dumps({"code": code}),
                            }
                        ],
                    },
                }
            ]
            trace["calls"] = [{"node": 0, "client_session_id": session_id}]
            traces.append(trace)
            audits.append(
                {
                    "schema_version": validator.ROUTE_SCHEMA,
                    "mode": validator.FORCED_ROUTE_MODE,
                    "endpoint": "/inference/v1/generate",
                    "role": "coordinator",
                    "expert_id": "source_inspector",
                    "status": 200,
                    "session_sha256": hashlib.sha256(session_id.encode()).hexdigest(),
                    "action_sha256": action_sha,
                }
            )

    report = validator._validate_p2_forced_assignment_routes(traces, audits)
    assert report["scope"] == "all_raw_rollouts"
    assert report["events"] == 24
    assert report["matched_raw_events"] == 24
    assert report["groups"] == 3
    assert sorted(report["raw_action_sha256_counts"].values()) == [8, 8, 8]

    with pytest.raises(validator.AuditFailure, match="exact action multiplicity"):
        validator._validate_p2_forced_assignment_routes(traces, audits[:-1])
    with pytest.raises(validator.AuditFailure, match="more than once in a rollout"):
        validator._validate_p2_forced_assignment_routes(traces, [*audits, audits[0]])


def _p2_route_trace(validator, *, third_root: bool = False) -> tuple[dict, list[dict]]:
    rollout_session = "a" * 64
    root_session = "root-branch-session"
    child_session = "child-branch-session"
    nodes = [
        {
            "parent": None,
            "message": {
                "role": "user",
                "content": (
                    "Recursive agent depth: 0\n[specialist worker routing contract]\n"
                    f"/tmp/vf-prime-agent-runs/{rollout_session[:16]}/root"
                ),
            },
        },
        {"parent": 0, "message": {"role": "assistant", "content": "root"}},
        {
            "parent": None,
            "message": {
                "role": "user",
                "content": (
                    "Recursive agent depth: 1\n"
                    f"/tmp/vf-prime-agent-runs/{rollout_session[:16]}/child"
                ),
            },
        },
        {
            "parent": 2,
            "message": {
                "role": "user",
                "content": validator.SPECIALIST_CHILD_PREFIX + "\ncan_delegate=false",
            },
        },
        {"parent": 3, "message": {"role": "assistant", "content": "child"}},
    ]
    calls = [
        {
            "node": 1,
            "client_session_id": root_session,
            "time": {"start": 10.0, "end": 11.0},
            "error": None,
        },
        {
            "node": 4,
            "client_session_id": child_session,
            "time": {"start": 12.0, "end": 13.0},
            "error": None,
        },
    ]
    if third_root:
        nodes.extend(
            [
                {
                    "parent": None,
                    "message": {
                        "role": "user",
                        "content": "Recursive agent depth: 2\nYou are a child agent spawned by task-worker",
                    },
                },
                {"parent": 5, "message": {"role": "assistant", "content": "nested"}},
            ]
        )
        calls.append(
            {
                "node": 6,
                "client_session_id": "nested-branch-session",
                "time": {"start": 14.0, "end": 15.0},
                "error": None,
            }
        )
    trace = {
        "id": "trace-1",
        "nodes": nodes,
        "calls": calls,
        "stop_condition": "agent_completed",
    }
    audits = [
        {
            "schema_version": validator.ROUTE_SCHEMA,
            "sequence": 0,
            "role": "coordinator",
            "endpoint": "/inference/v1/generate",
            "request_sha256": "1" * 64,
            "session_sha256": rollout_session,
            "branch_session_sha256": hashlib.sha256(root_session.encode()).hexdigest(),
            "upstream_model": str(validator.E33_PATH),
            "status": 200,
            "mode": validator.FORCED_ROUTE_MODE,
            "route_evidence": "coordinator_without_specialist_child_prefix",
            "request_start_unix_s": 10.1,
            "request_end_unix_s": 10.9,
        },
        {
            "schema_version": validator.ROUTE_SCHEMA,
            "sequence": 1,
            "role": "child",
            "endpoint": "/inference/v1/generate",
            "request_sha256": "2" * 64,
            "session_sha256": rollout_session,
            "branch_session_sha256": hashlib.sha256(child_session.encode()).hexdigest(),
            "upstream_model": str(validator.S5_PATH),
            "expert_id": "source_inspector",
            "status": 200,
            "mode": "forwarded",
            "route_evidence": "exact_specialist_child_prefix",
            "request_start_unix_s": 12.1,
            "request_end_unix_s": 12.9,
        },
    ]
    return trace, audits


def test_p2_branch_join_uses_distinct_rollout_and_client_sessions() -> None:
    validator = _validator_module()
    trace, audits = _p2_route_trace(validator)

    report = validator._validate_p2_branch_call_routes([trace], [trace], audits)

    assert report["branch_rollout_timing_bijection"] is True
    assert report["terminal_worker_descendants"] == 0
    assert audits[0]["session_sha256"] != audits[0]["branch_session_sha256"]
    audits[1]["role"] = "coordinator"
    with pytest.raises(validator.AuditFailure, match="route|branch role"):
        validator._validate_p2_branch_call_routes([trace], [trace], audits)


def test_p2_rejects_p1_style_named_grandchild_third_root() -> None:
    validator = _validator_module()
    trace, _ = _p2_route_trace(validator, third_root=True)

    with pytest.raises(validator.AuditFailure, match="only coordinator and terminal"):
        validator._validate_no_terminal_worker_descendant(trace, 0)


def test_p2_wrapper_executes_group_atomic_base_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _p2_validator_module()
    models = {"model": {"path": "/model", "model_sha256": "f" * 64}}
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        validator,
        "digest",
        lambda path: "c" * 64
        if path == validator.CONFIG
        else "e" * 64
        if path == validator.SAMPLING_CONTRACT
        else "a" * 64,
    )
    monkeypatch.setattr(validator, "_actual_model_hashes", lambda: models)
    monkeypatch.setattr(validator, "_run_artifact_hashes", lambda run_dir: {})
    monkeypatch.setattr(
        validator.BASE,
        "_read_json",
        lambda path: {
            "schema_version": (
                "q35-2b-source-route-parity-p2-model-preflight/v1"
                if path == validator.MODEL_PREFLIGHT
                else "q35-2b-source-route-parity-p2-model-postflight/v1"
            ),
            "execution_revision": "a" * 40,
            "verifiers_revision": "f" * 40,
            "config_sha256": "c" * 64,
            "models": models,
            "run_artifacts": {},
            "checkpoint_written": False,
            "learning_rate": 0.0,
        },
    )

    def fake_base_runtime(*args, **kwargs):
        captured.update(kwargs)
        return {
            "exported_rl_tokens": 6000,
            "gradient_norm": 1.0,
            "effective_call_routes": {},
            "trace_sets": {},
        }

    monkeypatch.setattr(validator.BASE, "validate_runtime", fake_base_runtime)
    monkeypatch.setattr(validator.BASE, "_read_jsonl", lambda path: [])
    monkeypatch.setattr(validator, "_validate_resolved_p2_mechanism", lambda path: {})
    monkeypatch.setattr(validator.BASE, "_observe_lr0_health", lambda metrics, traces: {"metrics": {}})
    monkeypatch.setattr(validator, "_validate_p2_health", lambda *args: {})
    monkeypatch.setattr(
        validator,
        "_materialize_task_bank",
        lambda: {
            "task_bank_sha256": "b" * 64,
            "task_key_set_sha256": "d" * 64,
            "tasks": 64,
            "p0_tasks": 64,
            "p1_tasks": 64,
            "pairwise_disjoint": True,
            "overlaps": {"P2_P0": [], "P2_P1": [], "P0_P1": []},
            "p0_task_bank_sha256": "0" * 64,
            "p0_task_key_set_sha256": "1" * 64,
            "p1_task_bank_sha256": "2" * 64,
            "p1_task_key_set_sha256": "3" * 64,
        },
    )

    validator.validate_runtime(
        validator.RUN_DIR,
        config_sha256="c" * 64,
        task_bank_sha256="b" * 64,
        task_key_set_sha256="d" * 64,
        sampling_contract_sha256="e" * 64,
        execution_revision="a" * 40,
        verifiers_revision="f" * 40,
        preflight_model_hashes=validator.MODEL_PREFLIGHT,
        postflight_model_hashes=validator.MODEL_POSTFLIGHT,
    )

    assert captured["p2_group_atomic_route_admission"] is True
    assert "p2_timed_route_admission" not in captured


def test_p2_launch_assets_are_calibration_only_and_hash_locked() -> None:
    config = P2_CONFIG_PATH.read_text()
    launcher = P2_LAUNCHER_PATH.read_text()
    validator = P2_VALIDATOR_PATH.read_text()
    hasher = P2_HASHER_PATH.read_text()

    assert config.count("seed = 20270925") == 2
    assert config.count("instance_offset = 72000") == 2
    assert config.count('RLM_MAX_DEPTH = "1"') == 2
    assert "require_client_session_id = true" in config
    assert "[trainer.ckpt]" not in config and "[orchestrator.ckpt]" not in config
    assert "eval" not in launcher.lower()
    assert "P2_DRY_RUN" in launcher and "preflight_archive" in launcher
    assert 'mv "$output_root" "$preflight_archive/output-root"' in launcher
    assert launcher.index("--write-postflight-model-hashes") < launcher.index(
        '"$run_dir" --runtime'
    )
    assert "c1a2f5bf3db3f34206e45b04442e64ca6a7770de" in launcher
    assert "pairwise_disjoint" in hasher and "P2_P1" in hasher
    assert "rejected_trace_exports" in validator
    assert BAD_TRACE_ID_IN_FORENSIC in P1_FORENSIC_PATH.read_text()


def test_p2r_core_import_provenance_requires_worktree_module_and_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = _p2r_validator_module()
    train_path = tmp_path / "deps/verifiers/verifiers/v1/clients/train.py"
    client_path = tmp_path / "deps/verifiers/verifiers/v1/clients/client.py"
    train_path.parent.mkdir(parents=True)
    train_path.write_text("# reviewed train client\n")
    client_path.write_text("# reviewed client constants\n")

    train = types.ModuleType("verifiers.v1.clients.train")
    train.__file__ = str(train_path)
    train.forwarded_session_headers = lambda session_id, headers: {
        "X-Session-ID": session_id,
        "X-Client-Session-ID": headers["session_id"],
    }
    client = types.ModuleType("verifiers.v1.clients.client")
    client.__file__ = str(client_path)
    client.SESSION_ID_HEADER = "X-Session-ID"
    client.CLIENT_SESSION_ID_HEADER = "X-Client-Session-ID"
    clients = types.ModuleType("verifiers.v1.clients")
    clients.train = train
    clients.client = client
    v1 = types.ModuleType("verifiers.v1")
    v1.clients = clients
    verifiers = types.ModuleType("verifiers")
    verifiers.v1 = v1
    for name, module in {
        "verifiers": verifiers,
        "verifiers.v1": v1,
        "verifiers.v1.clients": clients,
        "verifiers.v1.clients.train": train,
        "verifiers.v1.clients.client": client,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    report = validator.verify_core_import_provenance(tmp_path)
    assert report["train_client_module"] == str(train_path)
    assert report["branch_header"] == "X-Client-Session-ID"
    assert report["exact_behavior_probe_passed"] is True

    train.__file__ = str(tmp_path / "shared-environment/train.py")
    with pytest.raises(validator.BASE.AuditFailure, match="import provenance"):
        validator.verify_core_import_provenance(tmp_path)


def test_p2r_launcher_prioritizes_reviewed_core_verifiers() -> None:
    launcher = P2R_LAUNCHER_PATH.read_text()
    config = P2R_CONFIG_PATH.read_text()

    assert 'p2r_pythonpath="$root/deps/verifiers:$root/src:' in launcher
    assert "--verify-core-import-provenance" in launcher
    assert "--write-core-import-provenance" in launcher
    assert '--core-import-provenance "$core_import_provenance"' in launcher
    assert "--validate-failed-start-evidence" in launcher
    assert '--failed-start-evidence-sha256 "$failed_start_sha"' in launcher
    assert "q35-2b-source-route-parity-p2r-v1" in config
    assert "q35-2b-source-route-parity-p2-v1" not in config


def test_p2r_failed_start_proof_matches_durable_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = _p2r_validator_module()
    snapshot = REPO.parent / "durable-snapshots/2026-09-05-p2-failed-start"
    if not snapshot.is_dir():
        pytest.skip("durable P2 failed-start snapshot is not present")
    run_dir = (
        snapshot
        / "outputs/lr0-calibration/source-route-parity-p2-lr0-admission"
    )
    result_root = snapshot / "results"
    evidence = json.loads(P2_FAILED_START_PATH.read_text())
    evidence["failed_run_dir"] = str(run_dir)
    evidence["failed_result_root"] = str(result_root)
    evidence_path = tmp_path / "failed-start.json"
    evidence_path.write_text(json.dumps(evidence, sort_keys=True) + "\n")
    monkeypatch.setattr(validator, "FAILED_RUN_DIR", run_dir)
    monkeypatch.setattr(validator, "FAILED_RESULT_ROOT", result_root)
    monkeypatch.setattr(validator, "FAILED_START_EVIDENCE", evidence_path)

    report = validator.validate_failed_start_evidence(
        hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )

    assert report["raw_traces"] == 8
    assert report["failed_calls"] == 32
    assert report["upstream_model_responses"] == 0
    assert report["same_seed_and_bank_reuse_justified"] is True


BAD_TRACE_ID_IN_FORENSIC = "57f6214886ce4f70a69f3eb2754770ce"
