from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
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
EXPERIMENT = REPO / "experiments/qwen35-2b-document-recursion-zero-update-v1"


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
        def __init__(self, token_ids, session):
            self.headers = {"x-session-id": session}
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
    )
    client = Client()
    proxy.client = client

    forced = asyncio.run(proxy.generate(Request([1, 2, 3], "rollout-root")))
    child = asyncio.run(
        proxy.generate(Request([4, *marker, 5], "rollout-child"))
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
    assert events[1]["role"] == "child"
    assert events[1]["expert_id"] == "source_inspector"
    assert events[1]["upstream_model"] == "S5"
    assert events[1]["route_evidence"] == "exact_specialist_child_prefix"


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
