from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
REWARD_PATH = (
    REPO
    / "patches/verifiers/environments/source_worker_first_call_v1/"
    "source_worker_first_call_v1/reward.py"
)
SPEC = importlib.util.spec_from_file_location("source_worker_first_call_reward", REWARD_PATH)
assert SPEC is not None and SPEC.loader is not None
REWARD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REWARD
SPEC.loader.exec_module(REWARD)
CellEvidence = REWARD.CellEvidence
score_first_call = REWARD.score_first_call


AST_PATHS = ("/workspace/sample/alpha.py", "/workspace/sample/beta.py")
CONFIG_PATHS = ("/workspace/sample/service.toml", "/workspace/sample/features.env")
LAUNCHER_PATH = REPO / "scripts/run_q35_2b_source_first_call_grpo_s6_v1.sh"
VALIDATOR_PATH = REPO / "scripts/validate_q35_2b_source_first_call_grpo_s6_v1.py"


def _validator_module():
    spec = importlib.util.spec_from_file_location(
        "source_first_call_validator", VALIDATOR_PATH
    )
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
    assert result.exception_free_first_call == 1
    assert result.correct_file_api == 0
    assert result.score == pytest.approx(0.1)


def test_two_hop_loop_proves_each_path_is_read() -> None:
    code = """import ast
from pathlib import Path
paths = ('/workspace/sample/alpha.py', '/workspace/sample/beta.py')
trees = []
for path in paths:
    trees.append(ast.parse(Path(path).read_text()))
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
    assert recovered.score == pytest.approx(0.09)
    assert recovered.score < baseline.score


def test_extra_send_prevents_atomic_credit_and_is_penalized() -> None:
    first = """import ast
from pathlib import Path
trees = [ast.parse(Path(path).read_text()) for path in ('/workspace/sample/alpha.py', '/workspace/sample/beta.py')]
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


def test_runtime_validator_rejects_non_source_reward_leakage() -> None:
    validator = VALIDATOR_PATH.read_text()
    assert "if name != EXPECTED_REWARD" in validator
    assert 'payload.get("score") not in (0, 0.0)' in validator
    assert "has non-S6 reward leakage" in validator


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
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "name": "ipython",
                                        "arguments": json.dumps({"code": code}),
                                    }
                                ],
                            }
                        }
                    ],
                    "calls": [
                        {"node": 0, "client_session_id": client_session_id}
                    ],
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
                    "session_sha256": hashlib.sha256(
                        client_session_id.encode()
                    ).hexdigest(),
                    "action_sha256": action_sha,
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
        match="lacks its exact forced-assignment route event",
    ):
        validator._validate_forced_assignment_routes(traces, audits[:-1])


def test_runtime_validator_rejects_child_or_unmatched_forced_route() -> None:
    validator = _validator_module()
    code = (
        "task_worker = await rlm("
        "'[selected terminal capability]\\nexpert_id=source_inspector', "
        'name="task-worker")'
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
            "session_sha256": hashlib.sha256(
                f"session-{index}".encode()
            ).hexdigest(),
            "action_sha256": action_sha,
        }
        for index in range(16)
    ]
    with pytest.raises(validator.AuditFailure, match="invalid forced-assignment"):
        validator._validate_forced_assignment_routes(traces, audits)
