import importlib.util
import json
import sys
from pathlib import Path


def _module():
    path = Path(__file__).parents[2] / "scripts/summarize_q35_2b_specialist_utility_screen_v1.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _code(code: str) -> dict:
    return {
        "sampled": True,
        "message": {
            "role": "assistant",
            "tool_calls": [{"name": "ipython", "arguments": json.dumps({"code": code})}],
        },
    }


def _trace(index: int, expert: str, *, table_cost: float = 1.0) -> dict:
    registry = [
        {
            "expert_id": "generic_worker",
            "role": "terminal_worker",
            "capability": "single JSON",
            "affordances": ["single_json_arithmetic"],
            "limitations": "single artifact",
            "relative_cost": 0.5,
        },
        {
            "expert_id": "table_analyst",
            "role": "terminal_worker",
            "capability": "tables and JSON",
            "affordances": ["single_json_arithmetic", "multi_artifact_table"],
            "limitations": "no source",
            "relative_cost": table_cost,
        },
        {
            "expert_id": "source_inspector",
            "role": "terminal_worker",
            "capability": "source",
            "affordances": ["source_config_inspection"],
            "limitations": "no tables",
            "relative_cost": 1.0,
        },
    ]
    assignment = {
        "worker_name": "task-worker",
        "objective": "Read one JSON list and compute its weighted sum.",
        "paths": ["/workspace/specialist-worker/data.json"],
    }
    prompt = "\n".join(
        [
            "[specialist worker routing contract]",
            "[capability registry]",
            *(json.dumps(row, separators=(",", ":")) for row in registry),
            "[terminal specialist assignment]",
            json.dumps(assignment, separators=(",", ":")),
        ]
    )
    return {
        "id": f"trace-{index}",
        "task": {
            "data": {
                "name": f"specialist_generic-{index}",
                "family": "specialist_generic",
                "prompt": prompt,
            }
        },
        "metrics": {
            "coordination_spawn_calls": 1,
            "answer_accuracy": 1,
            "protocol_aligned": 1,
            "clean_protocol_aligned": 1,
        },
        "stop_condition": "agent_completed",
        "nodes": [
            _code(
                'await rlm("[selected terminal capability]\\n'
                f'expert_id={expert}\\nobjective=x")'
            ),
            _code(
                "text = Path('/workspace/specialist-worker/data.json').read_text()\n"
                "await agent_message.send(text, receiver_role='parent')"
            ),
        ],
    }


def _write(tmp_path: Path, trace: dict, expert: str) -> tuple[Path, Path]:
    traces = tmp_path / "traces.jsonl"
    audit = tmp_path / "audit.jsonl"
    traces.write_text(json.dumps({"traces": [trace]}) + "\n", encoding="utf-8")
    rows = [
        {
            "sequence": 0,
            "mode": f"forwarded_specialist_expert_router_{expert}",
            "expert_id": expert,
            "upstream_model": "C1L",
            "response_sha256": "router-response",
            "status": 200,
            "session_sha256": "session",
        },
        {
            "sequence": 1,
            "mode": f"forwarded_specialist_cognitive_action_delegate_terminal_{expert}",
            "expert_id": expert,
            "upstream_model": "E33",
            "session_sha256": "session",
        },
        {
            "sequence": 2,
            "mode": "forwarded",
            "expert_id": expert,
            "upstream_model": "H176",
            "session_sha256": "session",
        },
    ]
    audit.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return traces, audit


def test_cheaper_public_generic_is_nondominated(tmp_path: Path) -> None:
    module = _module()
    trace = _trace(0, "generic_worker")
    traces, audit = _write(tmp_path, trace, "generic_worker")

    result = module.summarize(
        traces_path=traces,
        audit_path=audit,
        expected_models={"generic_worker": "H176"},
        expected_router_model="C1L",
        expected_tasks=1,
        minimum_complete_qualifying=1,
    )

    assert result["tasks"][0]["capability_valid"] is True
    assert result["tasks"][0]["utility_nondominated"] is True
    assert result["complete_qualifying_count"] == 1


def test_more_expensive_capable_route_is_not_nondominated(tmp_path: Path) -> None:
    module = _module()
    trace = _trace(0, "table_analyst")
    traces, audit = _write(tmp_path, trace, "table_analyst")

    result = module.summarize(
        traces_path=traces,
        audit_path=audit,
        expected_models={"table_analyst": "H176"},
        expected_router_model="C1L",
        expected_tasks=1,
        minimum_complete_qualifying=1,
    )

    assert result["tasks"][0]["capability_valid"] is True
    assert result["tasks"][0]["utility_nondominated"] is False
    assert result["complete_qualifying_count"] == 0


def test_equal_cost_capable_routes_are_both_accepted(tmp_path: Path) -> None:
    module = _module()
    trace = _trace(0, "table_analyst", table_cost=0.5)
    traces, audit = _write(tmp_path, trace, "table_analyst")

    result = module.summarize(
        traces_path=traces,
        audit_path=audit,
        expected_models={"table_analyst": "H176"},
        expected_router_model="C1L",
        expected_tasks=1,
        minimum_complete_qualifying=1,
    )

    assert result["tasks"][0]["valid_experts"] == ["generic_worker", "table_analyst"]
    assert result["tasks"][0]["utility_nondominated"] is True
    assert result["complete_qualifying_count"] == 1
