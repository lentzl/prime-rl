import importlib.util
import json
import sys
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[2]
        / "scripts/summarize_q35_2b_specialist_routing_screen_v1.py"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _code_call(code: str) -> dict:
    return {
        "sampled": True,
        "message": {
            "role": "assistant",
            "tool_calls": [
                {"name": "ipython", "arguments": json.dumps({"code": code})}
            ],
        },
    }


def _trace(index: int, family: str, *, wrong_expert: str | None = None) -> dict:
    expected = {
        "specialist_generic": "generic_worker",
        "specialist_table_join": "table_analyst",
        "specialist_table_reconcile": "table_analyst",
        "specialist_source_ast": "source_inspector",
        "specialist_source_config": "source_inspector",
        "specialist_recursive_table": "table_analyst",
        "specialist_recursive_source": "source_inspector",
    }.get(family)
    nodes = []
    if family == "specialist_local":
        nodes.append(
            _code_call("specialist_local_values = [1, 2]\nsum(specialist_local_values)")
        )
    else:
        if family.startswith("specialist_recursive_"):
            nodes.append(_code_call('specialist_manager = await rlm("manager")'))
        expert = wrong_expert or expected
        nodes.append(
            _code_call(
                'await rlm("[selected terminal capability]\\n'
                f'expert_id={expert}\\nobjective=x")'
            )
        )
    return {
        "id": f"trace-{index}",
        "task": {"data": {"name": f"{family}-{index}", "family": family}},
        "metrics": {"coordination_spawn_calls": int(family != "specialist_local")},
        "nodes": nodes,
    }


def _write_audit(
    path: Path,
    traces: list[dict],
    worker: str = "H176",
    router: str | None = None,
) -> None:
    sequence = 0
    with path.open("w", encoding="utf-8") as stream:
        for trace in traces:
            family = trace["task"]["data"]["family"]
            expected = {
                "specialist_generic": "generic_worker",
                "specialist_table_join": "table_analyst",
                "specialist_table_reconcile": "table_analyst",
                "specialist_source_ast": "source_inspector",
                "specialist_source_config": "source_inspector",
                "specialist_recursive_table": "table_analyst",
                "specialist_recursive_source": "source_inspector",
            }.get(family)
            if expected is None:
                continue
            session = f"session-{sequence}"
            if router is not None:
                stream.write(
                    json.dumps(
                        {
                            "sequence": sequence,
                            "mode": f"forwarded_specialist_expert_router_{expected}",
                            "expert_id": expected,
                            "upstream_model": router,
                            "response_sha256": f"response-{sequence}",
                            "status": 200,
                            "session_sha256": session,
                        }
                    )
                    + "\n"
                )
                sequence += 1
            for mode, model in (
                (
                    f"forwarded_specialist_cognitive_action_delegate_terminal_{expected}",
                    "COORDINATOR",
                ),
                ("forwarded", worker),
            ):
                stream.write(
                    json.dumps(
                        {
                            "sequence": sequence,
                            "mode": mode,
                            "expert_id": expected,
                            "upstream_model": model,
                            "session_sha256": session,
                        }
                    )
                    + "\n"
                )
                sequence += 1


def test_routing_screen_requires_all_public_route_classes(tmp_path: Path) -> None:
    module = _module()
    families = [
        "specialist_local",
        "specialist_generic",
        "specialist_table_join",
        "specialist_table_reconcile",
        "specialist_source_ast",
        "specialist_source_config",
        "specialist_recursive_table",
        "specialist_recursive_source",
    ] * 2
    traces = [_trace(index, family) for index, family in enumerate(families)]
    traces_path = tmp_path / "traces.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    traces_path.write_text(json.dumps({"traces": traces}) + "\n", encoding="utf-8")
    _write_audit(audit_path, traces)

    result = module.summarize(
        traces_path=traces_path,
        audit_path=audit_path,
        expected_models={
            "generic_worker": "H176",
            "table_analyst": "H176",
            "source_inspector": "H176",
        },
    )

    assert result["root_routes_correct"] == 16
    assert result["recursive_manager_routes_correct"] == 4
    assert result["exact_route_sequence"] is True
    assert result["activated_worker_models_exact"] is True
    assert result["routing_screen_gate_passed"] is True


def test_wrong_source_expert_fails_source_and_sequence_gates(tmp_path: Path) -> None:
    module = _module()
    traces = [_trace(0, "specialist_source_ast", wrong_expert="table_analyst")]
    traces_path = tmp_path / "traces.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    traces_path.write_text(json.dumps({"traces": traces}) + "\n", encoding="utf-8")
    _write_audit(audit_path, traces)

    result = module.summarize(
        traces_path=traces_path,
        audit_path=audit_path,
        expected_models={"source_inspector": "H176"},
        expected_tasks=1,
    )

    assert result["root_routes_correct"] == 0
    assert result["exact_route_sequence"] is False
    assert result["routing_screen_gate_passed"] is False


def test_live_split_gate_requires_four_complete_routes_and_router_provenance(
    tmp_path: Path,
) -> None:
    module = _module()
    families = [
        "specialist_local",
        "specialist_generic",
        "specialist_table_join",
        "specialist_table_reconcile",
        "specialist_source_ast",
        "specialist_source_config",
        "specialist_recursive_table",
        "specialist_recursive_source",
    ] * 2
    traces = [_trace(index, family) for index, family in enumerate(families)]
    for trace in traces[:4]:
        trace["metrics"].update(
            {
                "answer_accuracy": 1,
                "protocol_aligned": 1,
                "clean_protocol_aligned": 1,
            }
        )
        trace["stop_condition"] = "agent_completed"
        if trace["task"]["data"]["family"] != "specialist_local":
            trace["nodes"].append(
                _code_call(
                    "text = Path('/workspace/data.json').read_text()\n"
                    "await agent_message.send(text, receiver_role='parent')"
                )
            )
    traces_path = tmp_path / "traces.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    traces_path.write_text(json.dumps({"traces": traces}) + "\n", encoding="utf-8")
    _write_audit(audit_path, traces, router="C1L")

    result = module.summarize(
        traces_path=traces_path,
        audit_path=audit_path,
        expected_models={
            "generic_worker": "H176",
            "table_analyst": "H176",
            "source_inspector": "H176",
        },
        expected_router_model="C1L",
        minimum_complete_qualifying=4,
    )

    assert result["complete_qualifying_count"] == 4
    assert result["isolated_router_provenance_exact"] is True
    assert result["routing_screen_gate_passed"] is True
