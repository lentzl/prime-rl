import importlib.util
import json
import sys
from pathlib import Path


def _module():
    path = Path(__file__).parents[2] / "scripts/summarize_q35_2b_specialist_population_v1.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _competence_module():
    scripts = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "summarize_q35_2b_specialist_competence_v1.py"
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _trace(index: int, family: str, expert: str | None, success: bool) -> dict:
    root = f"/workspace/specialist-worker/v4-i{35100 + index}"
    metrics = {
        "answer_accuracy": int(success),
        "protocol_aligned": int(success),
        "clean_protocol_aligned": int(success),
        "coordination_spawn_calls": int(expert is not None),
        "failed_cells": int(not success),
    }
    nodes = []
    if expert is not None:
        code = (
            f"assignment = '''[selected terminal capability]\nexpert_id={expert}\n'''\n"
            f"text = Path('{root}/data.json').read_text()\n"
            "await agent_message.send(json.dumps({'value': len(text)}), receiver_role='parent')"
        )
        nodes.append(
            {
                "sampled": True,
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"name": "ipython", "arguments": json.dumps({"code": code})}],
                },
            }
        )
    return {
        "id": f"trace-{index}",
        "task": {
            "data": {
                "name": f"{family}-v4-i{35100 + index}",
                "family": family,
                "template_variant": 4,
                "answer": {"result": index},
                "files": {f"{root}/data.json": "[]"} if expert else {},
                "child_paths": {"task-worker": root} if expert else {},
                "preferred_expert": expert,
            }
        },
        "metrics": metrics,
        "nodes": nodes,
        "stop_condition": "agent_completed",
    }


def test_selected_expert_accepts_literal_newlines_from_rlm_assignment() -> None:
    module = _module()
    assert module.SELECTED_EXPERT.findall(
        'await rlm("[selected terminal capability]\\nexpert_id=table_analyst\\nobjective=x")'
    ) == ["table_analyst"]


def _write_traces(path: Path, traces: list[dict]) -> None:
    path.write_text(json.dumps({"traces": traces}) + "\n", encoding="utf-8")


def _write_audit(
    path: Path,
    traces: list[dict],
    models: dict[str, str],
    *,
    worker_model_override: str | None = None,
) -> None:
    sequence = 0
    with path.open("w", encoding="utf-8") as stream:
        for trace in traces:
            expert = trace["task"]["data"].get("preferred_expert")
            if expert is None:
                continue
            stream.write(
                json.dumps(
                    {
                        "sequence": sequence,
                        "mode": f"forwarded_specialist_cognitive_action_delegate_terminal_{expert}",
                        "expert_id": expert,
                        "upstream_model": "COORDINATOR",
                        "latency_ms": 10.0,
                        "session_sha256": f"session-{sequence}",
                    }
                )
                + "\n"
            )
            stream.write(
                json.dumps(
                    {
                        "sequence": sequence + 1,
                        "mode": "forwarded",
                        "expert_id": expert,
                        "upstream_model": worker_model_override or models[expert],
                        "latency_ms": 10.0,
                        "session_sha256": f"session-{sequence}",
                    }
                )
                + "\n"
            )
            sequence += 2


def test_paired_specialist_gate_requires_real_multi_niche_recoveries(tmp_path: Path) -> None:
    module = _module()
    families = [
        ("specialist_local", None),
        ("specialist_local", None),
        ("specialist_generic", "generic_worker"),
        ("specialist_generic", "generic_worker"),
        ("specialist_table_join", "table_analyst"),
        ("specialist_table_reconcile", "table_analyst"),
        ("specialist_source_ast", "source_inspector"),
        ("specialist_source_config", "source_inspector"),
    ]
    control = [
        _trace(index, family, "generic_worker" if expert else None, expert is None)
        for index, (family, expert) in enumerate(families)
    ]
    treatment = [_trace(index, family, expert, True) for index, (family, expert) in enumerate(families)]
    control_traces = tmp_path / "control.jsonl"
    treatment_traces = tmp_path / "treatment.jsonl"
    control_audit = tmp_path / "control-audit.jsonl"
    treatment_audit = tmp_path / "treatment-audit.jsonl"
    _write_traces(control_traces, control)
    _write_traces(treatment_traces, treatment)
    _write_audit(control_audit, control, {"generic_worker": "H176"})
    _write_audit(
        treatment_audit,
        treatment,
        {
            "generic_worker": "H176",
            "table_analyst": "TABLE",
            "source_inspector": "SOURCE",
        },
    )

    result = module.summarize(
        control_traces=control_traces,
        control_audit=control_audit,
        treatment_traces=treatment_traces,
        treatment_audit=treatment_audit,
        control_models={"generic_worker": "H176"},
        treatment_models={
            "generic_worker": "H176",
            "table_analyst": "TABLE",
            "source_inspector": "SOURCE",
        },
        expected_tasks=8,
        recovery_floor=4,
    )

    assert result["paired_recovery_count"] == 6
    assert result["recoveries_attributed_to_specialists"] == {
        "source_inspector": 2,
        "table_analyst": 2,
    }
    assert result["maximum_worker_activation_fraction"] == 1 / 3
    assert result["paired_population_gate_passed"] is True
    assert result["acceptance_gates_relaxed"] is False


def test_route_model_mismatch_fails_provenance_gate(tmp_path: Path) -> None:
    module = _module()
    control = [_trace(0, "specialist_local", None, True)]
    treatment = [_trace(0, "specialist_local", None, True)]
    control_traces = tmp_path / "control.jsonl"
    treatment_traces = tmp_path / "treatment.jsonl"
    control_audit = tmp_path / "control-audit.jsonl"
    treatment_audit = tmp_path / "treatment-audit.jsonl"
    _write_traces(control_traces, control)
    _write_traces(treatment_traces, treatment)
    control_audit.write_text("", encoding="utf-8")
    _write_audit(
        treatment_audit,
        [_trace(0, "specialist_table_join", "table_analyst", False)],
        {"table_analyst": "TABLE"},
        worker_model_override="WRONG",
    )

    result = module.summarize(
        control_traces=control_traces,
        control_audit=control_audit,
        treatment_traces=treatment_traces,
        treatment_audit=treatment_audit,
        control_models={},
        treatment_models={"table_analyst": "TABLE"},
        expected_tasks=1,
        recovery_floor=1,
    )

    assert result["treatment"]["exact_route_sequence"] is False
    assert result["treatment"]["route_models_exact"] is False
    assert result["acceptance"]["exact_route_provenance"] is False
    assert result["paired_population_gate_passed"] is False


def test_fixed_route_competence_summary_is_router_independent(tmp_path: Path) -> None:
    module = _competence_module()
    families = [
        "specialist_table_join",
        "specialist_table_reconcile",
        "specialist_table_join",
        "specialist_table_reconcile",
    ]
    control = [
        _trace(index, family, "table_analyst", False)
        for index, family in enumerate(families)
    ]
    treatment = [
        _trace(index, family, "table_analyst", True)
        for index, family in enumerate(families)
    ]
    control_traces = tmp_path / "control.jsonl"
    treatment_traces = tmp_path / "treatment.jsonl"
    control_audit = tmp_path / "control-audit.jsonl"
    treatment_audit = tmp_path / "treatment-audit.jsonl"
    _write_traces(control_traces, control)
    _write_traces(treatment_traces, treatment)
    _write_audit(control_audit, control, {"table_analyst": "H176"})
    _write_audit(treatment_audit, treatment, {"table_analyst": "H_TABLE"})

    result = module.summarize(
        expert_id="table_analyst",
        control_traces=control_traces,
        control_audit=control_audit,
        treatment_traces=treatment_traces,
        treatment_audit=treatment_audit,
        control_model="H176",
        treatment_model="H_TABLE",
        expected_tasks=4,
        minimum_worker_activations=4,
        minimum_treatment_hard_successes=4,
        minimum_hard_successes_per_family=2,
        minimum_paired_recoveries=4,
        maximum_paired_regressions=0,
    )

    assert result["paired_recovery_count"] == 4
    assert result["paired_regression_count"] == 0
    assert result["router_taxonomy_evaluated"] is False
    assert result["acceptance"]["fixed_route_exact"] is True
    assert result["acceptance"]["router_absent"] is True
    assert result["competence_gate_passed"] is True
