#!/usr/bin/env python3
"""Audit one fixed-route terminal specialist against frozen H176."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from summarize_q35_2b_specialist_population_v1 import (
    _arm,
    _load_jsonl,
    _load_traces,
    _pair_contract,
    _task_data,
    _task_name,
)

FAMILIES = {
    "table_analyst": {"specialist_table_join", "specialist_table_reconcile"},
    "source_inspector": {"specialist_source_ast", "specialist_source_config"},
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_rows(arm: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["task"]: row for row in arm["tasks"]}


def _router_calls(path: Path) -> int:
    return sum(row.get("role") == "specialist_router" for row in _load_jsonl(path))


def summarize(
    *,
    expert_id: str,
    control_traces: Path,
    control_audit: Path,
    treatment_traces: Path,
    treatment_audit: Path,
    control_model: str,
    treatment_model: str,
    expected_tasks: int,
    minimum_worker_activations: int,
    minimum_treatment_hard_successes: int,
    minimum_hard_successes_per_family: int,
    minimum_paired_recoveries: int,
    maximum_paired_regressions: int,
) -> dict[str, Any]:
    if expert_id not in FAMILIES:
        raise ValueError(f"unsupported specialist: {expert_id}")
    if expected_tasks < 1 or minimum_worker_activations < 1:
        raise ValueError("task and worker-activation counts must be positive")
    if (
        minimum_treatment_hard_successes < 0
        or minimum_hard_successes_per_family < 0
        or minimum_paired_recoveries < 0
        or maximum_paired_regressions < 0
    ):
        raise ValueError("competence thresholds must be non-negative")
    control_values = _load_traces(control_traces)
    treatment_values = _load_traces(treatment_traces)
    control_contracts = {
        _task_name(trace): _pair_contract(trace) for trace in control_values
    }
    treatment_contracts = {
        _task_name(trace): _pair_contract(trace) for trace in treatment_values
    }
    if control_contracts != treatment_contracts:
        raise ValueError("control and treatment task contracts differ")
    observed_families = {
        _task_data(trace).get("family") for trace in control_values
    }
    if not observed_families or not observed_families.issubset(FAMILIES[expert_id]):
        raise ValueError(
            f"unexpected {expert_id} competence families: {sorted(observed_families, key=str)}"
        )

    control = _arm(
        control_traces,
        control_audit,
        {expert_id: control_model},
        expected_tasks,
    )
    treatment = _arm(
        treatment_traces,
        treatment_audit,
        {expert_id: treatment_model},
        expected_tasks,
    )
    control_rows = _task_rows(control)
    treatment_rows = _task_rows(treatment)
    paired_recoveries = [
        name
        for name in sorted(control_rows)
        if not control_rows[name]["hard_success"]
        and treatment_rows[name]["hard_success"]
    ]
    paired_regressions = [
        name
        for name in sorted(control_rows)
        if control_rows[name]["hard_success"]
        and not treatment_rows[name]["hard_success"]
    ]
    control_sequence = [
        expert
        for row in control["tasks"]
        for expert in row["selected_experts"]
    ]
    treatment_sequence = [
        expert
        for row in treatment["tasks"]
        for expert in row["selected_experts"]
    ]
    fixed_route_exact = (
        control_sequence == treatment_sequence
        and all(expert == expert_id for expert in control_sequence)
    )
    no_router_calls = (
        _router_calls(control_audit) == 0 and _router_calls(treatment_audit) == 0
    )
    thresholds = {
        "minimum_worker_activations": minimum_worker_activations,
        "minimum_treatment_hard_successes": minimum_treatment_hard_successes,
        "minimum_hard_successes_per_family": minimum_hard_successes_per_family,
        "minimum_paired_recoveries": minimum_paired_recoveries,
        "maximum_paired_regressions": maximum_paired_regressions,
    }
    evidence = {
        "minimum_worker_activations": (
            treatment["worker_activations"].get(expert_id, 0)
            >= minimum_worker_activations
        ),
        "minimum_treatment_hard_successes": (
            treatment["hard_successes"] >= minimum_treatment_hard_successes
        ),
        "minimum_hard_successes_per_family": all(
            treatment["family_hard_successes"].get(family, 0)
            >= minimum_hard_successes_per_family
            for family in FAMILIES[expert_id]
        ),
        "minimum_paired_recoveries": (
            len(paired_recoveries) >= minimum_paired_recoveries
        ),
        "maximum_paired_regressions": (
            len(paired_regressions) <= maximum_paired_regressions
        ),
        "fixed_route_exact": fixed_route_exact,
        "router_absent": no_router_calls,
        "control_provenance_exact": (
            control["exact_route_sequence"] and control["route_models_exact"]
        ),
        "treatment_provenance_exact": (
            treatment["exact_route_sequence"] and treatment["route_models_exact"]
        ),
    }
    return {
        "schema_version": "q35-2b-specialist-competence-paired-summary/v1",
        "expert_id": expert_id,
        "task_contracts_identical": True,
        "router_taxonomy_evaluated": False,
        "acceptance_gates_relaxed": False,
        "expected_tasks": expected_tasks,
        "inputs": {
            "control_traces_sha256": _sha256_file(control_traces),
            "control_audit_sha256": _sha256_file(control_audit),
            "treatment_traces_sha256": _sha256_file(treatment_traces),
            "treatment_audit_sha256": _sha256_file(treatment_audit),
            "control_model": control_model,
            "treatment_model": treatment_model,
        },
        "control": control,
        "treatment": treatment,
        "paired_recoveries": paired_recoveries,
        "paired_recovery_count": len(paired_recoveries),
        "paired_regressions": paired_regressions,
        "paired_regression_count": len(paired_regressions),
        "thresholds": thresholds,
        "acceptance": evidence,
        "competence_gate_passed": all(evidence.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expert-id", choices=tuple(FAMILIES), required=True
    )
    parser.add_argument("--control-traces", type=Path, required=True)
    parser.add_argument("--control-audit", type=Path, required=True)
    parser.add_argument("--treatment-traces", type=Path, required=True)
    parser.add_argument("--treatment-audit", type=Path, required=True)
    parser.add_argument("--control-model", required=True)
    parser.add_argument("--treatment-model", required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--minimum-worker-activations", type=int, required=True)
    parser.add_argument("--minimum-treatment-hard-successes", type=int, required=True)
    parser.add_argument("--minimum-hard-successes-per-family", type=int, required=True)
    parser.add_argument("--minimum-paired-recoveries", type=int, required=True)
    parser.add_argument("--maximum-paired-regressions", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    values = vars(args).copy()
    output = values.pop("output")
    result = summarize(**values)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is not None:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite immutable result: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
