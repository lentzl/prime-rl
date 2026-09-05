import copy
import json
from pathlib import Path

import pytest

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a0nc import (
    _ASSETS,
    _FAILURE_CLASSIFICATION,
    _INTERPRETATION,
    _MECHANISM,
    _REJECTION,
    CacheAllocationDetected,
    DiagnosticIncomplete,
    classify_failure,
    load_plan,
    recursive_subclass_closure,
    validate_plan,
)

EXPERIMENT = Path("experiments/qwen35-2b-latent-workspace-v1")


def _inputs():
    prior = json.loads((EXPERIMENT / "a0-cache-calibration-plan-v1.json").read_text())
    bank_sha = file_sha256(EXPERIMENT / "a0-nocache-bank-v1.json")
    return prior, bank_sha


def _plan(prior, bank_sha):
    resources = copy.deepcopy(prior["resource_bounds"])
    resources["output_root"] = "/home/ubuntu/rlm/outputs/latent-a0-nocache-receiver-v1"
    plan = {
        "schema_version": "prime-rl/latent-a0-nocache-plan/v1",
        "status": "preregistered",
        "execution_authorization": "root_and_evaluator_review_required",
        "mechanism_code_commit": "a" * 40,
        "asset_sha256": {path: f"{index:x}"[-1] * 64 for index, path in enumerate(sorted(_ASSETS), 1)},
        "plan_sha256": "",
        "bank_sha256": bank_sha,
        "prior_cache_rejection": copy.deepcopy(_REJECTION),
        "protected_checkpoints": copy.deepcopy(prior["protected_checkpoints"]),
        "remote_paths": copy.deepcopy(prior["remote_paths"]),
        "runtime": copy.deepcopy(prior["runtime"]),
        "mechanism": copy.deepcopy(_MECHANISM),
        "resource_bounds": resources,
        "failure_classification": copy.deepcopy(_FAILURE_CLASSIFICATION),
        "interpretation_boundary": _INTERPRETATION,
    }
    plan["plan_sha256"] = canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    return plan


def test_a0nc_accepts_exact_fresh_four_probe_nocache_design():
    prior, bank_sha = _inputs()
    validate_plan(_plan(prior, bank_sha), prior=prior, bank_sha=bank_sha)


def test_a0nc_frozen_plan_closes_rejected_evidence_and_disjoint_banks():
    plan, bank = load_plan(EXPERIMENT / "a0-nocache-plan-v1.json", EXPERIMENT / "a0-nocache-bank-v1.json")
    assert plan["plan_sha256"] == "e39b9ddad4e851070d46eef9c9d76ee7537214460fafdfc7ce33e56850e89a97"
    assert len(bank["examples"]) == 4


def test_a0nc_recursive_cache_subclass_closure_includes_indirect_classes():
    class Base:
        pass

    class Direct(Base):
        pass

    class Indirect(Direct):
        pass

    assert recursive_subclass_closure(Base) == {Base, Direct, Indirect}


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            CacheAllocationDetected("actual cache allocation"),
            ("nocache_receiver_mechanism_rejected", "cache_allocation_or_past_key_values_detected"),
        ),
        (
            DiagnosticIncomplete("nonfinite or diagnostic contract failure"),
            ("diagnostic_incomplete", "diagnostic_execution_or_finiteness_failure"),
        ),
        (TimeoutError("deadline"), ("infrastructure_invalid", "environment_provenance_timeout_or_oom")),
        (RuntimeError("environment failure"), ("infrastructure_invalid", "environment_provenance_timeout_or_oom")),
    ],
)
def test_a0nc_terminal_classification_is_exact(error, expected):
    assert classify_failure(error) == expected


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: plan["mechanism"].update(complete_distinct_probe_floor=3),
        lambda plan: plan["mechanism"].update(use_cache=True),
        lambda plan: plan["mechanism"].update(past_key_values_output_must_be_none=False),
        lambda plan: plan["mechanism"].update(l_id_l_e_logits_finite_each_step=False),
        lambda plan: plan["mechanism"].update(soft_repeat_logits_bitwise_equal_each_step=False),
        lambda plan: plan["prior_cache_rejection"].update(status="relative_cache_calibrated"),
    ],
)
def test_a0nc_rejects_probe_cache_determinism_or_evidence_drift(mutation):
    prior, bank_sha = _inputs()
    plan = _plan(prior, bank_sha)
    mutation(plan)
    plan["plan_sha256"] = canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    with pytest.raises(ValueError):
        validate_plan(plan, prior=prior, bank_sha=bank_sha)
