import copy
import json
from pathlib import Path

import pytest

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a0cal import (
    _A0DR2_EVIDENCE,
    _ASSET_PATHS,
    _CALIBRATION_BASIS,
    _CRITERION,
    _EXPECTED_PROBES,
    _FIXED_IDS,
    _FIXED_IDS_SHA256,
    _INTERPRETATION,
    calculate_probe_criterion,
    validate_a0cal_plan,
)

EXPERIMENT = Path("experiments/qwen35-2b-latent-workspace-v1")


def _inputs() -> tuple[dict[str, object], str]:
    prior = json.loads((EXPERIMENT / "a0dr2-cache-diagnostic-plan-v1.json").read_text())
    return prior, file_sha256(EXPERIMENT / "a0-mechanism-bank-v1.json")


def _plan(prior: dict[str, object], bank_hash: str) -> dict[str, object]:
    resources = copy.deepcopy(prior["resource_bounds"])
    resources["output_root"] = "/home/ubuntu/rlm/outputs/latent-a0-cache-calibration-v1"
    plan: dict[str, object] = {
        "schema_version": "prime-rl/latent-a0-cache-calibration-plan/v1",
        "status": "preregistered",
        "execution_authorization": "root_and_evaluator_review_required",
        "mechanism_code_commit": "a" * 40,
        "asset_sha256": {path: f"{index:x}"[-1] * 64 for index, path in enumerate(sorted(_ASSET_PATHS), start=1)},
        "plan_sha256": "",
        "bank_sha256": bank_hash,
        "a0dr2_evidence": copy.deepcopy(_A0DR2_EVIDENCE),
        "calibration_basis": copy.deepcopy(_CALIBRATION_BASIS),
        "protected_checkpoints": copy.deepcopy(prior["protected_checkpoints"]),
        "remote_paths": copy.deepcopy(prior["remote_paths"]),
        "runtime": copy.deepcopy(prior["runtime"]),
        "expected_probes": copy.deepcopy(_EXPECTED_PROBES),
        "length_control": {
            "token_ids": _FIXED_IDS,
            "token_ids_sha256": _FIXED_IDS_SHA256,
            "tokens_must_be_non_special": True,
            "insertion_count": 8,
            "same_boundary_mask_and_positions_as_soft": True,
        },
        "criterion": copy.deepcopy(_CRITERION),
        "resource_bounds": resources,
        "failure_classification": copy.deepcopy(prior["failure_classification"]),
        "interpretation_boundary": _INTERPRETATION,
    }
    plan["plan_sha256"] = canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    return plan


def _arm(name: str, values: list[float]) -> dict[str, object]:
    return {
        "arm": name,
        "initial_cache_sequence_length": 55,
        "prefill_last_logits_sha256": "1" * 64,
        "steps": [
            {
                "cache_sequence_length": 56 + index,
                "maximum_absolute_logit_difference": 0.2,
                "normalized_rms": value,
                "greedy_equal": True,
                "cached_logits_sha256": f"{index + 2}" * 64,
                "full_logits_sha256": f"{index + 6}" * 64,
            }
            for index, value in enumerate(values)
        ],
    }


def test_relative_criterion_accepts_bounded_soft_excess() -> None:
    discrete = _arm("L_E", [0.02] * 4)
    result = calculate_probe_criterion(
        {"L_ID": copy.deepcopy(discrete), "L_E": discrete, "S": _arm("S", [0.024, 0.02, 0.021, 0.022])}
    )
    assert result["qualifies"] is True
    assert result["mean_excess_allowance"] == pytest.approx(0.005)


def test_relative_criterion_rejects_single_step_excess_without_averaging_it_away() -> None:
    discrete = _arm("L_E", [0.02] * 4)
    result = calculate_probe_criterion(
        {"L_ID": copy.deepcopy(discrete), "L_E": discrete, "S": _arm("S", [0.026, 0.018, 0.018, 0.018])}
    )
    assert result["mean_relative_ok"] is True
    assert result["step_relative_ok"] is False
    assert result["qualifies"] is False


def test_a0cal_accepts_exact_prospective_four_probe_design() -> None:
    prior, bank_hash = _inputs()
    validate_a0cal_plan(_plan(prior, bank_hash), bank_sha256=bank_hash, prior=prior)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: plan["criterion"].update(complete_distinct_probe_floor=3),
        lambda plan: plan["criterion"].update(maximum_soft_minus_discrete_nrms_per_step=0.006),
        lambda plan: plan["criterion"].update(posthoc_threshold_change_allowed=True),
        lambda plan: plan["calibration_basis"].update(threshold_freeze_timing="after model exposure"),
        lambda plan: plan["length_control"].update(token_ids=[1] * 8),
        lambda plan: plan["interpretation_boundary"].replace("A1 remains blocked", "A1 authorized"),
    ],
)
def test_a0cal_rejects_probe_threshold_control_or_scope_drift(mutation) -> None:
    prior, bank_hash = _inputs()
    plan = _plan(prior, bank_hash)
    changed = mutation(plan)
    if isinstance(changed, str):
        plan["interpretation_boundary"] = changed
    plan["plan_sha256"] = canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    with pytest.raises(ValueError):
        validate_a0cal_plan(plan, bank_sha256=bank_hash, prior=prior)
