import copy
import json
from pathlib import Path

import pytest

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a0dr2 import (
    _ASSET_PATHS,
    _FAILED_RUN,
    _INTERPRETATION,
    _REPAIR,
    invoke_after_predecode_snapshot,
    validate_a0dr2_plan,
    validate_a0dr2_receipt,
)
from tests.unit.latent.test_a0dr import _receipt as _a0dr_receipt

EXPERIMENT = Path("experiments/qwen35-2b-latent-workspace-v1")


def _inputs() -> tuple[dict[str, object], str]:
    prior = json.loads((EXPERIMENT / "a0dr-cache-diagnostic-plan-v1.json").read_text())
    return prior, file_sha256(EXPERIMENT / "a0-mechanism-bank-v1.json")


def _plan(prior: dict[str, object], bank_hash: str) -> dict[str, object]:
    resources = copy.deepcopy(prior["resource_bounds"])
    resources["output_root"] = "/home/ubuntu/rlm/outputs/latent-a0dr2-cache-diagnostic-v1"
    plan: dict[str, object] = {
        "schema_version": "prime-rl/latent-a0dr2-cache-diagnostic-plan/v1",
        "status": "preregistered",
        "execution_authorization": "root_review_required",
        "mechanism_code_commit": "a" * 40,
        "asset_sha256": {path: f"{index:x}"[-1] * 64 for index, path in enumerate(sorted(_ASSET_PATHS), start=1)},
        "plan_sha256": "",
        "bank_sha256": bank_hash,
        "supersedes_failed_run": copy.deepcopy(_FAILED_RUN),
        "evidence_capture_repair": copy.deepcopy(_REPAIR),
        "bound_evidence": copy.deepcopy(prior["bound_evidence"]),
        "protected_checkpoints": copy.deepcopy(prior["protected_checkpoints"]),
        "remote_paths": copy.deepcopy(prior["remote_paths"]),
        "runtime": copy.deepcopy(prior["runtime"]),
        "diagnostic": copy.deepcopy(prior["diagnostic"]),
        "resource_bounds": resources,
        "failure_classification": copy.deepcopy(prior["failure_classification"]),
        "interpretation_boundary": _INTERPRETATION,
    }
    plan["plan_sha256"] = canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    return plan


def test_snapshot_precedes_mutation_of_aliased_cache() -> None:
    cache = {"sequence_length": 55}
    prepared: dict[str, object] = {"past_key_values": cache}

    def summarize(value: dict[str, object]) -> dict[str, object]:
        return {"sequence_length": value["past_key_values"]["sequence_length"]}

    def invoke() -> str:
        cache["sequence_length"] += 1
        return "decoded"

    result, snapshot = invoke_after_predecode_snapshot(prepared, summarize=summarize, invoke=invoke)
    assert result == "decoded"
    assert snapshot == {"sequence_length": 55}
    assert cache == {"sequence_length": 56}


def test_a0dr2_accepts_only_capture_timing_repair() -> None:
    prior, bank_hash = _inputs()
    validate_a0dr2_plan(_plan(prior, bank_hash), bank_sha256=bank_hash, a0dr_plan=prior)


def test_a0dr2_receipt_binds_predecode_capture_and_failed_run() -> None:
    prior, bank_hash = _inputs()
    plan = _plan(prior, bank_hash)
    receipt = _a0dr_receipt(plan)
    receipt["schema_version"] = "prime-rl/latent-a0dr2-cache-diagnostic-receipt/v1"
    receipt["supersedes_failed_run"] = copy.deepcopy(_FAILED_RUN)
    receipt["evidence_capture_timing"] = (
        "immediately_after_prepare_inputs_for_generation_before_full_or_cached_decode"
    )
    receipt["receipt_sha256"] = canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    validate_a0dr2_receipt(receipt, plan=plan)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: plan["supersedes_failed_run"].update(stage="arm_S55_auto_position"),
        lambda plan: plan["evidence_capture_repair"].update(
            numerical_arms_ids_positions_masks_and_references_changed=True
        ),
        lambda plan: plan["diagnostic"].update(reference_is_promotion_gate=True),
        lambda plan: plan["resource_bounds"].update(maximum_wall_minutes=60),
    ],
)
def test_a0dr2_rejects_evidence_numerical_gate_or_resource_drift(mutation) -> None:
    prior, bank_hash = _inputs()
    plan = _plan(prior, bank_hash)
    mutation(plan)
    plan["plan_sha256"] = canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    with pytest.raises(ValueError):
        validate_a0dr2_plan(plan, bank_sha256=bank_hash, a0dr_plan=prior)
