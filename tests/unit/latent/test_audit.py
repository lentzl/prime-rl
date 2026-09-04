import copy

import pytest

from prime_rl.latent.audit import canonical_plan_hash, validate_launch_plan

_E33 = "e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47"
_H176 = "77980e247bbccd6463ddda02cd42d2c357e15f8ec1ad0ea84627e008a8674a1e"


def _launch_plan() -> dict[str, object]:
    plan: dict[str, object] = {
        "schema_version": "prime-rl/latent-a0-a1-preregistration/v1",
        "status": "preregistered",
        "execution_authorization": "owner_approved",
        "frozen_by": "independent-evaluator",
        "frozen_at_utc": "2026-09-05T00:00:00Z",
        "base_code_commit": "abcdef0",
        "tokenizer_template_sha256": "3" * 64,
        "bridge_config_sha256": "4" * 64,
        "capture_spec_sha256": "5" * 64,
        "protected_checkpoints": {"coordinator_e33": _E33, "worker_h176": _H176},
        "workspace_contract": {
            "schema_version": "prime-rl/latent-workspace/v1",
            "slots": 8,
            "workspace_width": 256,
            "receiver_embedding_width": 2048,
            "capture_layer": -1,
            "capture_boundary": "accepted_delegation",
            "maximum_capture_tokens": 128,
            "parent_gradient": "detached",
            "receiver_injection_boundary": "after_child_objective_before_assistant_opening",
            "operational_direction": "coordinator_parent_to_coordinator_child",
        },
        "primary_audit_arms": ["M0", "MOTH", "MSELF", "MCUR"],
        "diagnostic_arms": ["ZERO", "NOISE"],
        "split_information_bank": {
            "held_out_seed": 123,
            "train_seed": 121,
            "validation_seed": 122,
            "arm_order_seed": 124,
            "train_manifest_sha256": "1" * 64,
            "validation_manifest_sha256": "6" * 64,
            "held_out_manifest_sha256": "2" * 64,
            "families": ["keyed_numeric", "relational_join", "config_structure", "ownership_graph"],
            "queries_per_evidence": 3,
            "train_examples_per_family": 8,
            "validation_examples_per_family": 4,
            "held_out_examples_per_family": 4,
        },
        "admission_thresholds": {
            "minimum_complete_held_out_tasks": 16,
            "minimum_paired_recoveries": 4,
            "minimum_recovery_families": 2,
            "minimum_current_vs_other_utility_gap": 0.1,
            "practical_equivalence_margin": 0.05,
            "mself_gpu_seconds_relative_tolerance": 0.1,
        },
        "mself_compute_match": {
            "primary_method": "equal_operation",
            "sender_feature_forwards": 1,
            "self_feature_forwards": 1,
            "sender_bridge_forwards": 1,
            "self_bridge_forwards": 1,
            "sender_feature_input_tokens": 128,
            "self_feature_input_tokens": 128,
            "receiver_decode_budget_tokens": 32,
            "same_physical_gpu": True,
        },
    }
    plan["plan_sha256"] = canonical_plan_hash(plan)
    return plan


def test_independently_frozen_plan_passes() -> None:
    validate_launch_plan(_launch_plan())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: plan.update(status="draft_requires_independent_freeze"),
        lambda plan: plan["split_information_bank"].update(held_out_seed=None),
        lambda plan: plan["admission_thresholds"].update(practical_equivalence_margin=None),
        lambda plan: plan["mself_compute_match"].update(self_feature_forwards=2),
    ],
)
def test_incomplete_or_compute_unmatched_plan_fails_closed(mutation) -> None:
    plan = copy.deepcopy(_launch_plan())
    mutation(plan)
    plan["plan_sha256"] = canonical_plan_hash(plan)

    with pytest.raises(ValueError):
        validate_launch_plan(plan)
