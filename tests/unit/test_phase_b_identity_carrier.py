import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from prime_rl.phase_b_contract import PhaseBContractError, canonical_json_sha256
from prime_rl.phase_b_identity_carrier import (
    ACTIONS,
    ARMS,
    EXPECTED_CACHE_CLASSES,
    IDENTITY_FIELDS,
    SAFETY_FIELDS,
    aligned_suffix_geometry,
    build_cache_guard_labels,
    evaluate_hic0,
    expected_memory_checkpoint_labels,
    ordered_subclass_closure,
    recursive_subclass_closure,
    validate_hic0_selection,
    validate_hic0_terminal_receipt,
    validate_suffix_target_ids,
)


def _rows() -> list[dict[str, object]]:
    rows = []
    for index in range(12):
        base = 1.0
        arms = {
            "BASE": {"nll": base, "margin": 0.0, "finite": True},
            "INSERT_ZERO": {"nll": base + 0.02, "margin": -0.01, "finite": True},
            "INSERT_EPS": {"nll": base + 0.10, "margin": -0.10, "finite": True},
            "INPLACE_ZERO": {"nll": base, "margin": 0.0, "finite": True},
            "INPLACE_EPS": {"nll": base + 0.005, "margin": -0.01, "finite": True},
        }
        rows.append(
            {
                "task_key": f"row-{index}",
                "action": ACTIONS[index % 3],
                "arms": arms,
                "same_residual_bytes_insert_and_inplace": True,
                "inplace_zero_identity": dict.fromkeys(IDENTITY_FIELDS, True),
                "rmsnorm_amplification": {
                    "A_insert": [12.0] * 8,
                    "A_inplace": [1.0] * 8,
                    "insert_norm_residual_cosine": [0.5] * 8,
                    "inplace_norm_cosine": [0.99] * 8,
                },
                "drift": {
                    "hidden_insert_eps_nrms": 1.0,
                    "hidden_inplace_eps_nrms": 0.1,
                    "logit_insert_eps_nrms": 1.0,
                    "logit_inplace_eps_nrms": 0.1,
                },
            }
        )
    return rows


def test_hic0_nomination_applies_all_frozen_gates() -> None:
    result = evaluate_hic0(_rows(), safety=dict.fromkeys(SAFETY_FIELDS, True))

    assert result["nominated"] is True
    assert result["disposition"] == "b_hic0_inplace_carrier_nominated"
    assert all(result["gates"].values())
    assert result["summaries"]["penalty_removal_fraction"] == pytest.approx(0.95)
    assert result["summaries"]["inplace_strict_wins"] == 12


def test_hic0_threshold_miss_is_valid_not_nominated() -> None:
    rows = _rows()
    rows[0]["arms"]["INPLACE_EPS"]["nll"] = 1.20

    result = evaluate_hic0(rows, safety=dict.fromkeys(SAFETY_FIELDS, True))

    assert result["nominated"] is False
    assert result["disposition"] == "b_hic0_inplace_carrier_not_nominated"
    assert result["gates"]["5_inplace_penalty_removed"] is False


def test_hic0_zero_drift_denominator_is_incomplete_gate_evidence() -> None:
    rows = _rows()
    rows[3]["drift"]["hidden_insert_eps_nrms"] = 0.0

    result = evaluate_hic0(rows, safety=dict.fromkeys(SAFETY_FIELDS, True))

    assert result["gates"]["6_hidden_and_logit_drift_removed"] is False
    assert result["summaries"]["zero_drift_denominators"] == [
        {"task_key": "row-3", "kind": "hidden"}
    ]


def test_hic0_selection_requires_exact_order_hashes_and_balance() -> None:
    pairs = [
        {"task_key": f"row-{index}", "expected_action": ACTIONS[index % 3]}
        for index in range(12)
    ]
    selection = {
        "schema_version": "q35-2b-b-hic0-identity-carrier-selection/v1",
        "task_keys": [pair["task_key"] for pair in pairs],
        "key_actions": pairs,
    }
    selection["ordered_task_key_sha256"] = canonical_json_sha256(selection["task_keys"])
    selection["ordered_key_action_sha256"] = canonical_json_sha256(pairs)

    assert validate_hic0_selection(selection) == pairs
    broken = deepcopy(selection)
    broken["key_actions"][0]["expected_action"] = "delegate_terminal"
    with pytest.raises(PhaseBContractError, match="action-balanced"):
        validate_hic0_selection(broken)


def test_hic0_rejects_missing_or_reordered_safety_evidence() -> None:
    with pytest.raises(PhaseBContractError, match="safety evidence"):
        evaluate_hic0(_rows(), safety={"cache": True})
    reordered = {key: True for key in reversed(SAFETY_FIELDS)}
    with pytest.raises(PhaseBContractError, match="safety evidence"):
        evaluate_hic0(_rows(), safety=reordered)


def test_hic0_arm_mapping_order_is_frozen() -> None:
    rows = _rows()
    rows[0]["arms"] = {arm: rows[0]["arms"][arm] for arm in reversed(ARMS)}
    with pytest.raises(PhaseBContractError, match="arm order"):
        evaluate_hic0(rows, safety=dict.fromkeys(SAFETY_FIELDS, True))


def test_hic0_cache_schedule_and_alignment_are_exact() -> None:
    labels = build_cache_guard_labels()
    assert len(labels) == 147
    assert labels[:4] == [
        "CACHE_GUARD_ENTRY",
        "CACHE_GUARD_PRE_HIC0_R01_SOURCE_CAPTURE",
        "CACHE_GUARD_POST_HIC0_R01_SOURCE_CAPTURE",
        "CACHE_GUARD_PRE_HIC0_R01_BASE",
    ]
    assert labels[-2:] == ["CACHE_GUARD_FINAL", "CACHE_GUARD_EXIT"]
    assert aligned_suffix_geometry(total=100, supervised_start=80, insertion_index=70) == {
        "T": 100,
        "S": 80,
        "I": 70,
        "B": 79,
        "K": 21,
        "Q": 8,
        "TQ": 108,
        "SQ": 88,
        "BQ": 87,
    }


def test_hic0_drift_gate_uses_ratio_of_means() -> None:
    rows = _rows()
    rows[0]["drift"].update(
        hidden_insert_eps_nrms=100.0,
        hidden_inplace_eps_nrms=19.0,
        logit_insert_eps_nrms=100.0,
        logit_inplace_eps_nrms=19.0,
    )
    for row in rows[1:]:
        row["drift"].update(
            hidden_insert_eps_nrms=0.001,
            hidden_inplace_eps_nrms=0.001,
            logit_insert_eps_nrms=0.001,
            logit_inplace_eps_nrms=0.001,
        )

    result = evaluate_hic0(rows, safety=dict.fromkeys(SAFETY_FIELDS, True))

    assert result["summaries"]["hidden_drift_ratio_mean"] > 0.9
    assert result["gates"]["6_hidden_and_logit_drift_removed"] is True


def test_hic0_positive_row_gate_uses_strict_one_e_minus_six() -> None:
    rows = _rows()
    for row in rows[:4]:
        row["arms"]["INSERT_EPS"]["nll"] = 1.0000005

    result = evaluate_hic0(rows, safety=dict.fromkeys(SAFETY_FIELDS, True))

    assert result["summaries"]["P_insert"]["positive_rows_gt_1e_6"] == 8
    assert result["gates"]["3_insert_penalty_replicates"] is False


def _terminal_plan() -> dict[str, object]:
    return {
        "_file_sha256": "p" * 64,
        "diagnostic_bank": {"selection_sha256": "s" * 64},
        "immutable_input_hashes": {"selection": "s" * 64, "bank": "b" * 64},
        "protected_model": {"weight_sha256": "w" * 64},
        "model_metadata_sha256": {"config.json": "m" * 64},
        "resources": {
            "minimum_host_ram_bytes": 64 * 1024**3,
            "minimum_free_disk_bytes": 60 * 1024**3,
        },
    }


def _seal(receipt: dict[str, object]) -> dict[str, object]:
    receipt["receipt_sha256"] = canonical_json_sha256(
        receipt, omitted_fields=("receipt_sha256",)
    )
    return receipt


def _success_receipt() -> dict[str, object]:
    plan = _terminal_plan()
    rows = _rows()
    nomination = evaluate_hic0(rows, safety=dict.fromkeys(SAFETY_FIELDS, True))
    cache_classes = [
        {
            "fqcn": fqcn,
            "module_path": f"/frozen/.venv/{relative_path}",
            "module_sha256": source_sha,
            "distribution": distribution,
        }
        for fqcn, relative_path, source_sha, distribution in EXPECTED_CACHE_CLASSES
    ]
    return _seal({
        "schema_version": "q35-2b-phase-b-hic0-identity-carrier-success/v1",
        "status": "b_hic0_inplace_carrier_nominated",
        "terminal": "SUCCESS",
        "disposition": "b_hic0_inplace_carrier_nominated",
        "claim_class": "zero_update_identity_carrier_causal_diagnostic_nomination_only",
        "execution_commit": "e" * 40,
        "plan_sha256": plan["_file_sha256"],
        "selection_sha256": "s" * 64,
        "model_loaded": True,
        "saved_model_state": False,
        "B1R_candidates_reused": False,
        "optimizer": None,
        "optimizer_updates": 0,
        "generation": False,
        "cache": False,
        "worker_loaded": False,
        "H176_loaded": False,
        "strand_a_combined": False,
        "source_forwards": 12,
        "receiver_forwards": 60,
        "backward_forwards_reused": 1,
        "rows": rows,
        "nomination": nomination,
        "backward": {
            "row": "document_adaptive_d2-v4-i35100",
            "receiver_forward_reused": True,
            "extra_receiver_forwards": 0,
            "residual_gradient": {"finite": True, "nonzero": True},
            "encoder_group": {"finite": True, "nonzero": True},
            "receiver_group": {"finite": True, "nonzero": True},
            "all_named_gradients_finite": True,
            "e33_gradients_absent": True,
        },
        "cache_guard": {
            "complete": True,
            "label_count": 147,
            "canonical_label_sha256": "8230eae3b60a7fd00d7bfb557563a9d9ca32764ace262447275bd40538818471",
            "closure_check_count": 147,
            "closure_checked_at_every_recorded_label": True,
            "restored_in_finally": True,
            "model_calls": 72,
            "dynamic_cache_trip_count": 1,
            "exit_recorded": True,
            "recursively_closed_config_count": 3,
            "classes": cache_classes,
        },
        "protection": {
            "e33_tensor_pre": "t" * 64,
            "e33_tensor_post": "t" * 64,
            "e33_file_pre": "w" * 64,
            "e33_file_post": "w" * 64,
            "metadata_pre": {"config.json": "m" * 64},
            "metadata_post": {"config.json": "m" * 64},
            "codec_pre": "c" * 64,
            "codec_post": "c" * 64,
        },
        "immutable_input_hashes": {
            "plan": "p" * 64,
            "selection": "s" * 64,
            "bank": "b" * 64,
        },
        "preflight_resources": {
            "available_host_ram_bytes": 70 * 1024**3,
            "free_disk_bytes": 70 * 1024**3,
            "offline_environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        },
        "allocator": {
            "cap_bytes": 32 * 1024**3,
            "requested_fraction": 0.7,
            "observed_fraction": 0.7,
        },
        "cuda_memory": {"maximum_allocated_bytes": 1, "maximum_reserved_bytes": 1},
        "cuda_memory_ledger": [
            {"checkpoint": label, "maximum_allocated_bytes": 1, "maximum_reserved_bytes": 1}
            for label in expected_memory_checkpoint_labels()
        ],
        "promotion": {
            "admitted": False,
            "diagnostic_rows_count_as_live_trajectories": False,
            "complete_live_trajectory_count": 0,
            "minimum_complete_live_trajectories_unchanged": 4,
        },
    })


def test_hic0_terminal_receipt_requires_deep_literal_and_internal_hash() -> None:
    receipt = _success_receipt()
    validate_hic0_terminal_receipt(
        receipt,
        success_file=True,
        plan=_terminal_plan(),
        execution_commit="e" * 40,
    )
    receipt["status"] = "SUCCESS"
    with pytest.raises(PhaseBContractError, match="status literal"):
        validate_hic0_terminal_receipt(
            receipt,
            success_file=True,
            plan=_terminal_plan(),
            execution_commit="e" * 40,
        )


def test_hic0_terminal_receipt_rejects_deep_cache_tamper() -> None:
    receipt = _success_receipt()
    receipt["cache_guard"]["classes"][0]["module_sha256"] = "0" * 64
    _seal(receipt)
    with pytest.raises(PhaseBContractError, match="cache evidence"):
        validate_hic0_terminal_receipt(
            receipt,
            success_file=True,
            plan=_terminal_plan(),
            execution_commit="e" * 40,
        )


def test_hic0_failure_receipt_requires_class_and_protected_audit() -> None:
    plan = _terminal_plan()
    failure = _seal(
        {
            "schema_version": "q35-2b-phase-b-hic0-identity-carrier-failure/v1",
            "status": "b_hic0_incomplete",
            "terminal": "FAILURE",
            "disposition": "b_hic0_incomplete",
            "failure_class": "contract_or_evidence_incomplete",
            "execution_commit": "e" * 40,
            "plan_sha256": "p" * 64,
            "selection_sha256": "s" * 64,
            "model_loaded": True,
            "saved_model_state": False,
            "B1R_candidates_reused": False,
            "optimizer": None,
            "optimizer_updates": 0,
            "generation": False,
            "cache": False,
            "worker_loaded": False,
            "H176_loaded": False,
            "strand_a_combined": False,
            "post_failure_hash_audit": {
                "audit_complete": True,
                "immutable_input_hashes": {
                    "plan": "p" * 64,
                    "selection": "s" * 64,
                    "bank": "b" * 64,
                },
                "immutable_input_hashes_match": True,
                "e33_tensor_post": "t" * 64,
                "e33_tensor_reference_available": True,
                "e33_tensor_preserved": True,
                "e33_disk_and_metadata_exact": True,
            },
        }
    )
    validate_hic0_terminal_receipt(
        failure,
        success_file=False,
        plan=plan,
        execution_commit="e" * 40,
    )
    failure["failure_class"] = "infrastructure"
    _seal(failure)
    with pytest.raises(PhaseBContractError, match="class or postflight"):
        validate_hic0_terminal_receipt(
            failure,
            success_file=False,
            plan=plan,
            execution_commit="e" * 40,
        )


def test_recursive_subclass_closure_is_transitive() -> None:
    class Base:
        pass

    class Child(Base):
        pass

    class Grandchild(Child):
        pass

    assert recursive_subclass_closure(Base) == {Base, Child, Grandchild}
    assert ordered_subclass_closure(Base) == tuple(
        sorted((Base, Child, Grandchild), key=lambda cls: (cls.__module__, cls.__qualname__))
    )


def test_hic0_cache_class_census_is_exact_and_deterministic() -> None:
    assert len(EXPECTED_CACHE_CLASSES) == 8
    assert [item[0] for item in EXPECTED_CACHE_CLASSES] == sorted(item[0] for item in EXPECTED_CACHE_CLASSES)
    assert {item[3] for item in EXPECTED_CACHE_CLASSES} == {
        "flash-linear-attention==0.5.2",
        "transformers==5.6.2",
    }
    assert all(len(item[2]) == 64 for item in EXPECTED_CACHE_CLASSES)


def test_hic0_explicit_cache_imports_precede_guard_and_generation_config_is_closed() -> None:
    source = (
        Path(__file__).parents[2] / "scripts/latent/run_phase_b_identity_carrier_v1.py"
    ).read_text(encoding="utf-8")
    qwen_import = source.index(
        'importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")'
    )
    fla_import = source.index('importlib.import_module("fla.models.utils")')
    guard = source.index("guard = _CacheGuard(model, transformers=transformers)")
    assert qwen_import < guard and fla_import < guard
    assert 'config_candidates.append(getattr(self.model, "generation_config", None))' in source


def test_hic0_failure_classification_checks_contract_before_runtime() -> None:
    path = Path(__file__).parents[2] / "scripts/latent/run_phase_b_identity_carrier_v1.py"
    spec = importlib.util.spec_from_file_location("hic0_failure_classification_test", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    assert runner._failure_class(PhaseBContractError("alignment"))[0] == "b_hic0_incomplete"
    assert runner._failure_class(RuntimeError("native runtime"))[0] == "infrastructure_invalid"
    assert runner._failure_class(runner.CacheContractViolated("cache"))[0] == "b_hic0_nocache_rejected"
    assert runner._failure_class(runner.ResourceContractExceeded("cap"))[0] == "infrastructure_invalid"
    assert runner._failure_class(runner.ProvenanceContractViolated("hash"))[0] == "infrastructure_invalid"


def test_hic0_suffix_target_ids_must_be_inside_vocabulary() -> None:
    validate_suffix_target_ids([0, 4, 9], vocabulary_size=10)
    with pytest.raises(PhaseBContractError, match="outside the vocabulary"):
        validate_suffix_target_ids([0, 10], vocabulary_size=10)
    with pytest.raises(PhaseBContractError, match="outside the vocabulary"):
        validate_suffix_target_ids([-1, 3], vocabulary_size=10)
