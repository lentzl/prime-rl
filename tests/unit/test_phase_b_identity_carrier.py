from copy import deepcopy

import pytest

from prime_rl.phase_b_contract import PhaseBContractError, canonical_json_sha256
from prime_rl.phase_b_identity_carrier import (
    ACTIONS,
    ARMS,
    IDENTITY_FIELDS,
    SAFETY_FIELDS,
    aligned_suffix_geometry,
    build_cache_guard_labels,
    evaluate_hic0,
    recursive_subclass_closure,
    validate_hic0_selection,
    validate_hic0_terminal_receipt,
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


def test_hic0_terminal_receipt_requires_literal_and_internal_hash() -> None:
    receipt = {
        "status": "b_hic0_inplace_carrier_not_nominated",
        "terminal": "SUCCESS",
        "disposition": "b_hic0_inplace_carrier_not_nominated",
        "optimizer": None,
        "optimizer_updates": 0,
        "generation": False,
        "cache": False,
        "worker_loaded": False,
        "H176_loaded": False,
        "strand_a_combined": False,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt, omitted_fields=("receipt_sha256",))
    validate_hic0_terminal_receipt(receipt, success_file=True)
    receipt["status"] = "SUCCESS"
    with pytest.raises(PhaseBContractError, match="status literal"):
        validate_hic0_terminal_receipt(receipt, success_file=True)


def test_recursive_subclass_closure_is_transitive() -> None:
    class Base:
        pass

    class Child(Base):
        pass

    class Grandchild(Child):
        pass

    assert recursive_subclass_closure(Base) == {Base, Child, Grandchild}
