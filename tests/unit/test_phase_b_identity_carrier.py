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
    validate_hic0_selection,
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
                "inplace_zero_identity": dict.fromkeys(IDENTITY_FIELDS, True),
                "rmsnorm_amplification": {
                    "A_insert": [12.0] * 8,
                    "A_inplace": [1.0] * 8,
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
