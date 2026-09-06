from __future__ import annotations

import math
from collections import Counter, deque
from typing import Any, Sequence

from prime_rl.phase_b_contract import PhaseBContractError, canonical_json_sha256

ARMS = ("BASE", "INSERT_ZERO", "INSERT_EPS", "INPLACE_ZERO", "INPLACE_EPS")
ACTIONS = ("solve_owned", "delegate_terminal", "delegate_coordinator")
SLOTS = 8
EPSILON = 0.001
STRICT_WIN_EPSILON = 1e-6
EXPECTED_ROWS = 12
CACHE_LABEL_COUNT = 147
CACHE_LABEL_SHA256 = "8230eae3b60a7fd00d7bfb557563a9d9ca32764ace262447275bd40538818471"
IDENTITY_FIELDS = (
    "inputs_embeds",
    "attention_mask",
    "position_ids",
    "labels",
    "final_hidden",
    "first_suffix_logits",
    "nll",
    "margin",
)
SAFETY_FIELDS = ("backward", "provenance", "cache", "protection", "resource")
SUCCESS_STATUSES = (
    "b_hic0_inplace_carrier_nominated",
    "b_hic0_inplace_carrier_not_nominated",
)
FAILURE_STATUSES = ("b_hic0_nocache_rejected", "b_hic0_incomplete", "infrastructure_invalid")


def mean64(values: Sequence[float]) -> float:
    if not values:
        raise PhaseBContractError("B-HIC0 cannot aggregate an empty sequence")
    if not all(math.isfinite(value) for value in values):
        raise PhaseBContractError("B-HIC0 aggregate contains a non-finite value")
    return math.fsum(values) / len(values)


def normalized_rms_difference(candidate: Any, reference: Any, *, torch: Any, epsilon: float = 1e-12) -> float:
    if candidate.shape != reference.shape or candidate.numel() == 0:
        raise PhaseBContractError("B-HIC0 drift tensors must have one nonempty shared shape")
    candidate_cpu = candidate.detach().cpu().double()
    reference_cpu = reference.detach().cpu().double()
    difference = (candidate_cpu - reference_cpu).square().mean().sqrt()
    denominator = reference_cpu.square().mean().sqrt()
    if not bool(torch.isfinite(difference)) or not bool(torch.isfinite(denominator)):
        raise PhaseBContractError("B-HIC0 drift is non-finite")
    return float(difference / denominator.clamp_min(epsilon))


def recursive_subclass_closure(base: type) -> set[type]:
    closure: set[type] = {base}
    pending = deque([base])
    while pending:
        parent = pending.popleft()
        for child in parent.__subclasses__():
            if child not in closure:
                closure.add(child)
                pending.append(child)
    return closure


def build_cache_guard_labels() -> list[str]:
    labels = ["CACHE_GUARD_ENTRY"]
    for row_index in range(1, EXPECTED_ROWS + 1):
        for operation in ("SOURCE_CAPTURE", *ARMS):
            labels.extend(
                (
                    f"CACHE_GUARD_PRE_HIC0_R{row_index:02d}_{operation}",
                    f"CACHE_GUARD_POST_HIC0_R{row_index:02d}_{operation}",
                )
            )
    labels.extend(("CACHE_GUARD_FINAL", "CACHE_GUARD_EXIT"))
    if len(labels) != CACHE_LABEL_COUNT or canonical_json_sha256(labels) != CACHE_LABEL_SHA256:
        raise PhaseBContractError("B-HIC0 cache-guard label schedule differs")
    return labels


def aligned_suffix_geometry(*, total: int, supervised_start: int, insertion_index: int, slots: int = SLOTS) -> dict[str, int]:
    if not 0 < insertion_index <= supervised_start < total or slots != SLOTS:
        raise PhaseBContractError("B-HIC0 aligned suffix geometry is invalid")
    predictor = supervised_start - 1
    keep = total - supervised_start + 1
    return {
        "T": total,
        "S": supervised_start,
        "I": insertion_index,
        "B": predictor,
        "K": keep,
        "Q": slots,
        "TQ": total + slots,
        "SQ": supervised_start + slots,
        "BQ": predictor + slots,
    }


def validate_hic0_terminal_receipt(receipt: dict[str, Any], *, success_file: bool) -> None:
    statuses = SUCCESS_STATUSES if success_file else FAILURE_STATUSES
    if receipt.get("status") not in statuses or receipt.get("terminal") != ("SUCCESS" if success_file else "FAILURE"):
        raise PhaseBContractError("B-HIC0 terminal status literal differs")
    if receipt.get("disposition") != receipt.get("status"):
        raise PhaseBContractError("B-HIC0 disposition and top-level status differ")
    if receipt.get("receipt_sha256") != canonical_json_sha256(receipt, omitted_fields=("receipt_sha256",)):
        raise PhaseBContractError("B-HIC0 internal receipt hash differs")
    if receipt.get("optimizer") is not None or receipt.get("optimizer_updates") != 0:
        raise PhaseBContractError("B-HIC0 receipt crossed the zero-update boundary")
    if any(receipt.get(key) is not False for key in ("generation", "cache", "worker_loaded")):
        raise PhaseBContractError("B-HIC0 receipt crossed the no-generation/cache/worker boundary")
    if receipt.get("H176_loaded") is not False or receipt.get("strand_a_combined") is not False:
        raise PhaseBContractError("B-HIC0 receipt crossed the model/strand boundary")


def validate_hic0_selection(selection: dict[str, Any]) -> list[dict[str, str]]:
    if selection.get("schema_version") != "q35-2b-b-hic0-identity-carrier-selection/v1":
        raise PhaseBContractError("B-HIC0 selection schema differs")
    pairs = selection.get("key_actions")
    keys = selection.get("task_keys")
    if not isinstance(pairs, list) or not isinstance(keys, list) or len(pairs) != EXPECTED_ROWS:
        raise PhaseBContractError("B-HIC0 selection does not contain exactly 12 rows")
    if keys != [pair.get("task_key") for pair in pairs] or len(set(keys)) != EXPECTED_ROWS:
        raise PhaseBContractError("B-HIC0 selection order or uniqueness differs")
    if Counter(pair.get("expected_action") for pair in pairs) != Counter({action: 4 for action in ACTIONS}):
        raise PhaseBContractError("B-HIC0 selection is not action-balanced 4/4/4")
    if canonical_json_sha256(keys) != selection.get("ordered_task_key_sha256"):
        raise PhaseBContractError("B-HIC0 ordered key hash differs")
    if canonical_json_sha256(pairs) != selection.get("ordered_key_action_sha256"):
        raise PhaseBContractError("B-HIC0 key/action hash differs")
    return pairs


def evaluate_hic0(rows: list[dict[str, Any]], *, safety: dict[str, bool]) -> dict[str, Any]:
    if len(rows) != EXPECTED_ROWS:
        raise PhaseBContractError("B-HIC0 requires exactly 12 complete metric rows")
    if len({row.get("task_key") for row in rows}) != EXPECTED_ROWS:
        raise PhaseBContractError("B-HIC0 metric rows are not unique")
    if tuple(safety) != SAFETY_FIELDS or any(type(value) is not bool for value in safety.values()):
        raise PhaseBContractError("B-HIC0 safety evidence is absent or out of order")
    safety_passed = all(safety.values())
    finite_complete = True
    identity_zero = True
    p_insert: list[float] = []
    p_inplace: list[float] = []
    inplace_wins: list[bool] = []
    margin_delta: list[float] = []
    structural_insert_zero: list[float] = []
    insert_epsilon_increment: list[float] = []
    a_insert: list[float] = []
    a_inplace: list[float] = []
    row_amplification_ratio: list[float] = []
    hidden_drift_ratio: list[float] = []
    logit_drift_ratio: list[float] = []
    hidden_insert_drift: list[float] = []
    hidden_inplace_drift: list[float] = []
    logit_insert_drift: list[float] = []
    logit_inplace_drift: list[float] = []
    zero_drift_denominators: list[dict[str, str]] = []
    for row in rows:
        arms = row.get("arms")
        if not isinstance(arms, dict) or tuple(arms) != ARMS:
            raise PhaseBContractError("B-HIC0 arm order differs")
        for arm in ARMS:
            metric = arms[arm]
            finite_complete = finite_complete and metric.get("finite") is True and all(
                math.isfinite(float(metric[key])) for key in ("nll", "margin")
            )
        finite_complete = finite_complete and row.get("same_residual_bytes_insert_and_inplace") is True
        identity = row.get("inplace_zero_identity")
        identity_zero = (
            identity_zero
            and isinstance(identity, dict)
            and tuple(identity) == IDENTITY_FIELDS
            and all(value is True for value in identity.values())
        )
        base = arms["BASE"]
        insert = arms["INSERT_EPS"]
        inplace = arms["INPLACE_EPS"]
        p_insert.append(float(insert["nll"]) - float(base["nll"]))
        p_inplace.append(float(inplace["nll"]) - float(base["nll"]))
        structural_insert_zero.append(float(arms["INSERT_ZERO"]["nll"]) - float(base["nll"]))
        insert_epsilon_increment.append(float(insert["nll"]) - float(arms["INSERT_ZERO"]["nll"]))
        inplace_wins.append(float(inplace["nll"]) - float(insert["nll"]) <= -STRICT_WIN_EPSILON)
        margin_delta.append(float(inplace["margin"]) - float(base["margin"]))
        amplification = row.get("rmsnorm_amplification")
        if not isinstance(amplification, dict):
            raise PhaseBContractError("B-HIC0 RMSNorm evidence is absent")
        insert_slots = [float(value) for value in amplification.get("A_insert", [])]
        inplace_slots = [float(value) for value in amplification.get("A_inplace", [])]
        if len(insert_slots) != SLOTS or len(inplace_slots) != SLOTS:
            raise PhaseBContractError("B-HIC0 requires 8 RMSNorm ratios per arm and row")
        a_insert.extend(insert_slots)
        a_inplace.extend(inplace_slots)
        cosine_values = [
            float(value)
            for key in ("insert_norm_residual_cosine", "inplace_norm_cosine")
            for value in amplification.get(key, [])
        ]
        if len(cosine_values) != 2 * SLOTS:
            raise PhaseBContractError("B-HIC0 requires 16 descriptive cosines per row")
        finite_complete = finite_complete and all(
            math.isfinite(value) for value in (*insert_slots, *inplace_slots, *cosine_values)
        )
        row_inplace = mean64(inplace_slots)
        row_amplification_ratio.append(float("inf") if row_inplace == 0.0 else mean64(insert_slots) / row_inplace)
        drift = row.get("drift")
        if not isinstance(drift, dict):
            raise PhaseBContractError("B-HIC0 drift evidence is absent")
        for kind, destination in (("hidden", hidden_drift_ratio), ("logit", logit_drift_ratio)):
            denominator = float(drift[f"{kind}_insert_eps_nrms"])
            numerator = float(drift[f"{kind}_inplace_eps_nrms"])
            finite_complete = finite_complete and math.isfinite(denominator) and math.isfinite(numerator)
            if kind == "hidden":
                hidden_insert_drift.append(denominator)
                hidden_inplace_drift.append(numerator)
            else:
                logit_insert_drift.append(denominator)
                logit_inplace_drift.append(numerator)
            if denominator == 0.0:
                zero_drift_denominators.append({"task_key": row["task_key"], "kind": kind})
            else:
                destination.append(numerator / denominator)
    p_insert_mean = mean64(p_insert)
    p_inplace_mean = mean64(p_inplace)
    removal = (p_insert_mean - p_inplace_mean) / p_insert_mean if p_insert_mean > 0.0 else float("-inf")
    summaries = {
        "P_insert": {
            "values": p_insert,
            "mean": p_insert_mean,
            "positive_rows_gt_1e_6": sum(v > STRICT_WIN_EPSILON for v in p_insert),
        },
        "P_inplace": {"values": p_inplace, "mean": p_inplace_mean, "worst": max(p_inplace)},
        "structural_INSERT_ZERO_minus_BASE_descriptive": {
            "values": structural_insert_zero,
            "mean": mean64(structural_insert_zero),
        },
        "epsilon_INSERT_EPS_minus_INSERT_ZERO_descriptive": {
            "values": insert_epsilon_increment,
            "mean": mean64(insert_epsilon_increment),
        },
        "penalty_removal_fraction": removal,
        "inplace_strict_wins": sum(inplace_wins),
        "A_insert_mean_96_slots": mean64(a_insert),
        "A_inplace_mean_96_slots": mean64(a_inplace),
        "row_amplification_ratio": row_amplification_ratio,
        "row_amplification_ratio_gt_5": sum(value > 5.0 for value in row_amplification_ratio),
        "hidden_drift_ratio": hidden_drift_ratio,
        "hidden_drift_ratio_mean": None if not hidden_drift_ratio else mean64(hidden_drift_ratio),
        "hidden_insert_drift_mean": mean64(hidden_insert_drift),
        "hidden_inplace_drift_mean": mean64(hidden_inplace_drift),
        "logit_drift_ratio": logit_drift_ratio,
        "logit_drift_ratio_mean": None if not logit_drift_ratio else mean64(logit_drift_ratio),
        "logit_insert_drift_mean": mean64(logit_insert_drift),
        "logit_inplace_drift_mean": mean64(logit_inplace_drift),
        "zero_drift_denominators": zero_drift_denominators,
        "inplace_margin_delta": {
            "values": margin_delta,
            "mean": mean64(margin_delta),
            "worst": min(margin_delta),
        },
    }
    gates = {
        "1_complete_finite_safe": finite_complete and safety_passed,
        "2_inplace_zero_bitwise_identity": identity_zero,
        "3_insert_penalty_replicates": p_insert_mean >= 0.05
        and sum(value > STRICT_WIN_EPSILON for value in p_insert) >= 9,
        "4_rmsnorm_amplification": mean64(a_insert) >= 5.0 * mean64(a_inplace)
        and sum(value > 5.0 for value in row_amplification_ratio) >= 10,
        "5_inplace_penalty_removed": p_inplace_mean <= 0.01
        and max(p_inplace) <= 0.05
        and removal >= 0.80
        and sum(inplace_wins) >= 10,
        "6_hidden_and_logit_drift_removed": not zero_drift_denominators
        and len(hidden_drift_ratio) == EXPECTED_ROWS
        and len(logit_drift_ratio) == EXPECTED_ROWS
        and mean64(hidden_inplace_drift) <= 0.20 * mean64(hidden_insert_drift)
        and mean64(logit_inplace_drift) <= 0.20 * mean64(logit_insert_drift),
        "7_inplace_margin_noninferior": mean64(margin_delta) >= -0.05 and min(margin_delta) >= -0.50,
        "8_backward_and_provenance": safety_passed,
    }
    nominated = all(gates.values())
    return {
        "nominated": nominated,
        "disposition": (
            "b_hic0_inplace_carrier_nominated" if nominated else "b_hic0_inplace_carrier_not_nominated"
        ),
        "gates": gates,
        "summaries": summaries,
        "safety": dict(safety),
    }
