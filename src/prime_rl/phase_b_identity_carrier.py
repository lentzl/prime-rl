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
EXPECTED_CACHE_CLASSES = (
    ("fla.models.utils.Cache", "lib/python3.12/site-packages/fla/models/utils.py", "3785d027727370b6eb8da96050109a249254c90caa12a3531a4034cc79f256a1", "flash-linear-attention==0.5.2"),
    ("fla.models.utils.FLACache", "lib/python3.12/site-packages/fla/models/utils.py", "3785d027727370b6eb8da96050109a249254c90caa12a3531a4034cc79f256a1", "flash-linear-attention==0.5.2"),
    ("fla.models.utils.LegacyFLACache", "lib/python3.12/site-packages/fla/models/utils.py", "3785d027727370b6eb8da96050109a249254c90caa12a3531a4034cc79f256a1", "flash-linear-attention==0.5.2"),
    ("transformers.cache_utils.Cache", "lib/python3.12/site-packages/transformers/cache_utils.py", "a51d2cd525f4458941a20cb5b3b8a8d989d8ca5bcd451855a27bb41743fed586", "transformers==5.6.2"),
    ("transformers.cache_utils.DynamicCache", "lib/python3.12/site-packages/transformers/cache_utils.py", "a51d2cd525f4458941a20cb5b3b8a8d989d8ca5bcd451855a27bb41743fed586", "transformers==5.6.2"),
    ("transformers.cache_utils.EncoderDecoderCache", "lib/python3.12/site-packages/transformers/cache_utils.py", "a51d2cd525f4458941a20cb5b3b8a8d989d8ca5bcd451855a27bb41743fed586", "transformers==5.6.2"),
    ("transformers.cache_utils.QuantizedCache", "lib/python3.12/site-packages/transformers/cache_utils.py", "a51d2cd525f4458941a20cb5b3b8a8d989d8ca5bcd451855a27bb41743fed586", "transformers==5.6.2"),
    ("transformers.cache_utils.StaticCache", "lib/python3.12/site-packages/transformers/cache_utils.py", "a51d2cd525f4458941a20cb5b3b8a8d989d8ca5bcd451855a27bb41743fed586", "transformers==5.6.2"),
)
EXPECTED_GATE_KEYS = tuple(f"{index}_{name}" for index, name in enumerate(
    (
        "complete_finite_safe",
        "inplace_zero_bitwise_identity",
        "insert_penalty_replicates",
        "rmsnorm_amplification",
        "inplace_penalty_removed",
        "hidden_and_logit_drift_removed",
        "inplace_margin_noninferior",
        "backward_and_provenance",
    ),
    start=1,
))


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


def ordered_subclass_closure(base: type) -> tuple[type, ...]:
    return tuple(sorted(recursive_subclass_closure(base), key=lambda cls: (cls.__module__, cls.__qualname__)))


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


def expected_memory_checkpoint_labels() -> list[str]:
    labels = ["after_model_load", "after_codec_construction"]
    for row_index in range(1, EXPECTED_ROWS + 1):
        labels.append(f"capture:r{row_index:02d}")
        labels.extend(f"receiver:r{row_index:02d}:{arm}" for arm in ARMS)
        if row_index == 3:
            labels.append("after_backward")
    labels.append("before_success")
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


def validate_suffix_target_ids(targets: Sequence[int], *, vocabulary_size: int) -> None:
    if not targets or vocabulary_size <= 0:
        raise PhaseBContractError("B-HIC0 suffix target domain is empty")
    if any(type(target) is not int or not 0 <= target < vocabulary_size for target in targets):
        raise PhaseBContractError("B-HIC0 suffix target ID is outside the vocabulary")


def validate_hic0_terminal_receipt(
    receipt: dict[str, Any],
    *,
    success_file: bool,
    plan: dict[str, Any],
    execution_commit: str,
) -> None:
    statuses = SUCCESS_STATUSES if success_file else FAILURE_STATUSES
    if receipt.get("status") not in statuses or receipt.get("terminal") != ("SUCCESS" if success_file else "FAILURE"):
        raise PhaseBContractError("B-HIC0 terminal status literal differs")
    if receipt.get("disposition") != receipt.get("status"):
        raise PhaseBContractError("B-HIC0 disposition and top-level status differ")
    if receipt.get("receipt_sha256") != canonical_json_sha256(receipt, omitted_fields=("receipt_sha256",)):
        raise PhaseBContractError("B-HIC0 internal receipt hash differs")
    if receipt.get("optimizer") is not None or receipt.get("optimizer_updates") != 0:
        raise PhaseBContractError("B-HIC0 receipt crossed the zero-update boundary")
    if receipt.get("saved_model_state") is not False or receipt.get("B1R_candidates_reused") is not False:
        raise PhaseBContractError("B-HIC0 receipt crossed the state/candidate boundary")
    if any(receipt.get(key) is not False for key in ("generation", "cache", "worker_loaded")):
        raise PhaseBContractError("B-HIC0 receipt crossed the no-generation/cache/worker boundary")
    if receipt.get("H176_loaded") is not False or receipt.get("strand_a_combined") is not False:
        raise PhaseBContractError("B-HIC0 receipt crossed the model/strand boundary")
    if (
        receipt.get("plan_sha256") != plan.get("_file_sha256")
        or receipt.get("execution_commit") != execution_commit
        or receipt.get("selection_sha256") != plan.get("diagnostic_bank", {}).get("selection_sha256")
    ):
        raise PhaseBContractError("B-HIC0 terminal plan, commit, or selection binding differs")
    expected_immutable = {"plan": plan.get("_file_sha256"), **plan.get("immutable_input_hashes", {})}
    if success_file:
        rows = receipt.get("rows")
        nomination = receipt.get("nomination")
        cache_guard = receipt.get("cache_guard")
        protection = receipt.get("protection")
        promotion = receipt.get("promotion")
        ledger = receipt.get("cuda_memory_ledger")
        if (
            receipt.get("schema_version") != "q35-2b-phase-b-hic0-identity-carrier-success/v1"
            or receipt.get("claim_class") != "zero_update_identity_carrier_causal_diagnostic_nomination_only"
            or receipt.get("model_loaded") is not True
            or receipt.get("source_forwards") != 12
            or receipt.get("receiver_forwards") != 60
            or receipt.get("backward_forwards_reused") != 1
            or not isinstance(rows, list)
            or len(rows) != EXPECTED_ROWS
            or len({row.get("task_key") for row in rows}) != EXPECTED_ROWS
            or Counter(row.get("action") for row in rows)
            != Counter({action: 4 for action in ACTIONS})
            or any(tuple(row.get("arms", {})) != ARMS for row in rows)
        ):
            raise PhaseBContractError("B-HIC0 SUCCESS row/count evidence differs")
        recomputed = evaluate_hic0(rows, safety=nomination.get("safety", {})) if isinstance(nomination, dict) else None
        if (
            not isinstance(nomination, dict)
            or nomination.get("disposition") != receipt["status"]
            or nomination.get("nominated") is not (receipt["status"] == SUCCESS_STATUSES[0])
            or not isinstance(nomination.get("gates"), dict)
            or tuple(nomination["gates"]) != EXPECTED_GATE_KEYS
            or any(type(value) is not bool for value in nomination["gates"].values())
            or (receipt["status"] == SUCCESS_STATUSES[0]) is not all(nomination["gates"].values())
            or recomputed != nomination
        ):
            raise PhaseBContractError("B-HIC0 SUCCESS nomination/status evidence differs")
        observed_cache_classes = cache_guard.get("classes", []) if isinstance(cache_guard, dict) else []
        cache_identities_match = len(observed_cache_classes) == len(EXPECTED_CACHE_CLASSES) and all(
            item.get("fqcn") == expected[0]
            and str(item.get("module_path", "")).endswith(expected[1])
            and item.get("module_sha256") == expected[2]
            and item.get("distribution") == expected[3]
            for item, expected in zip(observed_cache_classes, EXPECTED_CACHE_CLASSES, strict=True)
        )
        if (
            not isinstance(cache_guard, dict)
            or cache_guard.get("complete") is not True
            or cache_guard.get("label_count") != CACHE_LABEL_COUNT
            or cache_guard.get("canonical_label_sha256") != CACHE_LABEL_SHA256
            or cache_guard.get("closure_check_count") != CACHE_LABEL_COUNT
            or cache_guard.get("closure_checked_at_every_recorded_label") is not True
            or cache_guard.get("restored_in_finally") is not True
            or cache_guard.get("model_calls") != 72
            or cache_guard.get("dynamic_cache_trip_count") != 1
            or cache_guard.get("exit_recorded") is not True
            or cache_guard.get("recursively_closed_config_count", 0) < 1
            or not cache_identities_match
        ):
            raise PhaseBContractError("B-HIC0 SUCCESS cache evidence differs")
        backward = receipt.get("backward")
        if (
            not isinstance(backward, dict)
            or backward.get("row") != "document_adaptive_d2-v4-i35100"
            or backward.get("receiver_forward_reused") is not True
            or backward.get("extra_receiver_forwards") != 0
            or any(
                evidence.get("finite") is not True or evidence.get("nonzero") is not True
                for evidence in (backward.get("residual_gradient", {}), backward.get("encoder_group", {}), backward.get("receiver_group", {}))
            )
            or backward.get("all_named_gradients_finite") is not True
            or backward.get("e33_gradients_absent") is not True
        ):
            raise PhaseBContractError("B-HIC0 SUCCESS backward evidence differs")
        if (
            not isinstance(protection, dict)
            or protection.get("e33_tensor_pre") != protection.get("e33_tensor_post")
            or protection.get("e33_file_pre") != protection.get("e33_file_post")
            or protection.get("metadata_pre") != protection.get("metadata_post")
            or protection.get("codec_pre") != protection.get("codec_post")
            or protection.get("e33_file_pre") != plan.get("protected_model", {}).get("weight_sha256")
            or protection.get("metadata_pre") != plan.get("model_metadata_sha256")
            or receipt.get("immutable_input_hashes") != expected_immutable
        ):
            raise PhaseBContractError("B-HIC0 SUCCESS protection/provenance evidence differs")
        resources = plan.get("resources", {})
        preflight = receipt.get("preflight_resources", {})
        allocator = receipt.get("allocator", {})
        memory = receipt.get("cuda_memory", {})
        if (
            not isinstance(ledger, list)
            or [item.get("checkpoint") for item in ledger] != expected_memory_checkpoint_labels()
            or any(
                value > 32 * 1024**3
                for item in ledger
                for key, value in item.items()
                if key.endswith("_bytes")
            )
            or any(value > 32 * 1024**3 for key, value in memory.items() if key.endswith("_bytes"))
            or allocator.get("cap_bytes") != 32 * 1024**3
            or allocator.get("observed_fraction") != allocator.get("requested_fraction")
            or preflight.get("available_host_ram_bytes", 0) < resources.get("minimum_host_ram_bytes", 0)
            or preflight.get("free_disk_bytes", 0) < resources.get("minimum_free_disk_bytes", 0)
            or preflight.get("offline_environment") != {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            or not isinstance(promotion, dict)
            or promotion.get("admitted") is not False
            or promotion.get("diagnostic_rows_count_as_live_trajectories") is not False
            or promotion.get("complete_live_trajectory_count") != 0
            or promotion.get("minimum_complete_live_trajectories_unchanged") != 4
        ):
            raise PhaseBContractError("B-HIC0 SUCCESS resource/promotion evidence differs")
    else:
        expected_failure_class = {
            "b_hic0_nocache_rejected": "scientific_cache_rejection",
            "b_hic0_incomplete": "contract_or_evidence_incomplete",
            "infrastructure_invalid": "infrastructure",
        }[receipt["status"]]
        audit = receipt.get("post_failure_hash_audit")
        if (
            receipt.get("schema_version") != "q35-2b-phase-b-hic0-identity-carrier-failure/v1"
            or receipt.get("failure_class") != expected_failure_class
            or not isinstance(audit, dict)
            or audit.get("audit_complete") is not True
            or audit.get("immutable_input_hashes") != expected_immutable
            or audit.get("immutable_input_hashes_match") is not True
        ):
            raise PhaseBContractError("B-HIC0 FAILURE class or postflight audit differs")
        if type(receipt.get("model_loaded")) is not bool:
            raise PhaseBContractError("B-HIC0 FAILURE model-load evidence differs")
        if receipt["model_loaded"] is True:
            tensor_post = audit.get("e33_tensor_post")
            if (
                not isinstance(tensor_post, str)
                or len(tensor_post) != 64
                or audit.get("e33_disk_and_metadata_exact") is not True
                or (
                    audit.get("e33_tensor_reference_available") is True
                    and audit.get("e33_tensor_preserved") is not True
                )
            ):
                raise PhaseBContractError("B-HIC0 FAILURE lacks protected e33 evidence")


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
        if row_inplace == 0.0:
            raise PhaseBContractError("B-HIC0 row RMSNorm amplification denominator is zero")
        row_amplification_ratio.append(mean64(insert_slots) / row_inplace)
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
    if p_insert_mean == 0.0:
        raise PhaseBContractError("B-HIC0 insertion penalty does not define finite removal")
    removal = (p_insert_mean - p_inplace_mean) / p_insert_mean
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
