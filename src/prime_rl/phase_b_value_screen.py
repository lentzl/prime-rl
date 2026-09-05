from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from prime_rl.phase_b_contract import PhaseBContractError, canonical_json_sha256

ACTIONS = ("solve_owned", "delegate_terminal", "delegate_coordinator")
TRAINING_ARMS = ("STATIC", "FFN", "RECURRENT")
EVALUATION_DEPTHS = (1, 2, 4, 8)
EXPECTED_TRAINING_KEYS_SHA256 = "cd24743d41543e07485a6c9c690e3622847807f6ec813374f99cc5115236fb9a"
EXPECTED_TRAINING_BATCHES_SHA256 = "4bc83af062e1ca3f152daf32199ea6c1708c1e7f5dc54cad281ad67322170afb"
EXPECTED_HELDOUT_KEYS_SHA256 = "d50a1ef713303c38db16f72a5c8bbde60d13e6445dd896091b0a07a4f41155b4"
EXPECTED_HELDOUT_KEY_ACTION_SHA256 = "180b8067ecabcf8451aba5f0a454f3f6d28b6a8b35978de88f4505524f7d43e9"
INITIALIZATION_SEED_PAYLOAD = "q35-2b-b1-teacher-forced-value-screen-v1:init"
INITIALIZATION_DERIVATION_SHA256 = "0c597a15ad1983c226a6426f9d9c7dde1017190915705483672427c6a87e93ed"
INITIALIZATION_SEED = 207190549
STRICT_WIN_EPSILON = 1e-6
PENDING_PREFIX = "ASSIGNED_BY_STRAND_E_OR_ROOT"


@dataclass(frozen=True)
class TrainingBatch:
    update_index: int
    task_keys: tuple[str, ...]


def canonical_plan_sha256(plan: dict[str, Any]) -> str:
    """Hash a B1 plan while omitting only its self-referential hash field."""

    if "plan_sha256" not in plan:
        raise PhaseBContractError("B1 plan lacks its internal canonical hash field")
    payload = dict(plan)
    payload.pop("plan_sha256")
    return canonical_json_sha256(payload)


def validate_value_screen_plan(plan: dict[str, Any], *, require_authorized: bool = True) -> None:
    """Validate the prospective B1 matched-learning screen."""

    if plan.get("schema_version") != "q35-2b-phase-b-teacher-forced-value-screen/v1-repair1":
        raise PhaseBContractError("B1 plan schema differs from the implemented value screen")
    implementation_commit = plan.get("implementation_commit")
    if (
        not isinstance(implementation_commit, str)
        or len(implementation_commit) != 40
        or any(character not in "0123456789abcdef" for character in implementation_commit)
        or plan.get("mechanism_code_commit") != implementation_commit
    ):
        raise PhaseBContractError("B1 plan does not bind one exact mechanism code commit")
    if plan.get("plan_sha256") != canonical_plan_sha256(plan):
        raise PhaseBContractError("B1 internal canonical plan hash differs")
    values = {key: plan.get(key) for key in ("training", "evaluation", "boundaries", "heldout")}
    if not all(isinstance(value, dict) for value in values.values()):
        raise PhaseBContractError("B1 plan lacks training, evaluation, heldout, or boundary mappings")
    training = values["training"]
    evaluation = values["evaluation"]
    boundaries = values["boundaries"]
    heldout = values["heldout"]
    assert isinstance(training, dict) and isinstance(evaluation, dict)
    assert isinstance(boundaries, dict) and isinstance(heldout, dict)
    if training.get("arm_order") != list(TRAINING_ARMS):
        raise PhaseBContractError("B1 arms must train sequentially in STATIC/FFN/RECURRENT order")
    if (
        training.get("optimizer_updates_per_arm") != 4
        or training.get("rows_per_update") != 12
        or training.get("rows_per_action_per_update") != 4
    ):
        raise PhaseBContractError("B1 requires four balanced 4/4/4 twelve-row updates per arm")
    if training.get("unique_row_exposures_per_arm") != 48 or training.get("early_stop") is not False:
        raise PhaseBContractError("B1 must expose all 48 unique rows once per arm without early stop")
    if not isinstance(training.get("restart_rule"), str) or "do not load or reuse" not in training["restart_rule"]:
        raise PhaseBContractError("B1R must restart every arm without partial-state reuse")
    if training.get("recurrent_depth") != 4 or training.get("bptt_window") != 4:
        raise PhaseBContractError("B1 recurrence must use exact T=4 full BPTT")
    optimizer = training.get("optimizer")
    if not isinstance(optimizer, dict) or optimizer != {
        "name": "AdamW",
        "learning_rate": 0.0001,
        "betas": [0.9, 0.95],
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
    }:
        raise PhaseBContractError("B1 optimizer differs from the frozen matched-learning contract")
    if (
        training.get("initialization_seed_payload") != INITIALIZATION_SEED_PAYLOAD
        or training.get("initialization_derivation_sha256") != INITIALIZATION_DERIVATION_SHA256
        or training.get("initialization_seed") != INITIALIZATION_SEED
    ):
        raise PhaseBContractError("B1 initialization seed derivation differs")
    if evaluation.get("recurrent_depths") != list(EVALUATION_DEPTHS):
        raise PhaseBContractError("B1 evaluation depth grid must remain 1/2/4/8")
    if evaluation.get("primary_contrast") != "RECURRENT_T4_minus_FFN":
        raise PhaseBContractError("B1 primary contrast must remain RECURRENT-T4 minus FFN")
    if evaluation.get("strict_win_epsilon") != STRICT_WIN_EPSILON:
        raise PhaseBContractError("B1 strict-win epsilon differs")
    if heldout.get("generator_seed") != 20261113 or heldout.get("template_variants") != [4, 5]:
        raise PhaseBContractError("B1 heldout generation must use seed 20261113 and variants 4/5")
    if heldout.get("ordered_task_key_sha256") != EXPECTED_HELDOUT_KEYS_SHA256:
        raise PhaseBContractError("B1 heldout ordered-key hash differs")
    if heldout.get("ordered_key_action_sha256") != EXPECTED_HELDOUT_KEY_ACTION_SHA256:
        raise PhaseBContractError("B1 heldout key/action hash differs")
    if heldout.get("taskset_commit") != "5283a85a01b5e8a065b3d2db17f9efa6aa0f3b2f":
        raise PhaseBContractError("B1 heldout taskset commit differs")
    training_source = plan.get("training_source")
    if not isinstance(training_source, dict) or (
        training_source.get("ordered_task_key_sha256"),
        training_source.get("nested_batch_sha256"),
    ) != (EXPECTED_TRAINING_KEYS_SHA256, EXPECTED_TRAINING_BATCHES_SHA256):
        raise PhaseBContractError("B1 training source order hashes differ")
    if any(boundaries.get(key) is not False for key in ("generation", "cache", "load_H176", "combine_with_A")):
        raise PhaseBContractError("B1 forbids generation, cache, H176, and Strand A")
    if boundaries.get("e33_trainable") is not False or boundaries.get("save_e33_checkpoint") is not False:
        raise PhaseBContractError("B1 keeps e33 immutable and never checkpoints it")
    if boundaries.get("nomination_only") is not True:
        raise PhaseBContractError("B1 is nomination-only")
    if boundaries.get("live_promotion_minimum_complete_trajectories") != 4:
        raise PhaseBContractError("B1 must preserve the four-complete-live-trajectory promotion floor")
    if boundaries.get("teacher_forced_rows_are_live_trajectories") is not False:
        raise PhaseBContractError("teacher-forced B1 rows must not be called live trajectories")
    if boundaries.get("teacher_forced_live_trajectory_count") != 0:
        raise PhaseBContractError("B1 must record zero live trajectories")
    resources = plan.get("resources")
    outputs = plan.get("outputs")
    if not isinstance(resources, dict) or not isinstance(outputs, dict):
        raise PhaseBContractError("B1 resource or output contract is absent")
    if (
        resources.get("outer_wall_clock_seconds") != 14_400
        or resources.get("compute_limit_seconds") != 14_040
        or resources.get("failure_audit_limit_seconds") != 300
        or resources.get("terminal_publication_headroom_seconds") != 60
        or resources.get("minimum_free_disk_bytes") != 60 * 1024**3
        or resources.get("artifact_cap_bytes") != 512 * 1024**2
        or resources.get("cuda_memory_cap_bytes") != 32 * 1024**3
        or resources.get("memory_checkpoint_count") != 404
        or outputs.get("artifact_cap_bytes") != 512 * 1024**2
    ):
        raise PhaseBContractError("B1 resource caps differ")
    invalid_dependency = plan.get("b1_invalid_dependency")
    if not isinstance(invalid_dependency, dict) or (
        invalid_dependency.get("receipt_file_sha256"),
        invalid_dependency.get("prior_freeze_commit"),
        invalid_dependency.get("prior_mechanism_commit"),
        invalid_dependency.get("prior_plan_file_sha256"),
        invalid_dependency.get("prior_plan_internal_sha256"),
    ) != (
        "27cb02d3907acd0d203a56e70e4de8f1f82a8672a7ed1264d4abb2cc39033025",
        "d438b07c2549794c4a533b369c9ef6f5a8dc28b9",
        "d2130f9407151ab93985c761bc63f16da8beb9d2",
        "6013e4ade15fba3a85c7e202008ef8222504796203348b8cee2d3b1e366dc00d",
        "1dfd449499da846fc49844e310a94f8d1c27e44b3343fda94ca6a6b6026fce67",
    ):
        raise PhaseBContractError("B1R invalid-run provenance binding differs")
    material_violation = invalid_dependency.get("material_violation")
    if not isinstance(material_violation, dict) or (
        material_violation.get("cap_bytes"),
        material_violation.get("maximum_allocated_bytes"),
        material_violation.get("maximum_reserved_bytes"),
        material_violation.get("candidate_files"),
        material_violation.get("audit_complete"),
    ) != (32 * 1024**3, 33_097_241_600, 34_978_398_208, 0, True):
        raise PhaseBContractError("B1R material memory violation evidence differs")
    if require_authorized:
        pending_paths = _pending_paths(plan)
        if pending_paths:
            raise PhaseBContractError(f"B1 evaluator-owned fields remain pending: {pending_paths}")
        if plan.get("status") != "frozen_pending_independent_review":
            raise PhaseBContractError("B1 plan is not the prospectively frozen review object")
        if plan.get("execution_authorization") != "independent_gatekeeper_review_then_root_schedule_required":
            raise PhaseBContractError("B1 plan lacks the exact root scheduling boundary")


def validate_training_batches(rows: list[dict[str, Any]], selection: dict[str, Any]) -> tuple[TrainingBatch, ...]:
    """Require the exact four manifest-order balanced batches."""

    by_key: dict[str, dict[str, Any]] = {}
    ordered_source_keys: list[str] = []
    for row in rows:
        task_key = row.get("task_key")
        if not isinstance(task_key, str) or not task_key:
            raise PhaseBContractError("B1 source row lacks a task key")
        if task_key in by_key:
            raise PhaseBContractError(f"duplicate B1 source task key: {task_key}")
        by_key[task_key] = row
        ordered_source_keys.append(task_key)
    if len(by_key) != 48:
        raise PhaseBContractError("B1 training source must contain exactly 48 unique rows")
    if canonical_json_sha256(ordered_source_keys) != EXPECTED_TRAINING_KEYS_SHA256:
        raise PhaseBContractError("B1 source rows differ from the frozen manifest order")
    batches = selection.get("batches")
    if not isinstance(batches, list) or len(batches) != 4:
        raise PhaseBContractError("B1 selection must define exactly four batches")
    if canonical_json_sha256(batches) != EXPECTED_TRAINING_BATCHES_SHA256:
        raise PhaseBContractError("B1 update batches differ from the frozen manifest order")
    result: list[TrainingBatch] = []
    flattened: list[str] = []
    for update_index, raw_batch in enumerate(batches, start=1):
        if not isinstance(raw_batch, list) or len(raw_batch) != 12:
            raise PhaseBContractError(f"B1 update {update_index} does not contain twelve rows")
        if any(not isinstance(key, str) or key not in by_key for key in raw_batch):
            raise PhaseBContractError(f"B1 update {update_index} names an absent source row")
        if len(set(raw_batch)) != 12:
            raise PhaseBContractError(f"B1 update {update_index} repeats a source row")
        counts = Counter(by_key[key].get("action") for key in raw_batch)
        if counts != Counter({action: 4 for action in ACTIONS}):
            raise PhaseBContractError(f"B1 update {update_index} is not action-balanced 4/4/4")
        flattened.extend(raw_batch)
        result.append(TrainingBatch(update_index=update_index, task_keys=tuple(raw_batch)))
    if flattened != ordered_source_keys:
        raise PhaseBContractError("B1 update order must equal the exact source manifest order")
    return tuple(result)


def validate_evaluation_keys(
    train_keys: set[str], evaluation_rows: list[dict[str, Any]], selection: dict[str, Any]
) -> tuple[str, ...]:
    keys = selection.get("task_keys")
    if not isinstance(keys, list) or len(keys) != 12 or any(not isinstance(key, str) for key in keys):
        raise PhaseBContractError("B1 evaluation selection must contain twelve task keys")
    if len(keys) != len(set(keys)):
        raise PhaseBContractError("B1 evaluation selection repeats a task key")
    if canonical_json_sha256(keys) != EXPECTED_HELDOUT_KEYS_SHA256:
        raise PhaseBContractError("B1 heldout order differs from the prospectively frozen keys")
    by_key = {row.get("task_key"): row for row in evaluation_rows}
    if len(by_key) != len(evaluation_rows) or any(key not in by_key for key in keys):
        raise PhaseBContractError("B1 evaluation keys are not uniquely present in the heldout bank")
    if train_keys.intersection(keys):
        raise PhaseBContractError("B1 heldout task keys overlap training")
    key_actions = [{"task_key": key, "expected_action": by_key[key].get("action")} for key in keys]
    if canonical_json_sha256(key_actions) != EXPECTED_HELDOUT_KEY_ACTION_SHA256:
        raise PhaseBContractError("B1 heldout key/action pairing differs")
    if Counter(item["expected_action"] for item in key_actions) != Counter({action: 4 for action in ACTIONS}):
        raise PhaseBContractError("B1 heldout action counts must be exactly 4/4/4")
    return tuple(keys)


def build_action_trie(candidates: dict[str, Sequence[int]], *, correct_action: str) -> dict[str, Any]:
    """Freeze correct-leaf branch positions for the three-action token trie."""

    if tuple(candidates) != ACTIONS or correct_action not in ACTIONS:
        raise PhaseBContractError("B1 action trie must use the frozen action order")
    sequences: dict[str, list[int]] = {}
    for action, raw in candidates.items():
        values = list(raw)
        if not values or any(type(token) is not int or token < 0 for token in values):
            raise PhaseBContractError(f"B1 action trie has invalid tokens for {action}")
        sequences[action] = values
    if len({tuple(value) for value in sequences.values()}) != len(ACTIONS):
        raise PhaseBContractError("B1 action trie leaves are not unique")
    for left, left_values in sequences.items():
        for right, right_values in sequences.items():
            if (
                left != right
                and len(left_values) <= len(right_values)
                and right_values[: len(left_values)] == left_values
            ):
                raise PhaseBContractError(f"B1 action trie leaf {left} prefixes {right}")
    correct = sequences[correct_action]
    branches: list[dict[str, Any]] = []
    for offset, correct_token in enumerate(correct):
        live = [action for action in ACTIONS if sequences[action][:offset] == correct[:offset]]
        outgoing = sorted({sequences[action][offset] for action in live if offset < len(sequences[action])})
        if len(outgoing) > 1:
            branches.append(
                {
                    "target_offset": offset,
                    "logit_offset": offset,
                    "correct_token_id": correct_token,
                    "other_token_ids": [token for token in outgoing if token != correct_token],
                    "live_actions": live,
                }
            )
    if not branches:
        raise PhaseBContractError("B1 action trie has no branch along the correct leaf")
    trie_payload = {"action_order": list(ACTIONS), "candidate_suffix_token_ids": sequences}
    return {
        **trie_payload,
        "correct_action": correct_action,
        "branches": branches,
        "branch_count": len(branches),
        "canonical_trie_sha256": canonical_json_sha256(trie_payload),
    }


def action_margin_from_logits(
    trie: dict[str, Any], logit: Callable[[int, int], float]
) -> tuple[float, list[dict[str, Any]]]:
    branch_metrics: list[dict[str, Any]] = []
    for branch in trie["branches"]:
        offset = branch["logit_offset"]
        correct = float(logit(offset, branch["correct_token_id"]))
        alternatives = [float(logit(offset, token)) for token in branch["other_token_ids"]]
        if not alternatives or not all(math.isfinite(value) for value in (correct, *alternatives)):
            raise PhaseBContractError("B1 action trie margin is non-finite or has no alternative")
        margin = correct - max(alternatives)
        branch_metrics.append(
            {**branch, "correct_logit": correct, "max_other_logit": max(alternatives), "margin": margin}
        )
    return min(item["margin"] for item in branch_metrics), branch_metrics


def midpoint_median(values: Sequence[float]) -> float:
    ordered = sorted(_finite_values(values))
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def summarize(values: Sequence[float], actions: Sequence[str]) -> dict[str, Any]:
    finite = _finite_values(values)
    if len(finite) != 12 or len(actions) != 12:
        raise PhaseBContractError("B1 aggregation requires twelve ordered paired rows")
    if any(action not in ACTIONS for action in actions):
        raise PhaseBContractError("B1 aggregation contains an unknown action")
    if Counter(actions) != Counter({action: 4 for action in ACTIONS}):
        raise PhaseBContractError("B1 aggregation requires exact heldout 4/4/4 action balance")
    return {
        "values": finite,
        "mean": math.fsum(finite) / len(finite),
        "median": midpoint_median(finite),
        "minimum": min(finite),
        "maximum": max(finite),
        "strict_wins": sum(value > STRICT_WIN_EPSILON for value in finite),
        "per_action_means": {
            action: math.fsum(value for value, observed in zip(finite, actions, strict=True) if observed == action)
            / sum(observed == action for observed in actions)
            for action in ACTIONS
        },
    }


def evaluate_nomination(
    step0: dict[str, list[dict[str, Any]]],
    final: dict[str, list[dict[str, Any]]],
    *,
    safety_gate_passed: bool,
) -> dict[str, Any]:
    """Apply the prospectively frozen B1 gates without exclusions or selection."""

    required = ("STATIC", "FFN", *(f"RECURRENT_T{depth}" for depth in EVALUATION_DEPTHS))
    for metrics in (step0, final):
        if any(name not in metrics or len(metrics[name]) != 12 for name in required):
            raise PhaseBContractError("B1 nomination metrics lack an exact twelve-row arm")
    ordered_keys = [row["task_key"] for row in final["RECURRENT_T4"]]
    actions = [row["action"] for row in final["RECURRENT_T4"]]
    for metrics in (step0, final):
        for name in required:
            if [row["task_key"] for row in metrics[name]] != ordered_keys:
                raise PhaseBContractError("B1 nomination metric order differs across arms")
            if [row["action"] for row in metrics[name]] != actions:
                raise PhaseBContractError("B1 nomination action pairing differs across arms")

    def metric_values(name: str, field: str, phase: dict[str, list[dict[str, Any]]]) -> list[float]:
        return _finite_values([row[field] for row in phase[name]])

    rec_nll = _subtract(metric_values("RECURRENT_T4", "nll", step0), metric_values("RECURRENT_T4", "nll", final))
    ffn_nll = _subtract(metric_values("FFN", "nll", step0), metric_values("FFN", "nll", final))
    rec_margin = _subtract(
        metric_values("RECURRENT_T4", "margin", final), metric_values("RECURRENT_T4", "margin", step0)
    )
    ffn_margin = _subtract(metric_values("FFN", "margin", final), metric_values("FFN", "margin", step0))
    summaries = {
        "L_REC4": summarize(rec_nll, actions),
        "M_REC4": summarize(rec_margin, actions),
        "A_N": summarize(_subtract(rec_nll, ffn_nll), actions),
        "A_M": summarize(_subtract(rec_margin, ffn_margin), actions),
        "R4_minus_R1_nll_improvement": summarize(
            _subtract(metric_values("RECURRENT_T1", "nll", final), metric_values("RECURRENT_T4", "nll", final)),
            actions,
        ),
        "R4_minus_R1_margin": summarize(
            _subtract(metric_values("RECURRENT_T4", "margin", final), metric_values("RECURRENT_T1", "margin", final)),
            actions,
        ),
        "R8_minus_R4_nll": summarize(
            _subtract(metric_values("RECURRENT_T8", "nll", final), metric_values("RECURRENT_T4", "nll", final)),
            actions,
        ),
        "R8_minus_R4_margin": summarize(
            _subtract(metric_values("RECURRENT_T8", "margin", final), metric_values("RECURRENT_T4", "margin", final)),
            actions,
        ),
    }
    s = summaries
    gate2 = (
        s["L_REC4"]["mean"] >= 0.001
        and s["L_REC4"]["median"] >= 0.0
        and s["L_REC4"]["strict_wins"] >= 7
        and min(s["L_REC4"]["per_action_means"].values()) >= -0.005
        and s["L_REC4"]["minimum"] >= -0.05
        and s["M_REC4"]["mean"] >= 0.0
        and s["M_REC4"]["minimum"] >= -0.50
    )
    gate3 = (
        s["A_N"]["mean"] >= -0.0025
        and s["A_N"]["median"] >= -0.005
        and min(s["A_N"]["per_action_means"].values()) >= -0.010
        and s["A_N"]["minimum"] >= -0.050
        and s["A_M"]["mean"] >= -0.050
        and min(s["A_M"]["per_action_means"].values()) >= -0.10
        and s["A_M"]["minimum"] >= -0.50
    )
    nll_route = (
        s["A_N"]["mean"] >= 0.001
        and s["A_N"]["median"] >= 0.0
        and s["A_N"]["strict_wins"] >= 7
        and sum(value >= 0.0 for value in s["A_N"]["per_action_means"].values()) >= 2
    )
    action_route = (
        s["A_M"]["mean"] >= 0.020
        and s["A_M"]["median"] >= 0.0
        and s["A_M"]["strict_wins"] >= 7
        and sum(value >= 0.0 for value in s["A_M"]["per_action_means"].values()) >= 2
    )
    gate5 = (
        s["R4_minus_R1_nll_improvement"]["mean"] >= -0.0025
        and s["R4_minus_R1_nll_improvement"]["minimum"] >= -0.05
        and s["R4_minus_R1_margin"]["mean"] >= -0.05
        and s["R4_minus_R1_margin"]["minimum"] >= -0.50
        and (s["R4_minus_R1_nll_improvement"]["mean"] >= 0.0005 or s["R4_minus_R1_margin"]["mean"] >= 0.010)
    )
    recurrent_rows = final["RECURRENT_T8"]
    gate6 = all(
        metric["retention"][f"T{depth}"]["cosine"] >= 0.995
        and 0.95 <= metric["retention"][f"T{depth}"]["norm_ratio"] <= 1.05
        and metric["retention"][f"T{depth}"]["relative_l2"] <= 0.10
        for metric in recurrent_rows
        for depth in EVALUATION_DEPTHS
    )
    gate7 = (
        all(
            min(metric["stability_T8"]["memory_change_rms"]) > 1e-6
            and metric["stability_T8"]["median_memory_contraction_steps_2_8"] <= 0.90
            and metric["stability_T8"]["max_memory_contraction_steps_2_8"] <= 1.25
            and metric["stability_T8"]["memory_oscillation_rate"] <= 0.25
            and metric["stability_T8"]["finite"] is True
            for metric in recurrent_rows
        )
        and math.fsum(metric["stability_T8"]["memory_oscillation_rate"] for metric in recurrent_rows) / 12 <= 0.10
    )
    gate8 = (
        s["R8_minus_R4_nll"]["mean"] <= 0.005
        and s["R8_minus_R4_nll"]["maximum"] <= 0.05
        and s["R8_minus_R4_margin"]["mean"] >= -0.10
        and s["R8_minus_R4_margin"]["minimum"] >= -0.50
    )
    gates = {
        "1_safety_and_exact_execution": bool(safety_gate_passed),
        "2_recurrent_own_learning": gate2,
        "3_matched_noninferiority": gate3,
        "4_positive_recurrent_over_ffn": nll_route or action_route,
        "4_nll_route": nll_route,
        "4_action_route": action_route,
        "5_depth_T4_over_T1": gate5,
        "6_learned_retention": gate6,
        "7_recurrent_stability": gate7,
        "8_T8_nonregression": gate8,
    }
    required_gates = (
        "1_safety_and_exact_execution",
        "2_recurrent_own_learning",
        "3_matched_noninferiority",
        "4_positive_recurrent_over_ffn",
        "5_depth_T4_over_T1",
        "6_learned_retention",
        "7_recurrent_stability",
        "8_T8_nonregression",
    )
    nominated = all(gates[key] for key in required_gates)
    return {
        "nominated": nominated,
        "disposition": "b1_nominated" if nominated else "b1_not_nominated",
        "gates": gates,
        "summaries": summaries,
    }


def paired_loss_deltas(metrics: dict[str, list[dict[str, Any]]], left: str, right: str) -> list[dict[str, float | str]]:
    if left not in metrics or right not in metrics:
        raise PhaseBContractError("B1 paired contrast names an absent metric arm")
    left_by_key = {row.get("task_key"): row.get("nll", row.get("loss")) for row in metrics[left]}
    right_by_key = {row.get("task_key"): row.get("nll", row.get("loss")) for row in metrics[right]}
    if len(left_by_key) != len(metrics[left]) or len(right_by_key) != len(metrics[right]):
        raise PhaseBContractError("B1 paired contrast contains duplicate task keys")
    if not left_by_key or left_by_key.keys() != right_by_key.keys():
        raise PhaseBContractError("B1 paired contrast task keys differ")
    result: list[dict[str, float | str]] = []
    for task_key in left_by_key:
        delta = float(left_by_key[task_key]) - float(right_by_key[task_key])
        if not math.isfinite(delta):
            raise PhaseBContractError("B1 paired contrast contains a nonfinite loss")
        result.append({"task_key": str(task_key), "left_minus_right": delta})
    return result


def _subtract(left: Sequence[float], right: Sequence[float]) -> list[float]:
    if len(left) != len(right):
        raise PhaseBContractError("B1 paired arrays have different lengths")
    return [a - b for a, b in zip(left, right, strict=True)]


def _finite_values(values: Sequence[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise PhaseBContractError("B1 aggregation contains no values or a nonfinite value")
    return result


def _pending_paths(value: Any, path: str = "$") -> list[str]:
    result: list[str] = []
    if isinstance(value, str) and value.startswith(PENDING_PREFIX):
        result.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            result.extend(_pending_paths(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_pending_paths(child, f"{path}[{index}]"))
    return result
