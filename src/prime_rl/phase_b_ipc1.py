from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from prime_rl.phase_b_contract import PhaseBContractError
from prime_rl.phase_b_value_screen import EVALUATION_DEPTHS, midpoint_median

ACTIONS = ("solve_owned", "delegate_terminal", "delegate_coordinator")
TRAINING_ARMS = ("STATIC", "FFN", "RECURRENT")
EVALUATION_ARMS = ("STATIC", "FFN", "RECURRENT_T1", "RECURRENT_T2", "RECURRENT_T4", "RECURRENT_T8")
SLOTS = 8
STRICT_WIN_EPSILON = 1e-6
TRAIN_ROWS = 48
EVALUATION_ROWS = 24
INITIALIZATION_SEED_PAYLOAD = "q35-2b-b-ipc1-matched-learning-v1:init"
INITIALIZATION_DERIVATION_SHA256 = "0bd65b6f8536fba9fc0914cd2252ca91d5632e708b66dce6ba6d1113d2ded26a"
INITIALIZATION_SEED = 198597487
SELECTIONS = {
    "train": {
        "seed_payload": "q35-2b-b-ipc1-matched-learning-v1:train",
        "derivation_sha256": "03ffe55cee2227446460672212ef05b1a01d4c04f032eaef75b9ede361ae3ec7",
        "seed": 67_102_044,
        "instance_start": 36_100,
        "instance_stop": 36_116,
        "rows_per_action": 16,
    },
    "validation": {
        "seed_payload": "q35-2b-b-ipc1-matched-learning-v1:validation",
        "derivation_sha256": "0f2a3f07f5def290f8e501a89463f3f2d908658fb40b68f4129f1a28fc054b51",
        "seed": 254_426_887,
        "instance_start": 36_200,
        "instance_stop": 36_208,
        "rows_per_action": 8,
    },
    "heldout": {
        "seed_payload": "q35-2b-b-ipc1-matched-learning-v1:heldout",
        "derivation_sha256": "e48061b02620818c4b21fe55ea2f92e26ccd7ece68ca0c94fa8cb6a66867b040",
        "seed": 3_833_618_864,
        "instance_start": 36_300,
        "instance_stop": 36_308,
        "rows_per_action": 8,
    },
}
SUCCESS_STATUSES = (
    "b_ipc1_inplace_learning_recurrent_nominated",
    "b_ipc1_inplace_learning_nominated",
    "b_ipc1_inplace_learning_not_nominated",
    "b_ipc1_validation_not_nominated",
)
FAILURE_STATUS_CLASSES = (
    ("b_ipc1_mechanism_rejected", "scientific_mechanism_rejection"),
    ("b_ipc1_nocache_rejected", "scientific_cache_rejection"),
    ("b_ipc1_incomplete", "contract_or_evidence_incomplete"),
    ("infrastructure_invalid", "infrastructure"),
)
CUDA_MEMORY_CAP_BYTES = 32 * 1024**3
MINIMUM_HOST_RAM_BYTES = 64 * 1024**3
MINIMUM_FREE_DISK_BYTES = 60 * 1024**3
ARTIFACT_CAP_BYTES = 512 * 1024**2
EXPECTED_SUCCESS_TAMPERS = (
    "top_extra",
    "top_missing",
    "wrong_update_index",
    "wrong_action",
    "wrong_exposure",
    "output_scale_wrong_length",
    "output_scale_nonfinite",
    "output_scale_all_zero_step1",
    "objective_extra",
    "preprobe_backward",
    "render_extra",
    "aggregate_tamper",
    "cache_count",
    "memory_order",
    "immutable_hash",
    "candidate_filename",
    "candidate_state",
    "nomination",
)
EXPECTED_LATE_FAILURE_TAMPERS = (
    "post-model-pre-candidate:full_freeze",
    "post-model-pre-candidate:module_initial",
    "post-model-pre-candidate:cache_prefix",
    "post-model-pre-candidate:memory_prefix",
    "post-model-pre-candidate:progress",
    "post-model-pre-candidate:cache_truncated_against_progress",
    "post-model-pre-candidate:memory_truncated_against_progress",
    "post-model-pre-candidate:backward_progress",
    "post-model-pre-candidate:optimizer_progress",
    "late-post-candidate:full_freeze",
    "late-post-candidate:module_initial",
    "late-post-candidate:cache_prefix",
    "late-post-candidate:memory_prefix",
    "late-post-candidate:progress",
    "late-post-candidate:cache_truncated_against_progress",
    "late-post-candidate:memory_truncated_against_progress",
    "late-post-candidate:backward_progress",
    "late-post-candidate:optimizer_progress",
    "late-post-candidate:candidate_inventory",
    "late-post-candidate:candidate_safe_load",
    "late-post-candidate:module_current",
)


def require_exact_mapping_keys(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise PhaseBContractError(f"B-IPC1 {label} keyset differs: {observed}")
    return value


def canonical_plan_sha256(plan: dict[str, Any]) -> str:
    if "plan_sha256" not in plan:
        raise PhaseBContractError("B-IPC1 plan lacks its self hash")
    payload = dict(plan)
    payload.pop("plan_sha256")
    return canonical_bank_sha256(payload)


def validate_ipc1_plan(plan: dict[str, Any], *, require_authorized: bool = True) -> None:
    require_exact_mapping_keys(
        plan,
        {
            "schema_version",
            "status",
            "run_identity",
            "supersedes_run_identity",
            "run2_failure_dependency",
            "claim_class",
            "mechanism_code_commit",
            "implementation_commit",
            "plan_sha256",
            "execution_authorization",
            "terminal_proof",
            "antecedents",
            "protected_model",
            "model_metadata_sha256",
            "bank_manifest",
            "overlap_closure",
            "runtime_source",
            "generator_source",
            "taskset_source",
            "banks",
            "training",
            "receiver",
            "pre_update_mechanism_gate",
            "evaluation_firewall",
            "nomination_gates",
            "schedules",
            "terminal_statuses",
            "boundaries",
            "resources",
            "software",
            "execution_environment",
            "outputs",
        },
        label="plan",
    )
    if plan.get("schema_version") != "q35-2b-phase-b-ipc1-matched-learning/v1":
        raise PhaseBContractError("B-IPC1 plan schema differs")
    dependency = require_exact_mapping_keys(
        plan.get("run2_failure_dependency"),
        {
            "binding_path",
            "binding_sha256",
            "failure_path",
            "failure_file_sha256",
            "failure_internal_receipt_sha256",
            "run_log_path",
            "run_log_sha256",
            "archive_path",
            "archive_sha256",
            "manifest_path",
            "manifest_sha256",
            "failed_execution_commit",
            "failed_plan_file_sha256",
            "model_calls_completed",
            "backward_calls_completed",
            "optimizer_steps_completed",
            "static_rows_exposed",
            "ffn_rows_exposed",
            "partial_state_reusable",
            "validation_started",
            "heldout_opened",
            "scientific_result",
            "candidates_created",
        },
        label="run2 failure dependency",
    )
    if (
        plan.get("run_identity") != "b-ipc1-r1-matched-learning-run1"
        or plan.get("supersedes_run_identity") != "b-ipc1-matched-learning-run2"
        or dependency["failure_file_sha256"]
        != "7a5aef7085a53280d18caad6c2d354248d26d153525f90aa21442db707d72f78"
        or dependency["failure_internal_receipt_sha256"]
        != "b123dfc3795604d67913e6a6f9ff4b058d988facbec822ab089f85cc3ef61d0c"
        or dependency["run_log_sha256"]
        != "0b4f98163c8d418dab62a153906d2a4127ce671cbf4b00545b9f563a6d3afdbd"
        or dependency["archive_sha256"]
        != "edd4354db156c48b5044f2ebe030861a77557ad901c7dbfbd89cd48dfa807cae"
        or dependency["manifest_sha256"]
        != "b9ac5c045ee929b663480a1f44cab1da881e243d22ee5a083e8a6b9d7d600bdd"
        or dependency["failed_execution_commit"] != "3cd894e945f5b8239fe68073413205bee43e3877"
        or dependency["failed_plan_file_sha256"]
        != "4957dadb79da68acb5a60ffe5e3cf4c76379ab108f8bad41d7387c56bd4e61a1"
        or [
            dependency[key]
            for key in (
                "model_calls_completed",
                "backward_calls_completed",
                "optimizer_steps_completed",
                "static_rows_exposed",
                "ffn_rows_exposed",
            )
        ]
        != [184, 63, 5, 48, 12]
        or any(
            dependency[key] is not False
            for key in (
                "partial_state_reusable",
                "validation_started",
                "heldout_opened",
                "scientific_result",
                "candidates_created",
            )
        )
    ):
        raise PhaseBContractError("B-IPC1 R1 superseded-run boundary differs")
    commit = plan.get("mechanism_code_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or plan.get("implementation_commit") != commit
    ):
        raise PhaseBContractError("B-IPC1 plan lacks one exact mechanism commit")
    if plan.get("plan_sha256") != canonical_plan_sha256(plan):
        raise PhaseBContractError("B-IPC1 internal plan hash differs")
    boundaries = plan.get("boundaries")
    if not isinstance(boundaries, dict) or boundaries != {
        "nomination_only": True,
        "live_trajectory_count": 0,
        "later_live_admission_minimum_complete_trajectories": 4,
        "e33_trainable": False,
        "H176_loaded": False,
        "strand_a_combined": False,
        "generation": False,
        "cache": False,
        "teacher_forced_rows_are_live_trajectories": False,
    }:
        raise PhaseBContractError("B-IPC1 claim boundary differs")
    require_exact_mapping_keys(
        plan["protected_model"],
        {"name", "path", "weight_sha256", "architecture", "text_hidden_size", "trainable"},
        label="protected model",
    )
    require_exact_mapping_keys(
        plan["model_metadata_sha256"],
        {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "generation_config.json",
            "processor_config.json",
        },
        label="model metadata",
    )
    for field in ("bank_manifest", "overlap_closure", "runtime_source", "generator_source"):
        require_exact_mapping_keys(plan[field], {"path", "sha256"}, label=field)
    require_exact_mapping_keys(plan["taskset_source"], {"commit", "path", "sha256"}, label="taskset source")
    antecedents = plan["antecedents"]
    if not isinstance(antecedents, list) or [record.get("name") for record in antecedents] != ["HIC0_R1", "B1R"]:
        raise PhaseBContractError("B-IPC1 antecedent order differs")
    require_exact_mapping_keys(
        antecedents[0],
        {
            "name",
            "binding_path",
            "binding_sha256",
            "classification",
            "success_file_sha256",
            "internal_receipt_sha256",
            "execution_commit",
            "plan_file_sha256",
            "formal_terminal_valid",
            "validator_error",
            "invalidating_cause",
            "descriptive_evidence_only",
            "candidate_reused",
            "rows_reused",
            "seeds_reused",
            "admission",
            "threshold_source",
        },
        label="HIC0 antecedent",
    )
    require_exact_mapping_keys(
        antecedents[0]["descriptive_evidence_only"],
        {
            "insert_eps_minus_base_mean_nll",
            "inplace_eps_minus_base_mean_nll",
            "penalty_removal_fraction",
            "hidden_drift_ratio_mean",
            "logit_drift_ratio_mean",
            "inplace_margin_delta_mean",
            "inplace_margin_delta_worst",
            "formal_gate7_descriptive_value",
        },
        label="HIC0 descriptive evidence",
    )
    require_exact_mapping_keys(
        antecedents[1],
        {
            "name",
            "binding_path",
            "binding_sha256",
            "status",
            "disposition",
            "success_receipt_sha256",
            "candidate_reused",
            "rows_reused",
            "seeds_reused",
            "admission",
            "threshold_source",
        },
        label="B1R antecedent",
    )
    proof = require_exact_mapping_keys(
        plan["terminal_proof"],
        {
            "manifest_path",
            "manifest_sha256",
            "proof_execution_commit",
            "runner_sha256",
            "decoded_proof_sha256",
            "log_sha256",
            "exit_status_sha256",
            "exit_status",
            "model_loaded",
            "cuda_initialized",
            "exact_host_repository",
            "maximal_success_file_sha256",
            "failure_terminal_count",
            "tamper_cases_rejected",
            "late_failure_terminal_count",
            "late_failure_tamper_cases_rejected",
            "full_freeze_target_count",
            "mapping_insertion_permutation_canonical_equal",
            "global_exactly_one_terminal",
            "superseded_invocation_log_sha256",
            "superseded_invocation_exit_sha256",
        },
        label="terminal proof binding",
    )
    if (
        proof["exit_status"] != 0
        or proof["model_loaded"] is not False
        or proof["cuda_initialized"] is not False
        or proof["exact_host_repository"] is not True
        or proof["failure_terminal_count"] != 4
        or proof["tamper_cases_rejected"] != 18
        or proof["late_failure_terminal_count"] != 2
        or proof["late_failure_tamper_cases_rejected"] != 21
        or not isinstance(proof["full_freeze_target_count"], int)
        or proof["full_freeze_target_count"] < 1
        or proof["mapping_insertion_permutation_canonical_equal"] is not True
        or proof["global_exactly_one_terminal"] is not True
    ):
        raise PhaseBContractError("B-IPC1 terminal proof binding differs")
    training = plan.get("training")
    require_exact_mapping_keys(
        training,
        {
            "arm_order",
            "optimizer_updates_per_arm",
            "rows_per_update",
            "rows_per_action_per_update",
            "unique_row_exposures_per_arm",
            "same_schedule_every_arm",
            "recurrent_depth",
            "bptt_window",
            "early_stop",
            "initialization_seed_payload",
            "initialization_derivation_sha256",
            "initialization_seed",
            "construction_order",
            "optimizer",
            "objective",
        },
        label="training",
    )
    if not isinstance(training, dict) or (
        training.get("arm_order") != list(TRAINING_ARMS)
        or training.get("optimizer_updates_per_arm") != 4
        or training.get("rows_per_update") != 12
        or training.get("rows_per_action_per_update") != 4
        or training.get("unique_row_exposures_per_arm") != 48
        or training.get("recurrent_depth") != 4
        or training.get("bptt_window") != 4
        or training.get("early_stop") is not False
        or training.get("initialization_seed_payload") != INITIALIZATION_SEED_PAYLOAD
        or training.get("initialization_derivation_sha256") != INITIALIZATION_DERIVATION_SHA256
        or training.get("initialization_seed") != INITIALIZATION_SEED
        or training.get("optimizer")
        != {
            "name": "AdamW",
            "learning_rate": 0.0001,
            "betas": [0.9, 0.95],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
        }
    ):
        raise PhaseBContractError("B-IPC1 matched training contract differs")
    objective = training.get("objective")
    if not isinstance(objective, dict) or objective != {
        "formula": "aligned_suffix_CE + 0.1*w(action)*mean_branch_relu(BASE_margin_detached-candidate_margin)",
        "retention_coefficient": 0.1,
        "action_weights": [
            {"action": "solve_owned", "weight": 1.0},
            {"action": "delegate_terminal", "weight": 1.0},
            {"action": "delegate_coordinator", "weight": 2.0},
        ],
        "branch_reduction": "arithmetic_mean_over_every_correct_leaf_branch",
        "candidate_logit_dtype": "float32",
        "base_detached": True,
        "same_forward": True,
    }:
        raise PhaseBContractError("B-IPC1 action-margin retention objective differs")
    banks = plan.get("banks")
    if not isinstance(banks, list) or [bank.get("split") for bank in banks] != [
        "train",
        "validation",
        "heldout",
    ]:
        raise PhaseBContractError("B-IPC1 bank bindings are not ordered records")
    for bank in banks:
        split = bank["split"]
        expected_bank_keys = {
            "split",
            "taskset_split",
            "variants",
            "instance_start",
            "instance_stop",
            "seed_payload",
            "derivation_sha256",
            "seed",
            "rows_per_action",
            "rows",
            "selection_path",
            "selection_sha256",
            "parquet_path",
            "parquet_sha256",
            "ordered_task_key_sha256",
            "ordered_key_action_sha256",
            "row_list_canonical_sha256",
        }
        if split == "train":
            expected_bank_keys.add("ordered_update_schedule_sha256")
        require_exact_mapping_keys(bank, expected_bank_keys, label=f"{split} bank")
        expected = SELECTIONS[split]
        if (
            bank.get("seed_payload") != expected["seed_payload"]
            or bank.get("derivation_sha256") != expected["derivation_sha256"]
            or bank.get("seed") != expected["seed"]
            or bank.get("instance_start") != expected["instance_start"]
            or bank.get("instance_stop") != expected["instance_stop"]
            or bank.get("rows_per_action") != expected["rows_per_action"]
        ):
            raise PhaseBContractError(f"B-IPC1 {split} bank derivation differs")
    require_exact_mapping_keys(
        plan["receiver"],
        {
            "geometry",
            "slots",
            "target_span",
            "token_shift",
            "mask_shift",
            "position_shift",
            "label_shift",
            "initial_receiver_gate",
            "evaluation_depths",
        },
        label="receiver",
    )
    require_exact_mapping_keys(
        plan["pre_update_mechanism_gate"],
        {
            "probe_selection_indices",
            "probe_identity_sha256",
            "eps_backward_assignments",
            "index5_backward",
            "zero_identity_fields",
            "failure_disposition",
        },
        label="pre-update mechanism gate",
    )
    for assignment in plan["pre_update_mechanism_gate"]["eps_backward_assignments"]:
        require_exact_mapping_keys(assignment, {"arm", "selection_index"}, label="EPS backward assignment")
    require_exact_mapping_keys(
        plan["evaluation_firewall"],
        {
            "all_arms_frozen_before_validation",
            "validation_rows",
            "heldout_rows_if_opened",
            "open_heldout_iff_at_least_one_common_validation_pass",
            "validation_cannot_select_checkpoint_threshold_or_hyperparameter",
            "heldout_cannot_select_checkpoint_threshold_or_hyperparameter",
            "reruns",
            "row_exclusions",
        },
        label="evaluation firewall",
    )
    gates = require_exact_mapping_keys(
        plan["nomination_gates"], {"common_each_split", "recurrent_each_split"}, label="nomination gates"
    )
    require_exact_mapping_keys(
        gates["common_each_split"],
        {"delta_n_pre_minus_post", "delta_m_post_minus_base", "coordinator_delta_m_post_minus_pre", "noncollapse"},
        label="common nomination gates",
    )
    require_exact_mapping_keys(
        gates["recurrent_each_split"],
        {
            "positive_nll_route",
            "positive_margin_route",
            "depth_T4_over_T1",
            "retention",
            "stability",
            "T8_nonregression",
        },
        label="recurrent nomination gates",
    )
    require_exact_mapping_keys(plan["schedules"], {"validation_reject", "heldout_open"}, label="schedules")
    schedule_keys = {
        "model_forwards",
        "source_forwards",
        "receiver_forwards",
        "backward_calls",
        "optimizer_steps",
        "model_call_schedule_sha256",
        "cache_label_count",
        "cache_label_sha256",
        "memory_checkpoint_count",
        "memory_label_sha256",
    }
    for name, schedule in plan["schedules"].items():
        require_exact_mapping_keys(schedule, schedule_keys, label=f"{name} schedule")
    require_exact_mapping_keys(
        plan["terminal_statuses"],
        {
            "recurrent_nominated",
            "inplace_learning_nominated",
            "heldout_opened_not_nominated",
            "validation_not_nominated",
            "mechanism_rejected",
            "nocache_rejected",
            "incomplete",
            "infrastructure",
        },
        label="terminal statuses",
    )
    require_exact_mapping_keys(
        plan["execution_environment"],
        {
            "uv",
            "uv_project",
            "UV_PROJECT_ENVIRONMENT",
            "PYTHONPATH",
            "CUDA_VISIBLE_DEVICES",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        },
        label="execution environment",
    )
    require_exact_mapping_keys(
        plan["outputs"],
        {
            "directory",
            "artifact_cap_bytes",
            "candidate_files",
            "terminal",
            "candidate_valid_only_with_roundtrip_valid_success",
        },
        label="outputs",
    )
    software = require_exact_mapping_keys(
        plan["software"],
        {"transformers", "torch", "torch_distribution", "transformers_source_sha256"},
        label="software",
    )
    require_exact_mapping_keys(
        software["transformers_source_sha256"],
        {
            "transformers.cache_utils",
            "transformers.generation.utils",
            "transformers.loss.loss_utils",
            "transformers.models.qwen3_5.modeling_qwen3_5",
        },
        label="Transformers source hashes",
    )
    resources = plan.get("resources")
    if not isinstance(resources, dict) or resources != {
        "visible_gpus": 1,
        "required_gpu_name": "NVIDIA RTX A6000",
        "second_gpu_idle": True,
        "cuda_memory_cap_bytes": CUDA_MEMORY_CAP_BYTES,
        "minimum_host_ram_bytes": MINIMUM_HOST_RAM_BYTES,
        "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "artifact_cap_bytes": ARTIFACT_CAP_BYTES,
        "outer_wall_clock_seconds": 14_400,
        "compute_limit_seconds": 14_040,
        "failure_audit_limit_seconds": 300,
        "terminal_publication_headroom_seconds": 60,
    }:
        raise PhaseBContractError("B-IPC1 resource contract differs")
    if require_authorized and (
        plan.get("status") != "frozen_pending_independent_review"
        or plan.get("execution_authorization") != "independent_gatekeeper_review_then_root_schedule_required"
    ):
        raise PhaseBContractError("B-IPC1 plan is not independently authorized")


def canonical_bank_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_seed_derivations() -> None:
    values = ((INITIALIZATION_SEED_PAYLOAD, INITIALIZATION_DERIVATION_SHA256, INITIALIZATION_SEED),)
    values += tuple(
        (selection["seed_payload"], selection["derivation_sha256"], selection["seed"])
        for selection in SELECTIONS.values()
    )
    for payload, expected_hash, expected_seed in values:
        observed_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if observed_hash != expected_hash or int(observed_hash[:8], 16) != expected_seed:
            raise PhaseBContractError(f"B-IPC1 seed derivation differs for {payload}")


def select_balanced_rows(rows: Sequence[dict[str, Any]], *, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select an action-balanced bank without depending on input ordering."""

    verify_seed_derivations()
    if split not in SELECTIONS:
        raise PhaseBContractError(f"B-IPC1 unknown split {split}")
    specification = SELECTIONS[split]
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("task_key")
        action = row.get("action")
        if not isinstance(key, str) or not key or action not in ACTIONS:
            raise PhaseBContractError(f"B-IPC1 {split} pool has an invalid key/action")
        if key in by_key:
            raise PhaseBContractError(f"B-IPC1 {split} pool repeats task key {key}")
        by_key[key] = row
    selected_by_action: dict[str, list[str]] = {}
    for action in ACTIONS:
        candidates = sorted(key for key, row in by_key.items() if row["action"] == action)
        rng = random.Random(specification["seed"])
        rng.shuffle(candidates)
        count = specification["rows_per_action"]
        if len(candidates) < count:
            raise PhaseBContractError(f"B-IPC1 {split} pool is insufficient for {action}")
        selected_by_action[action] = candidates[:count]
    selected_keys = [
        selected_by_action[action][index] for index in range(specification["rows_per_action"]) for action in ACTIONS
    ]
    selected = [by_key[key] for key in selected_keys]
    expected_rows = specification["rows_per_action"] * len(ACTIONS)
    if len(selected) != expected_rows or Counter(row["action"] for row in selected) != Counter(
        {action: specification["rows_per_action"] for action in ACTIONS}
    ):
        raise PhaseBContractError(f"B-IPC1 {split} selection is not exactly balanced")
    selection = {
        "schema_version": "q35-2b-b-ipc1-selection/v1",
        "split": split,
        "seed_payload": specification["seed_payload"],
        "derivation_sha256": specification["derivation_sha256"],
        "seed": specification["seed"],
        "candidate_pool_rows": len(rows),
        "candidate_pool_order_independent": True,
        "algorithm": "lexicographic_per_action_then_seeded_shuffle_then_take_then_action_interleave",
        "action_order": list(ACTIONS),
        "rows_per_action": specification["rows_per_action"],
        "selected": [
            {"position": index, "task_key": row["task_key"], "expected_action": row["action"]}
            for index, row in enumerate(selected)
        ],
        "ordered_task_key_sha256": canonical_bank_sha256(selected_keys),
        "ordered_key_action_sha256": canonical_bank_sha256(
            [{"task_key": row["task_key"], "expected_action": row["action"]} for row in selected]
        ),
        "row_list_canonical_sha256": canonical_bank_sha256(selected),
        "row_canonical_sha256": [canonical_bank_sha256(row) for row in selected],
    }
    if split == "train":
        batches = [selected_keys[index : index + 12] for index in range(0, TRAIN_ROWS, 12)]
        if len(batches) != 4 or any(
            Counter(by_key[key]["action"] for key in batch) != Counter({action: 4 for action in ACTIONS})
            for batch in batches
        ):
            raise PhaseBContractError("B-IPC1 training schedule is not four exact 4/4/4 updates")
        selection["updates"] = [
            {
                "update_index": index,
                "rows": [
                    {"position": position, "task_key": key, "expected_action": by_key[key]["action"]}
                    for position, key in enumerate(batch)
                ],
            }
            for index, batch in enumerate(batches, start=1)
        ]
        selection["ordered_update_schedule_sha256"] = canonical_bank_sha256(selection["updates"])
    return selected, selection


def validate_bank_disjointness(
    selected_by_split: dict[str, Sequence[dict[str, Any]]],
    *,
    excluded_key_sets: dict[str, set[str]],
    excluded_row_hash_sets: dict[str, set[str]],
) -> dict[str, Any]:
    if tuple(selected_by_split) != ("train", "validation", "heldout"):
        raise PhaseBContractError("B-IPC1 split order differs")
    split_keys = {split: {str(row["task_key"]) for row in rows} for split, rows in selected_by_split.items()}
    split_hashes = {split: {canonical_bank_sha256(row) for row in rows} for split, rows in selected_by_split.items()}
    records: list[dict[str, Any]] = []
    splits = tuple(selected_by_split)
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            key_overlap = sorted(split_keys[left].intersection(split_keys[right]))
            row_overlap = sorted(split_hashes[left].intersection(split_hashes[right]))
            records.append(
                {
                    "left": left,
                    "right": right,
                    "task_key_overlap_count": len(key_overlap),
                    "row_hash_overlap_count": len(row_overlap),
                }
            )
            if key_overlap or row_overlap:
                raise PhaseBContractError(f"B-IPC1 selected splits overlap: {left}/{right}")
    if set(excluded_key_sets) != set(excluded_row_hash_sets):
        raise PhaseBContractError("B-IPC1 exclusion key/hash source closure differs")
    all_selected_keys = set().union(*split_keys.values())
    all_selected_hashes = set().union(*split_hashes.values())
    for source in sorted(excluded_key_sets):
        key_overlap = sorted(all_selected_keys.intersection(excluded_key_sets[source]))
        row_overlap = sorted(all_selected_hashes.intersection(excluded_row_hash_sets[source]))
        records.append(
            {
                "left": "all_ipc1_selected",
                "right": source,
                "task_key_overlap_count": len(key_overlap),
                "row_hash_overlap_count": len(row_overlap),
            }
        )
        if key_overlap or row_overlap:
            raise PhaseBContractError(f"B-IPC1 rows overlap frozen source {source}")
    return {
        "comparisons": records,
        "all_zero": True,
        "selected_rows_permanently_excluded_from_future_training": 96,
    }


def build_model_call_schedule(
    train_keys: Sequence[str], validation_keys: Sequence[str], heldout_keys: Sequence[str], *, open_heldout: bool
) -> list[dict[str, Any]]:
    if (len(train_keys), len(validation_keys), len(heldout_keys)) != (48, 24, 24):
        raise PhaseBContractError("B-IPC1 call schedule requires exact 48/24/24 key banks")
    if any(len(set(keys)) != len(keys) for keys in (train_keys, validation_keys, heldout_keys)):
        raise PhaseBContractError("B-IPC1 call schedule contains duplicate keys")
    calls: list[dict[str, Any]] = []

    def add(phase: str, kind: str, key: str, arm: str, *, backward: bool = False) -> None:
        calls.append(
            {
                "call_index": len(calls) + 1,
                "phase": phase,
                "kind": kind,
                "task_key": key,
                "arm": arm,
                "backward": backward,
            }
        )

    for key in train_keys:
        add("pre_learning", "source_capture", key, "SOURCE")
    for key in train_keys:
        add("pre_learning", "receiver", key, "BASE")
    probe_indices = (0, 1, 2, 5)
    backward_designations = {("STATIC", 0), ("FFN", 1), ("RECURRENT", 2)}
    for index in probe_indices:
        key = train_keys[index]
        add("pre_update_probe", "receiver", key, "ZERO")
        for arm in TRAINING_ARMS:
            add("pre_update_probe", "receiver", key, f"INPLACE_ZERO_{arm}")
            add(
                "pre_update_probe",
                "receiver",
                key,
                f"INPLACE_EPS_{arm}",
                backward=(arm, index) in backward_designations,
            )
    for arm in TRAINING_ARMS:
        for key in train_keys:
            add("learning", "receiver", key, arm, backward=True)

    def evaluation(split: str, keys: Sequence[str]) -> None:
        for key in keys:
            add(split, "source_capture", key, "SOURCE")
        for key in keys:
            for arm in (
                "BASE",
                "PRE_STATIC",
                "PRE_FFN",
                "PRE_RECURRENT_T4",
                "POST_STATIC",
                "POST_FFN",
                "POST_RECURRENT_T1",
                "POST_RECURRENT_T2",
                "POST_RECURRENT_T4",
                "POST_RECURRENT_T8",
            ):
                add(split, "receiver", key, arm)

    evaluation("validation", validation_keys)
    if open_heldout:
        evaluation("heldout", heldout_keys)
    expected_calls = 796 if open_heldout else 532
    expected_receivers = 700 if open_heldout else 460
    if len(calls) != expected_calls or sum(call["kind"] == "receiver" for call in calls) != expected_receivers:
        raise PhaseBContractError("B-IPC1 model-call schedule count differs")
    if sum(call["backward"] for call in calls) != 147:
        raise PhaseBContractError("B-IPC1 backward count differs")
    return calls


def build_cache_guard_labels(calls: Sequence[dict[str, Any]]) -> list[str]:
    labels = ["CACHE_GUARD_ENTRY"]
    for call in calls:
        index = call.get("call_index")
        if type(index) is not int or index != (len(labels) - 1) // 2 + 1:
            raise PhaseBContractError("B-IPC1 cache schedule call order differs")
        labels.extend((f"CACHE_GUARD_PRE_IPC1_C{index:04d}", f"CACHE_GUARD_POST_IPC1_C{index:04d}"))
    labels.extend(("CACHE_GUARD_FINAL", "CACHE_GUARD_EXIT"))
    if len(labels) != 2 * len(calls) + 3:
        raise PhaseBContractError("B-IPC1 cache label count differs")
    return labels


def build_memory_checkpoint_labels(calls: Sequence[dict[str, Any]]) -> list[str]:
    labels = ["after_model_load", "after_module_construction"]
    learning_counts = {arm: 0 for arm in TRAINING_ARMS}
    for call in calls:
        index = call["call_index"]
        labels.append(f"call:{index:04d}:complete")
        if call["backward"]:
            labels.append(f"call:{index:04d}:post_backward")
        if call["phase"] == "learning":
            arm = call["arm"]
            learning_counts[arm] += 1
            if learning_counts[arm] % 12 == 0:
                update = learning_counts[arm] // 12
                labels.extend(
                    (
                        f"optimizer:{arm}:update{update}:post_clip",
                        f"optimizer:{arm}:update{update}:post_step",
                    )
                )
            if learning_counts[arm] == 48:
                labels.append(f"optimizer:{arm}:destroyed")
    labels.extend(("after_final_audits", "before_candidate_writes"))
    for arm in TRAINING_ARMS:
        labels.extend((f"candidate:{arm}:before_write", f"candidate:{arm}:after_write"))
    labels.append("before_terminal")
    if len(labels) != len(set(labels)):
        raise PhaseBContractError("B-IPC1 memory checkpoint labels are not unique")
    return labels


def differentiable_margin_retention_from_baseline_margins(
    *,
    candidate_logits: Any,
    base_branch_margins: Sequence[Any],
    branches: Sequence[dict[str, Any]],
    action: str,
    aligned_suffix_ce: Any,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    if len(base_branch_margins) != len(branches) or not branches or action not in ACTIONS:
        raise PhaseBContractError("B-IPC1 sparse baseline-margin evidence differs")
    penalties = []
    records: list[dict[str, Any]] = []
    for baseline, branch in zip(base_branch_margins, branches, strict=True):
        offset = branch["logit_offset"]
        correct_token = branch["correct_token_id"]
        other_tokens = branch["other_token_ids"]
        if not other_tokens:
            raise PhaseBContractError("B-IPC1 action branch lacks another live child")
        logits = candidate_logits[0, offset].float()
        margin = logits[correct_token] - logits[other_tokens].max()
        baseline_value = torch.as_tensor(baseline, device=margin.device, dtype=torch.float32).detach()
        penalties.append(torch.relu(baseline_value - margin))
        records.append(
            {
                "logit_offset": int(offset),
                "correct_token_id": int(correct_token),
                "other_token_ids": [int(token) for token in other_tokens],
            }
        )
    mean_penalty = torch.stack(penalties).mean()
    action_weight = 2.0 if action == "delegate_coordinator" else 1.0
    return aligned_suffix_ce + 0.1 * action_weight * mean_penalty, {
        "action_weight": action_weight,
        "retention_coefficient": 0.1,
        "branch_count": len(branches),
        "branches": records,
        "baseline_margins_detached": True,
    }


def summarize24(values: Sequence[float], actions: Sequence[str]) -> dict[str, Any]:
    finite = [float(value) for value in values]
    if len(finite) != EVALUATION_ROWS or len(actions) != EVALUATION_ROWS:
        raise PhaseBContractError("B-IPC1 aggregation requires 24 paired rows")
    if not all(math.isfinite(value) for value in finite):
        raise PhaseBContractError("B-IPC1 aggregation contains a nonfinite value")
    if Counter(actions) != Counter({action: 8 for action in ACTIONS}):
        raise PhaseBContractError("B-IPC1 aggregation requires 8/8/8 action balance")
    return {
        "values": finite,
        "mean": math.fsum(finite) / EVALUATION_ROWS,
        "median": midpoint_median(finite),
        "minimum": min(finite),
        "maximum": max(finite),
        "strict_wins": sum(value > STRICT_WIN_EPSILON for value in finite),
        "per_action_means": [
            {
                "action": action,
                "mean": math.fsum(value for value, observed in zip(finite, actions, strict=True) if observed == action)
                / 8,
            }
            for action in ACTIONS
        ],
    }


def _per_action(summary: dict[str, Any], action: str) -> float:
    records = summary.get("per_action_means")
    if not isinstance(records, list) or [record.get("action") for record in records] != list(ACTIONS):
        raise PhaseBContractError("B-IPC1 per-action summaries are not ordered named records")
    return float(next(record["mean"] for record in records if record["action"] == action))


def evaluate_common_arm(
    *,
    action_order: Sequence[str],
    base: Sequence[dict[str, Any]],
    pre: Sequence[dict[str, Any]],
    post: Sequence[dict[str, Any]],
    safety_and_noncollapse: bool,
) -> dict[str, Any]:
    if any(len(rows) != EVALUATION_ROWS for rows in (base, pre, post)):
        raise PhaseBContractError("B-IPC1 common arm lacks 24 rows")
    keys = [row.get("task_key") for row in base]
    if any([row.get("task_key") for row in rows] != keys for rows in (pre, post)):
        raise PhaseBContractError("B-IPC1 common arm row order differs")
    actions = list(action_order)
    if [row.get("action") for row in base] != actions or any(
        [row.get("action") for row in rows] != actions for rows in (pre, post)
    ):
        raise PhaseBContractError("B-IPC1 common arm action order differs")
    delta_n = summarize24([float(a["nll"]) - float(b["nll"]) for a, b in zip(pre, post, strict=True)], actions)
    retention_m = summarize24(
        [float(a["margin"]) - float(b["margin"]) for a, b in zip(post, base, strict=True)], actions
    )
    correction_m = summarize24(
        [float(a["margin"]) - float(b["margin"]) for a, b in zip(post, pre, strict=True)], actions
    )
    coordinator_values = [
        value for value, action in zip(correction_m["values"], actions, strict=True) if action == "delegate_coordinator"
    ]
    gates = [
        {"name": "exact_complete_finite_safe_noncollapse", "passed": bool(safety_and_noncollapse)},
        {
            "name": "nll_learning",
            "passed": delta_n["mean"] >= 0.001
            and delta_n["median"] >= 0.0
            and delta_n["strict_wins"] >= 13
            and min(_per_action(delta_n, action) for action in ACTIONS) >= -0.005
            and delta_n["minimum"] >= -0.05,
        },
        {
            "name": "base_margin_retention",
            "passed": retention_m["mean"] >= -0.05
            and min(_per_action(retention_m, action) for action in ACTIONS) >= -0.10
            and retention_m["minimum"] >= -0.50
            and _per_action(retention_m, "delegate_coordinator") >= -0.05
            and min(
                value
                for value, action in zip(retention_m["values"], actions, strict=True)
                if action == "delegate_coordinator"
            )
            >= -0.50,
        },
        {
            "name": "coordinator_margin_correction",
            "passed": math.fsum(coordinator_values) / 8 >= 0.05
            and midpoint_median(coordinator_values) >= 0.0
            and sum(value > STRICT_WIN_EPSILON for value in coordinator_values) >= 5,
        },
    ]
    return {
        "passed": all(record["passed"] for record in gates),
        "gates": gates,
        "summaries": [
            {"name": "delta_n_pre_minus_post", "value": delta_n},
            {"name": "delta_m_post_minus_base", "value": retention_m},
            {
                "name": "coordinator_delta_m_post_minus_pre",
                "value": {
                    "values": coordinator_values,
                    "mean": math.fsum(coordinator_values) / 8,
                    "median": midpoint_median(coordinator_values),
                    "strict_wins": sum(value > STRICT_WIN_EPSILON for value in coordinator_values),
                },
            },
        ],
    }


def evaluate_recurrent_value(
    *,
    actions: Sequence[str],
    recurrent: dict[str, Sequence[dict[str, Any]]],
    ffn: Sequence[dict[str, Any]],
    retention_and_stability_passed: bool,
) -> dict[str, Any]:
    required = tuple(f"T{depth}" for depth in EVALUATION_DEPTHS)
    if tuple(recurrent) != required or len(ffn) != EVALUATION_ROWS:
        raise PhaseBContractError("B-IPC1 recurrent evaluation grid differs")
    keys = [row.get("task_key") for row in ffn]
    if any(len(recurrent[name]) != EVALUATION_ROWS for name in required) or any(
        [row.get("task_key") for row in recurrent[name]] != keys for name in required
    ):
        raise PhaseBContractError("B-IPC1 recurrent rows differ across depths")
    rec4 = recurrent["T4"]
    contrasts = {
        "A_N": summarize24([float(a["nll"]) - float(b["nll"]) for a, b in zip(ffn, rec4, strict=True)], actions),
        "A_M": summarize24([float(a["margin"]) - float(b["margin"]) for a, b in zip(rec4, ffn, strict=True)], actions),
        "R4_minus_R1_nll": summarize24(
            [float(a["nll"]) - float(b["nll"]) for a, b in zip(recurrent["T1"], rec4, strict=True)], actions
        ),
        "R4_minus_R1_margin": summarize24(
            [float(a["margin"]) - float(b["margin"]) for a, b in zip(rec4, recurrent["T1"], strict=True)], actions
        ),
        "R8_minus_R4_nll": summarize24(
            [float(a["nll"]) - float(b["nll"]) for a, b in zip(recurrent["T8"], rec4, strict=True)], actions
        ),
        "R8_minus_R4_margin": summarize24(
            [float(a["margin"]) - float(b["margin"]) for a, b in zip(recurrent["T8"], rec4, strict=True)], actions
        ),
    }
    nll = contrasts["A_N"]
    margin = contrasts["A_M"]
    nll_route = (
        nll["mean"] >= 0.001
        and nll["median"] >= 0
        and nll["strict_wins"] >= 13
        and sum(_per_action(nll, action) >= 0 for action in ACTIONS) >= 2
    )
    margin_route = (
        margin["mean"] >= 0.020
        and margin["median"] >= 0
        and margin["strict_wins"] >= 13
        and sum(_per_action(margin, action) >= 0 for action in ACTIONS) >= 2
    )
    depth_nll = contrasts["R4_minus_R1_nll"]
    depth_margin = contrasts["R4_minus_R1_margin"]
    depth_gate = (
        depth_nll["mean"] >= -0.0025
        and depth_nll["minimum"] >= -0.05
        and depth_margin["mean"] >= -0.05
        and depth_margin["minimum"] >= -0.50
        and (depth_nll["mean"] >= 0.0005 or depth_margin["mean"] >= 0.010)
    )
    r8_nll = contrasts["R8_minus_R4_nll"]
    r8_margin = contrasts["R8_minus_R4_margin"]
    r8_gate = (
        r8_nll["mean"] <= 0.005
        and r8_nll["maximum"] <= 0.05
        and r8_margin["mean"] >= -0.10
        and r8_margin["minimum"] >= -0.50
    )
    gates = [
        {"name": "positive_recurrence_over_ffn", "passed": nll_route or margin_route},
        {"name": "nll_route", "passed": nll_route},
        {"name": "action_margin_route", "passed": margin_route},
        {"name": "depth_T4_over_T1", "passed": depth_gate},
        {"name": "retention_and_stability", "passed": bool(retention_and_stability_passed)},
        {"name": "T8_nonregression", "passed": r8_gate},
    ]
    return {
        "passed": all(
            record["passed"] for record in gates if record["name"] not in {"nll_route", "action_margin_route"}
        ),
        "gates": gates,
        "summaries": [{"name": name, "value": value} for name, value in contrasts.items()],
    }


def canonical_terminal_bytes(receipt: dict[str, Any]) -> bytes:
    try:
        return json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PhaseBContractError("B-IPC1 terminal is not finite canonical JSON") from error


def strict_json_loads(payload: bytes | str) -> Any:
    """Parse terminal JSON while rejecting duplicate keys and non-finite constants."""

    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PhaseBContractError(f"B-IPC1 terminal repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise PhaseBContractError(f"B-IPC1 terminal contains non-finite JSON constant {value}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=object_without_duplicates,
            parse_constant=reject_constant,
        )
    except PhaseBContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseBContractError("B-IPC1 terminal is not strict UTF-8 JSON") from error


def validate_ordered_records(records: Any, expected_names: Sequence[str], *, label: str) -> None:
    if not isinstance(records, list) or [record.get("name") for record in records] != list(expected_names):
        raise PhaseBContractError(f"B-IPC1 {label} must be an ordered list of named records")


def roundtrip_validate_terminal(
    receipt: dict[str, Any], *, validator: Any, validator_kwargs: dict[str, Any]
) -> tuple[dict[str, Any], bytes, str]:
    """Validate exactly the canonical parsed object that may be published."""

    candidate = dict(receipt)
    candidate.pop("receipt_sha256", None)
    candidate["receipt_sha256"] = hashlib.sha256(canonical_terminal_bytes(candidate)).hexdigest()
    payload = canonical_terminal_bytes(candidate)
    parsed = strict_json_loads(payload)
    validator(parsed, **validator_kwargs)
    reparsed = strict_json_loads(payload)
    if canonical_terminal_bytes(reparsed) != payload:
        raise PhaseBContractError("B-IPC1 canonical terminal bytes do not round-trip exactly")
    validator(reparsed, **validator_kwargs)
    return reparsed, payload, hashlib.sha256(payload).hexdigest()


def verify_published_terminal(path: Path, payload: bytes, *, validator: Any, validator_kwargs: dict[str, Any]) -> str:
    observed = path.read_bytes()
    if observed != payload:
        raise PhaseBContractError("B-IPC1 published terminal bytes differ from validated bytes")
    parsed = strict_json_loads(observed)
    validator(parsed, **validator_kwargs)
    return hashlib.sha256(observed).hexdigest()
