#!/usr/bin/env python3
"""Exercise the production B-IPC1 terminal validators and writers without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import run_phase_b_ipc1_matched_learning_v1 as runtime
import torch

from prime_rl.phase_b_contract import PhaseBContractError, file_sha256, load_json_file
from prime_rl.phase_b_ipc1 import (
    ACTIONS,
    EVALUATION_DEPTHS,
    FAILURE_STATUS_CLASSES,
    TRAINING_ARMS,
    build_cache_guard_labels,
    build_memory_checkpoint_labels,
    build_model_call_schedule,
    canonical_terminal_bytes,
    roundtrip_validate_terminal,
    verify_published_terminal,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--repository-root", type=Path, default=runtime.WORKTREE)
    return parser.parse_args()


def _relocate_repository_paths(plan: dict[str, Any], repository_root: Path) -> None:
    original = str(runtime.WORKTREE)

    def relocate(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str) and value.startswith(f"{original}/"):
            return f"{repository_root}/{value.removeprefix(f'{original}/')}"
        return value

    relocated = relocate(plan)
    plan.clear()
    plan.update(relocated)


def _metric(task_key: str, action: str, *, nll: float, margin: float, recurrent_t8: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "task_key": task_key,
        "action": action,
        "nll": nll,
        "native_output_loss_descriptive": nll,
        "native_minus_float64_nll_descriptive": 0.0,
        "margin": margin,
        "finite": True,
        "branch_metrics": [
            {
                "target_offset": 0,
                "logit_offset": 0,
                "correct_token_id": 1,
                "other_token_ids": [2],
                "live_actions": list(ACTIONS),
                "correct_logit": margin,
                "max_other_logit": 0.0,
                "margin": margin,
            }
        ],
        "hashes": {
            "inputs_embeds": "0" * 64,
            "attention_mask": "1" * 64,
            "position_ids": "2" * 64,
            "labels": "3" * 64,
            "final_hidden": "4" * 64,
            "first_suffix_logits": "5" * 64,
        },
    }
    if recurrent_t8:
        record["retention"] = {
            f"T{depth}": {"cosine": 1.0, "norm_ratio": 1.0, "relative_l2": 0.0} for depth in EVALUATION_DEPTHS
        }
        record["stability_T8"] = {
            "memory_change_rms": [0.1] * 8,
            "memory_contraction_steps_2_8": [0.5] * 7,
            "median_memory_contraction_steps_2_8": 0.5,
            "max_memory_contraction_steps_2_8": 0.5,
            "memory_oscillation_rate": 0.0,
            "finite": True,
        }
    return record


def _evaluation(split: str, selection: dict[str, Any]) -> dict[str, Any]:
    specifications = {
        "BASE": (1.0, 0.0),
        "PRE_STATIC": (1.0, -0.1),
        "PRE_FFN": (1.0, -0.1),
        "PRE_RECURRENT_T4": (1.0, -0.1),
        "POST_STATIC": (0.99, 0.10),
        "POST_FFN": (0.99, 0.10),
        "POST_RECURRENT_T1": (0.99, 0.08),
        "POST_RECURRENT_T2": (0.985, 0.10),
        "POST_RECURRENT_T4": (0.98, 0.12),
        "POST_RECURRENT_T8": (0.981, 0.11),
    }
    metrics = []
    for name, (nll, margin) in specifications.items():
        rows = [
            _metric(
                selected["task_key"],
                selected["expected_action"],
                nll=nll,
                margin=margin,
                recurrent_t8=name == "POST_RECURRENT_T8",
            )
            for selected in selection["selected"]
        ]
        metrics.append({"name": name, "rows": rows})
    residuals = [
        {
            "name": arm,
            "rows": [
                {"task_key": selected["task_key"], "finite_nonzero": True, "l2": 1.0}
                for selected in selection["selected"]
            ],
        }
        for arm in TRAINING_ARMS
    ]
    value = {
        "split": split,
        "rows": 24,
        "metrics": metrics,
        "common_arm_gates": [],
        "recurrent_gates": {},
        "post_residual_checks": residuals,
    }
    recomputed = runtime._recompute_evaluation(value, {arm: True for arm in TRAINING_ARMS})
    value["common_arm_gates"] = recomputed["common"]
    value["recurrent_gates"] = recomputed["recurrent"]
    return value


def _objective_evidence(action: str) -> dict[str, Any]:
    return {
        "action_weight": 2.0 if action == "delegate_coordinator" else 1.0,
        "retention_coefficient": 0.1,
        "branch_count": 1,
        "branches": [{"logit_offset": 0, "correct_token_id": 1, "other_token_ids": [2]}],
        "baseline_margins_detached": True,
    }


def _training(selection: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    histories = []
    evidence = []
    groups = {
        "STATIC": ["codec"],
        "FFN": ["codec", "ffn_internal"],
        "RECURRENT": ["codec", "transition", "memory", "workspace"],
    }
    for arm in TRAINING_ARMS:
        updates = []
        for update in selection["updates"]:
            rows = [
                {
                    "task_key": selected["task_key"],
                    "action": selected["expected_action"],
                    "aligned_suffix_ce": 1.0,
                    "total_objective": 1.0,
                    "objective_evidence": _objective_evidence(selected["expected_action"]),
                    "reporting_nll": 1.0,
                    "reporting_margin": 0.0,
                }
                for selected in update["rows"]
            ]
            updates.append(
                {
                    "update_index": update["update_index"],
                    "rows": rows,
                    "gradient_l2": [{"name": "codec.weight", "l2": 1.0}],
                    "preclip_global_norm": 1.0,
                    "sidecar_output_scale": [] if arm == "STATIC" else [0.01],
                }
            )
        histories.append({"name": arm, "updates": updates})
        evidence.append(
            {
                "name": arm,
                "value": {
                    "finite_nonzero_gradient_updates": {
                        name: [1, 2, 3, 4] if name == "codec" else [2, 3, 4] for name in groups[arm]
                    },
                    "optimizer_destroyed": True,
                },
            }
        )
    return histories, evidence


def _probe_metric() -> dict[str, Any]:
    metric = _metric("unused", "solve_owned", nll=1.0, margin=0.0)
    metric.pop("task_key")
    metric.pop("action")
    return metric


def _mechanism(selection: dict[str, Any]) -> dict[str, Any]:
    backward_for = {"STATIC": 0, "FFN": 1, "RECURRENT": 2}
    probes = []
    for index in (0, 1, 2, 5):
        selected = selection["selected"][index]
        arms = []
        for arm in TRAINING_ARMS:
            gradient = None
            if backward_for[arm] == index:
                gradient = {
                    "objective": 1.0,
                    "objective_evidence": _objective_evidence(selected["expected_action"]),
                    "residual": {"finite": True, "nonzero": True},
                    "codec": {"tensor_count": 1, "finite": True, "nonzero": True},
                    "sidecar": None if arm == "STATIC" else {"tensor_count": 1, "finite": True, "nonzero": True},
                    "all_named_present_gradients_finite": True,
                    "e33_gradients_absent": True,
                }
            arms.append(
                {
                    "name": arm,
                    "inplace_zero_identity": True,
                    "inplace_eps": _probe_metric(),
                    "backward": gradient,
                }
            )
        probes.append(
            {
                "selection_index": index,
                "task_key": selected["task_key"],
                "action": selected["expected_action"],
                "base_equals_direct_zero": True,
                "arms": arms,
            }
        )
    hashes = {
        arm: {
            "codec": hashlib.sha256(f"{arm}:codec".encode()).hexdigest(),
            "sidecar": None if arm == "STATIC" else hashlib.sha256(f"{arm}:sidecar".encode()).hexdigest(),
        }
        for arm in TRAINING_ARMS
    }
    return {"probes": probes, "pre_tensor_hashes": hashes, "post_tensor_hashes": deepcopy(hashes)}


def _render_proofs(split: str, selection: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for selected, row_hash in zip(selection["selected"], selection["row_canonical_sha256"], strict=True):
        rows.append(
            {
                "task_key": selected["task_key"],
                "action": selected["expected_action"],
                "source_row_sha256": row_hash,
                "reasoning_content_sha256": "6" * 64,
                "modified_path": "messages.2.tool_calls.0.function.arguments",
                "plain_ids_sha256": "7" * 64,
                "opening_ids_sha256": "8" * 64,
                "full_ids_sha256": "9" * 64,
                "plain_tokens": 10,
                "opening_tokens": 11,
                "full_tokens": 12,
                "counterfactual_target_sha256": {
                    action: hashlib.sha256(action.encode()).hexdigest() for action in ACTIONS
                },
                "action_trie_sha256": "a" * 64,
                "action_trie_branch_count": 1,
            }
        )
    return {"name": split, "rows": rows}


def _candidate_fixture(output: Path, arm: str, value: float) -> tuple[dict[str, Any], list[dict[str, str]]]:
    codec = {"weight": torch.tensor([value], dtype=torch.float32)}
    sidecar = None if arm == "STATIC" else {"weight": torch.tensor([value + 0.5], dtype=torch.float32)}
    states = runtime._candidate_state_records(codec, sidecar, torch=torch)
    payload = {
        "schema_version": "q35-2b-phase-b-ipc1-candidate/v1",
        "arm": arm,
        "codec": codec,
        "sidecar": sidecar,
        "module_post_state": states,
    }
    name = f"{arm}.final.pt"
    sha = runtime._exclusive_candidate_save(output, name, payload, torch=torch)
    candidate = {
        "name": name,
        "arm": arm,
        "sha256": sha,
        "valid_only_with_terminal": "SUCCESS.json",
        "module_post_state": states,
    }
    post = [{"name": f"{arm}.codec", "sha256": states[0]["tensor_sha256"]}]
    if arm != "STATIC":
        post.append({"name": f"{arm}.sidecar", "sha256": states[1]["tensor_sha256"]})
    return candidate, post


def _cache_class_records() -> list[dict[str, str]]:
    return [
        {
            "fqcn": fqcn,
            "module_path": f"/frozen/{suffix}",
            "module_sha256": module_sha256,
            "distribution": distribution,
        }
        for fqcn, suffix, module_sha256, distribution in runtime.EXPECTED_CACHE_CLASSES
    ]


def _success(plan: dict[str, Any], output: Path, execution_commit: str) -> dict[str, Any]:
    selections = {
        split: load_json_file(Path(runtime._bank_record(plan, split)["selection_path"]))
        for split in ("train", "validation", "heldout")
    }
    histories, training_evidence = _training(selections["train"])
    validation = _evaluation("validation", selections["validation"])
    heldout = _evaluation("heldout", selections["heldout"])
    candidates = []
    module_post = []
    for index, arm in enumerate(TRAINING_ARMS, start=1):
        candidate, states = _candidate_fixture(output, arm, float(index))
        candidates.append(candidate)
        module_post.extend(states)
    module_pre = [
        {"name": record["name"], "sha256": hashlib.sha256(f"pre:{record['name']}".encode()).hexdigest()}
        for record in module_post
    ]
    restored = [
        {
            "name": arm,
            "codec": next(record["sha256"] for record in module_pre if record["name"] == f"{arm}.codec"),
            "sidecar": None
            if arm == "STATIC"
            else next(record["sha256"] for record in module_pre if record["name"] == f"{arm}.sidecar"),
        }
        for arm in TRAINING_ARMS
    ]
    delta_names = {
        "STATIC": ["codec_encoder", "codec_receiver"],
        "FFN": ["codec_encoder", "codec_receiver", "ffn_internal", "ffn_output_scale"],
        "RECURRENT": [
            "codec_encoder",
            "codec_receiver",
            "recurrent_transition",
            "recurrent_memory",
            "recurrent_workspace",
            "recurrent_output_scale",
        ],
    }
    schedule = build_model_call_schedule(
        [row["task_key"] for row in selections["train"]["selected"]],
        [row["task_key"] for row in selections["validation"]["selected"]],
        [row["task_key"] for row in selections["heldout"]["selected"]],
        open_heldout=True,
    )
    cache_labels = build_cache_guard_labels(schedule)
    memory_labels = build_memory_checkpoint_labels(schedule)
    receipt = {
        "schema_version": "q35-2b-phase-b-ipc1-matched-learning-success/v1",
        "terminal": "SUCCESS",
        "status": "b_ipc1_inplace_learning_recurrent_nominated",
        "disposition": "b_ipc1_inplace_learning_recurrent_nominated",
        "claim_class": "nomination_only_inplace_carrier_matched_learning_screen",
        "execution_commit": execution_commit,
        "plan_sha256": plan["_file_sha256"],
        "run_identity": plan["run_identity"],
        "optimizer_steps": 12,
        "backward_calls": 147,
        "model_forwards": 796,
        "source_forwards": 96,
        "receiver_forwards": 700,
        "heldout_opened": True,
        "training": histories,
        "training_evidence": training_evidence,
        "pre_update_mechanism_gate": _mechanism(selections["train"]),
        "evaluations": [{"name": "validation", "value": validation}, {"name": "heldout", "value": heldout}],
        "nomination": {
            "status": "b_ipc1_inplace_learning_recurrent_nominated",
            "common_nominated_arms": list(TRAINING_ARMS),
            "recurrent_nominated": True,
            "admitted": False,
            "complete_live_trajectory_count": 0,
            "minimum_complete_live_trajectories_unchanged": 4,
        },
        "module_hashes": {
            "pre": module_pre,
            "restored_pre": restored,
            "post": module_post,
            "delta_groups": [
                {
                    "name": arm,
                    "groups": [
                        {"name": name, "finite": True, "nonzero": True, "delta_l2": 1.0} for name in delta_names[arm]
                    ],
                }
                for arm in TRAINING_ARMS
            ],
            "receiver_gates": [{"name": arm, "value": 0.01} for arm in TRAINING_ARMS],
        },
        "candidates": candidates,
        "cache_guard": {
            "complete": True,
            "labels": cache_labels,
            "label_count": len(cache_labels),
            "canonical_label_sha256": runtime.canonical_bank_sha256(cache_labels),
            "expected_label_sha256": runtime.canonical_bank_sha256(cache_labels),
            "exact_prefix": True,
            "exit_recorded": True,
            "dynamic_cache_trip_count": 1,
            "closure_check_count": len(cache_labels),
            "closure_checked_at_every_label": True,
            "restored_in_finally": True,
            "model_calls": 796,
            "recursively_closed_config_count": 1,
            "configs": [
                {
                    "position": 0,
                    "fqcn": "transformers.GenerationConfig",
                    "original_use_cache": True,
                    "current_use_cache": True,
                }
            ],
            "classes": _cache_class_records(),
        },
        "cuda_memory": {
            "cap_bytes": runtime.CUDA_MEMORY_CAP_BYTES,
            "allocator": {
                "device_total_bytes": 48 * 1024**3,
                "cap_bytes": runtime.CUDA_MEMORY_CAP_BYTES,
                "requested_fraction": 2 / 3,
                "observed_fraction": 2 / 3,
            },
            "ledger": [
                {
                    "checkpoint": label,
                    "current_allocated_bytes": 0,
                    "current_reserved_bytes": 0,
                    "maximum_allocated_bytes": 0,
                    "maximum_reserved_bytes": 0,
                }
                for label in memory_labels
            ],
            "ordered_label_sha256": runtime.canonical_bank_sha256(memory_labels),
        },
        "protection": {
            "e33_tensor_pre": "b" * 64,
            "e33_tensor_post": "b" * 64,
            "e33_file_pre": plan["protected_model"]["weight_sha256"],
            "e33_file_post": plan["protected_model"]["weight_sha256"],
            "metadata_pre": plan["model_metadata_sha256"],
            "metadata_post": plan["model_metadata_sha256"],
            "e33_gradients_absent": True,
        },
        "immutable_inputs": runtime._immutable_input_records(plan),
        "full_freeze": runtime._full_freeze_audit(plan, execution_commit=execution_commit),
        "render_proofs": [_render_proofs(split, selections[split]) for split in ("train", "validation", "heldout")],
        "bank_bindings": [
            {
                "name": bank["split"],
                "selection_sha256": bank["selection_sha256"],
                "parquet_sha256": bank["parquet_sha256"],
            }
            for bank in plan["banks"]
        ],
        "schedule_binding": {"name": "heldout_open", **plan["schedules"]["heldout_open"]},
        "antecedent_bindings": [
            {"name": item["name"], "binding_sha256": item["binding_sha256"]} for item in plan["antecedents"]
        ],
        "boundaries": {
            "generation": False,
            "cache": False,
            "H176_loaded": False,
            "strand_a_combined": False,
            "live_trajectory_count": 0,
            "admitted": False,
        },
        "elapsed_seconds": 1.0,
    }
    return receipt


def _failure(
    plan: dict[str, Any],
    execution_commit: str,
    pair: tuple[str, str],
    error_type: str,
    *,
    output: Path,
) -> dict[str, Any]:
    audit = {
        "audit_complete": True,
        "audit_errors": [],
        "immutable_inputs_preserved": True,
        "immutable_inputs": runtime._immutable_input_records(plan),
        "full_freeze": runtime._full_freeze_audit(plan, execution_commit=execution_commit),
        "e33_tensor_preserved": None,
        "e33_disk_preserved": None,
        "metadata_preserved": None,
        "e33_gradients_absent": None,
        "output_inventory": runtime._output_inventory(output),
        "candidate_files_present": [],
        "candidate_files_valid": False,
        "candidate_file_audits": [],
        "candidate_module_state": [],
        "candidate_module_state_sha256": runtime.canonical_bank_sha256([]),
        "candidate_initial_state": None,
        "candidate_initial_state_sha256": None,
        "cache_guard": None,
        "cuda_memory": None,
        "execution_progress": {
            "model_calls_completed": 0,
            "backward_calls_completed": 0,
            "optimizer_steps_completed": 0,
            "schedule_name": None,
            "completed_call_prefix_sha256": runtime.canonical_bank_sha256([]),
        },
    }
    return {
        "schema_version": "q35-2b-phase-b-ipc1-matched-learning-failure/v1",
        "terminal": "FAILURE",
        "status": pair[0],
        "disposition": pair[0],
        "failure_class": pair[1],
        "error_type": error_type,
        "error": "synthetic production writer schema proof",
        "execution_commit": execution_commit,
        "plan_sha256": plan["_file_sha256"],
        "run_identity": plan["run_identity"],
        "model_loaded": False,
        "candidate_files_valid": False,
        "candidate_files_present": audit["candidate_files_present"],
        "execution_breadcrumbs": {"stage": "schema_proof", "task_key": None, "arm": None, "call_index": None},
        "post_failure_audit": audit,
        "elapsed_seconds": 1.0,
    }


def _late_failure(
    plan: dict[str, Any], execution_commit: str, *, output: Path, post_candidate: bool
) -> dict[str, Any]:
    selections = {
        split: load_json_file(Path(runtime._bank_record(plan, split)["selection_path"]))
        for split in ("train", "validation", "heldout")
    }
    schedule = build_model_call_schedule(
        [row["task_key"] for row in selections["train"]["selected"]],
        [row["task_key"] for row in selections["validation"]["selected"]],
        [row["task_key"] for row in selections["heldout"]["selected"]],
        open_heldout=True,
    )
    initial = [
        {"name": name, "sha256": hashlib.sha256(f"initial:{name}".encode()).hexdigest()}
        for name in runtime.MODULE_NAMES
    ]
    current = [
        {"name": name, "sha256": hashlib.sha256(f"current:{name}".encode()).hexdigest()}
        for name in runtime.MODULE_NAMES
    ]
    if post_candidate:
        _candidate, states = _candidate_fixture(output, "STATIC", 1.0)
        current[0]["sha256"] = states[0]["sha256"]
    inventory = runtime._output_inventory(output)
    candidate_files = [record["name"] for record in inventory if record["name"].endswith(".pt")]
    cache_labels = build_cache_guard_labels(schedule)
    memory_labels = build_memory_checkpoint_labels(schedule)
    memory_stop = "candidate:STATIC:after_write" if post_candidate else "before_candidate_writes"
    memory_prefix = memory_labels[: memory_labels.index(memory_stop) + 1]
    audit = {
        "audit_complete": True,
        "audit_errors": [],
        "immutable_inputs_preserved": True,
        "immutable_inputs": runtime._immutable_input_records(plan),
        "full_freeze": runtime._full_freeze_audit(plan, execution_commit=execution_commit),
        "e33_tensor_preserved": True,
        "e33_disk_preserved": True,
        "metadata_preserved": True,
        "e33_gradients_absent": True,
        "output_inventory": inventory,
        "candidate_files_present": candidate_files,
        "candidate_files_valid": False,
        "candidate_file_audits": runtime._failure_candidate_audit(
            output, current_modules=current, torch=torch
        ),
        "candidate_module_state": current,
        "candidate_module_state_sha256": runtime.canonical_bank_sha256(current),
        "candidate_initial_state": initial,
        "candidate_initial_state_sha256": runtime.canonical_bank_sha256(initial),
        "cache_guard": {
            "complete": True,
            "labels": cache_labels,
            "label_count": len(cache_labels),
            "canonical_label_sha256": runtime.canonical_bank_sha256(cache_labels),
            "expected_label_sha256": runtime.canonical_bank_sha256(cache_labels),
            "exact_prefix": True,
            "exit_recorded": True,
            "dynamic_cache_trip_count": 1,
            "closure_check_count": len(cache_labels),
            "closure_checked_at_every_label": True,
            "restored_in_finally": True,
            "model_calls": len(schedule),
            "recursively_closed_config_count": 1,
            "configs": [
                {
                    "position": 0,
                    "fqcn": "transformers.GenerationConfig",
                    "original_use_cache": True,
                    "current_use_cache": True,
                }
            ],
            "classes": _cache_class_records(),
        },
        "cuda_memory": {
            "allocator": {
                "device_total_bytes": 48 * 1024**3,
                "cap_bytes": runtime.CUDA_MEMORY_CAP_BYTES,
                "requested_fraction": 2 / 3,
                "observed_fraction": 2 / 3,
            },
            "ledger": [
                {
                    "checkpoint": label,
                    "current_allocated_bytes": 0,
                    "current_reserved_bytes": 0,
                    "maximum_allocated_bytes": 0,
                    "maximum_reserved_bytes": 0,
                }
                for label in memory_prefix
            ],
            "current": {
                "current_allocated_bytes": 0,
                "current_reserved_bytes": 0,
                "maximum_allocated_bytes": 0,
                "maximum_reserved_bytes": 0,
            },
        },
        "execution_progress": {
            "model_calls_completed": len(schedule),
            "backward_calls_completed": 147,
            "optimizer_steps_completed": 12,
            "schedule_name": "heldout_open",
            "completed_call_prefix_sha256": runtime.canonical_bank_sha256(schedule),
        },
    }
    return {
        "schema_version": "q35-2b-phase-b-ipc1-matched-learning-failure/v1",
        "terminal": "FAILURE",
        "status": "b_ipc1_incomplete",
        "disposition": "b_ipc1_incomplete",
        "failure_class": "contract_or_evidence_incomplete",
        "error_type": "RuntimeError",
        "error": "synthetic late post-candidate failure" if post_candidate else "synthetic post-model pre-candidate failure",
        "execution_commit": execution_commit,
        "plan_sha256": plan["_file_sha256"],
        "run_identity": plan["run_identity"],
        "model_loaded": True,
        "candidate_files_valid": False,
        "candidate_files_present": candidate_files,
        "execution_breadcrumbs": {
            "stage": "candidate_write" if post_candidate else "final_audit",
            "task_key": None,
            "arm": "STATIC" if post_candidate else None,
            "call_index": len(schedule),
        },
        "post_failure_audit": audit,
        "elapsed_seconds": 1.0,
    }


def _reverse_mappings(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _reverse_mappings(item) for key, item in reversed(list(value.items()))}
    if isinstance(value, list):
        return [_reverse_mappings(item) for item in value]
    return value


def _truncate_cache_evidence(receipt: dict[str, Any]) -> None:
    cache = receipt["post_failure_audit"]["cache_guard"]
    labels = ["CACHE_GUARD_ENTRY", "CACHE_GUARD_EXIT"]
    cache.update(
        {
            "complete": False,
            "labels": labels,
            "label_count": len(labels),
            "canonical_label_sha256": runtime.canonical_bank_sha256(labels),
            "exact_prefix": True,
            "exit_recorded": True,
            "closure_check_count": len(labels),
            "closure_checked_at_every_label": True,
            "restored_in_finally": True,
        }
    )


def _truncate_memory_evidence(receipt: dict[str, Any]) -> None:
    receipt["post_failure_audit"]["cuda_memory"]["ledger"] = receipt["post_failure_audit"][
        "cuda_memory"
    ]["ledger"][:2]


def _expect_rejected(receipt: dict[str, Any], *, plan: dict[str, Any], execution_commit: str, output: Path) -> None:
    try:
        roundtrip_validate_terminal(
            receipt,
            validator=runtime.validate_success_receipt,
            validator_kwargs={"plan": plan, "execution_commit": execution_commit, "output_dir": output, "torch": torch},
        )
    except PhaseBContractError:
        return
    raise AssertionError("B-IPC1 terminal tamper unexpectedly validated")


def _expect_failure_rejected(
    receipt: dict[str, Any], *, plan: dict[str, Any], execution_commit: str, output: Path, label: str
) -> None:
    try:
        roundtrip_validate_terminal(
            receipt,
            validator=runtime.validate_failure_receipt,
            validator_kwargs={
                "plan": plan,
                "execution_commit": execution_commit,
                "output_dir": output,
                "torch": torch,
            },
        )
    except PhaseBContractError:
        return
    raise AssertionError(f"B-IPC1 late FAILURE tamper unexpectedly validated: {label}")


def main() -> int:
    args = _args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit("proof output must be fresh")
    if (
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True).stdout.strip()
        != args.execution_commit
    ):
        raise SystemExit("proof execution commit differs from HEAD")
    plan = load_json_file(args.plan)
    _relocate_repository_paths(plan, args.repository_root.resolve())
    plan["_path"] = str(args.plan)
    plan["_file_sha256"] = file_sha256(args.plan)
    freeze_manifest = args.repository_root.resolve() / runtime.FREEZE_MANIFEST.relative_to(runtime.WORKTREE)
    plan["_freeze_manifest_path"] = str(freeze_manifest)
    plan["_freeze_manifest_sha256"] = file_sha256(freeze_manifest)
    args.output_dir.mkdir(mode=0o700)
    runtime._fsync_directory(args.output_dir.parent)

    success_dir = args.output_dir / "maximal-success"
    success_dir.mkdir(mode=0o700)
    success = _success(plan, success_dir, args.execution_commit)
    parsed, payload, success_sha = roundtrip_validate_terminal(
        success,
        validator=runtime.validate_success_receipt,
        validator_kwargs={
            "plan": plan,
            "execution_commit": args.execution_commit,
            "output_dir": success_dir,
            "torch": torch,
        },
    )
    reversed_receipt = _reverse_mappings(success)
    _parsed_reversed, reversed_payload, _ = roundtrip_validate_terminal(
        reversed_receipt,
        validator=runtime.validate_success_receipt,
        validator_kwargs={
            "plan": plan,
            "execution_commit": args.execution_commit,
            "output_dir": success_dir,
            "torch": torch,
        },
    )
    if reversed_payload != payload:
        raise AssertionError("mapping insertion permutation changed canonical terminal bytes")

    tampers = []
    for name, mutate in (
        ("top_extra", lambda item: item.__setitem__("unexpected", True)),
        ("top_missing", lambda item: item.pop("claim_class")),
        ("wrong_update_index", lambda item: item["training"][0]["updates"][0].__setitem__("update_index", 2)),
        (
            "wrong_action",
            lambda item: item["training"][0]["updates"][0]["rows"][0].__setitem__("action", "delegate_terminal"),
        ),
        ("wrong_exposure", lambda item: item["training"][0]["updates"][0]["rows"][0].__setitem__("task_key", "reused")),
        (
            "objective_extra",
            lambda item: item["training"][0]["updates"][0]["rows"][0]["objective_evidence"].__setitem__("square", True),
        ),
        (
            "preprobe_backward",
            lambda item: item["pre_update_mechanism_gate"]["probes"][3]["arms"][0].__setitem__("backward", {}),
        ),
        ("render_extra", lambda item: item["render_proofs"][0]["rows"][0].__setitem__("unexpected", True)),
        (
            "aggregate_tamper",
            lambda item: item["evaluations"][0]["value"]["common_arm_gates"][0]["value"].__setitem__("passed", False),
        ),
        ("cache_count", lambda item: item["cache_guard"].__setitem__("model_calls", 795)),
        ("memory_order", lambda item: item["cuda_memory"]["ledger"].reverse()),
        ("immutable_hash", lambda item: item["immutable_inputs"][0].__setitem__("observed_sha256", "f" * 64)),
        ("candidate_filename", lambda item: item["candidates"][0].__setitem__("name", "../STATIC.final.pt")),
        (
            "candidate_state",
            lambda item: item["candidates"][0]["module_post_state"][0].__setitem__("tensor_sha256", "f" * 64),
        ),
        ("nomination", lambda item: item["nomination"].__setitem__("recurrent_nominated", False)),
    ):
        candidate = deepcopy(success)
        mutate(candidate)
        _expect_rejected(candidate, plan=plan, execution_commit=args.execution_commit, output=success_dir)
        tampers.append(name)

    success_path = runtime._atomic_publish_bytes(success_dir, "SUCCESS.json", payload)
    verify_published_terminal(
        success_path,
        payload,
        validator=runtime.validate_success_receipt,
        validator_kwargs={
            "plan": plan,
            "execution_commit": args.execution_commit,
            "output_dir": success_dir,
            "torch": torch,
        },
    )
    try:
        runtime._atomic_publish_bytes(success_dir, "FAILURE.json", canonical_terminal_bytes({"x": 1}))
    except (FileExistsError, PhaseBContractError):
        pass
    else:
        raise AssertionError("global terminal exclusivity did not reject a second terminal")

    failure_records = []
    failure_mapping_permutations_equal = []
    error_types = ["MechanismRejected", "CacheContractViolated", "PhaseBContractError", "ResourceContractExceeded"]
    for pair, error_type in zip(FAILURE_STATUS_CLASSES, error_types, strict=True):
        directory = args.output_dir / pair[0]
        directory.mkdir(mode=0o700)
        failure = _failure(plan, args.execution_commit, pair, error_type, output=directory)
        parsed_failure, failure_payload, failure_sha = roundtrip_validate_terminal(
            failure,
            validator=runtime.validate_failure_receipt,
            validator_kwargs={
                "plan": plan,
                "execution_commit": args.execution_commit,
                "output_dir": directory,
                "torch": torch,
            },
        )
        _parsed_reversed_failure, reversed_failure_payload, _ = roundtrip_validate_terminal(
            _reverse_mappings(failure),
            validator=runtime.validate_failure_receipt,
            validator_kwargs={
                "plan": plan,
                "execution_commit": args.execution_commit,
                "output_dir": directory,
                "torch": torch,
            },
        )
        if reversed_failure_payload != failure_payload:
            raise AssertionError("FAILURE mapping insertion permutation changed canonical terminal bytes")
        failure_mapping_permutations_equal.append(pair[0])
        path = runtime._atomic_publish_bytes(directory, "FAILURE.json", failure_payload)
        verify_published_terminal(
            path,
            failure_payload,
            validator=runtime.validate_failure_receipt,
            validator_kwargs={
                "plan": plan,
                "execution_commit": args.execution_commit,
                "output_dir": directory,
                "torch": torch,
            },
        )
        failure_records.append(
            {
                "status": pair[0],
                "failure_class": pair[1],
                "file_sha256": failure_sha,
                "internal_sha256": parsed_failure["receipt_sha256"],
            }
        )

    late_failure_records = []
    late_tampers = []
    for name, post_candidate in (("post-model-pre-candidate", False), ("late-post-candidate", True)):
        directory = args.output_dir / name
        directory.mkdir(mode=0o700)
        failure = _late_failure(
            plan,
            args.execution_commit,
            output=directory,
            post_candidate=post_candidate,
        )
        tamper_specs = [
            ("full_freeze", lambda item: item["post_failure_audit"]["full_freeze"].__setitem__("complete", False)),
            (
                "module_initial",
                lambda item: item["post_failure_audit"]["candidate_initial_state"][0].__setitem__(
                    "sha256", "f" * 64
                ),
            ),
            (
                "cache_prefix",
                lambda item: item["post_failure_audit"]["cache_guard"]["labels"].reverse(),
            ),
            (
                "memory_prefix",
                lambda item: item["post_failure_audit"]["cuda_memory"]["ledger"].reverse(),
            ),
            (
                "progress",
                lambda item: item["post_failure_audit"]["execution_progress"].__setitem__(
                    "model_calls_completed", 795
                ),
            ),
            ("cache_truncated_against_progress", _truncate_cache_evidence),
            ("memory_truncated_against_progress", _truncate_memory_evidence),
            (
                "backward_progress",
                lambda item: item["post_failure_audit"]["execution_progress"].__setitem__(
                    "backward_calls_completed", 0
                ),
            ),
            (
                "optimizer_progress",
                lambda item: item["post_failure_audit"]["execution_progress"].__setitem__(
                    "optimizer_steps_completed", 0
                ),
            ),
        ]
        if post_candidate:
            tamper_specs.extend(
                [
                    (
                        "candidate_inventory",
                        lambda item: item["post_failure_audit"]["output_inventory"][0].__setitem__(
                            "sha256", "f" * 64
                        ),
                    ),
                    (
                        "candidate_safe_load",
                        lambda item: item["post_failure_audit"]["candidate_file_audits"][0].__setitem__(
                            "scientific_valid", True
                        ),
                    ),
                    (
                        "module_current",
                        lambda item: item["post_failure_audit"]["candidate_module_state"][0].__setitem__(
                            "sha256", "f" * 64
                        ),
                    ),
                ]
            )
        for suffix, mutate in tamper_specs:
            candidate = deepcopy(failure)
            mutate(candidate)
            _expect_failure_rejected(
                candidate,
                plan=plan,
                execution_commit=args.execution_commit,
                output=directory,
                label=f"{name}:{suffix}",
            )
            late_tampers.append(f"{name}:{suffix}")
        parsed_failure, failure_payload, failure_sha = roundtrip_validate_terminal(
            failure,
            validator=runtime.validate_failure_receipt,
            validator_kwargs={
                "plan": plan,
                "execution_commit": args.execution_commit,
                "output_dir": directory,
                "torch": torch,
            },
        )
        _parsed_reversed_failure, reversed_failure_payload, _ = roundtrip_validate_terminal(
            _reverse_mappings(failure),
            validator=runtime.validate_failure_receipt,
            validator_kwargs={
                "plan": plan,
                "execution_commit": args.execution_commit,
                "output_dir": directory,
                "torch": torch,
            },
        )
        if reversed_failure_payload != failure_payload:
            raise AssertionError("late FAILURE mapping insertion permutation changed canonical terminal bytes")
        failure_mapping_permutations_equal.append(name)
        path = runtime._atomic_publish_bytes(directory, "FAILURE.json", failure_payload)
        verify_published_terminal(
            path,
            failure_payload,
            validator=runtime.validate_failure_receipt,
            validator_kwargs={
                "plan": plan,
                "execution_commit": args.execution_commit,
                "output_dir": directory,
                "torch": torch,
            },
        )
        late_failure_records.append(
            {
                "name": name,
                "file_sha256": failure_sha,
                "internal_sha256": parsed_failure["receipt_sha256"],
                "model_loaded": True,
                "candidate_files_present": parsed_failure["candidate_files_present"],
            }
        )

    proof = {
        "schema_version": "q35-2b-phase-b-ipc1-terminal-proof/v1",
        "execution_commit": args.execution_commit,
        "runner_sha256": file_sha256(Path(runtime.__file__)),
        "plan_file_sha256": plan["_file_sha256"],
        "maximal_success": {
            "status": parsed["status"],
            "file_sha256": success_sha,
            "internal_sha256": parsed["receipt_sha256"],
        },
        "failure_terminals": failure_records,
        "late_failure_terminals": late_failure_records,
        "tamper_cases_rejected": tampers,
        "late_failure_tamper_cases_rejected": late_tampers,
        "full_freeze_target_count": success["full_freeze"]["target_count"],
        "mapping_insertion_permutation_canonical_equal": True,
        "failure_mapping_insertion_permutation_canonical_equal": (
            failure_mapping_permutations_equal
            == [
                "b_ipc1_mechanism_rejected",
                "b_ipc1_nocache_rejected",
                "b_ipc1_incomplete",
                "infrastructure_invalid",
                "post-model-pre-candidate",
                "late-post-candidate",
            ]
        ),
        "prepublish_validation": True,
        "postpublish_reopen_byte_compare_hash_parse_validation": True,
        "global_exactly_one_terminal": True,
        "model_loaded": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "repository_root": str(args.repository_root.resolve()),
        "exact_host_repository": args.repository_root.resolve() == runtime.WORKTREE,
    }
    proof_payload = canonical_terminal_bytes(proof)
    proof_path = args.output_dir / "PROOF.json"
    with proof_path.open("xb") as handle:
        handle.write(proof_payload)
        handle.flush()
        os.fsync(handle.fileno())
    runtime._fsync_directory(args.output_dir)
    runtime._fsync_directory(args.output_dir.parent)
    print(json.dumps({**proof, "proof_file_sha256": file_sha256(proof_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
