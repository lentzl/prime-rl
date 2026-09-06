from __future__ import annotations

import hashlib
import json
import math
from typing import Any

MECHANISM = "q35-2b-h-iter-phase1-cap0-v1"
RUN_ID = "h-iter-phase1-cap0-capture-run1"
OUTPUT_ROOT = "/home/ubuntu/rlm/outputs/q35-2b-h-iter-phase1-cap0-capture-run1"
ARTIFACT_DIR = "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-cap0-v1"
PLAN_SCHEMA = "prime-rl/latent-h-iter-phase1-cap0-plan/v1"
CONTRACT_SCHEMA = "prime-rl/latent-h-iter-phase1-cap0-contract/v1"
PREFLIGHT_SCHEMA = "prime-rl/latent-h-iter-phase1-cap0-preflight/v1"
PROOF_SCHEMA = "prime-rl/latent-h-iter-phase1-cap0-proof/v1"
FAILURE_SCHEMA = "prime-rl/latent-h-iter-phase1-cap0-failure/v1"
PREFLIGHT_STATUS = "h_iter_phase1_cap0_preflight_validated"
PROOF_STATUS = "h_iter_phase1_cap0_capture_mechanism_validated"
REJECT_STATUS = "h_iter_phase1_cap0_capture_mechanism_rejected"
INCOMPLETE_STATUS = "h_iter_phase1_cap0_incomplete"
EXPOSURE_STATUS = "h_iter_phase1_cap0_exposure_boundary_rejected"
INFRASTRUCTURE_STATUS = "infrastructure_invalid"

REPO_ROOT = "/home/ubuntu/rlm/worktrees/q35-2b-latent-workspace-v1"
SHARED_PROJECT = "/home/ubuntu/rlm/prime-rl"
SHARED_PYTHON = "/home/ubuntu/rlm/prime-rl/.venv/bin/python3"
E33_PATH = "/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2"
H176_PATH = "/home/ubuntu/rlm/outputs/q35-2b-document-child-sft-v1/h176child8-document-child-real12-step8-v2/weights/step_8"
E33_TREE_SHA256 = "e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47"
H176_TREE_SHA256 = "77980e247bbccd6463ddda02cd42d2c357e15f8ec1ad0ea84627e008a8674a1e"
E33_STATE_SHA256 = "dd6a76377c6e43a28efe484927e0a8427026cc3517fac0aea5dd9d6972cc1bf9"
METADATA_SHA256 = {
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "config.json": "22949388ed61c1100b20a3cae55bb22122554c74e06fc23f1be50cca1fec3b8c",
    "generation_config.json": "93f19a5ed0fb9f9e8e65dafae7a9bc4c6a32b3e37f6278980d05d3f4ca29f17b",
    "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "747ba36a06ba5428bb74e984d75136b37cf5dafe97b8dd315f701b361a9f417f",
}
RUNTIME = {
    "python": "3.12.14",
    "transformers": "5.6.2",
    "tokenizers": "0.22.2",
    "flash_linear_attention": "0.5.2",
    "torch_distribution": "2.11.0+cu128",
    "torch_runtime": "2.11.0+cu128",
    "shared_project_pyproject_sha256": "504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656",
    "shared_project_uv_lock_sha256": "fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5",
    "sys_executable": SHARED_PYTHON,
    "sys_prefix": f"{SHARED_PROJECT}/.venv",
    "model_class": "Qwen3_5ForConditionalGeneration",
    "hidden_size": 2048,
    "vocab_size": 248320,
    "gpu_model": "NVIDIA RTX A6000",
}
MF0_BINDING = {
    "archival_freeze_commit": "4087ecde6da743f1a248bf99493264ecac459c63",
    "evidence_commit": "197fb0ba67273015c9db98b52f230c875c745ca9",
    "manifest_file_sha256": "79caa566a74bd73ef4b56002f67f9584c5ac76d2521ab96b13afa6ad07aa0140",
    "manifest_internal_sha256": "c0a9034efe192a93efd3d755e0769e2dfadc2745b5772c8955fc43f319fa9758",
    "proof_file_sha256": "7b1f99f06adbc1282511a0e05306304e0b44ffc9c7411eaad8963543c67fa6fc",
    "proof_internal_sha256": "7101a4f19783911567c9b301dfd416cdcaac7b763d1d80f46355821367898a06",
    "launcher_log_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "exit_file_sha256": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "proof_status": "h_iter_phase1_train_calibration_preregistered",
}
SELECTION_SHA256 = "eb63e5984ddcbbc57e319048e0e5df160f0923272c785adebdd8351cd100f946"
RESOURCE_BOUNDS = {
    "minimum_gpu_free_gib": 44,
    "allocator_cap_gib": 40,
    "minimum_host_ram_gib": 64,
    "minimum_free_disk_gib": 16,
    "maximum_terminal_bytes": 16 * 2**20,
    "contract_compute_seconds": 3000,
    "contract_audit_seconds": 240,
    "contract_failure_seconds": 180,
    "compute_timeout_seconds": 2700,
    "audit_timeout_seconds": 180,
    "failure_timeout_seconds": 120,
    "terminal_timeout_seconds": 60,
    "outer_timeout_seconds": 3600,
    "startup_seconds": 120,
    "postexit_seconds": 60,
    "success_terminal_entry_maximum_seconds": 2880,
    "compute_failure_terminal_entry_maximum_seconds": 2820,
    "audit_failure_terminal_entry_maximum_seconds": 3000,
    "prior_terminal_failure_entry_maximum_seconds": 3060,
    "worst_external_seconds": 3300,
    "reserve_seconds": 300,
    "output_root": OUTPUT_ROOT,
}
CALL_NAMES = [f"CAP0_P{probe:02d}_R{repeat}" for probe in range(1, 5) for repeat in (1, 2)]
CACHE_LABELS = ["CACHE_ENTRY", *[label for call in CALL_NAMES for label in (f"CACHE_PRE_{call}", f"CACHE_POST_{call}")], "CACHE_EXIT"]
MEMORY_LABELS = [
    "runtime_verified", "full_freeze_preflight_verified", "mf0_archive_binding_validated",
    "train_bank_and_selection_validated", "protected_disk_preflight_verified", "model_loaded_frozen",
    "cache_guard_entered", *[label for call in CALL_NAMES for label in (f"pre_{call}", f"post_{call}")],
    "cache_guard_audit_complete", "protected_postflight_complete", "model_released",
    "full_freeze_postflight_validated", "proof_prewrite_ready",
]
TAMPERS = [
    "mf0_archive_manifest_file_hash_changed", "mf0_archive_manifest_internal_hash_changed",
    "mf0_archive_proof_file_hash_changed", "mf0_archive_proof_internal_hash_changed",
    "mf0_archive_log_hash_changed", "mf0_archive_exit_hash_changed", "train_bank_hash_changed",
    "selection_hash_changed", "probe_order_changed", "probe_not_fit_changed", "node_order_changed",
    "local_text_changed", "answer_or_supervision_added", "tokenizer_max_length_changed",
    "tokenizer_truncation_enabled", "tokenizer_padding_side_changed", "model_checkpoint_path_changed",
    "model_checkpoint_hash_changed", "model_metadata_hash_changed", "model_dtype_changed",
    "attention_implementation_changed", "inference_mode_disabled", "use_cache_enabled",
    "output_hidden_states_disabled", "logits_to_keep_changed", "cache_class_removed",
    "cache_source_hash_changed", "dynamic_cache_negative_control_removed", "pkv_none_gate_disabled",
    "repeat_count_changed", "repeat_hidden_parity_gate_disabled", "finite_gate_disabled",
    "node_diversity_gate_disabled", "model_forward_count_changed", "h176_load_allowed",
    "model_update_allowed", "candidate_or_optimizer_allowed", "validation_path_added",
    "heldout_path_added", "resource_or_memory_cap_changed", "proof_status_changed",
    "proof_self_hash_changed",
]
MECHANISM_CAUSES = {
    "cache_allocation_detected", "cache_configuration_drift", "returned_pkv_non_none", "nonfinite_output",
    "repeat_parity_failed", "node_diversity_failed",
}
COUNTS = {
    "train_rows_read": 96, "selected_rows": 4, "nodes_per_probe": 24,
    "tokenizer_calls": 4, "model_forwards": 8, "sequences": 192, "backwards": 0,
    "optimizer_objects": 0, "optimizer_steps": 0, "candidate_objects": 0,
    "checkpoint_files": 0, "validation_opens": 0, "heldout_opens": 0,
    "cache_checks": 18, "tampers": 42, "memory_rows": 28,
}
DECISION = {
    "admission": False, "cap0_capture_mechanism_validated": True,
    "claim": "train_only_frozen_e33_local_feature_capture_mechanism",
    "four_live_floor_unchanged": True, "live_trajectory_count": 0, "model_updated": False,
    "nomination": False, "promotion": False, "t0_authorized": False,
    "training_authorized": False, "validation_or_heldout_opened": False,
}


class CAP0ContractError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CAP0ContractError("CAP0 value is not canonical JSON") from error


def strict_loads(data: bytes) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise CAP0ContractError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(CAP0ContractError(f"nonfinite JSON value: {value}")))
    except CAP0ContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CAP0ContractError("invalid CAP0 JSON") from error
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finish(value: dict[str, Any], field: str) -> dict[str, Any]:
    value[field] = sha256_bytes(canonical_json({key: item for key, item in value.items() if key != field}))
    return value


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def build_contract(selection: dict[str, Any], capture: dict[str, Any]) -> dict[str, Any]:
    return finish({
        "schema_version": CONTRACT_SCHEMA,
        "status": "cap0_contract_preregistered",
        "mechanism": MECHANISM,
        "mf0_archive_binding": MF0_BINDING,
        "selection": {"selection_sha256": SELECTION_SHA256, "ordered_probes": selection["ordered_probes"]},
        "tokenizer_contract": {"source": "e33", "calls": 4, "sequences": 192, "local_text_utf8_bytes": 68, "add_special_tokens": True, "padding": "max_length", "padding_side": "left", "max_length": 128, "truncation": False, "return_tensors": "pt", "input_shape": [24, 128], "dtype": "torch.int64"},
        "model_contract": {"checkpoint": E33_PATH, "checkpoint_tree_sha256": E33_TREE_SHA256, "loader": "AutoModelForImageTextToText.from_pretrained", "local_files_only": True, "torch_dtype": "torch.bfloat16", "attn_implementation": "eager", "device": "cuda:0", "eval": True, "inference_mode": True, "requires_grad": False, "forward_kwargs": {"use_cache": False, "output_hidden_states": True, "return_dict": True, "logits_to_keep": 1}, "forwards": 8, "repeat_order": CALL_NAMES},
        "output_contract": {"logits_shape": [24, 1, 248320], "full_hidden_shape": [24, 128, 2048], "capture_shape": [24, 2048], "dtype": "torch.bfloat16", "finite": True, "pkv_none": True, "repeat_input_mask_hidden_capture_bitwise": True, "logit_repeat_equality_descriptive_only": True, "unique_capture_rows_minimum": 2},
        "cache_contract": {"class_closure": capture["cache_guard"]["class_closure"], "check_labels": CACHE_LABELS, "checks": 18, "mandatory_negative_trips": 1, "actual_allocations": 0, "config_restored": True},
        "protected_contract": {"e33_path": E33_PATH, "e33_tree_sha256": E33_TREE_SHA256, "e33_state_sha256": E33_STATE_SHA256, "h176_path": H176_PATH, "h176_tree_sha256": H176_TREE_SHA256, "metadata_sha256": METADATA_SHA256, "h176_loaded": False, "model_updated": False, "grads_none": True},
        "resource_bounds": RESOURCE_BOUNDS,
        "memory_label_schedule": {"labels": MEMORY_LABELS, "count": 28, "label_sha256": sha256_bytes(canonical_json(MEMORY_LABELS))},
        "tamper_schedule": TAMPERS,
        "safety_boundary": {"train_only": True, "validation_opens": 0, "heldout_opens": 0, "generation": False, "backward": False, "optimizer": False, "candidate": False, "checkpoint": False, "update": False, "network_attempts": 0},
        "decision_boundary": DECISION,
        "contract_sha256": "",
    }, "contract_sha256")


def validate_contract(value: dict[str, Any], selection: dict[str, Any], capture: dict[str, Any]) -> None:
    if value != build_contract(selection, capture):
        raise CAP0ContractError("CAP0 contract differs")


def validate_memory(rows: Any, complete: bool) -> None:
    if not isinstance(rows, list) or len(rows) > len(MEMORY_LABELS):
        raise CAP0ContractError("CAP0 memory rows differ")
    labels = [row.get("label") for row in rows if isinstance(row, dict)]
    if labels != MEMORY_LABELS[: len(rows)] or complete and labels != MEMORY_LABELS:
        raise CAP0ContractError("CAP0 memory label prefix differs")
    keys = {"label", "rss_bytes", "peak_rss_bytes", "allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes"}
    last_rss = last_alloc = last_reserved = -1
    cap = 40 * 2**30
    for row in rows:
        if set(row) != keys:
            raise CAP0ContractError("CAP0 memory row schema differs")
        for key in keys - {"label"}:
            if not isinstance(row[key], int) or isinstance(row[key], bool) or row[key] < 0:
                raise CAP0ContractError("CAP0 memory value differs")
        if row["peak_rss_bytes"] < row["rss_bytes"] or row["peak_allocated_bytes"] < row["allocated_bytes"] or row["peak_reserved_bytes"] < row["reserved_bytes"]:
            raise CAP0ContractError("CAP0 memory peak dominance differs")
        if row["peak_rss_bytes"] < last_rss or row["peak_allocated_bytes"] < last_alloc or row["peak_reserved_bytes"] < last_reserved:
            raise CAP0ContractError("CAP0 memory peak monotonicity differs")
        if any(row[key] > cap for key in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes")):
            raise CAP0ContractError("CAP0 memory cap exceeded")
        last_rss, last_alloc, last_reserved = row["peak_rss_bytes"], row["peak_allocated_bytes"], row["peak_reserved_bytes"]


def tensor_evidence_valid(value: dict[str, Any], shape: list[int]) -> bool:
    return value.get("shape") == shape and value.get("dtype") == "torch.bfloat16" and value.get("finite") is True and isinstance(value.get("sha256"), str) and len(value["sha256"]) == 64
