from __future__ import annotations

import json
import math
import re
from pathlib import Path

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a1cap768 import (
    _CACHE_CLASS_CLOSURE,
    _E33,
    _H176,
    _RUNTIME,
)
from prime_rl.latent.a1cap768 import (
    ASSET_PATHS as CAP_ASSET_PATHS,
)
from prime_rl.latent.a1cap768 import (
    RESOURCE_BOUNDS as CAP_RESOURCE_BOUNDS,
)

PLAN_SCHEMA = "prime-rl/latent-a1-nc0-cap768-flag-isolation-plan/v1"
RECEIPT_SCHEMA = "prime-rl/latent-a1-nc0-cap768-flag-isolation-receipt/v1"
FAILURE_SCHEMA = "prime-rl/latent-a1-nc0-cap768-flag-isolation-failure/v1"
AUTHORIZED_RUN_ID = "a1-nc0-cap768-flag-isolation-run1"

FIXTURE = {
    "family": "keyed_numeric",
    "evidence_id": "train-keyed_numeric-685221bdf4d3058c",
    "query_id": "train-keyed_numeric-685221bdf4d3058c-q1",
    "modality": "PARENT",
    "unpadded_tokens": 517,
    "padded_tokens": 768,
    "left_pad_tokens": 251,
    "capture_indices": [640, 768],
    "capture_shape": [1, 128, 2048],
}
FIXTURE_SHA256 = "2ebbd769a6e50cf4cc41ac99a020497be3e0902f78e79ae0bca131224fee89c9"
TRAIN_BANK_SHA256 = "cea92c57536ec7a93e68c64c4a01669e11dab1dc483a86c711c97c1181bb08d8"

OPERATION_SCHEDULE = [
    {"operation_index": 1, "kind": "embedding_lookup", "name": "CAP768_FLAG_P01_PARENT_EXACT_EMBED_LOOKUP"},
    {"operation_index": 2, "kind": "model_forward", "name": "CAP768_FLAG_P01_PARENT_L_ID_KEEP1"},
    {"operation_index": 3, "kind": "model_forward", "name": "CAP768_FLAG_P01_PARENT_L_E_KEEP1"},
    {"operation_index": 4, "kind": "model_forward", "name": "CAP768_FLAG_P01_PARENT_L_E_REPEAT_KEEP1"},
    {"operation_index": 5, "kind": "model_forward", "name": "CAP768_FLAG_P01_PARENT_L_ID_KEEP0_CONTROL"},
    {"operation_index": 6, "kind": "lm_head_projection", "name": "CAP768_FLAG_P01_PARENT_PROJ_ID1_LAST"},
    {"operation_index": 7, "kind": "lm_head_projection", "name": "CAP768_FLAG_P01_PARENT_PROJ_ID0_LAST"},
]
OPERATION_SCHEDULE_SHA256 = "bf211a9dbfae00d76f4abf0dc5be5728d01c6a46916c726e5827edc955c9276a"

FLAG_NAMES = [
    "left_padding_exact",
    "attention_mask_exact",
    "position_ids_exact",
    "no_truncation",
    "id_embed_keep1_logits_bitwise",
    "id_embed_keep1_full_hidden_bitwise",
    "id_embed_keep1_capture_bitwise",
    "repeat_same_embedding_object",
    "repeat_embedding_unchanged",
    "repeat_logits_bitwise",
    "repeat_full_hidden_bitwise",
    "repeat_capture_bitwise",
    "keep0_keep1_full_hidden_bitwise",
    "keep0_keep1_capture_bitwise",
    "keep0_last_logits_keep1_bitwise",
    "all_outputs_finite",
    "all_output_logits_finite",
    "all_output_full_hidden_finite",
    "all_capture_finite",
    "exact_embeddings_finite",
    "exact_embeddings_requires_grad_false",
    "proj_id1_matches_id1_logits_bitwise",
    "proj_id0_matches_id0_last_logits_bitwise",
    "proj_id1_proj_id0_bitwise",
    "id1_logits_proj_id0_bitwise",
]
FLAG_NAMES_SHA256 = "7527bc4baab849eb819c30d209e7ade244c456166d418987a588c7c959fbe3ee"
RUN4_FLAG_NAMES = FLAG_NAMES[:16]
RUN4_FLAG_NAMES_SHA256 = "2ba84a6ce26dbf74fd200086bb58fa7302fe0c222b83cfeb181ce34c09647969"

COMPARISON_SCHEDULE = [
    {"comparison_index": 1, "name": "id_embed_keep1_logits", "lhs": "L_ID_KEEP1.logits", "rhs": "L_E_KEEP1.logits"},
    {
        "comparison_index": 2,
        "name": "id_embed_keep1_full_hidden",
        "lhs": "L_ID_KEEP1.hidden",
        "rhs": "L_E_KEEP1.hidden",
    },
    {"comparison_index": 3, "name": "id_embed_keep1_capture", "lhs": "L_ID_KEEP1.capture", "rhs": "L_E_KEEP1.capture"},
    {"comparison_index": 4, "name": "repeat_logits", "lhs": "L_E_KEEP1.logits", "rhs": "L_E_REPEAT_KEEP1.logits"},
    {"comparison_index": 5, "name": "repeat_full_hidden", "lhs": "L_E_KEEP1.hidden", "rhs": "L_E_REPEAT_KEEP1.hidden"},
    {"comparison_index": 6, "name": "repeat_capture", "lhs": "L_E_KEEP1.capture", "rhs": "L_E_REPEAT_KEEP1.capture"},
    {
        "comparison_index": 7,
        "name": "keep0_keep1_full_hidden",
        "lhs": "L_ID_KEEP1.hidden",
        "rhs": "L_ID_KEEP0_CONTROL.hidden",
    },
    {
        "comparison_index": 8,
        "name": "keep0_keep1_capture",
        "lhs": "L_ID_KEEP1.capture",
        "rhs": "L_ID_KEEP0_CONTROL.capture",
    },
    {
        "comparison_index": 9,
        "name": "keep0_last_logits_keep1",
        "lhs": "L_ID_KEEP1.logits",
        "rhs": "L_ID_KEEP0_CONTROL.last_logits",
    },
    {
        "comparison_index": 10,
        "name": "proj_id1_matches_id1_logits",
        "lhs": "PROJ_ID1_LAST.logits",
        "rhs": "L_ID_KEEP1.logits",
    },
    {
        "comparison_index": 11,
        "name": "proj_id0_matches_id0_last_logits",
        "lhs": "PROJ_ID0_LAST.logits",
        "rhs": "L_ID_KEEP0_CONTROL.last_logits",
    },
    {"comparison_index": 12, "name": "proj_id1_proj_id0", "lhs": "PROJ_ID1_LAST.logits", "rhs": "PROJ_ID0_LAST.logits"},
    {"comparison_index": 13, "name": "id1_logits_proj_id0", "lhs": "L_ID_KEEP1.logits", "rhs": "PROJ_ID0_LAST.logits"},
]
COMPARISON_SCHEDULE_SHA256 = "f8df624428819cd5dd5a6ee234cc0407bb6b97ebe3cb9ab5f916ab79de66cfc5"


def memory_labels() -> list[str]:
    labels = ["model_loaded_frozen"]
    for operation in OPERATION_SCHEDULE:
        labels.extend([f"pre_{operation['name']}", f"post_{operation['name']}"])
    labels.extend(["cache_guard_audit_complete", "protected_postflight_complete"])
    if len(labels) != 17 or len(set(labels)) != 17:
        raise ValueError("FLAG0 memory schedule changed")
    return labels


MEMORY_LABELS_SHA256 = "02fbf26af98251d66e96841fb069321a0c5ad6e4ccd9ab44515b99bff8231ade"

RESOURCE_BOUNDS = {
    **CAP_RESOURCE_BOUNDS,
    "output_root": "/home/ubuntu/rlm/outputs/latent-a1-nc0-cap768-flag-isolation-v1",
}
INTERPRETATION = (
    "capture768 flag isolation only; run4 remains rejected, CAP768 is not validated, A1-NC0 remains blocked, "
    "and no training, nomination, admission, promotion, or four-live-floor change is authorized"
)
DECISION_BOUNDARY = {
    "claim": "capture768_flag_isolation_only",
    "training_authorized": False,
    "CAP768_validated": False,
    "A1_NC0_unblocked": False,
    "nomination": False,
    "admission": False,
    "promotion": False,
    "live_trajectories": 0,
    "four_live_floor_unchanged": True,
    "run4_remains_rejected": True,
}

RUN4_REJECTION_EVIDENCE = {
    "failure_file_sha256": "c2e2219fdf4753496871f5a8b56bbd37875e0b3988279d12b53a3bb2f8d45723",
    "failure_internal_sha256": "e9887d9f1b894dab09f1b88ab56bc26ac71001f56d0022a8bdfec253db69c9a0",
    "launch_log_sha256": "bc50647f012fa8fbfb2a08fc5daa8aa76bfb3f53271742d682dda8d5aa1a90d6",
    "status": "capture768_mechanism_rejected",
    "failure_category": "cache_pkv_parity_repeat_capture_geometry_or_finiteness",
    "error_type": "CaptureMechanismRejected",
    "exact_error": "CAP768 parity/repeat/capture predicate rejected",
    "run_id": "a1-nc0-cap768-run4",
    "execution_commit": "1cbd298e3bb65de38b96c1fa1bafd2e0504bd8e3",
    "mechanism_code_commit": "03df4ce5465373cb529105f97b767306cb738ae7",
    "plan_file_sha256": "f23edabbb4214de6ef84546b9222eb3cf6b16f9e144c501ab9d29eea4e056292",
    "plan_internal_sha256": "05414dc4df70326f5571646209c086aa54709db9e7763582605e9cfcd7da4206",
    "model_loaded": True,
    "memory_labels": [
        "model_loaded_frozen",
        "pre_CAP768_P01_PARENT_L_ID_KEEP1",
        "post_CAP768_P01_PARENT_L_ID_KEEP1",
        "pre_CAP768_P01_PARENT_L_E_KEEP1",
        "post_CAP768_P01_PARENT_L_E_KEEP1",
        "pre_CAP768_P01_PARENT_L_E_REPEAT_KEEP1",
        "post_CAP768_P01_PARENT_L_E_REPEAT_KEEP1",
        "pre_CAP768_P01_PARENT_L_ID_KEEP0_CONTROL",
        "post_CAP768_P01_PARENT_L_ID_KEEP0_CONTROL",
    ],
    "cache_closure_check_count": 10,
    "cache_guard_restored": True,
    "protected_disk_state_metadata_exact": True,
    "e33_gradients_absent": True,
    "worker_h176_loaded": False,
    "model_update_attempted": False,
    "bridge_created": False,
    "optimizer_created": False,
    "backward_used": False,
    "checkpoint_created": False,
    "candidate_created": False,
    "asset_count": 35,
    "failure_audit_errors": [],
    "snapshot_manifest_exists": False,
}

ASSET_PATHS = set(CAP_ASSET_PATHS) | {
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-run4-mechanism-rejection-failure.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-run4-mechanism-rejection-run.log",
    "scripts/latent/run_a1_nc0_cap768_flag0_v1.py",
    "scripts/latent/run_a1_nc0_cap768_flag0_v1.sh",
    "src/prime_rl/latent/a1cap768_flag0.py",
    "tests/unit/latent/test_a1cap768_flag0.py",
}

_SHA = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


class DiagnosticIncomplete(RuntimeError):
    pass


class NoCacheRejected(RuntimeError):
    pass


class ResourceFitRejected(RuntimeError):
    pass


def causal_interpretation(flags: dict[str, bool]) -> str:
    core = (
        "id_embed_keep1_full_hidden_bitwise",
        "id_embed_keep1_capture_bitwise",
        "repeat_full_hidden_bitwise",
        "repeat_capture_bitwise",
        "keep0_keep1_full_hidden_bitwise",
        "keep0_keep1_capture_bitwise",
    )
    if (
        all(flags[name] for name in core)
        and flags["proj_id1_proj_id0_bitwise"]
        and flags["proj_id1_matches_id1_logits_bitwise"]
        and flags["id1_logits_proj_id0_bitwise"]
        and not flags["proj_id0_matches_id0_last_logits_bitwise"]
        and not flags["keep0_last_logits_keep1_bitwise"]
    ):
        return "bf16_lm_head_shape_rounding_redesign_discussion_only"
    if not all(flags[name] for name in core):
        return "capture768_carrier_support_absent_under_flag0"
    if all(flags[name] for name in RUN4_FLAG_NAMES):
        return "run4_failure_nonreproduced_no_retry"
    return "other_run4_flag_pattern_no_authorization"


def _validate_constants() -> None:
    checks = {
        "fixture": (FIXTURE, FIXTURE_SHA256),
        "operations": (OPERATION_SCHEDULE, OPERATION_SCHEDULE_SHA256),
        "flags": (FLAG_NAMES, FLAG_NAMES_SHA256),
        "run4_flags": (RUN4_FLAG_NAMES, RUN4_FLAG_NAMES_SHA256),
        "comparisons": (COMPARISON_SCHEDULE, COMPARISON_SCHEDULE_SHA256),
        "memory": (memory_labels(), MEMORY_LABELS_SHA256),
    }
    for name, (value, expected) in checks.items():
        if canonical_json_hash(value) != expected:
            raise ValueError(f"FLAG0 {name} constant changed")


def validate_run4_rejection(repo: Path) -> dict[str, object]:
    experiment = repo / "experiments/qwen35-2b-latent-workspace-v1"
    failure_path = experiment / "a1-nc0-cap768-run4-mechanism-rejection-failure.json"
    log_path = experiment / "a1-nc0-cap768-run4-mechanism-rejection-run.log"
    plan_path = experiment / "a1-nc0-cap768-plan-v1.json"
    if any(path.is_symlink() or not path.is_file() for path in (failure_path, log_path, plan_path)):
        raise ValueError("FLAG0 run4 evidence absent or symlinked")
    failure = json.loads(failure_path.read_text())
    prior_plan = json.loads(plan_path.read_text())
    memory = failure.get("memory_ledger_partial")
    cache = failure.get("cache_guard_partial")
    if (
        file_sha256(failure_path) != RUN4_REJECTION_EVIDENCE["failure_file_sha256"]
        or file_sha256(log_path) != RUN4_REJECTION_EVIDENCE["launch_log_sha256"]
        or file_sha256(plan_path) != RUN4_REJECTION_EVIDENCE["plan_file_sha256"]
        or prior_plan.get("plan_sha256") != RUN4_REJECTION_EVIDENCE["plan_internal_sha256"]
        or prior_plan.get("plan_sha256") != canonical_json_hash(prior_plan, omitted_fields=("plan_sha256",))
        or failure.get("failure_sha256") != RUN4_REJECTION_EVIDENCE["failure_internal_sha256"]
        or failure.get("failure_sha256") != canonical_json_hash(failure, omitted_fields=("failure_sha256",))
        or failure.get("status") != RUN4_REJECTION_EVIDENCE["status"]
        or failure.get("failure_category") != RUN4_REJECTION_EVIDENCE["failure_category"]
        or failure.get("error_type") != RUN4_REJECTION_EVIDENCE["error_type"]
        or failure.get("error") != RUN4_REJECTION_EVIDENCE["exact_error"]
        or failure.get("run_id") != RUN4_REJECTION_EVIDENCE["run_id"]
        or failure.get("execution_commit") != RUN4_REJECTION_EVIDENCE["execution_commit"]
        or failure.get("mechanism_code_commit") != RUN4_REJECTION_EVIDENCE["mechanism_code_commit"]
        or failure.get("plan_sha256") != RUN4_REJECTION_EVIDENCE["plan_internal_sha256"]
        or failure.get("model_loaded") is not True
        or not isinstance(memory, list)
        or [row.get("label") for row in memory] != RUN4_REJECTION_EVIDENCE["memory_labels"]
        or not isinstance(cache, dict)
        or cache.get("closure_check_count") != 10
        or cache.get("restored_in_finally") is not True
        or cache.get("classes") != _CACHE_CLASS_CLOSURE
        or cache.get("negative_control_dynamic_cache_tripped") is not True
        or failure.get("protected_hashes_before") != {"coordinator_e33": _E33, "worker_h176": _H176}
        or failure.get("protected_hashes_before") != failure.get("protected_hash_probe_after_failure")
        or failure.get("checkpoint_metadata_before")
        != {
            "coordinator_e33": _RUNTIME["checkpoint_metadata_sha256"],
            "worker_h176": _RUNTIME["checkpoint_metadata_sha256"],
        }
        or failure.get("checkpoint_metadata_before") != failure.get("checkpoint_metadata_probe_after_failure")
        or failure.get("e33_state_tree_before") != failure.get("e33_state_tree_failure_audit")
        or failure.get("e33_gradients_absent_failure_audit") is not True
        or failure.get("worker_h176_loaded") is not False
        or any(
            failure.get(name) is not False
            for name in (
                "model_update_attempted",
                "bridge_created",
                "optimizer_created",
                "backward_used",
                "checkpoint_created",
                "candidate_created",
            )
        )
        or failure.get("asset_hashes_match_plan") is not True
        or len(failure.get("asset_hash_probe_after_failure", {})) != 35
        or failure.get("asset_hash_probe_after_failure") != prior_plan.get("asset_sha256")
        or failure.get("failure_audit_errors") != []
    ):
        raise ValueError("FLAG0 run4 rejection evidence changed")
    return RUN4_REJECTION_EVIDENCE


def load_plan(plan_path: Path, repo: Path) -> dict[str, object]:
    _validate_constants()
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError("FLAG0 plan absent or symlinked")
    plan = json.loads(plan_path.read_text())
    required = {
        "schema_version",
        "status",
        "mechanism_code_commit",
        "plan_sha256",
        "asset_sha256",
        "fixture",
        "fixture_sha256",
        "train_bank_sha256",
        "operation_schedule",
        "operation_schedule_sha256",
        "flag_names",
        "flag_names_sha256",
        "run4_flag_names",
        "run4_flag_names_sha256",
        "comparison_schedule",
        "comparison_schedule_sha256",
        "memory_labels",
        "memory_labels_sha256",
        "run4_rejection_evidence",
        "protected_checkpoints",
        "runtime",
        "resource_bounds",
        "interpretation_boundary",
        "decision_boundary",
        "execution_authorization",
        "authorized_run_id",
    }
    assets = plan.get("asset_sha256")
    if (
        set(plan) != required
        or plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "preregistered"
        or plan.get("execution_authorization") != "root_and_evaluator_review_required"
        or plan.get("authorized_run_id") != AUTHORIZED_RUN_ID
        or not _COMMIT.fullmatch(str(plan.get("mechanism_code_commit", "")))
        or plan.get("fixture") != FIXTURE
        or plan.get("fixture_sha256") != FIXTURE_SHA256
        or plan.get("train_bank_sha256") != TRAIN_BANK_SHA256
        or plan.get("operation_schedule") != OPERATION_SCHEDULE
        or plan.get("operation_schedule_sha256") != OPERATION_SCHEDULE_SHA256
        or plan.get("flag_names") != FLAG_NAMES
        or plan.get("flag_names_sha256") != FLAG_NAMES_SHA256
        or plan.get("run4_flag_names") != RUN4_FLAG_NAMES
        or plan.get("run4_flag_names_sha256") != RUN4_FLAG_NAMES_SHA256
        or plan.get("comparison_schedule") != COMPARISON_SCHEDULE
        or plan.get("comparison_schedule_sha256") != COMPARISON_SCHEDULE_SHA256
        or plan.get("memory_labels") != memory_labels()
        or plan.get("memory_labels_sha256") != MEMORY_LABELS_SHA256
        or plan.get("run4_rejection_evidence") != RUN4_REJECTION_EVIDENCE
        or plan.get("protected_checkpoints") != {"coordinator_e33": _E33, "worker_h176": _H176}
        or plan.get("runtime") != _RUNTIME
        or plan.get("resource_bounds") != RESOURCE_BOUNDS
        or plan.get("interpretation_boundary") != INTERPRETATION
        or plan.get("decision_boundary") != DECISION_BOUNDARY
        or not isinstance(assets, dict)
        or set(assets) != ASSET_PATHS
        or any(not _SHA.fullmatch(str(value)) for value in assets.values())
        or plan.get("plan_sha256") != canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    ):
        raise ValueError("FLAG0 plan changed")
    for relative, expected in assets.items():
        path = repo / relative
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"FLAG0 asset changed: {relative}")
    if validate_run4_rejection(repo) != plan["run4_rejection_evidence"]:
        raise ValueError("FLAG0 run4 evidence binding changed")
    return plan


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _validate_comparison(row: object, expected: dict[str, object]) -> None:
    keys = {
        "comparison_index",
        "name",
        "lhs",
        "rhs",
        "lhs_dtype",
        "rhs_dtype",
        "lhs_shape",
        "rhs_shape",
        "lhs_sha256",
        "rhs_sha256",
        "torch_equal",
        "element_count",
        "mismatch_count",
        "count_nonzero",
        "first_flat_mismatch",
        "metrics_defined",
        "max_abs",
        "rms_diff",
        "rhs_rms",
        "normalized_rms",
    }
    if not isinstance(row, dict) or set(row) != keys or any(row.get(key) != value for key, value in expected.items()):
        raise DiagnosticIncomplete("FLAG0 comparison identity changed")
    for key in ("lhs_sha256", "rhs_sha256"):
        if not _SHA.fullmatch(str(row.get(key, ""))):
            raise DiagnosticIncomplete("FLAG0 comparison hash changed")
    count = row.get("element_count")
    mismatch = row.get("mismatch_count")
    nonzero = row.get("count_nonzero")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or isinstance(mismatch, bool)
        or not isinstance(mismatch, int)
        or not 0 <= mismatch <= count
        or nonzero != mismatch
        or row.get("torch_equal") is not (mismatch == 0)
        or row.get("first_flat_mismatch") != (None if mismatch == 0 else row.get("first_flat_mismatch"))
        or (
            mismatch > 0
            and (
                isinstance(row.get("first_flat_mismatch"), bool)
                or not isinstance(row.get("first_flat_mismatch"), int)
                or not 0 <= row["first_flat_mismatch"] < count
            )
        )
    ):
        raise DiagnosticIncomplete("FLAG0 comparison counts changed")
    metrics = ("max_abs", "rms_diff", "rhs_rms", "normalized_rms")
    if row.get("metrics_defined") is True:
        if any(not _finite_number(row.get(key)) or row[key] < 0 for key in metrics):
            raise DiagnosticIncomplete("FLAG0 comparison metrics changed")
        if row["normalized_rms"] != row["rms_diff"] / max(row["rhs_rms"], 1e-12):
            raise DiagnosticIncomplete("FLAG0 normalized RMS changed")
    elif row.get("metrics_defined") is False:
        if any(row.get(key) is not None for key in metrics):
            raise DiagnosticIncomplete("FLAG0 undefined metrics changed")
    else:
        raise DiagnosticIncomplete("FLAG0 comparison metric state changed")


def validate_receipt(receipt: dict[str, object], *, plan: dict[str, object]) -> None:
    required = {
        "schema_version",
        "status",
        "plan_sha256",
        "mechanism_code_commit",
        "execution_commit",
        "asset_sha256",
        "run_id",
        "fixture",
        "fixture_sha256",
        "train_bank_sha256",
        "operation_schedule",
        "operation_schedule_sha256",
        "operation_counts",
        "flag_names",
        "flag_names_sha256",
        "run4_flag_names",
        "run4_flag_names_sha256",
        "flags",
        "run4_aggregate_reproduced",
        "comparison_schedule",
        "comparison_schedule_sha256",
        "comparisons",
        "tensor_evidence",
        "input_evidence",
        "run4_rejection_evidence",
        "versions",
        "runtime_sources",
        "static_guard",
        "protected_hashes_before",
        "protected_hashes_after",
        "checkpoint_metadata_before",
        "checkpoint_metadata_after",
        "e33_state_tree_before",
        "e33_state_tree_after",
        "e33_parameters_frozen_no_grad",
        "worker_h176_loaded",
        "model_runtime",
        "no_cache_contract",
        "cache_guard",
        "memory_ledger",
        "memory_labels_sha256",
        "resources",
        "timings",
        "decision_boundary",
        "interpretation_boundary",
        "receipt_sha256",
    }
    flags = receipt.get("flags")
    reproduced = isinstance(flags, dict) and not all(flags.get(name) is True for name in RUN4_FLAG_NAMES)
    expected_status = "capture768_flag_isolation_complete" if reproduced else "capture768_flag_isolation_nonreproduced"
    decision = receipt.get("decision_boundary")
    if (
        set(receipt) != required
        or receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != expected_status
        or receipt.get("plan_sha256") != plan.get("plan_sha256")
        or receipt.get("mechanism_code_commit") != plan.get("mechanism_code_commit")
        or receipt.get("asset_sha256") != plan.get("asset_sha256")
        or not _COMMIT.fullmatch(str(receipt.get("execution_commit", "")))
        or receipt.get("execution_commit") == receipt.get("mechanism_code_commit")
        or receipt.get("run_id") != AUTHORIZED_RUN_ID
        or receipt.get("fixture") != FIXTURE
        or receipt.get("fixture_sha256") != FIXTURE_SHA256
        or receipt.get("train_bank_sha256") != TRAIN_BANK_SHA256
        or receipt.get("operation_schedule") != OPERATION_SCHEDULE
        or receipt.get("operation_schedule_sha256") != OPERATION_SCHEDULE_SHA256
        or receipt.get("operation_counts")
        != {
            "embedding_lookup": 1,
            "e33_forward": 4,
            "lm_head_projection": 2,
            "capture": 4,
            "generation": 0,
            "h176_forward": 0,
            "bridge": 0,
            "optimizer": 0,
            "backward": 0,
            "step": 0,
            "checkpoint": 0,
            "candidate": 0,
        }
        or receipt.get("flag_names") != FLAG_NAMES
        or receipt.get("flag_names_sha256") != FLAG_NAMES_SHA256
        or receipt.get("run4_flag_names") != RUN4_FLAG_NAMES
        or receipt.get("run4_flag_names_sha256") != RUN4_FLAG_NAMES_SHA256
        or not isinstance(flags, dict)
        or list(flags) != FLAG_NAMES
        or any(not isinstance(value, bool) for value in flags.values())
        or receipt.get("run4_aggregate_reproduced") is not reproduced
        or receipt.get("comparison_schedule") != COMPARISON_SCHEDULE
        or receipt.get("comparison_schedule_sha256") != COMPARISON_SCHEDULE_SHA256
        or receipt.get("run4_rejection_evidence") != RUN4_REJECTION_EVIDENCE
        or not isinstance(decision, dict)
        or set(decision) != set(DECISION_BOUNDARY) | {"causal_interpretation"}
        or {key: decision.get(key) for key in DECISION_BOUNDARY} != DECISION_BOUNDARY
        or decision.get("causal_interpretation") != causal_interpretation(flags)
        or receipt.get("interpretation_boundary") != INTERPRETATION
        or receipt.get("protected_hashes_before") != plan.get("protected_checkpoints")
        or receipt.get("protected_hashes_after") != plan.get("protected_checkpoints")
        or receipt.get("checkpoint_metadata_before")
        != {
            "coordinator_e33": _RUNTIME["checkpoint_metadata_sha256"],
            "worker_h176": _RUNTIME["checkpoint_metadata_sha256"],
        }
        or receipt.get("checkpoint_metadata_after") != receipt.get("checkpoint_metadata_before")
        or not _SHA.fullmatch(str(receipt.get("e33_state_tree_before", "")))
        or receipt.get("e33_state_tree_after") != receipt.get("e33_state_tree_before")
        or receipt.get("e33_parameters_frozen_no_grad") is not True
        or receipt.get("worker_h176_loaded") is not False
        or receipt.get("versions")
        != {
            key: _RUNTIME[key]
            for key in (
                "python",
                "transformers",
                "flash_linear_attention",
                "torch_distribution",
                "torch_runtime",
            )
        }
        or receipt.get("model_runtime")
        != {
            "class": _RUNTIME["model_class"],
            "hidden_size": 2048,
            "vocab_size": 248320,
            "dtype": "torch.bfloat16",
            "device": "cuda:0",
        }
        or receipt.get("receipt_sha256") != canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    ):
        raise DiagnosticIncomplete("FLAG0 receipt identity/boundary changed")
    comparisons = receipt.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 13:
        raise DiagnosticIncomplete("FLAG0 comparison cardinality changed")
    for row, expected in zip(comparisons, COMPARISON_SCHEDULE, strict=True):
        _validate_comparison(row, expected)
    tensor_evidence = receipt.get("tensor_evidence")
    if not isinstance(tensor_evidence, dict) or set(tensor_evidence) != {
        "exact_embeddings",
        "L_ID_KEEP1.logits",
        "L_ID_KEEP1.hidden",
        "L_ID_KEEP1.capture",
        "L_E_KEEP1.logits",
        "L_E_KEEP1.hidden",
        "L_E_KEEP1.capture",
        "L_E_REPEAT_KEEP1.logits",
        "L_E_REPEAT_KEEP1.hidden",
        "L_E_REPEAT_KEEP1.capture",
        "L_ID_KEEP0_CONTROL.logits",
        "L_ID_KEEP0_CONTROL.last_logits",
        "L_ID_KEEP0_CONTROL.hidden",
        "L_ID_KEEP0_CONTROL.capture",
        "PROJ_ID1_LAST.logits",
        "PROJ_ID0_LAST.logits",
    }:
        raise DiagnosticIncomplete("FLAG0 tensor inventory changed")
    for item in tensor_evidence.values():
        if (
            not isinstance(item, dict)
            or set(item) != {"dtype", "shape", "sha256"}
            or not isinstance(item["dtype"], str)
            or not isinstance(item["shape"], list)
            or not _SHA.fullmatch(str(item["sha256"]))
        ):
            raise DiagnosticIncomplete("FLAG0 tensor evidence changed")
    expected_shapes = {
        "exact_embeddings": [1, 768, 2048],
        "L_ID_KEEP1.logits": [1, 1, 248320],
        "L_ID_KEEP1.hidden": [1, 768, 2048],
        "L_ID_KEEP1.capture": [1, 128, 2048],
        "L_E_KEEP1.logits": [1, 1, 248320],
        "L_E_KEEP1.hidden": [1, 768, 2048],
        "L_E_KEEP1.capture": [1, 128, 2048],
        "L_E_REPEAT_KEEP1.logits": [1, 1, 248320],
        "L_E_REPEAT_KEEP1.hidden": [1, 768, 2048],
        "L_E_REPEAT_KEEP1.capture": [1, 128, 2048],
        "L_ID_KEEP0_CONTROL.logits": [1, 768, 248320],
        "L_ID_KEEP0_CONTROL.last_logits": [1, 1, 248320],
        "L_ID_KEEP0_CONTROL.hidden": [1, 768, 2048],
        "L_ID_KEEP0_CONTROL.capture": [1, 128, 2048],
        "PROJ_ID1_LAST.logits": [1, 1, 248320],
        "PROJ_ID0_LAST.logits": [1, 1, 248320],
    }
    if any(
        tensor_evidence[name]["dtype"] != "torch.bfloat16" or tensor_evidence[name]["shape"] != shape
        for name, shape in expected_shapes.items()
    ):
        raise DiagnosticIncomplete("FLAG0 tensor dtype/shape changed")
    input_evidence = receipt.get("input_evidence")
    if (
        not isinstance(input_evidence, dict)
        or set(input_evidence)
        != {
            "rendered_ids_shape",
            "rendered_ids_dtype",
            "rendered_ids_contiguous",
            "rendered_ids_sha256",
            "padded_ids_sha256",
            "attention_mask_sha256",
            "position_ids_sha256",
            "capture_mask_sha256",
        }
        or input_evidence.get("rendered_ids_shape") != [1, 517]
        or input_evidence.get("rendered_ids_dtype") != "torch.int64"
        or input_evidence.get("rendered_ids_contiguous") is not True
        or any(
            not _SHA.fullmatch(str(input_evidence.get(key, "")))
            for key in (
                "rendered_ids_sha256",
                "padded_ids_sha256",
                "attention_mask_sha256",
                "position_ids_sha256",
                "capture_mask_sha256",
            )
        )
        or any(flags[name] is not True for name in FLAG_NAMES[:4])
    ):
        raise DiagnosticIncomplete("FLAG0 input geometry evidence changed")
    comparison_flags = {
        "id_embed_keep1_logits": "id_embed_keep1_logits_bitwise",
        "id_embed_keep1_full_hidden": "id_embed_keep1_full_hidden_bitwise",
        "id_embed_keep1_capture": "id_embed_keep1_capture_bitwise",
        "repeat_logits": "repeat_logits_bitwise",
        "repeat_full_hidden": "repeat_full_hidden_bitwise",
        "repeat_capture": "repeat_capture_bitwise",
        "keep0_keep1_full_hidden": "keep0_keep1_full_hidden_bitwise",
        "keep0_keep1_capture": "keep0_keep1_capture_bitwise",
        "keep0_last_logits_keep1": "keep0_last_logits_keep1_bitwise",
        "proj_id1_matches_id1_logits": "proj_id1_matches_id1_logits_bitwise",
        "proj_id0_matches_id0_last_logits": "proj_id0_matches_id0_last_logits_bitwise",
        "proj_id1_proj_id0": "proj_id1_proj_id0_bitwise",
        "id1_logits_proj_id0": "id1_logits_proj_id0_bitwise",
    }
    for row in comparisons:
        if (
            row["torch_equal"] is not flags[comparison_flags[row["name"]]]
            or row["lhs_dtype"] != tensor_evidence[row["lhs"]]["dtype"]
            or row["rhs_dtype"] != tensor_evidence[row["rhs"]]["dtype"]
            or row["lhs_shape"] != tensor_evidence[row["lhs"]]["shape"]
            or row["rhs_shape"] != tensor_evidence[row["rhs"]]["shape"]
            or row["lhs_sha256"] != tensor_evidence[row["lhs"]]["sha256"]
            or row["rhs_sha256"] != tensor_evidence[row["rhs"]]["sha256"]
        ):
            raise DiagnosticIncomplete("FLAG0 comparison/tensor binding changed")
    no_cache = receipt.get("no_cache_contract")
    cache = receipt.get("cache_guard")
    if (
        no_cache
        != {
            "calls": 4,
            "use_cache_false": True,
            "pkv_input_none": True,
            "pkv_output_none": True,
            "rope_reset_every_call": True,
            "model_config_use_cache": False,
            "generation_config_use_cache": False,
        }
        or not isinstance(cache, dict)
        or set(cache)
        != {"classes", "negative_control_dynamic_cache_tripped", "closure_check_count", "restored_in_finally"}
        or cache.get("classes") != _CACHE_CLASS_CLOSURE
        or cache.get("negative_control_dynamic_cache_tripped") is not True
        or cache.get("closure_check_count") != 11
        or cache.get("restored_in_finally") is not True
    ):
        raise DiagnosticIncomplete("FLAG0 cache evidence changed")
    labels = receipt.get("memory_ledger")
    if not isinstance(labels, list) or [row.get("label") for row in labels] != memory_labels():
        raise DiagnosticIncomplete("FLAG0 memory ledger changed")
    if receipt.get("memory_labels_sha256") != MEMORY_LABELS_SHA256:
        raise DiagnosticIncomplete("FLAG0 memory-label hash changed")
    cap_bytes = RESOURCE_BOUNDS["allocator_cap_gib"] * 2**30
    previous_peaks = (0, 0)
    for row in labels:
        if set(row) != {
            "label",
            "allocated_bytes",
            "reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        }:
            raise DiagnosticIncomplete("FLAG0 memory row schema changed")
        values = [row[key] for key in row if key.endswith("_bytes")]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= cap_bytes for value in values
        ):
            raise DiagnosticIncomplete("FLAG0 memory value changed")
        if (
            row["peak_allocated_bytes"] < row["allocated_bytes"]
            or row["peak_reserved_bytes"] < row["reserved_bytes"]
            or row["peak_allocated_bytes"] < previous_peaks[0]
            or row["peak_reserved_bytes"] < previous_peaks[1]
        ):
            raise DiagnosticIncomplete("FLAG0 memory peak changed")
        previous_peaks = (row["peak_allocated_bytes"], row["peak_reserved_bytes"])
    runtime_sources = receipt.get("runtime_sources")
    if not isinstance(runtime_sources, dict) or set(runtime_sources) != set(_RUNTIME["transformers_source_sha256"]):
        raise DiagnosticIncomplete("FLAG0 runtime source inventory changed")
    for name, expected_sha in _RUNTIME["transformers_source_sha256"].items():
        source = runtime_sources[name]
        if (
            not isinstance(source, dict)
            or set(source) != {"path", "sha256"}
            or source.get("sha256") != expected_sha
            or not isinstance(source.get("path"), str)
        ):
            raise DiagnosticIncomplete("FLAG0 runtime source identity changed")
    if receipt.get("static_guard") != {
        "runner_sha256": plan["asset_sha256"]["scripts/latent/run_a1_nc0_cap768_flag0_v1.py"],
        "forbidden_calls": [],
    }:
        raise DiagnosticIncomplete("FLAG0 static guard changed")
    resources = receipt.get("resources")
    if (
        not isinstance(resources, dict)
        or set(resources)
        != {
            "gpu_name",
            "total_gpu_memory_bytes",
            "allocator_cap_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "host_ram_bytes",
            "free_disk_bytes_preflight",
            "cuda_visible_devices",
            "network_disabled",
            "physical_gpu_before",
            "physical_gpu_after",
            "physical_gpu1_unused_before_after",
        }
        or resources.get("gpu_name") != RESOURCE_BOUNDS["gpu_model"]
        or resources.get("allocator_cap_bytes") != cap_bytes
        or resources.get("peak_allocated_bytes") != max(row["peak_allocated_bytes"] for row in labels)
        or resources.get("peak_reserved_bytes") != max(row["peak_reserved_bytes"] for row in labels)
        or resources.get("cuda_visible_devices") != "0"
        or resources.get("network_disabled") != {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        or resources.get("physical_gpu1_unused_before_after") is not True
    ):
        raise DiagnosticIncomplete("FLAG0 resource evidence changed")
    for key, minimum in (
        ("total_gpu_memory_bytes", RESOURCE_BOUNDS["minimum_gpu_memory_gib"]),
        ("host_ram_bytes", RESOURCE_BOUNDS["minimum_host_ram_gib"]),
        ("free_disk_bytes_preflight", RESOURCE_BOUNDS["minimum_free_disk_gib"]),
    ):
        if (
            isinstance(resources.get(key), bool)
            or not isinstance(resources.get(key), int)
            or resources[key] < minimum * 2**30
        ):
            raise DiagnosticIncomplete("FLAG0 resource floor changed")
    for physical in (resources["physical_gpu_before"], resources["physical_gpu_after"]):
        if (
            not isinstance(physical, dict)
            or set(physical) != {"names", "uuids", "memory_used_mib", "compute_apps"}
            or physical.get("names") != [RESOURCE_BOUNDS["gpu_model"], RESOURCE_BOUNDS["gpu_model"]]
            or not isinstance(physical.get("uuids"), list)
            or len(physical["uuids"]) != 2
            or not isinstance(physical.get("memory_used_mib"), list)
            or len(physical["memory_used_mib"]) != 2
            or physical["memory_used_mib"][1] > 512
        ):
            raise DiagnosticIncomplete("FLAG0 physical GPU evidence changed")
    timings = receipt.get("timings")
    if not isinstance(timings, dict) or set(timings) != {
        "operations",
        "operation_cuda_event_seconds_sum",
        "operation_wall_seconds_sum",
        "tokenizer_load_seconds",
        "model_load_seconds",
        "compute_seconds",
        "audit_seconds",
        "total_seconds",
    }:
        raise DiagnosticIncomplete("FLAG0 timing schema changed")
    operations = timings["operations"]
    if not isinstance(operations, list) or len(operations) != 7:
        raise DiagnosticIncomplete("FLAG0 operation timing cardinality changed")
    for observed, expected in zip(operations, OPERATION_SCHEDULE, strict=True):
        if (
            set(observed) != set(expected) | {"cuda_event_seconds", "wall_seconds"}
            or any(observed.get(key) != value for key, value in expected.items())
            or any(
                not _finite_number(observed.get(key)) or observed[key] < 0
                for key in ("cuda_event_seconds", "wall_seconds")
            )
        ):
            raise DiagnosticIncomplete("FLAG0 operation timing changed")
    scalar_timings = (
        "operation_cuda_event_seconds_sum",
        "operation_wall_seconds_sum",
        "tokenizer_load_seconds",
        "model_load_seconds",
        "compute_seconds",
        "audit_seconds",
        "total_seconds",
    )
    if any(not _finite_number(timings.get(key)) or timings[key] < 0 for key in scalar_timings):
        raise DiagnosticIncomplete("FLAG0 timing value changed")
    if (
        timings["operation_cuda_event_seconds_sum"] != math.fsum(item["cuda_event_seconds"] for item in operations)
        or timings["operation_wall_seconds_sum"] != math.fsum(item["wall_seconds"] for item in operations)
        or timings["compute_seconds"] > RESOURCE_BOUNDS["compute_seconds"]
        or timings["audit_seconds"] > RESOURCE_BOUNDS["audit_seconds"]
    ):
        raise DiagnosticIncomplete("FLAG0 timing aggregate changed")


def classify_failure(error: BaseException) -> tuple[str, str]:
    if isinstance(error, NoCacheRejected):
        return "capture768_nocache_rejected", "cache_allocation_pkv_or_rope"
    if isinstance(error, ResourceFitRejected):
        return "capture768_resource_fit_rejected", "allocator_cap_oom_or_compute_timeout"
    if isinstance(error, DiagnosticIncomplete):
        return "capture768_flag_isolation_incomplete", "diagnostic_operation_or_evidence_incomplete"
    return "infrastructure_invalid", "runtime_checkpoint_host_asset_or_publication"
