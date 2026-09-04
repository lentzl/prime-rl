from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PRIMARY_ARMS = ("M0", "MOTH", "MSELF", "MCUR")
DIAGNOSTIC_ARMS = ("ZERO", "NOISE")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
_E33 = "e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47"
_H176 = "77980e247bbccd6463ddda02cd42d2c357e15f8ec1ad0ea84627e008a8674a1e"
_FAMILIES = ("keyed_numeric", "relational_join", "config_structure", "ownership_graph")
_REQUIRED_THRESHOLDS = {
    "minimum_complete_held_out_tasks",
    "minimum_paired_recoveries",
    "minimum_recovery_families",
    "minimum_current_vs_other_utility_gap",
    "practical_equivalence_margin",
    "mself_gpu_seconds_relative_tolerance",
}


def canonical_plan_hash(plan: dict[str, object]) -> str:
    payload = dict(plan)
    payload.pop("plan_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_launch_plan(plan: dict[str, object]) -> None:
    """Fail closed unless an independently frozen A0/A1 plan is launch-ready."""

    if plan.get("schema_version") != "prime-rl/latent-a0-a1-preregistration/v1":
        raise ValueError("unknown A0/A1 preregistration schema")
    if plan.get("status") != "preregistered":
        raise ValueError("plan is not independently frozen for launch")
    if plan.get("execution_authorization") != "owner_approved":
        raise ValueError("plan lacks explicit owner execution authorization")
    if not isinstance(plan.get("frozen_by"), str) or not plan["frozen_by"]:
        raise ValueError("independent evaluator identity is missing")
    if not isinstance(plan.get("frozen_at_utc"), str) or not plan["frozen_at_utc"].endswith("Z"):
        raise ValueError("independent freeze timestamp is missing")
    if not isinstance(plan.get("base_code_commit"), str) or not _GIT_COMMIT_RE.fullmatch(plan["base_code_commit"]):
        raise ValueError("base code commit is missing or malformed")
    for field in ("tokenizer_template_sha256", "bridge_config_sha256", "capture_spec_sha256"):
        value = plan.get(field)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ValueError(f"{field} must be frozen before launch")
    plan_hash = plan.get("plan_sha256")
    if not isinstance(plan_hash, str) or not _SHA256_RE.fullmatch(plan_hash):
        raise ValueError("plan_sha256 is missing or malformed")
    if plan_hash != canonical_plan_hash(plan):
        raise ValueError("plan_sha256 does not match the canonical plan")
    if tuple(plan.get("primary_audit_arms", ())) != PRIMARY_ARMS:
        raise ValueError("primary audit arms must be M0/MOTH/MSELF/MCUR in order")
    if tuple(plan.get("diagnostic_arms", ())) != DIAGNOSTIC_ARMS:
        raise ValueError("diagnostic arms must be ZERO/NOISE")

    protected = plan.get("protected_checkpoints")
    if not isinstance(protected, dict) or set(protected) != {"coordinator_e33", "worker_h176"}:
        raise ValueError("both protected checkpoints must be declared")
    if protected != {"coordinator_e33": _E33, "worker_h176": _H176}:
        raise ValueError("protected checkpoint hashes do not match canonical e33/H176")

    workspace = plan.get("workspace_contract")
    expected_workspace = {
        "schema_version": "prime-rl/latent-workspace/v1",
        "slots": 8,
        "workspace_width": 256,
        "receiver_embedding_width": 2048,
        "capture_layer": -1,
        "capture_boundary": "accepted_delegation",
        "maximum_capture_tokens": 128,
        "parent_gradient": "detached",
        "receiver_injection_boundary": "after_child_objective_before_assistant_opening",
        "operational_direction": "coordinator_parent_to_coordinator_child",
    }
    if workspace != expected_workspace:
        raise ValueError("workspace contract differs from the A0/A1 v1 contract")

    split = plan.get("split_information_bank")
    if not isinstance(split, dict):
        raise ValueError("split-information bank declaration is missing")
    for field in ("train_seed", "validation_seed", "held_out_seed", "arm_order_seed"):
        if isinstance(split.get(field), bool) or not isinstance(split.get(field), int):
            raise ValueError(f"{field} must be independently assigned before launch")
    for field in ("train_manifest_sha256", "validation_manifest_sha256", "held_out_manifest_sha256"):
        value = split.get(field)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ValueError(f"{field} must be frozen before launch")
    families = split.get("families")
    if families != list(_FAMILIES):
        raise ValueError("split-information families or order differ from the v1 bank")
    for field in (
        "train_examples_per_family",
        "validation_examples_per_family",
        "held_out_examples_per_family",
    ):
        if isinstance(split.get(field), bool) or not isinstance(split.get(field), int) or split[field] < 2:
            raise ValueError(f"{field} must be at least two")
    if split.get("queries_per_evidence") != 3:
        raise ValueError("v1 requires three queries per evidence packet")

    thresholds = plan.get("admission_thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != _REQUIRED_THRESHOLDS:
        raise ValueError("all admission thresholds must be independently fixed")
    if any(value is None or not isinstance(value, (int, float)) for value in thresholds.values()):
        raise ValueError("admission thresholds cannot contain placeholders")
    for field in (
        "minimum_complete_held_out_tasks",
        "minimum_paired_recoveries",
        "minimum_recovery_families",
    ):
        if isinstance(thresholds[field], bool) or int(thresholds[field]) != thresholds[field] or thresholds[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    for field in (
        "minimum_current_vs_other_utility_gap",
        "practical_equivalence_margin",
        "mself_gpu_seconds_relative_tolerance",
    ):
        if not 0 <= thresholds[field] <= 1:
            raise ValueError(f"{field} must be between zero and one")
    held_out_tasks = len(_FAMILIES) * split["held_out_examples_per_family"] * split["queries_per_evidence"]
    if thresholds["minimum_complete_held_out_tasks"] > held_out_tasks:
        raise ValueError("minimum complete task count exceeds the held-out bank")
    if thresholds["minimum_paired_recoveries"] > thresholds["minimum_complete_held_out_tasks"]:
        raise ValueError("paired recovery floor exceeds the complete task floor")
    if thresholds["minimum_recovery_families"] > len(_FAMILIES):
        raise ValueError("recovery family floor exceeds the bank family count")

    compute_match = plan.get("mself_compute_match")
    if not isinstance(compute_match, dict):
        raise ValueError("MSELF compute-match declaration is missing")
    if compute_match.get("primary_method") != "equal_operation":
        raise ValueError("MSELF primary comparison must use equal-operation matching")
    if compute_match.get("sender_feature_forwards") != compute_match.get("self_feature_forwards"):
        raise ValueError("MSELF feature-extraction forward counts do not match")
    if compute_match.get("sender_bridge_forwards") != compute_match.get("self_bridge_forwards"):
        raise ValueError("MSELF bridge forward counts do not match")
    if compute_match.get("sender_feature_input_tokens") != compute_match.get("self_feature_input_tokens"):
        raise ValueError("MSELF feature-extraction token budgets do not match")
    for field in (
        "sender_feature_forwards",
        "self_feature_forwards",
        "sender_bridge_forwards",
        "self_bridge_forwards",
        "sender_feature_input_tokens",
        "self_feature_input_tokens",
        "receiver_decode_budget_tokens",
    ):
        if isinstance(compute_match.get(field), bool) or not isinstance(compute_match.get(field), int):
            raise ValueError(f"{field} must be a frozen integer")
        if compute_match[field] < 1:
            raise ValueError(f"{field} must be positive")
    if compute_match.get("same_physical_gpu") is not True:
        raise ValueError("MSELF cost audit must use the same physical GPU")


def load_and_validate_launch_plan(path: Path) -> dict[str, object]:
    plan = json.loads(path.read_text())
    validate_launch_plan(plan)
    return plan
