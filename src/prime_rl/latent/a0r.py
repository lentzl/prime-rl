from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from prime_rl.latent.a0 import canonical_json_hash, file_sha256, validate_a0_bank, validate_a0_plan

A0R_PLAN_SCHEMA = "prime-rl/latent-a0r-mechanism-plan/v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_A0R_ASSET_PATHS = {
    "experiments/qwen35-2b-latent-workspace-v1/a0-failed-runtime-version-failure.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0-failed-runtime-version.log",
    "experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-plan-v1.json",
    "scripts/latent/run_a0_mechanism_v1.py",
    "scripts/latent/run_a0r_mechanism_v1.sh",
    "src/prime_rl/latent/__init__.py",
    "src/prime_rl/latent/a0.py",
    "src/prime_rl/latent/a0r.py",
    "src/prime_rl/latent/policy_adapter.py",
}
_FAILED_EVIDENCE = {
    "schema_version": "prime-rl/latent-a0-failed-start-evidence/v1",
    "failed_plan_sha256": "444e10f6f9ca92f19bba5c393eb8308a8cd3674970b9695d0854ab2f1f39d9a8",
    "failed_execution_commit": "9d96a2143795d97f16aaf16c21440a906e1eab3f",
    "failed_mechanism_commit": "57aef2c09eea031427be70cc4351bdf774856aa6",
    "failure_file_sha256": "60a8d884b338bf035788ff5db07115aa1fcc413a95f484917f8c386e22b413fc",
    "failure_internal_sha256": "ee131bf5e46e721133ce9538a7eccfb6ec925a7d46d2aea6874728e8a590f930",
    "launch_log_sha256": "429d201c769b8ea7de51afddefc7b25f1c95fb4b8419fafd020cfb4b9da1bdc2",
    "exact_error": "torch version '2.11.0+cu128' differs from frozen '2.11.0'",
    "failure_stage": "protected_preflight_verified",
    "failure_status": "infrastructure_invalid",
    "failure_category": "environment_provenance_timeout_or_oom",
    "observed_torch_distribution": "2.11.0+cu128",
    "model_loaded": False,
    "cuda_allocated": False,
    "model_update_attempted": False,
    "bank_reuse_basis": "failed before tokenizer/model construction and before any CUDA query or allocation",
}


def validate_a0r_plan(
    plan: dict[str, object],
    *,
    bank_sha256: str,
    base_plan: dict[str, object],
    failure: dict[str, object],
    failure_file_sha256: str,
    launch_log_sha256: str,
) -> None:
    required = {
        "schema_version",
        "status",
        "execution_authorization",
        "mechanism_code_commit",
        "asset_sha256",
        "plan_sha256",
        "bank_sha256",
        "supersedes_plan_sha256",
        "prior_failed_start_evidence",
        "protected_checkpoints",
        "remote_paths",
        "runtime",
        "mechanism",
        "admission",
        "resource_bounds",
        "failure_classification",
        "promotion_boundary",
    }
    if set(plan) != required:
        raise ValueError("A0R plan fields differ from the v1 schema")
    if plan.get("schema_version") != A0R_PLAN_SCHEMA or plan.get("status") != "preregistered":
        raise ValueError("A0R schema or preregistration status changed")
    if plan.get("execution_authorization") != "root_review_required":
        raise ValueError("A0R execution authorization boundary changed")
    commit = plan.get("mechanism_code_commit")
    if not isinstance(commit, str) or not _GIT_COMMIT_RE.fullmatch(commit):
        raise ValueError("A0R mechanism code commit is missing or malformed")
    assets = plan.get("asset_sha256")
    if not isinstance(assets, dict) or set(assets) != _A0R_ASSET_PATHS:
        raise ValueError("A0R executable/evidence asset set differs from the freeze")
    if any(not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in assets.values()):
        raise ValueError("A0R asset hash is malformed")
    if plan.get("bank_sha256") != bank_sha256 or not _SHA256_RE.fullmatch(bank_sha256):
        raise ValueError("A0R bank hash differs from the failed no-exposure run")
    plan_hash = plan.get("plan_sha256")
    if not isinstance(plan_hash, str) or not _SHA256_RE.fullmatch(plan_hash):
        raise ValueError("A0R plan hash is missing or malformed")
    if plan_hash != canonical_json_hash(plan, omitted_fields=("plan_sha256",)):
        raise ValueError("A0R plan hash does not match its canonical content")
    if plan.get("supersedes_plan_sha256") != base_plan["plan_sha256"]:
        raise ValueError("A0R does not point to the immutable failed A0 plan")
    if plan.get("prior_failed_start_evidence") != _FAILED_EVIDENCE:
        raise ValueError("A0R failed-start evidence contract changed")

    if failure_file_sha256 != _FAILED_EVIDENCE["failure_file_sha256"]:
        raise ValueError("A0R bound failure artifact hash changed")
    if launch_log_sha256 != _FAILED_EVIDENCE["launch_log_sha256"]:
        raise ValueError("A0R bound launch log hash changed")
    if canonical_json_hash(failure, omitted_fields=("failure_sha256",)) != _FAILED_EVIDENCE["failure_internal_sha256"]:
        raise ValueError("A0R bound failure internal hash changed")
    expected_failure_fields = {
        "error": _FAILED_EVIDENCE["exact_error"],
        "execution_commit": _FAILED_EVIDENCE["failed_execution_commit"],
        "failure_category": _FAILED_EVIDENCE["failure_category"],
        "failure_sha256": _FAILED_EVIDENCE["failure_internal_sha256"],
        "mechanism_code_commit": _FAILED_EVIDENCE["failed_mechanism_commit"],
        "model_update_attempted": False,
        "plan_sha256": _FAILED_EVIDENCE["failed_plan_sha256"],
        "stage": _FAILED_EVIDENCE["failure_stage"],
        "status": _FAILED_EVIDENCE["failure_status"],
    }
    if any(failure.get(key) != value for key, value in expected_failure_fields.items()):
        raise ValueError("A0R failure artifact does not prove the frozen failed-start facts")
    if (
        failure.get("protected_hash_probe_after_failure", {}).get("coordinator_e33", {}).get("model_sha256")
        != base_plan["protected_checkpoints"]["coordinator_e33"]
        or failure.get("protected_hash_probe_after_failure", {}).get("worker_h176", {}).get("model_sha256")
        != base_plan["protected_checkpoints"]["worker_h176"]
    ):
        raise ValueError("A0R failed-start evidence does not preserve protected model hashes")

    for field in (
        "protected_checkpoints",
        "remote_paths",
        "mechanism",
        "admission",
        "failure_classification",
        "promotion_boundary",
    ):
        if plan.get(field) != base_plan[field]:
            raise ValueError(f"A0R changed frozen A0 field: {field}")
    expected_runtime = copy.deepcopy(base_plan["runtime"])
    expected_runtime.pop("torch")
    expected_runtime["torch_distribution"] = "2.11.0+cu128"
    expected_runtime["torch_runtime"] = "2.11.0+cu128"
    if plan.get("runtime") != expected_runtime:
        raise ValueError("A0R runtime differs beyond the exact Torch distribution/runtime repair")
    expected_resources = copy.deepcopy(base_plan["resource_bounds"])
    expected_resources["output_root"] = "/home/ubuntu/rlm/outputs/latent-a0r-mechanism-v1"
    if plan.get("resource_bounds") != expected_resources:
        raise ValueError("A0R resources differ beyond the fresh immutable output namespace")


def load_and_validate_a0r_plan(plan_path: Path, bank_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    directory = plan_path.parent
    base_plan_path = directory / "a0-mechanism-plan-v1.json"
    failure_path = directory / "a0-failed-runtime-version-failure.json"
    log_path = directory / "a0-failed-runtime-version.log"
    paths = (plan_path, bank_path, base_plan_path, failure_path, log_path)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("A0R plan, bank, base plan, or failed-start evidence is absent or symlinked")
    plan = json.loads(plan_path.read_text())
    bank = json.loads(bank_path.read_text())
    base_plan = json.loads(base_plan_path.read_text())
    failure = json.loads(failure_path.read_text())
    validate_a0_bank(bank)
    validate_a0_plan(base_plan, bank_sha256=file_sha256(bank_path))
    validate_a0r_plan(
        plan,
        bank_sha256=file_sha256(bank_path),
        base_plan=base_plan,
        failure=failure,
        failure_file_sha256=file_sha256(failure_path),
        launch_log_sha256=file_sha256(log_path),
    )
    return plan, bank
