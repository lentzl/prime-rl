from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, TypeVar

from prime_rl.latent.a0 import canonical_json_hash, file_sha256, validate_a0_bank
from prime_rl.latent.a0dr import load_and_validate_a0dr_plan, validate_a0dr_receipt

A0DR2_PLAN_SCHEMA = "prime-rl/latent-a0dr2-cache-diagnostic-plan/v1"

_T = TypeVar("_T")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ASSET_PATHS = {
    "experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0c-carrier-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0c-carrier-success-receipt.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0d-cache-diagnostic-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0dr-cache-diagnostic-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0dr-incomplete-evidence-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0dr-incomplete-failure.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0dr-incomplete-run.log",
    "experiments/qwen35-2b-latent-workspace-v1/a0r-mechanism-rejected-failure.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0r-mechanism-rejected.log",
    "scripts/latent/run_a0dr_cache_diagnostic_v1.py",
    "scripts/latent/run_a0dr2_cache_diagnostic_v1.py",
    "scripts/latent/run_a0dr2_cache_diagnostic_v1.sh",
    "src/prime_rl/latent/__init__.py",
    "src/prime_rl/latent/a0.py",
    "src/prime_rl/latent/a0c.py",
    "src/prime_rl/latent/a0d.py",
    "src/prime_rl/latent/a0dr.py",
    "src/prime_rl/latent/a0dr2.py",
    "src/prime_rl/latent/a0r.py",
    "src/prime_rl/latent/policy_adapter.py",
}
_FAILED_RUN = {
    "status": "diagnostic_incomplete_receipt_validation",
    "failure_file_sha256": "d6f498c77c23cceb5018149b7f18ab4472965c730f1f831d32279c4614fba064",
    "failure_internal_sha256": "c06a83fad2a591b0192c8bf28308a906d033580dfbcced2c89cd2cff94a12f26",
    "launch_log_sha256": "84d650cdbfacef48e718a6d4c356fa266a56606f458631e604cf607c0644f3eb",
    "evidence_manifest_sha256": "66feddf692b464116c705be94bae90de1c716db98bf19c640f6894033851f8a8",
    "mechanism_commit": "c3e8e951d198e4c2e7d4b425a60237bc4f288b6e",
    "execution_commit": "dbb377b816d0838e10f7e4b8ed44913a6d6aa5b9",
    "plan_sha256": "1226c9598875df8e4874cc2dee6da2c538f8a3b12cb8ac7dc1109fe3847601a2",
    "plan_file_sha256": "c897c1f6136b2aa62a4d7c816984539bae6cdec83266e2de90ea15720e5129e2",
    "stage": "protected_postflight_verified",
    "error": "A0D receipt contract failed: A0DR prepared cache provenance changed",
    "protected_hashes_exact": True,
    "model_update_attempted": False,
}
_REPAIR = {
    "diagnosis": "prepared_summary observed a mutable past_key_values alias after cached decode incremented it",
    "change": "snapshot prepared evidence immediately after prepare_inputs_for_generation and before full or cached decode",
    "numerical_arms_ids_positions_masks_and_references_changed": False,
    "mutable_cache_regression_required": True,
    "reuse_of_incomplete_numerical_results": False,
}
_INTERPRETATION = (
    "non-promotional receipt-instrumentation repair only; A0R remains rejected and A1 remains blocked regardless "
    "of result; A0DR emitted no valid numerical receipt and its incomplete result is not reused"
)


def invoke_after_predecode_snapshot(
    prepared: dict[str, object],
    *,
    summarize: Callable[[dict[str, object]], dict[str, object]],
    invoke: Callable[[], _T],
) -> tuple[_T, dict[str, object]]:
    """Snapshot mutable prepared inputs before invoking a decode that can mutate cache aliases."""
    snapshot = summarize(prepared)
    result = invoke()
    return result, snapshot


def validate_a0dr2_plan(plan: dict[str, object], *, bank_sha256: str, a0dr_plan: dict[str, object]) -> None:
    required = {
        "schema_version",
        "status",
        "execution_authorization",
        "mechanism_code_commit",
        "asset_sha256",
        "plan_sha256",
        "bank_sha256",
        "supersedes_failed_run",
        "evidence_capture_repair",
        "bound_evidence",
        "protected_checkpoints",
        "remote_paths",
        "runtime",
        "diagnostic",
        "resource_bounds",
        "failure_classification",
        "interpretation_boundary",
    }
    if set(plan) != required:
        raise ValueError("A0DR2 plan fields differ from schema")
    if plan.get("schema_version") != A0DR2_PLAN_SCHEMA or plan.get("status") != "preregistered":
        raise ValueError("A0DR2 schema or status changed")
    if plan.get("execution_authorization") != "root_review_required":
        raise ValueError("A0DR2 authorization changed")
    commit = plan.get("mechanism_code_commit")
    if not isinstance(commit, str) or not _GIT_COMMIT_RE.fullmatch(commit):
        raise ValueError("A0DR2 mechanism commit is malformed")
    assets = plan.get("asset_sha256")
    if (
        not isinstance(assets, dict)
        or set(assets) != _ASSET_PATHS
        or any(not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in assets.values())
    ):
        raise ValueError("A0DR2 asset closure changed")
    if plan.get("bank_sha256") != bank_sha256 or not _SHA256_RE.fullmatch(bank_sha256):
        raise ValueError("A0DR2 bank hash changed")
    if plan.get("plan_sha256") != canonical_json_hash(plan, omitted_fields=("plan_sha256",)):
        raise ValueError("A0DR2 canonical plan hash changed")
    if plan.get("supersedes_failed_run") != _FAILED_RUN:
        raise ValueError("A0DR2 failed-run binding changed")
    if plan.get("evidence_capture_repair") != _REPAIR:
        raise ValueError("A0DR2 repair changed")
    for field in (
        "bound_evidence",
        "protected_checkpoints",
        "remote_paths",
        "runtime",
        "diagnostic",
        "failure_classification",
    ):
        if plan.get(field) != a0dr_plan[field]:
            raise ValueError(f"A0DR2 changed frozen A0DR field: {field}")
    resources = dict(a0dr_plan["resource_bounds"])
    resources["output_root"] = "/home/ubuntu/rlm/outputs/latent-a0dr2-cache-diagnostic-v1"
    if plan.get("resource_bounds") != resources:
        raise ValueError("A0DR2 resources differ beyond fresh namespace")
    if plan.get("interpretation_boundary") != _INTERPRETATION:
        raise ValueError("A0DR2 interpretation boundary changed")


def _validate_failed_evidence(
    failure_path: Path, manifest_path: Path, log_path: Path, plan: dict[str, object]
) -> None:
    if any(path.is_symlink() or not path.is_file() for path in (failure_path, manifest_path, log_path)):
        raise ValueError("A0DR2 failed-run evidence is absent or symlinked")
    if file_sha256(failure_path) != _FAILED_RUN["failure_file_sha256"]:
        raise ValueError("A0DR2 failure file changed")
    if file_sha256(manifest_path) != _FAILED_RUN["evidence_manifest_sha256"]:
        raise ValueError("A0DR2 evidence manifest changed")
    if file_sha256(log_path) != _FAILED_RUN["launch_log_sha256"]:
        raise ValueError("A0DR2 launch log changed")
    failure = json.loads(failure_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (
        failure.get("failure_sha256") != _FAILED_RUN["failure_internal_sha256"]
        or failure.get("failure_sha256") != canonical_json_hash(failure, omitted_fields=("failure_sha256",))
        or failure.get("status") != "diagnostic_incomplete"
        or failure.get("stage") != _FAILED_RUN["stage"]
        or failure.get("error") != _FAILED_RUN["error"]
        or failure.get("execution_commit") != _FAILED_RUN["execution_commit"]
        or failure.get("mechanism_code_commit") != _FAILED_RUN["mechanism_commit"]
        or failure.get("plan_sha256") != _FAILED_RUN["plan_sha256"]
        or failure.get("model_update_attempted") is not False
        or {
            name: value.get("model_sha256")
            for name, value in failure.get("protected_hash_probe_after_failure", {}).items()
        }
        != plan["protected_checkpoints"]
    ):
        raise ValueError("A0DR2 internal failed-run evidence changed")
    if manifest != {
        "schema_version": "prime-rl/latent-a0dr-incomplete-evidence/v1",
        "durable_snapshot": "/Users/ilkkalehto/Documents/rlm/tmp/durable-snapshots/2026-09-05-a0dr-incomplete-receipt",
        "failure_file_sha256": _FAILED_RUN["failure_file_sha256"],
        "failure_internal_sha256": _FAILED_RUN["failure_internal_sha256"],
        "launch_log_sha256": _FAILED_RUN["launch_log_sha256"],
    }:
        raise ValueError("A0DR2 evidence manifest content changed")


def load_and_validate_a0dr2_plan(plan_path: Path, bank_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    prior_path = plan_path.with_name("a0dr-cache-diagnostic-plan-v1.json")
    failure_path = plan_path.with_name("a0dr-incomplete-failure.json")
    manifest_path = plan_path.with_name("a0dr-incomplete-evidence-v1.json")
    log_path = plan_path.with_name("a0dr-incomplete-run.log")
    if prior_path.is_symlink() or not prior_path.is_file() or file_sha256(prior_path) != _FAILED_RUN["plan_file_sha256"]:
        raise ValueError("A0DR2 prior plan changed")
    a0dr_plan, _ = load_and_validate_a0dr_plan(prior_path, bank_path)
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError("A0DR2 plan is absent or symlinked")
    plan = json.loads(plan_path.read_text())
    bank = json.loads(bank_path.read_text())
    validate_a0_bank(bank)
    validate_a0dr2_plan(plan, bank_sha256=file_sha256(bank_path), a0dr_plan=a0dr_plan)
    _validate_failed_evidence(failure_path, manifest_path, log_path, plan)
    return plan, bank


def validate_a0dr2_receipt(receipt: dict[str, object], *, plan: dict[str, object]) -> None:
    if (
        receipt.get("schema_version") != "prime-rl/latent-a0dr2-cache-diagnostic-receipt/v1"
        or receipt.get("supersedes_failed_run") != _FAILED_RUN
        or receipt.get("evidence_capture_timing")
        != "immediately_after_prepare_inputs_for_generation_before_full_or_cached_decode"
    ):
        raise ValueError("A0DR2 repair receipt binding changed")
    base_receipt = dict(receipt)
    base_receipt.pop("supersedes_failed_run", None)
    base_receipt.pop("evidence_capture_timing", None)
    base_receipt["schema_version"] = "prime-rl/latent-a0dr-cache-diagnostic-receipt/v1"
    base_receipt["receipt_sha256"] = canonical_json_hash(base_receipt, omitted_fields=("receipt_sha256",))
    validate_a0dr_receipt(base_receipt, plan=plan)
    receipt_hash = receipt.get("receipt_sha256")
    if not isinstance(receipt_hash, str) or receipt_hash != canonical_json_hash(
        receipt, omitted_fields=("receipt_sha256",)
    ):
        raise ValueError("A0DR2 receipt hash changed")
