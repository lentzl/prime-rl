import copy
import json
from pathlib import Path

import pytest

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a0r import validate_a0r_plan

EXPERIMENT = Path("experiments/qwen35-2b-latent-workspace-v1")


def _plan(base_plan: dict[str, object], bank_hash: str) -> dict[str, object]:
    runtime = copy.deepcopy(base_plan["runtime"])
    runtime.pop("torch")
    runtime["torch_distribution"] = "2.11.0+cu128"
    runtime["torch_runtime"] = "2.11.0+cu128"
    resources = copy.deepcopy(base_plan["resource_bounds"])
    resources["output_root"] = "/home/ubuntu/rlm/outputs/latent-a0r-mechanism-v1"
    plan: dict[str, object] = {
        "schema_version": "prime-rl/latent-a0r-mechanism-plan/v1",
        "status": "preregistered",
        "execution_authorization": "root_review_required",
        "mechanism_code_commit": "a" * 40,
        "asset_sha256": {
            "experiments/qwen35-2b-latent-workspace-v1/a0-failed-runtime-version-failure.json": "1" * 64,
            "experiments/qwen35-2b-latent-workspace-v1/a0-failed-runtime-version.log": "2" * 64,
            "experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-bank-v1.json": "3" * 64,
            "experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-plan-v1.json": "4" * 64,
            "scripts/latent/run_a0_mechanism_v1.py": "5" * 64,
            "scripts/latent/run_a0r_mechanism_v1.sh": "6" * 64,
            "src/prime_rl/latent/__init__.py": "7" * 64,
            "src/prime_rl/latent/a0.py": "8" * 64,
            "src/prime_rl/latent/a0r.py": "9" * 64,
            "src/prime_rl/latent/policy_adapter.py": "a" * 64,
        },
        "plan_sha256": "",
        "bank_sha256": bank_hash,
        "supersedes_plan_sha256": "444e10f6f9ca92f19bba5c393eb8308a8cd3674970b9695d0854ab2f1f39d9a8",
        "prior_failed_start_evidence": {
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
        },
        "protected_checkpoints": copy.deepcopy(base_plan["protected_checkpoints"]),
        "remote_paths": copy.deepcopy(base_plan["remote_paths"]),
        "runtime": runtime,
        "mechanism": copy.deepcopy(base_plan["mechanism"]),
        "admission": copy.deepcopy(base_plan["admission"]),
        "resource_bounds": resources,
        "failure_classification": copy.deepcopy(base_plan["failure_classification"]),
        "promotion_boundary": base_plan["promotion_boundary"],
    }
    plan["plan_sha256"] = canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    return plan


def _inputs() -> tuple[dict[str, object], dict[str, object], str, str, str]:
    base_plan = json.loads((EXPERIMENT / "a0-mechanism-plan-v1.json").read_text())
    failure_path = EXPERIMENT / "a0-failed-runtime-version-failure.json"
    log_path = EXPERIMENT / "a0-failed-runtime-version.log"
    failure = json.loads(failure_path.read_text())
    bank_hash = file_sha256(EXPERIMENT / "a0-mechanism-bank-v1.json")
    return base_plan, failure, bank_hash, file_sha256(failure_path), file_sha256(log_path)


def test_a0r_accepts_only_exact_runtime_repair_and_bound_failed_start() -> None:
    base_plan, failure, bank_hash, failure_hash, log_hash = _inputs()
    validate_a0r_plan(
        _plan(base_plan, bank_hash),
        bank_sha256=bank_hash,
        base_plan=base_plan,
        failure=failure,
        failure_file_sha256=failure_hash,
        launch_log_sha256=log_hash,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: plan["runtime"].update(torch_distribution="2.11.0"),
        lambda plan: plan["prior_failed_start_evidence"].update(model_loaded=True),
        lambda plan: plan["resource_bounds"].update(output_root="/home/ubuntu/rlm/outputs/latent-a0-mechanism-v1"),
    ],
)
def test_a0r_rejects_runtime_evidence_or_namespace_drift(mutation) -> None:
    base_plan, failure, bank_hash, failure_hash, log_hash = _inputs()
    plan = _plan(base_plan, bank_hash)
    mutation(plan)
    plan["plan_sha256"] = canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    with pytest.raises(ValueError):
        validate_a0r_plan(
            plan,
            bank_sha256=bank_hash,
            base_plan=base_plan,
            failure=failure,
            failure_file_sha256=failure_hash,
            launch_log_sha256=log_hash,
        )
