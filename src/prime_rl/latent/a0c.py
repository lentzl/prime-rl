from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

from prime_rl.latent.a0 import canonical_json_hash, file_sha256, validate_a0_bank
from prime_rl.latent.a0r import load_and_validate_a0r_plan

A0C_PLAN_SCHEMA = "prime-rl/latent-a0c-carrier-plan/v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ASSET_PATHS = {
    "experiments/qwen35-2b-latent-workspace-v1/a0-failed-runtime-version-failure.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0-failed-runtime-version.log",
    "experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0r-mechanism-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0r-mechanism-rejected-failure.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0r-mechanism-rejected.log",
    "scripts/latent/run_a0_mechanism_v1.py",
    "scripts/latent/run_a0c_carrier_v1.sh",
    "src/prime_rl/latent/__init__.py",
    "src/prime_rl/latent/a0.py",
    "src/prime_rl/latent/a0c.py",
    "src/prime_rl/latent/a0r.py",
    "src/prime_rl/latent/policy_adapter.py",
}
_REJECTION_EVIDENCE = {
    "schema_version": "prime-rl/latent-a0r-rejection-evidence/v1",
    "failure_file_sha256": "b38319aedb72f1cffe9cf1b4cb90adca430813a24f1d1120efd9213ea0256a20",
    "failure_internal_sha256": "8856b213a9796186ff328ba0d20584387ffeba3554859f430c7a29fe52ac4746",
    "launch_log_sha256": "815da07bedc55794db1cb87cc0222bf18034d074c8678005b04d9227674111fa",
    "execution_commit": "d2a3c47530db59c4829b579aed8874e92b9b8af2",
    "mechanism_commit": "5abc829ff04d7fc39f3e43a5b55f28eff7c61799",
    "plan_sha256": "d920e79d8dfe4334d99dfc875bea9dbc1ba01fdaf6f4d7c028b2b437814efed1",
    "status": "mechanism_rejected",
    "failure_category": "mechanism_predicate_failure",
    "stage": "probe_a0-mechanism-0001",
    "exact_error": (
        "cached/full disagreement at step 1: length=56, max_abs=0.1875, "
        "normalized_rms=0.015026240609586239, greedy_equal=True"
    ),
    "complete_probes": 0,
    "posthoc_partial_receipt_allowed": False,
}
_FIXTURE_REUSE = {
    "bank_kind": "mechanism-only synthetic fixtures, not held-out semantic evaluation",
    "reason": "A0C repeats pre-cache engineering predicates only and makes no capability or communication claim",
    "minimum_complete_probes": 4,
}
_DEPENDENCY_SCOPE = {
    "may_support": ["B frozen-decoder teacher-forced carrier dependency"],
    "does_not_support": [
        "A0 cache mechanism",
        "A1 latent bridge",
        "A1 semantic communication",
        "generation or cached decoding",
        "model admission",
    ],
}


def validate_a0c_plan(
    plan: dict[str, object],
    *,
    bank_sha256: str,
    base_plan: dict[str, object],
    rejection: dict[str, object],
    rejection_file_sha256: str,
    rejection_log_sha256: str,
) -> None:
    required = {
        "schema_version",
        "status",
        "execution_authorization",
        "mechanism_code_commit",
        "asset_sha256",
        "plan_sha256",
        "bank_sha256",
        "derived_from_rejected_plan_sha256",
        "prior_rejection_evidence",
        "fixture_reuse",
        "protected_checkpoints",
        "remote_paths",
        "runtime",
        "mechanism",
        "admission",
        "resource_bounds",
        "failure_classification",
        "dependency_scope",
        "promotion_boundary",
    }
    if set(plan) != required:
        raise ValueError("A0C plan fields differ from the v1 schema")
    if plan.get("schema_version") != A0C_PLAN_SCHEMA or plan.get("status") != "preregistered":
        raise ValueError("A0C schema or preregistration status changed")
    if plan.get("execution_authorization") != "root_review_required":
        raise ValueError("A0C execution authorization boundary changed")
    commit = plan.get("mechanism_code_commit")
    if not isinstance(commit, str) or not _GIT_COMMIT_RE.fullmatch(commit):
        raise ValueError("A0C mechanism commit is missing or malformed")
    assets = plan.get("asset_sha256")
    if not isinstance(assets, dict) or set(assets) != _ASSET_PATHS:
        raise ValueError("A0C executable asset set differs from the freeze")
    if any(not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in assets.values()):
        raise ValueError("A0C asset hash is malformed")
    if plan.get("bank_sha256") != bank_sha256 or not _SHA256_RE.fullmatch(bank_sha256):
        raise ValueError("A0C bank hash differs from the frozen mechanism fixtures")
    plan_hash = plan.get("plan_sha256")
    if not isinstance(plan_hash, str) or not _SHA256_RE.fullmatch(plan_hash):
        raise ValueError("A0C plan hash is missing or malformed")
    if plan_hash != canonical_json_hash(plan, omitted_fields=("plan_sha256",)):
        raise ValueError("A0C plan hash does not match canonical content")
    if plan.get("derived_from_rejected_plan_sha256") != base_plan["plan_sha256"]:
        raise ValueError("A0C does not point to the rejected A0R plan")
    if plan.get("prior_rejection_evidence") != _REJECTION_EVIDENCE:
        raise ValueError("A0C rejection evidence changed")
    if rejection_file_sha256 != _REJECTION_EVIDENCE["failure_file_sha256"]:
        raise ValueError("A0C bound A0R failure artifact hash changed")
    if rejection_log_sha256 != _REJECTION_EVIDENCE["launch_log_sha256"]:
        raise ValueError("A0C bound A0R launch log hash changed")
    if (
        canonical_json_hash(rejection, omitted_fields=("failure_sha256",))
        != _REJECTION_EVIDENCE["failure_internal_sha256"]
    ):
        raise ValueError("A0C bound A0R failure internal hash changed")
    expected_rejection = {
        "error": _REJECTION_EVIDENCE["exact_error"],
        "error_type": "MechanismRejected",
        "execution_commit": _REJECTION_EVIDENCE["execution_commit"],
        "failure_category": _REJECTION_EVIDENCE["failure_category"],
        "failure_sha256": _REJECTION_EVIDENCE["failure_internal_sha256"],
        "mechanism_code_commit": _REJECTION_EVIDENCE["mechanism_commit"],
        "model_update_attempted": False,
        "plan_sha256": _REJECTION_EVIDENCE["plan_sha256"],
        "schema_version": "prime-rl/latent-a0r-mechanism-failure/v1",
        "stage": _REJECTION_EVIDENCE["stage"],
        "status": _REJECTION_EVIDENCE["status"],
    }
    if any(rejection.get(key) != value for key, value in expected_rejection.items()):
        raise ValueError("A0C A0R artifact does not prove the frozen rejection facts")
    protected_after = rejection.get("protected_hash_probe_after_failure")
    if not isinstance(protected_after, dict):
        raise ValueError("A0C A0R artifact has no protected post-failure evidence")
    for name, expected_hash in base_plan["protected_checkpoints"].items():
        observed = protected_after.get(name)
        if not isinstance(observed, dict) or (
            observed.get("model_sha256") != expected_hash
            or observed.get("metadata_sha256") != base_plan["runtime"]["checkpoint_metadata_sha256"]
        ):
            raise ValueError("A0C A0R artifact does not preserve canonical checkpoints")
    if plan.get("fixture_reuse") != _FIXTURE_REUSE:
        raise ValueError("A0C fixture-reuse rationale changed")
    if plan.get("dependency_scope") != _DEPENDENCY_SCOPE:
        raise ValueError("A0C dependency scope changed")

    for field in ("protected_checkpoints", "remote_paths", "runtime", "failure_classification"):
        if plan.get(field) != base_plan[field]:
            raise ValueError(f"A0C changed frozen A0R field: {field}")
    expected_mechanism = copy.deepcopy(base_plan["mechanism"])
    expected_mechanism.pop("cache_full_recompute_max_abs")
    expected_mechanism.pop("cache_full_recompute_normalized_rms")
    if plan.get("mechanism") != expected_mechanism:
        raise ValueError("A0C mechanism differs beyond removal of cache comparison")
    expected_admission = copy.deepcopy(base_plan["admission"])
    for field in (
        "cache_prefill_finite",
        "cache_sequence_length_increments_exactly",
        "four_step_decode_finite",
        "cached_vs_full_max_abs_at_most",
        "cached_vs_full_normalized_rms_at_most",
        "cached_vs_full_greedy_tokens_equal",
    ):
        expected_admission.pop(field)
    if plan.get("admission") != expected_admission:
        raise ValueError("A0C admission differs from the exact pre-cache predicate set")
    expected_resources = copy.deepcopy(base_plan["resource_bounds"])
    expected_resources["output_root"] = "/home/ubuntu/rlm/outputs/latent-a0c-carrier-v1"
    if plan.get("resource_bounds") != expected_resources:
        raise ValueError("A0C resources differ beyond the fresh output namespace")
    if plan.get("promotion_boundary") != (
        "A0C carrier-only dependency evidence for B; no A0/A1, bridge, semantic communication, generation, cache, "
        "training, or model admission claim"
    ):
        raise ValueError("A0C promotion boundary changed")


def validate_a0c_receipt(receipt: dict[str, object], *, plan: dict[str, object]) -> None:
    """Fail closed on the carrier-only claim and atomic four-probe evidence."""
    required = {
        "schema_version",
        "status",
        "claim",
        "plan_sha256",
        "bank_sha256",
        "mechanism_code_commit",
        "execution_commit",
        "asset_sha256",
        "versions",
        "model_runtime",
        "transformers_runtime_sources",
        "gpu",
        "host",
        "checkpoint_metadata_before",
        "checkpoint_metadata_after",
        "protected_hashes_before",
        "protected_hashes_after",
        "complete_probes",
        "probes",
        "resources",
        "optimizer_created",
        "checkpoint_created",
        "artifact_contract",
        "a1_blocker",
        "promotion_boundary",
        "dependency_scope",
        "receipt_sha256",
    }
    if set(receipt) != required:
        raise ValueError("A0C receipt fields differ from the carrier-only schema")
    if receipt.get("schema_version") != "prime-rl/latent-a0c-mechanism-receipt/v1":
        raise ValueError("A0C receipt schema changed")
    if receipt.get("status") != "carrier_only_mechanism_validated_for_b_dependency":
        raise ValueError("A0C receipt status changed")
    if receipt.get("claim") != "four-probe pre-cache carrier/autograd validation for B dependency only":
        raise ValueError("A0C receipt claim changed")
    if receipt.get("dependency_scope") != _DEPENDENCY_SCOPE:
        raise ValueError("A0C receipt dependency scope changed")
    for field in ("plan_sha256", "bank_sha256", "mechanism_code_commit", "asset_sha256", "promotion_boundary"):
        if receipt.get(field) != plan[field]:
            raise ValueError(f"A0C receipt differs from its frozen plan: {field}")
    execution_commit = receipt.get("execution_commit")
    if not isinstance(execution_commit, str) or not _GIT_COMMIT_RE.fullmatch(execution_commit):
        raise ValueError("A0C receipt execution commit is malformed")
    versions = receipt.get("versions")
    if not isinstance(versions, dict) or (
        set(versions) != {"python", "transformers", "torch_distribution", "torch_runtime"}
        or not isinstance(versions.get("python"), str)
        or not versions["python"].startswith(f"{plan['runtime']['python']}.")
        or versions.get("transformers") != plan["runtime"]["transformers"]
        or versions.get("torch_distribution") != plan["runtime"]["torch_distribution"]
        or versions.get("torch_runtime") != plan["runtime"]["torch_runtime"]
    ):
        raise ValueError("A0C receipt runtime identity differs from the freeze")
    if receipt.get("model_runtime") != {
        "class": plan["runtime"]["model_class"],
        "hidden_size": plan["runtime"]["hidden_size"],
        "device": plan["runtime"]["device"],
        "dtype": plan["runtime"]["dtype"],
    }:
        raise ValueError("A0C receipt model runtime identity differs from the freeze")
    sources = receipt.get("transformers_runtime_sources")
    source_suffixes = {
        "transformers.cache_utils": "/lib/python3.12/site-packages/transformers/cache_utils.py",
        "transformers.generation.utils": "/lib/python3.12/site-packages/transformers/generation/utils.py",
        "transformers.models.qwen3_5.modeling_qwen3_5": (
            "/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py"
        ),
    }
    if not isinstance(sources, dict) or set(sources) != set(source_suffixes):
        raise ValueError("A0C receipt Transformers source set changed")
    for name, suffix in source_suffixes.items():
        source = sources.get(name)
        if not isinstance(source, dict) or (
            source.get("sha256") != plan["runtime"]["transformers_source_sha256"][name]
            or not isinstance(source.get("path"), str)
            or not source["path"].startswith("/home/ubuntu/rlm/prime-rl/.venv/")
            or not source["path"].endswith(suffix)
        ):
            raise ValueError("A0C receipt Transformers source identity changed")
    expected_metadata = {
        "coordinator_e33": plan["runtime"]["checkpoint_metadata_sha256"],
        "worker_h176": plan["runtime"]["checkpoint_metadata_sha256"],
    }
    if (
        receipt.get("protected_hashes_before") != plan["protected_checkpoints"]
        or receipt.get("protected_hashes_after") != plan["protected_checkpoints"]
        or receipt.get("checkpoint_metadata_before") != expected_metadata
        or receipt.get("checkpoint_metadata_after") != expected_metadata
    ):
        raise ValueError("A0C receipt does not preserve canonical checkpoint identities")
    gpu = receipt.get("gpu")
    host = receipt.get("host")
    if not isinstance(gpu, dict) or (
        gpu.get("name") != plan["resource_bounds"]["gpu_model"]
        or not isinstance(gpu.get("total_memory_gib"), (int, float))
        or not math.isfinite(gpu["total_memory_gib"])
        or gpu["total_memory_gib"] < plan["resource_bounds"]["minimum_gpu_memory_gib"]
    ):
        raise ValueError("A0C receipt GPU identity or capacity changed")
    if not isinstance(host, dict) or (
        not isinstance(host.get("ram_bytes"), int)
        or host["ram_bytes"] < plan["resource_bounds"]["minimum_host_ram_gib"] * 2**30
        or not isinstance(host.get("free_disk_bytes_before"), int)
        or host["free_disk_bytes_before"] < plan["resource_bounds"]["minimum_free_disk_gib"] * 2**30
    ):
        raise ValueError("A0C receipt host capacity changed")
    if receipt.get("optimizer_created") is not False or receipt.get("checkpoint_created") is not False:
        raise ValueError("A0C receipt indicates forbidden optimizer or checkpoint creation")
    resources = receipt.get("resources")
    if (
        not isinstance(resources, dict)
        or set(resources) != {"wall_seconds", "peak_cuda_memory_bytes"}
        or (
            not isinstance(resources.get("wall_seconds"), (int, float))
            or not math.isfinite(resources["wall_seconds"])
            or resources["wall_seconds"] <= 0
            or not isinstance(resources.get("peak_cuda_memory_bytes"), int)
            or resources["peak_cuda_memory_bytes"] <= 0
        )
    ):
        raise ValueError("A0C receipt resource evidence is malformed")
    if receipt.get("artifact_contract") != {
        "expected_files": ["receipt.json"],
        "maximum_directory_bytes": plan["resource_bounds"]["maximum_output_directory_bytes"],
    }:
        raise ValueError("A0C receipt artifact contract changed")
    if receipt.get("a1_blocker") != (
        "A0R cached/full gate remains rejected; live typed-harness action acceptance and capture timing remain "
        "unvalidated"
    ):
        raise ValueError("A0C receipt does not preserve both A0/A1 blockers")
    receipt_hash = receipt.get("receipt_sha256")
    if not isinstance(receipt_hash, str) or receipt_hash != canonical_json_hash(
        receipt, omitted_fields=("receipt_sha256",)
    ):
        raise ValueError("A0C receipt hash is missing or incorrect")
    if receipt.get("complete_probes") != 4:
        raise ValueError("A0C requires exactly four complete probes")
    probes = receipt.get("probes")
    if not isinstance(probes, list) or len(probes) != 4:
        raise ValueError("A0C receipt requires one atomic four-probe set")
    expected_ids = tuple(f"a0-mechanism-{index:04d}" for index in range(1, 5))
    if tuple(probe.get("example_id") for probe in probes if isinstance(probe, dict)) != expected_ids:
        raise ValueError("A0C receipt probe order or identity changed")
    for probe in probes:
        if not isinstance(probe, dict) or set(probe) != {
            "example_id",
            "status",
            "prompt_tokens",
            "hard_bypass",
            "capture",
            "soft_insertion",
        }:
            raise ValueError("A0C probe fields differ from the carrier-only schema")
        if probe.get("status") != "complete":
            raise ValueError("A0C contains an incomplete probe")
        hard = probe.get("hard_bypass")
        capture = probe.get("capture")
        soft = probe.get("soft_insertion")
        prompt_tokens = probe.get("prompt_tokens")
        if (
            not isinstance(prompt_tokens, dict)
            or set(prompt_tokens)
            != {
                "parent",
                "child_without_assistant_opening",
                "child_with_assistant_opening",
            }
            or any(not isinstance(value, int) or value <= 0 for value in prompt_tokens.values())
        ):
            raise ValueError("A0C prompt-token evidence is malformed")
        if not isinstance(hard, dict) or (
            hard.get("bitwise_equal") is not True
            or hard.get("logits_finite") is not True
            or hard.get("maximum_absolute_logit_difference") != 0.0
            or hard.get("additional_positions") != 0
            or hard.get("labels_mask_positions_preserved") is not True
            or hard.get("four_token_greedy_continuation_equal") is not True
            or hard.get("four_token_greedy_logits_finite") is not True
            or not isinstance(hard.get("greedy_token_ids"), list)
            or len(hard["greedy_token_ids"]) != 4
            or any(not isinstance(token, int) for token in hard["greedy_token_ids"])
        ):
            raise ValueError("A0C hard-bypass predicate is incomplete or false")
        if not isinstance(capture, dict) or (
            capture.get("claim_boundary") != "rendered_transcript_end_not_harness_acceptance"
            or capture.get("finite") is not True
            or capture.get("detached") is not True
            or capture.get("repeat_bitwise_equal") is not True
            or capture.get("mask_and_indices_exact") is not True
            or capture.get("hidden_padding_content_exact") is not True
            or capture.get("layer") != -1
            or capture.get("shape") != [1, 128, plan["runtime"]["hidden_size"]]
            or not isinstance(capture.get("tensor_bytes_sha256"), str)
            or not _SHA256_RE.fullmatch(capture["tensor_bytes_sha256"])
            or not isinstance(capture.get("capture_spec_sha256"), str)
            or not _SHA256_RE.fullmatch(capture["capture_spec_sha256"])
        ):
            raise ValueError("A0C capture predicate is incomplete or false")
        workspace_span = soft.get("workspace_span") if isinstance(soft, dict) else None
        if not isinstance(soft, dict) or (
            soft.get("claim") != "carrier_and_autograd_connectivity_only"
            or soft.get("positions") != 8
            or soft.get("logits_finite") is not True
            or soft.get("loss_finite") is not True
            or soft.get("original_tokens_mask_labels_preserved") is not True
            or soft.get("positions_sequential_shifted") is not True
            or soft.get("inserted_attention_mask_ones") != 8
            or soft.get("inserted_loss_mask_negative_100") != 8
            or soft.get("no_other_loss_masking") is not True
            or soft.get("workspace_gradient_finite_nonzero") is not True
            or soft.get("gate_gradient_finite_nonzero") is not True
            or soft.get("base_parameter_gradients") != 0
            or not isinstance(workspace_span, list)
            or len(workspace_span) != 2
            or any(not isinstance(position, int) for position in workspace_span)
            or workspace_span[1] - workspace_span[0] != 8
        ):
            raise ValueError("A0C soft-carrier predicate is incomplete or false")


def load_and_validate_a0c_plan(plan_path: Path, bank_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    base_plan_path = plan_path.with_name("a0r-mechanism-plan-v1.json")
    rejection_path = plan_path.with_name("a0r-mechanism-rejected-failure.json")
    rejection_log_path = plan_path.with_name("a0r-mechanism-rejected.log")
    paths = (plan_path, bank_path, base_plan_path, rejection_path, rejection_log_path)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("A0C plan, bank, A0R plan, or rejection evidence is absent or symlinked")
    base_plan, _ = load_and_validate_a0r_plan(base_plan_path, bank_path)
    plan = json.loads(plan_path.read_text())
    bank = json.loads(bank_path.read_text())
    rejection = json.loads(rejection_path.read_text())
    validate_a0_bank(bank)
    validate_a0c_plan(
        plan,
        bank_sha256=file_sha256(bank_path),
        base_plan=base_plan,
        rejection=rejection,
        rejection_file_sha256=file_sha256(rejection_path),
        rejection_log_sha256=file_sha256(rejection_log_path),
    )
    return plan, bank
