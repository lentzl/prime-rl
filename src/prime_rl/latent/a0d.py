from __future__ import annotations

import json
import math
import re
from pathlib import Path

from prime_rl.latent.a0 import canonical_json_hash, file_sha256, validate_a0_bank
from prime_rl.latent.a0c import load_and_validate_a0c_plan, validate_a0c_receipt

A0D_PLAN_SCHEMA = "prime-rl/latent-a0d-cache-diagnostic-plan/v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_A0C_RECEIPT_FILE_SHA256 = "d88dd97eb37c9c3dd61bc07fe422df6c7fa0034837897346e43ed16bb634e63c"
_A0C_RECEIPT_INTERNAL_SHA256 = "40dde68d34deb592f864739b48da8c22faafe470f2f0bc6708bce608ae482de7"
_ASSET_PATHS = {
    "experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0c-carrier-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0c-carrier-success-receipt.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0r-mechanism-rejected-failure.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0r-mechanism-rejected.log",
    "scripts/latent/run_a0d_cache_diagnostic_v1.py",
    "scripts/latent/run_a0d_cache_diagnostic_v1.sh",
    "src/prime_rl/latent/__init__.py",
    "src/prime_rl/latent/a0.py",
    "src/prime_rl/latent/a0c.py",
    "src/prime_rl/latent/a0d.py",
    "src/prime_rl/latent/a0r.py",
    "src/prime_rl/latent/policy_adapter.py",
}
_EVIDENCE = {
    "a0r": {
        "status": "mechanism_rejected",
        "failure_file_sha256": "b38319aedb72f1cffe9cf1b4cb90adca430813a24f1d1120efd9213ea0256a20",
        "failure_internal_sha256": "8856b213a9796186ff328ba0d20584387ffeba3554859f430c7a29fe52ac4746",
        "launch_log_sha256": "815da07bedc55794db1cb87cc0222bf18034d074c8678005b04d9227674111fa",
        "plan_sha256": "d920e79d8dfe4334d99dfc875bea9dbc1ba01fdaf6f4d7c028b2b437814efed1",
        "execution_commit": "d2a3c47530db59c4829b579aed8874e92b9b8af2",
        "error": (
            "cached/full disagreement at step 1: length=56, max_abs=0.1875, "
            "normalized_rms=0.015026240609586239, greedy_equal=True"
        ),
    },
    "a0c": {
        "status": "carrier_only_mechanism_validated_for_b_dependency",
        "receipt_file_sha256": _A0C_RECEIPT_FILE_SHA256,
        "receipt_internal_sha256": _A0C_RECEIPT_INTERNAL_SHA256,
        "plan_sha256": "0def45abb5981761eaf34a9228313cf9494e2fdf23c39055a36e095fd70482be",
        "execution_commit": "81e10c5ff362a4ac586fc9ecd81cdfe6f2196583",
        "complete_probes": 4,
    },
}
_DIAGNOSTIC = {
    "example_id": "a0-mechanism-0001",
    "fixture_status": "already_exposed_mechanism_fixture_not_heldout",
    "continuation_text_sha256": "d2a9291c35fc42fadedff20c365f38da2813504f980dd6ba6bdda413a79bd6e0",
    "fixed_continuation_tokens": 4,
    "expected_rendered_lengths": {"parent": 93, "child": 47, "soft_prompt": 55},
    "arms": {
        "D": "real token ids for full recompute and cached prefill/decode",
        "E": "exact token embeddings with explicit sequential prefill/full positions and token-id cached decode",
        "S": "A0C soft-slot embeddings with explicit sequential prefill/full positions and token-id cached decode",
    },
    "position_branches": {
        "auto_position": "prepare_inputs_for_generation without explicit next position",
        "explicit_next_position": "prepare_inputs_for_generation with exact next position_ids and cache_position",
    },
    "fresh_cache_per_arm_and_position_branch": True,
    "reset_rope_state_before_each_arm_and_position_branch": True,
    "record": [
        "cached_vs_full_max_abs",
        "cached_vs_full_normalized_rms",
        "cached_vs_full_greedy_equal",
        "cache_sequence_length",
        "prefill_cache_type_and_logit_hash",
        "cached_and_full_logit_hashes",
        "prepared_keys_and_tensor_metadata",
        "rope_state_before_prefill_after_prefill_after_prepare_after_decode",
        "rendered_prompt_and_continuation_token_hashes",
    ],
    "reference_normalized_rms": 0.01,
    "reference_is_promotion_gate": False,
    "per_layer_diagnostics": False,
    "model_update": False,
    "optimizer": None,
    "checkpoint": False,
}
_INTERPRETATION_BOUNDARY = (
    "non-promotional causal diagnostic only; A0R remains rejected and A1 remains blocked regardless of result; "
    "the 0.01 value is a historical reference, not an A0D gate"
)


def validate_a0d_plan(plan: dict[str, object], *, bank_sha256: str, a0c_plan: dict[str, object]) -> None:
    required = {
        "schema_version",
        "status",
        "execution_authorization",
        "mechanism_code_commit",
        "asset_sha256",
        "plan_sha256",
        "bank_sha256",
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
        raise ValueError("A0D plan fields differ from the v1 schema")
    if plan.get("schema_version") != A0D_PLAN_SCHEMA or plan.get("status") != "preregistered":
        raise ValueError("A0D schema or preregistration status changed")
    if plan.get("execution_authorization") != "root_review_required":
        raise ValueError("A0D execution authorization boundary changed")
    commit = plan.get("mechanism_code_commit")
    if not isinstance(commit, str) or not _GIT_COMMIT_RE.fullmatch(commit):
        raise ValueError("A0D mechanism commit is missing or malformed")
    assets = plan.get("asset_sha256")
    if not isinstance(assets, dict) or set(assets) != _ASSET_PATHS:
        raise ValueError("A0D asset set differs from the freeze")
    if any(not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in assets.values()):
        raise ValueError("A0D asset hash is malformed")
    if plan.get("bank_sha256") != bank_sha256 or not _SHA256_RE.fullmatch(bank_sha256):
        raise ValueError("A0D bank hash differs from the exposed mechanism fixture bank")
    plan_hash = plan.get("plan_sha256")
    if not isinstance(plan_hash, str) or plan_hash != canonical_json_hash(plan, omitted_fields=("plan_sha256",)):
        raise ValueError("A0D canonical plan hash differs")
    if plan.get("bound_evidence") != _EVIDENCE:
        raise ValueError("A0D rejection/success evidence binding changed")
    for field in ("protected_checkpoints", "remote_paths", "runtime"):
        if plan.get(field) != a0c_plan[field]:
            raise ValueError(f"A0D changed frozen A0C field: {field}")
    if plan.get("diagnostic") != _DIAGNOSTIC:
        raise ValueError("A0D causal arms, observables, or no-update contract changed")
    resources = dict(a0c_plan["resource_bounds"])
    resources["output_root"] = "/home/ubuntu/rlm/outputs/latent-a0d-cache-diagnostic-v1"
    if plan.get("resource_bounds") != resources:
        raise ValueError("A0D resources differ beyond the fresh output namespace")
    if plan.get("failure_classification") != {
        "diagnostic_execution_or_finiteness_failure": "diagnostic_incomplete",
        "environment_provenance_timeout_or_oom": "infrastructure_invalid",
        "run_id_reusable": False,
    }:
        raise ValueError("A0D failure classification changed")
    if plan.get("interpretation_boundary") != _INTERPRETATION_BOUNDARY:
        raise ValueError("A0D interpretation boundary changed")


def load_and_validate_a0d_plan(plan_path: Path, bank_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    directory = plan_path.parent
    a0c_plan_path = directory / "a0c-carrier-plan-v1.json"
    receipt_path = directory / "a0c-carrier-success-receipt.json"
    paths = (plan_path, bank_path, a0c_plan_path, receipt_path)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("A0D plan, bank, A0C plan, or success receipt is absent or symlinked")
    a0c_plan, _ = load_and_validate_a0c_plan(a0c_plan_path, bank_path)
    receipt = json.loads(receipt_path.read_text())
    if file_sha256(receipt_path) != _A0C_RECEIPT_FILE_SHA256:
        raise ValueError("A0D bound A0C receipt file hash changed")
    if canonical_json_hash(receipt, omitted_fields=("receipt_sha256",)) != _A0C_RECEIPT_INTERNAL_SHA256:
        raise ValueError("A0D bound A0C receipt internal hash changed")
    validate_a0c_receipt(receipt, plan=a0c_plan)
    plan = json.loads(plan_path.read_text())
    bank = json.loads(bank_path.read_text())
    validate_a0_bank(bank)
    validate_a0d_plan(plan, bank_sha256=file_sha256(bank_path), a0c_plan=a0c_plan)
    return plan, bank


def validate_a0d_receipt(receipt: dict[str, object], *, plan: dict[str, object]) -> None:
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
        "transformers_runtime_sources",
        "model_runtime",
        "gpu",
        "host",
        "protected_hashes_before",
        "protected_hashes_after",
        "checkpoint_metadata_before",
        "checkpoint_metadata_after",
        "fixture",
        "reference_normalized_rms",
        "reference_is_promotion_gate",
        "arms",
        "optimizer_created",
        "checkpoint_created",
        "model_update_attempted",
        "resources",
        "interpretation_boundary",
        "receipt_sha256",
    }
    if set(receipt) != required:
        raise ValueError("A0D receipt fields differ from the diagnostic schema")
    if (
        receipt.get("schema_version") != "prime-rl/latent-a0d-cache-diagnostic-receipt/v1"
        or receipt.get("status") != "diagnostic_complete"
        or receipt.get("claim") != "non-promotional cache causal measurements only"
    ):
        raise ValueError("A0D receipt status or claim changed")
    for field in ("plan_sha256", "bank_sha256", "mechanism_code_commit", "asset_sha256", "interpretation_boundary"):
        if receipt.get(field) != plan[field]:
            raise ValueError(f"A0D receipt differs from plan: {field}")
    commit = receipt.get("execution_commit")
    if not isinstance(commit, str) or not _GIT_COMMIT_RE.fullmatch(commit):
        raise ValueError("A0D execution commit is malformed")
    if (
        receipt.get("reference_normalized_rms") != 0.01
        or receipt.get("reference_is_promotion_gate") is not False
        or receipt.get("optimizer_created") is not False
        or receipt.get("checkpoint_created") is not False
        or receipt.get("model_update_attempted") is not False
    ):
        raise ValueError("A0D reference or no-update boundary changed")
    versions = receipt.get("versions")
    if not isinstance(versions, dict) or (
        set(versions) != {"python", "transformers", "torch_distribution", "torch_runtime"}
        or not isinstance(versions.get("python"), str)
        or not versions["python"].startswith(f"{plan['runtime']['python']}.")
        or versions.get("transformers") != plan["runtime"]["transformers"]
        or versions.get("torch_distribution") != plan["runtime"]["torch_distribution"]
        or versions.get("torch_runtime") != plan["runtime"]["torch_runtime"]
    ):
        raise ValueError("A0D receipt runtime versions differ from freeze")
    if receipt.get("model_runtime") != {
        "class": plan["runtime"]["model_class"],
        "hidden_size": plan["runtime"]["hidden_size"],
        "device": plan["runtime"]["device"],
        "dtype": plan["runtime"]["dtype"],
    }:
        raise ValueError("A0D receipt model runtime differs from freeze")
    sources = receipt.get("transformers_runtime_sources")
    if not isinstance(sources, dict) or set(sources) != set(plan["runtime"]["transformers_source_sha256"]):
        raise ValueError("A0D receipt Transformers source set differs")
    for name, expected_hash in plan["runtime"]["transformers_source_sha256"].items():
        source = sources.get(name)
        if not isinstance(source, dict) or (
            source.get("sha256") != expected_hash
            or not isinstance(source.get("path"), str)
            or not source["path"].startswith("/home/ubuntu/rlm/prime-rl/.venv/")
        ):
            raise ValueError("A0D receipt Transformers source identity differs")
    gpu = receipt.get("gpu")
    host = receipt.get("host")
    if not isinstance(gpu, dict) or (
        gpu.get("name") != plan["resource_bounds"]["gpu_model"]
        or not isinstance(gpu.get("total_memory_gib"), (int, float))
        or not math.isfinite(gpu["total_memory_gib"])
        or gpu["total_memory_gib"] < plan["resource_bounds"]["minimum_gpu_memory_gib"]
    ):
        raise ValueError("A0D receipt GPU differs from freeze")
    if not isinstance(host, dict) or (
        not isinstance(host.get("ram_bytes"), int)
        or host["ram_bytes"] < plan["resource_bounds"]["minimum_host_ram_gib"] * 2**30
        or not isinstance(host.get("free_disk_bytes_before"), int)
        or host["free_disk_bytes_before"] < plan["resource_bounds"]["minimum_free_disk_gib"] * 2**30
    ):
        raise ValueError("A0D receipt host resources differ from freeze")
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
        raise ValueError("A0D receipt does not preserve canonical checkpoints")
    fixture = receipt.get("fixture")
    if not isinstance(fixture, dict) or (
        fixture.get("example_id") != plan["diagnostic"]["example_id"]
        or fixture.get("parent_token_count") != plan["diagnostic"]["expected_rendered_lengths"]["parent"]
        or fixture.get("child_token_count") != plan["diagnostic"]["expected_rendered_lengths"]["child"]
        or fixture.get("soft_prompt_length") != plan["diagnostic"]["expected_rendered_lengths"]["soft_prompt"]
        or not isinstance(fixture.get("continuation_token_ids"), list)
        or len(fixture["continuation_token_ids"]) != plan["diagnostic"]["fixed_continuation_tokens"]
        or any(
            not isinstance(fixture.get(field), str) or not _SHA256_RE.fullmatch(fixture[field])
            for field in (
                "child_input_ids_sha256",
                "continuation_input_ids_sha256",
                "workspace_source_sha256",
                "soft_prompt_sha256",
            )
        )
    ):
        raise ValueError("A0D receipt fixture evidence changed")
    arms = receipt.get("arms")
    expected_pairs = {
        (arm, branch) for arm in ("D", "E", "S") for branch in ("auto_position", "explicit_next_position")
    }
    if (
        not isinstance(arms, list)
        or len(arms) != 6
        or {(arm.get("arm"), arm.get("position_branch")) for arm in arms if isinstance(arm, dict)} != expected_pairs
    ):
        raise ValueError("A0D receipt arm/position matrix is incomplete")
    for arm in arms:
        if (
            not isinstance(arm, dict)
            or set(arm)
            != {
                "arm",
                "position_branch",
                "fresh_cache",
                "initial_cache_sequence_length",
                "initial_logits_finite",
                "prefill_cache_type",
                "prefill_last_logits_sha256",
                "rope_state",
                "steps",
            }
            or arm.get("fresh_cache") is not True
            or arm.get("initial_logits_finite") is not True
            or not isinstance(arm.get("prefill_cache_type"), str)
            or not isinstance(arm.get("prefill_last_logits_sha256"), str)
            or not _SHA256_RE.fullmatch(arm["prefill_last_logits_sha256"])
        ):
            raise ValueError("A0D arm lacks fresh finite cache evidence")
        initial = arm.get("initial_cache_sequence_length")
        steps = arm.get("steps")
        expected_initial = fixture["soft_prompt_length"] if arm["arm"] == "S" else fixture["child_token_count"]
        arm_rope = arm.get("rope_state")
        if (
            not isinstance(initial, int)
            or initial != expected_initial
            or not isinstance(steps, list)
            or len(steps) != 4
            or not isinstance(arm_rope, dict)
            or set(arm_rope) != {"before_prefill", "after_prefill"}
        ):
            raise ValueError("A0D arm cache length or continuation is malformed")
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict) or (
                set(step)
                != {
                    "step",
                    "cache_sequence_length",
                    "maximum_absolute_logit_difference",
                    "normalized_rms",
                    "greedy_equal",
                    "cached_logits_sha256",
                    "full_logits_sha256",
                    "prepared",
                    "rope_state",
                }
                or step.get("step") != index
                or step.get("cache_sequence_length") != initial + index
                or not isinstance(step.get("maximum_absolute_logit_difference"), (int, float))
                or not isinstance(step.get("normalized_rms"), (int, float))
                or not isinstance(step.get("greedy_equal"), bool)
                or not isinstance(step.get("cached_logits_sha256"), str)
                or not _SHA256_RE.fullmatch(step["cached_logits_sha256"])
                or not isinstance(step.get("full_logits_sha256"), str)
                or not _SHA256_RE.fullmatch(step["full_logits_sha256"])
            ):
                raise ValueError("A0D step evidence is incomplete")
            if not all(math.isfinite(step[field]) for field in ("maximum_absolute_logit_difference", "normalized_rms")):
                raise ValueError("A0D step metric is non-finite")
            prepared = step.get("prepared")
            if not isinstance(prepared, dict) or not isinstance(prepared.get("keys"), list):
                raise ValueError("A0D prepared-input evidence is absent")
            keys = set(prepared["keys"])
            if not {"input_ids", "past_key_values", "attention_mask", "use_cache"}.issubset(keys):
                raise ValueError("A0D prepared inputs omit a required decode input")
            if arm["position_branch"] == "auto_position" and {"position_ids", "cache_position"} & keys:
                raise ValueError("A0D auto-position branch contains explicit positions")
            if arm["position_branch"] == "explicit_next_position" and not {
                "position_ids",
                "cache_position",
            }.issubset(keys):
                raise ValueError("A0D explicit-position branch lacks positions")
            prepared_values = prepared.get("values")
            if (
                not isinstance(prepared_values, dict)
                or prepared_values.get("past_key_values", {}).get("sequence_length") != initial + index - 1
            ):
                raise ValueError("A0D prepared cache provenance is malformed")
            expected_token = fixture["continuation_token_ids"][index - 1]
            if prepared_values.get("input_ids", {}).get("values") != [expected_token]:
                raise ValueError("A0D prepared token differs from the fixed continuation")
            if arm["position_branch"] == "explicit_next_position":
                expected_position = [initial + index - 1]
                if (
                    prepared_values.get("position_ids", {}).get("values") != expected_position
                    or prepared_values.get("cache_position", {}).get("values") != expected_position
                ):
                    raise ValueError("A0D explicit prepared position value is incorrect")
            rope = step.get("rope_state")
            if not isinstance(rope, dict) or set(rope) != {"before_prepare", "after_prepare", "after_decode"}:
                raise ValueError("A0D rope-state transition evidence is absent")
    receipt_hash = receipt.get("receipt_sha256")
    if not isinstance(receipt_hash, str) or receipt_hash != canonical_json_hash(
        receipt, omitted_fields=("receipt_sha256",)
    ):
        raise ValueError("A0D receipt hash is absent or incorrect")
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
        raise ValueError("A0D receipt resource evidence is malformed")
