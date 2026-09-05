from __future__ import annotations

import json
import math
import re
from pathlib import Path

from prime_rl.latent.a0 import canonical_json_hash, file_sha256, validate_a0_bank
from prime_rl.latent.a0d import load_and_validate_a0d_plan

A0DR_PLAN_SCHEMA = "prime-rl/latent-a0dr-cache-diagnostic-plan/v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_FIXED_IDS = [40, 4021, 2528, 8976, 35139, 635, 524, 599]
_FIXED_IDS_SHA256 = "e86e01e61315008783cc217a5bb83a1b3aced0daaecbc920b8d3b45ab4b205d8"
_ASSET_PATHS = {
    "experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0c-carrier-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0c-carrier-success-receipt.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0d-cache-diagnostic-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0r-mechanism-rejected-failure.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0r-mechanism-rejected.log",
    "scripts/latent/run_a0dr_cache_diagnostic_v1.py",
    "scripts/latent/run_a0dr_cache_diagnostic_v1.sh",
    "src/prime_rl/latent/__init__.py",
    "src/prime_rl/latent/a0.py",
    "src/prime_rl/latent/a0c.py",
    "src/prime_rl/latent/a0d.py",
    "src/prime_rl/latent/a0dr.py",
    "src/prime_rl/latent/a0r.py",
    "src/prime_rl/latent/policy_adapter.py",
}
_PRIOR_NO_GO = {
    "status": "evaluator_no_go_unlaunched",
    "mechanism_commit": "ed77eacb85b0751d10a0a5c5ddc18a6de185b237",
    "freeze_commit": "2c693e25e736b737e5e39fcc6980184e90eb8013",
    "plan_sha256": "f1e96ad0fa66da2bc4c3629b07870a01382e8d2b3c22d122a78a9c386c1f54ab",
    "plan_file_sha256": "f415990d7dbf0df98fa0835020b5e4e5a0147742e9d4d340503f36c5e17a41a4",
    "model_exposure": False,
    "reason": "D/E prefill length 47 versus S length 55 confounds soft content with hybrid chunk padding/length",
}
_DIAGNOSTIC = {
    "example_id": "a0-mechanism-0001",
    "fixture_status": "already_exposed_mechanism_fixture_not_heldout",
    "continuation_text_sha256": "d2a9291c35fc42fadedff20c365f38da2813504f980dd6ba6bdda413a79bd6e0",
    "fixed_continuation_tokens": 4,
    "injection_index": 40,
    "length_control_token_ids": _FIXED_IDS,
    "length_control_token_ids_sha256": _FIXED_IDS_SHA256,
    "length_control_tokens_must_be_non_special": True,
    "causal_core_prefill_length": 55,
    "expected_rendered_lengths": {"parent": 93, "child": 47, "soft_prompt": 55},
    "arms": {
        "D47": "optional length-sensitivity control: original real token ids with inferred prefill/full positions",
        "E47": "optional length-sensitivity control: original exact embeddings with explicit sequential positions",
        "L_ID55": "causal core: eight fixed ordinary IDs inserted at boundary with mask one and sequential positions",
        "L_E55": "causal core: exact embeddings of the identical length-55 discrete sequence and positions",
        "S55": "causal core: A0C soft-slot embeddings at the identical length, boundary, mask, and positions",
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
        "rendered_prompt_continuation_and_length_control_hashes",
    ],
    "reference_normalized_rms": 0.01,
    "reference_is_promotion_gate": False,
    "per_layer_diagnostics": False,
    "model_update": False,
    "optimizer": None,
    "checkpoint": False,
}
_INTERPRETATION = (
    "non-promotional length-matched causal diagnostic only; A0R remains rejected and A1 remains blocked regardless "
    "of result; the 0.01 value is historical reference only"
)


def validate_a0dr_plan(plan: dict[str, object], *, bank_sha256: str, a0d_plan: dict[str, object]) -> None:
    required = {
        "schema_version",
        "status",
        "execution_authorization",
        "mechanism_code_commit",
        "asset_sha256",
        "plan_sha256",
        "bank_sha256",
        "supersedes_no_go",
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
        raise ValueError("A0DR plan fields differ from schema")
    if plan.get("schema_version") != A0DR_PLAN_SCHEMA or plan.get("status") != "preregistered":
        raise ValueError("A0DR schema or status changed")
    if plan.get("execution_authorization") != "root_review_required":
        raise ValueError("A0DR authorization changed")
    commit = plan.get("mechanism_code_commit")
    if not isinstance(commit, str) or not _GIT_COMMIT_RE.fullmatch(commit):
        raise ValueError("A0DR mechanism commit is malformed")
    assets = plan.get("asset_sha256")
    if (
        not isinstance(assets, dict)
        or set(assets) != _ASSET_PATHS
        or any(not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in assets.values())
    ):
        raise ValueError("A0DR asset closure changed")
    if plan.get("bank_sha256") != bank_sha256 or not _SHA256_RE.fullmatch(bank_sha256):
        raise ValueError("A0DR bank hash changed")
    if plan.get("plan_sha256") != canonical_json_hash(plan, omitted_fields=("plan_sha256",)):
        raise ValueError("A0DR canonical plan hash changed")
    if plan.get("supersedes_no_go") != _PRIOR_NO_GO:
        raise ValueError("A0DR prior NO-GO binding changed")
    if plan.get("bound_evidence") != a0d_plan["bound_evidence"]:
        raise ValueError("A0DR A0R/A0C evidence binding changed")
    for field in ("protected_checkpoints", "remote_paths", "runtime"):
        if plan.get(field) != a0d_plan[field]:
            raise ValueError(f"A0DR changed protected A0D field: {field}")
    if plan.get("diagnostic") != _DIAGNOSTIC:
        raise ValueError("A0DR length-matched causal design changed")
    resources = dict(a0d_plan["resource_bounds"])
    resources["output_root"] = "/home/ubuntu/rlm/outputs/latent-a0dr-cache-diagnostic-v1"
    if plan.get("resource_bounds") != resources:
        raise ValueError("A0DR resources differ beyond fresh namespace")
    if plan.get("failure_classification") != a0d_plan["failure_classification"]:
        raise ValueError("A0DR failure classification changed")
    if plan.get("interpretation_boundary") != _INTERPRETATION:
        raise ValueError("A0DR interpretation boundary changed")


def load_and_validate_a0dr_plan(plan_path: Path, bank_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    prior_path = plan_path.with_name("a0d-cache-diagnostic-plan-v1.json")
    if any(path.is_symlink() or not path.is_file() for path in (plan_path, bank_path, prior_path)):
        raise ValueError("A0DR plan, bank, or prior NO-GO plan is absent or symlinked")
    a0d_plan, _ = load_and_validate_a0d_plan(prior_path, bank_path)
    if file_sha256(prior_path) != _PRIOR_NO_GO["plan_file_sha256"]:
        raise ValueError("A0DR prior NO-GO plan file changed")
    plan = json.loads(plan_path.read_text())
    bank = json.loads(bank_path.read_text())
    validate_a0_bank(bank)
    validate_a0dr_plan(plan, bank_sha256=file_sha256(bank_path), a0d_plan=a0d_plan)
    return plan, bank


def validate_a0dr_receipt(receipt: dict[str, object], *, plan: dict[str, object]) -> None:
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
        raise ValueError("A0DR receipt fields differ from schema")
    if (
        receipt.get("schema_version") != "prime-rl/latent-a0dr-cache-diagnostic-receipt/v1"
        or receipt.get("status") != "diagnostic_complete"
        or receipt.get("claim") != "non-promotional cache causal measurements only"
        or receipt.get("reference_normalized_rms") != 0.01
        or receipt.get("reference_is_promotion_gate") is not False
    ):
        raise ValueError("A0DR receipt claim or reference changed")
    for field in ("plan_sha256", "bank_sha256", "mechanism_code_commit", "asset_sha256", "interpretation_boundary"):
        if receipt.get(field) != plan[field]:
            raise ValueError(f"A0DR receipt differs from plan: {field}")
    commit = receipt.get("execution_commit")
    if not isinstance(commit, str) or not _GIT_COMMIT_RE.fullmatch(commit):
        raise ValueError("A0DR execution commit is malformed")
    if any(
        receipt.get(field) is not False
        for field in ("optimizer_created", "checkpoint_created", "model_update_attempted")
    ):
        raise ValueError("A0DR receipt violates no-update boundary")
    versions = receipt.get("versions")
    if not isinstance(versions, dict) or (
        set(versions) != {"python", "transformers", "torch_distribution", "torch_runtime"}
        or not isinstance(versions.get("python"), str)
        or not versions["python"].startswith(f"{plan['runtime']['python']}.")
        or versions.get("transformers") != plan["runtime"]["transformers"]
        or versions.get("torch_distribution") != plan["runtime"]["torch_distribution"]
        or versions.get("torch_runtime") != plan["runtime"]["torch_runtime"]
    ):
        raise ValueError("A0DR runtime versions changed")
    if receipt.get("model_runtime") != {
        "class": plan["runtime"]["model_class"],
        "hidden_size": plan["runtime"]["hidden_size"],
        "device": plan["runtime"]["device"],
        "dtype": plan["runtime"]["dtype"],
    }:
        raise ValueError("A0DR model runtime changed")
    sources = receipt.get("transformers_runtime_sources")
    if not isinstance(sources, dict) or set(sources) != set(plan["runtime"]["transformers_source_sha256"]):
        raise ValueError("A0DR source set changed")
    for name, expected_hash in plan["runtime"]["transformers_source_sha256"].items():
        source = sources.get(name)
        if not isinstance(source, dict) or (
            source.get("sha256") != expected_hash
            or not isinstance(source.get("path"), str)
            or not source["path"].startswith("/home/ubuntu/rlm/prime-rl/.venv/")
        ):
            raise ValueError("A0DR source identity changed")
    gpu = receipt.get("gpu")
    host = receipt.get("host")
    if not isinstance(gpu, dict) or (
        gpu.get("name") != plan["resource_bounds"]["gpu_model"]
        or not isinstance(gpu.get("total_memory_gib"), (int, float))
        or not math.isfinite(gpu["total_memory_gib"])
        or gpu["total_memory_gib"] < plan["resource_bounds"]["minimum_gpu_memory_gib"]
    ):
        raise ValueError("A0DR GPU identity changed")
    if not isinstance(host, dict) or (
        not isinstance(host.get("ram_bytes"), int)
        or host["ram_bytes"] < plan["resource_bounds"]["minimum_host_ram_gib"] * 2**30
        or not isinstance(host.get("free_disk_bytes_before"), int)
        or host["free_disk_bytes_before"] < plan["resource_bounds"]["minimum_free_disk_gib"] * 2**30
    ):
        raise ValueError("A0DR host resources changed")
    if (
        receipt.get("protected_hashes_before") != plan["protected_checkpoints"]
        or receipt.get("protected_hashes_after") != plan["protected_checkpoints"]
    ):
        raise ValueError("A0DR receipt does not preserve checkpoints")
    expected_metadata = {
        "coordinator_e33": plan["runtime"]["checkpoint_metadata_sha256"],
        "worker_h176": plan["runtime"]["checkpoint_metadata_sha256"],
    }
    if (
        receipt.get("checkpoint_metadata_before") != expected_metadata
        or receipt.get("checkpoint_metadata_after") != expected_metadata
    ):
        raise ValueError("A0DR receipt metadata changed")
    fixture = receipt.get("fixture")
    if not isinstance(fixture, dict) or (
        fixture.get("example_id") != _DIAGNOSTIC["example_id"]
        or fixture.get("parent_token_count") != 93
        or fixture.get("child_token_count") != 47
        or fixture.get("soft_prompt_length") != 55
        or fixture.get("injection_index") != 40
        or fixture.get("length_control_token_ids") != _FIXED_IDS
        or fixture.get("length_control_token_ids_sha256") != _FIXED_IDS_SHA256
        or fixture.get("length_control_tokens_non_special") is not True
        or fixture.get("length_matched_masks_and_positions_exact") is not True
        or not isinstance(fixture.get("continuation_token_ids"), list)
        or len(fixture["continuation_token_ids"]) != 4
        or any(not isinstance(token, int) for token in fixture["continuation_token_ids"])
        or not isinstance(fixture.get("length_control_input_ids_sha256"), str)
        or not _SHA256_RE.fullmatch(fixture["length_control_input_ids_sha256"])
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
        raise ValueError("A0DR length-control fixture evidence changed")
    arms = receipt.get("arms")
    names = ("D47", "E47", "L_ID55", "L_E55", "S55")
    pairs = {(name, branch) for name in names for branch in ("auto_position", "explicit_next_position")}
    if (
        not isinstance(arms, list)
        or len(arms) != 10
        or {(arm.get("arm"), arm.get("position_branch")) for arm in arms if isinstance(arm, dict)} != pairs
    ):
        raise ValueError("A0DR arm matrix is incomplete")
    for arm in arms:
        initial = arm.get("initial_cache_sequence_length")
        expected_initial = 47 if arm.get("arm") in {"D47", "E47"} else 55
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
            or initial != expected_initial
            or not isinstance(arm.get("prefill_last_logits_sha256"), str)
            or not _SHA256_RE.fullmatch(arm["prefill_last_logits_sha256"])
            or not isinstance(arm.get("steps"), list)
            or len(arm["steps"]) != 4
            or not isinstance(arm.get("prefill_cache_type"), str)
            or not arm["prefill_cache_type"]
            or not isinstance(arm.get("rope_state"), dict)
            or set(arm["rope_state"]) != {"before_prefill", "after_prefill"}
        ):
            raise ValueError("A0DR arm prefill evidence changed")
        for index, step in enumerate(arm["steps"], start=1):
            if (
                not isinstance(step, dict)
                or set(step)
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
                or not isinstance(step.get("greedy_equal"), bool)
                or not isinstance(step.get("cached_logits_sha256"), str)
                or not _SHA256_RE.fullmatch(step["cached_logits_sha256"])
                or not isinstance(step.get("full_logits_sha256"), str)
                or not _SHA256_RE.fullmatch(step["full_logits_sha256"])
                or not all(
                    isinstance(step.get(field), (int, float)) and math.isfinite(step[field])
                    for field in ("maximum_absolute_logit_difference", "normalized_rms")
                )
            ):
                raise ValueError("A0DR step metric or length is invalid")
            prepared = step.get("prepared")
            if not isinstance(prepared, dict) or not isinstance(prepared.get("keys"), list):
                raise ValueError("A0DR prepared-input evidence is absent")
            keys = set(prepared["keys"])
            if not {"input_ids", "past_key_values", "attention_mask", "use_cache"}.issubset(keys):
                raise ValueError("A0DR prepared decode inputs are incomplete")
            if arm["position_branch"] == "auto_position" and {"position_ids", "cache_position"} & keys:
                raise ValueError("A0DR auto branch contains explicit position")
            if arm["position_branch"] == "explicit_next_position" and not {
                "position_ids",
                "cache_position",
            }.issubset(keys):
                raise ValueError("A0DR explicit branch lacks position")
            values = prepared.get("values")
            past = values.get("past_key_values") if isinstance(values, dict) else None
            input_ids = values.get("input_ids") if isinstance(values, dict) else None
            if not isinstance(values, dict) or not isinstance(past, dict) or past.get("sequence_length") != (
                initial + index - 1
            ):
                raise ValueError("A0DR prepared cache provenance changed")
            expected_token = fixture["continuation_token_ids"][index - 1]
            if not isinstance(input_ids, dict) or input_ids.get("values") != [expected_token]:
                raise ValueError("A0DR prepared continuation token changed")
            if arm["position_branch"] == "explicit_next_position":
                expected_position = [initial + index - 1]
                prepared_position_ids = values.get("position_ids")
                prepared_cache_position = values.get("cache_position")
                if (
                    not isinstance(prepared_position_ids, dict)
                    or prepared_position_ids.get("values") != expected_position
                    or not isinstance(prepared_cache_position, dict)
                    or prepared_cache_position.get("values") != expected_position
                ):
                    raise ValueError("A0DR explicit position value changed")
            rope = step.get("rope_state")
            if not isinstance(rope, dict) or set(rope) != {"before_prepare", "after_prepare", "after_decode"}:
                raise ValueError("A0DR rope transition evidence is absent")
    receipt_hash = receipt.get("receipt_sha256")
    if not isinstance(receipt_hash, str) or receipt_hash != canonical_json_hash(
        receipt, omitted_fields=("receipt_sha256",)
    ):
        raise ValueError("A0DR receipt hash changed")
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
        raise ValueError("A0DR resource evidence changed")
