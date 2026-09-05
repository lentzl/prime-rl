from __future__ import annotations

import json
import math
import re
from pathlib import Path

from prime_rl.latent.a0 import canonical_json_hash, file_sha256, validate_a0_bank
from prime_rl.latent.a0dr2 import load_and_validate_a0dr2_plan, validate_a0dr2_receipt

A0CAL_PLAN_SCHEMA = "prime-rl/latent-a0-cache-calibration-plan/v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_FIXED_IDS = [40, 4021, 2528, 8976, 35139, 635, 524, 599]
_FIXED_IDS_SHA256 = "e86e01e61315008783cc217a5bb83a1b3aced0daaecbc920b8d3b45ab4b205d8"
_EXPECTED_PROBES = {
    "a0-mechanism-0001": {"parent": 93, "child": 47, "injection": 40, "matched": 55},
    "a0-mechanism-0002": {"parent": 84, "child": 42, "injection": 35, "matched": 50},
    "a0-mechanism-0003": {"parent": 77, "child": 35, "injection": 28, "matched": 43},
    "a0-mechanism-0004": {"parent": 74, "child": 40, "injection": 33, "matched": 48},
}
_CRITERION = {
    "complete_distinct_probe_floor": 4,
    "required_probe_ids": list(_EXPECTED_PROBES),
    "continuation_tokens_per_probe": 4,
    "arms": ["L_ID", "L_E", "S"],
    "position_branch": "explicit_next_position",
    "fresh_cache_per_arm": True,
    "discrete_interface_parity": "L_ID and L_E prefill/step metrics and cached/full logit hashes exact",
    "maximum_absolute_logit_difference_per_step": 0.5,
    "cached_full_greedy_equal_each_step": True,
    "maximum_soft_minus_discrete_nrms_per_step": 0.005,
    "soft_mean_nrms_rule": "soft_mean <= discrete_mean + max(0.0025, 0.25 * discrete_mean)",
    "mean_excess_floor": 0.0025,
    "mean_relative_excess_fraction": 0.25,
    "all_probes_must_qualify": True,
    "posthoc_threshold_change_allowed": False,
}
_A0DR2_EVIDENCE = {
    "status": "diagnostic_complete",
    "receipt_file_sha256": "6ed279ce60a7162aee7fb70f07b554d84a6ebd57e87ddc1a70a8e3f350238fab",
    "receipt_internal_sha256": "1f5490dc20016f655f8b7b1c7b02e6a62157250eb6217c23667ef2879f983f1b",
    "launch_log_sha256": "778b7655044f3d0c0697d28b616afae88c026423958fa29da2f102997286c544",
    "mechanism_commit": "945c262399b3794c74ddc3604f00233757f7d4dd",
    "execution_commit": "f6d45f3fd15c424304a4cb755de121e1a6917bcf",
    "plan_sha256": "be6077168a09ea25f6f96716b66472d00583ee0174126e7bd448dc8edf5d016b",
    "plan_file_sha256": "22b6a441b5a6b4a62d2f2a9c30f5544c1c97d15b9e30c90b2f44ca11d59695f0",
    "probe_1_role": "calibration_anchor_not_heldout_confirmation",
    "probes_2_to_4_role": "prospective_first_cache_exposure",
}
_CALIBRATION_BASIS = {
    "anchor_probe": "a0-mechanism-0001",
    "anchor_discrete_mean_nrms": 0.01765758660621941,
    "anchor_soft_mean_nrms": 0.01621906436048448,
    "anchor_max_positive_soft_minus_discrete_step_nrms": 0.001720189116895199,
    "threshold_freeze_timing": "before any cache exposure of probes 2 through 4",
    "step_excess_limit_rationale": "0.005 is more than 2.9 times the anchor positive excess but remains bounded",
    "mean_rule_rationale": "25 percent relative allowance with a 0.0025 numerical floor prevents near-zero ratio instability",
}
_ASSET_PATHS = {
    "experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0c-carrier-success-receipt.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0dr2-cache-diagnostic-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0dr2-diagnostic-complete-receipt.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0dr2-diagnostic-complete-run.log",
    "scripts/latent/run_a0dr_cache_diagnostic_v1.py",
    "scripts/latent/run_a0dr2_cache_diagnostic_v1.py",
    "scripts/latent/run_a0_cache_calibration_v1.py",
    "scripts/latent/run_a0_cache_calibration_v1.sh",
    "src/prime_rl/latent/__init__.py",
    "src/prime_rl/latent/a0.py",
    "src/prime_rl/latent/a0cal.py",
    "src/prime_rl/latent/a0c.py",
    "src/prime_rl/latent/a0d.py",
    "src/prime_rl/latent/a0dr.py",
    "src/prime_rl/latent/a0dr2.py",
    "src/prime_rl/latent/a0r.py",
    "src/prime_rl/latent/policy_adapter.py",
}
_INTERPRETATION = (
    "prospective relative cache calibration only; A0R remains rejected and A1 remains blocked during this run; "
    "a validated result clears only the cache-specific relative-calibration question for later root/evaluator review, "
    "not live harness capture, bridge learnability, model admission, training, or A1 authorization"
)


def validate_a0cal_plan(plan: dict[str, object], *, bank_sha256: str, prior: dict[str, object]) -> None:
    required = {
        "schema_version",
        "status",
        "execution_authorization",
        "mechanism_code_commit",
        "asset_sha256",
        "plan_sha256",
        "bank_sha256",
        "a0dr2_evidence",
        "calibration_basis",
        "protected_checkpoints",
        "remote_paths",
        "runtime",
        "expected_probes",
        "length_control",
        "criterion",
        "resource_bounds",
        "failure_classification",
        "interpretation_boundary",
    }
    if set(plan) != required:
        raise ValueError("A0 cache-calibration plan fields changed")
    if plan.get("schema_version") != A0CAL_PLAN_SCHEMA or plan.get("status") != "preregistered":
        raise ValueError("A0 cache-calibration schema or status changed")
    if plan.get("execution_authorization") != "root_and_evaluator_review_required":
        raise ValueError("A0 cache-calibration authorization changed")
    commit = plan.get("mechanism_code_commit")
    assets = plan.get("asset_sha256")
    if not isinstance(commit, str) or not _GIT_COMMIT_RE.fullmatch(commit):
        raise ValueError("A0 cache-calibration mechanism commit malformed")
    if (
        not isinstance(assets, dict)
        or set(assets) != _ASSET_PATHS
        or any(not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in assets.values())
    ):
        raise ValueError("A0 cache-calibration asset closure changed")
    if plan.get("plan_sha256") != canonical_json_hash(plan, omitted_fields=("plan_sha256",)):
        raise ValueError("A0 cache-calibration plan hash changed")
    if plan.get("bank_sha256") != bank_sha256 or not _SHA256_RE.fullmatch(bank_sha256):
        raise ValueError("A0 cache-calibration bank changed")
    if plan.get("a0dr2_evidence") != _A0DR2_EVIDENCE:
        raise ValueError("A0DR2 evidence binding changed")
    if plan.get("calibration_basis") != _CALIBRATION_BASIS:
        raise ValueError("A0 cache-calibration prospective basis changed")
    for field in ("protected_checkpoints", "remote_paths", "runtime", "failure_classification"):
        if plan.get(field) != prior[field]:
            raise ValueError(f"A0 cache calibration changed protected field: {field}")
    if plan.get("expected_probes") != _EXPECTED_PROBES:
        raise ValueError("A0 cache-calibration probes changed")
    if plan.get("length_control") != {
        "token_ids": _FIXED_IDS,
        "token_ids_sha256": _FIXED_IDS_SHA256,
        "tokens_must_be_non_special": True,
        "insertion_count": 8,
        "same_boundary_mask_and_positions_as_soft": True,
    }:
        raise ValueError("A0 cache-calibration length control changed")
    if plan.get("criterion") != _CRITERION:
        raise ValueError("A0 cache-calibration criterion changed")
    resources = dict(prior["resource_bounds"])
    resources["output_root"] = "/home/ubuntu/rlm/outputs/latent-a0-cache-calibration-v1"
    if plan.get("resource_bounds") != resources:
        raise ValueError("A0 cache-calibration resources changed")
    if plan.get("interpretation_boundary") != _INTERPRETATION:
        raise ValueError("A0 cache-calibration interpretation changed")


def load_and_validate_a0cal_plan(plan_path: Path, bank_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    prior_path = plan_path.with_name("a0dr2-cache-diagnostic-plan-v1.json")
    receipt_path = plan_path.with_name("a0dr2-diagnostic-complete-receipt.json")
    log_path = plan_path.with_name("a0dr2-diagnostic-complete-run.log")
    if any(
        path.is_symlink() or not path.is_file() for path in (plan_path, bank_path, prior_path, receipt_path, log_path)
    ):
        raise ValueError("A0 cache-calibration plan/bank/evidence absent or symlinked")
    if file_sha256(prior_path) != _A0DR2_EVIDENCE["plan_file_sha256"]:
        raise ValueError("A0DR2 plan file changed")
    if file_sha256(receipt_path) != _A0DR2_EVIDENCE["receipt_file_sha256"]:
        raise ValueError("A0DR2 receipt file changed")
    if file_sha256(log_path) != _A0DR2_EVIDENCE["launch_log_sha256"]:
        raise ValueError("A0DR2 launch log changed")
    prior, _ = load_and_validate_a0dr2_plan(prior_path, bank_path)
    receipt = json.loads(receipt_path.read_text())
    validate_a0dr2_receipt(receipt, plan=prior)
    if receipt.get("receipt_sha256") != _A0DR2_EVIDENCE["receipt_internal_sha256"]:
        raise ValueError("A0DR2 internal receipt hash changed")
    plan = json.loads(plan_path.read_text())
    bank = json.loads(bank_path.read_text())
    validate_a0_bank(bank)
    validate_a0cal_plan(plan, bank_sha256=file_sha256(bank_path), prior=prior)
    return plan, bank


def arm_signature(arm: dict[str, object]) -> tuple[object, ...]:
    return (
        arm.get("initial_cache_sequence_length"),
        arm.get("initial_logits_finite"),
        arm.get("prefill_cache_type"),
        arm.get("prefill_last_logits_sha256"),
        [
            (
                step.get("cache_sequence_length"),
                step.get("maximum_absolute_logit_difference"),
                step.get("normalized_rms"),
                step.get("greedy_equal"),
                step.get("cached_logits_sha256"),
                step.get("full_logits_sha256"),
            )
            for step in arm.get("steps", [])
        ],
    )


def calculate_probe_criterion(arms: dict[str, dict[str, object]]) -> dict[str, object]:
    discrete = arms["L_E"]
    soft = arms["S"]
    discrete_steps = discrete["steps"]
    soft_steps = soft["steps"]
    discrete_mean = sum(step["normalized_rms"] for step in discrete_steps) / 4
    soft_mean = sum(step["normalized_rms"] for step in soft_steps) / 4
    allowance = max(_CRITERION["mean_excess_floor"], _CRITERION["mean_relative_excess_fraction"] * discrete_mean)
    step_excess = [
        soft_step["normalized_rms"] - discrete_step["normalized_rms"]
        for discrete_step, soft_step in zip(discrete_steps, soft_steps, strict=True)
    ]
    interface_exact = arm_signature(arms["L_ID"]) == arm_signature(discrete)
    finite = all(
        math.isfinite(step[field])
        for arm in arms.values()
        for step in arm["steps"]
        for field in ("maximum_absolute_logit_difference", "normalized_rms")
    )
    absolute_ok = all(
        step["maximum_absolute_logit_difference"] <= _CRITERION["maximum_absolute_logit_difference_per_step"]
        for arm in arms.values()
        for step in arm["steps"]
    )
    greedy_ok = all(step["greedy_equal"] is True for arm in arms.values() for step in arm["steps"])
    step_relative_ok = max(step_excess) <= _CRITERION["maximum_soft_minus_discrete_nrms_per_step"]
    mean_relative_ok = soft_mean <= discrete_mean + allowance
    return {
        "discrete_interface_exact": interface_exact,
        "finite": finite,
        "absolute_max_ok": absolute_ok,
        "greedy_equal_all": greedy_ok,
        "discrete_mean_nrms": discrete_mean,
        "soft_mean_nrms": soft_mean,
        "mean_excess_allowance": allowance,
        "soft_minus_discrete_nrms_by_step": step_excess,
        "step_relative_ok": step_relative_ok,
        "mean_relative_ok": mean_relative_ok,
        "qualifies": interface_exact and finite and absolute_ok and greedy_ok and step_relative_ok and mean_relative_ok,
    }


def validate_a0cal_receipt(receipt: dict[str, object], *, plan: dict[str, object]) -> None:
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
        "criterion",
        "probes",
        "complete_distinct_probes",
        "optimizer_created",
        "checkpoint_created",
        "model_update_attempted",
        "resources",
        "interpretation_boundary",
        "receipt_sha256",
    }
    if set(receipt) != required:
        raise ValueError("A0 cache-calibration receipt fields changed")
    if (
        receipt.get("schema_version") != "prime-rl/latent-a0-cache-calibration-receipt/v1"
        or receipt.get("status") not in {"relative_cache_calibrated", "relative_cache_rejected"}
        or receipt.get("claim") != "relative soft-vs-discrete cache calibration only"
    ):
        raise ValueError("A0 cache-calibration receipt status changed")
    for field in (
        "plan_sha256",
        "bank_sha256",
        "mechanism_code_commit",
        "asset_sha256",
        "criterion",
        "interpretation_boundary",
    ):
        if receipt.get(field) != plan[field]:
            raise ValueError(f"A0 cache-calibration receipt differs from plan: {field}")
    execution = receipt.get("execution_commit")
    if not isinstance(execution, str) or not _GIT_COMMIT_RE.fullmatch(execution):
        raise ValueError("A0 cache-calibration execution commit malformed")
    if any(
        receipt.get(field) is not False
        for field in ("optimizer_created", "checkpoint_created", "model_update_attempted")
    ):
        raise ValueError("A0 cache-calibration no-update boundary changed")
    if (
        receipt.get("protected_hashes_before") != plan["protected_checkpoints"]
        or receipt.get("protected_hashes_after") != plan["protected_checkpoints"]
    ):
        raise ValueError("A0 cache-calibration protected hashes changed")
    versions = receipt.get("versions")
    if not isinstance(versions, dict) or (
        versions.get("transformers") != plan["runtime"]["transformers"]
        or versions.get("torch_distribution") != plan["runtime"]["torch_distribution"]
        or versions.get("torch_runtime") != plan["runtime"]["torch_runtime"]
    ):
        raise ValueError("A0 cache-calibration runtime versions changed")
    expected_metadata = {
        "coordinator_e33": plan["runtime"]["checkpoint_metadata_sha256"],
        "worker_h176": plan["runtime"]["checkpoint_metadata_sha256"],
    }
    if (
        receipt.get("checkpoint_metadata_before") != expected_metadata
        or receipt.get("checkpoint_metadata_after") != expected_metadata
    ):
        raise ValueError("A0 cache-calibration metadata changed")
    sources = receipt.get("transformers_runtime_sources")
    if (
        not isinstance(sources, dict)
        or {name: value.get("sha256") for name, value in sources.items() if isinstance(value, dict)}
        != plan["runtime"]["transformers_source_sha256"]
    ):
        raise ValueError("A0 cache-calibration source hashes changed")
    model = receipt.get("model_runtime")
    if model != {
        "class": plan["runtime"]["model_class"],
        "hidden_size": plan["runtime"]["hidden_size"],
        "device": plan["runtime"]["device"],
        "dtype": plan["runtime"]["dtype"],
    }:
        raise ValueError("A0 cache-calibration model runtime changed")
    gpu = receipt.get("gpu")
    host = receipt.get("host")
    if not isinstance(gpu, dict) or (
        gpu.get("name") != plan["resource_bounds"]["gpu_model"]
        or not isinstance(gpu.get("total_memory_gib"), (int, float))
        or not math.isfinite(gpu["total_memory_gib"])
        or gpu["total_memory_gib"] < plan["resource_bounds"]["minimum_gpu_memory_gib"]
    ):
        raise ValueError("A0 cache-calibration GPU evidence changed")
    if not isinstance(host, dict) or (
        not isinstance(host.get("ram_bytes"), int)
        or host["ram_bytes"] < plan["resource_bounds"]["minimum_host_ram_gib"] * 2**30
        or not isinstance(host.get("free_disk_bytes_before"), int)
        or host["free_disk_bytes_before"] < plan["resource_bounds"]["minimum_free_disk_gib"] * 2**30
    ):
        raise ValueError("A0 cache-calibration host evidence changed")
    probes = receipt.get("probes")
    if (
        not isinstance(probes, list)
        or len(probes) != 4
        or {probe.get("example_id") for probe in probes if isinstance(probe, dict)} != set(_EXPECTED_PROBES)
    ):
        raise ValueError("A0 cache-calibration probe floor changed")
    for probe in probes:
        if not isinstance(probe, dict) or set(probe) != {"example_id", "fixture", "arms", "criterion_result"}:
            raise ValueError("A0 cache-calibration probe schema changed")
        expected = _EXPECTED_PROBES[probe["example_id"]]
        fixture = probe["fixture"]
        if not isinstance(fixture, dict) or (
            fixture.get("parent_token_count") != expected["parent"]
            or fixture.get("child_token_count") != expected["child"]
            or fixture.get("injection_index") != expected["injection"]
            or fixture.get("matched_prompt_length") != expected["matched"]
            or fixture.get("length_control_token_ids") != _FIXED_IDS
            or fixture.get("length_control_token_ids_sha256") != _FIXED_IDS_SHA256
            or fixture.get("length_control_tokens_non_special") is not True
            or fixture.get("matched_mask_and_positions_exact") is not True
            or any(
                not isinstance(fixture.get(field), str) or not _SHA256_RE.fullmatch(fixture[field])
                for field in (
                    "child_input_ids_sha256",
                    "length_control_input_ids_sha256",
                    "continuation_input_ids_sha256",
                    "workspace_source_sha256",
                    "soft_prompt_sha256",
                )
            )
            or not isinstance(fixture.get("continuation_token_ids"), list)
            or len(fixture["continuation_token_ids"]) != 4
        ):
            raise ValueError("A0 cache-calibration fixture changed")
        arms = probe.get("arms")
        if (
            not isinstance(arms, list)
            or len(arms) != 3
            or {arm.get("arm") for arm in arms if isinstance(arm, dict)} != {"L_ID", "L_E", "S"}
        ):
            raise ValueError("A0 cache-calibration arm set changed")
        arm_map = {arm["arm"]: arm for arm in arms}
        for arm in arms:
            if (
                set(arm)
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
                or arm.get("position_branch") != "explicit_next_position"
                or arm.get("fresh_cache") is not True
                or arm.get("initial_logits_finite") is not True
                or arm.get("initial_cache_sequence_length") != expected["matched"]
                or not isinstance(arm.get("prefill_cache_type"), str)
                or not isinstance(arm.get("prefill_last_logits_sha256"), str)
                or not _SHA256_RE.fullmatch(arm["prefill_last_logits_sha256"])
                or len(arm.get("steps", [])) != 4
            ):
                raise ValueError("A0 cache-calibration arm evidence incomplete")
            for index, step in enumerate(arm["steps"], start=1):
                prepared = step.get("prepared")
                prepared_values = prepared.get("values") if isinstance(prepared, dict) else None
                past = prepared_values.get("past_key_values") if isinstance(prepared_values, dict) else None
                if (
                    step.get("step") != index
                    or step.get("cache_sequence_length") != expected["matched"] + index
                    or not isinstance(step.get("greedy_equal"), bool)
                    or not all(
                        isinstance(step.get(field), (int, float)) and math.isfinite(step[field])
                        for field in ("maximum_absolute_logit_difference", "normalized_rms")
                    )
                    or any(
                        not isinstance(step.get(field), str) or not _SHA256_RE.fullmatch(step[field])
                        for field in ("cached_logits_sha256", "full_logits_sha256")
                    )
                    or not isinstance(past, dict)
                    or past.get("sequence_length") != expected["matched"] + index - 1
                ):
                    raise ValueError("A0 cache-calibration step evidence incomplete")
        observed = calculate_probe_criterion(arm_map)
        result = probe.get("criterion_result")
        if not isinstance(result, dict) or result != observed:
            raise ValueError("A0 cache-calibration criterion result changed")
    qualified = all(probe["criterion_result"]["qualifies"] for probe in probes)
    expected_status = "relative_cache_calibrated" if qualified else "relative_cache_rejected"
    if receipt.get("complete_distinct_probes") != 4 or receipt.get("status") != expected_status:
        raise ValueError("A0 cache-calibration terminal classification changed")
    receipt_hash = receipt.get("receipt_sha256")
    if not isinstance(receipt_hash, str) or receipt_hash != canonical_json_hash(
        receipt, omitted_fields=("receipt_sha256",)
    ):
        raise ValueError("A0 cache-calibration receipt hash changed")
    resources = receipt.get("resources")
    if (
        not isinstance(resources, dict)
        or not isinstance(resources.get("peak_cuda_memory_bytes"), int)
        or resources["peak_cuda_memory_bytes"] <= 0
        or not isinstance(resources.get("wall_seconds"), (int, float))
        or not math.isfinite(resources["wall_seconds"])
    ):
        raise ValueError("A0 cache-calibration resources changed")
