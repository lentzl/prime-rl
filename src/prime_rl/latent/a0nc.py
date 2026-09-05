from __future__ import annotations

import json
import math
import re
from collections import deque
from pathlib import Path

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a0cal import validate_a0cal_receipt

A0NC_PLAN_SCHEMA = "prime-rl/latent-a0-nocache-plan/v1"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDS = ["a0-nocache-0001", "a0-nocache-0002", "a0-nocache-0003", "a0-nocache-0004"]
_CONTINUATION = [49265, 48338, 3438, 321]
_REJECTION = {
    "status": "relative_cache_rejected",
    "receipt_file_sha256": "707d507bd24f64ea7b4d872268d5335520bd7f1e7771f380b37fa1fb5be97ab1",
    "receipt_internal_sha256": "10b5008dcc8ba45f8375590d850fd4306e1d591d6e4e6da7b1df3c7a98deae23",
    "launch_log_sha256": "c3b3d2ff004e46543bcc9435256c39b00f3e8d957c273dfac819150e76cb949d",
    "snapshot_manifest_sha256": "b1bc015a9711015592d89edfdaefe1343da69e78ac09584131a7c24d49102cc9",
    "execution_commit": "2b7d3c8b26c6813da4fb7e534c0061b3814769ec",
    "plan_sha256": "eeda1c571359b29882f16acbd50f573ecd3fa71238882f9f95bb99040e8e7578",
    "complete_distinct_probes": 4,
    "qualifying_probes": 2,
}
_MECHANISM = {
    "probe_status": "fresh_unexposed_before_this_freeze",
    "required_probe_ids": _IDS,
    "complete_distinct_probe_floor": 4,
    "steps_per_probe": 4,
    "continuation_token_ids": _CONTINUATION,
    "arms": ["L_ID", "L_E", "S"],
    "full_prefix_recompute_every_step": True,
    "use_cache": False,
    "past_key_values_input": None,
    "past_key_values_output_must_be_none": True,
    "cache_subclass_allocation_sentinel": "raise_on___new___during_probe_inference_window",
    "l_id_l_e_logits_bitwise_equal_each_step": True,
    "l_id_l_e_logits_finite_each_step": True,
    "soft_logits_finite_each_step": True,
    "soft_repeat_logits_bitwise_equal_each_step": True,
    "attention_mask_all_visible_each_step": True,
    "position_ids_strictly_sequential_each_step": True,
    "all_four_probes_must_complete_and_qualify": True,
    "optimizer": None,
    "checkpoint": False,
    "model_update": False,
    "claim": "no_cache_full_recompute_diagnostic_valid_for_B",
}
_INTERPRETATION = (
    "no-cache full-prefix receiver mechanism only; A0R and relative-cache calibration remain rejected and A1/live "
    "harness remain blocked; no bridge learnability, handoff, capability, admission, training, or A+B claim"
)
_FAILURE_CLASSIFICATION = {
    "cache_allocation_or_past_key_values_detected": "nocache_receiver_mechanism_rejected",
    "diagnostic_execution_or_finiteness_failure": "diagnostic_incomplete",
    "environment_provenance_timeout_or_oom": "infrastructure_invalid",
    "run_id_reusable": False,
}
_ASSETS = {
    "experiments/qwen35-2b-latent-workspace-v1/a0-cache-calibration-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0-cache-calibration-rejected-receipt.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0-cache-calibration-rejected-run.log",
    "experiments/qwen35-2b-latent-workspace-v1/a0-cache-calibration-rejected-manifest.sha256",
    "experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0-nocache-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0-nocache-disjointness-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0c-carrier-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0d-cache-diagnostic-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0dr-cache-diagnostic-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0dr2-cache-diagnostic-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0r-mechanism-plan-v1.json",
    "scripts/latent/run_a0_nocache_receiver_v1.py",
    "scripts/latent/run_a0_nocache_receiver_v1.sh",
    "src/prime_rl/latent/__init__.py",
    "src/prime_rl/latent/a0.py",
    "src/prime_rl/latent/a0cal.py",
    "src/prime_rl/latent/a0c.py",
    "src/prime_rl/latent/a0d.py",
    "src/prime_rl/latent/a0dr.py",
    "src/prime_rl/latent/a0dr2.py",
    "src/prime_rl/latent/a0nc.py",
    "src/prime_rl/latent/a0r.py",
    "src/prime_rl/latent/policy_adapter.py",
}


def validate_plan(plan: dict[str, object], *, prior: dict[str, object], bank_sha: str) -> None:
    required = {
        "schema_version",
        "status",
        "execution_authorization",
        "mechanism_code_commit",
        "asset_sha256",
        "plan_sha256",
        "bank_sha256",
        "prior_cache_rejection",
        "protected_checkpoints",
        "remote_paths",
        "runtime",
        "mechanism",
        "resource_bounds",
        "failure_classification",
        "interpretation_boundary",
    }
    if set(plan) != required or plan.get("schema_version") != A0NC_PLAN_SCHEMA or plan.get("status") != "preregistered":
        raise ValueError("A0NC plan schema changed")
    if plan.get("execution_authorization") != "root_and_evaluator_review_required":
        raise ValueError("A0NC authorization changed")
    if not isinstance(plan.get("mechanism_code_commit"), str) or not _COMMIT.fullmatch(plan["mechanism_code_commit"]):
        raise ValueError("A0NC commit malformed")
    assets = plan.get("asset_sha256")
    if (
        not isinstance(assets, dict)
        or set(assets) != _ASSETS
        or any(not isinstance(v, str) or not _SHA.fullmatch(v) for v in assets.values())
    ):
        raise ValueError("A0NC asset closure changed")
    if (
        plan.get("plan_sha256") != canonical_json_hash(plan, omitted_fields=("plan_sha256",))
        or plan.get("bank_sha256") != bank_sha
    ):
        raise ValueError("A0NC plan or bank hash changed")
    if plan.get("prior_cache_rejection") != _REJECTION or plan.get("mechanism") != _MECHANISM:
        raise ValueError("A0NC evidence or mechanism changed")
    for field in ("protected_checkpoints", "remote_paths", "runtime"):
        if plan.get(field) != prior[field]:
            raise ValueError(f"A0NC protected field changed: {field}")
    resources = dict(prior["resource_bounds"])
    resources["output_root"] = "/home/ubuntu/rlm/outputs/latent-a0-nocache-receiver-v1"
    if (
        plan.get("resource_bounds") != resources
        or plan.get("failure_classification") != _FAILURE_CLASSIFICATION
        or plan.get("interpretation_boundary") != _INTERPRETATION
    ):
        raise ValueError("A0NC resource or interpretation changed")


def load_plan(plan_path: Path, bank_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    prior_path = plan_path.with_name("a0-cache-calibration-plan-v1.json")
    receipt_path = plan_path.with_name("a0-cache-calibration-rejected-receipt.json")
    log_path = plan_path.with_name("a0-cache-calibration-rejected-run.log")
    manifest_path = plan_path.with_name("a0-cache-calibration-rejected-manifest.sha256")
    prior_bank_path = plan_path.with_name("a0-mechanism-bank-v1.json")
    disjointness_path = plan_path.with_name("a0-nocache-disjointness-v1.json")
    evidence_paths = (
        plan_path,
        bank_path,
        prior_path,
        receipt_path,
        log_path,
        manifest_path,
        prior_bank_path,
        disjointness_path,
    )
    if any(p.is_symlink() or not p.is_file() for p in evidence_paths):
        raise ValueError("A0NC plan/bank/evidence absent or symlinked")
    if (
        file_sha256(receipt_path) != _REJECTION["receipt_file_sha256"]
        or file_sha256(log_path) != _REJECTION["launch_log_sha256"]
        or file_sha256(manifest_path) != _REJECTION["snapshot_manifest_sha256"]
    ):
        raise ValueError("A0NC rejection evidence changed")
    prior = json.loads(prior_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    validate_a0cal_receipt(receipt, plan=prior)
    qualifying = sum(probe.get("criterion_result", {}).get("qualifies") is True for probe in receipt.get("probes", []))
    if (
        receipt.get("receipt_sha256") != _REJECTION["receipt_internal_sha256"]
        or receipt.get("status") != _REJECTION["status"]
        or receipt.get("execution_commit") != _REJECTION["execution_commit"]
        or receipt.get("plan_sha256") != _REJECTION["plan_sha256"]
        or receipt.get("complete_distinct_probes") != _REJECTION["complete_distinct_probes"]
        or qualifying != _REJECTION["qualifying_probes"]
    ):
        raise ValueError("A0NC rejection internal hash changed")
    plan = json.loads(plan_path.read_text())
    bank = json.loads(bank_path.read_text())
    if (
        bank.get("schema_version") != "prime-rl/latent-a0-nocache-bank/v1"
        or bank.get("status") != "frozen_before_model_exposure"
        or bank.get("continuation_token_ids") != _CONTINUATION
        or [e.get("example_id") for e in bank.get("examples", [])] != _IDS
    ):
        raise ValueError("A0NC fresh bank changed")
    prior_bank_sha = file_sha256(prior_bank_path)
    fresh_bank_sha = file_sha256(bank_path)
    prior_bank = json.loads(prior_bank_path.read_text())
    disjointness = json.loads(disjointness_path.read_text())

    def content_hashes(payload: dict[str, object]) -> dict[str, str]:
        return {
            example["example_id"]: canonical_json_hash(
                {"parent_messages": example["parent_messages"], "child_messages": example["child_messages"]}
            )
            for example in payload["examples"]
        }

    prior_hashes = set(disjointness.get("prior_content_sha256", {}).values())
    fresh_hashes = set(disjointness.get("fresh_content_sha256", {}).values())
    campaign_union = disjointness.get("prior_campaign_union", {})
    expected_campaigns = {"A0", "A0R", "A0C", "A0D", "A0DR", "A0DR2", "A0_CACHE_CALIBRATION"}
    campaign_mapping_valid = isinstance(campaign_union, dict) and set(campaign_union) == expected_campaigns
    if campaign_mapping_valid:
        for campaign in campaign_union.values():
            campaign_plan = plan_path.with_name(campaign["plan"])
            if (
                campaign_plan.is_symlink()
                or not campaign_plan.is_file()
                or file_sha256(campaign_plan) != campaign["plan_file_sha256"]
                or campaign["bank_sha256"] != prior_bank_sha
                or not set(campaign["probe_ids"]).issubset(
                    set(_IDS).union({id.replace("nocache", "mechanism") for id in _IDS})
                )
            ):
                campaign_mapping_valid = False
                break
    if (
        disjointness.get("schema_version") != "prime-rl/latent-a0-nocache-disjointness/v1"
        or disjointness.get("prior_bank_sha256") != prior_bank_sha
        or disjointness.get("fresh_bank_sha256") != fresh_bank_sha
        or disjointness.get("content_hash_rule")
        != "canonical JSON SHA-256 of {parent_messages,child_messages}, sorted keys, compact separators, ensure_ascii=true"
        or disjointness.get("prior_content_sha256") != content_hashes(prior_bank)
        or disjointness.get("fresh_content_sha256") != content_hashes(bank)
        or not campaign_mapping_valid
        or disjointness.get("prior_union_collapses_to_exact_probe_ids")
        != ["a0-mechanism-0001", "a0-mechanism-0002", "a0-mechanism-0003", "a0-mechanism-0004"]
        or len(prior_hashes) != 4
        or len(fresh_hashes) != 4
        or prior_hashes & fresh_hashes
        or disjointness.get("content_hash_sets_disjoint") is not True
        or disjointness.get("runtime_rule")
        != "all full rendered parent and child token-tensor hashes must be unique across prior and fresh banks"
    ):
        raise ValueError("A0NC frozen disjointness evidence changed")
    validate_plan(plan, prior=prior, bank_sha=file_sha256(bank_path))
    return plan, bank


def recursive_subclass_closure(base: type) -> set[type]:
    """Return the transitive subclass closure, including base, without instantiation."""
    closure: set[type] = {base}
    pending = deque([base])
    while pending:
        parent = pending.popleft()
        for child in parent.__subclasses__():
            if child not in closure:
                closure.add(child)
                pending.append(child)
    return closure


def validate_receipt(receipt: dict[str, object], *, plan: dict[str, object]) -> None:
    required = {
        "schema_version",
        "status",
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
        "cache_guard",
        "no_cache_call_contract",
        "disjointness",
        "protected_hashes_before",
        "protected_hashes_after",
        "checkpoint_metadata_before",
        "checkpoint_metadata_after",
        "probes",
        "complete_distinct_probes",
        "prior_cache_rejection",
        "claim",
        "optimizer_created",
        "checkpoint_created",
        "model_update_attempted",
        "tensor_persistence",
        "resources",
        "interpretation_boundary",
        "receipt_sha256",
    }
    if set(receipt) != required:
        raise ValueError("A0NC receipt fields changed")
    if receipt.get("schema_version") != "prime-rl/latent-a0-nocache-receipt/v1" or receipt.get("status") not in {
        "nocache_receiver_mechanism_validated",
        "nocache_receiver_mechanism_rejected",
    }:
        raise ValueError("A0NC receipt status changed")
    for field in ("plan_sha256", "bank_sha256", "mechanism_code_commit", "asset_sha256", "interpretation_boundary"):
        if receipt.get(field) != plan[field]:
            raise ValueError(f"A0NC receipt differs from plan: {field}")
    if (
        receipt.get("protected_hashes_before") != plan["protected_checkpoints"]
        or receipt.get("protected_hashes_after") != plan["protected_checkpoints"]
        or any(
            receipt.get(f) is not False for f in ("optimizer_created", "checkpoint_created", "model_update_attempted")
        )
    ):
        raise ValueError("A0NC protection boundary changed")
    versions = receipt.get("versions")
    sources = receipt.get("transformers_runtime_sources")
    if (
        not isinstance(versions, dict)
        or versions.get("transformers") != plan["runtime"]["transformers"]
        or versions.get("torch_distribution") != plan["runtime"]["torch_distribution"]
        or versions.get("torch_runtime") != plan["runtime"]["torch_runtime"]
        or not isinstance(sources, dict)
        or {name: value.get("sha256") for name, value in sources.items()}
        != plan["runtime"]["transformers_source_sha256"]
        or receipt.get("checkpoint_metadata_before")
        != {
            "coordinator_e33": plan["runtime"]["checkpoint_metadata_sha256"],
            "worker_h176": plan["runtime"]["checkpoint_metadata_sha256"],
        }
        or receipt.get("checkpoint_metadata_after") != receipt.get("checkpoint_metadata_before")
    ):
        raise ValueError("A0NC runtime provenance changed")
    if (
        receipt.get("claim") != _MECHANISM["claim"]
        or receipt.get("prior_cache_rejection") != _REJECTION
        or receipt.get("tensor_persistence") is not False
        or receipt.get("no_cache_call_contract")
        != {
            "use_cache_false_every_call": True,
            "past_key_values_input_none_every_call": True,
            "past_key_values_output_none_every_call": True,
            "generate_used": False,
            "prepare_inputs_for_generation_used": False,
            "cached_decode_used": False,
            "feedback_used": False,
            "observed_forward_calls": 68,
            "observed_soft_inputs_embeds_calls": 32,
        }
    ):
        raise ValueError("A0NC claim or no-cache boundary changed")
    guard = receipt.get("cache_guard")
    if (
        not isinstance(guard, dict)
        or guard.get("negative_control_dynamic_cache_tripped") is not True
        or guard.get("restored_in_finally") is not True
        or guard.get("closure_rechecked_after_each_probe_and_finally") is not True
        or not isinstance(guard.get("classes"), list)
        or not guard["classes"]
        or any(
            set(item) != {"fqcn", "module_path", "module_sha256", "package"}
            or not _SHA.fullmatch(item["module_sha256"])
            for item in guard["classes"]
        )
    ):
        raise ValueError("A0NC cache-allocation guard changed")
    disjointness = receipt.get("disjointness")
    if (
        not isinstance(disjointness, dict)
        or disjointness.get("all_parent_child_token_hashes_unique") is not True
        or len(disjointness.get("rendered_token_sha256", {})) != 16
        or disjointness.get("reference_sha256")
        != plan["asset_sha256"]["experiments/qwen35-2b-latent-workspace-v1/a0-nocache-disjointness-v1.json"]
        or disjointness.get("prior_bank_sha256")
        != plan["asset_sha256"]["experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-bank-v1.json"]
        or disjointness.get("fresh_bank_sha256") != plan["bank_sha256"]
    ):
        raise ValueError("A0NC runtime disjointness changed")
    probes = receipt.get("probes")
    if not isinstance(probes, list) or len(probes) != 4 or [p.get("example_id") for p in probes] != _IDS:
        raise ValueError("A0NC probe floor changed")
    for probe in probes:
        if (
            probe.get("complete") is not True
            or probe.get("cache_allocations") != 0
            or probe.get("past_key_values_outputs") != 0
            or len(probe.get("steps", [])) != 4
            or any(
                not isinstance(probe.get("fixture", {}).get(field), bool)
                for field in (
                    "soft_span_active",
                    "soft_span_differs_from_hard_span",
                    "outside_soft_span_exact",
                    "mask_positions_exact",
                    "soft_used_inputs_embeds_without_input_ids",
                )
            )
        ):
            raise ValueError("A0NC no-cache evidence changed")
        for index, step in enumerate(probe["steps"], start=1):
            if (
                step.get("step") != index
                or step.get("continuation_token_id") != _CONTINUATION[index - 1]
                or any(
                    not isinstance(step.get(field), bool)
                    for field in (
                        "l_id_l_e_bitwise_equal",
                        "l_id_l_e_finite",
                        "soft_finite",
                        "soft_repeat_bitwise_equal",
                        "attention_mask_exact_all_visible",
                        "position_ids_exact_sequential",
                    )
                )
                or any(
                    not isinstance(step.get(f), str) or not _SHA.fullmatch(step[f])
                    for f in (
                        "l_id_logits_sha256",
                        "l_e_logits_sha256",
                        "soft_logits_sha256",
                        "soft_repeat_logits_sha256",
                        "attention_mask_sha256",
                        "position_ids_sha256",
                        "continuation_prefix_sha256",
                        "l_id_prefix_sha256",
                        "l_e_prefix_sha256",
                        "soft_prefix_sha256",
                        "soft_repeat_input_sha256",
                    )
                )
                or step.get("prefix_length") != probe["fixture"]["matched_prompt_length"] + index - 1
                or step.get("continuation_prefix_token_ids") != _CONTINUATION[: index - 1]
                or step.get("soft_prefix_sha256") != step.get("soft_repeat_input_sha256")
            ):
                raise ValueError("A0NC step evidence changed")
    qualifies = all(
        p["past_key_values_outputs"] == 0
        and all(
            p["fixture"][field]
            for field in (
                "soft_span_active",
                "soft_span_differs_from_hard_span",
                "outside_soft_span_exact",
                "mask_positions_exact",
                "soft_used_inputs_embeds_without_input_ids",
            )
        )
        and all(
            s["l_id_l_e_bitwise_equal"]
            and s["l_id_l_e_finite"]
            and s["soft_finite"]
            and s["soft_repeat_bitwise_equal"]
            and s["attention_mask_exact_all_visible"]
            and s["position_ids_exact_sequential"]
            for s in p["steps"]
        )
        for p in probes
    )
    if receipt.get("complete_distinct_probes") != 4 or receipt.get("status") != (
        "nocache_receiver_mechanism_validated" if qualifies else "nocache_receiver_mechanism_rejected"
    ):
        raise ValueError("A0NC terminal result changed")
    if receipt.get("receipt_sha256") != canonical_json_hash(receipt, omitted_fields=("receipt_sha256",)):
        raise ValueError("A0NC receipt hash changed")
    resources = receipt.get("resources")
    if not isinstance(resources, dict) or not math.isfinite(resources.get("wall_seconds", math.nan)):
        raise ValueError("A0NC resources changed")
