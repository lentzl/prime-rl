import copy
import json
from pathlib import Path

import pytest

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a0dr import (
    _ASSET_PATHS,
    _DIAGNOSTIC,
    _INTERPRETATION,
    _PRIOR_NO_GO,
    validate_a0dr_plan,
    validate_a0dr_receipt,
)

EXPERIMENT = Path("experiments/qwen35-2b-latent-workspace-v1")


def _plan(a0d_plan: dict[str, object], bank_hash: str) -> dict[str, object]:
    resources = copy.deepcopy(a0d_plan["resource_bounds"])
    resources["output_root"] = "/home/ubuntu/rlm/outputs/latent-a0dr-cache-diagnostic-v1"
    plan: dict[str, object] = {
        "schema_version": "prime-rl/latent-a0dr-cache-diagnostic-plan/v1",
        "status": "preregistered",
        "execution_authorization": "root_review_required",
        "mechanism_code_commit": "a" * 40,
        "asset_sha256": {path: f"{index:x}" * 64 for index, path in enumerate(sorted(_ASSET_PATHS), start=1)},
        "plan_sha256": "",
        "bank_sha256": bank_hash,
        "supersedes_no_go": copy.deepcopy(_PRIOR_NO_GO),
        "bound_evidence": copy.deepcopy(a0d_plan["bound_evidence"]),
        "protected_checkpoints": copy.deepcopy(a0d_plan["protected_checkpoints"]),
        "remote_paths": copy.deepcopy(a0d_plan["remote_paths"]),
        "runtime": copy.deepcopy(a0d_plan["runtime"]),
        "diagnostic": copy.deepcopy(_DIAGNOSTIC),
        "resource_bounds": resources,
        "failure_classification": copy.deepcopy(a0d_plan["failure_classification"]),
        "interpretation_boundary": _INTERPRETATION,
    }
    plan["plan_sha256"] = canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    return plan


def _inputs() -> tuple[dict[str, object], str]:
    prior = json.loads((EXPERIMENT / "a0d-cache-diagnostic-plan-v1.json").read_text())
    return prior, file_sha256(EXPERIMENT / "a0-mechanism-bank-v1.json")


def _receipt(plan: dict[str, object]) -> dict[str, object]:
    continuation = [11, 12, 13, 14]
    names = ("D47", "E47", "L_ID55", "L_E55", "S55")
    arms = []
    for name in names:
        initial = 47 if name in {"D47", "E47"} else 55
        for branch in ("auto_position", "explicit_next_position"):
            steps = []
            for index, token in enumerate(continuation, start=1):
                keys = ["attention_mask", "input_ids", "past_key_values", "use_cache"]
                values: dict[str, object] = {
                    "input_ids": {"values": [token]},
                    "past_key_values": {"sequence_length": initial + index - 1},
                }
                if branch == "explicit_next_position":
                    keys.extend(("cache_position", "position_ids"))
                    values["cache_position"] = {"values": [initial + index - 1]}
                    values["position_ids"] = {"values": [initial + index - 1]}
                steps.append(
                    {
                        "step": index,
                        "cache_sequence_length": initial + index,
                        "maximum_absolute_logit_difference": 0.2,
                        "normalized_rms": 0.02,
                        "greedy_equal": True,
                        "cached_logits_sha256": "5" * 64,
                        "full_logits_sha256": "6" * 64,
                        "prepared": {"keys": sorted(keys), "values": values},
                        "rope_state": {
                            "before_prepare": {"state": "none"},
                            "after_prepare": {"state": "none"},
                            "after_decode": {"state": "none"},
                        },
                    }
                )
            arms.append(
                {
                    "arm": name,
                    "position_branch": branch,
                    "fresh_cache": True,
                    "initial_cache_sequence_length": initial,
                    "initial_logits_finite": True,
                    "prefill_cache_type": "DynamicCache",
                    "prefill_last_logits_sha256": "7" * 64,
                    "rope_state": {"before_prefill": {"state": "none"}, "after_prefill": {"state": "none"}},
                    "steps": steps,
                }
            )
    metadata = {
        "coordinator_e33": plan["runtime"]["checkpoint_metadata_sha256"],
        "worker_h176": plan["runtime"]["checkpoint_metadata_sha256"],
    }
    receipt: dict[str, object] = {
        "schema_version": "prime-rl/latent-a0dr-cache-diagnostic-receipt/v1",
        "status": "diagnostic_complete",
        "claim": "non-promotional cache causal measurements only",
        "plan_sha256": plan["plan_sha256"],
        "bank_sha256": plan["bank_sha256"],
        "mechanism_code_commit": plan["mechanism_code_commit"],
        "execution_commit": "b" * 40,
        "asset_sha256": copy.deepcopy(plan["asset_sha256"]),
        "versions": {
            "python": "3.12.14",
            "transformers": "5.6.2",
            "torch_distribution": "2.11.0+cu128",
            "torch_runtime": "2.11.0+cu128",
        },
        "transformers_runtime_sources": {
            name: {"path": f"/home/ubuntu/rlm/prime-rl/.venv/{name}.py", "sha256": digest}
            for name, digest in plan["runtime"]["transformers_source_sha256"].items()
        },
        "model_runtime": {
            "class": "Qwen3_5ForConditionalGeneration",
            "hidden_size": 2048,
            "device": "cuda:0",
            "dtype": "bfloat16",
        },
        "gpu": {"name": "NVIDIA RTX A6000", "total_memory_gib": 47.4},
        "host": {"ram_bytes": 80 * 2**30, "free_disk_bytes_before": 20 * 2**30},
        "protected_hashes_before": copy.deepcopy(plan["protected_checkpoints"]),
        "protected_hashes_after": copy.deepcopy(plan["protected_checkpoints"]),
        "checkpoint_metadata_before": copy.deepcopy(metadata),
        "checkpoint_metadata_after": copy.deepcopy(metadata),
        "fixture": {
            "example_id": "a0-mechanism-0001",
            "parent_token_count": 93,
            "child_token_count": 47,
            "soft_prompt_length": 55,
            "injection_index": 40,
            "length_control_token_ids": [40, 4021, 2528, 8976, 35139, 635, 524, 599],
            "length_control_token_ids_sha256": "e86e01e61315008783cc217a5bb83a1b3aced0daaecbc920b8d3b45ab4b205d8",
            "length_control_tokens_non_special": True,
            "length_control_input_ids_sha256": "8" * 64,
            "length_matched_masks_and_positions_exact": True,
            "child_input_ids_sha256": "1" * 64,
            "continuation_input_ids_sha256": "2" * 64,
            "continuation_token_ids": continuation,
            "workspace_source_sha256": "3" * 64,
            "soft_prompt_sha256": "4" * 64,
        },
        "reference_normalized_rms": 0.01,
        "reference_is_promotion_gate": False,
        "arms": arms,
        "optimizer_created": False,
        "checkpoint_created": False,
        "model_update_attempted": False,
        "resources": {"wall_seconds": 1.0, "peak_cuda_memory_bytes": 1024},
        "interpretation_boundary": plan["interpretation_boundary"],
    }
    receipt["receipt_sha256"] = canonical_json_hash(receipt)
    return receipt


def test_a0dr_accepts_exact_length_matched_superseding_design() -> None:
    prior, bank_hash = _inputs()
    validate_a0dr_plan(_plan(prior, bank_hash), bank_sha256=bank_hash, a0d_plan=prior)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: plan["supersedes_no_go"].update(status="go"),
        lambda plan: plan["diagnostic"].update(length_control_token_ids=[1] * 8),
        lambda plan: plan["diagnostic"].update(length_control_tokens_must_be_non_special=False),
        lambda plan: plan["diagnostic"].update(causal_core_prefill_length=47),
        lambda plan: plan["diagnostic"].update(reference_is_promotion_gate=True),
        lambda plan: plan["diagnostic"].update(model_update=True),
    ],
)
def test_a0dr_plan_rejects_no_go_length_control_or_scope_drift(mutation) -> None:
    prior, bank_hash = _inputs()
    plan = _plan(prior, bank_hash)
    mutation(plan)
    plan["plan_sha256"] = canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    with pytest.raises(ValueError):
        validate_a0dr_plan(plan, bank_sha256=bank_hash, a0d_plan=prior)


def test_a0dr_receipt_accepts_complete_length_matched_matrix() -> None:
    prior, bank_hash = _inputs()
    plan = _plan(prior, bank_hash)
    validate_a0dr_receipt(_receipt(plan), plan=plan)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda receipt: receipt["fixture"].update(length_control_token_ids=[1] * 8),
        lambda receipt: receipt["fixture"].update(length_matched_masks_and_positions_exact=False),
        lambda receipt: receipt["arms"].pop(),
        lambda receipt: receipt["arms"][4].update(initial_cache_sequence_length=54),
        lambda receipt: receipt["arms"][0]["steps"][0].update(normalized_rms=float("nan")),
        lambda receipt: receipt.update(reference_is_promotion_gate=True),
    ],
)
def test_a0dr_receipt_rejects_length_matrix_metric_or_promotion_drift(mutation) -> None:
    prior, bank_hash = _inputs()
    plan = _plan(prior, bank_hash)
    receipt = _receipt(plan)
    mutation(receipt)
    receipt["receipt_sha256"] = canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    with pytest.raises(ValueError):
        validate_a0dr_receipt(receipt, plan=plan)
