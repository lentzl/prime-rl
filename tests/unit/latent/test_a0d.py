import copy
import json
from pathlib import Path

import pytest

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a0d import validate_a0d_plan, validate_a0d_receipt

EXPERIMENT = Path("experiments/qwen35-2b-latent-workspace-v1")


def _plan(a0c_plan: dict[str, object], bank_hash: str) -> dict[str, object]:
    asset_paths = (
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
    )
    resources = copy.deepcopy(a0c_plan["resource_bounds"])
    resources["output_root"] = "/home/ubuntu/rlm/outputs/latent-a0d-cache-diagnostic-v1"
    plan: dict[str, object] = {
        "schema_version": "prime-rl/latent-a0d-cache-diagnostic-plan/v1",
        "status": "preregistered",
        "execution_authorization": "root_review_required",
        "mechanism_code_commit": "a" * 40,
        "asset_sha256": {path: f"{index:x}" * 64 for index, path in enumerate(asset_paths, start=1)},
        "plan_sha256": "",
        "bank_sha256": bank_hash,
        "bound_evidence": {
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
                "receipt_file_sha256": "d88dd97eb37c9c3dd61bc07fe422df6c7fa0034837897346e43ed16bb634e63c",
                "receipt_internal_sha256": "40dde68d34deb592f864739b48da8c22faafe470f2f0bc6708bce608ae482de7",
                "plan_sha256": "0def45abb5981761eaf34a9228313cf9494e2fdf23c39055a36e095fd70482be",
                "execution_commit": "81e10c5ff362a4ac586fc9ecd81cdfe6f2196583",
                "complete_probes": 4,
            },
        },
        "protected_checkpoints": copy.deepcopy(a0c_plan["protected_checkpoints"]),
        "remote_paths": copy.deepcopy(a0c_plan["remote_paths"]),
        "runtime": copy.deepcopy(a0c_plan["runtime"]),
        "diagnostic": {
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
                "explicit_next_position": (
                    "prepare_inputs_for_generation with exact next position_ids and cache_position"
                ),
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
        },
        "resource_bounds": resources,
        "failure_classification": {
            "diagnostic_execution_or_finiteness_failure": "diagnostic_incomplete",
            "environment_provenance_timeout_or_oom": "infrastructure_invalid",
            "run_id_reusable": False,
        },
        "interpretation_boundary": (
            "non-promotional causal diagnostic only; A0R remains rejected and A1 remains blocked regardless of "
            "result; the 0.01 value is a historical reference, not an A0D gate"
        ),
    }
    plan["plan_sha256"] = canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    return plan


def _inputs() -> tuple[dict[str, object], str]:
    plan = json.loads((EXPERIMENT / "a0c-carrier-plan-v1.json").read_text())
    bank_hash = file_sha256(EXPERIMENT / "a0-mechanism-bank-v1.json")
    return plan, bank_hash


def _receipt(plan: dict[str, object]) -> dict[str, object]:
    continuation = [11, 12, 13, 14]
    arms = []
    for arm_name in ("D", "E", "S"):
        initial = 55 if arm_name == "S" else 47
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
                        "maximum_absolute_logit_difference": 0.125,
                        "normalized_rms": 0.015,
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
                    "arm": arm_name,
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
        "schema_version": "prime-rl/latent-a0d-cache-diagnostic-receipt/v1",
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


def test_a0d_accepts_exact_nonpromotional_causal_design() -> None:
    a0c_plan, bank_hash = _inputs()
    validate_a0d_plan(_plan(a0c_plan, bank_hash), bank_sha256=bank_hash, a0c_plan=a0c_plan)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: plan["bound_evidence"]["a0r"].update(status="validated"),
        lambda plan: plan["bound_evidence"]["a0c"].update(complete_probes=1),
        lambda plan: plan["diagnostic"].update(example_id="a0-mechanism-0002"),
        lambda plan: plan["diagnostic"].update(reference_is_promotion_gate=True),
        lambda plan: plan["diagnostic"]["arms"].update(S="trained bridge"),
        lambda plan: plan["diagnostic"].update(per_layer_diagnostics=True),
        lambda plan: plan["diagnostic"].update(model_update=True),
        lambda plan: plan["resource_bounds"].update(output_root="/home/ubuntu/rlm/outputs/latent-a0r-mechanism-v1"),
        lambda plan: plan.update(interpretation_boundary="A1 authorized"),
    ],
)
def test_a0d_rejects_evidence_scope_threshold_or_update_drift(mutation) -> None:
    a0c_plan, bank_hash = _inputs()
    plan = _plan(a0c_plan, bank_hash)
    mutation(plan)
    plan["plan_sha256"] = canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    with pytest.raises(ValueError):
        validate_a0d_plan(plan, bank_sha256=bank_hash, a0c_plan=a0c_plan)


def test_a0d_receipt_accepts_complete_nonpromotional_matrix() -> None:
    a0c_plan, bank_hash = _inputs()
    plan = _plan(a0c_plan, bank_hash)
    validate_a0d_receipt(_receipt(plan), plan=plan)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda receipt: receipt.update(reference_is_promotion_gate=True),
        lambda receipt: receipt.update(model_update_attempted=True),
        lambda receipt: receipt["arms"].pop(),
        lambda receipt: receipt["arms"][0]["steps"].pop(),
        lambda receipt: receipt["arms"][0]["steps"][0].update(normalized_rms=float("nan")),
        lambda receipt: receipt["arms"][1]["steps"][0]["prepared"]["values"]["position_ids"].update(values=[999]),
        lambda receipt: receipt["protected_hashes_after"].update(coordinator_e33="0" * 64),
    ],
)
def test_a0d_receipt_rejects_promotion_update_partial_or_provenance_drift(mutation) -> None:
    a0c_plan, bank_hash = _inputs()
    plan = _plan(a0c_plan, bank_hash)
    receipt = _receipt(plan)
    mutation(receipt)
    receipt["receipt_sha256"] = canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    with pytest.raises(ValueError):
        validate_a0d_receipt(receipt, plan=plan)
