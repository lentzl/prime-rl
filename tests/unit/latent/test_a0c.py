import copy
import json
from pathlib import Path

import pytest

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a0c import validate_a0c_plan, validate_a0c_receipt

EXPERIMENT = Path("experiments/qwen35-2b-latent-workspace-v1")


def _plan(base_plan: dict[str, object], bank_hash: str) -> dict[str, object]:
    mechanism = copy.deepcopy(base_plan["mechanism"])
    mechanism.pop("cache_full_recompute_max_abs")
    mechanism.pop("cache_full_recompute_normalized_rms")
    admission = copy.deepcopy(base_plan["admission"])
    for field in (
        "cache_prefill_finite",
        "cache_sequence_length_increments_exactly",
        "four_step_decode_finite",
        "cached_vs_full_max_abs_at_most",
        "cached_vs_full_normalized_rms_at_most",
        "cached_vs_full_greedy_tokens_equal",
    ):
        admission.pop(field)
    resources = copy.deepcopy(base_plan["resource_bounds"])
    resources["output_root"] = "/home/ubuntu/rlm/outputs/latent-a0c-carrier-v1"
    asset_paths = (
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
    )
    plan: dict[str, object] = {
        "schema_version": "prime-rl/latent-a0c-carrier-plan/v1",
        "status": "preregistered",
        "execution_authorization": "root_review_required",
        "mechanism_code_commit": "a" * 40,
        "asset_sha256": {path: f"{index:x}" * 64 for index, path in enumerate(asset_paths, start=1)},
        "plan_sha256": "",
        "bank_sha256": bank_hash,
        "derived_from_rejected_plan_sha256": base_plan["plan_sha256"],
        "prior_rejection_evidence": {
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
        },
        "fixture_reuse": {
            "bank_kind": "mechanism-only synthetic fixtures, not held-out semantic evaluation",
            "reason": "A0C repeats pre-cache engineering predicates only and makes no capability or communication claim",
            "minimum_complete_probes": 4,
        },
        "protected_checkpoints": copy.deepcopy(base_plan["protected_checkpoints"]),
        "remote_paths": copy.deepcopy(base_plan["remote_paths"]),
        "runtime": copy.deepcopy(base_plan["runtime"]),
        "mechanism": mechanism,
        "admission": admission,
        "resource_bounds": resources,
        "failure_classification": copy.deepcopy(base_plan["failure_classification"]),
        "dependency_scope": {
            "may_support": ["B frozen-decoder teacher-forced carrier dependency"],
            "does_not_support": [
                "A0 cache mechanism",
                "A1 latent bridge",
                "A1 semantic communication",
                "generation or cached decoding",
                "model admission",
            ],
        },
        "promotion_boundary": (
            "A0C carrier-only dependency evidence for B; no A0/A1, bridge, semantic communication, generation, "
            "cache, training, or model admission claim"
        ),
    }
    plan["plan_sha256"] = canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    return plan


def _inputs() -> tuple[dict[str, object], str, dict[str, object], str, str]:
    base_plan = json.loads((EXPERIMENT / "a0r-mechanism-plan-v1.json").read_text())
    bank_hash = file_sha256(EXPERIMENT / "a0-mechanism-bank-v1.json")
    rejection_path = EXPERIMENT / "a0r-mechanism-rejected-failure.json"
    rejection_log_path = EXPERIMENT / "a0r-mechanism-rejected.log"
    rejection = json.loads(rejection_path.read_text())
    return (
        base_plan,
        bank_hash,
        rejection,
        file_sha256(rejection_path),
        file_sha256(rejection_log_path),
    )


def _receipt(plan: dict[str, object]) -> dict[str, object]:
    probes = []
    for index in range(1, 5):
        probes.append(
            {
                "example_id": f"a0-mechanism-{index:04d}",
                "status": "complete",
                "prompt_tokens": {
                    "parent": 10,
                    "child_without_assistant_opening": 11,
                    "child_with_assistant_opening": 12,
                },
                "hard_bypass": {
                    "bitwise_equal": True,
                    "logits_finite": True,
                    "maximum_absolute_logit_difference": 0.0,
                    "additional_positions": 0,
                    "labels_mask_positions_preserved": True,
                    "four_token_greedy_continuation_equal": True,
                    "four_token_greedy_logits_finite": True,
                    "greedy_token_ids": [1, 2, 3, 4],
                },
                "capture": {
                    "claim_boundary": "rendered_transcript_end_not_harness_acceptance",
                    "layer": -1,
                    "shape": [1, 128, 2048],
                    "finite": True,
                    "detached": True,
                    "repeat_bitwise_equal": True,
                    "mask_and_indices_exact": True,
                    "hidden_padding_content_exact": True,
                    "tensor_bytes_sha256": "a" * 64,
                    "capture_spec_sha256": "b" * 64,
                },
                "soft_insertion": {
                    "claim": "carrier_and_autograd_connectivity_only",
                    "workspace_span": [2, 10],
                    "positions": 8,
                    "logits_finite": True,
                    "loss_finite": True,
                    "original_tokens_mask_labels_preserved": True,
                    "positions_sequential_shifted": True,
                    "inserted_attention_mask_ones": 8,
                    "inserted_loss_mask_negative_100": 8,
                    "no_other_loss_masking": True,
                    "workspace_gradient_finite_nonzero": True,
                    "gate_gradient_finite_nonzero": True,
                    "base_parameter_gradients": 0,
                },
            }
        )
    receipt: dict[str, object] = {
        "schema_version": "prime-rl/latent-a0c-mechanism-receipt/v1",
        "status": "carrier_only_mechanism_validated_for_b_dependency",
        "claim": "four-probe pre-cache carrier/autograd validation for B dependency only",
        "plan_sha256": plan["plan_sha256"],
        "bank_sha256": plan["bank_sha256"],
        "mechanism_code_commit": plan["mechanism_code_commit"],
        "execution_commit": "b" * 40,
        "asset_sha256": copy.deepcopy(plan["asset_sha256"]),
        "versions": {
            "python": "3.12.11",
            "transformers": "5.6.2",
            "torch_distribution": "2.11.0+cu128",
            "torch_runtime": "2.11.0+cu128",
        },
        "model_runtime": {
            "class": "Qwen3_5ForConditionalGeneration",
            "hidden_size": 2048,
            "device": "cuda:0",
            "dtype": "bfloat16",
        },
        "transformers_runtime_sources": {
            name: {
                "path": "/home/ubuntu/rlm/prime-rl/.venv" + suffix,
                "sha256": plan["runtime"]["transformers_source_sha256"][name],
            }
            for name, suffix in {
                "transformers.cache_utils": "/lib/python3.12/site-packages/transformers/cache_utils.py",
                "transformers.generation.utils": "/lib/python3.12/site-packages/transformers/generation/utils.py",
                "transformers.models.qwen3_5.modeling_qwen3_5": (
                    "/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py"
                ),
            }.items()
        },
        "gpu": {"name": "NVIDIA RTX A6000", "total_memory_gib": 47.99},
        "host": {"ram_bytes": 128 * 2**30, "free_disk_bytes_before": 20 * 2**30},
        "checkpoint_metadata_before": {
            "coordinator_e33": plan["runtime"]["checkpoint_metadata_sha256"],
            "worker_h176": plan["runtime"]["checkpoint_metadata_sha256"],
        },
        "checkpoint_metadata_after": {
            "coordinator_e33": plan["runtime"]["checkpoint_metadata_sha256"],
            "worker_h176": plan["runtime"]["checkpoint_metadata_sha256"],
        },
        "protected_hashes_before": copy.deepcopy(plan["protected_checkpoints"]),
        "protected_hashes_after": copy.deepcopy(plan["protected_checkpoints"]),
        "complete_probes": 4,
        "probes": probes,
        "resources": {"wall_seconds": 1.0, "peak_cuda_memory_bytes": 1024},
        "optimizer_created": False,
        "checkpoint_created": False,
        "artifact_contract": {
            "expected_files": ["receipt.json"],
            "maximum_directory_bytes": plan["resource_bounds"]["maximum_output_directory_bytes"],
        },
        "a1_blocker": (
            "A0R cached/full gate remains rejected; live typed-harness action acceptance and capture timing remain "
            "unvalidated"
        ),
        "promotion_boundary": plan["promotion_boundary"],
        "dependency_scope": plan["dependency_scope"],
    }
    receipt["receipt_sha256"] = canonical_json_hash(receipt)
    return receipt


def _validate(plan: dict[str, object], inputs: tuple[dict[str, object], str, dict[str, object], str, str]) -> None:
    base_plan, bank_hash, rejection, rejection_hash, rejection_log_hash = inputs
    validate_a0c_plan(
        plan,
        bank_sha256=bank_hash,
        base_plan=base_plan,
        rejection=rejection,
        rejection_file_sha256=rejection_hash,
        rejection_log_sha256=rejection_log_hash,
    )


def test_a0c_accepts_only_exact_carrier_only_derivation() -> None:
    inputs = _inputs()
    base_plan, bank_hash = inputs[:2]
    _validate(_plan(base_plan, bank_hash), inputs)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: plan["mechanism"].update(cache_full_recompute_normalized_rms=0.02),
        lambda plan: plan["admission"].update(cache_prefill_finite=True),
        lambda plan: plan["fixture_reuse"].update(bank_kind="held-out evaluation"),
        lambda plan: plan["dependency_scope"]["may_support"].append("A1 latent bridge"),
        lambda plan: plan["resource_bounds"].update(output_root="/home/ubuntu/rlm/outputs/latent-a0r-mechanism-v1"),
    ],
)
def test_a0c_plan_rejects_cache_claim_scope_or_namespace_drift(mutation) -> None:
    inputs = _inputs()
    base_plan, bank_hash = inputs[:2]
    plan = _plan(base_plan, bank_hash)
    mutation(plan)
    plan["plan_sha256"] = canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    with pytest.raises(ValueError):
        _validate(plan, inputs)


def test_a0c_plan_rejects_unbound_rejection_evidence() -> None:
    inputs = list(_inputs())
    base_plan, bank_hash = inputs[:2]
    inputs[3] = "0" * 64
    with pytest.raises(ValueError):
        _validate(_plan(base_plan, bank_hash), tuple(inputs))


def test_a0c_receipt_accepts_atomic_four_probe_carrier_evidence() -> None:
    base_plan, bank_hash = _inputs()[:2]
    plan = _plan(base_plan, bank_hash)
    validate_a0c_receipt(_receipt(plan), plan=plan)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda receipt: receipt.update(complete_probes=3),
        lambda receipt: receipt["probes"].pop(),
        lambda receipt: receipt["probes"][0].update(cache_probe={}),
        lambda receipt: receipt["probes"][0]["capture"].update(repeat_bitwise_equal=False),
        lambda receipt: receipt["probes"][0]["soft_insertion"].update(gate_gradient_finite_nonzero=False),
        lambda receipt: receipt["protected_hashes_after"].update(coordinator_e33="0" * 64),
        lambda receipt: receipt.update(optimizer_created=True),
        lambda receipt: receipt.update(a1_blocker="live harness unvalidated"),
    ],
)
def test_a0c_receipt_rejects_partial_cache_or_false_predicate(mutation) -> None:
    base_plan, bank_hash = _inputs()[:2]
    plan = _plan(base_plan, bank_hash)
    receipt = _receipt(plan)
    mutation(receipt)
    receipt["receipt_sha256"] = canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    with pytest.raises(ValueError):
        validate_a0c_receipt(receipt, plan=plan)
