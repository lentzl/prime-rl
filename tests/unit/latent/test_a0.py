import copy
import json

import pytest

from prime_rl.latent.a0 import canonical_json_hash, file_sha256, validate_a0_bank, validate_a0_plan


def _bank() -> dict[str, object]:
    return {
        "schema_version": "prime-rl/latent-a0-mechanism-bank/v1",
        "examples": [
            {
                "example_id": f"a0-mechanism-{index:04d}",
                "parent_messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ],
                "child_messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                ],
                "continuation_text": " Acknowledged and continuing safely.",
            }
            for index in range(1, 5)
        ],
    }


def _plan(bank_hash: str) -> dict[str, object]:
    plan: dict[str, object] = {
        "schema_version": "prime-rl/latent-a0-mechanism-plan/v1",
        "status": "preregistered",
        "execution_authorization": "root_review_required",
        "mechanism_code_commit": "a" * 40,
        "asset_sha256": {
            "experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-bank-v1.json": "1" * 64,
            "scripts/latent/run_a0_mechanism_v1.py": "2" * 64,
            "scripts/latent/run_a0_mechanism_v1.sh": "3" * 64,
            "src/prime_rl/latent/__init__.py": "4" * 64,
            "src/prime_rl/latent/a0.py": "5" * 64,
            "src/prime_rl/latent/policy_adapter.py": "6" * 64,
        },
        "plan_sha256": "",
        "bank_sha256": bank_hash,
        "protected_checkpoints": {
            "coordinator_e33": "e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47",
            "worker_h176": "77980e247bbccd6463ddda02cd42d2c357e15f8ec1ad0ea84627e008a8674a1e",
        },
        "remote_paths": {
            "coordinator_e33": "/home/ubuntu/rlm/e33",
            "worker_h176": "/home/ubuntu/rlm/h176",
        },
        "runtime": {
            "python": "3.12",
            "transformers": "5.6.2",
            "torch": "2.11.0",
            "model_class": "Qwen3_5ForConditionalGeneration",
            "hidden_size": 2048,
            "device": "cuda:0",
            "dtype": "bfloat16",
            "attention_implementation": "eager",
            "local_files_only": True,
            "checkpoint_metadata_sha256": {
                "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
                "config.json": "22949388ed61c1100b20a3cae55bb22122554c74e06fc23f1be50cca1fec3b8c",
                "generation_config.json": "93f19a5ed0fb9f9e8e65dafae7a9bc4c6a32b3e37f6278980d05d3f4ca29f17b",
                "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
                "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
                "tokenizer_config.json": "747ba36a06ba5428bb74e984d75136b37cf5dafe97b8dd315f701b361a9f417f",
            },
            "transformers_source_sha256": {
                "transformers.cache_utils": "a51d2cd525f4458941a20cb5b3b8a8d989d8ca5bcd451855a27bb41743fed586",
                "transformers.generation.utils": "2cf094451e1f30391c3449cfd44828ab8b27ba721ae2cdc0a0415413689b0f9c",
                "transformers.models.qwen3_5.modeling_qwen3_5": (
                    "3e2b6239e4b2c3e512d4f9836a1dba12e975ae652407e84f2d3c2beebf0c9528"
                ),
            },
        },
        "mechanism": {
            "capture_layer": -1,
            "capture_boundary": "rendered_transcript_end_not_harness_acceptance",
            "capture_tokens": 128,
            "capture_detached": True,
            "workspace_slots": 8,
            "workspace_projection": "identity_final_hidden_2048_probe_only",
            "injection_boundary": "before_assistant_generation_opening",
            "hard_bypass_gate": 0.0,
            "soft_probe_gate": 0.125,
            "soft_probe_trainable": ["workspace_input", "receiver_gate"],
            "base_model_trainable_parameters": 0,
            "optimizer": None,
            "model_update": False,
            "decode_probe_tokens": 4,
            "cache_full_recompute_max_abs": 0.5,
            "cache_full_recompute_normalized_rms": 0.01,
        },
        "admission": {
            "standard_vs_hard_bypass_logits": "bitwise_equal",
            "standard_and_hard_bypass_logits_finite": True,
            "standard_vs_hard_bypass_four_token_greedy": "exact",
            "standard_and_hard_bypass_greedy_logits_finite": True,
            "hard_bypass_additional_positions": 0,
            "hard_bypass_labels_mask_positions_preserved": True,
            "soft_insertion_positions": 8,
            "soft_logits_finite": True,
            "soft_loss_finite": True,
            "soft_original_tokens_mask_labels_preserved": True,
            "soft_positions_sequential_shifted": True,
            "soft_inserted_attention_mask_ones": 8,
            "soft_inserted_loss_mask_negative_100": 8,
            "soft_no_other_loss_masking": True,
            "workspace_gradient_finite_nonzero": True,
            "gate_gradient_finite_nonzero": True,
            "captured_hidden_finite": True,
            "captured_hidden_detached": True,
            "capture_repeat_bitwise_equal": True,
            "capture_mask_and_indices_exact": True,
            "capture_hidden_padding_content_exact": True,
            "capture_tensor_sha256_recorded": True,
            "cache_prefill_finite": True,
            "cache_sequence_length_increments_exactly": True,
            "four_step_decode_finite": True,
            "cached_vs_full_max_abs_at_most": 0.5,
            "cached_vs_full_normalized_rms_at_most": 0.01,
            "cached_vs_full_greedy_tokens_equal": True,
            "protected_hashes_unchanged": True,
            "minimum_complete_probes": 4,
            "receipt_complete": True,
        },
        "resource_bounds": {
            "gpus_used": 1,
            "gpu_model": "NVIDIA RTX A6000",
            "minimum_gpu_memory_gib": 47,
            "minimum_host_ram_gib": 64,
            "minimum_free_disk_gib": 8,
            "maximum_output_bytes": 1048576,
            "maximum_output_directory_bytes": 16777216,
            "network": False,
            "maximum_wall_minutes": 30,
            "failure_cleanup_headroom_minutes": 2,
            "output_root": "/home/ubuntu/rlm/outputs/latent-a0-mechanism-v1",
        },
        "failure_classification": {
            "mechanism_predicate_failure": "mechanism_rejected",
            "environment_provenance_timeout_or_oom": "infrastructure_invalid",
            "run_id_reusable": False,
        },
        "promotion_boundary": (
            "A0 rendered-transcript carrier/autograd mechanism only; no harness acceptance/timing, bridge learnability, "
            "training, model admission, or A1 authorization"
        ),
    }
    plan["plan_sha256"] = canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    return plan


def test_a0_bank_and_plan_validate(tmp_path) -> None:
    bank = _bank()
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(bank))
    validate_a0_bank(bank)
    validate_a0_plan(_plan(file_sha256(bank_path)), bank_sha256=file_sha256(bank_path))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: plan.update(execution_authorization="owner_approved"),
        lambda plan: plan["asset_sha256"].pop("scripts/latent/run_a0_mechanism_v1.sh"),
        lambda plan: plan["runtime"].update(transformers="5.7.0"),
        lambda plan: plan["mechanism"].update(model_update=True),
        lambda plan: plan["admission"].update(standard_vs_hard_bypass_logits="close"),
        lambda plan: plan["resource_bounds"].update(minimum_free_disk_gib=250),
    ],
)
def test_a0_plan_fails_closed_on_changed_contract(tmp_path, mutation) -> None:
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(_bank()))
    plan = copy.deepcopy(_plan(file_sha256(bank_path)))
    mutation(plan)
    plan["plan_sha256"] = canonical_json_hash(plan, omitted_fields=("plan_sha256",))

    with pytest.raises(ValueError):
        validate_a0_plan(plan, bank_sha256=file_sha256(bank_path))
