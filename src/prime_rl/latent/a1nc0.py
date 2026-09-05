from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.transfer_bank import FAMILIES, build_transfer_bank

A1NC0_PLAN_SCHEMA = "prime-rl/latent-a1-nc0-r1-plan/v1"
A1NC0_RECEIPT_SCHEMA = "prime-rl/latent-a1-nc0-r1-receipt/v1"

_SHA = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_OPAQUE_IDENTIFIER = re.compile(r"(?<![A-Z0-9])[A-Z][A-Z0-9]{5}(?![A-Z0-9])")
_E33 = "e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47"
_H176 = "77980e247bbccd6463ddda02cd42d2c357e15f8ec1ad0ea84627e008a8674a1e"
_ARMS = ("M0", "MOTH", "MSELF", "MCUR", "ZERO", "NOISE")
_NOISE_NORM_RELATIVE_TOLERANCE = 1e-5
_SPLITS = {
    "train": {
        "seed": 771168341,
        "examples_per_family": 16,
        "records": 64,
        "queries": 192,
        "manifest_sha256": "5819ed6acba9524eeb8f9b537ee71c9b73e4babc13d3689fafb7ce3ef825c5ac",
        "file_sha256": "cea92c57536ec7a93e68c64c4a01669e11dab1dc483a86c711c97c1181bb08d8",
    },
    "validation": {
        "seed": 213801847,
        "examples_per_family": 4,
        "records": 16,
        "queries": 48,
        "manifest_sha256": "587a4de77e9607217055b70f3640cbc219a10d515f7936e9ee357c694d13b36e",
        "file_sha256": "8e87309df0c37cce7bbd0bb98a3daed098f0b354f0c00ae15c27024047539c5e",
    },
    "held_out": {
        "seed": 27267994,
        "examples_per_family": 4,
        "records": 16,
        "queries": 48,
        "manifest_sha256": "2201b93fe3e250d6816a09a86a69c72e6e0757d5eb06d5e5d457d2a6a56b2b7d",
        "file_sha256": "204e6605fca20f94b23468fbd9f56b86ca16e8296191dd09b877a021c9ab2381",
    },
}
_SEEDS = {
    "arm_order": 310138650,
    "bridge_init": 3772224393,
    "noise": 2671655313,
}
_TRAIN_SCHEDULE_SHA256 = "5c9c067a08f245716cab2330ac9151867fb0faded031c598e1d7e3742548a597"
_ARM_ORDER_SHA256 = {
    "validation": "813bb5a54ac8c3faf976e1bb5059f6f8b31dce1847afea5fa35ac4fda76f7bc2",
    "held_out": "d429872cd414f768b5e5be3aa65ece6087caaf3a34eb804ec89ff124f660e9d6",
    "combined": "9ee2bff9897ffe5f0ff2e0e375864f6c19dbe870b72c22564ec04d4594cab599",
}
_A0NC_REPEAT_SELECTION = [
    {
        "family": "keyed_numeric",
        "evidence_id": "train-keyed_numeric-7b852e1763dcb590",
        "query_id": "train-keyed_numeric-7b852e1763dcb590-q0",
    },
    {
        "family": "relational_join",
        "evidence_id": "train-relational_join-b555e514ce686052",
        "query_id": "train-relational_join-b555e514ce686052-q0",
    },
    {
        "family": "config_structure",
        "evidence_id": "train-config_structure-87f3a2c174d52881",
        "query_id": "train-config_structure-87f3a2c174d52881-q0",
    },
    {
        "family": "ownership_graph",
        "evidence_id": "train-ownership_graph-941cbc927f66476b",
        "query_id": "train-ownership_graph-941cbc927f66476b-q0",
    },
]
_A0NC_REPEAT_SELECTION_SHA256 = "cbdbdd1c6e3e6678165f12025663028f32522eb8f658472676af6b6e86d89333"
_TRAINING = {
    "epochs": 4,
    "updates_per_epoch": 16,
    "queries_per_update": 12,
    "total_updates": 64,
    "total_query_exposures": 768,
    "optimizer": "AdamW",
    "learning_rate": 0.0001,
    "betas": [0.9, 0.95],
    "epsilon": 1e-8,
    "weight_decay": 0.0,
    "gradient_clip_norm": 1.0,
    "early_stopping": False,
    "checkpoint_selection": False,
    "trainable_scope": "workspace_bridge_and_receiver_gate_only",
    "objective": "MCUR answer-token cross_entropy",
    "teacher_input_terminal_ids_excluded": True,
    "suffix_logits_to_keep": "answer_token_count_plus_one",
    "causal_alignment": "leading_minus100_then_contiguous_answer_tokens",
    "train_parent_feature_captures": 64,
    "train_parent_feature_cache": "detached_immutable_host_ram",
    "bridge_recomputed_per_query_exposure": True,
    "pretraining_repeat_distinct_probes": 4,
    "pretraining_repeat_e33_forwards": 78,
    "pretraining_repeat_optimizer_step": False,
    "optimizer_destroyed_before_evaluation": True,
    "optimizer_state_persisted": False,
    "initialization_seed_calls": ["torch.manual_seed", "torch.cuda.manual_seed_all"],
}
_EVALUATION = {
    "arms": list(_ARMS),
    "validation_before_held_out": True,
    "decode_steps": 12,
    "decode_mode": "greedy_full_prefix_recompute_use_cache_false",
    "after_first_eos": "append_eos_for_remaining_fixed_steps",
    "feature_input_tokens": 256,
    "capture_tokens": 128,
    "capture_layer": -1,
    "workspace_slots": 8,
    "workspace_width": 256,
    "receiver_width": 2048,
    "parent_query_visibility": False,
    "parent_answer_visibility": False,
    "workspace_reused_bitwise_across_three_queries": True,
    "mself_primary_compute_match": "equal_operation",
    "no_cache": True,
    "tokenizer_eos_and_pad_token_id": 248046,
    "terminal_marker_token_ids": [248046, 198],
    "forced_steps_after_first_terminal": True,
    "gold_nll_under_actual_rollout_prefix": True,
    "paired_nll_win_epsilon": 1e-6,
    "canonical_captures_and_bridges_per_split": 16,
    "query_witness_captures_and_bridges_per_split": 144,
    "receiver_forwards_per_split": 3456,
    "total_e33_forwards_through_validation": 4526,
    "total_e33_forwards_with_held_out": 8142,
    "mcur_mself_cuda_event_relative_difference_maximum": 0.10,
}
_VALIDATION_GATE = {
    "complete_tasks": 48,
    "mcur_exact_minimum": 12,
    "mcur_minus_moth_exact_net_minimum": 4,
    "m0_to_mcur_recoveries_minimum": 6,
    "m0_to_mcur_regressions_maximum": 3,
    "m0_to_mcur_recovery_families_minimum": 2,
    "mcur_minus_moth_mean_answer_token_nll_minimum": 0.02,
    "mcur_vs_moth_paired_nll_wins_minimum": 30,
}
_NOMINATION_GATE = {
    "complete_tasks": 48,
    "mcur_exact_minimum": 16,
    "m0_to_mcur_recoveries_minimum": 9,
    "m0_to_mcur_regressions_maximum": 2,
    "m0_to_mcur_recovery_families_minimum": 3,
    "mcur_vs_moth_exact_wins_minimum": 10,
    "mcur_vs_moth_exact_losses_maximum": 3,
    "mcur_minus_moth_exact_net_minimum": 7,
    "mcur_vs_moth_win_families_minimum": 3,
    "mcur_minus_moth_mean_answer_token_nll_minimum": 0.02,
    "mcur_vs_moth_paired_nll_wins_minimum": 30,
    "mcur_vs_mself_exact_wins_minimum": 7,
    "mcur_vs_mself_exact_losses_maximum": 3,
    "mcur_minus_mself_exact_net_minimum": 4,
    "mcur_minus_mself_mean_answer_token_nll_minimum": 0.01,
    "mcur_vs_mself_paired_nll_wins_minimum": 27,
    "mcur_minus_zero_exact_net_minimum": 6,
    "mcur_minus_zero_mean_answer_token_nll_minimum": 0.02,
    "mcur_minus_noise_exact_net_minimum": 6,
    "mcur_minus_noise_mean_answer_token_nll_minimum": 0.02,
    "moth_recovery_fraction_of_mcur_maximum": 0.25,
    "noise_recovery_fraction_of_mcur_maximum": 0.25,
    "mcur_exact_per_family_minimum": 2,
}
_INTERPRETATION = (
    "nomination-only no-cache bridge learnability; A0R and relative-cache calibration remain rejected; "
    "A1 admission, live typed-harness delivery, A2, model promotion, and any A+B claim remain blocked"
)
_A0NC_SUCCESS = {
    "status": "nocache_receiver_mechanism_validated",
    "claim": "no_cache_full_recompute_diagnostic_valid_for_B",
    "receipt_file_sha256": "4a713486110d8c17c1fb6e03ffddb87ffc514ecd840ddc4c3f5998021b36f4f2",
    "receipt_internal_sha256": "20186c98f3d2f9a56ec9db9b02be5dc1302aa12ca2049076d402a98c88a7c2ba",
    "plan_sha256": "6f50fc1c83cafac99f89212aa4b8349984caed4e4b389fff284edc1922e860fd",
    "bank_sha256": "b77df46145d67e9147f42b9dd1e403a6253955e2a67bab3c95490357fe255ea3",
    "mechanism_code_commit": "8d18937bdf4dedc92d4d9abf88070784a2a4a5c0",
    "execution_commit": "a150e2cee216c04fe4d600c039edf472d91fdc35",
    "complete_distinct_probes": 4,
    "protected_hashes_before": {"coordinator_e33": _E33, "worker_h176": _H176},
    "protected_hashes_after": {"coordinator_e33": _E33, "worker_h176": _H176},
    "model_update_attempted": False,
    "optimizer_created": False,
    "checkpoint_created": False,
    "prior_cache_rejection": {
        "status": "relative_cache_rejected",
        "receipt_file_sha256": "707d507bd24f64ea7b4d872268d5335520bd7f1e7771f380b37fa1fb5be97ab1",
        "receipt_internal_sha256": "10b5008dcc8ba45f8375590d850fd4306e1d591d6e4e6da7b1df3c7a98deae23",
        "launch_log_sha256": "c3b3d2ff004e46543bcc9435256c39b00f3e8d957c273dfac819150e76cb949d",
        "snapshot_manifest_sha256": "b1bc015a9711015592d89edfdaefe1343da69e78ac09584131a7c24d49102cc9",
        "execution_commit": "2b7d3c8b26c6813da4fb7e534c0061b3814769ec",
        "plan_sha256": "eeda1c571359b29882f16acbd50f573ecd3fa71238882f9f95bb99040e8e7578",
        "complete_distinct_probes": 4,
        "qualifying_probes": 2,
    },
}
_PRIOR_RENDER_REJECTION = {
    "status": "mechanism_rejected",
    "failure_category": "training_or_evaluation_contract_incomplete",
    "error_type": "ExperimentIncomplete",
    "error": "A1-NC0 materialized render boundary changed",
    "stage": "artifact_namespace_created",
    "failure_file_sha256": "9e41fb8107af77ae5258c10d9caff7f13773aaa26162850faab7c085c6440d80",
    "failure_internal_sha256": "d6e688e377054b667ab1889c217a0d3da35c578efd373c4deeb3a1e363ce85b1",
    "launch_log_sha256": "5d4f519dba35d5e8307fce54b1d979bfa937a7d636e4abb29fdddef14b7f0fc3",
    "snapshot_manifest_sha256": "755bd5ae763d2309f2cd77b0771be2c430e4370c88287bc14c3b3c4a835a6eb5",
    "execution_commit": "810d5610ab080e58776be610a6bcf927c16b964f",
    "mechanism_code_commit": "efa9a79baec118c80c279031a5e08a9c5ac9e015",
    "plan_sha256": "35c75821e7e5b1a359c31536dc3047d3d9e42785316f7a5631263d1d7e78458d",
    "cuda_runtime_contacted": True,
    "model_or_material_allocation_attempted": False,
    "base_model_update_attempted": False,
    "bridge_update_attempted": False,
    "optimizer_created": False,
    "checkpoint_created": False,
    "candidate_inventory": [],
    "candidate_valid": False,
    "memory_ledger_partial": [],
    "cache_guard_partial": None,
    "protected_hashes_exact": True,
    "frozen_assets_exact": True,
    "run_id_reusable": False,
}
_FAILURE_CLASSIFICATION = {
    "infrastructure_or_provenance_failure": "infrastructure_invalid",
    "mechanism_or_contract_failure": "mechanism_rejected",
    "validation_gate_not_met": "valid_not_nominated_validation",
    "held_out_nomination_gate_not_met": "valid_not_nominated",
    "held_out_nomination_gate_met": "a1_nc0_nominated",
    "run_id_reusable": False,
}
_BRIDGE = {
    "schema_version": "prime-rl/workspace-bridge/v1",
    "source_width": 2048,
    "workspace_width": 256,
    "receiver_width": 2048,
    "slots": 8,
    "attention_heads": 8,
    "initial_receiver_gate": 0.001,
    "trainable_parameter_count": 1321217,
    "receiver_gate_application": "inside_WorkspaceDecoder_then_compose_gate_exactly_1.0",
    "candidate_tensor_names": [
        "decoder.projection.bias",
        "decoder.projection.weight",
        "decoder.receiver_gate",
        "decoder.workspace_norm.bias",
        "decoder.workspace_norm.weight",
        "encoder.learned_queries",
        "encoder.output_norm.bias",
        "encoder.output_norm.weight",
        "encoder.resampler.in_proj_bias",
        "encoder.resampler.in_proj_weight",
        "encoder.resampler.out_proj.bias",
        "encoder.resampler.out_proj.weight",
        "encoder.source_norm.bias",
        "encoder.source_norm.weight",
        "encoder.source_projection.bias",
        "encoder.source_projection.weight",
    ],
    "candidate_tensor_shapes": {
        "decoder.projection.bias": [2048],
        "decoder.projection.weight": [2048, 256],
        "decoder.receiver_gate": [],
        "decoder.workspace_norm.bias": [256],
        "decoder.workspace_norm.weight": [256],
        "encoder.learned_queries": [8, 256],
        "encoder.output_norm.bias": [256],
        "encoder.output_norm.weight": [256],
        "encoder.resampler.in_proj_bias": [768],
        "encoder.resampler.in_proj_weight": [768, 256],
        "encoder.resampler.out_proj.bias": [256],
        "encoder.resampler.out_proj.weight": [256, 256],
        "encoder.source_norm.bias": [2048],
        "encoder.source_norm.weight": [2048],
        "encoder.source_projection.bias": [256],
        "encoder.source_projection.weight": [256, 2048],
    },
    "candidate_tensor_dtype": "torch.float32",
}
_DISJOINTNESS = {
    "file_sha256": "44c78fb0680d837a284c0d63b6154b51ea1b76ea4f3a0cf0dac7de56c96974e2",
    "report_sha256": "fc121902c22bbf6065b14ba010502b27a103550e0d252b0ff7540a3f1b4ca164",
    "axes": [
        "evidence_ids",
        "query_ids",
        "structured_evidence_sha256",
        "parent_evidence_text",
        "child_query_text",
        "opaque_identifiers",
    ],
    "all_pairwise_intersections_zero": True,
}
_RUNTIME = {
    "python": "3.12.14",
    "transformers": "5.6.2",
    "flash_linear_attention": "0.5.2",
    "torch_distribution": "2.11.0+cu128",
    "torch_runtime": "2.11.0+cu128",
    "model_class": "Qwen3_5ForConditionalGeneration",
    "hidden_size": 2048,
    "vocab_size": 248320,
    "device": "cuda:0",
    "dtype": "bfloat16",
    "attention_implementation": "eager",
    "local_files_only": True,
    "tokenizer_eos_token_id": 248046,
    "tokenizer_pad_token_id": 248046,
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
        "transformers.loss.loss_utils": "3df3438f0cb80cb7b903d6d06f522839756f4ca9301a02d793748ae099c27d9f",
        "transformers.models.qwen3_5.modeling_qwen3_5": "3e2b6239e4b2c3e512d4f9836a1dba12e975ae652407e84f2d3c2beebf0c9528",
    },
}
_RESOURCE_BOUNDS = {
    "gpus_used": 1,
    "gpu_model": "NVIDIA RTX A6000",
    "minimum_gpu_memory_gib": 47,
    "allocator_cap_gib": 40,
    "minimum_host_ram_gib": 64,
    "minimum_free_disk_gib": 60,
    "maximum_candidate_bytes": 8 * 1024 * 1024,
    "maximum_receipt_bytes": 512 * 1024 * 1024,
    "maximum_failure_bytes": 16 * 1024 * 1024,
    "maximum_output_directory_bytes": 1024 * 1024 * 1024,
    "outer_wall_seconds": 28800,
    "compute_seconds": 28260,
    "audit_seconds": 240,
    "failure_audit_seconds": 180,
    "terminal_seconds": 60,
    "network": False,
    "output_root": "/home/ubuntu/rlm/outputs/latent-a1-nc0-r1-nomination-v1",
}
_CACHE_CLASS_CLOSURE = [
    {
        "fqcn": fqcn,
        "module_path": module_path,
        "module_sha256": module_sha256,
        "distribution": distribution,
    }
    for fqcn, module_path, module_sha256, distribution in (
        (
            "fla.models.utils.Cache",
            "/home/ubuntu/rlm/prime-rl/.venv/lib/python3.12/site-packages/fla/models/utils.py",
            "3785d027727370b6eb8da96050109a249254c90caa12a3531a4034cc79f256a1",
            "flash-linear-attention==0.5.2",
        ),
        (
            "fla.models.utils.FLACache",
            "/home/ubuntu/rlm/prime-rl/.venv/lib/python3.12/site-packages/fla/models/utils.py",
            "3785d027727370b6eb8da96050109a249254c90caa12a3531a4034cc79f256a1",
            "flash-linear-attention==0.5.2",
        ),
        (
            "fla.models.utils.LegacyFLACache",
            "/home/ubuntu/rlm/prime-rl/.venv/lib/python3.12/site-packages/fla/models/utils.py",
            "3785d027727370b6eb8da96050109a249254c90caa12a3531a4034cc79f256a1",
            "flash-linear-attention==0.5.2",
        ),
        *(
            (
                f"transformers.cache_utils.{name}",
                "/home/ubuntu/rlm/prime-rl/.venv/lib/python3.12/site-packages/transformers/cache_utils.py",
                "a51d2cd525f4458941a20cb5b3b8a8d989d8ca5bcd451855a27bb41743fed586",
                "transformers==5.6.2",
            )
            for name in ("Cache", "DynamicCache", "EncoderDecoderCache", "QuantizedCache", "StaticCache")
        ),
    )
]
_ASSET_PATHS = {
    "experiments/qwen35-2b-latent-workspace-v1/a0-cache-calibration-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0-cache-calibration-rejected-manifest.sha256",
    "experiments/qwen35-2b-latent-workspace-v1/a0-cache-calibration-rejected-receipt.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0-cache-calibration-rejected-run.log",
    "experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0-nocache-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0-nocache-disjointness-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0-nocache-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0c-carrier-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0d-cache-diagnostic-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0dr-cache-diagnostic-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0dr2-cache-diagnostic-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0nc-success-receipt.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0r-mechanism-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-disjointness-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-render-rejection-failure.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-render-rejection-manifest.sha256",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-render-rejection-run.log",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-r1-tokenizer-proof-receipt.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-r1-tokenizer-proof.log",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-held_out-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-schedule-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-train-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-validation-bank-v1.json",
    "scripts/latent/run_a0_nocache_receiver_v1.py",
    "scripts/latent/run_a0_nocache_receiver_v1.sh",
    "scripts/latent/prove_a1_nc0_r1_tokenizer_v1.py",
    "scripts/latent/prove_a1_nc0_r1_tokenizer_v1.sh",
    "scripts/latent/run_a1_nc0_nomination_v1.py",
    "scripts/latent/run_a1_nc0_nomination_v1.sh",
    "src/prime_rl/latent/__init__.py",
    "src/prime_rl/latent/a0.py",
    "src/prime_rl/latent/a0c.py",
    "src/prime_rl/latent/a0cal.py",
    "src/prime_rl/latent/a0d.py",
    "src/prime_rl/latent/a0dr.py",
    "src/prime_rl/latent/a0dr2.py",
    "src/prime_rl/latent/a0nc.py",
    "src/prime_rl/latent/a0r.py",
    "src/prime_rl/latent/a1nc0.py",
    "src/prime_rl/latent/bridge.py",
    "src/prime_rl/latent/policy_adapter.py",
    "src/prime_rl/latent/transfer_bank.py",
    "tests/unit/latent/test_a1nc0.py",
}
_CACHE_GUARD_CONTRACT = {
    "recursive_class_closure": _CACHE_CLASS_CLOSURE,
    "negative_control": "DynamicCache allocation must raise",
    "closure_check_count": "memory_index_of_cache_guard_audit_complete_plus_2",
    "restore_in_finally": True,
}


class ExperimentIncomplete(RuntimeError):
    """The frozen train/evaluation contract did not complete interpretably."""


class CacheAllocationDetected(RuntimeError):
    """A forbidden cache object or past-key-value output was observed."""


class MechanismRejected(RuntimeError):
    """A frozen numerical, gradient, parity, or protection predicate failed."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


def classify_failure(error: BaseException) -> tuple[str, str]:
    if isinstance(error, CacheAllocationDetected):
        return "mechanism_rejected", "cache_allocation_or_past_key_values_detected"
    if isinstance(error, MechanismRejected):
        return "mechanism_rejected", error.reason
    if isinstance(error, ExperimentIncomplete):
        return "mechanism_rejected", "training_or_evaluation_contract_incomplete"
    return "infrastructure_invalid", "environment_provenance_timeout_or_oom"


def validate_bank_artifact(path: Path, split: str) -> dict[str, object]:
    if split not in _SPLITS or path.is_symlink() or not path.is_file():
        raise ValueError("A1-NC0 bank path or split is invalid")
    expected = _SPLITS[split]
    if file_sha256(path) != expected["file_sha256"]:
        raise ValueError(f"A1-NC0 {split} bank file hash changed")
    artifact = json.loads(path.read_text())
    regenerated = build_transfer_bank(
        seed=expected["seed"],
        split=split,  # type: ignore[arg-type]
        examples_per_family=expected["examples_per_family"],
    ).artifact_dict()
    if artifact != regenerated or artifact.get("manifest_sha256") != expected["manifest_sha256"]:
        raise ValueError(f"A1-NC0 {split} bank content changed")
    records = artifact.get("bank", {}).get("records", [])
    if (
        len(records) != expected["records"]
        or sum(len(item.get("queries", [])) for item in records) != expected["queries"]
    ):
        raise ValueError(f"A1-NC0 {split} bank cardinality changed")
    return artifact


def build_training_schedule(train_artifact: dict[str, object]) -> list[dict[str, object]]:
    records = train_artifact["bank"]["records"]
    by_family = {family: [record for record in records if record["family"] == family] for family in FAMILIES}
    schedule: list[dict[str, object]] = []
    update_index = 0
    for epoch in range(1, 5):
        for evidence_index in range(16):
            update_index += 1
            query_ids = [
                query["query_id"] for family in FAMILIES for query in by_family[family][evidence_index]["queries"]
            ]
            schedule.append({"epoch": epoch, "update_index": update_index, "query_ids": query_ids})
    if canonical_json_hash(schedule) != _TRAIN_SCHEDULE_SHA256:
        raise ValueError("A1-NC0 training schedule hash changed")
    return schedule


def build_arm_orders(artifact: dict[str, object], split: str) -> list[dict[str, object]]:
    if split not in {"validation", "held_out"}:
        raise ValueError("A1-NC0 arm orders are evaluation-only")
    orders = []
    for record in artifact["bank"]["records"]:
        for query in record["queries"]:
            query_id = query["query_id"]
            arms = sorted(
                _ARMS,
                key=lambda arm: hashlib.sha256(
                    f"q35-2b-a1-nc0-split-information-v1:arm_order:{split}:{query_id}:{arm}".encode()
                ).hexdigest(),
            )
            orders.append({"query_id": query_id, "arms": arms})
    if canonical_json_hash(orders) != _ARM_ORDER_SHA256[split]:
        raise ValueError(f"A1-NC0 {split} arm-order hash changed")
    return orders


def validate_schedule_artifact(schedule: dict[str, object], *, artifacts: dict[str, dict[str, object]]) -> None:
    required = {
        "schema_version",
        "train_updates",
        "train_schedule_sha256",
        "arm_orders",
        "arm_order_sha256",
        "a0nc_repeat_selection",
        "a0nc_repeat_selection_sha256",
        "memory_ledger_paths",
    }
    if set(schedule) != required or schedule.get("schema_version") != "prime-rl/latent-a1-nc0-schedule/v1":
        raise ValueError("A1-NC0 schedule schema changed")
    expected_train = build_training_schedule(artifacts["train"])
    expected_orders = {split: build_arm_orders(artifacts[split], split) for split in ("validation", "held_out")}
    if (
        schedule.get("train_updates") != expected_train
        or schedule.get("train_schedule_sha256") != _TRAIN_SCHEDULE_SHA256
        or schedule.get("arm_orders") != expected_orders
        or schedule.get("arm_order_sha256") != _ARM_ORDER_SHA256
        or canonical_json_hash(expected_orders) != _ARM_ORDER_SHA256["combined"]
        or schedule.get("a0nc_repeat_selection") != _A0NC_REPEAT_SELECTION
        or schedule.get("a0nc_repeat_selection_sha256") != _A0NC_REPEAT_SELECTION_SHA256
    ):
        raise ValueError("A1-NC0 schedule content changed")
    expected_memory_paths = build_memory_ledger_paths(schedule)
    if schedule.get("memory_ledger_paths") != {
        path: {
            "labels": labels,
            "label_count": len(labels),
            "labels_sha256": canonical_json_hash(labels),
        }
        for path, labels in expected_memory_paths.items()
    }:
        raise ValueError("A1-NC0 memory-ledger label freeze changed")


def build_memory_ledger_paths(schedule: dict[str, object]) -> dict[str, list[str]]:
    """Materialize both prospective terminal paths from the frozen operation schedule."""
    labels: list[str] = ["model_loaded_frozen"]
    e33 = 0
    bridge = 0

    def forward(arm: str) -> None:
        nonlocal e33
        e33 += 1
        labels.append(f"e33_forward_{e33:04d}_{arm}")

    def bridge_forward(arm: str) -> None:
        nonlocal bridge
        bridge += 1
        labels.append(f"bridge_forward_{bridge:04d}_{arm}")

    for _record in range(64):
        forward("TRAIN_PARENT_FEATURE")
    labels.extend(["train_feature_cache_host_complete", "bridge_initialized"])
    for probe in range(4):
        forward("A0NC_REPEAT_CAPTURE")
        if probe == 0:
            forward("A0NC_REPEAT_CAPTURE_KEEP0_CONTROL")
        forward("A0NC_REPEAT_CAPTURE_REPEAT")
        bridge_forward("A0NC_REPEAT_BRIDGE")
        bridge_forward("A0NC_REPEAT_BRIDGE_REPEAT")
        for _step in range(4):
            for arm in ("L_ID", "L_E", "S", "S_REPEAT"):
                forward(f"A0NC_REPEAT_{arm}")
        forward("A0NC_REPEAT_GRADIENT")
        if probe == 0:
            forward("A0NC_REPEAT_GRADIENT_FULL_LOGITS_CONTROL")
    labels.append("a0nc_repeat_gradient_complete")
    for update in schedule["train_updates"]:
        for row in range(1, 13):
            bridge_forward("TRAIN_MCUR")
            forward("TRAIN_MCUR")
            labels.append(f"train_update_{update['update_index']:02d}_row_{row:02d}_post_backward")
        labels.append(f"train_update_{update['update_index']:02d}_clip")
        labels.append(f"train_update_{update['update_index']:02d}_optimizer_step")
        if update["update_index"] % 16 == 0:
            labels.append(f"train_epoch_{update['epoch']}_complete")
    labels.append("optimizer_destroyed_before_evaluation")

    def evaluation(split: str) -> None:
        for _record in range(16):
            forward(f"{split}_MCUR_CANONICAL_SETUP_FEATURE")
            bridge_forward(f"{split}_MCUR_CANONICAL_SETUP_BRIDGE")
        for row in schedule["arm_orders"][split]:
            for arm in row["arms"]:
                if arm in {"MCUR", "MOTH", "MSELF"}:
                    forward(f"{split}_{arm}_FEATURE")
                    bridge_forward(f"{split}_{arm}_BRIDGE")
                for _step in range(12):
                    forward(f"{split}_{arm}_DECODE")
        labels.append(f"{split}_split_audit_complete")

    evaluation("validation")
    common = list(labels)
    validation_stop = [*common, "held_out_skipped_no_model_exposure"]
    labels.append("validation_evaluation_complete")
    # The completion marker belongs before branch selection on both paths.
    validation_stop.insert(len(common), "validation_evaluation_complete")
    evaluation("held_out")
    labels.append("held_out_evaluation_complete")
    full = labels
    tail = [
        "cache_guard_audit_complete",
        "e33_in_memory_post_hash_complete",
        "protected_disk_postflight_complete",
        "candidate_write_preflight",
        "candidate_write_complete",
        "preterminal_receipt_audit_complete",
    ]
    return {
        "validation_stop": [*validation_stop, *tail],
        "full_evaluation": [*full, *tail],
    }


def build_disjointness_report(artifacts: dict[str, dict[str, object]]) -> dict[str, object]:
    axes: dict[str, dict[str, list[str]]] = {}
    for split in _SPLITS:
        records = artifacts[split]["bank"]["records"]
        axes[split] = {
            "evidence_ids": [record["evidence_id"] for record in records],
            "query_ids": [query["query_id"] for record in records for query in record["queries"]],
            "structured_evidence_sha256": [record["structured_evidence_sha256"] for record in records],
            "parent_evidence_text": [record["parent_evidence"] for record in records],
            "child_query_text": [query["child_query"] for record in records for query in record["queries"]],
            "opaque_identifiers": sorted(
                {
                    identifier
                    for record in records
                    for text in [record["parent_evidence"], *(query["child_query"] for query in record["queries"])]
                    for identifier in _OPAQUE_IDENTIFIER.findall(text)
                }
            ),
        }
    intersections: dict[str, dict[str, object]] = {}
    split_names = list(_SPLITS)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            key = f"{left}__{right}"
            intersections[key] = {}
            for axis in axes[left]:
                overlap = sorted(set(axes[left][axis]) & set(axes[right][axis]))
                intersections[key][axis] = {"count": len(overlap), "values": overlap}
    report = {
        "schema_version": "prime-rl/latent-a1-nc0-disjointness/v1",
        "opaque_identifier_regex": _OPAQUE_IDENTIFIER.pattern,
        "split_axes": axes,
        "pairwise_intersections": intersections,
        "all_pairwise_intersections_zero": all(
            item["count"] == 0 for pair in intersections.values() for item in pair.values()
        ),
        "report_sha256": "",
    }
    report["report_sha256"] = canonical_json_hash(report, omitted_fields=("report_sha256",))
    return report


def validate_disjointness_report(report: dict[str, object], *, artifacts: dict[str, dict[str, object]]) -> None:
    expected = build_disjointness_report(artifacts)
    if report != expected or report.get("all_pairwise_intersections_zero") is not True:
        raise ValueError("A1-NC0 bank disjointness report changed")


def fixed_feature_inputs(
    input_ids: torch.Tensor, *, pad_token_id: int, budget: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ExperimentIncomplete("A1-NC0 feature input must be one nonempty token sequence")
    if input_ids.shape[1] > budget:
        raise ExperimentIncomplete("A1-NC0 feature input would require forbidden truncation")
    padding = budget - input_ids.shape[1]
    padded = torch.nn.functional.pad(input_ids, (padding, 0), value=pad_token_id)
    attention_mask = torch.nn.functional.pad(torch.ones_like(input_ids), (padding, 0), value=0)
    if (
        padded.shape != (1, budget)
        or attention_mask.sum().item() != input_ids.shape[1]
        or not torch.equal(padded[:, padding:], input_ids)
        or torch.count_nonzero(attention_mask[:, :padding]).item() != 0
        or not torch.all(attention_mask[:, padding:] == 1).item()
    ):
        raise ExperimentIncomplete("A1-NC0 fixed feature geometry changed")
    return padded, attention_mask


def noise_seed(split: str, evidence_id: str) -> int:
    payload = f"q35-2b-a1-nc0-split-information-v1:noise|2671655313|{split}|{evidence_id}"
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


def tensor_bytes_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def module_state_tree_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        entry = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": tensor_bytes_sha256(tensor),
        }
        digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NormMatchedNoise:
    tensor: torch.Tensor
    evidence: dict[str, object]


def norm_matched_noise(target: torch.Tensor, *, split: str, evidence_id: str) -> NormMatchedNoise:
    if target.shape != (8, 2048) or target.dtype != torch.bfloat16 or target.device.type != "cpu":
        raise ExperimentIncomplete("A1-NC0 NOISE target must be detached CPU BF16 [8,2048]")
    target_f32 = target.detach().float()
    if not torch.isfinite(target_f32).all():
        raise MechanismRejected("A1-NC0 NOISE target is nonfinite", reason="noise_numeric_contract_rejected")
    payload = f"q35-2b-a1-nc0-split-information-v1:noise|2671655313|{split}|{evidence_id}"
    payload_sha256 = hashlib.sha256(payload.encode()).hexdigest()
    seed = noise_seed(split, evidence_id)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    raw = torch.randn((8, 2048), dtype=torch.float32, device="cpu", generator=generator)
    raw_norms = torch.linalg.vector_norm(raw, dim=1, keepdim=True)
    target_norms = torch.linalg.vector_norm(target_f32, dim=1, keepdim=True)
    if not torch.isfinite(raw_norms).all() or not torch.isfinite(target_norms).all():
        raise MechanismRejected("A1-NC0 NOISE norm is nonfinite", reason="noise_numeric_contract_rejected")
    positive = target_norms.squeeze(1) > 0
    if torch.any(positive & (raw_norms.squeeze(1) == 0)):
        raise MechanismRejected(
            "A1-NC0 NOISE positive target has zero raw norm", reason="noise_numeric_contract_rejected"
        )
    scaled = torch.zeros_like(raw)
    scaled[positive] = raw[positive] * (target_norms[positive] / raw_norms[positive])
    precast_norms = torch.linalg.vector_norm(scaled, dim=1, keepdim=True)
    if torch.count_nonzero(precast_norms[~positive]).item() != 0 or not torch.allclose(
        precast_norms[positive],
        target_norms[positive],
        rtol=_NOISE_NORM_RELATIVE_TOLERANCE,
        atol=0.0,
    ):
        raise MechanismRejected("A1-NC0 NOISE float32 norm matching changed", reason="noise_numeric_contract_rejected")
    noise = scaled.to(dtype=torch.bfloat16).contiguous()
    if not torch.isfinite(noise.float()).all():
        raise MechanismRejected(
            "A1-NC0 NOISE BF16 cast produced nonfinite values", reason="noise_numeric_contract_rejected"
        )
    evidence = {
        "payload": payload,
        "payload_sha256": payload_sha256,
        "seed": seed,
        "target_bfloat16_sha256": tensor_bytes_sha256(target),
        "raw_float32_sha256": tensor_bytes_sha256(raw),
        "scaled_float32_sha256": tensor_bytes_sha256(scaled),
        "final_bfloat16_sha256": tensor_bytes_sha256(noise),
        "target_slot_norms_float32": target_norms.squeeze(1).tolist(),
        "precast_slot_norms_float32": precast_norms.squeeze(1).tolist(),
        "postcast_slot_norms_float32": torch.linalg.vector_norm(noise.float(), dim=1).tolist(),
        "zero_target_rows": torch.nonzero(~positive, as_tuple=False).flatten().tolist(),
        "precast_target_norm_relative_tolerance": _NOISE_NORM_RELATIVE_TOLERANCE,
    }
    validate_finite_metrics(evidence)
    return NormMatchedNoise(tensor=noise, evidence=evidence)


def validate_a1nc0_r1_evidence(repo: Path, plan: dict[str, object]) -> dict[str, object]:
    """Bind the clean failed start and the tokenizer-only repair proof before model load."""
    experiment = repo / "experiments/qwen35-2b-latent-workspace-v1"
    failure_path = experiment / "a1-nc0-render-rejection-failure.json"
    log_path = experiment / "a1-nc0-render-rejection-run.log"
    manifest_path = experiment / "a1-nc0-render-rejection-manifest.sha256"
    proof_path = experiment / "a1-nc0-r1-tokenizer-proof-receipt.json"
    proof_log_path = experiment / "a1-nc0-r1-tokenizer-proof.log"
    paths = (failure_path, log_path, manifest_path, proof_path, proof_log_path)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("A1-NC0-R1 prospective evidence is absent or symlinked")

    failure = json.loads(failure_path.read_text())
    manifest = manifest_path.read_text()
    prior_observed = {
        "status": failure.get("status"),
        "failure_category": failure.get("failure_category"),
        "error_type": failure.get("error_type"),
        "error": failure.get("error"),
        "stage": failure.get("stage"),
        "failure_file_sha256": file_sha256(failure_path),
        "failure_internal_sha256": failure.get("failure_sha256"),
        "launch_log_sha256": file_sha256(log_path),
        "snapshot_manifest_sha256": file_sha256(manifest_path),
        "execution_commit": failure.get("execution_commit"),
        "mechanism_code_commit": failure.get("mechanism_code_commit"),
        "plan_sha256": failure.get("plan_sha256"),
        "cuda_runtime_contacted": True,
        "model_or_material_allocation_attempted": failure.get("e33_parameter_tree_sha256_before") is not None,
        "base_model_update_attempted": failure.get("base_model_update_attempted"),
        "bridge_update_attempted": failure.get("bridge_update_attempted"),
        "optimizer_created": False,
        "checkpoint_created": failure.get("checkpoint_created"),
        "candidate_inventory": failure.get("candidate_inventory"),
        "candidate_valid": failure.get("candidate_valid"),
        "memory_ledger_partial": failure.get("memory_ledger_partial"),
        "cache_guard_partial": failure.get("cache_guard_partial"),
        "protected_hashes_exact": failure.get("protected_hash_probe_after_failure")
        == {"coordinator_e33": _E33, "worker_h176": _H176}
        and failure.get("protected_hashes_before") == {"coordinator_e33": _E33, "worker_h176": _H176},
        "frozen_assets_exact": failure.get("frozen_asset_hashes_match_plan"),
        "run_id_reusable": failure.get("run_id_reusable"),
    }
    if (
        prior_observed != _PRIOR_RENDER_REJECTION
        or plan.get("prior_render_rejection") != _PRIOR_RENDER_REJECTION
        or manifest
        != "9e41fb8107af77ae5258c10d9caff7f13773aaa26162850faab7c085c6440d80  failure.json\n"
        "5d4f519dba35d5e8307fce54b1d979bfa937a7d636e4abb29fdddef14b7f0fc3  run.log\n"
        or failure.get("e33_parameter_tree_sha256_before") is not None
        or failure.get("e33_parameter_tree_sha256_failure_audit") is not None
        or failure.get("e33_gradients_absent_failure_audit") is not None
        or failure.get("checkpoint_metadata_before") != failure.get("protected_metadata_probe_after_failure")
        or failure.get("a0nc_success_evidence_matches_plan") is not True
    ):
        raise ValueError("A1-NC0-R1 failed-start evidence changed")

    proof = json.loads(proof_path.read_text())
    preflight = proof.get("preflight")
    proof_observed = {
        "status": proof.get("status"),
        "receipt_file_sha256": file_sha256(proof_path),
        "receipt_internal_sha256": proof.get("receipt_sha256"),
        "proof_log_sha256": file_sha256(proof_log_path),
        "mechanism_commit": proof.get("mechanism_commit"),
        "versions": proof.get("versions"),
        "transformers_runtime_version": proof.get("transformers_runtime_version"),
        "tokenizer_class": proof.get("tokenizer_class"),
        "tokenizer_asset_sha256": proof.get("tokenizer_asset_sha256"),
        "bank_file_sha256": proof.get("bank_file_sha256"),
        "materialized_queries": preflight.get("materialized_queries") if isinstance(preflight, dict) else None,
        "batch_encoding_extraction_counts": preflight.get("batch_encoding_extraction_counts")
        if isinstance(preflight, dict)
        else None,
        "maximum_unpadded_feature_tokens": preflight.get("maximum_unpadded_feature_tokens")
        if isinstance(preflight, dict)
        else None,
        "render_hashes_sha256": preflight.get("render_hashes_sha256") if isinstance(preflight, dict) else None,
        "label_alignment_sha256": preflight.get("label_alignment_sha256")
        if isinstance(preflight, dict)
        else None,
        "repeat_preflight_bitwise_canonical_equal": proof.get("repeat_preflight_bitwise_canonical_equal"),
        "cuda_visible_devices": proof.get("cuda_visible_devices"),
        "torch_cuda_initialized_before": proof.get("torch_cuda_initialized_before"),
        "torch_cuda_initialized_after": proof.get("torch_cuda_initialized_after"),
        "model_from_pretrained_calls": proof.get("model_from_pretrained_calls"),
        "model_loaded": proof.get("model_loaded"),
        "optimizer_created": proof.get("optimizer_created"),
        "model_update_attempted": proof.get("model_update_attempted"),
    }
    expected_proof = plan.get("tokenizer_only_proof")
    if (
        not isinstance(expected_proof, dict)
        or proof_observed != expected_proof
        or proof.get("schema_version") != "prime-rl/latent-a1-nc0-r1-tokenizer-proof/v1"
        or proof.get("receipt_sha256") != canonical_json_hash(proof, omitted_fields=("receipt_sha256",))
        or proof.get("proof_log", {}).get("sha256") != proof_observed["proof_log_sha256"]
        or proof.get("repeat_render_hashes_sha256") != proof_observed["render_hashes_sha256"]
        or proof.get("repeat_label_alignment_sha256") != proof_observed["label_alignment_sha256"]
        or not _valid_sha(proof_observed["render_hashes_sha256"])
        or not _valid_sha(proof_observed["label_alignment_sha256"])
        or proof_observed["materialized_queries"] != 288
        or proof_observed["batch_encoding_extraction_counts"]
        != {"parent": 96, "child_plain": 288, "child_opening": 288, "child_full": 288, "mself_parent": 288}
        or proof_observed["torch_cuda_initialized_before"] is not False
        or proof_observed["torch_cuda_initialized_after"] is not False
        or proof_observed["model_from_pretrained_calls"] != 0
        or proof_observed["model_loaded"] is not False
        or proof_observed["optimizer_created"] is not False
        or proof_observed["model_update_attempted"] is not False
    ):
        raise ValueError("A1-NC0-R1 tokenizer-only proof changed")
    return {"prior_render_rejection": prior_observed, "tokenizer_only_proof": proof_observed}


def validate_plan(plan: dict[str, object], *, bank_file_hashes: dict[str, str]) -> None:
    required = {
        "schema_version",
        "status",
        "execution_authorization",
        "mechanism_code_commit",
        "evidence_commit",
        "asset_sha256",
        "plan_sha256",
        "a0nc_success_evidence",
        "prior_render_rejection",
        "tokenizer_only_proof",
        "protected_checkpoints",
        "remote_paths",
        "runtime",
        "cache_guard_contract",
        "split_information_bank",
        "bank_disjointness",
        "seeds",
        "bridge",
        "training",
        "evaluation",
        "validation_gate",
        "nomination_gate",
        "resource_bounds",
        "failure_classification",
        "interpretation_boundary",
    }
    if set(plan) != required or plan.get("schema_version") != A1NC0_PLAN_SCHEMA:
        raise ValueError("A1-NC0 plan schema changed")
    if (
        plan.get("status") != "preregistered"
        or plan.get("execution_authorization") != "root_and_evaluator_review_required"
    ):
        raise ValueError("A1-NC0 freeze or authorization changed")
    if (
        not isinstance(plan.get("mechanism_code_commit"), str)
        or not _COMMIT.fullmatch(plan["mechanism_code_commit"])
        or not isinstance(plan.get("evidence_commit"), str)
        or not _COMMIT.fullmatch(plan["evidence_commit"])
        or plan["evidence_commit"] == plan["mechanism_code_commit"]
    ):
        raise ValueError("A1-NC0 mechanism/evidence commit malformed")
    assets = plan.get("asset_sha256")
    if (
        not isinstance(assets, dict)
        or not assets
        or any(
            not isinstance(path, str) or not isinstance(digest, str) or not _SHA.fullmatch(digest)
            for path, digest in assets.items()
        )
    ):
        raise ValueError("A1-NC0 asset closure malformed")
    if set(assets) != _ASSET_PATHS:
        raise ValueError("A1-NC0 executable/evidence asset closure changed")
    if plan.get("plan_sha256") != canonical_json_hash(plan, omitted_fields=("plan_sha256",)):
        raise ValueError("A1-NC0 canonical plan hash changed")
    if plan.get("protected_checkpoints") != {"coordinator_e33": _E33, "worker_h176": _H176}:
        raise ValueError("A1-NC0 protected checkpoints changed")
    if plan.get("a0nc_success_evidence") != _A0NC_SUCCESS:
        raise ValueError("A1-NC0 A0NC dependency evidence changed")
    proof = plan.get("tokenizer_only_proof")
    if plan.get("prior_render_rejection") != _PRIOR_RENDER_REJECTION or not isinstance(proof, dict):
        raise ValueError("A1-NC0-R1 prospective evidence declaration changed")
    if (
        set(proof)
        != {
            "status",
            "receipt_file_sha256",
            "receipt_internal_sha256",
            "proof_log_sha256",
            "mechanism_commit",
            "versions",
            "transformers_runtime_version",
            "tokenizer_class",
            "tokenizer_asset_sha256",
            "bank_file_sha256",
            "materialized_queries",
            "batch_encoding_extraction_counts",
            "maximum_unpadded_feature_tokens",
            "render_hashes_sha256",
            "label_alignment_sha256",
            "repeat_preflight_bitwise_canonical_equal",
            "cuda_visible_devices",
            "torch_cuda_initialized_before",
            "torch_cuda_initialized_after",
            "model_from_pretrained_calls",
            "model_loaded",
            "optimizer_created",
            "model_update_attempted",
        }
        or proof.get("status") != "tokenizer_render_mechanism_validated"
        or proof.get("mechanism_commit") != plan["mechanism_code_commit"]
        or any(
            not _valid_sha(proof.get(key))
            for key in (
                "receipt_file_sha256",
                "receipt_internal_sha256",
                "proof_log_sha256",
                "render_hashes_sha256",
                "label_alignment_sha256",
            )
        )
    ):
        raise ValueError("A1-NC0-R1 tokenizer-proof declaration malformed")
    if plan.get("remote_paths") != {
        "coordinator_e33": "/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2",
        "worker_h176": "/home/ubuntu/rlm/outputs/q35-2b-document-child-sft-v1/h176child8-document-child-real12-step8-v2/weights/step_8",
    }:
        raise ValueError("A1-NC0 protected remote paths changed")
    banks = plan.get("split_information_bank")
    if not isinstance(banks, dict) or set(banks) != set(_SPLITS):
        raise ValueError("A1-NC0 bank declaration changed")
    for split, expected in _SPLITS.items():
        if banks.get(split) != expected or bank_file_hashes.get(split) != expected["file_sha256"]:
            raise ValueError(f"A1-NC0 {split} bank freeze changed")
    if plan.get("bank_disjointness") != _DISJOINTNESS:
        raise ValueError("A1-NC0 disjointness freeze changed")
    if plan.get("seeds") != _SEEDS or plan.get("training") != _TRAINING or plan.get("evaluation") != _EVALUATION:
        raise ValueError("A1-NC0 seeds, training, or evaluation contract changed")
    if plan.get("bridge") != _BRIDGE or plan.get("runtime") != _RUNTIME:
        raise ValueError("A1-NC0 bridge or runtime contract changed")
    if plan.get("cache_guard_contract") != _CACHE_GUARD_CONTRACT:
        raise ValueError("A1-NC0 cache guard contract changed")
    if plan.get("resource_bounds") != _RESOURCE_BOUNDS:
        raise ValueError("A1-NC0 resource bounds changed")
    if (
        _RESOURCE_BOUNDS["compute_seconds"]
        + _RESOURCE_BOUNDS["audit_seconds"]
        + _RESOURCE_BOUNDS["failure_audit_seconds"]
        + _RESOURCE_BOUNDS["terminal_seconds"]
        > _RESOURCE_BOUNDS["outer_wall_seconds"]
    ):
        raise ValueError("A1-NC0 failure-publication headroom changed")
    if plan.get("validation_gate") != _VALIDATION_GATE or plan.get("nomination_gate") != _NOMINATION_GATE:
        raise ValueError("A1-NC0 prospective gates changed")
    if (
        plan.get("failure_classification") != _FAILURE_CLASSIFICATION
        or plan.get("interpretation_boundary") != _INTERPRETATION
    ):
        raise ValueError("A1-NC0 terminal interpretation changed")


def load_plan(
    plan_path: Path,
    bank_paths: dict[str, Path],
    schedule_path: Path,
    disjointness_path: Path,
) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, object], dict[str, object]]:
    paths = [plan_path, schedule_path, disjointness_path, *bank_paths.values()]
    if (
        plan_path.is_symlink()
        or not plan_path.is_file()
        or any(path.is_symlink() or not path.is_file() for path in paths)
    ):
        raise ValueError("A1-NC0 plan/bank asset absent or symlinked")
    artifacts = {split: validate_bank_artifact(bank_paths[split], split) for split in _SPLITS}
    schedule = json.loads(schedule_path.read_text())
    validate_schedule_artifact(schedule, artifacts=artifacts)
    disjointness = json.loads(disjointness_path.read_text())
    validate_disjointness_report(disjointness, artifacts=artifacts)
    plan = json.loads(plan_path.read_text())
    validate_plan(plan, bank_file_hashes={split: file_sha256(path) for split, path in bank_paths.items()})
    if plan["asset_sha256"].get(str(schedule_path.relative_to(plan_path.parents[2]))) != file_sha256(schedule_path):
        raise ValueError("A1-NC0 schedule asset hash changed")
    if plan["asset_sha256"].get(str(disjointness_path.relative_to(plan_path.parents[2]))) != file_sha256(
        disjointness_path
    ):
        raise ValueError("A1-NC0 disjointness asset hash changed")
    return plan, artifacts, schedule, disjointness


def summarize_arm_results(
    rows: list[dict[str, object]], *, expected_queries: list[tuple[str, str]]
) -> dict[str, object]:
    expected_ids = [query_id for query_id, _family in expected_queries]
    if (
        len(rows) != 48
        or len(expected_queries) != 48
        or [row.get("query_id") for row in rows] != expected_ids
        or [row.get("family") for row in rows] != [family for _query_id, family in expected_queries]
        or len(set(expected_ids)) != 48
        or Counter(family for _query_id, family in expected_queries) != Counter({family: 12 for family in FAMILIES})
    ):
        raise ExperimentIncomplete("A1-NC0 split row identity/order/cardinality changed")
    for row in rows:
        arms = row.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(_ARMS):
            raise ExperimentIncomplete("A1-NC0 row arm set changed")
        for result in arms.values():
            if (
                not isinstance(result, dict)
                or set(result)
                != {
                    "exact_match",
                    "generated_text",
                    "expected_answer_sha256",
                    "answer_token_nll",
                    "answer_token_count",
                    "generated_text_sha256",
                    "generated_token_ids",
                    "fixed_decode_steps",
                }
                or not isinstance(result["exact_match"], bool)
                or not isinstance(result["generated_text"], str)
                or not _valid_sha(result["expected_answer_sha256"])
                or hashlib.sha256(result["generated_text"].encode()).hexdigest() != result["generated_text_sha256"]
                or result["exact_match"] is not (result["generated_text_sha256"] == result["expected_answer_sha256"])
                or isinstance(result["answer_token_count"], bool)
                or not isinstance(result["answer_token_count"], int)
                or result["answer_token_count"] < 1
                or result["answer_token_count"] > 12
                or not isinstance(result["answer_token_nll"], (int, float))
                or not math.isfinite(float(result["answer_token_nll"]))
                or float(result["answer_token_nll"]) < 0
                or not isinstance(result["generated_text_sha256"], str)
                or not _SHA.fullmatch(result["generated_text_sha256"])
                or not isinstance(result["generated_token_ids"], list)
                or any(isinstance(token, bool) or not isinstance(token, int) for token in result["generated_token_ids"])
                or len(result["generated_token_ids"]) != 12
                or result["fixed_decode_steps"] != 12
            ):
                raise ExperimentIncomplete("A1-NC0 arm result contract changed")
    arm_exact = {arm: sum(row["arms"][arm]["exact_match"] is True for row in rows) for arm in _ARMS}
    arm_mean_nll = {arm: math.fsum(float(row["arms"][arm]["answer_token_nll"]) for row in rows) / 48 for arm in _ARMS}

    def comparison(left: str, right: str) -> dict[str, object]:
        wins = [row for row in rows if row["arms"][left]["exact_match"] and not row["arms"][right]["exact_match"]]
        losses = [row for row in rows if row["arms"][right]["exact_match"] and not row["arms"][left]["exact_match"]]
        paired_nll_wins = sum(
            float(row["arms"][right]["answer_token_nll"]) - float(row["arms"][left]["answer_token_nll"]) > 1e-6
            for row in rows
        )
        return {
            "exact_wins": len(wins),
            "exact_losses": len(losses),
            "exact_net": len(wins) - len(losses),
            "win_families": sorted({row["family"] for row in wins}),
            "mean_answer_token_nll_improvement": arm_mean_nll[right] - arm_mean_nll[left],
            "paired_nll_wins": paired_nll_wins,
        }

    recoveries = [row for row in rows if row["arms"]["MCUR"]["exact_match"] and not row["arms"]["M0"]["exact_match"]]
    regressions = [row for row in rows if row["arms"]["M0"]["exact_match"] and not row["arms"]["MCUR"]["exact_match"]]
    family_exact = Counter(row["family"] for row in rows if row["arms"]["MCUR"]["exact_match"])
    moth_recoveries = sum(row["arms"]["MOTH"]["exact_match"] for row in recoveries)
    noise_recoveries = sum(row["arms"]["NOISE"]["exact_match"] for row in recoveries)
    contamination_floor = math.floor(len(recoveries) * _NOMINATION_GATE["moth_recovery_fraction_of_mcur_maximum"])

    def architecture_contrast(left: str, right: str) -> dict[str, object]:
        per_family = {}
        for family in FAMILIES:
            family_rows = [row for row in rows if row["family"] == family]
            exact_utility = math.fsum(
                float(row["arms"][left]["exact_match"]) - float(row["arms"][right]["exact_match"])
                for row in family_rows
            )
            nll_utility = (
                math.fsum(
                    float(row["arms"][right]["answer_token_nll"]) - float(row["arms"][left]["answer_token_nll"])
                    for row in family_rows
                )
                / 12
            )
            per_family[family] = {
                "query_count": 12,
                "exact_utility_count": int(exact_utility),
                "exact_utility_rate": exact_utility / 12,
                "mean_positive_favors_left_nll_utility": nll_utility,
            }
        exact_utility = math.fsum(
            float(row["arms"][left]["exact_match"]) - float(row["arms"][right]["exact_match"]) for row in rows
        )
        return {
            "left": left,
            "right": right,
            "query_count": 48,
            "exact_utility_count": int(exact_utility),
            "exact_utility_rate": exact_utility / 48,
            "mean_positive_favors_left_nll_utility": math.fsum(
                float(row["arms"][right]["answer_token_nll"]) - float(row["arms"][left]["answer_token_nll"])
                for row in rows
            )
            / 48,
            "per_family": per_family,
        }

    return {
        "complete_tasks": len(rows),
        "arm_exact": arm_exact,
        "arm_mean_answer_token_nll": arm_mean_nll,
        "m0_to_mcur_recoveries": len(recoveries),
        "m0_to_mcur_regressions": len(regressions),
        "m0_to_mcur_recovery_families": sorted({row["family"] for row in recoveries}),
        "mcur_exact_by_family": {family: family_exact[family] for family in FAMILIES},
        "mcur_vs_moth": comparison("MCUR", "MOTH"),
        "mcur_vs_mself": comparison("MCUR", "MSELF"),
        "mcur_vs_zero": comparison("MCUR", "ZERO"),
        "mcur_vs_noise": comparison("MCUR", "NOISE"),
        "recovery_contamination": {
            "allowed_floor_count": contamination_floor,
            "moth_observed_count": moth_recoveries,
            "noise_observed_count": noise_recoveries,
        },
        "architecture_contrasts": {
            "OPE": architecture_contrast("MCUR", "M0"),
            "OME": architecture_contrast("MOTH", "M0"),
            "CAG": architecture_contrast("MCUR", "MOTH"),
            "SSG": architecture_contrast("MCUR", "MSELF"),
            "DSC": architecture_contrast("MSELF", "MOTH"),
        },
    }


def validation_gate_passes(summary: dict[str, object]) -> bool:
    gate = _VALIDATION_GATE
    comparison = summary["mcur_vs_moth"]
    return bool(
        summary["complete_tasks"] == gate["complete_tasks"]
        and summary["arm_exact"]["MCUR"] >= gate["mcur_exact_minimum"]
        and comparison["exact_net"] >= gate["mcur_minus_moth_exact_net_minimum"]
        and summary["m0_to_mcur_recoveries"] >= gate["m0_to_mcur_recoveries_minimum"]
        and summary["m0_to_mcur_regressions"] <= gate["m0_to_mcur_regressions_maximum"]
        and len(summary["m0_to_mcur_recovery_families"]) >= gate["m0_to_mcur_recovery_families_minimum"]
        and comparison["mean_answer_token_nll_improvement"] >= gate["mcur_minus_moth_mean_answer_token_nll_minimum"]
        and comparison["paired_nll_wins"] >= gate["mcur_vs_moth_paired_nll_wins_minimum"]
    )


def nomination_gate_passes(summary: dict[str, object]) -> bool:
    gate = _NOMINATION_GATE
    mcur_moth = summary["mcur_vs_moth"]
    mcur_self = summary["mcur_vs_mself"]
    mcur_zero = summary["mcur_vs_zero"]
    mcur_noise = summary["mcur_vs_noise"]
    recoveries = summary["m0_to_mcur_recoveries"]
    contamination = summary["recovery_contamination"]
    ratios_ok = recoveries > 0 and (
        contamination["moth_observed_count"] <= contamination["allowed_floor_count"]
        and contamination["noise_observed_count"] <= contamination["allowed_floor_count"]
    )
    return bool(
        summary["complete_tasks"] == gate["complete_tasks"]
        and summary["arm_exact"]["MCUR"] >= gate["mcur_exact_minimum"]
        and recoveries >= gate["m0_to_mcur_recoveries_minimum"]
        and summary["m0_to_mcur_regressions"] <= gate["m0_to_mcur_regressions_maximum"]
        and len(summary["m0_to_mcur_recovery_families"]) >= gate["m0_to_mcur_recovery_families_minimum"]
        and mcur_moth["exact_wins"] >= gate["mcur_vs_moth_exact_wins_minimum"]
        and mcur_moth["exact_losses"] <= gate["mcur_vs_moth_exact_losses_maximum"]
        and mcur_moth["exact_net"] >= gate["mcur_minus_moth_exact_net_minimum"]
        and len(mcur_moth["win_families"]) >= gate["mcur_vs_moth_win_families_minimum"]
        and mcur_moth["mean_answer_token_nll_improvement"] >= gate["mcur_minus_moth_mean_answer_token_nll_minimum"]
        and mcur_moth["paired_nll_wins"] >= gate["mcur_vs_moth_paired_nll_wins_minimum"]
        and mcur_self["exact_wins"] >= gate["mcur_vs_mself_exact_wins_minimum"]
        and mcur_self["exact_losses"] <= gate["mcur_vs_mself_exact_losses_maximum"]
        and mcur_self["exact_net"] >= gate["mcur_minus_mself_exact_net_minimum"]
        and mcur_self["mean_answer_token_nll_improvement"] >= gate["mcur_minus_mself_mean_answer_token_nll_minimum"]
        and mcur_self["paired_nll_wins"] >= gate["mcur_vs_mself_paired_nll_wins_minimum"]
        and mcur_zero["exact_net"] >= gate["mcur_minus_zero_exact_net_minimum"]
        and mcur_zero["mean_answer_token_nll_improvement"] >= gate["mcur_minus_zero_mean_answer_token_nll_minimum"]
        and mcur_noise["exact_net"] >= gate["mcur_minus_noise_exact_net_minimum"]
        and mcur_noise["mean_answer_token_nll_improvement"] >= gate["mcur_minus_noise_mean_answer_token_nll_minimum"]
        and all(value >= gate["mcur_exact_per_family_minimum"] for value in summary["mcur_exact_by_family"].values())
        and ratios_ok
    )


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _require_capture_evidence(capture: object, *, keep0: bool | None = None) -> None:
    if not isinstance(capture, dict):
        raise ExperimentIncomplete("A1-NC0 capture evidence is absent")
    indices = capture.get("captured_token_indices")
    visible = capture.get("captured_visible_tokens")
    control = capture.get("keep0_control")
    if (
        not isinstance(visible, int)
        or isinstance(visible, bool)
        or not 1 <= visible <= 128
        or capture.get("padded_tokens") != 256
        or capture.get("tokens_truncated") != 0
        or not isinstance(capture.get("unpadded_tokens"), int)
        or capture["unpadded_tokens"] < visible
        or not isinstance(indices, list)
        or len(indices) != 128
        or indices != [-1] * (128 - visible) + list(range(256 - visible, 256))
        or capture.get("captured_zero_left_padding") is not True
        or capture.get("captured_suffix_matches_final_hidden_bitwise") is not True
        or any(
            not _valid_sha(capture.get(key))
            for key in (
                "input_ids_sha256",
                "attention_mask_sha256",
                "captured_hidden_sha256",
                "full_final_hidden_sha256",
                "keep1_logits_sha256",
                "captured_mask_sha256",
                "capture_spec_sha256",
            )
        )
        or not isinstance(capture.get("gpu_seconds"), (int, float))
        or isinstance(capture.get("gpu_seconds"), bool)
        or not math.isfinite(float(capture["gpu_seconds"]))
        or float(capture["gpu_seconds"]) < 0
    ):
        raise ExperimentIncomplete("A1-NC0 capture geometry or content evidence changed")
    if keep0 is True:
        if (
            not isinstance(control, dict)
            or control.get("full_hidden_bitwise_equal") is not True
            or control.get("selected_hidden_bitwise_equal") is not True
            or control.get("last_logits_bitwise_equal") is not True
            or control.get("keep0_last_logits_sha256") != control.get("keep1_logits_sha256")
            or control.get("keep1_logits_sha256") != capture.get("keep1_logits_sha256")
            or any(
                not _valid_sha(control.get(key))
                for key in ("keep0_full_hidden_sha256", "keep0_last_logits_sha256", "keep1_logits_sha256")
            )
        ):
            raise ExperimentIncomplete("A1-NC0 keep0 capture control changed")
    elif keep0 is False and control is not None:
        raise ExperimentIncomplete("A1-NC0 unexpected keep0 capture control appeared")


def _require_bridge_evidence(evidence: object) -> None:
    if (
        not isinstance(evidence, dict)
        or evidence.get("receiver_gate_applied_exactly_once") is not True
        or any(
            not _valid_sha(evidence.get(key))
            for key in (
                "encoder_workspace_float32_sha256",
                "receiver_precast_float32_sha256",
                "receiver_final_bfloat16_sha256",
            )
        )
        or not isinstance(evidence.get("gpu_seconds"), (int, float))
        or isinstance(evidence.get("gpu_seconds"), bool)
        or not math.isfinite(float(evidence["gpu_seconds"]))
        or float(evidence["gpu_seconds"]) < 0
    ):
        raise ExperimentIncomplete("A1-NC0 bridge evidence changed")


def _require_workspace_evidence(evidence: object) -> None:
    if not isinstance(evidence, dict):
        raise ExperimentIncomplete("A1-NC0 workspace witness is absent")
    _require_capture_evidence(evidence.get("feature"), keep0=False)
    _require_bridge_evidence(evidence.get("bridge"))
    for key in ("feature_bridge_cuda_event_seconds", "feature_bridge_wall_seconds"):
        value = evidence.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ExperimentIncomplete("A1-NC0 workspace timing evidence changed")


def _workspace_fingerprint(evidence: dict[str, object]) -> tuple[object, ...]:
    return (
        evidence["feature"]["captured_hidden_sha256"],
        evidence["feature"]["captured_mask_sha256"],
        tuple(evidence["feature"]["captured_token_indices"]),
        evidence["feature"]["full_final_hidden_sha256"],
        evidence["bridge"]["encoder_workspace_float32_sha256"],
        evidence["bridge"]["receiver_precast_float32_sha256"],
        evidence["bridge"]["receiver_final_bfloat16_sha256"],
    )


def _validate_render_receipt(render: object, *, plan: dict[str, object]) -> None:
    if not isinstance(render, dict):
        raise ExperimentIncomplete("A1-NC0 render preflight is absent")
    alignments = render.get("label_alignment")
    if (
        render.get("enable_thinking") is not False
        or render.get("tools_none_for_child") is not True
        or render.get("parent_fixture_messages") != 4
        or render.get("child_base_messages") != 2
        or render.get("terminal_token_ids") != [248046, 198]
        or render.get("fixed_continuation_token_ids") != [49265, 48338, 3438, 321]
        or render.get("length_control_token_ids") != [40, 4021, 2528, 8976, 35139, 635, 524, 599]
        or render.get("length_control_tokens_non_special") is not True
        or render.get("tokenizer_eos_token_id") != 248046
        or render.get("tokenizer_pad_token_id") != 248046
        or render.get("feature_sequences_truncated") != 0
        or render.get("materialized_queries") != 288
        or render.get("tokenized_template_container")
        != "transformers.tokenization_utils_base.BatchEncoding"
        or render.get("preflight_input_ids_extracted_from_batch_encoding") is not True
        or render.get("batch_encoding_extraction_counts")
        != {
            "parent": 96,
            "child_plain": 288,
            "child_opening": 288,
            "child_full": 288,
            "mself_parent": 288,
        }
        or render.get("answer_key_interpolation_scope") != "teacher_target_and_scoring_only"
        or render.get("answer_key_not_interpolated_into_parent_or_child_opening") is not True
        or not isinstance(render.get("maximum_unpadded_feature_tokens"), int)
        or not 1 <= render["maximum_unpadded_feature_tokens"] <= 256
        or not _valid_sha(render.get("render_hashes_sha256"))
        or not isinstance(alignments, dict)
        or len(alignments) != 288
        or render.get("label_alignment_sha256") != canonical_json_hash(alignments)
        or plan.get("runtime", {}).get("tokenizer_eos_token_id") != 248046
        or plan.get("runtime", {}).get("tokenizer_pad_token_id") != 248046
    ):
        raise ExperimentIncomplete("A1-NC0 render preflight changed")
    for alignment in alignments.values():
        if not isinstance(alignment, dict):
            raise ExperimentIncomplete("A1-NC0 label alignment is malformed")
        active = alignment.get("active_label_positions")
        logits = alignment.get("active_logit_positions")
        answers = alignment.get("raw_answer_token_ids")
        if (
            not isinstance(active, list)
            or not isinstance(logits, list)
            or not isinstance(answers, list)
            or not 1 <= len(answers) <= 12
            or len(active) != len(answers)
            or active != list(range(active[0], active[0] + len(active)))
            or logits != [position - 1 for position in active]
            or alignment.get("terminal_token_ids") != [248046, 198]
            or alignment.get("all_other_labels_masked") is not True
        ):
            raise ExperimentIncomplete("A1-NC0 causal label alignment changed")


def _validate_cache_guard(receipt: dict[str, object], *, expected_labels: list[str]) -> None:
    guard = receipt.get("cache_guard")
    if not isinstance(guard, dict):
        raise ExperimentIncomplete("A1-NC0 cache guard evidence is absent")
    classes = guard.get("classes")
    # One check enters the guard, every guarded ledger row checks again, and
    # the explicit final closure plus __exit__ contribute two terminal checks.
    expected_checks = expected_labels.index("cache_guard_audit_complete") + 2
    if (
        not isinstance(classes, list)
        or not classes
        or guard.get("negative_control_dynamic_cache_tripped") is not True
        or guard.get("restored_in_finally") is not True
        or guard.get("closure_check_count") != expected_checks
    ):
        raise ExperimentIncomplete("A1-NC0 cache guard closure changed")
    if classes != _CACHE_CLASS_CLOSURE:
        raise ExperimentIncomplete("A1-NC0 recursive cache subclass closure changed")


def _validate_suffix_objective(objective: object, *, query_id: str | None = None) -> None:
    if not isinstance(objective, dict) or (query_id is not None and objective.get("query_id") != query_id):
        raise ExperimentIncomplete("A1-NC0 suffix objective identity changed")
    active = objective.get("active_label_count")
    if (
        not isinstance(active, int)
        or isinstance(active, bool)
        or not 1 <= active <= 12
        or objective.get("logits_to_keep") != active + 1
        or objective.get("logit_suffix_start") != objective.get("first_active_label_index") - 1
        or objective.get("active_causal_pairs_unchanged") is not True
        or objective.get("terminal_ids_excluded_from_teacher_input") is not True
        or not _valid_sha(objective.get("active_causal_pairs_sha256"))
    ):
        raise ExperimentIncomplete("A1-NC0 suffix objective changed")


_GRADIENT_GROUPS = {
    "source_norm",
    "source_projection",
    "learned_queries",
    "resampler_in_proj",
    "resampler_out_proj",
    "output_norm",
    "decoder_workspace_norm",
    "decoder_projection",
    "receiver_gate",
}


def _validate_gradient_groups(groups: object, *, require_all_positive: bool) -> None:
    if not isinstance(groups, dict) or set(groups) != _GRADIENT_GROUPS:
        raise ExperimentIncomplete("A1-NC0 gradient group set changed")
    for value in groups.values():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
            or (require_all_positive and float(value) <= 0)
        ):
            raise ExperimentIncomplete("A1-NC0 gradient group evidence changed")


def _validate_pretraining_repeat(repeat: object, *, schedule: dict[str, object]) -> None:
    if not isinstance(repeat, dict):
        raise ExperimentIncomplete("A1-NC0 pretraining repeat is absent")
    expected_histogram = {
        "A0NC_REPEAT_CAPTURE": 4,
        "A0NC_REPEAT_CAPTURE_REPEAT": 4,
        "A0NC_REPEAT_L_ID": 16,
        "A0NC_REPEAT_L_E": 16,
        "A0NC_REPEAT_S": 16,
        "A0NC_REPEAT_S_REPEAT": 16,
        "A0NC_REPEAT_GRADIENT": 4,
        "A0NC_REPEAT_CAPTURE_KEEP0_CONTROL": 1,
        "A0NC_REPEAT_GRADIENT_FULL_LOGITS_CONTROL": 1,
    }
    if (
        repeat.get("selection") != schedule.get("a0nc_repeat_selection")
        or repeat.get("selection_sha256") != _A0NC_REPEAT_SELECTION_SHA256
        or repeat.get("fixed_continuation_text") != " Acknowledged and continuing safely."
        or repeat.get("fixed_continuation_token_ids") != [49265, 48338, 3438, 321]
        or repeat.get("length_control_token_ids") != [40, 4021, 2528, 8976, 35139, 635, 524, 599]
        or repeat.get("e33_forward_count") != 78
        or repeat.get("e33_call_histogram") != expected_histogram
        or repeat.get("bridge_forward_count") != 8
        or repeat.get("bridge_call_histogram") != {"A0NC_REPEAT_BRIDGE": 4, "A0NC_REPEAT_BRIDGE_REPEAT": 4}
        or repeat.get("base_model_gradients_absent") is not True
        or repeat.get("optimizer_step") is not False
        or repeat.get("bridge_parameter_sha256_before") != repeat.get("bridge_parameter_sha256_after")
        or not _valid_sha(repeat.get("bridge_parameter_sha256_before"))
    ):
        raise ExperimentIncomplete("A1-NC0 pretraining repeat aggregate changed")
    _validate_gradient_groups(repeat.get("gradient_group_l2"), require_all_positive=True)
    probes = repeat.get("probes")
    if not isinstance(probes, list) or len(probes) != 4:
        raise ExperimentIncomplete("A1-NC0 pretraining probe count changed")
    for index, (probe, selected) in enumerate(zip(probes, _A0NC_REPEAT_SELECTION, strict=True)):
        if (
            not isinstance(probe, dict)
            or {key: probe.get(key) for key in ("family", "evidence_id", "query_id")} != selected
            or probe.get("capture_repeat_bitwise") is not True
            or probe.get("bridge_repeat_bitwise") is not True
            or probe.get("soft_span_active") is not True
            or probe.get("soft_span_differs_from_hard") is not True
            or probe.get("outside_soft_span_exact") is not True
            or not isinstance(probe.get("loss"), (int, float))
            or isinstance(probe.get("loss"), bool)
            or not math.isfinite(float(probe["loss"]))
            or len(probe.get("capture", [])) != 2
            or len(probe.get("bridge", [])) != 2
            or len(probe.get("steps", [])) != 4
        ):
            raise ExperimentIncomplete("A1-NC0 pretraining probe evidence changed")
        _require_capture_evidence(probe["capture"][0], keep0=index == 0)
        _require_capture_evidence(probe["capture"][1], keep0=False)
        _require_bridge_evidence(probe["bridge"][0])
        _require_bridge_evidence(probe["bridge"][1])
        if any(
            probe["capture"][0][key] != probe["capture"][1][key]
            for key in (
                "captured_hidden_sha256",
                "captured_mask_sha256",
                "captured_token_indices",
                "full_final_hidden_sha256",
            )
        ) or any(
            probe["bridge"][0][key] != probe["bridge"][1][key]
            for key in (
                "encoder_workspace_float32_sha256",
                "receiver_precast_float32_sha256",
                "receiver_final_bfloat16_sha256",
            )
        ):
            raise ExperimentIncomplete("A1-NC0 repeated capture/bridge hashes changed")
        for step_index, step in enumerate(probe["steps"], start=1):
            required_hashes = (
                "l_id_input_ids_sha256",
                "l_e_inputs_embeds_sha256",
                "shared_soft_inputs_embeds_sha256",
                "attention_mask_sha256",
                "position_ids_sha256",
                "l_id_logits_sha256",
                "l_e_logits_sha256",
                "soft_logits_sha256",
                "soft_repeat_logits_sha256",
            )
            if (
                not isinstance(step, dict)
                or step.get("step") != step_index
                or step.get("continuation_token_id") != [49265, 48338, 3438, 321][step_index - 1]
                or step.get("l_id_l_e_bitwise_equal") is not True
                or step.get("soft_repeat_bitwise_equal") is not True
                or step.get("soft_same_tensor_object_for_repeat") is not True
                or step.get("soft_input_unchanged_after_forwards") is not True
                or step.get("l_id_logits_sha256") != step.get("l_e_logits_sha256")
                or step.get("soft_logits_sha256") != step.get("soft_repeat_logits_sha256")
                or any(not _valid_sha(step.get(key)) for key in required_hashes)
            ):
                raise ExperimentIncomplete("A1-NC0 pretraining step parity changed")
        _validate_suffix_objective(probe.get("suffix_objective"))
        if probe.get("answer_token_count") != probe["suffix_objective"]["active_label_count"]:
            raise ExperimentIncomplete("A1-NC0 pretraining supervised answer count changed")
        control = probe["suffix_objective"].get("full_logits_control")
        if index == 0:
            if (
                not isinstance(control, dict)
                or control.get("last_k_logits_bitwise_equal") is not True
                or control.get("loss_bitwise_equal") is not True
                or control.get("full_loss_sha256") != control.get("suffix_loss_sha256")
                or any(
                    not _valid_sha(control.get(key))
                    for key in (
                        "full_logits_sha256",
                        "suffix_logits_sha256",
                        "full_loss_sha256",
                        "suffix_loss_sha256",
                    )
                )
            ):
                raise ExperimentIncomplete("A1-NC0 full-logit objective control changed")
        elif control is not None:
            raise ExperimentIncomplete("A1-NC0 unexpected objective control appeared")


def _validate_training_receipt(receipt: dict[str, object], *, schedule: dict[str, object]) -> None:
    updates = receipt.get("training_updates")
    if not isinstance(updates, list) or len(updates) != 64:
        raise ExperimentIncomplete("A1-NC0 training update count changed")
    for observed, expected in zip(updates, schedule["train_updates"], strict=True):
        if (
            not isinstance(observed, dict)
            or observed.get("epoch") != expected["epoch"]
            or observed.get("update_index") != expected["update_index"]
            or observed.get("query_ids_sha256") != canonical_json_hash(expected["query_ids"])
            or observed.get("query_exposures") != 12
            or observed.get("base_model_gradients_absent_after_each_row") is not True
            or observed.get("receiver_gate_gradient_finite_nonzero") is not True
            or not isinstance(observed.get("mean_loss"), (int, float))
            or isinstance(observed.get("mean_loss"), bool)
            or not math.isfinite(float(observed["mean_loss"]))
            or not isinstance(observed.get("preclip_gradient_l2"), (int, float))
            or isinstance(observed.get("preclip_gradient_l2"), bool)
            or not math.isfinite(float(observed["preclip_gradient_l2"]))
            or float(observed["preclip_gradient_l2"]) < 0
            or not _valid_sha(observed.get("bridge_parameter_sha256_after"))
        ):
            raise ExperimentIncomplete("A1-NC0 training update identity changed")
        _validate_gradient_groups(observed.get("gradient_group_l2"), require_all_positive=False)
        objectives = observed.get("suffix_objectives")
        if not isinstance(objectives, list) or len(objectives) != 12:
            raise ExperimentIncomplete("A1-NC0 training suffix objective count changed")
        for objective, query_id in zip(objectives, expected["query_ids"], strict=True):
            _validate_suffix_objective(objective, query_id=query_id)
        workspace = observed.get("within_update_evidence_workspace_sha256")
        evidence_ids = {query_id.rsplit("-q", 1)[0] for query_id in expected["query_ids"]}
        if (
            not isinstance(workspace, dict)
            or set(workspace) != evidence_ids
            or any(not _valid_sha(value) for value in workspace.values())
        ):
            raise ExperimentIncomplete("A1-NC0 within-update workspace reuse changed")
    feature_cache = receipt.get("train_feature_cache")
    expected_ids = {
        query_id.rsplit("-q", 1)[0] for update in schedule["train_updates"][:16] for query_id in update["query_ids"]
    }
    if (
        not isinstance(feature_cache, list)
        or len(feature_cache) != 64
        or {item.get("evidence_id") for item in feature_cache if isinstance(item, dict)} != expected_ids
    ):
        raise ExperimentIncomplete("A1-NC0 train feature cache identity changed")
    for item in feature_cache:
        if (
            not isinstance(item, dict)
            or item.get("device") != "cpu"
            or item.get("detached") is not True
            or not _valid_sha(item.get("host_hidden_sha256"))
            or not _valid_sha(item.get("host_mask_sha256"))
        ):
            raise ExperimentIncomplete("A1-NC0 host feature cache evidence changed")
        _require_capture_evidence(item, keep0=False)
    invariants = receipt.get("training_invariants")
    after = invariants.get("feature_cache_after") if isinstance(invariants, dict) else None
    epochs = invariants.get("epoch_positive_gradient_groups") if isinstance(invariants, dict) else None
    if (
        not isinstance(after, dict)
        or set(after) != expected_ids
        or invariants.get("all_64_host_features_unchanged_after_768_exposures") is not True
        or epochs != {str(epoch): sorted(_GRADIENT_GROUPS) for epoch in range(1, 5)}
    ):
        raise ExperimentIncomplete("A1-NC0 training invariant summary changed")
    before_by_id = {item["evidence_id"]: item for item in feature_cache}
    for evidence_id, item in after.items():
        if (
            not isinstance(item, dict)
            or item.get("device") != "cpu"
            or item.get("hidden_sha256") != before_by_id[evidence_id]["host_hidden_sha256"]
            or item.get("mask_sha256") != before_by_id[evidence_id]["host_mask_sha256"]
        ):
            raise ExperimentIncomplete("A1-NC0 host feature cache mutation evidence changed")


def _validate_noise_evidence(noise: object, *, split: str, evidence_id: str, expected_target_sha256: str) -> None:
    if not isinstance(noise, dict):
        raise ExperimentIncomplete("A1-NC0 NOISE evidence is absent")
    payload = f"q35-2b-a1-nc0-split-information-v1:noise|2671655313|{split}|{evidence_id}"
    norm_keys = (
        "target_slot_norms_float32",
        "precast_slot_norms_float32",
        "postcast_slot_norms_float32",
    )
    if (
        noise.get("payload") != payload
        or noise.get("payload_sha256") != hashlib.sha256(payload.encode()).hexdigest()
        or noise.get("seed") != noise_seed(split, evidence_id)
        or noise.get("target_bfloat16_sha256") != expected_target_sha256
        or any(
            not _valid_sha(noise.get(key))
            for key in (
                "target_bfloat16_sha256",
                "raw_float32_sha256",
                "scaled_float32_sha256",
                "final_bfloat16_sha256",
            )
        )
        or any(not isinstance(noise.get(key), list) or len(noise[key]) != 8 for key in norm_keys)
        or not isinstance(noise.get("zero_target_rows"), list)
        or any(
            not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 8
            for index in noise["zero_target_rows"]
        )
        or noise.get("precast_target_norm_relative_tolerance") != _NOISE_NORM_RELATIVE_TOLERANCE
    ):
        raise ExperimentIncomplete("A1-NC0 NOISE provenance changed")
    zero_rows = set(noise["zero_target_rows"])
    target_norms = noise["target_slot_norms_float32"]
    precast_norms = noise["precast_slot_norms_float32"]
    for index, (target_norm, precast_norm) in enumerate(zip(target_norms, precast_norms, strict=True)):
        if index in zero_rows:
            if target_norm != 0.0 or precast_norm != 0.0:
                raise ExperimentIncomplete("A1-NC0 NOISE zero-row norm changed")
        elif (
            not isinstance(target_norm, (int, float))
            or isinstance(target_norm, bool)
            or float(target_norm) <= 0
            or abs(float(precast_norm) - float(target_norm)) > _NOISE_NORM_RELATIVE_TOLERANCE * float(target_norm)
        ):
            raise ExperimentIncomplete("A1-NC0 NOISE float32 norm match changed")
    validate_finite_metrics(noise)


def _validate_decode_evidence(
    evidence: object,
    *,
    generated_token_ids: list[int],
    expected_gold_token_ids: list[int],
) -> float:
    if not isinstance(evidence, list) or len(evidence) != 12:
        raise ExperimentIncomplete("A1-NC0 decode operation count changed")
    eos_seen = False
    initial_prefix = None
    nlls = []
    for index, (step, generated) in enumerate(zip(evidence, generated_token_ids, strict=True), start=1):
        if not isinstance(step, dict):
            raise ExperimentIncomplete("A1-NC0 decode evidence is malformed")
        prefix_length = step.get("prefix_length")
        if initial_prefix is None:
            initial_prefix = prefix_length
        gold_id = step.get("gold_token_id")
        gold_nll = step.get("gold_token_nll")
        forced = step.get("forced_after_eos")
        argmax = step.get("argmax_token_id")
        if (
            step.get("step") != index
            or not isinstance(prefix_length, int)
            or isinstance(prefix_length, bool)
            or prefix_length != initial_prefix + index - 1
            or step.get("appended_token_id") != generated
            or forced is not eos_seen
            or (eos_seen and (argmax is not None or generated != 248046))
            or (not eos_seen and (not isinstance(argmax, int) or isinstance(argmax, bool) or generated != argmax))
            or step.get("terminal_selected") is not (argmax == 248046 if argmax is not None else False)
            or any(
                not _valid_sha(step.get(key))
                for key in ("prefix_sha256", "attention_mask_sha256", "position_ids_sha256", "logits_sha256")
            )
        ):
            raise ExperimentIncomplete("A1-NC0 decode recurrence evidence changed")
        if index <= len(expected_gold_token_ids):
            if (
                gold_id != expected_gold_token_ids[index - 1]
                or not isinstance(gold_nll, (int, float))
                or isinstance(gold_nll, bool)
                or not math.isfinite(float(gold_nll))
                or float(gold_nll) < 0
            ):
                raise ExperimentIncomplete("A1-NC0 rollout-conditioned gold NLL changed")
            nlls.append(float(gold_nll))
        elif gold_id is not None or gold_nll is not None:
            raise ExperimentIncomplete("A1-NC0 gold NLL extended beyond answer span")
        eos_seen = eos_seen or generated == 248046
    if len(nlls) != len(expected_gold_token_ids):
        raise ExperimentIncomplete("A1-NC0 answer-token NLL operation count changed")
    return math.fsum(nlls) / len(nlls)


def _validate_evaluation_split(
    evaluation: object,
    *,
    split: str,
    schedule: dict[str, object],
    artifact: dict[str, object],
    label_alignment: dict[str, object],
    tokenizer,
) -> None:
    if not isinstance(evaluation, dict):
        raise ExperimentIncomplete(f"A1-NC0 {split} evaluation is absent")
    rows = evaluation.get("rows")
    setup = evaluation.get("setup")
    summary = evaluation.get("summary")
    if not isinstance(rows, list) or len(rows) != 48 or not isinstance(setup, list) or len(setup) != 16:
        raise ExperimentIncomplete(f"A1-NC0 {split} evaluation cardinality changed")
    setup_by_id = {}
    for item in setup:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("evidence_id"), str)
            or not isinstance(item.get("moth_donor_evidence_id"), str)
            or not _valid_sha(item.get("moth_canonical_source_sha256"))
        ):
            raise ExperimentIncomplete(f"A1-NC0 {split} canonical workspace setup changed")
        evidence_id = item["evidence_id"]
        if evidence_id in setup_by_id:
            raise ExperimentIncomplete(f"A1-NC0 {split} duplicate workspace setup appeared")
        _require_workspace_evidence(item.get("mcur"))
        _validate_noise_evidence(
            item.get("noise"),
            split=split,
            evidence_id=evidence_id,
            expected_target_sha256=item["mcur"]["bridge"]["receiver_final_bfloat16_sha256"],
        )
        if item["noise"]["final_bfloat16_sha256"] == item["noise"]["target_bfloat16_sha256"]:
            raise ExperimentIncomplete(f"A1-NC0 {split} NOISE silently equals MCUR")
        setup_by_id[evidence_id] = item
    expected_orders = schedule["arm_orders"][split]
    records = {record["evidence_id"]: record for record in artifact["bank"]["records"]}
    queries = {
        query["query_id"]: (record, query) for record in artifact["bank"]["records"] for query in record["queries"]
    }
    if list(setup_by_id) != list(records):
        raise ExperimentIncomplete(f"A1-NC0 {split} setup does not match frozen bank")
    for evidence_id, item in setup_by_id.items():
        donor_id = artifact["moth_donors"].get(evidence_id)
        if (
            item["moth_donor_evidence_id"] != donor_id
            or donor_id not in setup_by_id
            or item["moth_canonical_source_sha256"]
            != setup_by_id[donor_id]["mcur"]["bridge"]["receiver_final_bfloat16_sha256"]
        ):
            raise ExperimentIncomplete(f"A1-NC0 {split} MOTH donor binding changed")
    if [row.get("query_id") for row in rows] != [item["query_id"] for item in expected_orders]:
        raise ExperimentIncomplete(f"A1-NC0 {split} row order changed")
    evidence_query_counts: Counter[str] = Counter()
    for row, expected in zip(rows, expected_orders, strict=True):
        if (
            not isinstance(row, dict)
            or row.get("arm_order") != expected["arms"]
            or not isinstance(row.get("evidence_id"), str)
            or row["evidence_id"] not in setup_by_id
            or expected["query_id"] not in queries
            or row["evidence_id"] != queries[expected["query_id"]][0]["evidence_id"]
            or row.get("family") != queries[expected["query_id"]][0]["family"]
            or set(row.get("costs", {})) != set(_ARMS)
            or set(row.get("decode_evidence", {})) != set(_ARMS)
        ):
            raise ExperimentIncomplete(f"A1-NC0 {split} row operation contract changed")
        evidence_query_counts[row["evidence_id"]] += 1
        for arm in _ARMS:
            result = row["arms"][arm]
            cost = row["costs"][arm]
            feature_bridge = cost.get("feature_bridge") if isinstance(cost, dict) else None
            expected_feature = 1 if arm in {"MCUR", "MOTH", "MSELF"} else 0
            if (
                not isinstance(cost, dict)
                or cost.get("feature_forwards") != expected_feature
                or cost.get("bridge_forwards") != expected_feature
                or cost.get("receiver_forwards") != 12
                or any(
                    not isinstance(cost.get(key), (int, float))
                    or isinstance(cost.get(key), bool)
                    or not math.isfinite(float(cost[key]))
                    or float(cost[key]) < 0
                    for key in ("cuda_event_gpu_seconds", "wall_seconds", "receiver_cuda_event_seconds")
                )
                or (expected_feature == 0 and feature_bridge is not None)
            ):
                raise ExperimentIncomplete(f"A1-NC0 {split} arm compute receipt changed")
            if expected_feature:
                _require_workspace_evidence(feature_bridge)
            if arm == "MCUR" and _workspace_fingerprint(feature_bridge) != _workspace_fingerprint(
                setup_by_id[row["evidence_id"]]["mcur"]
            ):
                raise ExperimentIncomplete(f"A1-NC0 {split} MCUR canonical witness changed")
            if arm == "MOTH":
                donor_id = setup_by_id[row["evidence_id"]]["moth_donor_evidence_id"]
                if donor_id not in setup_by_id or _workspace_fingerprint(feature_bridge) != _workspace_fingerprint(
                    setup_by_id[donor_id]["mcur"]
                ):
                    raise ExperimentIncomplete(f"A1-NC0 {split} MOTH donor witness changed")
            expected_answer = queries[expected["query_id"]][1]["answer"]
            expected_answer_sha = hashlib.sha256(expected_answer.encode()).hexdigest()
            alignment = label_alignment.get(f"{split}:{expected['query_id']}")
            expected_gold_ids = tokenizer(expected_answer, add_special_tokens=False).input_ids
            if (
                result.get("expected_answer_sha256") != expected_answer_sha
                or result.get("exact_match") is not (result.get("generated_text") == expected_answer)
                or not isinstance(alignment, dict)
                or alignment.get("raw_answer_token_ids") != expected_gold_ids
            ):
                raise ExperimentIncomplete(f"A1-NC0 {split} exact-answer binding changed")
            recomputed_nll = _validate_decode_evidence(
                row["decode_evidence"][arm],
                generated_token_ids=result["generated_token_ids"],
                expected_gold_token_ids=alignment["raw_answer_token_ids"],
            )
            if (
                result["answer_token_count"] != len(alignment["raw_answer_token_ids"])
                or result["answer_token_nll"] != recomputed_nll
            ):
                raise ExperimentIncomplete(f"A1-NC0 {split} answer-token NLL summary changed")
    if set(evidence_query_counts.values()) != {3} or set(evidence_query_counts) != set(setup_by_id):
        raise ExperimentIncomplete(f"A1-NC0 {split} evidence/query grouping changed")
    if not isinstance(summary, dict):
        raise ExperimentIncomplete(f"A1-NC0 {split} summary is absent")
    match = summary.get("mself_compute_match")
    if (
        not isinstance(match, dict)
        or match.get("mcur_feature_forwards") != 48
        or match.get("mself_feature_forwards") != 48
        or match.get("mcur_bridge_forwards") != 48
        or match.get("mself_bridge_forwards") != 48
        or match.get("mcur_feature_input_tokens") != 48 * 256
        or match.get("mself_feature_input_tokens") != 48 * 256
        or match.get("receiver_forwards_each") != 48 * 12
    ):
        raise ExperimentIncomplete(f"A1-NC0 {split} MSELF compute matching changed")
    mcur = match.get("mcur_cuda_event_gpu_seconds")
    mself = match.get("mself_cuda_event_gpu_seconds")
    if (
        not isinstance(mcur, (int, float))
        or isinstance(mcur, bool)
        or not isinstance(mself, (int, float))
        or isinstance(mself, bool)
        or not math.isfinite(float(mcur))
        or not math.isfinite(float(mself))
        or float(mcur) < 0
        or float(mself) < 0
        or not math.isfinite(float(mcur) + float(mself))
        or float(mcur) + float(mself) <= 0
    ):
        raise ExperimentIncomplete(f"A1-NC0 {split} MSELF timing evidence changed")
    ratio = 2 * abs(float(mcur) - float(mself)) / (float(mcur) + float(mself))
    if match.get("relative_gpu_seconds_difference") != ratio or ratio > 0.10:
        raise ExperimentIncomplete(f"A1-NC0 {split} MSELF timing gate changed")
    reuse = summary.get("workspace_reuse")
    if not isinstance(reuse, dict) or set(reuse) != set(setup_by_id):
        raise ExperimentIncomplete(f"A1-NC0 {split} workspace reuse identities changed")
    for evidence_id, arms in reuse.items():
        if not isinstance(arms, dict) or set(arms) != {"MCUR", "MOTH", "NOISE"}:
            raise ExperimentIncomplete(f"A1-NC0 {split} workspace reuse arms changed")
        for arm, hashes in arms.items():
            if not isinstance(hashes, list) or len(hashes) != 3 or len(set(hashes)) != 1 or not _valid_sha(hashes[0]):
                raise ExperimentIncomplete(f"A1-NC0 {split} workspace reuse bytes changed")
        if arms["MCUR"][0] != setup_by_id[evidence_id]["mcur"]["bridge"]["receiver_final_bfloat16_sha256"]:
            raise ExperimentIncomplete(f"A1-NC0 {split} MCUR reuse hash changed")
        donor_id = setup_by_id[evidence_id]["moth_donor_evidence_id"]
        if arms["MOTH"][0] != setup_by_id[donor_id]["mcur"]["bridge"]["receiver_final_bfloat16_sha256"]:
            raise ExperimentIncomplete(f"A1-NC0 {split} MOTH reuse hash changed")
        if arms["NOISE"][0] != setup_by_id[evidence_id]["noise"]["final_bfloat16_sha256"]:
            raise ExperimentIncomplete(f"A1-NC0 {split} NOISE reuse hash changed")
    counts = summary.get("operation_counts")
    expected_histogram = {
        f"{split}_MCUR_CANONICAL_SETUP_FEATURE": 16,
        f"{split}_MCUR_FEATURE": 48,
        f"{split}_MOTH_FEATURE": 48,
        f"{split}_MSELF_FEATURE": 48,
        **{f"{split}_{arm}_DECODE": 576 for arm in _ARMS},
    }
    if counts != {
        "captures": 160,
        "bridges": 160,
        "receiver_forwards": 3456,
        "canonical_captures_and_bridges": 16,
        "mcur_query_captures_and_bridges": 48,
        "moth_query_captures_and_bridges": 48,
        "mself_query_captures_and_bridges": 48,
        "e33_call_histogram": expected_histogram,
    }:
        raise ExperimentIncomplete(f"A1-NC0 {split} operation histogram changed")


def _validate_receipt(
    receipt: dict[str, object],
    *,
    plan: dict[str, object],
    schedule: dict[str, object],
    artifacts: dict[str, dict[str, object]],
    tokenizer,
    candidate_path: Path,
) -> None:
    allowed_statuses = {
        "valid_not_nominated_validation",
        "valid_not_nominated",
        "a1_nc0_nominated",
    }
    if (
        receipt.get("schema_version") != A1NC0_RECEIPT_SCHEMA
        or receipt.get("status") not in allowed_statuses
        or receipt.get("plan_sha256") != plan.get("plan_sha256")
        or receipt.get("mechanism_code_commit") != plan.get("mechanism_code_commit")
        or receipt.get("evidence_commit") != plan.get("evidence_commit")
        or not isinstance(receipt.get("execution_commit"), str)
        or _COMMIT.fullmatch(receipt["execution_commit"]) is None
        or receipt.get("execution_commit") == receipt.get("mechanism_code_commit")
        or receipt.get("execution_commit_is_exact_child_of_evidence") is not True
        or receipt.get("asset_sha256") != plan.get("asset_sha256")
        or receipt.get("protected_hashes_before") != plan.get("protected_checkpoints")
        or receipt.get("protected_hashes_after") != plan.get("protected_checkpoints")
        or receipt.get("interpretation_boundary") != _INTERPRETATION
        or receipt.get("a1_admission") is not False
        or receipt.get("live_harness_authorized") is not False
        or receipt.get("a2_authorized") is not False
        or receipt.get("model_promotion_authorized") is not False
        or receipt.get("worker_h176_loaded") is not False
        or receipt.get("live_trajectory_count") != 0
        or receipt.get("a_plus_b_combined") is not False
        or receipt.get("resume_used") is not False
        or receipt.get("candidate_valid") is not True
        or receipt.get("candidate_valid_only_with_this_exact_terminal_receipt") is not True
        or receipt.get("claim") != "A1-NC0 nomination-only no-cache bridge learnability"
        or receipt.get("bound_a0nc_dependency_valid_for_B_only") is not True
        or receipt.get("a0nc_success_evidence") != plan.get("a0nc_success_evidence")
        or receipt.get("a1nc0_r1_evidence")
        != {
            "prior_render_rejection": plan.get("prior_render_rejection"),
            "tokenizer_only_proof": plan.get("tokenizer_only_proof"),
        }
        or receipt.get("bank_disjointness")
        != {
            "file_sha256": plan.get("bank_disjointness", {}).get("file_sha256"),
            "report_sha256": plan.get("bank_disjointness", {}).get("report_sha256"),
            "all_pairwise_intersections_zero": True,
        }
        or plan.get("cache_guard_contract") != _CACHE_GUARD_CONTRACT
    ):
        raise ExperimentIncomplete("A1-NC0 receipt identity or boundary changed")
    static_guard = receipt.get("static_no_generation_guard")
    if (
        not isinstance(static_guard, dict)
        or static_guard.get("runner_sha256")
        != plan.get("asset_sha256", {}).get("scripts/latent/run_a1_nc0_nomination_v1.py")
        or static_guard.get("forbidden_calls") != []
        or static_guard.get("generate_used") is not False
        or static_guard.get("prepare_inputs_for_generation_used") is not False
        or static_guard.get("torch_manual_seed_call_count") != 1
        or static_guard.get("torch_cuda_manual_seed_all_call_count") != 1
        or static_guard.get("compose_receiver_inputs_gate_values") != [1.0]
        or static_guard.get("receiver_gate_applied_by_bridge_then_compose_gate_one") is not True
    ):
        raise ExperimentIncomplete("A1-NC0 static generation/gate/seed guard changed")
    expected_runtime = plan.get("runtime")
    versions = receipt.get("versions")
    sources = receipt.get("runtime_sources")
    if (
        not isinstance(expected_runtime, dict)
        or versions
        != {
            "python": expected_runtime.get("python"),
            "transformers": expected_runtime.get("transformers"),
            "flash_linear_attention": expected_runtime.get("flash_linear_attention"),
            "torch_distribution": expected_runtime.get("torch_distribution"),
            "torch_runtime": expected_runtime.get("torch_runtime"),
        }
        or not isinstance(sources, dict)
        or set(sources) != set(expected_runtime.get("transformers_source_sha256", {}))
    ):
        raise ExperimentIncomplete("A1-NC0 runtime identity changed")
    for name, expected_sha in expected_runtime["transformers_source_sha256"].items():
        source = sources[name]
        if (
            not isinstance(source, dict)
            or set(source) != {"path", "sha256"}
            or not isinstance(source["path"], str)
            or source["sha256"] != expected_sha
        ):
            raise ExperimentIncomplete("A1-NC0 runtime source identity changed")
    _validate_render_receipt(receipt.get("render_preflight"), plan=plan)
    expected_metadata = expected_runtime.get("checkpoint_metadata_sha256")
    if (
        receipt.get("checkpoint_metadata_before")
        != {"coordinator_e33": expected_metadata, "worker_h176": expected_metadata}
        or receipt.get("checkpoint_metadata_after") != receipt.get("checkpoint_metadata_before")
        or receipt.get("e33_parameter_tree_sha256_before") != receipt.get("e33_parameter_tree_sha256_after")
        or not _valid_sha(receipt.get("e33_parameter_tree_sha256_before"))
        or receipt.get("e33_tensor_tree_hash_schema") != "sorted_state_dict_name_dtype_shape_tensor_sha256_lines/v1"
        or receipt.get("e33_parameters_require_grad_false") is not True
        or receipt.get("e33_gradients_absent") is not True
        or receipt.get("model_runtime")
        != {
            "class": expected_runtime.get("model_class"),
            "hidden_size": expected_runtime.get("hidden_size"),
            "vocab_size": expected_runtime.get("vocab_size"),
            "dtype": "torch.bfloat16",
            "device": expected_runtime.get("device"),
        }
    ):
        raise ExperimentIncomplete("A1-NC0 protected in-memory or metadata identity changed")
    bridge = receipt.get("bridge")
    if (
        not isinstance(bridge, dict)
        or bridge.get("config")
        != {
            "schema_version": _BRIDGE["schema_version"],
            "source_width": _BRIDGE["source_width"],
            "workspace_width": _BRIDGE["workspace_width"],
            "receiver_width": _BRIDGE["receiver_width"],
            "slots": _BRIDGE["slots"],
            "attention_heads": _BRIDGE["attention_heads"],
            "initial_receiver_gate": _BRIDGE["initial_receiver_gate"],
        }
        or bridge.get("trainable_parameter_count") != _BRIDGE["trainable_parameter_count"]
        or bridge.get("initialization_seed") != _SEEDS["bridge_init"]
        or bridge.get("torch_manual_seed_calls") != 1
        or bridge.get("torch_cuda_manual_seed_all_calls") != 1
        or bridge.get("parameter_tree_hash_schema") != "sorted_state_dict_name_dtype_shape_tensor_sha256_lines/v1"
        or not _valid_sha(bridge.get("parameter_sha256_initial"))
        or not _valid_sha(bridge.get("parameter_sha256_final"))
        or bridge.get("parameter_sha256_initial") == bridge.get("parameter_sha256_final")
        or bridge.get("optimizer_created") is not True
        or bridge.get("optimizer_updates") != 64
        or bridge.get("optimizer_destroyed_before_evaluation") is not True
        or bridge.get("optimizer_state_persisted") is not False
        or bridge.get("base_model_checkpoint_created") is not False
        or not isinstance(bridge.get("candidate"), dict)
        or bridge["candidate"].get("name") != "bridge-candidate.safetensors"
        or bridge["candidate"].get("contains_bridge_and_receiver_gate_only") is not True
        or bridge["candidate"].get("valid_only_with_exact_terminal_receipt") is not True
        or bridge["candidate"].get("promotion_authorized") is not False
        or not isinstance(bridge["candidate"].get("bytes"), int)
        or isinstance(bridge["candidate"].get("bytes"), bool)
        or not 0 < bridge["candidate"]["bytes"] <= plan["resource_bounds"]["maximum_candidate_bytes"]
    ):
        raise ExperimentIncomplete("A1-NC0 bridge/candidate receipt contract changed")
    if candidate_path.is_symlink() or not candidate_path.is_file():
        raise ExperimentIncomplete("A1-NC0 candidate path is absent or symlinked")
    candidate = bridge["candidate"]
    if file_sha256(candidate_path) != candidate.get("sha256") or candidate_path.stat().st_size != candidate.get(
        "bytes"
    ):
        raise ExperimentIncomplete("A1-NC0 candidate bytes changed")
    candidate_digest = hashlib.sha256()
    with safe_open(candidate_path, framework="pt", device="cpu") as handle:
        names = sorted(handle.keys())
        if (
            handle.metadata() != {"schema": "prime-rl/latent-a1-nc0-candidate/v1"}
            or names != _BRIDGE["candidate_tensor_names"]
        ):
            raise ExperimentIncomplete("A1-NC0 candidate schema or tensor names changed")
        for name in names:
            tensor = handle.get_tensor(name)
            if (
                str(tensor.dtype) != _BRIDGE["candidate_tensor_dtype"]
                or list(tensor.shape) != _BRIDGE["candidate_tensor_shapes"][name]
            ):
                raise ExperimentIncomplete("A1-NC0 candidate tensor contract changed")
            if not torch.isfinite(tensor).all():
                raise ExperimentIncomplete("A1-NC0 candidate tensor is nonfinite")
            entry = {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": tensor_bytes_sha256(tensor),
            }
            candidate_digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode())
            candidate_digest.update(b"\n")
    if candidate_digest.hexdigest() != bridge.get("parameter_sha256_final"):
        raise ExperimentIncomplete("A1-NC0 candidate parameter tree hash changed")
    _validate_pretraining_repeat(receipt.get("pretraining_a0nc_repeat"), schedule=schedule)
    _validate_training_receipt(receipt, schedule=schedule)
    validation = receipt.get("validation")
    if not isinstance(validation, dict) or len(validation.get("rows", [])) != 48:
        raise ExperimentIncomplete("A1-NC0 validation receipt is incomplete")
    validation_expected = [
        (
            item["query_id"],
            next(family for family in FAMILIES if item["query_id"].startswith(f"validation-{family}-")),
        )
        for item in schedule["arm_orders"]["validation"]
    ]
    recomputed_validation = summarize_arm_results(validation["rows"], expected_queries=validation_expected)
    for key, value in recomputed_validation.items():
        if validation["summary"].get(key) != value:
            raise ExperimentIncomplete("A1-NC0 validation summary changed")
    validation_passed = validation_gate_passes(validation["summary"])
    _validate_evaluation_split(
        validation,
        split="validation",
        schedule=schedule,
        artifact=artifacts["validation"],
        label_alignment=receipt["render_preflight"]["label_alignment"],
        tokenizer=tokenizer,
    )
    if validation.get("proceed_gate_passed") is not validation_passed:
        raise ExperimentIncomplete("A1-NC0 validation decision changed")
    held = receipt.get("held_out")
    if not validation_passed:
        expected_status = "valid_not_nominated_validation"
        if held is not None:
            raise ExperimentIncomplete("A1-NC0 held-out was exposed after failed validation")
    else:
        if not isinstance(held, dict) or len(held.get("rows", [])) != 48:
            raise ExperimentIncomplete("A1-NC0 held-out receipt is incomplete")
        held_expected = [
            (
                item["query_id"],
                next(family for family in FAMILIES if item["query_id"].startswith(f"held_out-{family}-")),
            )
            for item in schedule["arm_orders"]["held_out"]
        ]
        recomputed_held = summarize_arm_results(held["rows"], expected_queries=held_expected)
        for key, value in recomputed_held.items():
            if held["summary"].get(key) != value:
                raise ExperimentIncomplete("A1-NC0 held-out summary changed")
        nominated = nomination_gate_passes(held["summary"])
        _validate_evaluation_split(
            held,
            split="held_out",
            schedule=schedule,
            artifact=artifacts["held_out"],
            label_alignment=receipt["render_preflight"]["label_alignment"],
            tokenizer=tokenizer,
        )
        if held.get("nomination_gate_passed") is not nominated:
            raise ExperimentIncomplete("A1-NC0 held-out decision changed")
        expected_status = "a1_nc0_nominated" if nominated else "valid_not_nominated"
    if receipt["status"] != expected_status:
        raise ExperimentIncomplete("A1-NC0 terminal scientific status changed")
    expected_calls = 4526 + (3616 if held is not None else 0)
    no_cache = receipt.get("no_cache_call_contract")
    if (
        not isinstance(no_cache, dict)
        or no_cache.get("total_e33_forwards") != expected_calls
        or no_cache.get("expected_e33_forwards") != expected_calls
        or no_cache.get("use_cache_false_every_call") is not True
        or no_cache.get("past_key_values_input_none_every_call") is not True
        or no_cache.get("past_key_values_output_none_every_call") is not True
        or no_cache.get("generate_used") is not False
        or no_cache.get("prepare_inputs_for_generation_used") is not False
        or no_cache.get("rope_deltas_reset_before_every_call") is not True
        or no_cache.get("model_config_use_cache") is not False
        or no_cache.get("generation_config_use_cache") is not False
    ):
        raise ExperimentIncomplete("A1-NC0 no-cache receipt contract changed")
    path = "full_evaluation" if held is not None else "validation_stop"
    expected_labels = schedule["memory_ledger_paths"][path]["labels"]
    rows = receipt.get("memory_ledger")
    memory_keys = ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes")
    if (
        receipt.get("memory_ledger_path") != path
        or not isinstance(rows, list)
        or [row.get("label") for row in rows] != expected_labels
        or receipt.get("memory_ledger_labels_sha256") != canonical_json_hash(expected_labels)
        or any(
            not isinstance(row, dict)
            or set(row) != {"label", *memory_keys}
            or not isinstance(row.get(key), int)
            or isinstance(row.get(key), bool)
            or row[key] < 0
            or row[key] > _RESOURCE_BOUNDS["allocator_cap_gib"] * 2**30
            for row in rows
            for key in memory_keys
        )
    ):
        raise ExperimentIncomplete("A1-NC0 memory receipt contract changed")
    if any(
        row["peak_allocated_bytes"] < row["allocated_bytes"] or row["peak_reserved_bytes"] < row["reserved_bytes"]
        for row in rows
    ) or any(
        current[peak] < previous[peak]
        for previous, current in zip(rows, rows[1:])
        for peak in ("peak_allocated_bytes", "peak_reserved_bytes")
    ):
        raise ExperimentIncomplete("A1-NC0 memory peak accounting changed")
    _validate_cache_guard(receipt, expected_labels=expected_labels)
    resources = receipt.get("resources")
    bounds = plan.get("resource_bounds")
    if (
        not isinstance(resources, dict)
        or not isinstance(bounds, dict)
        or resources.get("gpu_name") != bounds.get("gpu_model")
        or not isinstance(resources.get("total_gpu_memory_bytes"), int)
        or resources["total_gpu_memory_bytes"] < bounds.get("minimum_gpu_memory_gib", 10**9) * 2**30
        or resources.get("allocator_cap_bytes") != bounds.get("allocator_cap_gib", -1) * 2**30
        or resources.get("peak_allocated_bytes", 10**30) > resources["allocator_cap_bytes"]
        or resources.get("peak_reserved_bytes", 10**30) > resources["allocator_cap_bytes"]
        or resources.get("peak_allocated_bytes") != max(row["peak_allocated_bytes"] for row in rows)
        or resources.get("peak_reserved_bytes") != max(row["peak_reserved_bytes"] for row in rows)
        or resources.get("host_ram_bytes", -1) < bounds.get("minimum_host_ram_gib", 10**9) * 2**30
        or resources.get("free_disk_bytes_before", -1) < bounds.get("minimum_free_disk_gib", 10**9) * 2**30
        or resources.get("compute_seconds", float("inf")) > bounds.get("compute_seconds", -1)
        or resources.get("audit_seconds_before_receipt_materialization", float("inf")) > bounds.get("audit_seconds", -1)
        or resources.get("wall_seconds", float("inf")) > bounds.get("outer_wall_seconds", -1)
        or resources.get("visible_cuda_device_count") != 1
        or resources.get("launcher_verified_two_a6000_idle_before_gpu0_exposure") is not True
        or resources.get("physical_gpu1_unused") is not True
        or resources.get("network_used") is not False
    ):
        raise ExperimentIncomplete("A1-NC0 resource receipt changed")
    validate_finite_metrics(receipt)
    if (
        not isinstance(receipt.get("receipt_sha256"), str)
        or not _SHA.fullmatch(receipt["receipt_sha256"])
        or receipt["receipt_sha256"] != canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    ):
        raise ExperimentIncomplete("A1-NC0 receipt self-hash changed")
    return None


def validate_receipt(
    receipt: dict[str, object],
    *,
    plan: dict[str, object],
    schedule: dict[str, object],
    artifacts: dict[str, dict[str, object]],
    tokenizer,
    candidate_path: Path,
) -> None:
    try:
        _validate_receipt(
            receipt,
            plan=plan,
            schedule=schedule,
            artifacts=artifacts,
            tokenizer=tokenizer,
            candidate_path=candidate_path,
        )
    except ExperimentIncomplete:
        raise
    except BaseException as error:
        raise ExperimentIncomplete("A1-NC0 receipt contract validation could not complete") from error


def validate_finite_metrics(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ExperimentIncomplete("A1-NC0 receipt contains a nonfinite metric")
    if isinstance(value, dict):
        for item in value.values():
            validate_finite_metrics(item)
    elif isinstance(value, list):
        for item in value:
            validate_finite_metrics(item)
