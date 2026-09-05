from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

A0_PLAN_SCHEMA = "prime-rl/latent-a0-mechanism-plan/v1"
A0_BANK_SCHEMA = "prime-rl/latent-a0-mechanism-bank/v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_E33 = "e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47"
_H176 = "77980e247bbccd6463ddda02cd42d2c357e15f8ec1ad0ea84627e008a8674a1e"
_ASSET_PATHS = {
    "experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-bank-v1.json",
    "scripts/latent/run_a0_mechanism_v1.py",
    "scripts/latent/run_a0_mechanism_v1.sh",
    "src/prime_rl/latent/__init__.py",
    "src/prime_rl/latent/a0.py",
    "src/prime_rl/latent/policy_adapter.py",
}


def canonical_json_hash(value: object, *, omitted_fields: tuple[str, ...] = ()) -> str:
    if isinstance(value, dict):
        payload = {key: item for key, item in value.items() if key not in omitted_fields}
    else:
        payload = value
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_a0_bank(bank: dict[str, object]) -> None:
    if set(bank) != {"schema_version", "examples"}:
        raise ValueError("A0 bank fields differ from the v1 schema")
    if bank.get("schema_version") != A0_BANK_SCHEMA:
        raise ValueError("unknown A0 bank schema")
    examples = bank.get("examples")
    if not isinstance(examples, list) or len(examples) != 4:
        raise ValueError("A0 v1 requires exactly four distinct fixed mechanism probes")
    expected_ids = tuple(f"a0-mechanism-{index:04d}" for index in range(1, 5))
    if tuple(example.get("example_id") for example in examples if isinstance(example, dict)) != expected_ids:
        raise ValueError("A0 probe IDs differ from the frozen ordered v1 bank")
    for example in examples:
        if not isinstance(example, dict) or set(example) != {
            "example_id",
            "parent_messages",
            "child_messages",
            "continuation_text",
        }:
            raise ValueError("A0 probe fields differ from the v1 schema")
        if not isinstance(example.get("continuation_text"), str) or not example["continuation_text"]:
            raise ValueError("A0 probe needs a fixed continuation text")
        for field, expected_roles in (
            ("parent_messages", ("system", "user", "assistant")),
            ("child_messages", ("system", "user")),
        ):
            messages = example.get(field)
            if not isinstance(messages, list) or tuple(message.get("role") for message in messages) != expected_roles:
                raise ValueError(f"{field} roles differ from the A0 contract")
        for message in (*example["parent_messages"], *example["child_messages"]):
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or not isinstance(message.get("content"), str)
                or not message["content"]
            ):
                raise ValueError("A0 text message fixture is malformed")


def validate_a0_plan(plan: dict[str, object], *, bank_sha256: str) -> None:
    required = {
        "schema_version",
        "status",
        "execution_authorization",
        "mechanism_code_commit",
        "asset_sha256",
        "plan_sha256",
        "bank_sha256",
        "protected_checkpoints",
        "remote_paths",
        "runtime",
        "mechanism",
        "admission",
        "resource_bounds",
        "failure_classification",
        "promotion_boundary",
    }
    if set(plan) != required:
        raise ValueError("A0 plan fields differ from the v1 schema")
    if plan.get("schema_version") != A0_PLAN_SCHEMA:
        raise ValueError("unknown A0 plan schema")
    if plan.get("status") != "preregistered":
        raise ValueError("A0 plan is not preregistered")
    if plan.get("execution_authorization") != "root_review_required":
        raise ValueError("A0 execution authorization boundary changed")
    commit = plan.get("mechanism_code_commit")
    if not isinstance(commit, str) or not _GIT_COMMIT_RE.fullmatch(commit):
        raise ValueError("A0 mechanism code commit is missing or malformed")
    assets = plan.get("asset_sha256")
    if not isinstance(assets, dict) or set(assets) != _ASSET_PATHS:
        raise ValueError("A0 executable asset set differs from the frozen mechanism")
    if any(not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in assets.values()):
        raise ValueError("A0 executable asset hash is malformed")
    if plan.get("bank_sha256") != bank_sha256 or not _SHA256_RE.fullmatch(bank_sha256):
        raise ValueError("A0 bank hash differs from the preregistration")
    plan_hash = plan.get("plan_sha256")
    if not isinstance(plan_hash, str) or not _SHA256_RE.fullmatch(plan_hash):
        raise ValueError("A0 plan hash is missing or malformed")
    if plan_hash != canonical_json_hash(plan, omitted_fields=("plan_sha256",)):
        raise ValueError("A0 plan hash does not match its canonical content")

    if plan.get("protected_checkpoints") != {"coordinator_e33": _E33, "worker_h176": _H176}:
        raise ValueError("A0 protected checkpoint hashes differ from canonical e33/H176")
    paths = plan.get("remote_paths")
    if not isinstance(paths, dict) or set(paths) != {"coordinator_e33", "worker_h176"}:
        raise ValueError("A0 remote paths are incomplete")
    if any(not isinstance(path, str) or not path.startswith("/home/ubuntu/rlm/") for path in paths.values()):
        raise ValueError("A0 remote model paths must be absolute paths under the retained checkout")

    expected_runtime = {
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
    }
    if plan.get("runtime") != expected_runtime:
        raise ValueError("A0 runtime differs from the frozen mechanism runtime")
    expected_mechanism = {
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
    }
    if plan.get("mechanism") != expected_mechanism:
        raise ValueError("A0 mechanism differs from the frozen final-hidden/soft-embedding probe")
    expected_admission = {
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
    }
    if plan.get("admission") != expected_admission:
        raise ValueError("A0 admission contract differs from v1")
    expected_resources = {
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
    }
    if plan.get("resource_bounds") != expected_resources:
        raise ValueError("A0 resources differ from the bounded one-GPU probe")
    if plan.get("failure_classification") != {
        "mechanism_predicate_failure": "mechanism_rejected",
        "environment_provenance_timeout_or_oom": "infrastructure_invalid",
        "run_id_reusable": False,
    }:
        raise ValueError("A0 failure classification changed")
    if plan.get("promotion_boundary") != (
        "A0 rendered-transcript carrier/autograd mechanism only; no harness acceptance/timing, bridge learnability, "
        "training, model admission, or A1 authorization"
    ):
        raise ValueError("A0 promotion boundary changed")


def load_and_validate_a0_plan(plan_path: Path, bank_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    if plan_path.is_symlink() or bank_path.is_symlink():
        raise ValueError("A0 plan and bank must not be symlinks")
    if not plan_path.is_file() or not bank_path.is_file():
        raise ValueError("A0 plan or bank is absent")
    plan = json.loads(plan_path.read_text())
    bank = json.loads(bank_path.read_text())
    validate_a0_bank(bank)
    validate_a0_plan(plan, bank_sha256=file_sha256(bank_path))
    return plan, bank
