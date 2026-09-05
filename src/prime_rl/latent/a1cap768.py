from __future__ import annotations

import json
import math
import re
from pathlib import Path

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a1nc0 import _CACHE_CLASS_CLOSURE, _E33, _H176, _RUNTIME

PLAN_SCHEMA = "prime-rl/latent-a1-nc0-cap768-r1-plan/v1"
RECEIPT_SCHEMA = "prime-rl/latent-a1-nc0-cap768-receipt/v1"
FAILURE_SCHEMA = "prime-rl/latent-a1-nc0-cap768-failure/v1"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")

SELECTION = [
    {
        "family": "keyed_numeric",
        "evidence_id": "train-keyed_numeric-685221bdf4d3058c",
        "query_id": "train-keyed_numeric-685221bdf4d3058c-q1",
        "parent_unpadded_tokens": 517,
        "mself_unpadded_tokens": 475,
    },
    {
        "family": "relational_join",
        "evidence_id": "train-relational_join-2060fb4da095535a",
        "query_id": "train-relational_join-2060fb4da095535a-q0",
        "parent_unpadded_tokens": 599,
        "mself_unpadded_tokens": 471,
    },
    {
        "family": "config_structure",
        "evidence_id": "held_out-config_structure-92edc7342716a68e",
        "query_id": "held_out-config_structure-92edc7342716a68e-q2",
        "parent_unpadded_tokens": 616,
        "mself_unpadded_tokens": 476,
    },
    {
        "family": "ownership_graph",
        "evidence_id": "held_out-ownership_graph-49bd63f3f5c80cfd",
        "query_id": "held_out-ownership_graph-49bd63f3f5c80cfd-q0",
        "parent_unpadded_tokens": 644,
        "mself_unpadded_tokens": 470,
    },
]
SELECTION_SHA256 = "4696e1e8e075bc6a525054387c7d09ea3890c71915be70847c9352660d829020"
SCHEDULE_SHA256 = "44a41c3f48b013366d318cdf520f1fc62ffa44b117edf966e3dbeb9888632216"
REPAIR_COMMIT = "5c7da8ee788b80a144bd48a89ddb2d037c3766b4"
AUTHORIZED_RUN_ID = "a1-nc0-cap768-run2"
RESOURCE_BOUNDS = {
    "gpus_used": 1,
    "gpu_model": "NVIDIA RTX A6000",
    "minimum_gpu_memory_gib": 47,
    "allocator_cap_gib": 40,
    "minimum_host_ram_gib": 64,
    "minimum_free_disk_gib": 60,
    "compute_seconds": 3000,
    "audit_seconds": 240,
    "failure_audit_seconds": 180,
    "terminal_seconds": 60,
    "outer_wall_seconds": 3600,
    "maximum_receipt_bytes": 64 * 1024 * 1024,
    "maximum_failure_bytes": 16 * 1024 * 1024,
    "maximum_output_directory_bytes": 128 * 1024 * 1024,
    "output_root": "/home/ubuntu/rlm/outputs/latent-a1-nc0-cap768-v1",
}
PRIOR_EVIDENCE = {
    "launch_failure_file_sha256": "9e41fb8107af77ae5258c10d9caff7f13773aaa26162850faab7c085c6440d80",
    "launch_failure_internal_sha256": "d6e688e377054b667ab1889c217a0d3da35c578efd373c4deeb3a1e363ce85b1",
    "launch_log_sha256": "5d4f519dba35d5e8307fce54b1d979bfa937a7d636e4abb29fdddef14b7f0fc3",
    "launch_manifest_sha256": "755bd5ae763d2309f2cd77b0771be2c430e4370c88287bc14c3b3c4a835a6eb5",
    "repair_commit": REPAIR_COMMIT,
    "census_file_sha256": "9bb6e6548553bc78a3f83e05d897fc19865ae8a96f653d33be713e88a4d965ea",
    "census_manifest_sha256": "6d526e7a54d4f2ba63f57277c02a801b90d4ca0a68b698320d3bae8196ed2372",
    "repair_attempt_created_namespace": False,
    "repair_attempt_artifact_inventory": [],
    "repair_attempt_receipt_sha256": None,
    "repair_attempt_log_sha256": None,
    "repair_attempt_status": None,
}
LAUNCHER_REJECTION_EVIDENCE = {
    "status": "launcher_rejected_pre_python",
    "failed_run_id": "a1-nc0-cap768-run1",
    "failed_execution_commit": "38f712b652da0cb86d8ad71087761a5ead9ecdba",
    "failed_mechanism_code_commit": "be5bf43b92a7b999bbf8e700ff49d439f3f7f538",
    "failed_plan_file_sha256": "5f6266d64fb523e11ff48d64246066e948b6d49b564169aa184dc9bbe1c7910b",
    "failed_plan_internal_sha256": "a8af6c2c9a925d8acbddfc103cd587adabc3f353a97430e66f1f189719a22bb7",
    "launch_log_sha256": "b70f54b49c6d9812f801fb793ee3ceee8683186b03213831db7692d7727d6078",
    "exact_error": "scripts/latent/run_a1_nc0_cap768_v1.sh: line 16: shared_project: unbound variable",
    "stage": "shell_readonly_initialization",
    "artifacts": [],
    "shell_exit_nonzero": True,
    "output_namespace_created": False,
    "python_started": False,
    "cuda_runtime_contacted": False,
    "model_loaded": False,
    "scientific_exposure": False,
    "model_update_attempted": False,
}
INTERPRETATION = (
    "capture geometry and resource fit only for a prospective A1 refreeze; no training authorization, bridge "
    "learning, nomination, semantic held-out output, A1 admission, or four-live-floor change"
)
ASSET_PATHS = {
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-render-rejection-failure.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-render-rejection-manifest.sha256",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-render-rejection-run.log",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-census.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-census-manifest.sha256",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-launcher-rejection-run.log",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-train-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-validation-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-held_out-bank-v1.json",
    "scripts/latent/run_a1_nc0_nomination_v1.py",
    "scripts/latent/run_a1_nc0_cap768_v1.py",
    "scripts/latent/run_a1_nc0_cap768_v1.sh",
    "src/prime_rl/latent/__init__.py",
    "src/prime_rl/latent/a0.py",
    "src/prime_rl/latent/a0nc.py",
    "src/prime_rl/latent/a1nc0.py",
    "src/prime_rl/latent/a1cap768.py",
    "src/prime_rl/latent/policy_adapter.py",
    "src/prime_rl/latent/transfer_bank.py",
    "tests/unit/latent/test_a1cap768.py",
}


class DiagnosticIncomplete(RuntimeError):
    pass


class CaptureMechanismRejected(RuntimeError):
    pass


class ResourceFitRejected(RuntimeError):
    pass


def build_schedule() -> list[dict[str, object]]:
    schedule = []
    call_index = 0
    for probe_index, probe in enumerate(SELECTION, 1):
        for modality in ("PARENT", "MSELF"):
            for operation in ("L_ID_KEEP1", "L_E_KEEP1", "L_E_REPEAT_KEEP1", "L_ID_KEEP0_CONTROL"):
                call_index += 1
                schedule.append(
                    {
                        "call_index": call_index,
                        "probe_index": probe_index,
                        "family": probe["family"],
                        "modality": modality,
                        "arm": f"CAP768_P{probe_index:02d}_{modality}_{operation}",
                    }
                )
    if canonical_json_hash(schedule) != SCHEDULE_SHA256:
        raise ValueError("CAP768 call schedule changed")
    return schedule


def memory_labels() -> list[str]:
    labels = ["model_loaded_frozen"]
    for call in build_schedule():
        labels.extend([f"pre_{call['arm']}", f"post_{call['arm']}"])
    labels.extend(["cache_guard_audit_complete", "protected_postflight_complete"])
    if len(labels) != 67 or len(set(labels)) != 67:
        raise ValueError("CAP768 memory schedule changed")
    return labels


def _validate_prior_evidence(repo: Path) -> dict[str, object]:
    experiment = repo / "experiments/qwen35-2b-latent-workspace-v1"
    paths = {
        "failure": experiment / "a1-nc0-render-rejection-failure.json",
        "log": experiment / "a1-nc0-render-rejection-run.log",
        "launch_manifest": experiment / "a1-nc0-render-rejection-manifest.sha256",
        "census": experiment / "a1-nc0-cap768-census.json",
        "census_manifest": experiment / "a1-nc0-cap768-census-manifest.sha256",
    }
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError("CAP768 prior evidence absent or symlinked")
    failure = json.loads(paths["failure"].read_text())
    census = json.loads(paths["census"].read_text())
    observed = {
        "launch_failure_file_sha256": file_sha256(paths["failure"]),
        "launch_failure_internal_sha256": failure.get("failure_sha256"),
        "launch_log_sha256": file_sha256(paths["log"]),
        "launch_manifest_sha256": file_sha256(paths["launch_manifest"]),
        "repair_commit": census.get("mechanism_commit"),
        "census_file_sha256": file_sha256(paths["census"]),
        "census_manifest_sha256": file_sha256(paths["census_manifest"]),
        "repair_attempt_created_namespace": False,
        "repair_attempt_artifact_inventory": [],
        "repair_attempt_receipt_sha256": None,
        "repair_attempt_log_sha256": None,
        "repair_attempt_status": None,
    }
    if (
        observed != PRIOR_EVIDENCE
        or census.get("counts") != {"parent": 96, "mself": 288}
        or census.get("global")
        != {"parent_min": 511, "parent_max": 644, "parent_over_256": 96, "mself_min": 466,
            "mself_max": 476, "mself_over_256": 288}
        or census.get("model_loaded") is not False
        or census.get("torch_cuda_initialized_before") is not False
        or census.get("torch_cuda_initialized_after") is not False
    ):
        raise ValueError("CAP768 prior evidence changed")
    return observed


def _validate_launcher_rejection_evidence(repo: Path) -> dict[str, object]:
    path = (
        repo
        / "experiments/qwen35-2b-latent-workspace-v1"
        / "a1-nc0-cap768-launcher-rejection-run.log"
    )
    if path.is_symlink() or not path.is_file():
        raise ValueError("CAP768 launcher-rejection evidence absent or symlinked")
    if file_sha256(path) != LAUNCHER_REJECTION_EVIDENCE["launch_log_sha256"]:
        raise ValueError("CAP768 launcher-rejection log changed")
    if path.read_text().rstrip("\n") != LAUNCHER_REJECTION_EVIDENCE["exact_error"]:
        raise ValueError("CAP768 launcher-rejection error changed")
    return LAUNCHER_REJECTION_EVIDENCE


def load_plan(plan_path: Path, repo: Path) -> dict[str, object]:
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError("CAP768 plan absent or symlinked")
    plan = json.loads(plan_path.read_text())
    required = {
        "schema_version", "status", "mechanism_code_commit", "plan_sha256", "asset_sha256",
        "selection", "selection_sha256", "call_schedule", "call_schedule_sha256", "memory_labels",
        "memory_labels_sha256", "prior_evidence", "protected_checkpoints", "runtime", "resource_bounds",
        "interpretation_boundary", "execution_authorization", "authorized_run_id",
        "launcher_rejection_evidence",
    }
    assets = plan.get("asset_sha256")
    if (
        set(plan) != required
        or plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "preregistered"
        or plan.get("execution_authorization") != "root_and_evaluator_review_required"
        or plan.get("authorized_run_id") != AUTHORIZED_RUN_ID
        or not _COMMIT.fullmatch(str(plan.get("mechanism_code_commit", "")))
        or plan.get("selection") != SELECTION
        or plan.get("selection_sha256") != SELECTION_SHA256
        or plan.get("call_schedule") != build_schedule()
        or plan.get("call_schedule_sha256") != SCHEDULE_SHA256
        or plan.get("memory_labels") != memory_labels()
        or plan.get("memory_labels_sha256") != canonical_json_hash(memory_labels())
        or plan.get("prior_evidence") != PRIOR_EVIDENCE
        or plan.get("launcher_rejection_evidence") != LAUNCHER_REJECTION_EVIDENCE
        or plan.get("protected_checkpoints") != {"coordinator_e33": _E33, "worker_h176": _H176}
        or plan.get("runtime") != _RUNTIME
        or plan.get("resource_bounds") != RESOURCE_BOUNDS
        or plan.get("interpretation_boundary") != INTERPRETATION
        or not isinstance(assets, dict)
        or not assets
        or set(assets) != ASSET_PATHS
        or any(not _SHA.fullmatch(str(digest)) for digest in assets.values())
        or plan.get("plan_sha256") != canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    ):
        raise ValueError("CAP768 plan changed")
    for relative, expected in assets.items():
        path = repo / relative
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"CAP768 asset changed: {relative}")
    if _validate_prior_evidence(repo) != plan["prior_evidence"]:
        raise ValueError("CAP768 prior evidence binding changed")
    if _validate_launcher_rejection_evidence(repo) != plan["launcher_rejection_evidence"]:
        raise ValueError("CAP768 launcher-rejection evidence binding changed")
    return plan


def validate_receipt(receipt: dict[str, object], *, plan: dict[str, object]) -> None:
    expected_top_keys = {
        "schema_version", "status", "plan_sha256", "mechanism_code_commit", "execution_commit",
        "asset_sha256", "selection", "selection_sha256", "call_schedule", "call_schedule_sha256",
        "prior_evidence", "launcher_rejection_evidence", "run_id", "versions", "runtime_sources",
        "static_guard", "render_preflight", "protected_hashes_before",
        "protected_hashes_after", "checkpoint_metadata_before", "checkpoint_metadata_after",
        "e33_state_tree_before", "e33_state_tree_after", "e33_parameters_frozen_no_grad",
        "worker_h176_loaded", "model_runtime", "probes", "calls", "no_cache_contract", "cache_guard", "memory_ledger",
        "memory_labels_sha256", "resources", "timings", "claim", "training_authorized", "bridge_created",
        "optimizer_created", "backward_used", "checkpoint_created", "candidate_created", "generation_used",
        "model_update_attempted", "semantic_heldout_output", "reusable_hidden_persisted",
        "interpretation_boundary", "receipt_sha256",
    }
    if (
        set(receipt) != expected_top_keys
        or receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != "capture768_mechanism_validated"
        or receipt.get("plan_sha256") != plan.get("plan_sha256")
        or receipt.get("mechanism_code_commit") != plan.get("mechanism_code_commit")
        or receipt.get("asset_sha256") != plan.get("asset_sha256")
        or not _COMMIT.fullmatch(str(receipt.get("execution_commit", "")))
        or receipt.get("execution_commit") == receipt.get("mechanism_code_commit")
        or receipt.get("selection") != SELECTION
        or receipt.get("selection_sha256") != SELECTION_SHA256
        or receipt.get("call_schedule") != build_schedule()
        or receipt.get("call_schedule_sha256") != SCHEDULE_SHA256
        or receipt.get("prior_evidence") != PRIOR_EVIDENCE
        or receipt.get("launcher_rejection_evidence") != LAUNCHER_REJECTION_EVIDENCE
        or receipt.get("run_id") != AUTHORIZED_RUN_ID
        or receipt.get("protected_hashes_before") != plan.get("protected_checkpoints")
        or receipt.get("protected_hashes_after") != plan.get("protected_checkpoints")
        or receipt.get("interpretation_boundary") != INTERPRETATION
        or receipt.get("claim") != "capture768_geometry_and_resource_fit_only"
        or receipt.get("training_authorized") is not False
        or receipt.get("bridge_created") is not False
        or receipt.get("optimizer_created") is not False
        or receipt.get("backward_used") is not False
        or receipt.get("checkpoint_created") is not False
        or receipt.get("candidate_created") is not False
        or receipt.get("generation_used") is not False
        or receipt.get("model_update_attempted") is not False
        or receipt.get("worker_h176_loaded") is not False
        or receipt.get("semantic_heldout_output") is not False
        or receipt.get("reusable_hidden_persisted") is not False
        or receipt.get("versions")
        != {key: _RUNTIME[key] for key in ("python", "transformers", "flash_linear_attention", "torch_distribution", "torch_runtime")}
        or not isinstance(receipt.get("runtime_sources"), dict)
        or set(receipt["runtime_sources"]) != set(_RUNTIME["transformers_source_sha256"])
        or receipt.get("checkpoint_metadata_before")
        != {"coordinator_e33": _RUNTIME["checkpoint_metadata_sha256"],
            "worker_h176": _RUNTIME["checkpoint_metadata_sha256"]}
        or receipt.get("checkpoint_metadata_after") != receipt.get("checkpoint_metadata_before")
        or not _SHA.fullmatch(str(receipt.get("e33_state_tree_before", "")))
        or receipt.get("e33_state_tree_after") != receipt.get("e33_state_tree_before")
        or receipt.get("e33_parameters_frozen_no_grad") is not True
        or receipt.get("model_runtime")
        != {"class": _RUNTIME["model_class"], "hidden_size": 2048, "vocab_size": 248320,
            "dtype": "torch.bfloat16", "device": "cuda:0"}
    ):
        raise DiagnosticIncomplete("CAP768 receipt identity/boundary changed")
    for name, expected_sha in _RUNTIME["transformers_source_sha256"].items():
        source = receipt["runtime_sources"][name]
        if (
            not isinstance(source, dict)
            or set(source) != {"path", "sha256"}
            or source.get("sha256") != expected_sha
            or not isinstance(source.get("path"), str)
        ):
            raise DiagnosticIncomplete("CAP768 runtime source evidence changed")
    preflight = receipt.get("render_preflight")
    preflight_keys = {
        "enable_thinking", "tools_none_for_child", "parent_fixture_messages", "child_base_messages",
        "terminal_token_ids", "fixed_continuation_token_ids", "length_control_token_ids",
        "length_control_tokens_non_special", "tokenizer_eos_token_id", "tokenizer_pad_token_id",
        "maximum_unpadded_feature_tokens", "feature_token_budget", "feature_sequences_truncated",
        "materialized_queries", "tokenized_template_container",
        "preflight_input_ids_extracted_from_batch_encoding", "batch_encoding_extraction_counts",
        "answer_key_interpolation_scope", "answer_key_not_interpolated_into_parent_or_child_opening",
        "render_hashes_sha256", "label_alignment_sha256",
    }
    if (
        not isinstance(preflight, dict)
        or set(preflight) != preflight_keys
        or "label_alignment" in preflight
        or preflight.get("materialized_queries") != 288
        or preflight.get("feature_token_budget") != 768
        or preflight.get("maximum_unpadded_feature_tokens") != 644
        or preflight.get("enable_thinking") is not False
        or preflight.get("tools_none_for_child") is not True
        or preflight.get("parent_fixture_messages") != 4
        or preflight.get("child_base_messages") != 2
        or preflight.get("terminal_token_ids") != [248046, 198]
        or preflight.get("tokenizer_eos_token_id") != 248046
        or preflight.get("tokenizer_pad_token_id") != 248046
        or preflight.get("feature_sequences_truncated") != 0
        or preflight.get("preflight_input_ids_extracted_from_batch_encoding") is not True
        or preflight.get("tokenized_template_container")
        != "transformers.tokenization_utils_base.BatchEncoding"
        or not _SHA.fullmatch(str(preflight.get("render_hashes_sha256", "")))
        or not _SHA.fullmatch(str(preflight.get("label_alignment_sha256", "")))
        or preflight.get("batch_encoding_extraction_counts")
        != {"parent": 96, "child_plain": 288, "child_opening": 288, "child_full": 288, "mself_parent": 288}
    ):
        raise DiagnosticIncomplete("CAP768 all-bank render preflight changed")
    probes = receipt.get("probes")
    calls = receipt.get("calls")
    if not isinstance(probes, list) or len(probes) != 4 or not isinstance(calls, list) or len(calls) != 32:
        raise DiagnosticIncomplete("CAP768 probe/call cardinality changed")
    if [call.get("arm") for call in calls] != [item["arm"] for item in build_schedule()]:
        raise DiagnosticIncomplete("CAP768 call order changed")
    call_keys = {
        "call_index", "probe_index", "family", "modality", "arm", "unpadded_tokens", "padded_tokens",
        "logits_to_keep", "cuda_event_seconds", "wall_seconds", "logits_sha256",
    }
    for call, expected_call in zip(calls, build_schedule(), strict=True):
        expected_probe = SELECTION[expected_call["probe_index"] - 1]
        expected_unpadded = expected_probe[f"{expected_call['modality'].lower()}_unpadded_tokens"]
        if (
            set(call) != call_keys
            or call.get("call_index") != expected_call["call_index"]
            or call.get("probe_index") != expected_call["probe_index"]
            or call.get("family") != expected_call["family"]
            or call.get("modality") != expected_call["modality"]
            or call.get("arm") != expected_call["arm"]
            or call.get("unpadded_tokens") != expected_unpadded
            or call.get("padded_tokens") != 768
            or call.get("logits_to_keep") != (0 if call["arm"].endswith("KEEP0_CONTROL") else 1)
            or not _SHA.fullmatch(str(call.get("logits_sha256", "")))
            or any(
                isinstance(call.get(key), bool)
                or not isinstance(call.get(key), (int, float))
                or not math.isfinite(float(call[key]))
                or call[key] < 0
                for key in ("cuda_event_seconds", "wall_seconds")
            )
        ):
            raise DiagnosticIncomplete("CAP768 call evidence changed")
    case_keys = {
        "unpadded_tokens", "padded_tokens", "padding_tokens", "capture_indices", "capture_shape",
        "input_ids_sha256", "attention_mask_sha256", "captured_mask_sha256", "position_ids_sha256",
        "exact_embeddings_sha256", "operation_hashes",
        "full_hidden_sha256", "capture_sha256", "keep1_logits_sha256", "exact_embeddings_finite",
        "exact_embeddings_requires_grad_false", "left_padding_exact", "attention_mask_exact",
        "position_ids_exact", "no_truncation", "id_embed_keep1_logits_bitwise",
        "id_embed_keep1_full_hidden_bitwise", "id_embed_keep1_capture_bitwise",
        "repeat_same_embedding_object", "repeat_embedding_unchanged", "repeat_logits_bitwise",
        "repeat_full_hidden_bitwise", "repeat_capture_bitwise", "keep0_keep1_full_hidden_bitwise",
        "keep0_keep1_capture_bitwise", "keep0_last_logits_keep1_bitwise", "all_outputs_finite",
        "embedding_lookup_cuda_event_seconds", "embedding_lookup_wall_seconds",
        "four_call_cuda_event_seconds", "four_call_wall_seconds",
    }
    for probe_index, (probe, expected) in enumerate(zip(probes, SELECTION, strict=True), 1):
        if (
            not isinstance(probe, dict)
            or set(probe) != {"selection", "modalities"}
            or probe.get("selection") != expected
            or set(probe.get("modalities", {})) != {"PARENT", "MSELF"}
        ):
            raise DiagnosticIncomplete("CAP768 probe identity changed")
        for modality in ("PARENT", "MSELF"):
            case = probe["modalities"][modality]
            expected_unpadded = expected[f"{modality.lower()}_unpadded_tokens"]
            modality_calls = [
                call for call in calls if call["probe_index"] == probe_index and call["modality"] == modality
            ]
            if (
                not isinstance(case, dict)
                or set(case) != case_keys
                or case.get("unpadded_tokens") != expected_unpadded
                or case.get("padded_tokens") != 768
                or case.get("padding_tokens") != 768 - expected_unpadded
                or case.get("capture_indices") != list(range(640, 768))
                or case.get("capture_shape") != [1, 128, 2048]
                or any(case.get(flag) is not True for flag in (
                    "left_padding_exact", "attention_mask_exact", "position_ids_exact", "no_truncation",
                    "id_embed_keep1_logits_bitwise", "id_embed_keep1_full_hidden_bitwise",
                    "id_embed_keep1_capture_bitwise", "repeat_same_embedding_object", "repeat_embedding_unchanged",
                    "repeat_logits_bitwise", "repeat_full_hidden_bitwise", "repeat_capture_bitwise",
                    "keep0_keep1_full_hidden_bitwise", "keep0_keep1_capture_bitwise",
                    "keep0_last_logits_keep1_bitwise", "all_outputs_finite", "exact_embeddings_finite",
                    "exact_embeddings_requires_grad_false",
                ))
                or any(not _SHA.fullmatch(str(case.get(key, ""))) for key in (
                    "input_ids_sha256", "attention_mask_sha256", "captured_mask_sha256", "position_ids_sha256",
                    "exact_embeddings_sha256", "full_hidden_sha256", "capture_sha256", "keep1_logits_sha256",
                ))
                or any(
                    isinstance(case.get(key), bool)
                    or not isinstance(case.get(key), (int, float))
                    or not math.isfinite(float(case[key]))
                    or case[key] < 0
                    for key in (
                        "embedding_lookup_cuda_event_seconds", "embedding_lookup_wall_seconds",
                        "four_call_cuda_event_seconds", "four_call_wall_seconds",
                    )
                )
                or case.get("four_call_cuda_event_seconds")
                != math.fsum(float(call["cuda_event_seconds"]) for call in modality_calls)
                or case.get("four_call_wall_seconds")
                != math.fsum(float(call["wall_seconds"]) for call in modality_calls)
            ):
                raise DiagnosticIncomplete("CAP768 modality evidence changed")
            operation_hashes = case["operation_hashes"]
            operations = ("L_ID_KEEP1", "L_E_KEEP1", "L_E_REPEAT_KEEP1", "L_ID_KEEP0_CONTROL")
            if (
                not isinstance(operation_hashes, dict)
                or set(operation_hashes) != {
                    f"CAP768_P{probe_index:02d}_{modality}_{operation}" for operation in operations
                }
                or any(
                    not isinstance(item, dict)
                    or set(item) != {"last_logits_sha256", "full_hidden_sha256", "capture_sha256"}
                    or any(not _SHA.fullmatch(str(value)) for value in item.values())
                    for item in operation_hashes.values()
                )
            ):
                raise DiagnosticIncomplete("CAP768 operation hashes changed")
            id1, e1, e2, id0 = [
                operation_hashes[f"CAP768_P{probe_index:02d}_{modality}_{operation}"]
                for operation in operations
            ]
            if not (
                id1["last_logits_sha256"] == e1["last_logits_sha256"] == e2["last_logits_sha256"]
                == id0["last_logits_sha256"]
                and id1["full_hidden_sha256"] == e1["full_hidden_sha256"] == e2["full_hidden_sha256"]
                == id0["full_hidden_sha256"]
                and id1["capture_sha256"] == e1["capture_sha256"] == e2["capture_sha256"]
                == id0["capture_sha256"]
                and case["full_hidden_sha256"] == id1["full_hidden_sha256"]
                and case["capture_sha256"] == id1["capture_sha256"]
                and case["keep1_logits_sha256"] == id1["last_logits_sha256"]
            ):
                raise DiagnosticIncomplete("CAP768 operation parity hashes changed")
    no_cache = receipt.get("no_cache_contract")
    cache = receipt.get("cache_guard")
    if (
        no_cache != {"calls": 32, "use_cache_false": True, "pkv_input_none": True, "pkv_output_none": True,
                     "rope_reset_every_call": True, "embedding_lookups": 8,
                     "model_config_use_cache": False, "generation_config_use_cache": False}
        or not isinstance(cache, dict)
        or set(cache)
        != {"classes", "negative_control_dynamic_cache_tripped", "closure_check_count", "restored_in_finally"}
        or cache.get("classes") != _CACHE_CLASS_CLOSURE
        or cache.get("negative_control_dynamic_cache_tripped") is not True
        or cache.get("closure_check_count") != 67
        or cache.get("restored_in_finally") is not True
    ):
        raise DiagnosticIncomplete("CAP768 cache/no-cache evidence changed")
    labels = plan["memory_labels"]
    rows = receipt.get("memory_ledger")
    if (
        not isinstance(rows, list)
        or [row.get("label") for row in rows] != labels
        or receipt.get("memory_labels_sha256") != canonical_json_hash(labels)
    ):
        raise DiagnosticIncomplete("CAP768 memory ledger changed")
    previous_alloc = previous_reserved = 0
    cap = RESOURCE_BOUNDS["allocator_cap_gib"] * 2**30
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "label", "allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes"
        }:
            raise DiagnosticIncomplete("CAP768 memory row schema changed")
        values = [row.get(key) for key in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes")]
        if (
            any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= cap for value in values)
            or values[2] < values[0] or values[3] < values[1]
            or values[2] < previous_alloc or values[3] < previous_reserved
        ):
            raise DiagnosticIncomplete("CAP768 memory accounting changed")
        previous_alloc, previous_reserved = values[2], values[3]
    resources = receipt.get("resources")
    expected_resource_keys = {
        "gpu_name", "total_gpu_memory_bytes", "allocator_cap_bytes", "peak_allocated_bytes",
        "peak_reserved_bytes", "host_ram_bytes", "free_disk_bytes_before", "visible_cuda_devices",
        "physical_gpu1_unused", "physical_gpu_audit_before", "physical_gpu_audit_after", "network_used",
    }
    if (
        not isinstance(resources, dict)
        or set(resources) != expected_resource_keys
        or resources.get("gpu_name") != RESOURCE_BOUNDS["gpu_model"]
        or resources.get("total_gpu_memory_bytes", 0) < RESOURCE_BOUNDS["minimum_gpu_memory_gib"] * 2**30
        or resources.get("allocator_cap_bytes") != cap
        or resources.get("peak_allocated_bytes") != max(row["peak_allocated_bytes"] for row in rows)
        or resources.get("peak_reserved_bytes") != max(row["peak_reserved_bytes"] for row in rows)
        or resources.get("host_ram_bytes", 0) < RESOURCE_BOUNDS["minimum_host_ram_gib"] * 2**30
        or resources.get("free_disk_bytes_before", 0) < RESOURCE_BOUNDS["minimum_free_disk_gib"] * 2**30
        or resources.get("visible_cuda_devices") != 1
        or resources.get("physical_gpu1_unused") is not True
        or resources.get("network_used") is not False
    ):
        raise DiagnosticIncomplete("CAP768 resource evidence changed")
    for audit in (resources["physical_gpu_audit_before"], resources["physical_gpu_audit_after"]):
        if (
            not isinstance(audit, dict)
            or set(audit) != {"names", "uuids", "memory_used_mib", "compute_apps"}
            or audit["names"] != ["NVIDIA RTX A6000", "NVIDIA RTX A6000"]
            or not isinstance(audit["uuids"], list)
            or len(audit["uuids"]) != 2
            or not isinstance(audit["memory_used_mib"], list)
            or len(audit["memory_used_mib"]) != 2
            or audit["memory_used_mib"][1] > 512
            or not isinstance(audit["compute_apps"], list)
            or any(
                not isinstance(app, dict)
                or set(app) != {"gpu_uuid", "pid"}
                or app["gpu_uuid"] == audit["uuids"][1]
                or isinstance(app["pid"], bool)
                or not isinstance(app["pid"], int)
                or app["pid"] <= 0
                for app in audit["compute_apps"]
            )
        ):
            raise DiagnosticIncomplete("CAP768 physical GPU evidence changed")
    static = receipt.get("static_guard")
    if (
        not isinstance(static, dict)
        or static.get("forbidden_calls") != []
        or static.get("runner_sha256")
        != plan["asset_sha256"].get("scripts/latent/run_a1_nc0_cap768_v1.py")
    ):
        raise DiagnosticIncomplete("CAP768 static source guard changed")
    timings = receipt.get("timings")
    timing_keys = {
        "tokenizer_seconds", "model_load_seconds", "compute_seconds", "audit_seconds",
        "call_cuda_event_seconds_sum", "call_wall_seconds_sum", "embedding_cuda_event_seconds_sum",
        "embedding_wall_seconds_sum", "per_probe", "total_seconds",
    }
    if not isinstance(timings, dict) or set(timings) != timing_keys:
        raise DiagnosticIncomplete("CAP768 timing schema changed")
    for key, value in timings.items():
        if key != "per_probe" and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
        ):
            raise DiagnosticIncomplete("CAP768 timing changed")
    expected_probe_timings = []
    for probe_index, probe in enumerate(probes, 1):
        probe_calls = [call for call in calls if call["probe_index"] == probe_index]
        expected_probe_timings.append({
            "probe_index": probe_index,
            "embedding_cuda_event_seconds": math.fsum(
                probe["modalities"][name]["embedding_lookup_cuda_event_seconds"] for name in ("PARENT", "MSELF")
            ),
            "embedding_wall_seconds": math.fsum(
                probe["modalities"][name]["embedding_lookup_wall_seconds"] for name in ("PARENT", "MSELF")
            ),
            "call_cuda_event_seconds": math.fsum(call["cuda_event_seconds"] for call in probe_calls),
            "call_wall_seconds": math.fsum(call["wall_seconds"] for call in probe_calls),
        })
    if (
        timings["per_probe"] != expected_probe_timings
        or timings["call_cuda_event_seconds_sum"] != math.fsum(call["cuda_event_seconds"] for call in calls)
        or timings["call_wall_seconds_sum"] != math.fsum(call["wall_seconds"] for call in calls)
        or timings["embedding_cuda_event_seconds_sum"]
        != math.fsum(
            probe["modalities"][name]["embedding_lookup_cuda_event_seconds"]
            for probe in probes for name in ("PARENT", "MSELF")
        )
        or timings["embedding_wall_seconds_sum"]
        != math.fsum(
            probe["modalities"][name]["embedding_lookup_wall_seconds"]
            for probe in probes for name in ("PARENT", "MSELF")
        )
        or timings["compute_seconds"] > RESOURCE_BOUNDS["compute_seconds"]
        or timings["audit_seconds"] > RESOURCE_BOUNDS["audit_seconds"]
        or timings["total_seconds"] > RESOURCE_BOUNDS["outer_wall_seconds"]
    ):
        raise DiagnosticIncomplete("CAP768 timing aggregate changed")
    if receipt.get("receipt_sha256") != canonical_json_hash(receipt, omitted_fields=("receipt_sha256",)):
        raise DiagnosticIncomplete("CAP768 receipt hash changed")


def classify_failure(error: BaseException) -> tuple[str, str]:
    if isinstance(error, CaptureMechanismRejected):
        return "capture768_mechanism_rejected", "cache_pkv_parity_repeat_capture_geometry_or_finiteness"
    if isinstance(error, ResourceFitRejected):
        return "capture768_resource_fit_rejected", "cap_oom_or_compute_timeout"
    if isinstance(error, DiagnosticIncomplete):
        return "diagnostic_incomplete", "tokenizer_bank_render_selection_call_receipt_or_asset"
    return "infrastructure_invalid", "runtime_checkpoint_host_gpu_disk_external_or_publication"
