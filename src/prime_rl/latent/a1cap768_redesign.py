from __future__ import annotations

import json
import math
import re
from pathlib import Path

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a1cap768 import (
    _CACHE_CLASS_CLOSURE,
    _E33,
    _H176,
    _RUNTIME,
    SELECTION,
    SELECTION_SHA256,
)
from prime_rl.latent.a1cap768_flag0 import (
    ASSET_PATHS as FLAG0_ASSET_PATHS,
)
from prime_rl.latent.a1cap768_flag0 import (
    COMPARISON_SCHEDULE as LOCAL_COMPARISONS,
)
from prime_rl.latent.a1cap768_flag0 import (
    COMPARISON_SCHEDULE_SHA256 as LOCAL_COMPARISONS_SHA256,
)
from prime_rl.latent.a1cap768_flag0 import (
    FLAG_NAMES,
    FLAG_NAMES_SHA256,
)
from prime_rl.latent.a1cap768_flag0 import (
    RESOURCE_BOUNDS as FLAG0_RESOURCE_BOUNDS,
)

PLAN_SCHEMA = "prime-rl/latent-a1-nc0-cap768-redesign-plan/v1"
RECEIPT_SCHEMA = "prime-rl/latent-a1-nc0-cap768-redesign-receipt/v1"
FAILURE_SCHEMA = "prime-rl/latent-a1-nc0-cap768-redesign-failure/v1"
AUTHORIZED_RUN_ID = "a1-nc0-cap768-redesign-run1"

CASE_SCHEDULE = [
    {"probe_index": probe_index, "family": selection["family"], "modality": modality}
    for probe_index, selection in enumerate(SELECTION, 1)
    for modality in ("PARENT", "MSELF")
]
CASE_SCHEDULE_SHA256 = "7830929d5b6135cf5641c748ba49d5ecb8229cf9023c73338859b39b3b715445"
SCIENTIFIC_EXPOSURE_BOUNDARY = {
    "schema": "prime-rl/latent-a1-nc0-cap768-redesign-scientific-exposure-boundary/v1",
    "selection_reused": True,
    "selection_adaptive": False,
    "all_eight_selection_and_token_lengths_previously_exposed_tokenizer_only": True,
    "scientific_forward_exposure_before_redesign": [
        {
            **case,
            "exposed": case["probe_index"] == 1 and case["modality"] == "PARENT",
            "sources": ["CAP768_RUN4", "FLAG0_RUN1"]
            if case["probe_index"] == 1 and case["modality"] == "PARENT"
            else [],
        }
        for case in CASE_SCHEDULE
    ],
    "reuse_scope": "interface_mechanism_only",
    "heldout_or_generalization_claim_allowed": False,
}
SCIENTIFIC_EXPOSURE_BOUNDARY_SHA256 = "b8a4398654b6d53966ad3e56ac685ee20a02593c5f260ca78fe3a04b3704ddfe"

LOCAL_OPERATIONS = (
    ("embedding_lookup", "EXACT_EMBED_LOOKUP"),
    ("model_forward", "L_ID_KEEP1"),
    ("model_forward", "L_E_KEEP1"),
    ("model_forward", "L_E_REPEAT_KEEP1"),
    ("model_forward", "L_ID_KEEP0_CONTROL"),
    ("lm_head_projection", "PROJ_ID1_LAST"),
    ("lm_head_projection", "PROJ_ID0_LAST"),
)


def build_operation_schedule() -> list[dict[str, object]]:
    operations = []
    operation_index = 0
    for case in CASE_SCHEDULE:
        for kind, suffix in LOCAL_OPERATIONS:
            operation_index += 1
            prefix = f"CAP768R_P{case['probe_index']:02d}_{case['modality']}"
            operations.append(
                {
                    "operation_index": operation_index,
                    "probe_index": case["probe_index"],
                    "family": case["family"],
                    "modality": case["modality"],
                    "kind": kind,
                    "name": f"{prefix}_{suffix}",
                }
            )
    return operations


OPERATION_SCHEDULE_SHA256 = "bd98a70e27c35e92217b663f4d67a39d91f2bc9b4a58b1be6e42c972a1d035bc"


def build_comparison_schedule() -> list[dict[str, object]]:
    comparisons = []
    comparison_index = 0
    for case in CASE_SCHEDULE:
        for local in LOCAL_COMPARISONS:
            comparison_index += 1
            comparisons.append(
                {
                    "comparison_index": comparison_index,
                    "probe_index": case["probe_index"],
                    "family": case["family"],
                    "modality": case["modality"],
                    "name": local["name"],
                    "lhs": local["lhs"],
                    "rhs": local["rhs"],
                }
            )
    return comparisons


COMPARISON_SCHEDULE_SHA256 = "71806dea43a463bf88d33761d0725626c158b8475e6c687d96ae1f46a89008b1"
DESCRIPTIVE_FLAG_NAMES = [
    "keep0_last_logits_keep1_bitwise",
    "proj_id0_matches_id0_last_logits_bitwise",
]
DESCRIPTIVE_FLAG_NAMES_SHA256 = "f6c7bfcc31a9dd8d8231511748fb954eb2c4fb745da102c6747b943e83d02bbf"
GATING_FLAG_NAMES = [name for name in FLAG_NAMES if name not in DESCRIPTIVE_FLAG_NAMES]
GATING_FLAG_NAMES_SHA256 = "54ea69aedd2e99bbf1a74692c2b5a31a85eb5bc083452bb794bec31648429e2e"


def memory_labels() -> list[str]:
    labels = ["model_loaded_frozen"]
    offset = 0
    operations = build_operation_schedule()
    for case in CASE_SCHEDULE:
        for operation in operations[offset : offset + len(LOCAL_OPERATIONS)]:
            labels.extend([f"pre_{operation['name']}", f"post_{operation['name']}"])
        offset += len(LOCAL_OPERATIONS)
        labels.append(f"post_CAP768R_P{case['probe_index']:02d}_{case['modality']}_RELEASE")
    labels.extend(["cache_guard_audit_complete", "protected_postflight_complete"])
    return labels


MEMORY_LABELS_SHA256 = "102d32905692dc9fe8794c55858e666fb7adcc7cad3a472204a7d486e4ba94d3"
RESOURCE_BOUNDS = {
    **FLAG0_RESOURCE_BOUNDS,
    "output_root": "/home/ubuntu/rlm/outputs/latent-a1-nc0-cap768-redesign-v1",
}
INTERPRETATION = (
    "reused fixed-length no-cache carrier/capture parity and resource fit only; A1-NC0 remains blocked, "
    "FLAG0 remains incomplete, and no training, nomination, admission, promotion, or live-floor change is authorized"
)
DECISION_BOUNDARY = {
    "claim": "capture768_fixed_length_nocache_carrier_capture_parity_and_resource_fit_only",
    "training_authorized": False,
    "A1_NC0_unblocked": False,
    "nomination": False,
    "admission": False,
    "promotion": False,
    "live_trajectories": 0,
    "four_live_floor_unchanged": True,
    "flag0_counted_as_valid": False,
}
FLAG0_INCOMPLETE_EVIDENCE = {
    "failure_file_sha256": "2e7934efe6f3ffaa2c63734dbd31f6de9dbee5650f41a2de2610fe5ffdbc25fe",
    "failure_internal_sha256": "716060c284cf9a5abc4aaa9e58ba272f8725eaf312fdc24689713108648c05f7",
    "launch_log_sha256": "fccf3b2f77b227cdeb20f33c52b38514481dcaad495e79aa19f67dd960c4ebb8",
    "execution_commit": "b6e2b43422cebafecc3ccc232912a028da7c0c3b",
    "mechanism_code_commit": "1a61d101e7164f2300148c1b452c1f614a15f99e",
    "plan_internal_sha256": "847ac4caa0e0420b127336650c8e88ea2fe8316e0da30ac9a25ddcddd8a62438",
    "status": "capture768_flag_isolation_incomplete",
    "failure_category": "diagnostic_operation_or_evidence_incomplete",
    "error_type": "DiagnosticIncomplete",
    "exact_error": "FLAG0 forbidden training/generation source appeared",
    "run_id": "a1-nc0-cap768-flag-isolation-run1",
    "model_loaded": True,
    "operation_count": 7,
    "flag_count": 25,
    "comparison_count": 13,
    "memory_count": 17,
    "cache_closure_check_count": 11,
    "cache_guard_restored": True,
    "false_flags": DESCRIPTIVE_FLAG_NAMES,
    "protected_disk_state_metadata_exact": True,
    "e33_gradients_absent": True,
    "worker_h176_loaded": False,
    "model_update_attempted": False,
    "bridge_created": False,
    "optimizer_created": False,
    "backward_used": False,
    "checkpoint_created": False,
    "candidate_created": False,
    "asset_count": 42,
    "failure_audit_errors": [],
}
STATIC_GUARD_PROOF1_EVIDENCE = {
    "receipt_file_sha256": "0ad3dcf09cdd69dcbd9a9f9cb6facb091cdd43ed360f3c8fc324607c30f66c00",
    "receipt_internal_sha256": "16b4439d836d3684060a407211e8f6f00095a7da333f80e01212f8c85691f0af",
    "command_file_sha256": "2aeff6b1293bade5081d46cab2b8dd58592a4692a0bc3b4efd34d19c88754011",
    "proof_log_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "exit_status_file_sha256": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "execution_commit": "6ca5caf13f1ace6f270ddcd8b09845df64922e90",
    "runner_sha256": "ff5ef962ec3e70d399e85d966f24dbe6493fb34c818a8063a73de1f6ae94d2e4",
    "guard_module_sha256": "ed7d6c0fc16cc0c24f8cc55907db69cf77079adebc963425f2cc1e7c78e568f6",
    "negative_fixture_count": 5,
    "cuda_hidden_uninitialized": True,
    "model_loaded": False,
    "model_forward_count": 0,
    "scientific_exposure": False,
    "model_update_attempted": False,
}
STATIC_GUARD_PROOF2_EVIDENCE = {
    "receipt_file_sha256": "6b566efd7b893520a7a0bf3edc51337e95f290aa1bf9812770c86ed5d973b431",
    "receipt_internal_sha256": "311da17d7fe19cb4f50d829048963a02eb2eeda0eb38f2cf2376fcda3ce50900",
    "command_file_sha256": "da36abc7f33b8927de8859d48b4ef92ca2c3fb67979fbe6fcf49cf45ab4cb53e",
    "proof_log_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "exit_status_file_sha256": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "execution_commit": "339613b77f515c95bf18f3cba4deeeb201d6b0f5",
    "runner_sha256": "1397b26e4b1d86acae9f928c290cd9b4dbd7304a260bd28b3a13e2f4263d9e99",
    "guard_module_sha256": "ed7d6c0fc16cc0c24f8cc55907db69cf77079adebc963425f2cc1e7c78e568f6",
    "negative_fixture_count": 5,
    "cuda_hidden_uninitialized": True,
    "model_loaded": False,
    "model_forward_count": 0,
    "scientific_exposure": False,
    "model_update_attempted": False,
}
STATIC_GUARD_PROOF_EVIDENCE = {
    "receipt_file_sha256": "32ff5413754deca01fa4e8beff66ddad5370d3c9121e485be5d2fd944afdb267",
    "receipt_internal_sha256": "d4d370aaad994dc0cd4be8886096152f04eab3106dbf8892edfdc32b2eb854d2",
    "command_file_sha256": "69f365e0dceb589092c8dee5838f8684ecf24427af6d4df1118627a5517f77cd",
    "proof_log_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "exit_status_file_sha256": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "execution_commit": "3bc863706ba881af5ac6e453969fbff312dc657d",
    "runner_sha256": "8403c3af2bb4f3cdcbb8fd866fb34c673c897c65cfefafd6b448688a5ceebea4",
    "guard_module_sha256": "ed7d6c0fc16cc0c24f8cc55907db69cf77079adebc963425f2cc1e7c78e568f6",
    "negative_fixture_count": 5,
    "cuda_hidden_uninitialized": True,
    "model_loaded": False,
    "model_forward_count": 0,
    "scientific_exposure": False,
    "model_update_attempted": False,
}
SUPERSEDED_PAIR_NO_GO = {
    "status": "gatekeeper_no_go_not_authorized",
    "mechanism_code_commit": "4fb657d1730727ecab747ca6bcc14188124c8e64",
    "freeze_commit": "add62ececae4c64c0a42ad746e66b08cfa8028b5",
    "plan_file_sha256": "2198d6edb30b8a3d216dea92a4e31d4a78262abb56af15859e4a9fdeb1fcc5a9",
    "plan_internal_sha256": "a9ee9d6759c20937b9a85385a1e7b20a9816d5fd073d795e63a5bb2810bc7aee",
    "exact_blocker": "release checkpoint retained CUDA ids, padded, mask, and positions tensors",
    "model_or_gpu_exposure": False,
    "execution_authorized": False,
}

ASSET_PATHS = set(FLAG0_ASSET_PATHS) | {
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-flag0-plan-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-flag0-incomplete-failure.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-flag0-incomplete-run.log",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-redesign-static-guard-proof-receipt.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-redesign-static-guard-proof-command.txt",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-redesign-static-guard-proof-exit.txt",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-redesign-static-guard-proof.log",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-redesign-static-guard-proof-v2-receipt.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-redesign-static-guard-proof-v2-command.txt",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-redesign-static-guard-proof-v2-exit.txt",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-redesign-static-guard-proof-v2.log",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-redesign-static-guard-proof-v3-receipt.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-redesign-static-guard-proof-v3-command.txt",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-redesign-static-guard-proof-v3-exit.txt",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-redesign-static-guard-proof-v3.log",
    "scripts/latent/run_a1_nc0_cap768_redesign_v1.py",
    "scripts/latent/run_a1_nc0_cap768_redesign_v1.sh",
    "scripts/latent/prove_a1_nc0_cap768_redesign_static_guard_v1.py",
    "src/prime_rl/latent/a1cap768_redesign.py",
    "src/prime_rl/latent/cap768_redesign_invariants.py",
    "tests/unit/latent/test_a1cap768_redesign.py",
    "tests/unit/latent/test_cap768_redesign_invariants.py",
}

_SHA = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


class DiagnosticIncomplete(RuntimeError):
    pass


class NoCacheRejected(RuntimeError):
    pass


class ResourceFitRejected(RuntimeError):
    pass


def validate_constants() -> None:
    values = (
        (SELECTION, SELECTION_SHA256),
        (CASE_SCHEDULE, CASE_SCHEDULE_SHA256),
        (SCIENTIFIC_EXPOSURE_BOUNDARY, SCIENTIFIC_EXPOSURE_BOUNDARY_SHA256),
        (build_operation_schedule(), OPERATION_SCHEDULE_SHA256),
        (LOCAL_COMPARISONS, LOCAL_COMPARISONS_SHA256),
        (build_comparison_schedule(), COMPARISON_SCHEDULE_SHA256),
        (FLAG_NAMES, FLAG_NAMES_SHA256),
        (GATING_FLAG_NAMES, GATING_FLAG_NAMES_SHA256),
        (DESCRIPTIVE_FLAG_NAMES, DESCRIPTIVE_FLAG_NAMES_SHA256),
        (memory_labels(), MEMORY_LABELS_SHA256),
    )
    if any(canonical_json_hash(value) != expected for value, expected in values):
        raise ValueError("CAP768R frozen constant changed")
    if (
        len(CASE_SCHEDULE) != 8
        or len(build_operation_schedule()) != 56
        or len(build_comparison_schedule()) != 104
        or len(GATING_FLAG_NAMES) != 23
        or len(memory_labels()) != 123
        or len(set(memory_labels())) != 123
    ):
        raise ValueError("CAP768R frozen cardinality changed")


def validate_flag0_incomplete(repo: Path) -> dict[str, object]:
    experiment = repo / "experiments/qwen35-2b-latent-workspace-v1"
    failure_path = experiment / "a1-nc0-cap768-flag0-incomplete-failure.json"
    log_path = experiment / "a1-nc0-cap768-flag0-incomplete-run.log"
    plan_path = experiment / "a1-nc0-cap768-flag0-plan-v1.json"
    if any(path.is_symlink() or not path.is_file() for path in (failure_path, log_path, plan_path)):
        raise ValueError("CAP768R FLAG0 evidence absent or symlinked")
    failure = json.loads(failure_path.read_text())
    prior_plan = json.loads(plan_path.read_text())
    flags = failure.get("flags_partial")
    comparisons = failure.get("comparisons_partial")
    operations = failure.get("operation_timings_partial")
    memory = failure.get("memory_ledger_partial")
    cache = failure.get("cache_guard_partial")
    if (
        file_sha256(failure_path) != FLAG0_INCOMPLETE_EVIDENCE["failure_file_sha256"]
        or file_sha256(log_path) != FLAG0_INCOMPLETE_EVIDENCE["launch_log_sha256"]
        or failure.get("failure_sha256") != FLAG0_INCOMPLETE_EVIDENCE["failure_internal_sha256"]
        or failure.get("failure_sha256") != canonical_json_hash(failure, omitted_fields=("failure_sha256",))
        or failure.get("execution_commit") != FLAG0_INCOMPLETE_EVIDENCE["execution_commit"]
        or failure.get("mechanism_code_commit") != FLAG0_INCOMPLETE_EVIDENCE["mechanism_code_commit"]
        or failure.get("plan_sha256") != FLAG0_INCOMPLETE_EVIDENCE["plan_internal_sha256"]
        or prior_plan.get("plan_sha256") != FLAG0_INCOMPLETE_EVIDENCE["plan_internal_sha256"]
        or any(
            failure.get(key) != FLAG0_INCOMPLETE_EVIDENCE[key]
            for key in (
                "status",
                "failure_category",
                "error_type",
                "run_id",
            )
        )
        or failure.get("error") != FLAG0_INCOMPLETE_EVIDENCE["exact_error"]
        or failure.get("model_loaded") is not True
        or not isinstance(flags, dict)
        or list(sorted(flags)) != list(sorted(FLAG_NAMES))
        or [name for name in FLAG_NAMES if flags[name] is False] != DESCRIPTIVE_FLAG_NAMES
        or not isinstance(comparisons, list)
        or len(comparisons) != 13
        or not isinstance(operations, list)
        or len(operations) != 7
        or not isinstance(memory, list)
        or len(memory) != 17
        or not isinstance(cache, dict)
        or cache.get("classes") != _CACHE_CLASS_CLOSURE
        or cache.get("closure_check_count") != 11
        or cache.get("restored_in_finally") is not True
        or failure.get("protected_hashes_before") != {"coordinator_e33": _E33, "worker_h176": _H176}
        or failure.get("protected_hashes_before") != failure.get("protected_hash_probe_after_failure")
        or failure.get("checkpoint_metadata_before")
        != {
            "coordinator_e33": _RUNTIME["checkpoint_metadata_sha256"],
            "worker_h176": _RUNTIME["checkpoint_metadata_sha256"],
        }
        or failure.get("checkpoint_metadata_before") != failure.get("checkpoint_metadata_probe_after_failure")
        or failure.get("e33_state_tree_before") != failure.get("e33_state_tree_failure_audit")
        or failure.get("e33_gradients_absent_failure_audit") is not True
        or failure.get("worker_h176_loaded") is not False
        or any(
            failure.get(key) is not False
            for key in (
                "model_update_attempted",
                "bridge_created",
                "optimizer_created",
                "backward_used",
                "checkpoint_created",
                "candidate_created",
            )
        )
        or failure.get("asset_hashes_match_plan") is not True
        or failure.get("asset_hash_probe_after_failure") != prior_plan.get("asset_sha256")
        or len(failure.get("asset_hash_probe_after_failure", {})) != 42
        or failure.get("failure_audit_errors") != []
    ):
        raise ValueError("CAP768R FLAG0 incomplete evidence changed")
    return FLAG0_INCOMPLETE_EVIDENCE


def _validate_static_guard_proof(
    repo: Path, *, expected: dict[str, object], stem: str
) -> dict[str, object]:
    experiment = repo / "experiments/qwen35-2b-latent-workspace-v1"
    paths = {
        "receipt": experiment / f"{stem}-receipt.json",
        "command": experiment / f"{stem}-command.txt",
        "log": experiment / f"{stem}.log",
        "exit": experiment / f"{stem}-exit.txt",
    }
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError("CAP768R static guard proof absent or symlinked")
    receipt = json.loads(paths["receipt"].read_text())
    expected_hashes = {
        "receipt": expected["receipt_file_sha256"],
        "command": expected["command_file_sha256"],
        "log": expected["proof_log_sha256"],
        "exit": expected["exit_status_file_sha256"],
    }
    if (
        any(file_sha256(paths[name]) != expected for name, expected in expected_hashes.items())
        or paths["exit"].read_bytes() != b"0\n"
        or receipt.get("proof_sha256") != expected["receipt_internal_sha256"]
        or receipt.get("proof_sha256") != canonical_json_hash({**receipt, "proof_sha256": ""})
        or receipt.get("execution_commit") != expected["execution_commit"]
        or receipt.get("runner_sha256") != expected["runner_sha256"]
        or receipt.get("guard_module_sha256") != expected["guard_module_sha256"]
        or receipt.get("status") != "static_guard_validated_cuda_hidden"
        or receipt.get("positive")
        != {
            "runner_sha256": expected["runner_sha256"],
            "forbidden_calls": [],
            "forbidden_identifiers": [],
            "forbidden_imports": [],
        }
        or receipt.get("negative_fixtures")
        != {
            "adamw_attribute": True,
            "backward_call": True,
            "generate_call": True,
            "optimizer_step_call": True,
            "workspace_bridge_name": True,
        }
        or receipt.get("cuda_visible_devices") != ""
        or receipt.get("cuda_initialized_before_after") is not True
        or receipt.get("model_loaded") is not False
        or receipt.get("model_forward_count") != 0
        or receipt.get("scientific_exposure") is not False
        or receipt.get("model_update_attempted") is not False
    ):
        raise ValueError("CAP768R static guard proof changed")
    return expected


def validate_static_guard_proof(repo: Path) -> dict[str, object]:
    return _validate_static_guard_proof(
        repo,
        expected=STATIC_GUARD_PROOF_EVIDENCE,
        stem="a1-nc0-cap768-redesign-static-guard-proof-v3",
    )


def load_plan(plan_path: Path, repo: Path) -> dict[str, object]:
    validate_constants()
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError("CAP768R plan absent or symlinked")
    plan = json.loads(plan_path.read_text())
    required = {
        "schema_version",
        "status",
        "mechanism_code_commit",
        "plan_sha256",
        "asset_sha256",
        "selection",
        "selection_sha256",
        "case_schedule",
        "case_schedule_sha256",
        "operation_schedule",
        "operation_schedule_sha256",
        "local_comparisons",
        "local_comparisons_sha256",
        "comparison_schedule",
        "comparison_schedule_sha256",
        "flag_names",
        "flag_names_sha256",
        "gating_flag_names",
        "gating_flag_names_sha256",
        "descriptive_flag_names",
        "descriptive_flag_names_sha256",
        "memory_labels",
        "memory_labels_sha256",
        "flag0_incomplete_evidence",
        "static_guard_proof_evidence",
        "superseded_static_guard_proof_evidence",
        "superseded_pair_no_go",
        "scientific_exposure_boundary",
        "scientific_exposure_boundary_sha256",
        "protected_checkpoints",
        "runtime",
        "resource_bounds",
        "interpretation_boundary",
        "decision_boundary",
        "execution_authorization",
        "authorized_run_id",
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
        or plan.get("case_schedule") != CASE_SCHEDULE
        or plan.get("case_schedule_sha256") != CASE_SCHEDULE_SHA256
        or plan.get("operation_schedule") != build_operation_schedule()
        or plan.get("operation_schedule_sha256") != OPERATION_SCHEDULE_SHA256
        or plan.get("local_comparisons") != LOCAL_COMPARISONS
        or plan.get("local_comparisons_sha256") != LOCAL_COMPARISONS_SHA256
        or plan.get("comparison_schedule") != build_comparison_schedule()
        or plan.get("comparison_schedule_sha256") != COMPARISON_SCHEDULE_SHA256
        or plan.get("flag_names") != FLAG_NAMES
        or plan.get("flag_names_sha256") != FLAG_NAMES_SHA256
        or plan.get("gating_flag_names") != GATING_FLAG_NAMES
        or plan.get("gating_flag_names_sha256") != GATING_FLAG_NAMES_SHA256
        or plan.get("descriptive_flag_names") != DESCRIPTIVE_FLAG_NAMES
        or plan.get("descriptive_flag_names_sha256") != DESCRIPTIVE_FLAG_NAMES_SHA256
        or plan.get("memory_labels") != memory_labels()
        or plan.get("memory_labels_sha256") != MEMORY_LABELS_SHA256
        or plan.get("flag0_incomplete_evidence") != FLAG0_INCOMPLETE_EVIDENCE
        or plan.get("static_guard_proof_evidence") != STATIC_GUARD_PROOF_EVIDENCE
        or plan.get("superseded_static_guard_proof_evidence")
        != [STATIC_GUARD_PROOF1_EVIDENCE, STATIC_GUARD_PROOF2_EVIDENCE]
        or plan.get("superseded_pair_no_go") != SUPERSEDED_PAIR_NO_GO
        or plan.get("scientific_exposure_boundary") != SCIENTIFIC_EXPOSURE_BOUNDARY
        or plan.get("scientific_exposure_boundary_sha256") != SCIENTIFIC_EXPOSURE_BOUNDARY_SHA256
        or plan.get("protected_checkpoints") != {"coordinator_e33": _E33, "worker_h176": _H176}
        or plan.get("runtime") != _RUNTIME
        or plan.get("resource_bounds") != RESOURCE_BOUNDS
        or plan.get("interpretation_boundary") != INTERPRETATION
        or plan.get("decision_boundary") != DECISION_BOUNDARY
        or not isinstance(assets, dict)
        or set(assets) != ASSET_PATHS
        or any(not _SHA.fullmatch(str(value)) for value in assets.values())
        or plan.get("plan_sha256") != canonical_json_hash(plan, omitted_fields=("plan_sha256",))
    ):
        raise ValueError("CAP768R plan changed")
    for relative, expected in assets.items():
        path = repo / relative
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"CAP768R asset changed: {relative}")
    if validate_flag0_incomplete(repo) != plan["flag0_incomplete_evidence"]:
        raise ValueError("CAP768R FLAG0 binding changed")
    if validate_static_guard_proof(repo) != plan["static_guard_proof_evidence"]:
        raise ValueError("CAP768R static guard proof binding changed")
    superseded_proofs = [
        _validate_static_guard_proof(
            repo,
            expected=STATIC_GUARD_PROOF1_EVIDENCE,
            stem="a1-nc0-cap768-redesign-static-guard-proof",
        ),
        _validate_static_guard_proof(
            repo,
            expected=STATIC_GUARD_PROOF2_EVIDENCE,
            stem="a1-nc0-cap768-redesign-static-guard-proof-v2",
        ),
    ]
    if superseded_proofs != plan["superseded_static_guard_proof_evidence"]:
        raise ValueError("CAP768R superseded proof binding changed")
    return plan


def classification(all_probes_qualify: bool) -> str:
    return "capture768_redesign_validated" if all_probes_qualify else "capture768_redesign_rejected"


def classify_failure(error: BaseException) -> tuple[str, str]:
    if isinstance(error, NoCacheRejected):
        return "capture768_redesign_nocache_rejected", "cache_allocation_pkv_or_rope"
    if isinstance(error, ResourceFitRejected):
        return "capture768_redesign_resource_fit_rejected", "allocator_cap_oom_or_compute_timeout"
    if isinstance(error, DiagnosticIncomplete):
        return "capture768_redesign_incomplete", "diagnostic_operation_or_evidence_incomplete"
    return "infrastructure_invalid", "runtime_checkpoint_host_asset_or_publication"


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


_COMPARISON_KEYS = {
    "comparison_index",
    "probe_index",
    "family",
    "modality",
    "name",
    "lhs",
    "rhs",
    "lhs_dtype",
    "rhs_dtype",
    "lhs_shape",
    "rhs_shape",
    "lhs_sha256",
    "rhs_sha256",
    "torch_equal",
    "element_count",
    "mismatch_count",
    "count_nonzero",
    "first_flat_mismatch",
    "metrics_defined",
    "max_abs",
    "rms_diff",
    "rhs_rms",
    "normalized_rms",
}


def validate_comparison(row: dict[str, object], expected: dict[str, object]) -> None:
    if set(row) != _COMPARISON_KEYS or any(row.get(key) != value for key, value in expected.items()):
        raise DiagnosticIncomplete("CAP768R comparison identity changed")
    for key in ("lhs_dtype", "rhs_dtype"):
        if not isinstance(row.get(key), str):
            raise DiagnosticIncomplete("CAP768R comparison dtype changed")
    for key in ("lhs_shape", "rhs_shape"):
        if not isinstance(row.get(key), list) or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in row[key]
        ):
            raise DiagnosticIncomplete("CAP768R comparison shape changed")
    for key in ("lhs_sha256", "rhs_sha256"):
        if not _SHA.fullmatch(str(row.get(key, ""))):
            raise DiagnosticIncomplete("CAP768R comparison tensor hash changed")
    count = row.get("element_count")
    mismatch = row.get("mismatch_count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or isinstance(mismatch, bool)
        or not isinstance(mismatch, int)
        or not 0 <= mismatch <= count
        or row.get("count_nonzero") != mismatch
        or not isinstance(row.get("torch_equal"), bool)
        or row["torch_equal"] is not (mismatch == 0)
        or (mismatch == 0 and row.get("first_flat_mismatch") is not None)
        or (
            mismatch > 0
            and (
                isinstance(row.get("first_flat_mismatch"), bool)
                or not isinstance(row.get("first_flat_mismatch"), int)
                or not 0 <= row["first_flat_mismatch"] < count
            )
        )
    ):
        raise DiagnosticIncomplete("CAP768R comparison counts changed")
    metrics = ("max_abs", "rms_diff", "rhs_rms", "normalized_rms")
    if row.get("metrics_defined") is True:
        if any(not _finite_number(row.get(key)) or row[key] < 0 for key in metrics):
            raise DiagnosticIncomplete("CAP768R comparison metric changed")
        if row["normalized_rms"] != row["rms_diff"] / max(row["rhs_rms"], 1e-12):
            raise DiagnosticIncomplete("CAP768R normalized RMS changed")
    elif row.get("metrics_defined") is False:
        if any(row.get(key) is not None for key in metrics):
            raise DiagnosticIncomplete("CAP768R undefined comparison metric changed")
    else:
        raise DiagnosticIncomplete("CAP768R comparison metric state changed")


def validate_receipt(receipt: dict[str, object], *, plan: dict[str, object]) -> None:
    """Fail-closed validation of the completed eight-case diagnostic receipt."""
    required = {
        "schema_version",
        "status",
        "plan_sha256",
        "mechanism_code_commit",
        "execution_commit",
        "asset_sha256",
        "run_id",
        "selection",
        "selection_sha256",
        "case_schedule",
        "case_schedule_sha256",
        "operation_schedule",
        "operation_schedule_sha256",
        "operation_counts",
        "flag_names",
        "flag_names_sha256",
        "gating_flag_names",
        "gating_flag_names_sha256",
        "descriptive_flag_names",
        "descriptive_flag_names_sha256",
        "comparison_schedule",
        "comparison_schedule_sha256",
        "cases",
        "probes",
        "flag0_incomplete_evidence",
        "versions",
        "runtime_sources",
        "static_guard",
        "rendering_preflight",
        "protected_hashes_before",
        "protected_hashes_after",
        "checkpoint_metadata_before",
        "checkpoint_metadata_after",
        "e33_state_tree_before",
        "e33_state_tree_after",
        "e33_parameters_frozen_no_grad",
        "worker_h176_loaded",
        "model_runtime",
        "no_cache_contract",
        "cache_guard",
        "memory_ledger",
        "memory_labels_sha256",
        "resources",
        "timings",
        "decision_boundary",
        "interpretation_boundary",
        "receipt_sha256",
    }
    cases = receipt.get("cases")
    probes = receipt.get("probes")
    if (
        set(receipt) != required
        or receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("plan_sha256") != plan.get("plan_sha256")
        or receipt.get("mechanism_code_commit") != plan.get("mechanism_code_commit")
        or receipt.get("asset_sha256") != plan.get("asset_sha256")
        or not _COMMIT.fullmatch(str(receipt.get("execution_commit", "")))
        or receipt.get("execution_commit") == receipt.get("mechanism_code_commit")
        or receipt.get("run_id") != AUTHORIZED_RUN_ID
        or receipt.get("selection") != SELECTION
        or receipt.get("selection_sha256") != SELECTION_SHA256
        or receipt.get("case_schedule") != CASE_SCHEDULE
        or receipt.get("case_schedule_sha256") != CASE_SCHEDULE_SHA256
        or receipt.get("operation_schedule") != build_operation_schedule()
        or receipt.get("operation_schedule_sha256") != OPERATION_SCHEDULE_SHA256
        or receipt.get("comparison_schedule") != build_comparison_schedule()
        or receipt.get("comparison_schedule_sha256") != COMPARISON_SCHEDULE_SHA256
        or receipt.get("flag_names") != FLAG_NAMES
        or receipt.get("flag_names_sha256") != FLAG_NAMES_SHA256
        or receipt.get("gating_flag_names") != GATING_FLAG_NAMES
        or receipt.get("gating_flag_names_sha256") != GATING_FLAG_NAMES_SHA256
        or receipt.get("descriptive_flag_names") != DESCRIPTIVE_FLAG_NAMES
        or receipt.get("descriptive_flag_names_sha256") != DESCRIPTIVE_FLAG_NAMES_SHA256
        or receipt.get("flag0_incomplete_evidence") != FLAG0_INCOMPLETE_EVIDENCE
        or receipt.get("decision_boundary") != DECISION_BOUNDARY
        or receipt.get("interpretation_boundary") != INTERPRETATION
        or not isinstance(cases, list)
        or len(cases) != 8
        or not isinstance(probes, list)
        or len(probes) != 4
        or receipt.get("receipt_sha256") != canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    ):
        raise DiagnosticIncomplete("CAP768R receipt identity changed")
    expected_comparisons = build_comparison_schedule()
    all_comparisons: list[dict[str, object]] = []
    all_timings: list[dict[str, object]] = []
    tensor_names = {
        "exact_embeddings",
        "L_ID_KEEP1.logits",
        "L_ID_KEEP1.hidden",
        "L_ID_KEEP1.capture",
        "L_E_KEEP1.logits",
        "L_E_KEEP1.hidden",
        "L_E_KEEP1.capture",
        "L_E_REPEAT_KEEP1.logits",
        "L_E_REPEAT_KEEP1.hidden",
        "L_E_REPEAT_KEEP1.capture",
        "L_ID_KEEP0_CONTROL.logits",
        "L_ID_KEEP0_CONTROL.last_logits",
        "L_ID_KEEP0_CONTROL.hidden",
        "L_ID_KEEP0_CONTROL.capture",
        "PROJ_ID1_LAST.logits",
        "PROJ_ID0_LAST.logits",
    }
    comparison_flags = {
        "id_embed_keep1_logits": "id_embed_keep1_logits_bitwise",
        "id_embed_keep1_full_hidden": "id_embed_keep1_full_hidden_bitwise",
        "id_embed_keep1_capture": "id_embed_keep1_capture_bitwise",
        "repeat_logits": "repeat_logits_bitwise",
        "repeat_full_hidden": "repeat_full_hidden_bitwise",
        "repeat_capture": "repeat_capture_bitwise",
        "keep0_keep1_full_hidden": "keep0_keep1_full_hidden_bitwise",
        "keep0_keep1_capture": "keep0_keep1_capture_bitwise",
        "keep0_last_logits_keep1": "keep0_last_logits_keep1_bitwise",
        "proj_id1_matches_id1_logits": "proj_id1_matches_id1_logits_bitwise",
        "proj_id0_matches_id0_last_logits": "proj_id0_matches_id0_last_logits_bitwise",
        "proj_id1_proj_id0": "proj_id1_proj_id0_bitwise",
        "id1_logits_proj_id0": "id1_logits_proj_id0_bitwise",
    }
    for case_index, (case, expected_case) in enumerate(zip(cases, CASE_SCHEDULE, strict=True)):
        required_case = {
            "probe_index",
            "family",
            "modality",
            "evidence_id",
            "query_id",
            "unpadded_tokens",
            "padded_tokens",
            "left_pad_tokens",
            "input_evidence",
            "tensor_evidence",
            "flags",
            "gating_flags_all_true",
            "qualifies",
            "comparisons",
            "operation_timings",
            "released",
        }
        flags = case.get("flags")
        comparisons = case.get("comparisons")
        selection = SELECTION[int(expected_case["probe_index"]) - 1]
        expected_unpadded = selection[
            "parent_unpadded_tokens" if expected_case["modality"] == "PARENT" else "mself_unpadded_tokens"
        ]
        if (
            set(case) != required_case
            or any(case.get(key) != value for key, value in expected_case.items())
            or case.get("padded_tokens") != 768
            or case.get("evidence_id") != selection["evidence_id"]
            or case.get("query_id") != selection["query_id"]
            or case.get("unpadded_tokens") != expected_unpadded
            or case.get("left_pad_tokens") != 768 - case.get("unpadded_tokens", 0)
            or not isinstance(flags, dict)
            or list(flags) != FLAG_NAMES
            or any(not isinstance(value, bool) for value in flags.values())
            or case.get("gating_flags_all_true") is not all(flags[name] for name in GATING_FLAG_NAMES)
            or case.get("qualifies") is not case.get("gating_flags_all_true")
            or case.get("released") is not True
            or not isinstance(comparisons, list)
            or len(comparisons) != 13
            or not isinstance(case.get("operation_timings"), list)
            or len(case["operation_timings"]) != 7
        ):
            raise DiagnosticIncomplete("CAP768R case evidence changed")
        input_evidence = case.get("input_evidence")
        if (
            not isinstance(input_evidence, dict)
            or set(input_evidence)
            != {
                "rendered_ids_shape",
                "rendered_ids_dtype",
                "rendered_ids_contiguous",
                "rendered_ids_sha256",
                "padded_ids_sha256",
                "attention_mask_sha256",
                "position_ids_sha256",
                "capture_mask_sha256",
                "capture_indices",
                "capture_shape",
            }
            or input_evidence.get("rendered_ids_shape") != [1, expected_unpadded]
            or input_evidence.get("rendered_ids_dtype") != "torch.int64"
            or input_evidence.get("rendered_ids_contiguous") is not True
            or input_evidence.get("capture_indices") != list(range(640, 768))
            or input_evidence.get("capture_shape") != [1, 128, 2048]
            or any(
                not _SHA.fullmatch(str(input_evidence.get(key, "")))
                for key in (
                    "rendered_ids_sha256",
                    "padded_ids_sha256",
                    "attention_mask_sha256",
                    "position_ids_sha256",
                    "capture_mask_sha256",
                )
            )
            or any(flags[name] is not True for name in FLAG_NAMES[:4])
        ):
            raise DiagnosticIncomplete("CAP768R input evidence changed")
        tensors = case.get("tensor_evidence")
        if not isinstance(tensors, dict) or set(tensors) != tensor_names:
            raise DiagnosticIncomplete("CAP768R tensor inventory changed")
        for name, tensor in tensors.items():
            expected_shape = (
                [1, 768, 2048]
                if name == "exact_embeddings" or name.endswith(".hidden")
                else [1, 128, 2048]
                if name.endswith(".capture")
                else [1, 768, 248320]
                if name == "L_ID_KEEP0_CONTROL.logits"
                else [1, 1, 248320]
            )
            if (
                not isinstance(tensor, dict)
                or set(tensor) != {"dtype", "shape", "sha256"}
                or tensor.get("dtype") != "torch.bfloat16"
                or tensor.get("shape") != expected_shape
                or not _SHA.fullmatch(str(tensor.get("sha256", "")))
            ):
                raise DiagnosticIncomplete("CAP768R tensor evidence changed")
        for row in comparisons:
            if (
                row["torch_equal"] is not flags[comparison_flags[row["name"]]]
                or row["lhs_dtype"] != tensors[row["lhs"]]["dtype"]
                or row["rhs_dtype"] != tensors[row["rhs"]]["dtype"]
                or row["lhs_shape"] != tensors[row["lhs"]]["shape"]
                or row["rhs_shape"] != tensors[row["rhs"]]["shape"]
                or row["lhs_sha256"] != tensors[row["lhs"]]["sha256"]
                or row["rhs_sha256"] != tensors[row["rhs"]]["sha256"]
            ):
                raise DiagnosticIncomplete("CAP768R comparison binding changed")
        expected_timings = build_operation_schedule()[case_index * 7 : case_index * 7 + 7]
        for timing, expected_timing in zip(case["operation_timings"], expected_timings, strict=True):
            if (
                set(timing) != set(expected_timing) | {"cuda_event_seconds", "wall_seconds"}
                or any(timing.get(key) != value for key, value in expected_timing.items())
                or any(
                    not _finite_number(timing.get(key)) or timing[key] < 0
                    for key in ("cuda_event_seconds", "wall_seconds")
                )
            ):
                raise DiagnosticIncomplete("CAP768R case timing changed")
        all_comparisons.extend(comparisons)
        all_timings.extend(case["operation_timings"])
    for row, expected in zip(all_comparisons, expected_comparisons, strict=True):
        validate_comparison(row, expected)
    for probe, selection in zip(probes, SELECTION, strict=True):
        matching = [case for case in cases if case["probe_index"] == probe.get("probe_index")]
        qualifies = len(matching) == 2 and all(case["qualifies"] for case in matching)
        if (
            set(probe) != {"probe_index", "family", "modalities_complete", "qualifies"}
            or probe.get("probe_index") != SELECTION.index(selection) + 1
            or probe.get("family") != selection["family"]
            or probe.get("modalities_complete") is not (len(matching) == 2)
            or probe.get("qualifies") is not qualifies
        ):
            raise DiagnosticIncomplete("CAP768R probe evidence changed")
    all_qualify = all(probe["qualifies"] for probe in probes)
    if receipt.get("status") != classification(all_qualify):
        raise DiagnosticIncomplete("CAP768R terminal status changed")
    expected_counts = {
        "embedding_lookup": 8,
        "e33_forward": 32,
        "lm_head_projection": 16,
        "capture": 32,
        "comparison": 104,
        "generation": 0,
        "h176_forward": 0,
        "bridge": 0,
        "optimizer": 0,
        "backward": 0,
        "step": 0,
        "checkpoint": 0,
        "candidate": 0,
    }
    if receipt.get("operation_counts") != expected_counts:
        raise DiagnosticIncomplete("CAP768R operation counts changed")
    if receipt.get("versions") != {
        key: _RUNTIME[key]
        for key in ("python", "transformers", "flash_linear_attention", "torch_distribution", "torch_runtime")
    }:
        raise DiagnosticIncomplete("CAP768R runtime versions changed")
    sources = receipt.get("runtime_sources")
    if not isinstance(sources, dict) or set(sources) != set(_RUNTIME["transformers_source_sha256"]):
        raise DiagnosticIncomplete("CAP768R runtime source inventory changed")
    for name, expected_sha in _RUNTIME["transformers_source_sha256"].items():
        source = sources[name]
        if (
            not isinstance(source, dict)
            or set(source) != {"path", "sha256"}
            or source.get("sha256") != expected_sha
            or not isinstance(source.get("path"), str)
        ):
            raise DiagnosticIncomplete("CAP768R runtime source changed")
    if receipt.get("static_guard") != {
        "runner_sha256": plan["asset_sha256"]["scripts/latent/run_a1_nc0_cap768_redesign_v1.py"],
        "forbidden_calls": [],
        "forbidden_identifiers": [],
        "forbidden_imports": [],
    }:
        raise DiagnosticIncomplete("CAP768R static guard changed")
    preflight = receipt.get("rendering_preflight")
    if (
        not isinstance(preflight, dict)
        or preflight.get("materialized_queries") != 288
        or preflight.get("feature_token_budget") != 768
        or preflight.get("maximum_unpadded_feature_tokens") != 644
        or preflight.get("feature_sequences_truncated") != 0
        or preflight.get("preflight_input_ids_extracted_from_batch_encoding") is not True
        or preflight.get("terminal_token_ids") != [248046, 198]
        or preflight.get("tokenizer_eos_token_id") != 248046
        or preflight.get("tokenizer_pad_token_id") != 248046
        or "label_alignment" in preflight
    ):
        raise DiagnosticIncomplete("CAP768R rendering preflight changed")
    no_cache = receipt.get("no_cache_contract")
    cache = receipt.get("cache_guard")
    if (
        no_cache
        != {
            "calls": 32,
            "use_cache_false": True,
            "pkv_input_none": True,
            "pkv_output_none": True,
            "rope_reset_every_call": True,
            "model_config_use_cache": False,
            "generation_config_use_cache": False,
        }
        or not isinstance(cache, dict)
        or cache.get("classes") != _CACHE_CLASS_CLOSURE
        or cache.get("negative_control_dynamic_cache_tripped") is not True
        or cache.get("closure_check_count") != 67
        or cache.get("restored_in_finally") is not True
    ):
        raise DiagnosticIncomplete("CAP768R cache evidence changed")
    ledger = receipt.get("memory_ledger")
    if not isinstance(ledger, list) or [row.get("label") for row in ledger] != memory_labels():
        raise DiagnosticIncomplete("CAP768R memory ledger changed")
    if receipt.get("memory_labels_sha256") != MEMORY_LABELS_SHA256:
        raise DiagnosticIncomplete("CAP768R memory hash changed")
    cap_bytes = RESOURCE_BOUNDS["allocator_cap_gib"] * 2**30
    peaks = (0, 0)
    for row in ledger:
        if set(row) != {"label", "allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes"}:
            raise DiagnosticIncomplete("CAP768R memory row changed")
        values = [row[key] for key in row if key.endswith("_bytes")]
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= cap_bytes for v in values):
            raise DiagnosticIncomplete("CAP768R memory value changed")
        if (
            row["peak_allocated_bytes"] < row["allocated_bytes"]
            or row["peak_reserved_bytes"] < row["reserved_bytes"]
            or row["peak_allocated_bytes"] < peaks[0]
            or row["peak_reserved_bytes"] < peaks[1]
        ):
            raise DiagnosticIncomplete("CAP768R memory peak changed")
        peaks = (row["peak_allocated_bytes"], row["peak_reserved_bytes"])
    if receipt.get("protected_hashes_before") != plan.get("protected_checkpoints") or receipt.get(
        "protected_hashes_after"
    ) != plan.get("protected_checkpoints"):
        raise DiagnosticIncomplete("CAP768R protected checkpoint changed")
    if receipt.get("checkpoint_metadata_before") != {
        "coordinator_e33": _RUNTIME["checkpoint_metadata_sha256"],
        "worker_h176": _RUNTIME["checkpoint_metadata_sha256"],
    } or receipt.get("checkpoint_metadata_after") != receipt.get("checkpoint_metadata_before"):
        raise DiagnosticIncomplete("CAP768R checkpoint metadata changed")
    if (
        receipt.get("e33_state_tree_after") != receipt.get("e33_state_tree_before")
        or receipt.get("e33_parameters_frozen_no_grad") is not True
        or receipt.get("worker_h176_loaded") is not False
    ):
        raise DiagnosticIncomplete("CAP768R protected runtime state changed")
    if receipt.get("model_runtime") != {
        "class": _RUNTIME["model_class"],
        "hidden_size": 2048,
        "vocab_size": 248320,
        "dtype": "torch.bfloat16",
        "device": "cuda:0",
    }:
        raise DiagnosticIncomplete("CAP768R model runtime changed")
    resources = receipt.get("resources")
    if (
        not isinstance(resources, dict)
        or resources.get("gpu_name") != RESOURCE_BOUNDS["gpu_model"]
        or resources.get("allocator_cap_bytes") != cap_bytes
        or resources.get("peak_allocated_bytes") != max(row["peak_allocated_bytes"] for row in ledger)
        or resources.get("peak_reserved_bytes") != max(row["peak_reserved_bytes"] for row in ledger)
        or resources.get("cuda_visible_devices") != "0"
        or resources.get("network_disabled") != {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        or resources.get("physical_gpu1_unused_before_after") is not True
    ):
        raise DiagnosticIncomplete("CAP768R resource evidence changed")
    for key, minimum in (
        ("total_gpu_memory_bytes", RESOURCE_BOUNDS["minimum_gpu_memory_gib"]),
        ("host_ram_bytes", RESOURCE_BOUNDS["minimum_host_ram_gib"]),
        ("free_disk_bytes_preflight", RESOURCE_BOUNDS["minimum_free_disk_gib"]),
    ):
        if (
            isinstance(resources.get(key), bool)
            or not isinstance(resources.get(key), int)
            or resources[key] < minimum * 2**30
        ):
            raise DiagnosticIncomplete("CAP768R resource floor changed")
    timings = receipt.get("timings")
    if (
        not isinstance(timings, dict)
        or set(timings)
        != {
            "operations",
            "operation_cuda_event_seconds_sum",
            "operation_wall_seconds_sum",
            "tokenizer_load_seconds",
            "model_load_seconds",
            "compute_seconds",
            "audit_seconds",
            "total_seconds",
        }
        or timings.get("operations") != all_timings
        or len(timings["operations"]) != 56
    ):
        raise DiagnosticIncomplete("CAP768R timing evidence changed")
    for observed, expected in zip(timings["operations"], build_operation_schedule(), strict=True):
        if (
            set(observed) != set(expected) | {"cuda_event_seconds", "wall_seconds"}
            or any(observed.get(k) != v for k, v in expected.items())
            or any(
                not _finite_number(observed.get(k)) or observed[k] < 0 for k in ("cuda_event_seconds", "wall_seconds")
            )
        ):
            raise DiagnosticIncomplete("CAP768R operation timing changed")
    if timings.get("operation_cuda_event_seconds_sum") != math.fsum(
        row["cuda_event_seconds"] for row in timings["operations"]
    ) or timings.get("operation_wall_seconds_sum") != math.fsum(row["wall_seconds"] for row in timings["operations"]):
        raise DiagnosticIncomplete("CAP768R timing aggregate changed")
    for key in (
        "tokenizer_load_seconds",
        "model_load_seconds",
        "compute_seconds",
        "audit_seconds",
        "total_seconds",
    ):
        if not _finite_number(timings.get(key)) or timings[key] < 0:
            raise DiagnosticIncomplete("CAP768R timing scalar changed")
    if (
        timings["compute_seconds"] > RESOURCE_BOUNDS["compute_seconds"]
        or timings["audit_seconds"] > RESOURCE_BOUNDS["audit_seconds"]
    ):
        raise DiagnosticIncomplete("CAP768R phase timing exceeded")
