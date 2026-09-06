#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

BASELINE_COMMIT = "b5cdb53fb14523d9adbb2c3f6f01458a488edcfd"
MODULE_PATH = "src/prime_rl/latent/h_iter_phase0.py"
RUNNER_PATH = "scripts/latent/run_h_iter_phase0_generator_locality_v1.py"
SCHEMA_VERSION = "prime-rl/latent-h-iter-phase0-scientific-surface/v1"

CONSTANT_NAMES = [
    "MECHANISM", "BANK_SCHEMA", "RECEIVER_SCHEMA", "SUPERVISION_SCHEMA",
    "PROBE_SCHEMA", "OVERLAP_SCHEMA", "OPERATION_SCHEMA", "TAMPER_SCHEMA",
    "ARTIFACT_DIR_REL", "NODE_COUNT", "FEATURE_DIM", "MARKERS", "ACTIONS",
    "SPLITS", "BANK_PAYLOADS", "ORDER_PAYLOADS", "EXPECTED_PAYLOAD_SHA256",
    "EXPECTED_PAYLOAD_SEEDS", "EXPECTED_ORDER_SHA256", "EXPECTED_ORDER_SEEDS",
    "PROBE_PAYLOAD", "PROBE_PAYLOAD_SHA256", "PROBE_SEED", "PERTURB_VECTOR",
    "ARM_NAMES", "MECHANISM_TAMPERS", "_ROW_KEYS", "_BANK_KEYS", "_TOKEN",
    "_IDENTITY_IN_TEXT",
]

FUNCTION_NAMES = [
    "strict_json_loads", "canonical_json", "sha256_bytes", "canonical_sha256",
    "seed_u64", "_digest", "_shuffle", "_node_id", "_nonce", "_local_text",
    "generate_row", "_finish_row", "generate_bank", "row_ring",
    "marker_from_local_text", "nonce_from_local_text", "donor_for_row",
    "_assert_answer_free", "validate_bank", "validate_banks",
    "build_probe_selection", "build_operation_schedule", "build_tamper_schedule",
    "iter_strings", "_receiver_candidates", "extract_prior_source",
    "new_identity_sets", "validate_no_threshold_fields", "validate_schedule",
    "expected_symbolic_counts", "finite_float", "validate_probe_selection",
    "_tensor_sha256", "_encode_graph", "_transition", "_arm_state", "_readout",
    "_radius", "_distance_indices", "run_locality_probe",
    "run_symbolic_dependency_audit", "run_all_locality_probes",
    "validate_locality_policy", "locality_policy", "validate_locality_evidence",
    "_rehash_row", "_rehash_bank", "run_mechanism_tamper_audit",
]

SCIENTIFIC_ASSET_PATHS = [
    "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/heldout-bank.json",
    "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/locality-probe-selection.json",
    "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/operation-schedule.json",
    "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/overlap-evidence.json",
    "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/tamper-schedule.json",
    "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/train-bank.json",
    "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/validation-bank.json",
    "scripts/latent/materialize_h_iter_phase0_assets_v1.py",
]

RECOVERY_SOURCE_ALLOWLIST = {
    "src/prime_rl/latent/h_iter_phase0.py": [
        "RUN_IDENTITY", "RESOURCE_BOUNDS", "DECISION_BOUNDARY",
        "PREEXECUTION_EVIDENCE_SCHEMA", "PREEXECUTION_EVIDENCE_SHA256",
        "validate_preexecution_evidence", "validate_terminal_entry_timing",
        "validate_phase_records", "validate_plan", "validate_proof", "validate_failure",
        "new_recovery_identity_or_evidence_constants",
        "new_recovery_antecedent_or_robustness_validators",
    ],
    "scripts/latent/run_h_iter_phase0_generator_locality_v1.py": [
        "imports_needed_only_for_census_or_recovery_evidence",
        "InfrastructureInvalid", "PLAN_RELATIVE_PATH", "PLAN_SIDECAR_RELATIVE_PATH",
        "PhaseTracker", "NetworkGuard", "ArtifactWriter", "MemoryLedger",
        "read_regular_file_bytes", "load_authorized_plan", "asset_hashes",
        "validate_preexecution_assets", "runtime_evidence", "static_forbidden_sites",
        "object_inventory", "safety_evidence", "host_resources", "receipt_tamper_audit",
        "parse_args", "run_outside_frozen_22_statement_scientific_slice",
        "failure_payload", "main", "new_recovery_evidence_helpers",
    ],
    "scripts/latent/run_h_iter_phase0_generator_locality_v1.sh": ["entire_file_contract_review_required"],
    "tests/unit/latent/test_h_iter_phase0.py": ["entire_file_test_review_required"],
    "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/phase0-plan.json": ["entire_file_contract_review_required"],
    "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/phase0-plan.sha256": ["entire_file_contract_review_required"],
    "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/recovery_evidence_assets": ["new_files_only_exact_hash_review_required"],
}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def git_blob(repo: Path, source_commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{source_commit}:{path}"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def ast_dump(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False, indent=None)


def assignment_name(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def is_runner_end(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "ledger"
        and node.value.func.attr == "checkpoint"
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == "mechanism_tampers_validated"
    )


def build(repo: Path, source_commit: str = BASELINE_COMMIT) -> dict[str, object]:
    module_tree = ast.parse(git_blob(repo, source_commit, MODULE_PATH).decode("utf-8"))
    assignments = {
        name: node
        for node in module_tree.body
        if (name := assignment_name(node)) is not None
    }
    functions = {
        node.name: node
        for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if any(name not in assignments for name in CONSTANT_NAMES):
        raise RuntimeError("scientific constant selection is incomplete")
    if any(name not in functions for name in FUNCTION_NAMES):
        raise RuntimeError("scientific function selection is incomplete")

    runner_tree = ast.parse(git_blob(repo, source_commit, RUNNER_PATH).decode("utf-8"))
    runner_matches = [
        node
        for node in runner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    ]
    if len(runner_matches) != 1:
        raise RuntimeError("runner run() selector is not singular")
    runner = runner_matches[0]
    starts = [
        index
        for index, node in enumerate(runner.body)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "banks"
    ]
    ends = [index for index, node in enumerate(runner.body) if is_runner_end(node)]
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise RuntimeError("runner scientific slice selector differs")
    statements = runner.body[starts[0] : ends[0] + 1]
    if len(statements) != 22:
        raise RuntimeError("runner scientific slice statement count differs")

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_commit": BASELINE_COMMIT,
        "canonicalizer": {
            "python": "3.12.14",
            "parser": "ast.parse_utf8_source",
            "node_encoding": "ast.dump(annotate_fields=True,include_attributes=False,indent=None)",
            "payload_encoding": "json.dumps(sort_keys=True,separators=(comma,colon),ensure_ascii=False,allow_nan=False).encode(utf-8)",
            "hash": "sha256",
            "self_hash_omits_exact_top_level_key": "surface_sha256",
        },
        "module_path": MODULE_PATH,
        "constant_names": CONSTANT_NAMES,
        "function_names": FUNCTION_NAMES,
        "module_constants": [
            {"name": name, "ast": ast_dump(assignments[name])}
            for name in CONSTANT_NAMES
        ],
        "module_functions": [
            {"name": name, "ast": ast_dump(functions[name])}
            for name in FUNCTION_NAMES
        ],
        "runner_path": RUNNER_PATH,
        "runner_block": {
            "function": "run",
            "start_selector": "first_top_level_Assign_single_Name_target_banks",
            "end_selector": "unique_top_level_Expr_ledger.checkpoint_constant_mechanism_tampers_validated_inclusive",
            "statement_count": 22,
            "statements": [ast_dump(node) for node in statements],
        },
        "scientific_asset_paths": SCIENTIFIC_ASSET_PATHS,
        "scientific_asset_sha256": {
            path: hashlib.sha256(git_blob(repo, source_commit, path)).hexdigest()
            for path in SCIENTIFIC_ASSET_PATHS
        },
        "historical_source_commits": {
            "a_lane": "a8f347c9a5fdf1c2d532c6527ce169cff0000a07",
            "b_lane": "4ae0308094a71d13520554da40cfe6375438b610",
        },
        "recovery_source_allowlist": RECOVERY_SOURCE_ALLOWLIST,
        "surface_sha256": "",
    }
    payload["surface_sha256"] = hashlib.sha256(
        canonical_json({key: value for key, value in payload.items() if key != "surface_sha256"})
    ).hexdigest()
    return payload


def main() -> None:
    if len(sys.argv) not in {3, 4}:
        raise SystemExit(f"usage: {sys.argv[0]} <repo> <output-json> [source-commit]")
    repo = Path(sys.argv[1]).resolve(strict=True)
    output = Path(sys.argv[2])
    source_commit = BASELINE_COMMIT if len(sys.argv) == 3 else sys.argv[3]
    payload = build(repo, source_commit)
    encoded = canonical_json(payload) + b"\n"
    output.write_bytes(encoded)
    print(payload["surface_sha256"])
    print(hashlib.sha256(encoded).hexdigest())


if __name__ == "__main__":
    main()
