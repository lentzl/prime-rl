#!/usr/bin/env python3
"""Model-free exact-host proof for IPC render-proof canonicalization hygiene."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from prime_rl.phase_b_contract import (
    PhaseBContractError,
    canonical_json_sha256,
    file_sha256,
    load_json_file,
)
from prime_rl.phase_b_ipc1 import canonical_bank_sha256, canonical_terminal_bytes, strict_json_loads

WORKTREE = Path("/home/ubuntu/rlm/worktrees/q35-2b-recurrent-sidecar-v1")
EXPERIMENT = WORKTREE / "experiments/qwen35-2b-latent-coordinator-v1"
DEFAULT_PLAN = EXPERIMENT / "phase-b-ipc1-render-proof-hygiene-v1-plan.json"
DEFAULT_FREEZE = EXPERIMENT / "phase-b-ipc1-render-proof-hygiene-v1.sha256"
EXPECTED_ENV = Path("/home/ubuntu/rlm/prime-rl/.venv")
EXPECTED_PYTHONPATH = f"{WORKTREE}/src:{WORKTREE}/packages/prime-rl-configs/src"
SUCCESS_STATUS = "ipc1_render_proof_hygiene_validated"
INCOMPLETE_STATUS = "ipc1_render_proof_hygiene_incomplete"
INFRASTRUCTURE_STATUS = "infrastructure_invalid"
SPLITS = (("train", 48), ("validation", 24), ("heldout", 24))
TAMPERS = (
    "unpatched_utf8_source_hash_train",
    "unpatched_utf8_source_hash_validation",
    "unpatched_utf8_source_hash_heldout",
    "wrong_source_hash",
    "missing_proof_key",
    "extra_proof_key",
    "mutation_non_source_field",
    "row_reorder",
    "row_deletion",
    "row_duplication",
    "split_name_change",
    "split_order_change",
)


class HygieneInfrastructureError(RuntimeError):
    """Raised for a provenance, resource, or execution-environment failure."""


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--authorized-plan-sha256", required=True)
    parser.add_argument("--authorized-freeze-sha256", required=True)
    return parser.parse_args()


def _canonical_plan_sha256(plan: dict[str, Any]) -> str:
    payload = dict(plan)
    payload.pop("plan_sha256", None)
    return canonical_bank_sha256(payload)


def _require_keys(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise PhaseBContractError(f"IPC render hygiene {label} keyset differs")
    return value


def _load_plan(args: argparse.Namespace) -> dict[str, Any]:
    if not args.plan.is_absolute() or not args.freeze_manifest.is_absolute() or not args.output_dir.is_absolute():
        raise HygieneInfrastructureError("IPC render hygiene requires absolute paths")
    if file_sha256(args.plan) != args.authorized_plan_sha256:
        raise HygieneInfrastructureError("IPC render hygiene plan authorization differs")
    if file_sha256(args.freeze_manifest) != args.authorized_freeze_sha256:
        raise HygieneInfrastructureError("IPC render hygiene freeze authorization differs")
    plan = load_json_file(args.plan)
    _require_keys(
        plan,
        {
            "schema_version",
            "mechanism",
            "status",
            "mechanism_commit",
            "plan_sha256",
            "execution_authorization",
            "source_ipc1",
            "prior_failure",
            "protected_model",
            "banks",
            "expected_evidence",
            "resources",
            "execution_environment",
            "outputs",
            "boundaries",
        },
        label="plan",
    )
    if (
        plan["schema_version"] != "q35-2b-ipc1-render-proof-hygiene-plan/v1"
        or plan["mechanism"] != "b-ipc1-render-proof-hygiene-v1"
        or plan["plan_sha256"] != _canonical_plan_sha256(plan)
        or plan["mechanism_commit"] != subprocess.run(
            ["git", "rev-parse", f"{args.execution_commit}^"],
            cwd=WORKTREE,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    ):
        raise HygieneInfrastructureError("IPC render hygiene plan identity differs")
    if (
        plan["resources"]
        != {
            "outer_wall_clock_seconds": 600,
            "compute_limit_seconds": 480,
            "failure_audit_limit_seconds": 90,
            "terminal_publication_headroom_seconds": 30,
            "minimum_available_ram_bytes": 8 * 1024**3,
            "minimum_free_disk_bytes": 8 * 1024**3,
            "artifact_cap_bytes": 64 * 1024**2,
        }
        or plan["execution_environment"]
        != {
            "UV_PROJECT_ENVIRONMENT": str(EXPECTED_ENV),
            "PYTHONPATH": EXPECTED_PYTHONPATH,
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        or plan["boundaries"]
        != {
            "evidence_only": True,
            "candidate_reuse": False,
            "ipc1_rerun": False,
            "nomination": False,
            "admission": False,
            "promotion": False,
            "revalidation": False,
            "model_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "updates": False,
        }
        or plan["expected_evidence"]
        != {
            "split_order": ["train", "validation", "heldout"],
            "row_counts": [48, 24, 24],
            "total_rows": 96,
            "authoritative_bank_hash_matches": 96,
            "inherited_utf8_hash_matches": 96,
            "inherited_vs_bank_mismatches": 96,
            "overwrite_only_matches": 96,
            "post_repair_validator_matches": 96,
            "tamper_cases": list(TAMPERS),
        }
    ):
        raise HygieneInfrastructureError("IPC render hygiene frozen contract differs")
    plan["_path"] = str(args.plan)
    plan["_file_sha256"] = args.authorized_plan_sha256
    plan["_freeze_path"] = str(args.freeze_manifest)
    plan["_freeze_sha256"] = args.authorized_freeze_sha256
    return plan


def _full_freeze(plan: dict[str, Any], *, execution_commit: str) -> dict[str, Any]:
    if subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WORKTREE, capture_output=True, check=True, text=True
    ).stdout.strip() != execution_commit:
        raise HygieneInfrastructureError("IPC render hygiene execution commit differs")
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=WORKTREE,
        capture_output=True,
        check=True,
        text=True,
    ).stdout:
        raise HygieneInfrastructureError("IPC render hygiene worktree is dirty")
    root = Path(plan["_freeze_path"]).parent.resolve()
    records: list[dict[str, Any]] = []
    for position, line in enumerate(Path(plan["_freeze_path"]).read_text().splitlines()):
        digest, relative = line.split("  ", 1)
        target = (root / relative).resolve()
        if target.is_symlink() or not target.is_file() or WORKTREE.resolve() not in target.parents:
            raise HygieneInfrastructureError("IPC render hygiene freeze target is unsafe")
        observed = file_sha256(target)
        if observed != digest:
            raise HygieneInfrastructureError(f"IPC render hygiene freeze target differs: {relative}")
        records.append(
            {"position": position, "path": relative, "expected_sha256": digest, "observed_sha256": observed}
        )
    if not records:
        raise HygieneInfrastructureError("IPC render hygiene freeze is empty")
    return {
        "execution_commit": execution_commit,
        "clean": True,
        "target_count": len(records),
        "targets": records,
        "targets_canonical_sha256": canonical_bank_sha256(records),
        "sidecar_sha256": plan["_freeze_sha256"],
    }


def _validate_prior_failure(plan: dict[str, Any]) -> dict[str, Any]:
    record = plan["prior_failure"]
    binding = load_json_file(Path(record["binding_path"]))
    if file_sha256(Path(record["binding_path"])) != record["binding_sha256"]:
        raise HygieneInfrastructureError("IPC render hygiene prior binding differs")
    for path_key, hash_key in (
        ("failure_path", "failure_file_sha256"),
        ("run_log_path", "run_log_sha256"),
        ("archive_path", "archive_sha256"),
        ("snapshot_manifest_path", "snapshot_manifest_sha256"),
    ):
        if record[path_key] != binding[path_key] or file_sha256(Path(binding[path_key])) != binding[hash_key]:
            raise HygieneInfrastructureError(f"IPC render hygiene prior {path_key} differs")
    failure = load_json_file(Path(binding["failure_path"]))
    unhashed = {key: value for key, value in failure.items() if key != "receipt_sha256"}
    if (
        failure.get("receipt_sha256") != binding["failure_internal_sha256"]
        or hashlib.sha256(canonical_terminal_bytes(unhashed)).hexdigest() != binding["failure_internal_sha256"]
        or failure.get("status") != "b_ipc1_incomplete"
        or failure.get("candidate_files_valid") is not False
        or failure.get("post_failure_audit", {}).get("execution_progress", {}).get("schedule_name")
        != "validation_reject"
    ):
        raise HygieneInfrastructureError("IPC render hygiene prior FAILURE differs")
    return {
        "failure_file_sha256": binding["failure_file_sha256"],
        "failure_internal_sha256": binding["failure_internal_sha256"],
        "classification": "valid_b_ipc1_incomplete_validation_reject_control_flow_only",
        "candidate_reuse": False,
        "rerun_authorized": False,
    }


def _resource_preflight(plan: dict[str, Any], output_dir: Path) -> dict[str, int]:
    if output_dir.exists() or output_dir.is_symlink():
        raise HygieneInfrastructureError("IPC render hygiene output namespace is not fresh")
    available_ram = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    free_disk = shutil.disk_usage(output_dir.parent).free
    if available_ram < plan["resources"]["minimum_available_ram_bytes"]:
        raise HygieneInfrastructureError("IPC render hygiene host RAM is insufficient")
    if free_disk < plan["resources"]["minimum_free_disk_bytes"]:
        raise HygieneInfrastructureError("IPC render hygiene disk is insufficient")
    return {"available_ram_bytes": available_ram, "free_disk_bytes": free_disk}


def _expect_rejected(runtime: Any, candidate: dict[str, Any], source_plan: dict[str, Any]) -> None:
    try:
        runtime._validate_render_proofs(candidate, plan=source_plan, heldout_open=True)
    except PhaseBContractError:
        return
    raise PhaseBContractError("IPC render hygiene tamper unexpectedly validated")


def _produce_evidence(plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    import pyarrow.parquet as parquet
    import torch
    from transformers import AutoTokenizer

    if torch.cuda.is_initialized():
        raise PhaseBContractError("CUDA initialized before IPC render hygiene proof")
    runtime = __import__("run_phase_b_ipc1_matched_learning_v1")
    source_plan = load_json_file(Path(plan["source_ipc1"]["plan_path"]))
    source_plan["_path"] = plan["source_ipc1"]["plan_path"]
    source_plan["_file_sha256"] = plan["source_ipc1"]["plan_sha256"]
    smoke = runtime._load_module(runtime.BR5_RUNNER, "ipc_render_hygiene_smoke")
    b1 = runtime._load_module(runtime.B1_RUNNER, "ipc_render_hygiene_b1")
    tokenizer = AutoTokenizer.from_pretrained(plan["protected_model"]["path"], local_files_only=True)
    context: dict[str, Any] = {"selections": {}, "paths": {}}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    proof_records: list[dict[str, Any]] = []
    split_evidence: list[dict[str, Any]] = []
    for split, expected_count in SPLITS:
        bank = next(record for record in plan["banks"] if record["name"] == split)
        context["selections"][split] = load_json_file(Path(bank["selection_path"]))
        context["paths"][f"{split}_parquet"] = Path(bank["parquet_path"])
        rows = runtime._ordered_rows(parquet, context["paths"][f"{split}_parquet"], context["selections"][split])
        all_rows[split] = rows
        rendered, proofs, hygiene = runtime._render_split(
            split, context, tokenizer=tokenizer, parquet=parquet, b1=b1, smoke=smoke
        )
        if len(rendered) != expected_count or hygiene != {
            "name": split,
            "row_count": expected_count,
            "authoritative_bank_hash_matches": expected_count,
            "inherited_utf8_hash_matches": expected_count,
            "inherited_vs_bank_mismatches": expected_count,
            "overwrite_only_matches": expected_count,
            "post_repair_validator_matches": expected_count,
        }:
            raise PhaseBContractError(f"IPC render hygiene {split} evidence differs")
        proof_records.append({"name": split, "rows": proofs})
        split_evidence.append(
            {
                **hygiene,
                "ordered_proof_sha256": canonical_bank_sha256(proofs),
                "ordered_length_sha256": canonical_bank_sha256(
                    [
                        {
                            "plain_tokens": proof["plain_tokens"],
                            "opening_tokens": proof["opening_tokens"],
                            "full_tokens": proof["full_tokens"],
                        }
                        for proof in proofs
                    ]
                ),
            }
        )
    ephemeral = {"render_proofs": proof_records}
    runtime._validate_render_proofs(ephemeral, plan=source_plan, heldout_open=True)
    canonical = canonical_terminal_bytes(ephemeral)
    parsed = strict_json_loads(canonical)
    runtime._validate_render_proofs(parsed, plan=source_plan, heldout_open=True)
    if canonical_terminal_bytes(parsed) != canonical:
        raise PhaseBContractError("IPC render hygiene canonical roundtrip differs")
    tampers: list[str] = []
    for split_index, split in enumerate(("train", "validation", "heldout")):
        candidate = deepcopy(parsed)
        candidate["render_proofs"][split_index]["rows"][0]["source_row_sha256"] = canonical_json_sha256(
            all_rows[split][0]
        )
        _expect_rejected(runtime, candidate, source_plan)
        tampers.append(f"unpatched_utf8_source_hash_{split}")
    mutations = [
        ("wrong_source_hash", lambda item: item["render_proofs"][0]["rows"][0].__setitem__("source_row_sha256", "0" * 64)),
        ("missing_proof_key", lambda item: item["render_proofs"][0]["rows"][0].pop("action_trie_sha256")),
        ("extra_proof_key", lambda item: item["render_proofs"][0]["rows"][0].__setitem__("unexpected", True)),
        ("mutation_non_source_field", lambda item: item["render_proofs"][0]["rows"][0].__setitem__("action", "delegate_terminal")),
        ("row_reorder", lambda item: item["render_proofs"][0]["rows"].__setitem__(slice(0, 2), list(reversed(item["render_proofs"][0]["rows"][:2])))),
        ("row_deletion", lambda item: item["render_proofs"][0]["rows"].pop()),
        ("row_duplication", lambda item: item["render_proofs"][0]["rows"].append(deepcopy(item["render_proofs"][0]["rows"][-1]))),
        ("split_name_change", lambda item: item["render_proofs"][0].__setitem__("name", "heldout")),
        ("split_order_change", lambda item: item["render_proofs"].__setitem__(slice(0, 2), list(reversed(item["render_proofs"][:2])))),
    ]
    for name, mutate in mutations:
        candidate = deepcopy(parsed)
        mutate(candidate)
        _expect_rejected(runtime, candidate, source_plan)
        tampers.append(name)
    if tampers != list(TAMPERS):
        raise PhaseBContractError("IPC render hygiene tamper order differs")
    if torch.cuda.is_initialized() or any(name.endswith("modeling_qwen3_5") for name in sys.modules):
        raise PhaseBContractError("model or CUDA initialized during IPC render hygiene proof")
    return (
        {
            "split_evidence": split_evidence,
            "total_rows": 96,
            "canonical_roundtrip_payload_sha256": hashlib.sha256(canonical).hexdigest(),
            "canonical_roundtrip_byte_equal": True,
            "validator_passed_before_and_after_roundtrip": True,
            "tamper_cases_rejected": tampers,
            "heldout_auditor_opened": True,
            "heldout_candidate_or_model_opened": False,
            "heldout_scientific_evaluation": False,
            "heldout_threshold_or_selection_use": False,
        },
        {"model_loaded": False, "cuda_initialized": False},
    )


def _validate_proof(receipt: dict[str, Any], *, plan: dict[str, Any], execution_commit: str) -> None:
    _require_keys(
        receipt,
        {
            "schema_version",
            "terminal",
            "status",
            "mechanism",
            "execution_commit",
            "plan_sha256",
            "prior_failure",
            "evidence",
            "safety",
            "resources",
            "full_freeze_pre",
            "full_freeze_post",
            "boundaries",
            "elapsed_seconds",
            "receipt_sha256",
        },
        label="PROOF",
    )
    unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    evidence = receipt["evidence"]
    if (
        receipt["schema_version"] != "q35-2b-ipc1-render-proof-hygiene-proof/v1"
        or receipt["terminal"] != "PROOF"
        or receipt["status"] != SUCCESS_STATUS
        or receipt["mechanism"] != "b-ipc1-render-proof-hygiene-v1"
        or receipt["execution_commit"] != execution_commit
        or receipt["plan_sha256"] != plan["_file_sha256"]
        or receipt["receipt_sha256"] != hashlib.sha256(canonical_terminal_bytes(unhashed)).hexdigest()
        or [record["name"] for record in evidence["split_evidence"]] != [name for name, _ in SPLITS]
        or [record["row_count"] for record in evidence["split_evidence"]] != [count for _, count in SPLITS]
        or any(
            record[key] != record["row_count"]
            for record in evidence["split_evidence"]
            for key in (
                "authoritative_bank_hash_matches",
                "inherited_utf8_hash_matches",
                "inherited_vs_bank_mismatches",
                "overwrite_only_matches",
                "post_repair_validator_matches",
            )
        )
        or evidence["tamper_cases_rejected"] != list(TAMPERS)
        or evidence["total_rows"] != 96
        or receipt["safety"]
        != {
            "CUDA_VISIBLE_DEVICES": "",
            "model_loaded": False,
            "cuda_initialized": False,
            "model_forward_count": 0,
            "backward_count": 0,
            "optimizer_steps": 0,
            "candidate_files": [],
            "update": False,
        }
        or receipt["full_freeze_pre"] != receipt["full_freeze_post"]
        or not isinstance(receipt["elapsed_seconds"], (int, float))
        or not math.isfinite(receipt["elapsed_seconds"])
    ):
        raise PhaseBContractError("IPC render hygiene PROOF differs")
    for key in (
        "heldout_auditor_opened",
        "canonical_roundtrip_byte_equal",
        "validator_passed_before_and_after_roundtrip",
    ):
        if evidence[key] is not True:
            raise PhaseBContractError("IPC render hygiene positive predicate differs")
    for key in (
        "heldout_candidate_or_model_opened",
        "heldout_scientific_evaluation",
        "heldout_threshold_or_selection_use",
    ):
        if evidence[key] is not False:
            raise PhaseBContractError("IPC render hygiene heldout boundary differs")
    if receipt["boundaries"] != plan["boundaries"]:
        raise PhaseBContractError("IPC render hygiene boundaries differ")


def _validate_failure(receipt: dict[str, Any], *, execution_commit: str) -> None:
    _require_keys(
        receipt,
        {
            "schema_version",
            "terminal",
            "status",
            "failure_class",
            "error_type",
            "error",
            "execution_commit",
            "plan_sha256",
            "candidate_files",
            "model_loaded",
            "cuda_initialized",
            "update",
            "elapsed_seconds",
            "receipt_sha256",
        },
        label="FAILURE",
    )
    unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    expected_class = (
        "infrastructure" if receipt["status"] == INFRASTRUCTURE_STATUS else "diagnostic_or_schema_incomplete"
    )
    if (
        receipt["schema_version"] != "q35-2b-ipc1-render-proof-hygiene-failure/v1"
        or receipt["terminal"] != "FAILURE"
        or receipt["status"] not in (INCOMPLETE_STATUS, INFRASTRUCTURE_STATUS)
        or receipt["failure_class"] != expected_class
        or receipt["execution_commit"] != execution_commit
        or receipt["candidate_files"] != []
        or receipt["model_loaded"] is not False
        or receipt["cuda_initialized"] is not False
        or receipt["update"] is not False
        or receipt["receipt_sha256"] != hashlib.sha256(canonical_terminal_bytes(unhashed)).hexdigest()
        or not isinstance(receipt["elapsed_seconds"], (int, float))
        or not math.isfinite(receipt["elapsed_seconds"])
    ):
        raise PhaseBContractError("IPC render hygiene FAILURE differs")


def _publish(output: Path, filename: str, payload: bytes) -> Path:
    output.mkdir(mode=0o700)
    existing = list(output.iterdir())
    if existing:
        raise FileExistsError("IPC render hygiene terminal namespace is not exclusive")
    temporary = output / f".{filename}.tmp"
    terminal = output / filename
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, terminal)
    directory_fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return terminal


def _failure_status(error: BaseException) -> tuple[str, str]:
    if isinstance(error, HygieneInfrastructureError) or isinstance(error, TimeoutError):
        return INFRASTRUCTURE_STATUS, "infrastructure"
    return INCOMPLETE_STATUS, "diagnostic_or_schema_incomplete"


def main() -> int:
    args = _args()
    started = time.time()
    plan: dict[str, Any] | None = None
    try:
        signal.signal(signal.SIGALRM, lambda _signum, _frame: (_ for _ in ()).throw(TimeoutError("compute timeout")))
        signal.alarm(480)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise HygieneInfrastructureError("IPC render hygiene offline/CUDA environment differs")
        if Path(sys.prefix).resolve() != EXPECTED_ENV.resolve() or os.environ.get("PYTHONPATH") != EXPECTED_PYTHONPATH:
            raise HygieneInfrastructureError("IPC render hygiene Python environment differs")
        plan = _load_plan(args)
        resources = _resource_preflight(plan, args.output_dir)
        prior = _validate_prior_failure(plan)
        pre = _full_freeze(plan, execution_commit=args.execution_commit)
        evidence, _model_state = _produce_evidence(plan)
        post = _full_freeze(plan, execution_commit=args.execution_commit)
        receipt = {
            "schema_version": "q35-2b-ipc1-render-proof-hygiene-proof/v1",
            "terminal": "PROOF",
            "status": SUCCESS_STATUS,
            "mechanism": "b-ipc1-render-proof-hygiene-v1",
            "execution_commit": args.execution_commit,
            "plan_sha256": plan["_file_sha256"],
            "prior_failure": prior,
            "evidence": evidence,
            "safety": {
                "CUDA_VISIBLE_DEVICES": "",
                "model_loaded": False,
                "cuda_initialized": False,
                "model_forward_count": 0,
                "backward_count": 0,
                "optimizer_steps": 0,
                "candidate_files": [],
                "update": False,
            },
            "resources": resources,
            "full_freeze_pre": pre,
            "full_freeze_post": post,
            "boundaries": plan["boundaries"],
            "elapsed_seconds": time.time() - started,
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical_terminal_bytes(receipt)).hexdigest()
        payload = canonical_terminal_bytes(receipt)
        if len(payload) > plan["resources"]["artifact_cap_bytes"]:
            raise HygieneInfrastructureError("IPC render hygiene artifact cap exceeded")
        parsed = strict_json_loads(payload)
        _validate_proof(parsed, plan=plan, execution_commit=args.execution_commit)
        signal.alarm(0)
        terminal = _publish(args.output_dir, "PROOF.json", payload)
        if terminal.read_bytes() != payload:
            raise RuntimeError("IPC render hygiene published bytes differ")
        _validate_proof(strict_json_loads(terminal.read_bytes()), plan=plan, execution_commit=args.execution_commit)
        return 0
    except BaseException as error:
        signal.alarm(0)
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            print(f"IPC render hygiene failed after terminal publication: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
        status, failure_class = _failure_status(error)
        failure = {
            "schema_version": "q35-2b-ipc1-render-proof-hygiene-failure/v1",
            "terminal": "FAILURE",
            "status": status,
            "failure_class": failure_class,
            "error_type": type(error).__name__,
            "error": str(error),
            "execution_commit": args.execution_commit,
            "plan_sha256": None if plan is None else plan.get("_file_sha256"),
            "candidate_files": [],
            "model_loaded": False,
            "cuda_initialized": False,
            "update": False,
            "elapsed_seconds": time.time() - started,
        }
        failure["receipt_sha256"] = hashlib.sha256(canonical_terminal_bytes(failure)).hexdigest()
        try:
            payload = canonical_terminal_bytes(failure)
            parsed = strict_json_loads(payload)
            _validate_failure(parsed, execution_commit=args.execution_commit)
            terminal = _publish(args.output_dir, "FAILURE.json", payload)
            if terminal.read_bytes() != payload:
                raise RuntimeError("IPC render hygiene FAILURE published bytes differ")
            _validate_failure(strict_json_loads(terminal.read_bytes()), execution_commit=args.execution_commit)
        except BaseException as publication_error:
            print(f"IPC render hygiene failure publication failed: {publication_error}", file=sys.stderr)
        print(f"IPC render hygiene failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
