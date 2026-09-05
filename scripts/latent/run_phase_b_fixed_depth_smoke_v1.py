#!/usr/bin/env python3
"""Run the prospectively bound, zero-update Phase B fixed-depth smoke.

The runner first performs pure-stdlib authorization/provenance checks, then a
tokenizer-only normalization/render gate over all 12 selected rows. Torch's
model path and CUDA-facing Phase B modules remain unreachable until that gate
passes. A future root-authorized freeze must pin the repair's exact constants.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from prime_rl.phase_b_contract import (
    PhaseBContractError,
    atomic_exclusive_json,
    canonical_json_sha256,
    file_sha256,
    load_json_file,
    normalize_assistant_tool_call_arguments,
    validate_a0c_binding,
    validate_br2_failure_evidence,
    validate_failed_start_evidence,
    validate_plan_authorization,
    validate_preflight_rejection_evidence,
)

WORKTREE = Path("/home/ubuntu/rlm/worktrees/q35-2b-recurrent-sidecar-v1")
EXPERIMENT = WORKTREE / "experiments/qwen35-2b-latent-coordinator-v1"
PLAN = EXPERIMENT / "phase-b-fixed-depth-smoke-a0c-br3-plan.json"
SELECTION = EXPERIMENT / "phase-b-fixed-depth-smoke-v1-selection.json"
PLAN_SHA256 = "901182cae2debc1b810824b14e97e1feab8b71e4bd8554d178b390ce1d15d068"
SELECTION_SHA256 = "8e160b9214aeb5cc971abf472cb31c0173bdfeee2d56fea98620dc87b166b3fe"
EXPECTED_ENV = Path("/home/ubuntu/rlm/prime-rl/.venv")
EXPECTED_PYTHONPATH = (
    "/home/ubuntu/rlm/worktrees/q35-2b-recurrent-sidecar-v1/src:"
    "/home/ubuntu/rlm/worktrees/q35-2b-recurrent-sidecar-v1/packages/prime-rl-configs/src"
)
ARTIFACT_CAP = 536_870_912
MINIMUM_FREE_BYTES = 60 * 1024**3
WALL_CLOCK_LIMIT_SECONDS = 2 * 60 * 60
FAILURE_AUDIT_HEADROOM_SECONDS = 5 * 60
TERMINAL_PUBLICATION_HEADROOM_SECONDS = 60
COMPUTE_LIMIT_SECONDS = (
    WALL_CLOCK_LIMIT_SECONDS - FAILURE_AUDIT_HEADROOM_SECONDS - TERMINAL_PUBLICATION_HEADROOM_SECONDS
)


class PhaseBWallClockExceeded(RuntimeError):
    pass


class PhaseBMechanismRejected(RuntimeError):
    pass


def _wall_clock_timeout(_signal_number: int, _frame: Any) -> None:
    raise PhaseBWallClockExceeded("Phase B crossed its 114-minute compute limit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--a0c-binding", type=Path, required=True)
    parser.add_argument("--a0c-binding-hash", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def exact_git_commit(worktree: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=worktree, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def preflight_before_heavy_imports(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any, Any, Any]:
    if args.plan != PLAN or args.selection != SELECTION:
        raise PhaseBContractError("plan and selection paths differ from the deployed worktree")
    if file_sha256(args.plan) != PLAN_SHA256:
        raise PhaseBContractError("Phase B plan differs from the exact checked-in freeze")
    if file_sha256(args.selection) != SELECTION_SHA256:
        raise PhaseBContractError("Phase B selection differs from the exact checked-in freeze")
    plan = load_json_file(args.plan)
    selection = load_json_file(args.selection)

    # Deliberately fails for the currently checked-in plan.  This must precede
    # Torch/Transformers imports and all filesystem output creation.
    validate_plan_authorization(plan)
    dependency = plan["a0c_dependency"]
    if args.a0c_binding != Path(dependency.get("binding_path", "")) or args.a0c_binding_hash != Path(
        dependency.get("binding_hash_path", "")
    ):
        raise PhaseBContractError("A0C binding paths differ from the authorized plan")
    binding = validate_a0c_binding(args.a0c_binding, args.a0c_binding_hash)
    if dependency.get("binding_sha256") != binding.binding_file_sha256:
        raise PhaseBContractError("authorized plan does not bind the exact A0C binding file")
    if binding.binding.get("a0c_plan_sha256") != dependency.get("a0c_plan_sha256"):
        raise PhaseBContractError("authorized plan and A0C binding disagree on the A0C plan")
    if binding.binding.get("a0c_execution_commit") != dependency.get("a0c_execution_commit"):
        raise PhaseBContractError("authorized plan and A0C binding disagree on the A0C execution commit")
    if (
        binding.receipt_file_sha256 != dependency.get("receipt_file_sha256")
        or binding.receipt_canonical_sha256 != dependency.get("receipt_internal_sha256")
        or binding.receipt.get("schema_version") != dependency.get("receipt_schema_version")
    ):
        raise PhaseBContractError("authorized plan and A0C binding disagree on exact receipt identity")
    required_identity = {
        "e33_sha256": plan["protected_models"]["coordinator"]["expected_sha256"],
        "config_sha256": plan["model_metadata_sha256"]["config.json"],
        "tokenizer_sha256": plan["model_metadata_sha256"]["tokenizer.json"],
        "tokenizer_config_sha256": plan["model_metadata_sha256"]["tokenizer_config.json"],
        "chat_template_sha256": plan["model_metadata_sha256"]["chat_template.jinja"],
    }
    if binding.binding.get("identity") != required_identity:
        raise PhaseBContractError("A0C binding identity differs from the Phase B model identity")

    repair_dependency = plan.get("repair_dependency")
    if not isinstance(repair_dependency, dict):
        raise PhaseBContractError("Phase B-R plan lacks its exact failed-start dependency")
    failed_start = validate_failed_start_evidence(
        Path(repair_dependency.get("binding_path", "")),
        Path(repair_dependency.get("binding_hash_path", "")),
    )
    if failed_start.binding_file_sha256 != repair_dependency.get("binding_sha256"):
        raise PhaseBContractError("Phase B-R plan does not bind the exact failed-start evidence")
    if (
        failed_start.failure_file_sha256 != repair_dependency.get("failure_file_sha256")
        or failed_start.log_file_sha256 != repair_dependency.get("log_file_sha256")
    ):
        raise PhaseBContractError("Phase B-R failed-start artifacts differ from the plan")
    control_flow = failed_start.binding.get("control_flow_proof")
    expected_control_identity = {
        "prior_execution_commit": repair_dependency.get("prior_execution_commit"),
        "prior_runner_path": repair_dependency.get("prior_runner_path"),
        "prior_runner_sha256": repair_dependency.get("prior_runner_sha256"),
    }
    if not isinstance(control_flow, dict) or any(
        control_flow.get(key) != expected for key, expected in expected_control_identity.items()
    ):
        raise PhaseBContractError("Phase B-R plan and failed-start control-flow proof disagree")
    if any(
        control_flow.get(field) is not False
        for field in (
            "optimizer_construction_present",
            "optimizer_step_present",
            "checkpoint_write_present",
            "generation_present",
        )
    ):
        raise PhaseBContractError("failed-start evidence does not prove the no-update/no-checkpoint boundary")
    prior_source = subprocess.run(
        [
            "git",
            "show",
            f"{repair_dependency.get('prior_execution_commit')}:{repair_dependency.get('prior_runner_path')}",
        ],
        cwd=WORKTREE,
        capture_output=True,
        check=True,
    ).stdout
    if hashlib.sha256(prior_source).hexdigest() != repair_dependency.get("prior_runner_sha256"):
        raise PhaseBContractError("prior Phase B runner source differs from the repair dependency")

    br1_dependency = plan.get("br1_preflight_dependency")
    if not isinstance(br1_dependency, dict):
        raise PhaseBContractError("Phase B-R2 plan lacks the exact B-R preflight rejection")
    preflight_rejection = validate_preflight_rejection_evidence(
        Path(br1_dependency.get("binding_path", "")),
        Path(br1_dependency.get("binding_hash_path", "")),
    )
    if preflight_rejection.binding_file_sha256 != br1_dependency.get("binding_sha256"):
        raise PhaseBContractError("Phase B-R2 plan does not bind the exact preflight rejection")
    if (
        preflight_rejection.manifest_file_sha256 != br1_dependency.get("manifest_file_sha256")
        or preflight_rejection.log_file_sha256 != br1_dependency.get("log_file_sha256")
    ):
        raise PhaseBContractError("Phase B-R preflight artifacts differ from the B-R2 plan")

    br2_dependency = plan.get("br2_failure_dependency")
    if not isinstance(br2_dependency, dict):
        raise PhaseBContractError("Phase B-R3 plan lacks the exact B-R2 scalar-hash failure")
    br2_failure = validate_br2_failure_evidence(
        Path(br2_dependency.get("binding_path", "")),
        Path(br2_dependency.get("binding_hash_path", "")),
    )
    if br2_failure.binding_file_sha256 != br2_dependency.get("binding_sha256"):
        raise PhaseBContractError("Phase B-R3 plan does not bind the exact B-R2 failure")
    if (
        br2_failure.failure_file_sha256 != br2_dependency.get("failure_file_sha256")
        or br2_failure.manifest_file_sha256 != br2_dependency.get("manifest_file_sha256")
        or br2_failure.preflight_log_file_sha256 != br2_dependency.get("preflight_log_file_sha256")
        or br2_failure.run_log_file_sha256 != br2_dependency.get("run_log_file_sha256")
    ):
        raise PhaseBContractError("Phase B-R2 scalar-hash artifacts differ from the B-R3 plan")

    if len(args.execution_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.execution_commit
    ):
        raise PhaseBContractError("execution commit must be an exact lowercase 40-character commit")
    if args.execution_commit != exact_git_commit(WORKTREE):
        raise PhaseBContractError("reported execution commit differs from deployed HEAD")
    worktree_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=WORKTREE,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    if worktree_status:
        raise PhaseBContractError("Phase B execution requires a clean deployed worktree")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", plan["implementation_commit"], args.execution_commit],
        cwd=WORKTREE,
        check=True,
    )
    if Path(sys.prefix).resolve() != EXPECTED_ENV.resolve(strict=True):
        raise PhaseBContractError("runner is outside the frozen shared environment")
    if os.environ.get("UV_PROJECT_ENVIRONMENT") != str(EXPECTED_ENV):
        raise PhaseBContractError("UV_PROJECT_ENVIRONMENT differs from the frozen value")
    if os.environ.get("PYTHONPATH") != EXPECTED_PYTHONPATH:
        raise PhaseBContractError("PYTHONPATH differs from the frozen value")
    if args.output_dir != Path(plan["outputs"]["directory"]):
        raise PhaseBContractError("output directory differs from the exact authorized plan")
    if plan["outputs"]["new_artifact_hard_cap_bytes"] != ARTIFACT_CAP:
        raise PhaseBContractError("authorized plan artifact cap differs from the runner")
    if args.output_dir.parent.is_symlink() or not args.output_dir.parent.is_dir():
        raise PhaseBContractError("Phase B output parent is absent or symlinked")
    if shutil.disk_usage(args.output_dir.parent).free < MINIMUM_FREE_BYTES:
        raise PhaseBContractError("host has less than the frozen 60 GiB free-disk floor")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise PhaseBContractError("Phase B output namespace already exists")
    return plan, selection, binding, failed_start, preflight_rejection, br2_failure


def main() -> int:
    args = parse_args()
    try:
        plan, selection, binding, failed_start, preflight_rejection, br2_failure = preflight_before_heavy_imports(args)
    except BaseException as error:
        print(f"Phase B preflight refusal: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    # Only the data reader and tokenizer are imported at this boundary. The
    # all-12 normalization/render preflight must finish before Torch, the model
    # class, or any accelerator-facing Phase B module is imported.
    import pyarrow.parquet as parquet
    import transformers
    from transformers import AutoTokenizer

    if args.preflight_only:
        try:
            tokenizer_context = _tokenizer_only_preflight(
                plan=plan,
                selection=selection,
                parquet=parquet,
                transformers=transformers,
                AutoTokenizer=AutoTokenizer,
            )
        except BaseException as error:
            print(f"Phase B tokenizer preflight refusal: {type(error).__name__}: {error}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "status": "tokenizer_preflight_only_passed",
                    "plan_sha256": PLAN_SHA256,
                    "selection_sha256": SELECTION_SHA256,
                    "binding_sha256": binding.binding_file_sha256,
                    "receipt_file_sha256": binding.receipt_file_sha256,
                    "receipt_internal_sha256": binding.receipt_canonical_sha256,
                    "execution_commit": args.execution_commit,
                    "failed_start_binding_sha256": failed_start.binding_file_sha256,
                    "br1_preflight_binding_sha256": preflight_rejection.binding_file_sha256,
                    "br2_failure_binding_sha256": br2_failure.binding_file_sha256,
                    "normalization_and_render_proofs": tokenizer_context["proofs"],
                    "model_loaded": False,
                    "cuda_initialized_during_preflight": tokenizer_context["cuda_initialized"],
                    "output_created": False,
                },
                sort_keys=True,
            )
        )
        return 0

    output_created = False
    started = time.time()
    signal.signal(signal.SIGALRM, _wall_clock_timeout)
    signal.alarm(COMPUTE_LIMIT_SECONDS)
    try:
        args.output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        output_created = True
        tokenizer_context = _tokenizer_only_preflight(
            plan=plan,
            selection=selection,
            parquet=parquet,
            transformers=transformers,
            AutoTokenizer=AutoTokenizer,
        )

        # The tokenizer-only repair gate above is complete for all 12 rows.
        # Model/CUDA imports and loading are deliberately unreachable before it.
        import torch
        from transformers import AutoModelForImageTextToText

        from prime_rl.latent.local_depth import LocalDepthCodec, compose_local_depth_inputs
        from prime_rl.latent.recurrent import (
            OneShotFeedForwardSidecar,
            TimestepFreeRecurrentSidecar,
            diagnose_recurrent_states,
        )

        receipt = execute_smoke(
            plan=plan,
            binding=binding,
            failed_start=failed_start,
            preflight_rejection=preflight_rejection,
            br2_failure=br2_failure,
            tokenizer_context=tokenizer_context,
            torch=torch,
            transformers=transformers,
            AutoModelForImageTextToText=AutoModelForImageTextToText,
            LocalDepthCodec=LocalDepthCodec,
            compose_local_depth_inputs=compose_local_depth_inputs,
            OneShotFeedForwardSidecar=OneShotFeedForwardSidecar,
            TimestepFreeRecurrentSidecar=TimestepFreeRecurrentSidecar,
            diagnose_recurrent_states=diagnose_recurrent_states,
            execution_commit=args.execution_commit,
        )
        receipt["elapsed_seconds"] = time.time() - started
        if shutil.disk_usage(args.output_dir).free < MINIMUM_FREE_BYTES:
            raise PhaseBContractError("host crossed the 60 GiB free-disk floor before success")
        signal.alarm(0)
        atomic_exclusive_json(args.output_dir, "SUCCESS.json", receipt, maximum_directory_bytes=ARTIFACT_CAP)
        return 0
    except BaseException as error:
        signal.alarm(0)
        if output_created:
            try:
                signal.alarm(FAILURE_AUDIT_HEADROOM_SECONDS)
                post_failure_audit = _post_failure_hash_audit(
                    plan, binding, failed_start, preflight_rejection, br2_failure
                )
            except BaseException as audit_error:
                post_failure_audit = {
                    "audit_complete": False,
                    "hash_probe_error": f"{type(audit_error).__name__}: {audit_error}",
                }
            finally:
                signal.alarm(0)
            failure_category = _classify_failure(error)
            failure = {
                "schema_version": "q35-2b-phase-b-fixed-depth-smoke-failure/v1",
                "status": failure_category,
                "failure_category": failure_category,
                "claim_class": "no_update_mechanism_connectivity_only",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.time() - started,
                "plan_sha256": PLAN_SHA256,
                "selection_sha256": SELECTION_SHA256,
                "failed_start": {
                    "binding_file_sha256": failed_start.binding_file_sha256,
                    "failure_file_sha256": failed_start.failure_file_sha256,
                    "log_file_sha256": failed_start.log_file_sha256,
                },
                "br1_preflight_rejection": {
                    "binding_file_sha256": preflight_rejection.binding_file_sha256,
                    "manifest_file_sha256": preflight_rejection.manifest_file_sha256,
                    "log_file_sha256": preflight_rejection.log_file_sha256,
                },
                "br2_failure": {
                    "binding_file_sha256": br2_failure.binding_file_sha256,
                    "failure_file_sha256": br2_failure.failure_file_sha256,
                    "manifest_file_sha256": br2_failure.manifest_file_sha256,
                    "preflight_log_file_sha256": br2_failure.preflight_log_file_sha256,
                    "run_log_file_sha256": br2_failure.run_log_file_sha256,
                },
                "post_failure_hash_audit": post_failure_audit,
            }
            try:
                atomic_exclusive_json(args.output_dir, "FAILURE.json", failure, maximum_directory_bytes=ARTIFACT_CAP)
            except BaseException as receipt_error:
                print(f"Phase B failure receipt publication failed: {receipt_error}", file=sys.stderr)
        print(f"Phase B failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


def _post_failure_hash_audit(
    plan: dict[str, Any],
    binding: Any,
    failed_start: Any = None,
    preflight_rejection: Any = None,
    br2_failure: Any = None,
) -> dict[str, Any]:
    """Best-effort durable preservation evidence after any started run fails."""

    errors: list[str] = []
    audit: dict[str, Any] = {}
    try:
        model_path = Path(plan["protected_models"]["coordinator"]["host_path"])
        weight_hash = file_sha256(model_path / "model.safetensors")
        metadata = _metadata_hashes(model_path)
        audit["e33"] = {
            "weight_file_sha256": weight_hash,
            "weight_file_matches": weight_hash == plan["protected_models"]["coordinator"]["expected_sha256"],
            "metadata_sha256": metadata,
            "metadata_matches": metadata == plan["model_metadata_sha256"],
        }
    except BaseException as error:
        errors.append(f"e33:{type(error).__name__}:{error}")
    try:
        rebound = validate_a0c_binding(binding.binding_path, binding.binding_hash_path)
        audit["a0c"] = {
            "binding_file_sha256": rebound.binding_file_sha256,
            "binding_matches": rebound.binding_file_sha256 == binding.binding_file_sha256,
            "receipt_file_sha256": rebound.receipt_file_sha256,
            "receipt_file_matches": rebound.receipt_file_sha256 == binding.receipt_file_sha256,
            "receipt_internal_sha256": rebound.receipt_canonical_sha256,
            "receipt_internal_matches": rebound.receipt_canonical_sha256 == binding.receipt_canonical_sha256,
            "receipt_whole_object_sha256": rebound.receipt_whole_object_sha256,
        }
        if rebound != binding:
            errors.append("a0c:validated binding object changed during run")
    except BaseException as error:
        errors.append(f"a0c:{type(error).__name__}:{error}")
    if failed_start is not None:
        try:
            rebound_failed_start = validate_failed_start_evidence(
                failed_start.binding_path, failed_start.binding_hash_path
            )
            audit["repair_dependency"] = {
                "binding_file_sha256": rebound_failed_start.binding_file_sha256,
                "binding_matches": rebound_failed_start.binding_file_sha256 == failed_start.binding_file_sha256,
                "failure_file_sha256": rebound_failed_start.failure_file_sha256,
                "failure_matches": rebound_failed_start.failure_file_sha256 == failed_start.failure_file_sha256,
                "log_file_sha256": rebound_failed_start.log_file_sha256,
                "log_matches": rebound_failed_start.log_file_sha256 == failed_start.log_file_sha256,
            }
            if rebound_failed_start != failed_start:
                errors.append("repair_dependency:validated evidence object changed during run")
        except BaseException as error:
            errors.append(f"repair_dependency:{type(error).__name__}:{error}")
    if preflight_rejection is not None:
        try:
            rebound_preflight = validate_preflight_rejection_evidence(
                preflight_rejection.binding_path, preflight_rejection.binding_hash_path
            )
            audit["br1_preflight_rejection"] = {
                "binding_file_sha256": rebound_preflight.binding_file_sha256,
                "binding_matches": rebound_preflight.binding_file_sha256
                == preflight_rejection.binding_file_sha256,
                "manifest_file_sha256": rebound_preflight.manifest_file_sha256,
                "manifest_matches": rebound_preflight.manifest_file_sha256
                == preflight_rejection.manifest_file_sha256,
                "log_file_sha256": rebound_preflight.log_file_sha256,
                "log_matches": rebound_preflight.log_file_sha256 == preflight_rejection.log_file_sha256,
            }
            if rebound_preflight != preflight_rejection:
                errors.append("br1_preflight_rejection:validated evidence object changed during run")
        except BaseException as error:
            errors.append(f"br1_preflight_rejection:{type(error).__name__}:{error}")
    if br2_failure is not None:
        try:
            rebound_br2 = validate_br2_failure_evidence(br2_failure.binding_path, br2_failure.binding_hash_path)
            audit["br2_failure"] = {
                "binding_file_sha256": rebound_br2.binding_file_sha256,
                "binding_matches": rebound_br2.binding_file_sha256 == br2_failure.binding_file_sha256,
                "failure_file_sha256": rebound_br2.failure_file_sha256,
                "failure_matches": rebound_br2.failure_file_sha256 == br2_failure.failure_file_sha256,
                "manifest_file_sha256": rebound_br2.manifest_file_sha256,
                "manifest_matches": rebound_br2.manifest_file_sha256 == br2_failure.manifest_file_sha256,
                "preflight_log_file_sha256": rebound_br2.preflight_log_file_sha256,
                "preflight_log_matches": rebound_br2.preflight_log_file_sha256
                == br2_failure.preflight_log_file_sha256,
                "run_log_file_sha256": rebound_br2.run_log_file_sha256,
                "run_log_matches": rebound_br2.run_log_file_sha256 == br2_failure.run_log_file_sha256,
            }
            if rebound_br2 != br2_failure:
                errors.append("br2_failure:validated evidence object changed during run")
        except BaseException as error:
            errors.append(f"br2_failure:{type(error).__name__}:{error}")
    try:
        audit["plan_selection"] = {
            "plan_sha256": file_sha256(PLAN),
            "selection_sha256": file_sha256(SELECTION),
        }
        if audit["plan_selection"] != {"plan_sha256": PLAN_SHA256, "selection_sha256": SELECTION_SHA256}:
            errors.append("plan_selection:hash changed during run")
    except BaseException as error:
        errors.append(f"plan_selection:{type(error).__name__}:{error}")
    audit["audit_complete"] = not errors
    audit["hash_probe_error"] = "; ".join(errors) if errors else None
    return audit


def _classify_failure(error: BaseException) -> str:
    if isinstance(error, PhaseBMechanismRejected):
        return "mechanism_rejected"
    return "infrastructure_invalid"


def _tokenizer_only_preflight(  # noqa: N803
    *,
    plan: dict[str, Any],
    selection: dict[str, Any],
    parquet: Any,
    transformers: Any,
    AutoTokenizer: Any,
) -> dict[str, Any]:
    """Normalize and render every selected target before model/CUDA access."""

    if _cuda_initialized_if_torch_loaded():
        raise PhaseBContractError("CUDA was already initialized before tokenizer preflight")
    _validate_transformers_runtime(plan, transformers=transformers)
    model_path = Path(plan["protected_models"]["coordinator"]["host_path"])
    model_file_hash = file_sha256(_model_file(model_path))
    if model_file_hash != plan["protected_models"]["coordinator"]["expected_sha256"]:
        raise PhaseBContractError("e33 weight-file hash differs from the plan")
    metadata = _metadata_hashes(model_path)
    if metadata != plan["model_metadata_sha256"]:
        raise PhaseBContractError("e33 metadata hashes differ from the plan")

    data_plan = plan["data"]
    parquet_path = Path(data_plan["host_parquet_path"])
    manifest_path = Path(data_plan["host_manifest_path"])
    if file_sha256(parquet_path) != data_plan["source_parquet_sha256"]:
        raise PhaseBContractError("source parquet hash differs from the plan")
    if file_sha256(manifest_path) != data_plan["source_manifest_sha256"]:
        raise PhaseBContractError("source manifest hash differs from the plan")
    selected_rows = _select_rows(parquet.read_table(parquet_path).to_pylist(), selection)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    prepared_rows: list[dict[str, Any]] = []
    proofs: list[dict[str, Any]] = []
    for source_row in selected_rows:
        task_key = source_row["task_key"]
        action = source_row["action"]
        try:
            raw_row_sha256 = canonical_json_sha256(source_row)
            messages = json.loads(json.dumps(source_row["messages"]))
            if not isinstance(messages, list) or len(messages) < 2:
                raise PhaseBContractError("row lacks a multi-message assistant target")
            raw_messages_sha256 = canonical_json_sha256(messages)
            raw_target_sha256 = canonical_json_sha256(messages[-1])
            reasoning_content = messages[-1].get("reasoning_content")
            if not isinstance(reasoning_content, str) or not reasoning_content.strip():
                raise PhaseBContractError("final assistant target lacks nonempty reasoning_content")
            reasoning_content_sha256 = hashlib.sha256(reasoning_content.encode()).hexdigest()
            normalized, records = normalize_assistant_tool_call_arguments(messages, expected_action=action)
            if canonical_json_sha256(messages) != raw_messages_sha256 or len(records) != 1:
                raise PhaseBContractError("normalization mutated the source or modified multiple paths")
            record = records[0]
            expected_path = f"messages.{len(messages) - 1}.tool_calls.0.function.arguments"
            if record["modified_path"] != expected_path:
                raise PhaseBContractError("normalization modified an unexpected path")
            restored = json.loads(json.dumps(normalized))
            restored[-1]["tool_calls"][0]["function"]["arguments"] = messages[-1]["tool_calls"][0]["function"][
                "arguments"
            ]
            if restored != messages:
                raise PhaseBContractError("normalization changed content outside the one allowed path")
            if normalized[-1].get("reasoning_content") != reasoning_content:
                raise PhaseBContractError("normalization changed reasoning_content")

            raw_tools = source_row.get("tools")
            tools = json.loads(raw_tools) if isinstance(raw_tools, str) else json.loads(json.dumps(raw_tools))
            if not isinstance(tools, list):
                raise PhaseBContractError("row tools must be a list")
            prefix = normalized[:-1]
            plain_rendered = _render(tokenizer, prefix, tools, False)
            open_rendered = _render(tokenizer, prefix, tools, True)
            full_rendered = _render(tokenizer, normalized, tools, False)
            if not open_rendered.startswith(plain_rendered):
                raise PhaseBContractError("explicit-thinking opening does not string-prefix the plain context")
            if not full_rendered.startswith(open_rendered):
                raise PhaseBContractError("full target does not preserve the explicit-thinking string opening")
            plain_ids = _token_ids_list(tokenizer, plain_rendered)
            open_ids = _token_ids_list(tokenizer, open_rendered)
            full_ids = _token_ids_list(tokenizer, full_rendered)
            if len(plain_ids) < 8 or len(open_ids) <= len(plain_ids):
                raise PhaseBContractError("target has no verified assistant-generation opening")
            if len(full_ids) <= len(open_ids) or full_ids[: len(open_ids)] != open_ids:
                raise PhaseBContractError("full target does not preserve the generation prefix")
            if open_ids[: len(plain_ids)] != plain_ids:
                raise PhaseBContractError("generation opening does not extend its plain prefix")
        except PhaseBContractError as error:
            raise PhaseBContractError(f"row {task_key} tokenizer preflight failed: {error}") from error
        except BaseException as error:
            raise PhaseBContractError(f"row {task_key} tokenizer preflight failed: {error}") from error

        proof = {
            "task_key": task_key,
            "action": action,
            "raw_row_sha256": raw_row_sha256,
            "raw_messages_sha256": raw_messages_sha256,
            "raw_target_sha256": raw_target_sha256,
            "normalized_target_sha256": canonical_json_sha256(normalized[-1]),
            "reasoning_content_sha256": reasoning_content_sha256,
            "reasoning_content_utf8_bytes": len(reasoning_content.encode()),
            "reasoning_content_preserved_byte_exact": True,
            "modified_paths": [record["modified_path"]],
            "raw_arguments_sha256": record["raw_arguments_sha256"],
            "normalized_arguments_sha256": record["normalized_arguments_sha256"],
            "normalized_arguments": record["normalized_arguments"],
            "plain_prefix_sha256": hashlib.sha256(plain_rendered.encode()).hexdigest(),
            "generation_prefix_sha256": hashlib.sha256(open_rendered.encode()).hexdigest(),
            "full_target_sha256": hashlib.sha256(full_rendered.encode()).hexdigest(),
            "plain_prefix_token_ids_sha256": canonical_json_sha256(plain_ids),
            "generation_prefix_token_ids_sha256": canonical_json_sha256(open_ids),
            "full_target_token_ids_sha256": canonical_json_sha256(full_ids),
            "plain_prefix_tokens": len(plain_ids),
            "generation_prefix_tokens": len(open_ids),
            "full_target_tokens": len(full_ids),
            "generation_opening_tokens": len(open_ids) - len(plain_ids),
            "latent_injection_token_index": len(plain_ids),
            "label_mask_through_token_index": len(open_ids),
        }
        proofs.append(proof)
        prepared_rows.append(
            {
                "task_key": task_key,
                "action": action,
                "plain_ids": plain_ids,
                "open_ids": open_ids,
                "full_ids": full_ids,
                "render_proof": proof,
            }
        )

    if len(prepared_rows) != 12 or len(proofs) != 12:
        raise PhaseBContractError("tokenizer preflight did not complete all 12 selected rows")
    if _cuda_initialized_if_torch_loaded():
        raise PhaseBContractError("CUDA initialized during tokenizer-only preflight")
    return {
        "rows": prepared_rows,
        "proofs": proofs,
        "backward_probe_task_key": selection["backward_probe_task_key"],
        "model_file_sha256": model_file_hash,
        "metadata_sha256": metadata,
        "cuda_initialized": False,
    }


def _cuda_initialized_if_torch_loaded() -> bool:
    torch_module = sys.modules.get("torch")
    cuda_module = getattr(torch_module, "cuda", None)
    return bool(cuda_module is not None and cuda_module.is_initialized())


def execute_smoke(  # noqa: PLR0913, PLR0915
    *,
    plan: dict[str, Any],
    binding: Any,
    failed_start: Any,
    preflight_rejection: Any,
    br2_failure: Any,
    tokenizer_context: dict[str, Any],
    torch: Any,
    transformers: Any,
    AutoModelForImageTextToText: Any,
    LocalDepthCodec: Any,
    compose_local_depth_inputs: Any,
    OneShotFeedForwardSidecar: Any,
    TimestepFreeRecurrentSidecar: Any,
    diagnose_recurrent_states: Any,
    execution_commit: str,
) -> dict[str, Any]:
    """Execute only full-recompute, teacher-forced forwards and diagnostics."""

    _validate_torch_runtime(plan, torch=torch)
    _validate_transformers_runtime(plan, transformers=transformers)
    model_path = Path(plan["protected_models"]["coordinator"]["host_path"])
    model_file = _model_file(model_path)
    model_file_before = tokenizer_context["model_file_sha256"]
    metadata_before = tokenizer_context["metadata_sha256"]
    data_plan = plan["data"]
    parquet_path = Path(data_plan["host_parquet_path"])
    manifest_path = Path(data_plan["host_manifest_path"])
    selected_rows = tokenizer_context["rows"]

    _require_gpu0_idle()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise PhaseBContractError("frozen host topology requires two visible CUDA GPUs")
    if torch.cuda.get_device_name(0) != "NVIDIA RTX A6000" or torch.cuda.get_device_name(1) != "NVIDIA RTX A6000":
        raise PhaseBContractError("frozen host topology requires two NVIDIA RTX A6000 GPUs")
    torch.cuda.set_device(0)
    torch.manual_seed(plan["matched_conditions"]["seed"])
    torch.cuda.manual_seed_all(plan["matched_conditions"]["seed"])

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda:0")
    model.eval()
    if model.__class__.__name__ != plan["protected_models"]["coordinator"]["architecture"]:
        raise PhaseBContractError("loaded e33 architecture differs from the plan")
    if int(model.config.text_config.hidden_size) != 2048:
        raise PhaseBContractError("loaded e33 text hidden size is not 2048")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise PhaseBContractError("protected e33 is not entirely frozen")

    model_tensor_before = _module_tensor_sha256(model, torch)
    embedding_shell_norm = _mean_embedding_norm(model.get_input_embeddings().weight, torch)
    examples = [_prepare_example(row, model=model, torch=torch) for row in selected_rows]

    codec_template = LocalDepthCodec().to(device="cuda:0", dtype=torch.bfloat16).eval()
    codec_state = {name: value.detach().clone() for name, value in codec_template.state_dict().items()}
    codecs = {
        name: LocalDepthCodec().to(device="cuda:0", dtype=torch.bfloat16).eval()
        for name in ("STATIC", "FFN", "RECURRENT")
    }
    for codec in codecs.values():
        codec.load_state_dict(codec_state, strict=True)
    del codec_template, codec_state
    ffn = OneShotFeedForwardSidecar().to(device="cuda:0", dtype=torch.bfloat16).eval()
    recurrent = TimestepFreeRecurrentSidecar().to(device="cuda:0", dtype=torch.bfloat16).eval()
    modules = {
        "STATIC.codec": codecs["STATIC"],
        "FFN.codec": codecs["FFN"],
        "RECURRENT.codec": codecs["RECURRENT"],
        "FFN.sidecar": ffn,
        "RECURRENT.sidecar": recurrent,
    }
    module_tensor_before = {name: _module_tensor_sha256(module, torch) for name, module in modules.items()}
    codec_hashes = {module_tensor_before[f"{arm}.codec"] for arm in ("STATIC", "FFN", "RECURRENT")}
    if len(codec_hashes) != 1:
        raise PhaseBContractError("arm codecs are not bitwise-identical at initialization")
    counts = {
        "STATIC": _parameter_count(codecs["STATIC"]),
        "FFN": _parameter_count(codecs["FFN"]) + _parameter_count(ffn),
        "RECURRENT": _parameter_count(codecs["RECURRENT"]) + _parameter_count(recurrent),
    }
    expected_counts = {arm: plan["arms"][arm]["trainable_parameters"] for arm in counts}
    if counts != expected_counts or counts["FFN"] - counts["RECURRENT"] != 107:
        raise PhaseBContractError(f"arm parameter counts differ from the frozen match: {counts}")

    metrics: dict[str, Any] = {"BASE": [], "STATIC": [], "FFN": [], "RECURRENT": []}
    recurrent_every_step_changed: list[bool] = []
    for example in examples:
        with torch.no_grad():
            base = model(
                input_ids=example["full_ids"],
                attention_mask=example["full_mask"],
                position_ids=example["full_positions"],
                labels=example["labels"],
                use_cache=False,
                return_dict=True,
            )
        _require_finite(base.loss, "BASE loss", torch)
        metrics["BASE"].append({"task_key": example["task_key"], "loss": float(base.loss)})

        anchors: dict[str, Any] = {}
        for arm in ("STATIC", "FFN", "RECURRENT"):
            hidden = example["captured_hidden"].to("cuda:0")
            hidden_mask = torch.ones(hidden.shape[:2], dtype=torch.long, device="cuda:0")
            anchors[arm] = codecs[arm].encode(hidden, hidden_mask)
        if not torch.equal(anchors["STATIC"], anchors["FFN"]) or not torch.equal(
            anchors["STATIC"], anchors["RECURRENT"]
        ):
            raise PhaseBMechanismRejected("identical codecs did not produce bitwise-identical task anchors")
        visible = {
            "STATIC": anchors["STATIC"],
            "FFN": ffn(anchors["FFN"]),
        }
        trajectory = recurrent.rollout(anchors["RECURRENT"], 4, return_trajectory=True)
        visible["RECURRENT"] = trajectory[-1].visible_workspace
        if not torch.equal(visible["STATIC"], visible["FFN"]) or not torch.equal(
            visible["STATIC"], visible["RECURRENT"]
        ):
            raise PhaseBMechanismRejected("zero-scale sidecars changed the STATIC visible workspace")

        for arm in ("STATIC", "FFN", "RECURRENT"):
            result = _latent_forward(
                example,
                visible[arm],
                codec=codecs[arm],
                model=model,
                embedding_shell_norm=embedding_shell_norm,
                compose_local_depth_inputs=compose_local_depth_inputs,
                torch=torch,
            )
            _require_finite(result.loss, f"{arm} loss", torch)
            row_metric: dict[str, Any] = {"task_key": example["task_key"], "loss": float(result.loss)}
            if arm == "RECURRENT":
                diagnostic = diagnose_recurrent_states(trajectory)
                if diagnostic.nonfinite:
                    raise PhaseBMechanismRejected("RECURRENT state diagnostics are non-finite")
                memory_changes = diagnostic.memory_change_norms
                if not bool(torch.count_nonzero(memory_changes)):
                    raise PhaseBMechanismRejected("RECURRENT private memory did not change")
                every_step_changed = bool(torch.all(memory_changes > 0))
                recurrent_every_step_changed.append(every_step_changed)
                row_metric["state"] = _diagnostic_json(diagnostic)
                row_metric["private_memory_changed_at_every_step"] = every_step_changed
            metrics[arm].append(row_metric)
    if not any(recurrent_every_step_changed):
        raise PhaseBMechanismRejected("no RECURRENT row changed private memory at every one of four steps")

    backward_key = tokenizer_context["backward_probe_task_key"]
    backward_example = next(example for example in examples if example["task_key"] == backward_key)
    gradients = {
        "canonical": _backward_probes(
            backward_example,
            codecs=codecs,
            ffn=ffn,
            recurrent=recurrent,
            model=model,
            embedding_shell_norm=embedding_shell_norm,
            compose_local_depth_inputs=compose_local_depth_inputs,
            torch=torch,
            residual_scale_override=None,
        ),
        "hypothetical_open_gate": _backward_probes(
            backward_example,
            codecs=codecs,
            ffn=ffn,
            recurrent=recurrent,
            model=model,
            embedding_shell_norm=embedding_shell_norm,
            compose_local_depth_inputs=compose_local_depth_inputs,
            torch=torch,
            residual_scale_override=0.001,
            include_static=False,
        ),
    }
    _validate_gradients(gradients)

    module_tensor_after = {name: _module_tensor_sha256(module, torch) for name, module in modules.items()}
    model_tensor_after = _module_tensor_sha256(model, torch)
    model_file_after = file_sha256(model_file)
    metadata_after = _metadata_hashes(model_path)
    if module_tensor_before != module_tensor_after:
        raise PhaseBMechanismRejected("one or more Phase B module parameters changed")
    if model_tensor_before != model_tensor_after or model_file_before != model_file_after:
        raise PhaseBContractError("protected e33 parameters changed")
    if metadata_before != metadata_after:
        raise PhaseBContractError("protected e33 metadata changed")
    if file_sha256(parquet_path) != data_plan["source_parquet_sha256"]:
        raise PhaseBContractError("source parquet changed during the smoke")
    if file_sha256(manifest_path) != data_plan["source_manifest_sha256"]:
        raise PhaseBContractError("source manifest changed during the smoke")
    if file_sha256(PLAN) != PLAN_SHA256 or file_sha256(SELECTION) != SELECTION_SHA256:
        raise PhaseBContractError("Phase B plan or selection changed during the smoke")
    rebound = validate_a0c_binding(binding.binding_path, binding.binding_hash_path)
    if rebound != binding:
        raise PhaseBContractError("A0C binding or receipt changed during the smoke")
    rebound_failed_start = validate_failed_start_evidence(failed_start.binding_path, failed_start.binding_hash_path)
    if rebound_failed_start != failed_start:
        raise PhaseBContractError("failed-start repair evidence changed during the smoke")
    rebound_preflight = validate_preflight_rejection_evidence(
        preflight_rejection.binding_path, preflight_rejection.binding_hash_path
    )
    if rebound_preflight != preflight_rejection:
        raise PhaseBContractError("B-R preflight rejection evidence changed during the smoke")
    rebound_br2 = validate_br2_failure_evidence(br2_failure.binding_path, br2_failure.binding_hash_path)
    if rebound_br2 != br2_failure:
        raise PhaseBContractError("B-R2 failure evidence changed during the smoke")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise PhaseBContractError("protected e33 accumulated gradients")

    return {
        "schema_version": "q35-2b-phase-b-fixed-depth-smoke-success/v1",
        "status": "SUCCESS",
        "claim_class": "no_update_mechanism_connectivity_only",
        "interpretation_boundary": plan["interpretation_boundary"],
        "plan_sha256": PLAN_SHA256,
        "selection_sha256": SELECTION_SHA256,
        "implementation_commit": plan["implementation_commit"],
        "execution_commit": execution_commit,
        "a0c": {
            "binding_file_sha256": binding.binding_file_sha256,
            "receipt_file_sha256": binding.receipt_file_sha256,
            "receipt_canonical_sha256": binding.receipt_canonical_sha256,
            "receipt_whole_object_sha256": binding.receipt_whole_object_sha256,
            "claim": binding.binding["required_claim"],
        },
        "repair_dependency": {
            "binding_file_sha256": failed_start.binding_file_sha256,
            "failure_file_sha256": failed_start.failure_file_sha256,
            "log_file_sha256": failed_start.log_file_sha256,
            "prior_status": failed_start.failure["status"],
            "prior_error_type": failed_start.failure["error_type"],
            "prior_error": failed_start.failure["error"],
        },
        "br1_preflight_rejection": {
            "binding_file_sha256": preflight_rejection.binding_file_sha256,
            "manifest_file_sha256": preflight_rejection.manifest_file_sha256,
            "log_file_sha256": preflight_rejection.log_file_sha256,
            "model_loaded": preflight_rejection.manifest["model_loaded"],
            "cuda_initialized": preflight_rejection.manifest["cuda_initialized"],
            "output_created": preflight_rejection.manifest["output_created"],
        },
        "br2_failure": {
            "binding_file_sha256": br2_failure.binding_file_sha256,
            "failure_file_sha256": br2_failure.failure_file_sha256,
            "manifest_file_sha256": br2_failure.manifest_file_sha256,
            "preflight_log_file_sha256": br2_failure.preflight_log_file_sha256,
            "run_log_file_sha256": br2_failure.run_log_file_sha256,
            "preflight_rows": len(br2_failure.preflight_report["normalization_and_render_proofs"]),
            "no_useful_forward": not br2_failure.manifest["useful_model_forward_completed"],
            "no_update_attempt": not br2_failure.manifest["model_update_attempted"],
        },
        "optimizer": None,
        "optimizer_updates": 0,
        "generation": False,
        "cache": False,
        "worker_loaded": False,
        "tokenizer_preflight_before_model_load": {
            "completed_rows": len(tokenizer_context["proofs"]),
            "model_loaded_during_preflight": False,
            "cuda_initialized_during_preflight": tokenizer_context["cuda_initialized"],
            "proofs": tokenizer_context["proofs"],
        },
        "parameter_counts": counts,
        "metrics": metrics,
        "gradients": gradients,
        "hashes": {
            "model_file_pre": model_file_before,
            "model_file_post": model_file_after,
            "model_tensor_tree_pre": model_tensor_before,
            "model_tensor_tree_post": model_tensor_after,
            "module_tensor_tree_pre": module_tensor_before,
            "module_tensor_tree_post": module_tensor_after,
            "metadata_pre": metadata_before,
            "metadata_post": metadata_after,
        },
        "rendered_inputs": [example["rendered_hashes"] for example in examples],
        "embedding_shell_norm": float(embedding_shell_norm),
        "maximum_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
    }


def _validate_transformers_runtime(plan: dict[str, Any], *, transformers: Any) -> None:
    if transformers.__version__ != plan["software"]["transformers"]:
        raise PhaseBContractError("Transformers version differs from the plan")
    if importlib.metadata.version("transformers") != transformers.__version__:
        raise PhaseBContractError("Transformers package metadata differs from its runtime version")
    for module_name, expected_hash in plan["software"]["transformers_source_sha256"].items():
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise PhaseBContractError(f"Transformers source module cannot be located: {module_name}")
        module_path = Path(spec.origin).resolve(strict=True)
        if (
            not module_path.is_relative_to(EXPECTED_ENV.resolve(strict=True))
            or file_sha256(module_path) != expected_hash
        ):
            raise PhaseBContractError(f"Transformers runtime source differs from the plan: {module_name}")


def _validate_torch_runtime(plan: dict[str, Any], *, torch: Any) -> None:
    if torch.__version__ != plan["software"]["torch"]:
        raise PhaseBContractError("Torch runtime version differs from the plan")
    if importlib.metadata.version("torch") != plan["software"]["torch_distribution"]:
        raise PhaseBContractError("Torch distribution version differs from the plan")


def _require_gpu0_idle() -> None:
    gpu_rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    gpu_zero = next((row.split(",", 1)[1].strip() for row in gpu_rows if row.split(",", 1)[0].strip() == "0"), None)
    if gpu_zero is None:
        raise PhaseBContractError("nvidia-smi did not report GPU 0")
    process_rows = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    if any(len(fields := row.split(",", 1)) == 2 and fields[1].strip() == gpu_zero for row in process_rows):
        raise PhaseBContractError("GPU 0 already has a compute process")


def _model_file(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or not (path / "STABLE").is_file():
        raise PhaseBContractError("e33 checkpoint is not a direct stable directory")
    weight = path / "model.safetensors"
    file_sha256(weight)
    return weight


def _metadata_hashes(path: Path) -> dict[str, str]:
    names = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
    )
    return {name: file_sha256(path / name) for name in names}


def _select_rows(rows: list[dict[str, Any]], selection: dict[str, Any]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("task_key")
        if key in by_key:
            raise PhaseBContractError(f"duplicate source task key: {key}")
        if isinstance(key, str):
            by_key[key] = row
    keys = selection["probe_task_keys"]
    if len(keys) != 12 or len(set(keys)) != 12 or any(key not in by_key for key in keys):
        raise PhaseBContractError("frozen 12-row selection is not uniquely present in the source parquet")
    selected = [by_key[key] for key in keys]
    actions = [row.get("action") for row in selected]
    if {action: actions.count(action) for action in set(actions)} != {
        "solve_owned": 4,
        "delegate_terminal": 4,
        "delegate_coordinator": 4,
    }:
        raise PhaseBContractError("frozen selection is not action-balanced 4/4/4")
    return selected


def _render(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]], generation: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=generation,
        enable_thinking=True,
    )


def _token_ids_list(tokenizer: Any, rendered: str) -> list[int]:
    encoded = tokenizer(rendered, add_special_tokens=False)
    values = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise PhaseBContractError("tokenizer returned an unexpected batch dimension")
        values = values[0]
    if not isinstance(values, list) or not values or any(type(value) is not int or value < 0 for value in values):
        raise PhaseBContractError("tokenizer returned invalid token IDs")
    return values


def _prepare_example(row: dict[str, Any], *, model: Any, torch: Any) -> dict[str, Any]:
    plain_ids = torch.tensor([row["plain_ids"]], device="cuda:0", dtype=torch.long)
    open_ids = torch.tensor([row["open_ids"]], device="cuda:0", dtype=torch.long)
    full_ids = torch.tensor([row["full_ids"]], device="cuda:0", dtype=torch.long)
    plain_mask = torch.ones_like(plain_ids)
    with torch.no_grad():
        capture = (
            model(
                input_ids=plain_ids,
                attention_mask=plain_mask,
                position_ids=torch.arange(plain_ids.shape[1], device="cuda:0")[None],
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            .hidden_states[-1][:, -8:, :]
            .detach()
            .cpu()
        )
    if capture.shape != (1, 8, 2048) or capture.requires_grad or not bool(torch.isfinite(capture).all()):
        raise PhaseBMechanismRejected(f"row {row['task_key']} produced an invalid detached local feature capture")
    labels = full_ids.clone()
    labels[:, : open_ids.shape[1]] = -100
    full_mask = torch.ones_like(full_ids)
    return {
        "task_key": row["task_key"],
        "full_ids": full_ids,
        "full_mask": full_mask,
        "full_positions": torch.arange(full_ids.shape[1], device="cuda:0")[None],
        "labels": labels,
        "injection_index": int(plain_ids.shape[1]),
        "captured_hidden": capture,
        "rendered_hashes": {
            **row["render_proof"],
            "plain_prefix_token_tensor_sha256": _tensor_bytes_sha256(plain_ids, torch),
            "generation_prefix_token_tensor_sha256": _tensor_bytes_sha256(open_ids, torch),
            "full_target_token_tensor_sha256": _tensor_bytes_sha256(full_ids, torch),
            "captured_hidden_sha256": _tensor_bytes_sha256(capture, torch),
            "injection_index": int(plain_ids.shape[1]),
        },
    }


def _latent_forward(
    example: dict[str, Any],
    visible: Any,
    *,
    codec: Any,
    model: Any,
    embedding_shell_norm: Any,
    compose_local_depth_inputs: Any,
    torch: Any,
) -> Any:
    latent = codec.decode(visible, embedding_shell_norm)
    token_embeddings = model.get_input_embeddings()(example["full_ids"])
    composed = compose_local_depth_inputs(
        token_embeddings,
        example["full_mask"],
        example["full_positions"],
        example["labels"],
        latent,
        example["injection_index"],
    )
    start, stop = composed.latent_span
    if stop - start != 8:
        raise PhaseBMechanismRejected("latent insertion does not contain exactly eight slots")
    if not bool(torch.all(composed.attention_mask[:, start:stop] == 1)):
        raise PhaseBMechanismRejected("latent attention mask differs from one")
    if not bool(torch.all(composed.labels[:, start:stop] == -100)):
        raise PhaseBMechanismRejected("latent labels differ from -100")
    expected_positions = torch.arange(start, stop, device="cuda:0")[None]
    if not torch.equal(composed.position_ids[:, start:stop], expected_positions):
        raise PhaseBMechanismRejected("latent positions are not sequential at the verified boundary")
    return model(
        inputs_embeds=composed.inputs_embeds,
        attention_mask=composed.attention_mask,
        position_ids=composed.position_ids,
        labels=composed.labels,
        use_cache=False,
        return_dict=True,
    )


def _backward_probes(  # noqa: PLR0913
    example: dict[str, Any],
    *,
    codecs: dict[str, Any],
    ffn: Any,
    recurrent: Any,
    model: Any,
    embedding_shell_norm: Any,
    compose_local_depth_inputs: Any,
    torch: Any,
    residual_scale_override: float | None,
    include_static: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    arms = ("STATIC", "FFN", "RECURRENT") if include_static else ("FFN", "RECURRENT")
    for arm in arms:
        for module in (codecs[arm], ffn, recurrent):
            for parameter in module.parameters():
                parameter.grad = None
        hidden = example["captured_hidden"].to("cuda:0")
        anchor = codecs[arm].encode(hidden, torch.ones(hidden.shape[:2], dtype=torch.long, device="cuda:0"))
        if arm == "STATIC":
            visible = anchor
        elif arm == "FFN":
            visible = ffn(anchor, residual_scale_override=residual_scale_override)
        else:
            visible = recurrent.rollout(
                anchor,
                4,
                residual_scale_override=residual_scale_override,
            ).visible_workspace
        output = _latent_forward(
            example,
            visible,
            codec=codecs[arm],
            model=model,
            embedding_shell_norm=embedding_shell_norm,
            compose_local_depth_inputs=compose_local_depth_inputs,
            torch=torch,
        )
        output.loss.backward()
        named_parameters = {f"codec.{name}": value for name, value in codecs[arm].named_parameters()}
        if arm == "FFN":
            named_parameters.update({f"sidecar.{name}": value for name, value in ffn.named_parameters()})
        elif arm == "RECURRENT":
            named_parameters.update({f"sidecar.{name}": value for name, value in recurrent.named_parameters()})
        result[arm] = {name: _gradient_summary(parameter.grad, torch) for name, parameter in named_parameters.items()}
    return result


def _gradient_summary(gradient: Any | None, torch: Any) -> dict[str, Any]:
    if gradient is None:
        return {"present": False, "finite": True, "nonzero": False, "l2": 0.0}
    detached = gradient.detach().float()
    return {
        "present": True,
        "finite": bool(torch.isfinite(detached).all()),
        "nonzero": bool(torch.count_nonzero(detached)),
        "l2": float(torch.linalg.vector_norm(detached)),
    }


def _validate_gradients(gradients: dict[str, Any]) -> None:
    canonical = gradients["canonical"]
    required_nonzero = (
        ("STATIC", "codec.source_projection.weight"),
        ("STATIC", "codec.receiver_projection.weight"),
        ("FFN", "sidecar.output_scale"),
        ("RECURRENT", "sidecar.output_scale"),
    )
    for arm, name in required_nonzero:
        summary = canonical[arm][name]
        if not summary["finite"] or not summary["nonzero"]:
            raise PhaseBMechanismRejected(f"canonical {arm} gradient {name} is not finite and nonzero")
    required_zero_prefixes = {
        "FFN": ("sidecar.hidden.", "sidecar.output."),
        "RECURRENT": (
            "sidecar.transition.",
            "sidecar.memory_candidate.",
            "sidecar.memory_gate.",
            "sidecar.workspace_delta.",
        ),
    }
    for arm, prefixes in required_zero_prefixes.items():
        for name, summary in canonical[arm].items():
            if name.startswith(prefixes) and (not summary["finite"] or summary["nonzero"]):
                raise PhaseBMechanismRejected(f"closed-gate {arm} internal gradient {name} is unexpectedly nonzero")
    open_gate = gradients["hypothetical_open_gate"]
    for arm, prefixes in required_zero_prefixes.items():
        for name, summary in open_gate[arm].items():
            if name.startswith(prefixes) and (not summary["finite"] or not summary["nonzero"]):
                raise PhaseBMechanismRejected(f"open-gate {arm} internal gradient {name} is not finite and nonzero")


def _parameter_count(module: Any) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _module_tensor_sha256(module: Any, torch: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode())
        digest.update(contiguous.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tensor_bytes_sha256(tensor: Any, torch: Any) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _mean_embedding_norm(weight: Any, torch: Any) -> Any:
    total = torch.zeros((), device=weight.device, dtype=torch.float64)
    for chunk in weight.split(4096, dim=0):
        total += torch.linalg.vector_norm(chunk.float(), dim=-1).double().sum()
    result = (total / weight.shape[0]).to(dtype=weight.dtype)
    _require_finite(result, "embedding shell norm", torch)
    if not bool(result > 0):
        raise PhaseBMechanismRejected("embedding shell norm is not positive")
    return result


def _require_finite(value: Any, label: str, torch: Any) -> None:
    if not bool(torch.isfinite(value).all()):
        raise PhaseBMechanismRejected(f"{label} is non-finite")


def _diagnostic_json(diagnostic: Any) -> dict[str, Any]:
    return {
        "visible_change_norms": diagnostic.visible_change_norms.tolist(),
        "memory_change_norms": diagnostic.memory_change_norms.tolist(),
        "visible_contraction_ratios": diagnostic.visible_contraction_ratios.tolist(),
        "memory_contraction_ratios": diagnostic.memory_contraction_ratios.tolist(),
        "visible_direction_cosines": diagnostic.visible_direction_cosines.tolist(),
        "memory_direction_cosines": diagnostic.memory_direction_cosines.tolist(),
        "visible_oscillation_rate": diagnostic.visible_oscillation_rate,
        "memory_oscillation_rate": diagnostic.memory_oscillation_rate,
    }


if __name__ == "__main__":
    raise SystemExit(main())
