#!/usr/bin/env python3
"""Run the nomination-only B-IPC1 matched in-place-carrier learning screen."""

from __future__ import annotations

import argparse
import base64
import contextlib
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence
from unittest import mock

from prime_rl.phase_b_contract import PhaseBContractError, file_sha256, load_json_file
from prime_rl.phase_b_ipc1 import (
    ACTIONS,
    ARTIFACT_CAP_BYTES,
    CUDA_MEMORY_CAP_BYTES,
    EVALUATION_DEPTHS,
    FAILURE_STATUS_CLASSES,
    INITIALIZATION_SEED,
    MINIMUM_FREE_DISK_BYTES,
    MINIMUM_HOST_RAM_BYTES,
    SLOTS,
    SUCCESS_STATUSES,
    TRAINING_ARMS,
    build_cache_guard_labels,
    build_memory_checkpoint_labels,
    build_model_call_schedule,
    canonical_bank_sha256,
    canonical_terminal_bytes,
    differentiable_margin_retention_from_baseline_margins,
    evaluate_common_arm,
    evaluate_recurrent_value,
    roundtrip_validate_terminal,
    strict_json_loads,
    validate_ipc1_plan,
    verify_published_terminal,
)

WORKTREE = Path("/home/ubuntu/rlm/worktrees/q35-2b-recurrent-sidecar-v1")
EXPERIMENT = WORKTREE / "experiments/qwen35-2b-latent-coordinator-v1"
DEFAULT_PLAN = EXPERIMENT / "phase-b-ipc1-matched-learning-run1-plan.json"
BR5_RUNNER = WORKTREE / "scripts/latent/run_phase_b_fixed_depth_smoke_v1.py"
B1_RUNNER = WORKTREE / "scripts/latent/run_phase_b_teacher_forced_value_screen_v1.py"
HIC0_RUNNER = WORKTREE / "scripts/latent/run_phase_b_identity_carrier_v1.py"
EXPECTED_ENV = Path("/home/ubuntu/rlm/prime-rl/.venv")
EXPECTED_PYTHONPATH = (
    "/home/ubuntu/rlm/worktrees/q35-2b-recurrent-sidecar-v1/src:"
    "/home/ubuntu/rlm/worktrees/q35-2b-recurrent-sidecar-v1/packages/prime-rl-configs/src"
)
COMPUTE_SECONDS = 14_040
AUDIT_SECONDS = 300


def _require_exact_keys(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise PhaseBContractError(f"B-IPC1 {label} keyset differs: {observed}")
    return value


def _require_finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PhaseBContractError(f"B-IPC1 {label} is not a finite number")
    return float(value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class MechanismRejected(PhaseBContractError):
    """Raised only for a completed pre-update identity/connectivity rejection."""


class CacheContractViolated(PhaseBContractError):
    """Raised only for actual cache/PKV/rope evidence."""


class ResourceContractExceeded(RuntimeError):
    """Raised for an allocator or host resource violation."""


class ProvenanceContractViolated(RuntimeError):
    """Raised for immutable external-input provenance loss."""


def _load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise PhaseBContractError(f"cannot load B-IPC1 frozen helper {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--authorized-plan-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _timeout_handler(_signum: int, _frame: Any) -> None:
    raise TimeoutError("B-IPC1 internal compute wall-clock limit reached")


def _available_ram_bytes() -> int:
    path = Path("/proc/meminfo")
    if not path.is_file():
        raise ResourceContractExceeded("B-IPC1 requires Linux /proc/meminfo")
    fields = dict(line.split(":", 1) for line in path.read_text(encoding="ascii").splitlines())
    value = fields.get("MemAvailable", "").strip()
    if not value.endswith(" kB"):
        raise ResourceContractExceeded("B-IPC1 available-RAM evidence is malformed")
    return int(value.removesuffix(" kB")) * 1024


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WORKTREE, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_parent(commit: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", f"{commit}^"], cwd=WORKTREE, check=True, capture_output=True, text=True
    ).stdout.strip()


def _memory(torch: Any) -> dict[str, int]:
    return {
        "current_allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "current_reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "maximum_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "maximum_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }


def _memory_checkpoint(torch: Any, audit: dict[str, Any], label: str) -> None:
    snapshot = {"checkpoint": label, **_memory(torch)}
    audit.setdefault("memory_ledger", []).append(snapshot)
    if any(value > CUDA_MEMORY_CAP_BYTES for key, value in snapshot.items() if key.endswith("_bytes")):
        raise ResourceContractExceeded(f"B-IPC1 CUDA memory cap exceeded at {label}")


def _deduplicated_use_cache_configs(model: Any) -> dict[int, tuple[Any, Any]]:
    candidates = [getattr(module, "config", None) for module in model.modules()]
    candidates.append(getattr(model, "generation_config", None))
    return {
        id(config): (config, config.use_cache)
        for config in candidates
        if config is not None and hasattr(config, "use_cache")
    }


def _cache_class_identity(cls: type, *, hic0: ModuleType) -> dict[str, str]:
    module = __import__(cls.__module__, fromlist=["__name__"])
    path = Path(module.__file__).resolve(strict=True)
    if not path.is_relative_to(EXPECTED_ENV.resolve(strict=True)):
        raise CacheContractViolated(f"B-IPC1 cache class outside frozen environment: {cls}")
    distribution = "flash-linear-attention" if cls.__module__.split(".", 1)[0] == "fla" else "transformers"
    identity = {
        "fqcn": f"{cls.__module__}.{cls.__qualname__}",
        "module_path": str(path),
        "module_sha256": file_sha256(path),
        "distribution": f"{distribution}=={importlib.metadata.version(distribution)}",
    }
    expected = {item[0]: item for item in hic0.EXPECTED_CACHE_CLASSES}.get(identity["fqcn"])
    if (
        expected is None
        or not str(path).endswith(expected[1])
        or identity["module_sha256"] != expected[2]
        or identity["distribution"] != expected[3]
    ):
        raise CacheContractViolated(f"B-IPC1 cache class identity differs: {identity['fqcn']}")
    return identity


class _CacheGuard:
    def __init__(self, model: Any, *, transformers: Any, hic0: ModuleType, calls: Sequence[dict[str, Any]]) -> None:
        self.model = model
        self.transformers = transformers
        self.hic0 = hic0
        self.schedule = list(calls)
        self.expected = build_cache_guard_labels(self.schedule)
        self.labels: list[str] = []
        self.base = transformers.cache_utils.Cache
        self.initial_classes = hic0.recursive_subclass_closure(self.base)
        self.patched_classes: set[type] = set()
        self.stack = contextlib.ExitStack()
        self.configs: dict[int, tuple[Any, Any]] = {}
        self.calls = 0
        self.trips = 0
        self.closure_checks = 0
        self.restored = False

    @staticmethod
    def _forbidden(cls: type, *_args: Any, **_kwargs: Any) -> None:
        del cls
        raise CacheContractViolated("B-IPC1 Cache subclass construction attempted")

    def _trip(self, cls: type, *_args: Any, **_kwargs: Any) -> None:
        self.trips += 1
        raise CacheContractViolated(f"B-IPC1 cache allocation attempted: {cls.__module__}.{cls.__qualname__}")

    def _verify_closure(self) -> None:
        new = self.hic0.recursive_subclass_closure(self.base) - self.patched_classes
        if new:
            raise CacheContractViolated(
                f"B-IPC1 unpatched cache classes appeared: {sorted(cls.__qualname__ for cls in new)}"
            )
        if any(config.use_cache is not False for config, _original in self.configs.values()):
            raise CacheContractViolated("B-IPC1 use_cache closure reopened")
        self.closure_checks += 1

    def _append(self, label: str) -> None:
        self._verify_closure()
        self.labels.append(label)

    def _restore(self) -> None:
        stack_error = None
        try:
            self.stack.close()
        except BaseException as error:
            stack_error = error
        for config, original in self.configs.values():
            config.use_cache = original
        if any(config.use_cache is not original for config, original in self.configs.values()):
            raise CacheContractViolated("B-IPC1 cache config originals did not restore")
        self.restored = True
        if stack_error is not None:
            raise stack_error

    def __enter__(self) -> "_CacheGuard":
        try:
            self.configs = _deduplicated_use_cache_configs(self.model)
            for config, _original in self.configs.values():
                config.use_cache = False
            classes = self.hic0.ordered_subclass_closure(self.base)
            if set(classes) != self.initial_classes or len(classes) != 8:
                raise CacheContractViolated("B-IPC1 pinned cache census differs")
            for cls in classes:
                _cache_class_identity(cls, hic0=self.hic0)
                replacement = self._trip if cls is self.transformers.cache_utils.DynamicCache else self._forbidden
                self.stack.enter_context(mock.patch.object(cls, "__new__", replacement))
                self.patched_classes.add(cls)
            try:
                self.transformers.cache_utils.DynamicCache()
            except CacheContractViolated:
                pass
            else:
                raise CacheContractViolated("B-IPC1 DynamicCache negative control did not trip")
            if self.trips != 1:
                raise CacheContractViolated("B-IPC1 DynamicCache trip count differs")
            self._append("CACHE_GUARD_ENTRY")
        except BaseException:
            self._restore()
            raise
        return self

    def call(self, call: dict[str, Any], **kwargs: Any) -> Any:
        expected = self.schedule[self.calls]
        if call != expected:
            raise PhaseBContractError("B-IPC1 runtime model-call order differs")
        index = expected["call_index"]
        self.model.model.rope_deltas = None
        self._append(f"CACHE_GUARD_PRE_IPC1_C{index:04d}")
        output = self.model(past_key_values=None, use_cache=False, return_dict=True, **kwargs)
        if getattr(output, "past_key_values", None) is not None or self.model.model.rope_deltas is not None:
            raise CacheContractViolated(f"B-IPC1 cache/rope closure failed at call {index}")
        self.calls += 1
        self._append(f"CACHE_GUARD_POST_IPC1_C{index:04d}")
        return output

    def final(self) -> None:
        if self.calls != len(self.schedule) or self.trips != 1:
            raise CacheContractViolated("B-IPC1 cache call/trip count differs")
        self._append("CACHE_GUARD_FINAL")

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        try:
            self._append("CACHE_GUARD_EXIT")
        finally:
            self._restore()

    def evidence(self) -> dict[str, Any]:
        prefix = self.labels[:-1] if self.labels[-1:] == ["CACHE_GUARD_EXIT"] else self.labels
        return {
            "complete": self.labels == self.expected,
            "labels": self.labels,
            "label_count": len(self.labels),
            "canonical_label_sha256": canonical_bank_sha256(self.labels),
            "expected_label_sha256": canonical_bank_sha256(self.expected),
            "exact_prefix": self.expected[: len(prefix)] == prefix,
            "exit_recorded": self.labels[-1:] == ["CACHE_GUARD_EXIT"],
            "dynamic_cache_trip_count": self.trips,
            "closure_check_count": self.closure_checks,
            "closure_checked_at_every_label": self.closure_checks == len(self.labels),
            "restored_in_finally": self.restored,
            "model_calls": self.calls,
            "recursively_closed_config_count": len(self.configs),
            "classes": [
                _cache_class_identity(cls, hic0=self.hic0) for cls in self.hic0.ordered_subclass_closure(self.base)
            ],
        }


def _validate_host_plan(plan: dict[str, Any], args: argparse.Namespace) -> None:
    validate_ipc1_plan(plan, require_authorized=True)
    if file_sha256(args.plan) != args.authorized_plan_sha256:
        raise ProvenanceContractViolated("B-IPC1 authorized plan file hash differs")
    if args.execution_commit != _git_head() or plan["mechanism_code_commit"] != _git_parent(args.execution_commit):
        raise ProvenanceContractViolated("B-IPC1 deployed exact-parent commit binding differs")
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=WORKTREE,
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise ProvenanceContractViolated("B-IPC1 deployed worktree is dirty")
    expected_environment = {
        "uv": "/home/ubuntu/.local/bin/uv",
        "uv_project": "/home/ubuntu/rlm/prime-rl",
        "UV_PROJECT_ENVIRONMENT": str(EXPECTED_ENV),
        "PYTHONPATH": EXPECTED_PYTHONPATH,
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if plan.get("execution_environment") != expected_environment:
        raise PhaseBContractError("B-IPC1 execution environment plan differs")
    if Path(sys.prefix).resolve() != EXPECTED_ENV.resolve(strict=True):
        raise ProvenanceContractViolated("B-IPC1 shared Python environment differs")
    for name in (
        "UV_PROJECT_ENVIRONMENT",
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        if os.environ.get(name) != expected_environment[name]:
            raise ProvenanceContractViolated(f"B-IPC1 environment value differs: {name}")
    if args.output_dir != Path(plan["outputs"]["directory"]):
        raise PhaseBContractError("B-IPC1 output namespace differs")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise PhaseBContractError("B-IPC1 output namespace is not fresh")
    if not args.output_dir.parent.is_dir() or args.output_dir.parent.is_symlink():
        raise ResourceContractExceeded("B-IPC1 output parent is absent or symlinked")
    if shutil.disk_usage(args.output_dir.parent).free < MINIMUM_FREE_DISK_BYTES:
        raise ResourceContractExceeded("B-IPC1 has less than 60 GiB free disk")
    if _available_ram_bytes() < MINIMUM_HOST_RAM_BYTES:
        raise ResourceContractExceeded("B-IPC1 has less than 64 GiB available RAM")


def _bank_record(plan: dict[str, Any], split: str) -> dict[str, Any]:
    records = plan["banks"]
    return next(record for record in records if record["split"] == split)


def _immutable_input_records(plan: dict[str, Any], *, require_match: bool = True) -> list[dict[str, str]]:
    specifications = [
        ("plan", plan["_path"], plan["_file_sha256"]),
        ("bank_manifest", plan["bank_manifest"]["path"], plan["bank_manifest"]["sha256"]),
        ("overlap_closure", plan["overlap_closure"]["path"], plan["overlap_closure"]["sha256"]),
        ("runtime_source", plan["runtime_source"]["path"], plan["runtime_source"]["sha256"]),
        ("generator_source", plan["generator_source"]["path"], plan["generator_source"]["sha256"]),
        ("taskset_source", plan["taskset_source"]["path"], plan["taskset_source"]["sha256"]),
        ("terminal_proof_manifest", plan["terminal_proof"]["manifest_path"], plan["terminal_proof"]["manifest_sha256"]),
    ]
    specifications.extend(
        (f"antecedent_{record['name']}", record["binding_path"], record["binding_sha256"])
        for record in plan["antecedents"]
    )
    for bank in plan["banks"]:
        specifications.extend(
            (
                (f"{bank['split']}_selection", bank["selection_path"], bank["selection_sha256"]),
                (f"{bank['split']}_parquet", bank["parquet_path"], bank["parquet_sha256"]),
            )
        )
    records = [
        {"name": name, "path": path, "expected_sha256": expected, "observed_sha256": file_sha256(Path(path))}
        for name, path, expected in specifications
    ]
    if require_match and any(record["expected_sha256"] != record["observed_sha256"] for record in records):
        raise ProvenanceContractViolated("B-IPC1 immutable input postflight hash differs")
    return records


def _validate_terminal_proof_binding(plan: dict[str, Any]) -> None:
    binding = plan["terminal_proof"]
    path = Path(binding["manifest_path"])
    if file_sha256(path) != binding["manifest_sha256"]:
        raise ProvenanceContractViolated("B-IPC1 terminal proof manifest hash differs")
    manifest = load_json_file(path)
    if (
        manifest.get("schema_version") != "q35-2b-phase-b-ipc1-terminal-proof-closure/v1"
        or manifest.get("proof_execution_commit") != binding["proof_execution_commit"]
        or manifest.get("runner_sha256") != binding["runner_sha256"]
    ):
        raise ProvenanceContractViolated("B-IPC1 terminal proof manifest identity differs")
    proof = manifest.get("successful_exact_host_proof")
    if not isinstance(proof, dict):
        raise ProvenanceContractViolated("B-IPC1 successful terminal proof record is absent")
    directory = path.parent
    encoded_path = directory / proof["proof_base64_path"]
    encoded = encoded_path.read_bytes()
    if (
        file_sha256(encoded_path) != proof["proof_base64_file_sha256"]
        or not encoded.endswith(b"\n")
        or any(byte in b" \t\r\n" for byte in encoded[:-1])
    ):
        raise ProvenanceContractViolated("B-IPC1 terminal proof encoding differs")
    try:
        decoded = base64.b64decode(encoded[:-1], validate=True)
    except ValueError as error:
        raise ProvenanceContractViolated("B-IPC1 terminal proof base64 is invalid") from error
    decoded_receipt = strict_json_loads(decoded)
    if (
        hashlib.sha256(decoded).hexdigest() != binding["decoded_proof_sha256"]
        or decoded_receipt.get("execution_commit") != binding["proof_execution_commit"]
        or decoded_receipt.get("runner_sha256") != binding["runner_sha256"]
        or decoded_receipt.get("model_loaded") is not False
        or decoded_receipt.get("cuda_initialized") is not False
        or decoded_receipt.get("exact_host_repository") is not True
        or decoded_receipt.get("maximal_success", {}).get("file_sha256") != binding["maximal_success_file_sha256"]
        or len(decoded_receipt.get("failure_terminals", [])) != binding["failure_terminal_count"]
        or len(decoded_receipt.get("tamper_cases_rejected", [])) != binding["tamper_cases_rejected"]
        or decoded_receipt.get("mapping_insertion_permutation_canonical_equal")
        is not binding["mapping_insertion_permutation_canonical_equal"]
        or decoded_receipt.get("global_exactly_one_terminal") is not binding["global_exactly_one_terminal"]
    ):
        raise ProvenanceContractViolated("B-IPC1 decoded terminal proof differs")
    for prefix in ("log", "exit_status"):
        if file_sha256(directory / proof[f"{prefix}_path"]) != proof[f"{prefix}_sha256"]:
            raise ProvenanceContractViolated(f"B-IPC1 terminal proof {prefix} differs")
    failed = manifest.get("superseded_preexecution_command_failure")
    if not isinstance(failed, dict):
        raise ProvenanceContractViolated("B-IPC1 superseded proof invocation record is absent")
    if (
        file_sha256(directory / failed["log_path"]) != binding["superseded_invocation_log_sha256"]
        or file_sha256(directory / failed["exit_status_path"]) != binding["superseded_invocation_exit_sha256"]
    ):
        raise ProvenanceContractViolated("B-IPC1 superseded proof invocation bytes differ")


def _validate_antecedents(plan: dict[str, Any]) -> None:
    antecedents = plan.get("antecedents")
    if not isinstance(antecedents, list) or [item.get("name") for item in antecedents] != [
        "HIC0_R1",
        "B1R",
    ]:
        raise PhaseBContractError("B-IPC1 antecedents are not ordered named records")
    hic0, b1r = antecedents
    if (
        hic0.get("classification") != "descriptive_antecedent_only_terminal_contract_invalid"
        or hic0.get("success_file_sha256") != "26dfb2c8942b767c0ca8697cc66eea9c0e0931123aa4ffd9c7426440323245c8"
        or hic0.get("internal_receipt_sha256") != "108342e04a3255afedbbf60d6dc8ccf86c5f0d2736b549634edbdeb8febbafc3"
        or hic0.get("validator_error") != "B-HIC0 SUCCESS row/count evidence differs"
        or hic0.get("candidate_reused") is not False
        or hic0.get("rows_reused") is not False
        or hic0.get("seeds_reused") is not False
    ):
        raise PhaseBContractError("B-IPC1 HIC0 invalid descriptive antecedent differs")
    binding = load_json_file(Path(hic0["binding_path"]))
    if file_sha256(Path(hic0["binding_path"])) != hic0["binding_sha256"] or binding != {
        key: value for key, value in hic0.items() if key not in {"name", "binding_path", "binding_sha256"}
    }:
        raise ProvenanceContractViolated("B-IPC1 HIC0 antecedent binding differs")
    b1r_binding = load_json_file(Path(b1r["binding_path"]))
    if (
        file_sha256(Path(b1r["binding_path"])) != b1r["binding_sha256"]
        or b1r_binding.get("disposition") != "b1_not_nominated"
        or b1r_binding.get("success_receipt_sha256")
        != "4cb2bf1e7e884f24c297381dc30698bce4dac2586e3f4522f231600ad76ef761"
        or b1r_binding.get("candidate_reuse") is not False
    ):
        raise ProvenanceContractViolated("B-IPC1 B1R negative antecedent differs")


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if not args.plan.is_file():
        raise ProvenanceContractViolated("B-IPC1 plan file is absent")
    plan = load_json_file(args.plan)
    _validate_host_plan(plan, args)
    _validate_antecedents(plan)
    _validate_terminal_proof_binding(plan)
    paths = {
        "bank_manifest": Path(plan["bank_manifest"]["path"]),
        "overlap_closure": Path(plan["overlap_closure"]["path"]),
        "runtime_source": Path(plan["runtime_source"]["path"]),
        "generator_source": Path(plan["generator_source"]["path"]),
        "taskset_source": Path(plan["taskset_source"]["path"]),
        "terminal_proof": Path(plan["terminal_proof"]["manifest_path"]),
    }
    expected = {
        "bank_manifest": plan["bank_manifest"]["sha256"],
        "overlap_closure": plan["overlap_closure"]["sha256"],
        "runtime_source": plan["runtime_source"]["sha256"],
        "generator_source": plan["generator_source"]["sha256"],
        "taskset_source": plan["taskset_source"]["sha256"],
        "terminal_proof": plan["terminal_proof"]["manifest_sha256"],
    }
    for bank in plan["banks"]:
        paths[f"{bank['split']}_selection"] = Path(bank["selection_path"])
        paths[f"{bank['split']}_parquet"] = Path(bank["parquet_path"])
        expected[f"{bank['split']}_selection"] = bank["selection_sha256"]
        expected[f"{bank['split']}_parquet"] = bank["parquet_sha256"]
    observed = {name: file_sha256(path) for name, path in paths.items()}
    if observed != expected:
        raise ProvenanceContractViolated("B-IPC1 immutable bank artifact closure differs")
    manifest = load_json_file(paths["bank_manifest"])
    closure = load_json_file(paths["overlap_closure"])
    if (
        manifest.get("freshness", {}).get("all_zero") is not True
        or closure.get("overlap_evidence", {}).get("all_zero") is not True
    ):
        raise PhaseBContractError("B-IPC1 bank overlap proof is incomplete")
    if closure.get("overlap_evidence", {}).get("selected_rows_permanently_excluded_from_future_training") != 96:
        raise PhaseBContractError("B-IPC1 future-training exclusion count differs")
    selections = {split: load_json_file(paths[f"{split}_selection"]) for split in ("train", "validation", "heldout")}
    for split, selection in selections.items():
        bank = _bank_record(plan, split)
        if (
            selection.get("ordered_task_key_sha256") != bank["ordered_task_key_sha256"]
            or selection.get("ordered_key_action_sha256") != bank["ordered_key_action_sha256"]
            or selection.get("row_list_canonical_sha256") != bank["row_list_canonical_sha256"]
        ):
            raise PhaseBContractError(f"B-IPC1 {split} selection hashes differ")
    train_keys = [record["task_key"] for record in selections["train"]["selected"]]
    validation_keys = [record["task_key"] for record in selections["validation"]["selected"]]
    heldout_keys = [record["task_key"] for record in selections["heldout"]["selected"]]
    schedules = {
        "validation_reject": build_model_call_schedule(train_keys, validation_keys, heldout_keys, open_heldout=False),
        "heldout_open": build_model_call_schedule(train_keys, validation_keys, heldout_keys, open_heldout=True),
    }
    for name, schedule in schedules.items():
        frozen = plan["schedules"][name]
        if (
            canonical_bank_sha256(schedule) != frozen["model_call_schedule_sha256"]
            or canonical_bank_sha256(build_cache_guard_labels(schedule)) != frozen["cache_label_sha256"]
            or canonical_bank_sha256(build_memory_checkpoint_labels(schedule)) != frozen["memory_label_sha256"]
        ):
            raise PhaseBContractError(f"B-IPC1 {name} schedule hash differs")
    return {
        "plan": plan | {"_path": str(args.plan), "_file_sha256": args.authorized_plan_sha256},
        "paths": paths,
        "selections": selections,
        "schedules": schedules,
        "resources": {
            "available_host_ram_bytes": _available_ram_bytes(),
            "free_disk_bytes": shutil.disk_usage(args.output_dir.parent).free,
        },
    }


def _ordered_rows(parquet: Any, path: Path, selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows = parquet.read_table(path).to_pylist()
    by_key = {row.get("task_key"): row for row in rows}
    ordered_records = selection.get("selected")
    if (
        not isinstance(ordered_records, list)
        or [record.get("position") for record in ordered_records] != list(range(len(ordered_records)))
        or any(record.get("task_key") not in by_key for record in ordered_records)
    ):
        raise PhaseBContractError("B-IPC1 selection records differ from parquet")
    ordered = [by_key[record["task_key"]] for record in ordered_records]
    if canonical_bank_sha256(ordered) != selection["row_list_canonical_sha256"]:
        raise PhaseBContractError("B-IPC1 selected parquet row-list hash differs")
    if any(row["action"] != record["expected_action"] for row, record in zip(ordered, ordered_records, strict=True)):
        raise PhaseBContractError("B-IPC1 selected parquet actions differ")
    return ordered


def _render_split(
    split: str, context: dict[str, Any], *, tokenizer: Any, parquet: Any, b1: ModuleType, smoke: ModuleType
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection = context["selections"][split]
    rows = _ordered_rows(parquet, context["paths"][f"{split}_parquet"], selection)
    rendered, proofs = b1._render_rows(rows, tokenizer=tokenizer, smoke=smoke)
    if [row["task_key"] for row in rendered] != [record["task_key"] for record in selection["selected"]]:
        raise PhaseBContractError(f"B-IPC1 {split} tokenizer row order differs")
    return rendered, proofs


def tokenizer_preflight(context: dict[str, Any], *, parquet: Any, AutoTokenizer: Any) -> dict[str, Any]:
    plan = context["plan"]
    smoke = _load_module(BR5_RUNNER, "b_ipc1_br5_runtime")
    b1 = _load_module(B1_RUNNER, "b_ipc1_b1_runtime")
    model_path = Path(plan["protected_model"]["path"])
    if file_sha256(smoke._model_file(model_path)) != plan["protected_model"]["weight_sha256"]:
        raise ProvenanceContractViolated("B-IPC1 e33 weight file differs")
    if smoke._metadata_hashes(model_path) != plan["model_metadata_sha256"]:
        raise ProvenanceContractViolated("B-IPC1 e33 metadata differs")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    train, train_proofs = _render_split("train", context, tokenizer=tokenizer, parquet=parquet, b1=b1, smoke=smoke)
    validation, validation_proofs = _render_split(
        "validation", context, tokenizer=tokenizer, parquet=parquet, b1=b1, smoke=smoke
    )
    probes = [
        {"selection_index": index, "task_key": train[index]["task_key"], "action": train[index]["action"]}
        for index in (0, 1, 2, 5)
    ]
    if canonical_bank_sha256(probes) != plan["pre_update_mechanism_gate"]["probe_identity_sha256"]:
        raise PhaseBContractError("B-IPC1 pre-update probe identities differ")
    if smoke._cuda_initialized_if_torch_loaded():
        raise PhaseBContractError("CUDA initialized during B-IPC1 tokenizer-only preflight")
    return {
        **context,
        "smoke": smoke,
        "b1": b1,
        "tokenizer": tokenizer,
        "rendered": {"train": train, "validation": validation},
        "render_proofs": {"train": train_proofs, "validation": validation_proofs},
        "probes": probes,
    }


def _guard_call(
    guard: _CacheGuard,
    audit: dict[str, Any],
    *,
    phase: str,
    kind: str,
    task_key: str,
    arm: str,
    backward: bool = False,
    **kwargs: Any,
) -> Any:
    call = {
        "call_index": guard.calls + 1,
        "phase": phase,
        "kind": kind,
        "task_key": task_key,
        "arm": arm,
        "backward": backward,
    }
    audit.update({"stage": phase, "task_key": task_key, "arm": arm, "call_index": call["call_index"]})
    output = guard.call(call, **kwargs)
    _memory_checkpoint(audit["torch"], audit, f"call:{call['call_index']:04d}:complete")
    return output


def _prepare_sources(
    rendered: Sequence[dict[str, Any]],
    *,
    phase: str,
    model: Any,
    guard: _CacheGuard,
    audit: dict[str, Any],
    torch: Any,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rendered:
        plain = torch.tensor([row["plain_ids"]], dtype=torch.long, device="cuda:0")
        full = torch.tensor([row["full_ids"]], dtype=torch.long, device="cuda:0")
        with torch.no_grad():
            output = _guard_call(
                guard,
                audit,
                phase=phase,
                kind="source_capture",
                task_key=row["task_key"],
                arm="SOURCE",
                input_ids=plain,
                attention_mask=torch.ones_like(plain),
                position_ids=torch.arange(plain.shape[1], device="cuda:0")[None],
                output_hidden_states=True,
                logits_to_keep=1,
            )
        captured = output.hidden_states[-1][:, -SLOTS:, :].detach().cpu()
        if captured.shape != (1, SLOTS, 2048) or not bool(torch.isfinite(captured).all()):
            raise PhaseBContractError(f"B-IPC1 source capture is invalid: {row['task_key']}")
        labels = full.clone()
        labels[:, : len(row["open_ids"])] = -100
        geometry = audit["hic0"].aligned_suffix_geometry(
            total=full.shape[1], supervised_start=len(row["open_ids"]), insertion_index=plain.shape[1]
        )
        if geometry["I"] < SLOTS:
            raise PhaseBContractError("B-IPC1 receiver does not have eight preceding token slots")
        examples.append(
            {
                **row,
                "full_tensor": full,
                "full_labels": labels,
                "geometry": geometry,
                "captured_hidden": captured,
            }
        )
        del output, plain
    return examples


def _receiver_inputs(
    example: dict[str, Any], base_embeddings: Any, residual: Any | None, *, zero: bool, torch: Any
) -> tuple[dict[str, Any], int]:
    geometry = example["geometry"]
    embeddings = base_embeddings if residual is None and not zero else base_embeddings.clone()
    if residual is not None or zero:
        target = slice(geometry["I"] - SLOTS, geometry["I"])
        value = (
            torch.zeros_like(residual)
            if zero and residual is not None
            else torch.zeros_like(base_embeddings[:, target, :])
        )
        if not zero:
            value = residual
        embeddings[:, target, :] = base_embeddings[:, target, :] + value
    labels = example["full_labels"]
    predictor = geometry["B"]
    suffix_labels = labels[:, predictor:]
    if suffix_labels.shape[1] != geometry["K"] or int(suffix_labels[0, 0]) != -100:
        raise PhaseBContractError("B-IPC1 aligned in-place suffix labels differ")
    return {
        "inputs_embeds": embeddings,
        "attention_mask": torch.ones_like(example["full_tensor"]),
        "position_ids": torch.arange(geometry["T"], device="cuda:0")[None],
        "labels": suffix_labels,
        "full_labels": labels,
        "logits_to_keep": geometry["K"],
        "output_hidden_states": True,
    }, predictor


def _metric(
    output: Any, example: dict[str, Any], predictor: int, inputs: dict[str, Any], *, audit: dict[str, Any]
) -> dict[str, Any]:
    metric = audit["hic0"]._metric(
        output,
        example,
        predictor,
        inputs,
        smoke=audit["smoke"],
        torch=audit["torch"],
    )
    return metric


def _reporting_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metric.items() if key not in {"final_hidden", "first_suffix_logits"}}


def _base_branch_margins(output: Any, example: dict[str, Any], *, torch: Any) -> list[float]:
    result: list[float] = []
    logits = output.logits.detach().float()
    for branch in example["action_trie"]["branches"]:
        row = logits[0, branch["logit_offset"]]
        value = row[branch["correct_token_id"]] - row[branch["other_token_ids"]].max()
        if not bool(torch.isfinite(value)):
            raise PhaseBContractError("B-IPC1 BASE branch margin is nonfinite")
        result.append(float(value.cpu()))
    if not result:
        raise PhaseBContractError("B-IPC1 BASE branch set is empty")
    return result


def _candidate_residual(
    example: dict[str, Any], arm: str, codec: Any, sidecar: Any | None, shell: Any, *, torch: Any, depth: int = 4
) -> tuple[Any, Any, Any]:
    hidden = example["captured_hidden"].to("cuda:0")
    mask = torch.ones(hidden.shape[:2], dtype=torch.long, device="cuda:0")
    anchor = codec.encode(hidden, mask)
    if arm == "STATIC":
        visible = anchor
    elif arm == "FFN":
        visible = sidecar(anchor)
    elif arm == "RECURRENT":
        visible = sidecar.rollout(anchor, depth).visible_workspace
    else:
        raise PhaseBContractError(f"B-IPC1 unknown candidate arm {arm}")
    residual = codec.decode(visible, shell)
    if residual.shape != (1, SLOTS, 2048) or not bool(torch.isfinite(residual).all()):
        raise PhaseBContractError("B-IPC1 candidate residual is invalid")
    return residual, anchor, visible


def _candidate_forward(
    example: dict[str, Any],
    *,
    phase: str,
    call_arm: str,
    candidate_arm: str,
    codec: Any,
    sidecar: Any | None,
    depth: int,
    model: Any,
    shell: Any,
    guard: _CacheGuard,
    audit: dict[str, Any],
    torch: Any,
    backward: bool = False,
    zero_receiver: bool = False,
) -> tuple[Any, dict[str, Any], Any, Any, Any]:
    residual, anchor, visible = _candidate_residual(
        example, candidate_arm, codec, sidecar, shell, torch=torch, depth=depth
    )
    base_embeddings = model.get_input_embeddings()(example["full_tensor"]).detach()
    inputs, predictor = _receiver_inputs(example, base_embeddings, residual, zero=zero_receiver, torch=torch)
    model_inputs = {key: value for key, value in inputs.items() if key != "full_labels"}
    output = _guard_call(
        guard,
        audit,
        phase=phase,
        kind="receiver",
        task_key=example["task_key"],
        arm=call_arm,
        backward=backward,
        **model_inputs,
    )
    metric = _metric(output, example, predictor, inputs, audit=audit)
    del base_embeddings
    return output, metric, residual, anchor, visible


def _initialize_modules(
    torch: Any,
    LocalDepthCodec: Any,
    OneShotFeedForwardSidecar: Any,
    TimestepFreeRecurrentSidecar: Any,
    smoke: ModuleType,
):
    torch.manual_seed(INITIALIZATION_SEED)
    torch.cuda.manual_seed_all(INITIALIZATION_SEED)
    template = LocalDepthCodec(2048, 256, SLOTS, initial_receiver_gate=0.001).to(device="cuda:0", dtype=torch.bfloat16)
    template_state = {name: tensor.detach().clone() for name, tensor in template.state_dict().items()}
    codecs = {
        arm: LocalDepthCodec(2048, 256, SLOTS, initial_receiver_gate=0.001).to(device="cuda:0", dtype=torch.bfloat16)
        for arm in TRAINING_ARMS
    }
    for codec in codecs.values():
        codec.load_state_dict(template_state, strict=True)
    sidecars = {
        "FFN": OneShotFeedForwardSidecar().to(device="cuda:0", dtype=torch.bfloat16),
        "RECURRENT": TimestepFreeRecurrentSidecar().to(device="cuda:0", dtype=torch.bfloat16),
    }
    del template, template_state
    modules = [
        {"name": "STATIC.codec", "module": codecs["STATIC"]},
        {"name": "FFN.codec", "module": codecs["FFN"]},
        {"name": "FFN.sidecar", "module": sidecars["FFN"]},
        {"name": "RECURRENT.codec", "module": codecs["RECURRENT"]},
        {"name": "RECURRENT.sidecar", "module": sidecars["RECURRENT"]},
    ]
    hashes = [{"name": item["name"], "sha256": smoke._module_tensor_sha256(item["module"], torch)} for item in modules]
    codec_hashes = [record["sha256"] for record in hashes if record["name"].endswith(".codec")]
    if len(set(codec_hashes)) != 1:
        raise PhaseBContractError("B-IPC1 initialized codec copies are not bitwise equal")
    if any(
        float(torch.tanh(codec.receiver_gate.detach()).cpu())
        != float(torch.tensor(0.001, dtype=codec.receiver_gate.dtype))
        for codec in codecs.values()
    ):
        raise PhaseBContractError("B-IPC1 initial receiver gate differs from BF16 0.001")
    snapshots = {
        arm: {
            "codec": {name: value.detach().cpu().clone() for name, value in codecs[arm].state_dict().items()},
            "sidecar": None
            if arm == "STATIC"
            else {name: value.detach().cpu().clone() for name, value in sidecars[arm].state_dict().items()},
        }
        for arm in TRAINING_ARMS
    }
    return codecs, sidecars, modules, hashes, snapshots


def _run_training_bases(
    examples: Sequence[dict[str, Any]], *, model: Any, guard: _CacheGuard, audit: dict[str, Any], torch: Any
) -> None:
    for example in examples:
        base_embeddings = model.get_input_embeddings()(example["full_tensor"]).detach()
        inputs, predictor = _receiver_inputs(example, base_embeddings, None, zero=False, torch=torch)
        with torch.no_grad():
            output = _guard_call(
                guard,
                audit,
                phase="pre_learning",
                kind="receiver",
                task_key=example["task_key"],
                arm="BASE",
                **{key: value for key, value in inputs.items() if key != "full_labels"},
            )
        metric = _metric(output, example, predictor, inputs, audit=audit)
        example["base_metric"] = _reporting_metric(metric)
        example["base_branch_margins"] = _base_branch_margins(output, example, torch=torch)
        example["base_identity"] = {
            "inputs_embeds": audit["smoke"]._tensor_bytes_sha256(inputs["inputs_embeds"], torch),
            "attention_mask": audit["smoke"]._tensor_bytes_sha256(inputs["attention_mask"], torch),
            "position_ids": audit["smoke"]._tensor_bytes_sha256(inputs["position_ids"], torch),
            "labels": audit["smoke"]._tensor_bytes_sha256(inputs["full_labels"], torch),
            "final_hidden": metric["hashes"]["final_hidden"],
            "first_suffix_logits": metric["hashes"]["first_suffix_logits"],
            "nll": metric["nll"],
            "margin": metric["margin"],
        }
        del output, metric, base_embeddings, inputs
        torch.cuda.empty_cache()


def _gradient_group(parameters: Sequence[tuple[str, Any]], prefixes: tuple[str, ...], *, torch: Any) -> dict[str, Any]:
    values = [
        parameter.grad for name, parameter in parameters if name.startswith(prefixes) and parameter.grad is not None
    ]
    return {
        "tensor_count": len(values),
        "finite": bool(values) and all(bool(torch.isfinite(value).all()) for value in values),
        "nonzero": bool(values) and any(bool(torch.count_nonzero(value)) for value in values),
    }


def _pre_update_gate(
    examples: Sequence[dict[str, Any]],
    *,
    model: Any,
    codecs: dict[str, Any],
    sidecars: dict[str, Any],
    shell: Any,
    guard: _CacheGuard,
    audit: dict[str, Any],
    torch: Any,
) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    backward_for = {"STATIC": 0, "FFN": 1, "RECURRENT": 2}
    initial_hashes = {
        arm: {
            "codec": audit["smoke"]._module_tensor_sha256(codecs[arm], torch),
            "sidecar": None if arm == "STATIC" else audit["smoke"]._module_tensor_sha256(sidecars[arm], torch),
        }
        for arm in TRAINING_ARMS
    }
    for index in (0, 1, 2, 5):
        example = examples[index]
        base_embeddings = model.get_input_embeddings()(example["full_tensor"]).detach()
        zero_inputs, predictor = _receiver_inputs(example, base_embeddings, None, zero=True, torch=torch)
        with torch.no_grad():
            zero_output = _guard_call(
                guard,
                audit,
                phase="pre_update_probe",
                kind="receiver",
                task_key=example["task_key"],
                arm="ZERO",
                **{key: value for key, value in zero_inputs.items() if key != "full_labels"},
            )
        zero_metric = _metric(zero_output, example, predictor, zero_inputs, audit=audit)
        zero_identity = {
            "inputs_embeds": audit["smoke"]._tensor_bytes_sha256(zero_inputs["inputs_embeds"], torch),
            "attention_mask": audit["smoke"]._tensor_bytes_sha256(zero_inputs["attention_mask"], torch),
            "position_ids": audit["smoke"]._tensor_bytes_sha256(zero_inputs["position_ids"], torch),
            "labels": audit["smoke"]._tensor_bytes_sha256(zero_inputs["full_labels"], torch),
            "final_hidden": zero_metric["hashes"]["final_hidden"],
            "first_suffix_logits": zero_metric["hashes"]["first_suffix_logits"],
            "nll": zero_metric["nll"],
            "margin": zero_metric["margin"],
        }
        if zero_identity != example["base_identity"]:
            raise MechanismRejected(f"B-IPC1 direct ZERO identity failed: {example['task_key']}")
        arm_records: list[dict[str, Any]] = []
        for arm in TRAINING_ARMS:
            sidecar = sidecars.get(arm)
            with torch.no_grad():
                zero_out, zero_arm_metric, zero_residual, zero_anchor, zero_visible = _candidate_forward(
                    example,
                    phase="pre_update_probe",
                    call_arm=f"INPLACE_ZERO_{arm}",
                    candidate_arm=arm,
                    codec=codecs[arm],
                    sidecar=sidecar,
                    depth=4,
                    model=model,
                    shell=shell,
                    guard=guard,
                    audit=audit,
                    torch=torch,
                    zero_receiver=True,
                )
            zero_arm_identity = {
                "inputs_embeds": zero_arm_metric["hashes"]["inputs_embeds"],
                "attention_mask": zero_arm_metric["hashes"]["attention_mask"],
                "position_ids": zero_arm_metric["hashes"]["position_ids"],
                "labels": zero_arm_metric["hashes"]["labels"],
                "final_hidden": zero_arm_metric["hashes"]["final_hidden"],
                "first_suffix_logits": zero_arm_metric["hashes"]["first_suffix_logits"],
                "nll": zero_arm_metric["nll"],
                "margin": zero_arm_metric["margin"],
            }
            if zero_arm_identity != example["base_identity"]:
                raise MechanismRejected(f"B-IPC1 {arm} INPLACE_ZERO identity failed: {example['task_key']}")
            do_backward = backward_for[arm] == index
            context = torch.enable_grad() if do_backward else torch.no_grad()
            with context:
                eps_out, eps_metric, residual, anchor, visible = _candidate_forward(
                    example,
                    phase="pre_update_probe",
                    call_arm=f"INPLACE_EPS_{arm}",
                    candidate_arm=arm,
                    codec=codecs[arm],
                    sidecar=sidecar,
                    depth=4,
                    model=model,
                    shell=shell,
                    guard=guard,
                    audit=audit,
                    torch=torch,
                    backward=do_backward,
                )
                if not math.isfinite(eps_metric["nll"]) or not math.isfinite(eps_metric["margin"]):
                    raise MechanismRejected(f"B-IPC1 {arm} INPLACE_EPS is nonfinite")
                gradient = None
                if do_backward:
                    residual.retain_grad()
                    objective, objective_evidence = differentiable_margin_retention_from_baseline_margins(
                        candidate_logits=eps_out.logits,
                        base_branch_margins=example["base_branch_margins"],
                        branches=example["action_trie"]["branches"],
                        action=example["action"],
                        aligned_suffix_ce=eps_out.loss,
                        torch=torch,
                    )
                    objective.backward()
                    named = [(f"codec.{name}", parameter) for name, parameter in codecs[arm].named_parameters()]
                    if sidecar is not None:
                        named.extend((f"sidecar.{name}", parameter) for name, parameter in sidecar.named_parameters())
                    codec_group = _gradient_group(named, ("codec.",), torch=torch)
                    sidecar_group = None if sidecar is None else _gradient_group(named, ("sidecar.",), torch=torch)
                    gradient = {
                        "objective": float(objective.detach()),
                        "objective_evidence": objective_evidence,
                        "residual": {
                            "finite": residual.grad is not None and bool(torch.isfinite(residual.grad).all()),
                            "nonzero": residual.grad is not None and bool(torch.count_nonzero(residual.grad)),
                        },
                        "codec": codec_group,
                        "sidecar": sidecar_group,
                        "all_named_present_gradients_finite": all(
                            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                            for _name, parameter in named
                        ),
                        "e33_gradients_absent": all(parameter.grad is None for parameter in model.parameters()),
                    }
                    if not all(
                        (
                            gradient["residual"]["finite"],
                            gradient["residual"]["nonzero"],
                            codec_group["finite"],
                            codec_group["nonzero"],
                            gradient["all_named_present_gradients_finite"],
                            gradient["e33_gradients_absent"],
                        )
                    ) or (sidecar_group is not None and not all((sidecar_group["finite"], sidecar_group["nonzero"]))):
                        raise MechanismRejected(f"B-IPC1 {arm} EPS connectivity failed")
                    for _name, parameter in named:
                        parameter.grad = None
                    _memory_checkpoint(torch, audit, f"call:{guard.calls:04d}:post_backward")
            arm_records.append(
                {
                    "name": arm,
                    "inplace_zero_identity": zero_arm_identity == example["base_identity"],
                    "inplace_eps": _reporting_metric(eps_metric),
                    "backward": gradient,
                }
            )
            del zero_out, zero_arm_metric, zero_residual, zero_anchor, zero_visible
            del eps_out, eps_metric, residual, anchor, visible
            torch.cuda.empty_cache()
        proof = {
            "selection_index": index,
            "task_key": example["task_key"],
            "action": example["action"],
            "base_equals_direct_zero": True,
            "arms": arm_records,
        }
        parsed = json.loads(canonical_terminal_bytes(proof))
        if parsed != proof:
            raise PhaseBContractError("B-IPC1 pre-update proof does not canonical-roundtrip")
        probes.append(parsed)
        del base_embeddings, zero_output, zero_metric, zero_inputs
    post_hashes = {
        arm: {
            "codec": audit["smoke"]._module_tensor_sha256(codecs[arm], torch),
            "sidecar": None if arm == "STATIC" else audit["smoke"]._module_tensor_sha256(sidecars[arm], torch),
        }
        for arm in TRAINING_ARMS
    }
    if initial_hashes != post_hashes:
        raise MechanismRejected("B-IPC1 pre-update probes changed candidate tensor state")
    return {"probes": probes, "pre_tensor_hashes": initial_hashes, "post_tensor_hashes": post_hashes}


def _parameter_grad_norms(module_records: Sequence[tuple[str, Any]], *, torch: Any) -> list[dict[str, Any]]:
    result = []
    for name, parameter in module_records:
        value = None if parameter.grad is None else float(torch.linalg.vector_norm(parameter.grad.detach().float()))
        if value is not None and not math.isfinite(value):
            raise PhaseBContractError(f"B-IPC1 nonfinite gradient: {name}")
        result.append({"name": name, "l2": value})
    return result


def _train_arm(
    arm: str,
    examples: Sequence[dict[str, Any]],
    update_records: Sequence[dict[str, Any]],
    *,
    model: Any,
    codec: Any,
    sidecar: Any | None,
    shell: Any,
    guard: _CacheGuard,
    audit: dict[str, Any],
    torch: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {example["task_key"]: example for example in examples}
    parameters = list(codec.parameters()) + ([] if sidecar is None else list(sidecar.parameters()))
    optimizer = torch.optim.AdamW(
        parameters,
        lr=0.0001,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    history: list[dict[str, Any]] = []
    group_prefixes = {
        "STATIC": {"codec": ("codec.",)},
        "FFN": {"codec": ("codec.",), "ffn_internal": ("sidecar.input_norm.", "sidecar.hidden.", "sidecar.output.")},
        "RECURRENT": {
            "codec": ("codec.",),
            "transition": ("sidecar.transition.",),
            "memory": ("sidecar.memory_candidate.", "sidecar.memory_gate.", "sidecar.memory_norm."),
            "workspace": ("sidecar.workspace_delta.",),
        },
    }[arm]
    group_updates: dict[str, list[int]] = {name: [] for name in group_prefixes}
    for update_record in update_records:
        update_index = update_record["update_index"]
        optimizer.zero_grad(set_to_none=True)
        row_records: list[dict[str, Any]] = []
        for selected in update_record["rows"]:
            example = by_key[selected["task_key"]]
            output, metric, residual, anchor, visible = _candidate_forward(
                example,
                phase="learning",
                call_arm=arm,
                candidate_arm=arm,
                codec=codec,
                sidecar=sidecar,
                depth=4,
                model=model,
                shell=shell,
                guard=guard,
                audit=audit,
                torch=torch,
                backward=True,
            )
            objective, objective_evidence = differentiable_margin_retention_from_baseline_margins(
                candidate_logits=output.logits,
                base_branch_margins=example["base_branch_margins"],
                branches=example["action_trie"]["branches"],
                action=example["action"],
                aligned_suffix_ce=output.loss,
                torch=torch,
            )
            (objective / 12).backward()
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise PhaseBContractError(f"B-IPC1 {arm} caused an e33 gradient")
            if any(
                parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                for parameter in parameters
            ):
                raise PhaseBContractError(f"B-IPC1 {arm} accumulated a nonfinite gradient")
            row_records.append(
                {
                    "task_key": example["task_key"],
                    "action": example["action"],
                    "aligned_suffix_ce": float(output.loss.detach()),
                    "total_objective": float(objective.detach()),
                    "objective_evidence": objective_evidence,
                    "reporting_nll": metric["nll"],
                    "reporting_margin": metric["margin"],
                }
            )
            del output, metric, objective, residual, anchor, visible
            torch.cuda.empty_cache()
            _memory_checkpoint(torch, audit, f"call:{guard.calls:04d}:post_backward")
        named = [(f"codec.{name}", parameter) for name, parameter in codec.named_parameters()]
        if sidecar is not None:
            named.extend((f"sidecar.{name}", parameter) for name, parameter in sidecar.named_parameters())
        gradient_norms = _parameter_grad_norms(named, torch=torch)
        groups = {name: _gradient_group(named, prefixes, torch=torch) for name, prefixes in group_prefixes.items()}
        for name, evidence in groups.items():
            if evidence["finite"] and evidence["nonzero"] and (name == "codec" or update_index in (2, 3, 4)):
                group_updates[name].append(update_index)
        preclip = float(torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0))
        if not math.isfinite(preclip):
            raise PhaseBContractError(f"B-IPC1 {arm} preclip gradient norm is nonfinite")
        _memory_checkpoint(torch, audit, f"optimizer:{arm}:update{update_index}:post_clip")
        optimizer.step()
        _memory_checkpoint(torch, audit, f"optimizer:{arm}:update{update_index}:post_step")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise PhaseBContractError(f"B-IPC1 {arm} e33 gradient appeared after optimizer step")
        scales = [] if sidecar is None else [float(sidecar.output_scale.detach().cpu())]
        history.append(
            {
                "update_index": update_index,
                "rows": row_records,
                "gradient_l2": gradient_norms,
                "preclip_global_norm": preclip,
                "sidecar_output_scale": scales,
            }
        )
    if any(not updates for updates in group_updates.values()):
        raise PhaseBContractError(f"B-IPC1 {arm} lacks finite nonzero trained-group gradients")
    if sidecar is not None and (not history[0]["sidecar_output_scale"] or history[0]["sidecar_output_scale"][0] == 0):
        raise PhaseBContractError(f"B-IPC1 {arm} sidecar output scale did not open after step1")
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    _memory_checkpoint(torch, audit, f"optimizer:{arm}:destroyed")
    return history, {"finite_nonzero_gradient_updates": group_updates, "optimizer_destroyed": True}


def _restore_pre_modules(
    snapshots: dict[str, Any],
    *,
    torch: Any,
    LocalDepthCodec: Any,
    OneShotFeedForwardSidecar: Any,
    TimestepFreeRecurrentSidecar: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    codecs = {
        arm: LocalDepthCodec(2048, 256, SLOTS, initial_receiver_gate=0.001).to(device="cuda:0", dtype=torch.bfloat16)
        for arm in TRAINING_ARMS
    }
    sidecars = {
        "FFN": OneShotFeedForwardSidecar().to(device="cuda:0", dtype=torch.bfloat16),
        "RECURRENT": TimestepFreeRecurrentSidecar().to(device="cuda:0", dtype=torch.bfloat16),
    }
    for arm in TRAINING_ARMS:
        codecs[arm].load_state_dict(snapshots[arm]["codec"], strict=True)
        if arm != "STATIC":
            sidecars[arm].load_state_dict(snapshots[arm]["sidecar"], strict=True)
        codecs[arm].eval()
        if arm != "STATIC":
            sidecars[arm].eval()
    return codecs, sidecars


def _evaluate_split(
    split: str,
    rendered: Sequence[dict[str, Any]],
    *,
    model: Any,
    post_codecs: dict[str, Any],
    post_sidecars: dict[str, Any],
    pre_codecs: dict[str, Any],
    pre_sidecars: dict[str, Any],
    shell: Any,
    guard: _CacheGuard,
    audit: dict[str, Any],
    torch: Any,
    diagnose_recurrent_states: Any,
    safety_by_arm: dict[str, bool],
) -> dict[str, Any]:
    examples = _prepare_sources(
        rendered,
        phase=split,
        model=model,
        guard=guard,
        audit=audit,
        torch=torch,
    )
    metrics = {
        "BASE": [],
        "PRE_STATIC": [],
        "PRE_FFN": [],
        "PRE_RECURRENT_T4": [],
        "POST_STATIC": [],
        "POST_FFN": [],
        **{f"POST_RECURRENT_T{depth}": [] for depth in EVALUATION_DEPTHS},
    }
    residual_checks = {arm: [] for arm in TRAINING_ARMS}
    with torch.no_grad():
        for example in examples:
            base_embeddings = model.get_input_embeddings()(example["full_tensor"]).detach()
            base_inputs, predictor = _receiver_inputs(example, base_embeddings, None, zero=False, torch=torch)
            base_output = _guard_call(
                guard,
                audit,
                phase=split,
                kind="receiver",
                task_key=example["task_key"],
                arm="BASE",
                **{key: value for key, value in base_inputs.items() if key != "full_labels"},
            )
            base_metric = _metric(base_output, example, predictor, base_inputs, audit=audit)
            metrics["BASE"].append(
                {"task_key": example["task_key"], "action": example["action"], **_reporting_metric(base_metric)}
            )
            del base_output, base_metric, base_inputs, base_embeddings
            for arm in TRAINING_ARMS:
                output, metric, residual, anchor, visible = _candidate_forward(
                    example,
                    phase=split,
                    call_arm=f"PRE_{arm}" if arm != "RECURRENT" else "PRE_RECURRENT_T4",
                    candidate_arm=arm,
                    codec=pre_codecs[arm],
                    sidecar=pre_sidecars.get(arm),
                    depth=4,
                    model=model,
                    shell=shell,
                    guard=guard,
                    audit=audit,
                    torch=torch,
                )
                metrics[f"PRE_{arm}" if arm != "RECURRENT" else "PRE_RECURRENT_T4"].append(
                    {"task_key": example["task_key"], "action": example["action"], **_reporting_metric(metric)}
                )
                del output, metric, residual, anchor, visible
            for arm in ("STATIC", "FFN"):
                output, metric, residual, anchor, visible = _candidate_forward(
                    example,
                    phase=split,
                    call_arm=f"POST_{arm}",
                    candidate_arm=arm,
                    codec=post_codecs[arm],
                    sidecar=post_sidecars.get(arm),
                    depth=4,
                    model=model,
                    shell=shell,
                    guard=guard,
                    audit=audit,
                    torch=torch,
                )
                residual_checks[arm].append(
                    {
                        "task_key": example["task_key"],
                        "finite_nonzero": bool(torch.isfinite(residual).all()) and bool(torch.count_nonzero(residual)),
                        "l2": float(torch.linalg.vector_norm(residual.detach().float()).cpu()),
                    }
                )
                metrics[f"POST_{arm}"].append(
                    {"task_key": example["task_key"], "action": example["action"], **_reporting_metric(metric)}
                )
                del output, metric, residual, anchor, visible
            hidden = example["captured_hidden"].to("cuda:0")
            mask = torch.ones(hidden.shape[:2], dtype=torch.long, device="cuda:0")
            rec_anchor = post_codecs["RECURRENT"].encode(hidden, mask)
            trajectory = post_sidecars["RECURRENT"].rollout(rec_anchor, 8, return_trajectory=True)
            diagnostic = diagnose_recurrent_states(trajectory)
            if diagnostic.nonfinite:
                raise PhaseBContractError("B-IPC1 recurrent evaluation state is nonfinite")
            retention = {
                f"T{depth}": audit["b1"]._retention(rec_anchor, trajectory[depth].visible_workspace, torch=torch)
                for depth in EVALUATION_DEPTHS
            }
            stability = audit["b1"]._stability(diagnostic)
            for depth in EVALUATION_DEPTHS:
                residual = post_codecs["RECURRENT"].decode(trajectory[depth].visible_workspace, shell)
                base_embeddings = model.get_input_embeddings()(example["full_tensor"]).detach()
                inputs, predictor = _receiver_inputs(example, base_embeddings, residual, zero=False, torch=torch)
                output = _guard_call(
                    guard,
                    audit,
                    phase=split,
                    kind="receiver",
                    task_key=example["task_key"],
                    arm=f"POST_RECURRENT_T{depth}",
                    **{key: value for key, value in inputs.items() if key != "full_labels"},
                )
                metric = _metric(output, example, predictor, inputs, audit=audit)
                row = {"task_key": example["task_key"], "action": example["action"], **_reporting_metric(metric)}
                if depth == 8:
                    row.update({"retention": retention, "stability_T8": stability})
                metrics[f"POST_RECURRENT_T{depth}"].append(row)
                if depth == 4:
                    residual_checks["RECURRENT"].append(
                        {
                            "task_key": example["task_key"],
                            "finite_nonzero": bool(torch.isfinite(residual).all())
                            and bool(torch.count_nonzero(residual)),
                            "l2": float(torch.linalg.vector_norm(residual.detach().float()).cpu()),
                        }
                    )
                del output, metric, residual, base_embeddings, inputs
            del hidden, mask, rec_anchor, trajectory, diagnostic, retention, stability
            torch.cuda.empty_cache()
    actions = [example["action"] for example in examples]
    common = []
    for arm in TRAINING_ARMS:
        suffix = arm if arm != "RECURRENT" else "RECURRENT_T4"
        result = evaluate_common_arm(
            action_order=actions,
            base=metrics["BASE"],
            pre=metrics[f"PRE_{suffix}"],
            post=metrics[f"POST_{suffix}"],
            safety_and_noncollapse=safety_by_arm[arm]
            and all(record["finite_nonzero"] and math.isfinite(record["l2"]) for record in residual_checks[arm]),
        )
        common.append({"name": arm, "value": result})
    recurrent = evaluate_recurrent_value(
        actions=actions,
        recurrent={f"T{depth}": metrics[f"POST_RECURRENT_T{depth}"] for depth in EVALUATION_DEPTHS},
        ffn=metrics["POST_FFN"],
        retention_and_stability_passed=_recurrent_retention_stability(metrics["POST_RECURRENT_T8"]),
    )
    return {
        "split": split,
        "rows": len(examples),
        "metrics": [{"name": name, "rows": rows} for name, rows in metrics.items()],
        "common_arm_gates": common,
        "recurrent_gates": recurrent,
        "post_residual_checks": [{"name": arm, "rows": residual_checks[arm]} for arm in TRAINING_ARMS],
    }


def _recurrent_retention_stability(rows: Sequence[dict[str, Any]]) -> bool:
    if len(rows) != 24:
        raise PhaseBContractError("B-IPC1 recurrent retention lacks 24 rows")
    retention = all(
        row["retention"][f"T{depth}"]["cosine"] >= 0.995
        and 0.95 <= row["retention"][f"T{depth}"]["norm_ratio"] <= 1.05
        and row["retention"][f"T{depth}"]["relative_l2"] <= 0.10
        for row in rows
        for depth in EVALUATION_DEPTHS
    )
    stability = all(
        min(row["stability_T8"]["memory_change_rms"]) > 1e-6
        and row["stability_T8"]["median_memory_contraction_steps_2_8"] <= 0.90
        and row["stability_T8"]["max_memory_contraction_steps_2_8"] <= 1.25
        and row["stability_T8"]["memory_oscillation_rate"] <= 0.25
        and row["stability_T8"]["finite"] is True
        for row in rows
    )
    aggregate = math.fsum(row["stability_T8"]["memory_oscillation_rate"] for row in rows) / 24 <= 0.10
    return retention and stability and aggregate


def _module_delta_groups(
    arm: str, codec: Any, sidecar: Any | None, snapshot: dict[str, Any], *, torch: Any
) -> list[dict[str, Any]]:
    current = {
        **{f"codec.{name}": value.detach().cpu().float() for name, value in codec.state_dict().items()},
        **(
            {}
            if sidecar is None
            else {f"sidecar.{name}": value.detach().cpu().float() for name, value in sidecar.state_dict().items()}
        ),
    }
    initial = {
        **{f"codec.{name}": value.float() for name, value in snapshot["codec"].items()},
        **(
            {}
            if snapshot["sidecar"] is None
            else {f"sidecar.{name}": value.float() for name, value in snapshot["sidecar"].items()}
        ),
    }
    groups = {
        "codec_encoder": ("codec.source_norm.", "codec.source_projection."),
        "codec_receiver": ("codec.workspace_norm.", "codec.receiver_projection.", "codec.receiver_gate"),
    }
    if arm == "FFN":
        groups |= {
            "ffn_internal": ("sidecar.input_norm.", "sidecar.hidden.", "sidecar.output."),
            "ffn_output_scale": ("sidecar.output_scale",),
        }
    elif arm == "RECURRENT":
        groups |= {
            "recurrent_transition": ("sidecar.anchor_norm.", "sidecar.visible_norm.", "sidecar.transition."),
            "recurrent_memory": (
                "sidecar.memory_norm.",
                "sidecar.memory_candidate.",
                "sidecar.memory_gate.",
            ),
            "recurrent_workspace": ("sidecar.workspace_delta.",),
            "recurrent_output_scale": ("sidecar.output_scale",),
        }
    records = []
    for name, prefixes in groups.items():
        values = [current[key] - initial[key] for key in current if key.startswith(prefixes)]
        if not values:
            raise PhaseBContractError(f"B-IPC1 trained parameter group is empty: {arm}/{name}")
        square_sum = math.fsum(float(value.square().sum()) for value in values)
        l2 = math.sqrt(square_sum)
        records.append({"name": name, "finite": math.isfinite(l2), "nonzero": l2 > 0.0, "delta_l2": l2})
    return records


def _common_by_arm(evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["name"]: record["value"] for record in evaluation["common_arm_gates"]}


def _status_for_evaluations(validation: dict[str, Any], heldout: dict[str, Any] | None) -> tuple[str, list[str]]:
    validation_pass = [arm for arm, result in _common_by_arm(validation).items() if result["passed"]]
    if not validation_pass:
        if heldout is not None:
            raise PhaseBContractError("B-IPC1 heldout was opened after validation rejected all arms")
        return "b_ipc1_validation_not_nominated", []
    if heldout is None:
        raise PhaseBContractError("B-IPC1 heldout was not opened after validation passed")
    heldout_common = _common_by_arm(heldout)
    jointly_passing = [arm for arm in TRAINING_ARMS if arm in validation_pass and heldout_common[arm]["passed"]]
    recurrent = (
        "RECURRENT" in jointly_passing
        and validation["recurrent_gates"]["passed"]
        and heldout["recurrent_gates"]["passed"]
    )
    if recurrent:
        return "b_ipc1_inplace_learning_recurrent_nominated", jointly_passing
    if jointly_passing:
        return "b_ipc1_inplace_learning_nominated", jointly_passing
    return "b_ipc1_inplace_learning_not_nominated", []


def _state_dict_tensor_sha256(state: dict[str, Any], *, torch: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise PhaseBContractError("B-IPC1 candidate state is not a tensor mapping")
        contiguous = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(contiguous).all()):
            raise PhaseBContractError(f"B-IPC1 candidate tensor is nonfinite: {name}")
        digest.update(name.encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode())
        digest.update(contiguous.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_dict_schema(state: dict[str, Any], *, torch: Any) -> list[dict[str, Any]]:
    records = []
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise PhaseBContractError("B-IPC1 candidate state is not a tensor mapping")
        if not bool(torch.isfinite(tensor).all()):
            raise PhaseBContractError(f"B-IPC1 candidate tensor is nonfinite: {name}")
        records.append({"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    if not records:
        raise PhaseBContractError("B-IPC1 candidate state is empty")
    return records


def _candidate_state_records(
    codec_state: dict[str, Any], sidecar_state: dict[str, Any] | None, *, torch: Any
) -> list[dict[str, Any]]:
    records = [
        {
            "name": "codec",
            "tensor_sha256": _state_dict_tensor_sha256(codec_state, torch=torch),
            "schema": _state_dict_schema(codec_state, torch=torch),
        }
    ]
    if sidecar_state is not None:
        records.append(
            {
                "name": "sidecar",
                "tensor_sha256": _state_dict_tensor_sha256(sidecar_state, torch=torch),
                "schema": _state_dict_schema(sidecar_state, torch=torch),
            }
        )
    return records


def _exclusive_candidate_save(output: Path, name: str, payload: dict[str, Any], *, torch: Any) -> str:
    final = output / name
    temporary = output / f".{name}.{os.getpid()}.tmp"
    if Path(name).name != name or final.exists() or final.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise PhaseBContractError(f"B-IPC1 candidate target is not fresh and safe: {name}")
    try:
        torch.save(payload, temporary)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, final)
        _fsync_directory(output)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
            _fsync_directory(output)
    return file_sha256(final)


def _save_candidates(
    output: Path,
    codecs: dict[str, Any],
    sidecars: dict[str, Any],
    *,
    b1: ModuleType,
    torch: Any,
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for arm in TRAINING_ARMS:
        name = f"{arm}.final.pt"
        _memory_checkpoint(torch, audit, f"candidate:{arm}:before_write")
        codec_state = b1._cpu_state_dict(codecs[arm])
        sidecar_state = None if arm == "STATIC" else b1._cpu_state_dict(sidecars[arm])
        state_records = _candidate_state_records(codec_state, sidecar_state, torch=torch)
        payload = {
            "schema_version": "q35-2b-phase-b-ipc1-candidate/v1",
            "arm": arm,
            "codec": codec_state,
            "sidecar": sidecar_state,
            "module_post_state": state_records,
        }
        sha = _exclusive_candidate_save(output, name, payload, torch=torch)
        _fsync_directory(output)
        _memory_checkpoint(torch, audit, f"candidate:{arm}:after_write")
        records.append(
            {
                "name": name,
                "arm": arm,
                "sha256": sha,
                "valid_only_with_terminal": "SUCCESS.json",
                "module_post_state": state_records,
            }
        )
    if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > ARTIFACT_CAP_BYTES:
        raise ResourceContractExceeded("B-IPC1 artifacts exceed 512 MiB")
    _fsync_directory(output)
    _fsync_directory(output.parent)
    return records


def execute(
    context: dict[str, Any],
    *,
    execution_commit: str,
    output: Path,
    torch: Any,
    transformers: Any,
    AutoModelForImageTextToText: Any,
    LocalDepthCodec: Any,
    OneShotFeedForwardSidecar: Any,
    TimestepFreeRecurrentSidecar: Any,
    diagnose_recurrent_states: Any,
    parquet: Any,
    audit: dict[str, Any],
) -> dict[str, Any]:
    plan = context["plan"]
    smoke = context["smoke"]
    b1 = context["b1"]
    hic0 = _load_module(HIC0_RUNNER, "b_ipc1_hic0_runtime")
    audit.update({"torch": torch, "smoke": smoke, "b1": b1, "hic0": hic0, "stage": "runtime_validation"})
    smoke._validate_torch_runtime(plan, torch=torch)
    smoke._validate_transformers_runtime(plan, transformers=transformers)
    smoke._require_gpu0_idle()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ResourceContractExceeded("B-IPC1 requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    if torch.cuda.get_device_name(0) != "NVIDIA RTX A6000":
        raise ResourceContractExceeded("B-IPC1 requires one NVIDIA RTX A6000")
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    fraction = CUDA_MEMORY_CAP_BYTES / total_memory
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    torch.cuda.reset_peak_memory_stats(0)
    audit["allocator"] = {
        "device_total_bytes": total_memory,
        "cap_bytes": CUDA_MEMORY_CAP_BYTES,
        "requested_fraction": fraction,
        "observed_fraction": float(torch.cuda.get_per_process_memory_fraction(0)),
    }
    model_path = Path(plan["protected_model"]["path"])
    audit.update({"stage": "model_load", "model_path": model_path})
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).to("cuda:0")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise PhaseBContractError("B-IPC1 e33 is not fully frozen")
    audit.update({"model": model, "stage": "model_loaded"})
    _memory_checkpoint(torch, audit, "after_model_load")
    e33_tensor_pre = smoke._module_tensor_sha256(model, torch)
    e33_file_pre = file_sha256(smoke._model_file(model_path))
    metadata_pre = smoke._metadata_hashes(model_path)
    audit.update(
        {
            "e33_tensor_pre": e33_tensor_pre,
            "e33_file_pre": e33_file_pre,
            "metadata_pre": metadata_pre,
        }
    )
    shell = smoke._mean_embedding_norm(model.get_input_embeddings().weight, torch)
    codecs, sidecars, modules, module_pre, snapshots = _initialize_modules(
        torch, LocalDepthCodec, OneShotFeedForwardSidecar, TimestepFreeRecurrentSidecar, smoke
    )
    audit.update({"modules": modules, "module_pre": module_pre, "stage": "modules_initialized"})
    _memory_checkpoint(torch, audit, "after_module_construction")
    full_schedule = context["schedules"]["heldout_open"]
    guard = _CacheGuard(model, transformers=transformers, hic0=hic0, calls=full_schedule)
    audit["cache_guard"] = guard
    with guard:
        train_examples = _prepare_sources(
            context["rendered"]["train"],
            phase="pre_learning",
            model=model,
            guard=guard,
            audit=audit,
            torch=torch,
        )
        _run_training_bases(train_examples, model=model, guard=guard, audit=audit, torch=torch)
        mechanism = _pre_update_gate(
            train_examples,
            model=model,
            codecs=codecs,
            sidecars=sidecars,
            shell=shell,
            guard=guard,
            audit=audit,
            torch=torch,
        )
        histories = []
        training_evidence = []
        updates = context["selections"]["train"]["updates"]
        for arm in TRAINING_ARMS:
            history, evidence = _train_arm(
                arm,
                train_examples,
                updates,
                model=model,
                codec=codecs[arm],
                sidecar=sidecars.get(arm),
                shell=shell,
                guard=guard,
                audit=audit,
                torch=torch,
            )
            histories.append({"name": arm, "updates": history})
            training_evidence.append({"name": arm, "value": evidence})
        module_post = [
            {"name": item["name"], "sha256": smoke._module_tensor_sha256(item["module"], torch)} for item in modules
        ]
        delta_groups = [
            {
                "name": arm,
                "groups": _module_delta_groups(arm, codecs[arm], sidecars.get(arm), snapshots[arm], torch=torch),
            }
            for arm in TRAINING_ARMS
        ]
        if any(not group["finite"] or not group["nonzero"] for arm in delta_groups for group in arm["groups"]):
            raise PhaseBContractError("B-IPC1 trained parameter group did not change finitely")
        pre_codecs, pre_sidecars = _restore_pre_modules(
            snapshots,
            torch=torch,
            LocalDepthCodec=LocalDepthCodec,
            OneShotFeedForwardSidecar=OneShotFeedForwardSidecar,
            TimestepFreeRecurrentSidecar=TimestepFreeRecurrentSidecar,
        )
        restored_pre_hashes = {
            arm: {
                "codec": smoke._module_tensor_sha256(pre_codecs[arm], torch),
                "sidecar": None if arm == "STATIC" else smoke._module_tensor_sha256(pre_sidecars[arm], torch),
            }
            for arm in TRAINING_ARMS
        }
        if restored_pre_hashes != mechanism["pre_tensor_hashes"]:
            raise PhaseBContractError("B-IPC1 restored PRE modules differ from initial snapshots")
        safety_by_arm = {}
        for arm in TRAINING_ARMS:
            arm_delta = next(record for record in delta_groups if record["name"] == arm)
            gate_value = float(torch.tanh(codecs[arm].receiver_gate.detach()).cpu())
            safety_by_arm[arm] = (
                all(group["finite"] and group["nonzero"] for group in arm_delta["groups"])
                and math.isfinite(gate_value)
                and gate_value != 0.0
            )
        validation = _evaluate_split(
            "validation",
            context["rendered"]["validation"],
            model=model,
            post_codecs=codecs,
            post_sidecars=sidecars,
            pre_codecs=pre_codecs,
            pre_sidecars=pre_sidecars,
            shell=shell,
            guard=guard,
            audit=audit,
            torch=torch,
            diagnose_recurrent_states=diagnose_recurrent_states,
            safety_by_arm=safety_by_arm,
        )
        validation_pass = any(result["value"]["passed"] for result in validation["common_arm_gates"])
        heldout = None
        if validation_pass:
            heldout_rendered, heldout_proofs = _render_split(
                "heldout",
                context,
                tokenizer=context["tokenizer"],
                parquet=parquet,
                b1=b1,
                smoke=smoke,
            )
            context["render_proofs"]["heldout"] = heldout_proofs
            heldout = _evaluate_split(
                "heldout",
                heldout_rendered,
                model=model,
                post_codecs=codecs,
                post_sidecars=sidecars,
                pre_codecs=pre_codecs,
                pre_sidecars=pre_sidecars,
                shell=shell,
                guard=guard,
                audit=audit,
                torch=torch,
                diagnose_recurrent_states=diagnose_recurrent_states,
                safety_by_arm=safety_by_arm,
            )
        else:
            guard.schedule = context["schedules"]["validation_reject"]
            guard.expected = build_cache_guard_labels(guard.schedule)
        guard.final()
    cache_evidence = guard.evidence()
    if not cache_evidence["complete"]:
        raise CacheContractViolated("B-IPC1 cache evidence is incomplete")
    status, common_nominated_arms = _status_for_evaluations(validation, heldout)
    e33_tensor_post = smoke._module_tensor_sha256(model, torch)
    e33_file_post = file_sha256(smoke._model_file(model_path))
    metadata_post = smoke._metadata_hashes(model_path)
    if (e33_tensor_pre, e33_file_pre, metadata_pre) != (e33_tensor_post, e33_file_post, metadata_post):
        raise PhaseBContractError("B-IPC1 protected e33 changed")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise PhaseBContractError("B-IPC1 e33 retained a gradient")
    immutable_inputs = _immutable_input_records(plan)
    _memory_checkpoint(torch, audit, "after_final_audits")
    _memory_checkpoint(torch, audit, "before_candidate_writes")
    candidates = _save_candidates(output, codecs, sidecars, b1=b1, torch=torch, audit=audit)
    _memory_checkpoint(torch, audit, "before_terminal")
    schedule_name = "heldout_open" if heldout is not None else "validation_reject"
    expected_memory = build_memory_checkpoint_labels(context["schedules"][schedule_name])
    if [item["checkpoint"] for item in audit["memory_ledger"]] != expected_memory:
        raise PhaseBContractError("B-IPC1 CUDA memory checkpoint order differs")
    if any(
        value > CUDA_MEMORY_CAP_BYTES
        for item in audit["memory_ledger"]
        for key, value in item.items()
        if key.endswith("_bytes")
    ):
        raise ResourceContractExceeded("B-IPC1 CUDA memory ledger contains a cap violation")
    return {
        "schema_version": "q35-2b-phase-b-ipc1-matched-learning-success/v1",
        "terminal": "SUCCESS",
        "status": status,
        "disposition": status,
        "claim_class": "nomination_only_inplace_carrier_matched_learning_screen",
        "execution_commit": execution_commit,
        "plan_sha256": plan["_file_sha256"],
        "run_identity": plan["run_identity"],
        "optimizer_steps": 12,
        "backward_calls": 147,
        "model_forwards": len(context["schedules"][schedule_name]),
        "source_forwards": 96 if heldout is not None else 72,
        "receiver_forwards": 700 if heldout is not None else 460,
        "heldout_opened": heldout is not None,
        "training": histories,
        "training_evidence": training_evidence,
        "pre_update_mechanism_gate": mechanism,
        "evaluations": [
            {"name": "validation", "value": validation},
            {"name": "heldout", "value": heldout},
        ],
        "nomination": {
            "status": status,
            "common_nominated_arms": common_nominated_arms,
            "recurrent_nominated": status == SUCCESS_STATUSES[0],
            "admitted": False,
            "complete_live_trajectory_count": 0,
            "minimum_complete_live_trajectories_unchanged": 4,
        },
        "module_hashes": {
            "pre": module_pre,
            "restored_pre": [{"name": arm, **restored_pre_hashes[arm]} for arm in TRAINING_ARMS],
            "post": module_post,
            "delta_groups": delta_groups,
            "receiver_gates": [
                {"name": arm, "value": float(torch.tanh(codecs[arm].receiver_gate.detach()).cpu())}
                for arm in TRAINING_ARMS
            ],
        },
        "candidates": candidates,
        "cache_guard": cache_evidence,
        "cuda_memory": {
            "cap_bytes": CUDA_MEMORY_CAP_BYTES,
            "allocator": audit["allocator"],
            "ledger": audit["memory_ledger"],
            "ordered_label_sha256": canonical_bank_sha256(expected_memory),
        },
        "protection": {
            "e33_tensor_pre": e33_tensor_pre,
            "e33_tensor_post": e33_tensor_post,
            "e33_file_pre": e33_file_pre,
            "e33_file_post": e33_file_post,
            "metadata_pre": metadata_pre,
            "metadata_post": metadata_post,
            "e33_gradients_absent": True,
        },
        "immutable_inputs": immutable_inputs,
        "render_proofs": [{"name": split, "rows": proofs} for split, proofs in context["render_proofs"].items()],
        "bank_bindings": [
            {
                "name": bank["split"],
                "selection_sha256": bank["selection_sha256"],
                "parquet_sha256": bank["parquet_sha256"],
            }
            for bank in plan["banks"]
        ],
        "schedule_binding": {
            "name": schedule_name,
            **plan["schedules"][schedule_name],
        },
        "antecedent_bindings": [
            {"name": item["name"], "binding_sha256": item["binding_sha256"]} for item in plan["antecedents"]
        ],
        "boundaries": {
            "generation": False,
            "cache": False,
            "H176_loaded": False,
            "strand_a_combined": False,
            "live_trajectory_count": 0,
            "admitted": False,
        },
    }


def _metrics_mapping(evaluation: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    records = evaluation.get("metrics")
    expected = (
        "BASE",
        "PRE_STATIC",
        "PRE_FFN",
        "PRE_RECURRENT_T4",
        "POST_STATIC",
        "POST_FFN",
        "POST_RECURRENT_T1",
        "POST_RECURRENT_T2",
        "POST_RECURRENT_T4",
        "POST_RECURRENT_T8",
    )
    if not isinstance(records, list) or [record.get("name") for record in records] != list(expected):
        raise PhaseBContractError("B-IPC1 evaluation metric records differ")
    return {record["name"]: record["rows"] for record in records}


def _recompute_evaluation(evaluation: dict[str, Any], module_safety: dict[str, bool]) -> dict[str, Any]:
    metrics = _metrics_mapping(evaluation)
    residual_records = evaluation.get("post_residual_checks")
    if not isinstance(residual_records, list) or [record.get("name") for record in residual_records] != list(
        TRAINING_ARMS
    ):
        raise PhaseBContractError("B-IPC1 residual-check records differ")
    residuals = {record["name"]: record["rows"] for record in residual_records}
    actions = [row["action"] for row in metrics["BASE"]]
    common = []
    for arm in TRAINING_ARMS:
        suffix = arm if arm != "RECURRENT" else "RECURRENT_T4"
        rows = residuals[arm]
        safety = (
            module_safety[arm]
            and len(rows) == 24
            and all(record.get("finite_nonzero") is True and math.isfinite(float(record.get("l2"))) for record in rows)
        )
        common.append(
            {
                "name": arm,
                "value": evaluate_common_arm(
                    action_order=actions,
                    base=metrics["BASE"],
                    pre=metrics[f"PRE_{suffix}"],
                    post=metrics[f"POST_{suffix}"],
                    safety_and_noncollapse=safety,
                ),
            }
        )
    recurrent = evaluate_recurrent_value(
        actions=actions,
        recurrent={f"T{depth}": metrics[f"POST_RECURRENT_T{depth}"] for depth in EVALUATION_DEPTHS},
        ffn=metrics["POST_FFN"],
        retention_and_stability_passed=_recurrent_retention_stability(metrics["POST_RECURRENT_T8"]),
    )
    return {"common": common, "recurrent": recurrent}


def _validate_render_proofs(receipt: dict[str, Any], *, plan: dict[str, Any], heldout_open: bool) -> None:
    records = receipt.get("render_proofs")
    expected_splits = ["train", "validation"] + (["heldout"] if heldout_open else [])
    if not isinstance(records, list) or [record.get("name") for record in records] != expected_splits:
        raise PhaseBContractError("B-IPC1 SUCCESS render-proof split order differs")
    proof_keys = {
        "task_key",
        "action",
        "source_row_sha256",
        "reasoning_content_sha256",
        "modified_path",
        "plain_ids_sha256",
        "opening_ids_sha256",
        "full_ids_sha256",
        "plain_tokens",
        "opening_tokens",
        "full_tokens",
        "counterfactual_target_sha256",
        "action_trie_sha256",
        "action_trie_branch_count",
    }
    for split_record in records:
        _require_exact_keys(split_record, {"name", "rows"}, label="render-proof split")
        split = split_record["name"]
        selection = load_json_file(Path(_bank_record(plan, split)["selection_path"]))
        expected = selection["selected"]
        proofs = split_record["rows"]
        if not isinstance(proofs, list) or len(proofs) != len(expected):
            raise PhaseBContractError(f"B-IPC1 {split} render-proof count differs")
        for proof, selected, row_hash in zip(proofs, expected, selection["row_canonical_sha256"], strict=True):
            _require_exact_keys(proof, proof_keys, label=f"{split} render proof")
            if (
                proof["task_key"] != selected["task_key"]
                or proof["action"] != selected["expected_action"]
                or proof["source_row_sha256"] != row_hash
                or proof["modified_path"] != "messages.2.tool_calls.0.function.arguments"
                or set(proof["counterfactual_target_sha256"]) != set(ACTIONS)
                or not (0 < proof["plain_tokens"] <= proof["opening_tokens"] < proof["full_tokens"])
                or type(proof["action_trie_branch_count"]) is not int
                or proof["action_trie_branch_count"] < 1
            ):
                raise PhaseBContractError(f"B-IPC1 {split} render proof differs")


def _validate_objective_evidence(value: Any, *, action: str, label: str) -> None:
    record = _require_exact_keys(
        value,
        {"action_weight", "retention_coefficient", "branch_count", "branches", "baseline_margins_detached"},
        label=label,
    )
    expected_weight = 2.0 if action == "delegate_coordinator" else 1.0
    if (
        record["action_weight"] != expected_weight
        or record["retention_coefficient"] != 0.1
        or record["baseline_margins_detached"] is not True
        or type(record["branch_count"]) is not int
        or record["branch_count"] < 1
        or not isinstance(record["branches"], list)
        or len(record["branches"]) != record["branch_count"]
    ):
        raise PhaseBContractError(f"B-IPC1 {label} differs")
    for branch in record["branches"]:
        _require_exact_keys(
            branch,
            {"logit_offset", "correct_token_id", "other_token_ids"},
            label=f"{label} branch",
        )
        if (
            type(branch["logit_offset"]) is not int
            or branch["logit_offset"] < 0
            or type(branch["correct_token_id"]) is not int
            or not isinstance(branch["other_token_ids"], list)
            or not branch["other_token_ids"]
            or any(type(token) is not int for token in branch["other_token_ids"])
        ):
            raise PhaseBContractError(f"B-IPC1 {label} branch differs")


def _validate_training_receipt(receipt: dict[str, Any], *, plan: dict[str, Any]) -> None:
    histories = receipt.get("training")
    evidence = receipt.get("training_evidence")
    if (
        not isinstance(histories, list)
        or [record.get("name") for record in histories] != list(TRAINING_ARMS)
        or not isinstance(evidence, list)
        or [record.get("name") for record in evidence] != list(TRAINING_ARMS)
    ):
        raise PhaseBContractError("B-IPC1 SUCCESS training arm order differs")
    selection = load_json_file(Path(_bank_record(plan, "train")["selection_path"]))
    expected_updates = selection["updates"]
    expected_keys = [record["task_key"] for record in selection["selected"]]
    trained_keys: dict[str, list[str]] = {}
    expected_groups = {
        "STATIC": ["codec"],
        "FFN": ["codec", "ffn_internal"],
        "RECURRENT": ["codec", "transition", "memory", "workspace"],
    }
    for arm_record, evidence_record in zip(histories, evidence, strict=True):
        arm = arm_record["name"]
        _require_exact_keys(arm_record, {"name", "updates"}, label=f"{arm} training")
        updates = arm_record["updates"]
        if not isinstance(updates, list) or len(updates) != 4:
            raise PhaseBContractError(f"B-IPC1 {arm} update count differs")
        flattened: list[str] = []
        for update, expected_update in zip(updates, expected_updates, strict=True):
            _require_exact_keys(
                update,
                {"update_index", "rows", "gradient_l2", "preclip_global_norm", "sidecar_output_scale"},
                label=f"{arm} update",
            )
            if update["update_index"] != expected_update["update_index"] or len(update["rows"]) != 12:
                raise PhaseBContractError(f"B-IPC1 {arm} update index/count differs")
            expected_rows = expected_update["rows"]
            for row, selected in zip(update["rows"], expected_rows, strict=True):
                _require_exact_keys(
                    row,
                    {
                        "task_key",
                        "action",
                        "aligned_suffix_ce",
                        "total_objective",
                        "objective_evidence",
                        "reporting_nll",
                        "reporting_margin",
                    },
                    label=f"{arm} training row",
                )
                if row["task_key"] != selected["task_key"] or row["action"] != selected["expected_action"]:
                    raise PhaseBContractError(f"B-IPC1 {arm} update exposure differs")
                for key in ("aligned_suffix_ce", "total_objective", "reporting_nll", "reporting_margin"):
                    _require_finite_number(row[key], label=f"{arm} {key}")
                _validate_objective_evidence(row["objective_evidence"], action=row["action"], label=f"{arm} objective")
                flattened.append(row["task_key"])
            gradients = update["gradient_l2"]
            if not isinstance(gradients, list) or not gradients:
                raise PhaseBContractError(f"B-IPC1 {arm} named gradients are absent")
            seen_names: set[str] = set()
            for gradient in gradients:
                _require_exact_keys(gradient, {"name", "l2"}, label=f"{arm} named gradient")
                if gradient["name"] in seen_names:
                    raise PhaseBContractError(f"B-IPC1 {arm} repeats a named gradient")
                seen_names.add(gradient["name"])
                if gradient["l2"] is not None:
                    _require_finite_number(gradient["l2"], label=f"{arm} gradient l2")
            _require_finite_number(update["preclip_global_norm"], label=f"{arm} preclip norm")
            scales = update["sidecar_output_scale"]
            if (arm == "STATIC" and scales != []) or (
                arm != "STATIC" and (not isinstance(scales, list) or len(scales) != 1)
            ):
                raise PhaseBContractError(f"B-IPC1 {arm} output-scale evidence differs")
            for scale in scales:
                _require_finite_number(scale, label=f"{arm} output scale")
        if flattened != expected_keys or len(set(flattened)) != 48:
            raise PhaseBContractError(f"B-IPC1 {arm} exact exposure order differs")
        trained_keys[arm] = flattened
        _require_exact_keys(evidence_record, {"name", "value"}, label=f"{arm} training evidence")
        value = _require_exact_keys(
            evidence_record["value"],
            {"finite_nonzero_gradient_updates", "optimizer_destroyed"},
            label=f"{arm} training evidence value",
        )
        group_updates = value["finite_nonzero_gradient_updates"]
        if set(group_updates) != set(expected_groups[arm]) or value["optimizer_destroyed"] is not True:
            raise PhaseBContractError(f"B-IPC1 {arm} gradient-group/destruction evidence differs")
        for group, passing_updates in group_updates.items():
            allowed = [1, 2, 3, 4] if group == "codec" else [2, 3, 4]
            if (
                not isinstance(passing_updates, list)
                or not passing_updates
                or any(index not in allowed for index in passing_updates)
            ):
                raise PhaseBContractError(f"B-IPC1 {arm}/{group} gradient update evidence differs")
    if any(keys != expected_keys for keys in trained_keys.values()):
        raise PhaseBContractError("B-IPC1 matched arm exposure differs")


def _validate_mechanism_receipt(receipt: dict[str, Any], *, plan: dict[str, Any]) -> None:
    mechanism = _require_exact_keys(
        receipt.get("pre_update_mechanism_gate"),
        {"probes", "pre_tensor_hashes", "post_tensor_hashes"},
        label="pre-update mechanism",
    )
    probes = mechanism["probes"]
    if not isinstance(probes, list) or [probe.get("selection_index") for probe in probes] != [0, 1, 2, 5]:
        raise PhaseBContractError("B-IPC1 SUCCESS mechanism probes differ")
    backward_for = {"STATIC": 0, "FFN": 1, "RECURRENT": 2}
    selected = load_json_file(Path(_bank_record(plan, "train")["selection_path"]))["selected"]
    for probe in probes:
        _require_exact_keys(
            probe,
            {"selection_index", "task_key", "action", "base_equals_direct_zero", "arms"},
            label="mechanism probe",
        )
        expected_probe = selected[probe["selection_index"]]
        if (
            probe["task_key"] != expected_probe["task_key"]
            or probe["action"] != expected_probe["expected_action"]
            or probe["base_equals_direct_zero"] is not True
        ):
            raise PhaseBContractError("B-IPC1 mechanism probe identity differs")
        arms = probe["arms"]
        if not isinstance(arms, list) or [record.get("name") for record in arms] != list(TRAINING_ARMS):
            raise PhaseBContractError("B-IPC1 mechanism arm order differs")
        for arm_record in arms:
            _require_exact_keys(
                arm_record,
                {"name", "inplace_zero_identity", "inplace_eps", "backward"},
                label="mechanism arm",
            )
            arm = arm_record["name"]
            if arm_record["inplace_zero_identity"] is not True:
                raise PhaseBContractError("B-IPC1 mechanism arm zero identity differs")
            eps = arm_record["inplace_eps"]
            _validate_reporting_metric(eps, label=f"{arm} probe")
            expected_backward = backward_for[arm] == probe["selection_index"]
            gradient = arm_record["backward"]
            if not expected_backward:
                if gradient is not None:
                    raise PhaseBContractError("B-IPC1 unexpected preprobe backward evidence")
                continue
            gradient = _require_exact_keys(
                gradient,
                {
                    "objective",
                    "objective_evidence",
                    "residual",
                    "codec",
                    "sidecar",
                    "all_named_present_gradients_finite",
                    "e33_gradients_absent",
                },
                label="preprobe backward",
            )
            _require_finite_number(gradient["objective"], label="preprobe objective")
            _validate_objective_evidence(
                gradient["objective_evidence"], action=probe["action"], label="preprobe objective"
            )
            for name in ("residual", "codec"):
                group = _require_exact_keys(
                    gradient[name],
                    {"finite", "nonzero"} if name == "residual" else {"tensor_count", "finite", "nonzero"},
                    label=f"preprobe {name}",
                )
                if group["finite"] is not True or group["nonzero"] is not True:
                    raise PhaseBContractError(f"B-IPC1 preprobe {name} gradient differs")
            sidecar = gradient["sidecar"]
            if arm == "STATIC":
                if sidecar is not None:
                    raise PhaseBContractError("B-IPC1 STATIC preprobe has sidecar evidence")
            else:
                sidecar = _require_exact_keys(sidecar, {"tensor_count", "finite", "nonzero"}, label="preprobe sidecar")
                if sidecar["finite"] is not True or sidecar["nonzero"] is not True:
                    raise PhaseBContractError("B-IPC1 preprobe sidecar gradient differs")
            if (
                gradient["all_named_present_gradients_finite"] is not True
                or gradient["e33_gradients_absent"] is not True
            ):
                raise PhaseBContractError("B-IPC1 preprobe gradient safety differs")
    if mechanism["pre_tensor_hashes"] != mechanism["post_tensor_hashes"] or set(mechanism["pre_tensor_hashes"]) != set(
        TRAINING_ARMS
    ):
        raise PhaseBContractError("B-IPC1 preprobe candidate state changed")
    for arm, hashes in mechanism["pre_tensor_hashes"].items():
        _require_exact_keys(hashes, {"codec", "sidecar"}, label=f"{arm} preprobe hashes")
        if not isinstance(hashes["codec"], str) or (arm == "STATIC") is not (hashes["sidecar"] is None):
            raise PhaseBContractError(f"B-IPC1 {arm} preprobe hash shape differs")


def _validate_candidate_records(
    candidates: Any,
    *,
    output_dir: Path,
    module_post: list[dict[str, Any]],
    torch: Any,
) -> None:
    if not isinstance(candidates, list) or [record.get("arm") for record in candidates] != list(TRAINING_ARMS):
        raise PhaseBContractError("B-IPC1 SUCCESS candidate records differ")
    expected_files = [f"{arm}.final.pt" for arm in TRAINING_ARMS]
    if sorted(path.name for path in output_dir.glob("*.pt")) != sorted(expected_files):
        raise PhaseBContractError("B-IPC1 SUCCESS candidate file set differs")
    post = {record["name"]: record["sha256"] for record in module_post}
    for record, arm in zip(candidates, TRAINING_ARMS, strict=True):
        _require_exact_keys(
            record,
            {"name", "arm", "sha256", "valid_only_with_terminal", "module_post_state"},
            label=f"{arm} candidate record",
        )
        expected_name = f"{arm}.final.pt"
        if record["name"] != expected_name or record["valid_only_with_terminal"] != "SUCCESS.json":
            raise PhaseBContractError(f"B-IPC1 {arm} candidate filename/terminal binding differs")
        path = output_dir / expected_name
        if (
            path.parent != output_dir
            or path.is_symlink()
            or not path.is_file()
            or file_sha256(path) != record["sha256"]
        ):
            raise PhaseBContractError(f"B-IPC1 {arm} candidate path/hash differs")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        _require_exact_keys(
            payload,
            {"schema_version", "arm", "codec", "sidecar", "module_post_state"},
            label=f"{arm} candidate payload",
        )
        if payload["schema_version"] != "q35-2b-phase-b-ipc1-candidate/v1" or payload["arm"] != arm:
            raise PhaseBContractError(f"B-IPC1 {arm} candidate schema differs")
        if (arm == "STATIC") is not (payload["sidecar"] is None):
            raise PhaseBContractError(f"B-IPC1 {arm} candidate sidecar shape differs")
        observed = _candidate_state_records(payload["codec"], payload["sidecar"], torch=torch)
        if observed != record["module_post_state"] or observed != payload["module_post_state"]:
            raise PhaseBContractError(f"B-IPC1 {arm} candidate state schema/hash differs")
        expected_states = [{"name": "codec", "sha256": post[f"{arm}.codec"]}]
        if arm != "STATIC":
            expected_states.append({"name": "sidecar", "sha256": post[f"{arm}.sidecar"]})
        if [{"name": item["name"], "sha256": item["tensor_sha256"]} for item in observed] != expected_states:
            raise PhaseBContractError(f"B-IPC1 {arm} candidate differs from exact post-module state")


def _validate_reporting_metric(value: Any, *, label: str, with_identity_hashes: bool = True) -> None:
    keys = {
        "nll",
        "native_output_loss_descriptive",
        "native_minus_float64_nll_descriptive",
        "margin",
        "finite",
        "branch_metrics",
        "hashes",
    }
    metric = _require_exact_keys(value, keys, label=label)
    for key in ("nll", "native_output_loss_descriptive", "native_minus_float64_nll_descriptive", "margin"):
        _require_finite_number(metric[key], label=f"{label} {key}")
    if metric["finite"] is not True or not isinstance(metric["branch_metrics"], list) or not metric["branch_metrics"]:
        raise PhaseBContractError(f"B-IPC1 {label} finite/branch evidence differs")
    branch_keys = {
        "target_offset",
        "logit_offset",
        "correct_token_id",
        "other_token_ids",
        "live_actions",
        "correct_logit",
        "max_other_logit",
        "margin",
    }
    for branch in metric["branch_metrics"]:
        _require_exact_keys(branch, branch_keys, label=f"{label} branch metric")
        for key in ("correct_logit", "max_other_logit", "margin"):
            _require_finite_number(branch[key], label=f"{label} branch {key}")
    hashes = _require_exact_keys(
        metric["hashes"],
        {"inputs_embeds", "attention_mask", "position_ids", "labels", "final_hidden", "first_suffix_logits"},
        label=f"{label} hashes",
    )
    if with_identity_hashes and any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
        raise PhaseBContractError(f"B-IPC1 {label} tensor hash differs")


def _validate_summary(value: Any, *, label: str, coordinator: bool = False) -> None:
    if coordinator:
        record = _require_exact_keys(value, {"values", "mean", "median", "strict_wins"}, label=label)
        expected_count = 8
    else:
        record = _require_exact_keys(
            value,
            {"values", "mean", "median", "minimum", "maximum", "strict_wins", "per_action_means"},
            label=label,
        )
        expected_count = 24
        per_action = record["per_action_means"]
        if not isinstance(per_action, list) or [item.get("action") for item in per_action] != list(ACTIONS):
            raise PhaseBContractError(f"B-IPC1 {label} per-action order differs")
        for item in per_action:
            _require_exact_keys(item, {"action", "mean"}, label=f"{label} action mean")
            _require_finite_number(item["mean"], label=f"{label} action mean")
        for key in ("minimum", "maximum"):
            _require_finite_number(record[key], label=f"{label} {key}")
    if not isinstance(record["values"], list) or len(record["values"]) != expected_count:
        raise PhaseBContractError(f"B-IPC1 {label} value count differs")
    for value_item in record["values"]:
        _require_finite_number(value_item, label=f"{label} value")
    for key in ("mean", "median"):
        _require_finite_number(record[key], label=f"{label} {key}")
    if type(record["strict_wins"]) is not int:
        raise PhaseBContractError(f"B-IPC1 {label} strict-win count differs")


def _validate_gate_result(value: Any, *, recurrent: bool, label: str) -> None:
    result = _require_exact_keys(value, {"passed", "gates", "summaries"}, label=label)
    expected_gates = (
        [
            "positive_recurrence_over_ffn",
            "nll_route",
            "action_margin_route",
            "depth_T4_over_T1",
            "retention_and_stability",
            "T8_nonregression",
        ]
        if recurrent
        else [
            "exact_complete_finite_safe_noncollapse",
            "nll_learning",
            "base_margin_retention",
            "coordinator_margin_correction",
        ]
    )
    if not isinstance(result["gates"], list) or [item.get("name") for item in result["gates"]] != expected_gates:
        raise PhaseBContractError(f"B-IPC1 {label} gate order differs")
    for gate in result["gates"]:
        _require_exact_keys(gate, {"name", "passed"}, label=f"{label} gate")
        if type(gate["passed"]) is not bool:
            raise PhaseBContractError(f"B-IPC1 {label} gate value differs")
    expected_summaries = (
        ["A_N", "A_M", "R4_minus_R1_nll", "R4_minus_R1_margin", "R8_minus_R4_nll", "R8_minus_R4_margin"]
        if recurrent
        else ["delta_n_pre_minus_post", "delta_m_post_minus_base", "coordinator_delta_m_post_minus_pre"]
    )
    if (
        not isinstance(result["summaries"], list)
        or [item.get("name") for item in result["summaries"]] != expected_summaries
    ):
        raise PhaseBContractError(f"B-IPC1 {label} summary order differs")
    for summary in result["summaries"]:
        _require_exact_keys(summary, {"name", "value"}, label=f"{label} summary")
        _validate_summary(
            summary["value"],
            label=f"{label} {summary['name']}",
            coordinator=summary["name"] == "coordinator_delta_m_post_minus_pre",
        )
    if type(result["passed"]) is not bool:
        raise PhaseBContractError(f"B-IPC1 {label} pass value differs")


def _validate_evaluation_schema(evaluation: Any, *, split: str, plan: dict[str, Any]) -> None:
    value = _require_exact_keys(
        evaluation,
        {"split", "rows", "metrics", "common_arm_gates", "recurrent_gates", "post_residual_checks"},
        label=f"{split} evaluation",
    )
    if value["split"] != split or value["rows"] != 24:
        raise PhaseBContractError(f"B-IPC1 {split} evaluation identity differs")
    selection = load_json_file(Path(_bank_record(plan, split)["selection_path"]))
    selected = selection["selected"]
    expected_keys = [record["task_key"] for record in selected]
    expected_actions = [record["expected_action"] for record in selected]
    metrics = _metrics_mapping(value)
    for name, rows in metrics.items():
        if not isinstance(rows, list) or len(rows) != 24:
            raise PhaseBContractError(f"B-IPC1 {split}/{name} metric count differs")
        for row, key, action in zip(rows, expected_keys, expected_actions, strict=True):
            extra = {"retention", "stability_T8"} if name == "POST_RECURRENT_T8" else set()
            _require_exact_keys(
                row,
                {
                    "task_key",
                    "action",
                    "nll",
                    "native_output_loss_descriptive",
                    "native_minus_float64_nll_descriptive",
                    "margin",
                    "finite",
                    "branch_metrics",
                    "hashes",
                    *extra,
                },
                label=f"{split}/{name} row",
            )
            if row["task_key"] != key or row["action"] != action:
                raise PhaseBContractError(f"B-IPC1 {split}/{name} row order differs")
            _validate_reporting_metric(
                {key: row[key] for key in row if key not in extra | {"task_key", "action"}}, label=f"{split}/{name}"
            )
            if extra:
                retention = row["retention"]
                if set(retention) != {f"T{depth}" for depth in EVALUATION_DEPTHS}:
                    raise PhaseBContractError(f"B-IPC1 {split} retention depths differ")
                for item in retention.values():
                    _require_exact_keys(item, {"cosine", "norm_ratio", "relative_l2"}, label="retention")
                    for metric_value in item.values():
                        _require_finite_number(metric_value, label="retention value")
                stability = _require_exact_keys(
                    row["stability_T8"],
                    {
                        "memory_change_rms",
                        "memory_contraction_steps_2_8",
                        "median_memory_contraction_steps_2_8",
                        "max_memory_contraction_steps_2_8",
                        "memory_oscillation_rate",
                        "finite",
                    },
                    label="stability",
                )
                if (
                    len(stability["memory_change_rms"]) != 8
                    or len(stability["memory_contraction_steps_2_8"]) != 7
                    or stability["finite"] is not True
                ):
                    raise PhaseBContractError("B-IPC1 stability evidence differs")
                for metric_value in (
                    *stability["memory_change_rms"],
                    *stability["memory_contraction_steps_2_8"],
                    stability["median_memory_contraction_steps_2_8"],
                    stability["max_memory_contraction_steps_2_8"],
                    stability["memory_oscillation_rate"],
                ):
                    _require_finite_number(metric_value, label="stability value")
    common = value["common_arm_gates"]
    if not isinstance(common, list) or [record.get("name") for record in common] != list(TRAINING_ARMS):
        raise PhaseBContractError(f"B-IPC1 {split} common gate order differs")
    for record in common:
        _require_exact_keys(record, {"name", "value"}, label=f"{split} common gate")
        _validate_gate_result(record["value"], recurrent=False, label=f"{split}/{record['name']}")
    _validate_gate_result(value["recurrent_gates"], recurrent=True, label=f"{split}/recurrent")
    residuals = value["post_residual_checks"]
    if not isinstance(residuals, list) or [record.get("name") for record in residuals] != list(TRAINING_ARMS):
        raise PhaseBContractError(f"B-IPC1 {split} residual arm order differs")
    for record in residuals:
        _require_exact_keys(record, {"name", "rows"}, label=f"{split} residual arm")
        if len(record["rows"]) != 24:
            raise PhaseBContractError(f"B-IPC1 {split} residual count differs")
        for row, key in zip(record["rows"], expected_keys, strict=True):
            _require_exact_keys(row, {"task_key", "finite_nonzero", "l2"}, label=f"{split} residual row")
            if row["task_key"] != key or row["finite_nonzero"] is not True:
                raise PhaseBContractError(f"B-IPC1 {split} residual evidence differs")
            _require_finite_number(row["l2"], label=f"{split} residual l2")


def validate_success_receipt(
    receipt: dict[str, Any], *, plan: dict[str, Any], execution_commit: str, output_dir: Path, torch: Any
) -> None:
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "terminal",
            "status",
            "disposition",
            "claim_class",
            "execution_commit",
            "plan_sha256",
            "run_identity",
            "optimizer_steps",
            "backward_calls",
            "model_forwards",
            "source_forwards",
            "receiver_forwards",
            "heldout_opened",
            "training",
            "training_evidence",
            "pre_update_mechanism_gate",
            "evaluations",
            "nomination",
            "module_hashes",
            "candidates",
            "cache_guard",
            "cuda_memory",
            "protection",
            "immutable_inputs",
            "render_proofs",
            "bank_bindings",
            "schedule_binding",
            "antecedent_bindings",
            "boundaries",
            "elapsed_seconds",
            "receipt_sha256",
        },
        label="SUCCESS",
    )
    unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != hashlib.sha256(canonical_terminal_bytes(unhashed)).hexdigest():
        raise PhaseBContractError("B-IPC1 SUCCESS internal hash differs")
    if (
        receipt.get("schema_version") != "q35-2b-phase-b-ipc1-matched-learning-success/v1"
        or receipt.get("terminal") != "SUCCESS"
        or receipt.get("status") not in SUCCESS_STATUSES
        or receipt.get("disposition") != receipt.get("status")
        or receipt.get("execution_commit") != execution_commit
        or receipt.get("plan_sha256") != plan["_file_sha256"]
        or receipt.get("run_identity") != plan["run_identity"]
        or receipt.get("claim_class") != "nomination_only_inplace_carrier_matched_learning_screen"
        or receipt.get("optimizer_steps") != 12
        or receipt.get("backward_calls") != 147
    ):
        raise PhaseBContractError("B-IPC1 SUCCESS top-level contract differs")
    _require_finite_number(receipt["elapsed_seconds"], label="SUCCESS elapsed_seconds")
    heldout_open = receipt.get("heldout_opened") is True
    expected_counts = (796, 96, 700) if heldout_open else (532, 72, 460)
    if (
        receipt.get("model_forwards"),
        receipt.get("source_forwards"),
        receipt.get("receiver_forwards"),
    ) != expected_counts:
        raise PhaseBContractError("B-IPC1 SUCCESS model-call counts differ")
    _validate_training_receipt(receipt, plan=plan)
    training = receipt["training"]
    expected_train_keys = [
        record["task_key"] for record in load_json_file(Path(_bank_record(plan, "train")["selection_path"]))["selected"]
    ]
    for arm in training:
        observed = [row["task_key"] for update in arm["updates"] for row in update["rows"]]
        if observed != expected_train_keys:
            raise PhaseBContractError(f"B-IPC1 SUCCESS {arm['name']} training order differs")
    _validate_mechanism_receipt(receipt, plan=plan)
    module_hashes = receipt.get("module_hashes")
    _require_exact_keys(
        module_hashes,
        {"pre", "restored_pre", "post", "delta_groups", "receiver_gates"},
        label="module hashes",
    )
    expected_module_names = ["STATIC.codec", "FFN.codec", "FFN.sidecar", "RECURRENT.codec", "RECURRENT.sidecar"]
    for field in ("pre", "post"):
        records = module_hashes[field]
        if not isinstance(records, list) or [record.get("name") for record in records] != expected_module_names:
            raise PhaseBContractError(f"B-IPC1 module {field} order differs")
        for record in records:
            _require_exact_keys(record, {"name", "sha256"}, label=f"module {field}")
    restored = module_hashes["restored_pre"]
    if not isinstance(restored, list) or [record.get("name") for record in restored] != list(TRAINING_ARMS):
        raise PhaseBContractError("B-IPC1 restored PRE order differs")
    for record in restored:
        _require_exact_keys(record, {"name", "codec", "sidecar"}, label="restored PRE")
    deltas = module_hashes.get("delta_groups")
    gates = module_hashes.get("receiver_gates")
    if (
        not isinstance(deltas, list)
        or [record.get("name") for record in deltas] != list(TRAINING_ARMS)
        or not isinstance(gates, list)
        or [record.get("name") for record in gates] != list(TRAINING_ARMS)
    ):
        raise PhaseBContractError("B-IPC1 SUCCESS module safety records differ")
    for delta in deltas:
        _require_exact_keys(delta, {"name", "groups"}, label="module delta")
        if not isinstance(delta["groups"], list) or not delta["groups"]:
            raise PhaseBContractError("B-IPC1 module delta groups are absent")
        expected_delta_names = {
            "STATIC": ["codec_encoder", "codec_receiver"],
            "FFN": ["codec_encoder", "codec_receiver", "ffn_internal", "ffn_output_scale"],
            "RECURRENT": [
                "codec_encoder",
                "codec_receiver",
                "recurrent_transition",
                "recurrent_memory",
                "recurrent_workspace",
                "recurrent_output_scale",
            ],
        }[delta["name"]]
        if [group.get("name") for group in delta["groups"]] != expected_delta_names:
            raise PhaseBContractError(f"B-IPC1 {delta['name']} module delta group order differs")
        for group in delta["groups"]:
            _require_exact_keys(group, {"name", "finite", "nonzero", "delta_l2"}, label="module delta group")
            _require_finite_number(group["delta_l2"], label="module delta l2")
    for gate in gates:
        _require_exact_keys(gate, {"name", "value"}, label="receiver gate")
        _require_finite_number(gate["value"], label="receiver gate")
    module_safety = {
        arm: all(group.get("finite") is True and group.get("nonzero") is True for group in delta["groups"])
        and math.isfinite(float(gate["value"]))
        and float(gate["value"]) != 0.0
        for arm, delta, gate in zip(TRAINING_ARMS, deltas, gates, strict=True)
    }
    evaluations = receipt.get("evaluations")
    if not isinstance(evaluations, list) or [record.get("name") for record in evaluations] != ["validation", "heldout"]:
        raise PhaseBContractError("B-IPC1 SUCCESS evaluation records differ")
    validation = evaluations[0]["value"]
    heldout = evaluations[1]["value"]
    if not isinstance(validation, dict) or (heldout is None) is heldout_open:
        raise PhaseBContractError("B-IPC1 SUCCESS heldout firewall evidence differs")
    for record in evaluations:
        _require_exact_keys(record, {"name", "value"}, label="evaluation record")
    _validate_evaluation_schema(validation, split="validation", plan=plan)
    if heldout is not None:
        _validate_evaluation_schema(heldout, split="heldout", plan=plan)
    for evaluation in (validation,) if heldout is None else (validation, heldout):
        recomputed = _recompute_evaluation(evaluation, module_safety)
        if recomputed["common"] != evaluation.get("common_arm_gates") or recomputed["recurrent"] != evaluation.get(
            "recurrent_gates"
        ):
            raise PhaseBContractError("B-IPC1 SUCCESS evaluation aggregates do not recompute")
    status, nominated_arms = _status_for_evaluations(validation, heldout)
    nomination = receipt.get("nomination")
    _require_exact_keys(
        nomination,
        {
            "status",
            "common_nominated_arms",
            "recurrent_nominated",
            "admitted",
            "complete_live_trajectory_count",
            "minimum_complete_live_trajectories_unchanged",
        },
        label="nomination",
    )
    if (
        status != receipt["status"]
        or not isinstance(nomination, dict)
        or nomination.get("status") != status
        or nomination.get("common_nominated_arms") != nominated_arms
        or nomination.get("recurrent_nominated") is not (status == SUCCESS_STATUSES[0])
        or nomination.get("admitted") is not False
        or nomination.get("complete_live_trajectory_count") != 0
        or nomination.get("minimum_complete_live_trajectories_unchanged") != 4
    ):
        raise PhaseBContractError("B-IPC1 SUCCESS nomination/status differs")
    schedule_name = "heldout_open" if heldout_open else "validation_reject"
    if receipt.get("schedule_binding") != {"name": schedule_name, **plan["schedules"][schedule_name]}:
        raise PhaseBContractError("B-IPC1 SUCCESS schedule binding differs")
    cache = receipt.get("cache_guard")
    _require_exact_keys(
        cache,
        {
            "complete",
            "labels",
            "label_count",
            "canonical_label_sha256",
            "expected_label_sha256",
            "exact_prefix",
            "exit_recorded",
            "dynamic_cache_trip_count",
            "closure_check_count",
            "closure_checked_at_every_label",
            "restored_in_finally",
            "model_calls",
            "recursively_closed_config_count",
            "classes",
        },
        label="cache guard",
    )
    expected_cache = plan["schedules"][schedule_name]["cache_label_sha256"]
    if (
        not isinstance(cache, dict)
        or cache.get("complete") is not True
        or cache.get("model_calls") != expected_counts[0]
        or cache.get("expected_label_sha256") != expected_cache
        or cache.get("canonical_label_sha256") != expected_cache
        or cache.get("restored_in_finally") is not True
        or cache.get("dynamic_cache_trip_count") != 1
        or cache.get("closure_check_count") != cache.get("label_count")
        or cache.get("closure_checked_at_every_label") is not True
        or cache.get("recursively_closed_config_count", 0) < 1
        or len(cache.get("classes", [])) != 8
    ):
        raise PhaseBContractError("B-IPC1 SUCCESS cache evidence differs")
    if not isinstance(cache["labels"], list) or len(cache["labels"]) != cache["label_count"]:
        raise PhaseBContractError("B-IPC1 SUCCESS cache labels differ")
    for cls in cache["classes"]:
        _require_exact_keys(cls, {"fqcn", "module_path", "module_sha256", "distribution"}, label="cache class")
    memory = receipt.get("cuda_memory")
    _require_exact_keys(memory, {"cap_bytes", "allocator", "ledger", "ordered_label_sha256"}, label="CUDA memory")
    _require_exact_keys(
        memory["allocator"],
        {"device_total_bytes", "cap_bytes", "requested_fraction", "observed_fraction"},
        label="CUDA allocator",
    )
    ledger = memory.get("ledger", []) if isinstance(memory, dict) else []
    expected_memory = build_memory_checkpoint_labels(
        build_model_call_schedule(
            expected_train_keys,
            [
                record["task_key"]
                for record in load_json_file(Path(_bank_record(plan, "validation")["selection_path"]))["selected"]
            ],
            [
                record["task_key"]
                for record in load_json_file(Path(_bank_record(plan, "heldout")["selection_path"]))["selected"]
            ],
            open_heldout=heldout_open,
        )
    )
    if (
        memory.get("cap_bytes") != CUDA_MEMORY_CAP_BYTES
        or memory.get("allocator", {}).get("cap_bytes") != CUDA_MEMORY_CAP_BYTES
        or not math.isclose(
            float(memory.get("allocator", {}).get("requested_fraction", -1.0)),
            float(memory.get("allocator", {}).get("observed_fraction", -2.0)),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or [record.get("checkpoint") for record in ledger] != expected_memory
        or memory.get("ordered_label_sha256") != canonical_bank_sha256(expected_memory)
        or any(
            value > CUDA_MEMORY_CAP_BYTES
            for record in ledger
            for key, value in record.items()
            if key.endswith("_bytes")
        )
    ):
        raise PhaseBContractError("B-IPC1 SUCCESS memory evidence differs")
    for record in ledger:
        _require_exact_keys(
            record,
            {
                "checkpoint",
                "current_allocated_bytes",
                "current_reserved_bytes",
                "maximum_allocated_bytes",
                "maximum_reserved_bytes",
            },
            label="memory checkpoint",
        )
    protection = receipt.get("protection")
    _require_exact_keys(
        protection,
        {
            "e33_tensor_pre",
            "e33_tensor_post",
            "e33_file_pre",
            "e33_file_post",
            "metadata_pre",
            "metadata_post",
            "e33_gradients_absent",
        },
        label="protection",
    )
    if (
        not isinstance(protection, dict)
        or protection.get("e33_tensor_pre") != protection.get("e33_tensor_post")
        or protection.get("e33_file_pre") != protection.get("e33_file_post")
        or protection.get("metadata_pre") != protection.get("metadata_post")
        or protection.get("e33_file_pre") != plan["protected_model"]["weight_sha256"]
        or protection.get("metadata_pre") != plan["model_metadata_sha256"]
        or protection.get("e33_gradients_absent") is not True
    ):
        raise PhaseBContractError("B-IPC1 SUCCESS e33 protection differs")
    immutable_inputs = receipt.get("immutable_inputs")
    expected_immutable = _immutable_input_records(plan)
    if immutable_inputs != expected_immutable:
        raise PhaseBContractError("B-IPC1 SUCCESS immutable postflight hashes differ")
    for record in immutable_inputs:
        _require_exact_keys(record, {"name", "path", "expected_sha256", "observed_sha256"}, label="immutable input")
    _validate_candidate_records(
        receipt.get("candidates"), output_dir=output_dir, module_post=module_hashes["post"], torch=torch
    )
    _validate_render_proofs(receipt, plan=plan, heldout_open=heldout_open)
    expected_banks = [
        {"name": bank["split"], "selection_sha256": bank["selection_sha256"], "parquet_sha256": bank["parquet_sha256"]}
        for bank in plan["banks"]
    ]
    expected_antecedents = [
        {"name": item["name"], "binding_sha256": item["binding_sha256"]} for item in plan["antecedents"]
    ]
    if receipt.get("bank_bindings") != expected_banks or receipt.get("antecedent_bindings") != expected_antecedents:
        raise PhaseBContractError("B-IPC1 SUCCESS bank or antecedent binding differs")
    for record in receipt["bank_bindings"]:
        _require_exact_keys(record, {"name", "selection_sha256", "parquet_sha256"}, label="bank binding")
    for record in receipt["antecedent_bindings"]:
        _require_exact_keys(record, {"name", "binding_sha256"}, label="antecedent binding")
    _require_exact_keys(
        receipt["schedule_binding"],
        {"name", *plan["schedules"][schedule_name]},
        label="schedule binding",
    )
    _require_exact_keys(
        receipt["boundaries"],
        {"generation", "cache", "H176_loaded", "strand_a_combined", "live_trajectory_count", "admitted"},
        label="claim boundaries",
    )
    if receipt.get("boundaries") != {
        "generation": False,
        "cache": False,
        "H176_loaded": False,
        "strand_a_combined": False,
        "live_trajectory_count": 0,
        "admitted": False,
    }:
        raise PhaseBContractError("B-IPC1 SUCCESS claim boundary differs")


def validate_failure_receipt(receipt: dict[str, Any], *, plan: dict[str, Any], execution_commit: str) -> None:
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "terminal",
            "status",
            "disposition",
            "failure_class",
            "error_type",
            "error",
            "execution_commit",
            "plan_sha256",
            "run_identity",
            "model_loaded",
            "candidate_files_valid",
            "candidate_files_present",
            "execution_breadcrumbs",
            "post_failure_audit",
            "elapsed_seconds",
            "receipt_sha256",
        },
        label="FAILURE",
    )
    unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != hashlib.sha256(canonical_terminal_bytes(unhashed)).hexdigest():
        raise PhaseBContractError("B-IPC1 FAILURE internal hash differs")
    pair = (receipt.get("status"), receipt.get("failure_class"))
    if (
        receipt.get("schema_version") != "q35-2b-phase-b-ipc1-matched-learning-failure/v1"
        or receipt.get("terminal") != "FAILURE"
        or pair not in FAILURE_STATUS_CLASSES
        or receipt.get("disposition") != receipt.get("status")
        or receipt.get("execution_commit") != execution_commit
        or receipt.get("plan_sha256") != plan.get("_file_sha256")
        or receipt.get("run_identity") != plan.get("run_identity")
        or receipt.get("candidate_files_valid") is not False
    ):
        raise PhaseBContractError("B-IPC1 FAILURE top-level contract differs")
    _require_finite_number(receipt["elapsed_seconds"], label="FAILURE elapsed_seconds")
    _require_exact_keys(
        receipt["execution_breadcrumbs"],
        {"stage", "task_key", "arm", "call_index"},
        label="FAILURE breadcrumbs",
    )
    expected_types = {
        FAILURE_STATUS_CLASSES[0]: {"MechanismRejected"},
        FAILURE_STATUS_CLASSES[1]: {"CacheContractViolated"},
        FAILURE_STATUS_CLASSES[3]: {
            "ResourceContractExceeded",
            "ProvenanceContractViolated",
            "TimeoutError",
            "MemoryError",
            "OutOfMemoryError",
        },
    }
    classified_oom = (
        pair == FAILURE_STATUS_CLASSES[3]
        and receipt["error_type"] == "RuntimeError"
        and "out of memory" in receipt["error"].lower()
    )
    if pair in expected_types and receipt["error_type"] not in expected_types[pair] and not classified_oom:
        raise PhaseBContractError("B-IPC1 FAILURE error type/status classification differs")
    audit = receipt.get("post_failure_audit")
    _require_exact_keys(
        audit,
        {
            "audit_complete",
            "audit_errors",
            "immutable_inputs_preserved",
            "immutable_inputs",
            "e33_tensor_preserved",
            "e33_disk_preserved",
            "metadata_preserved",
            "e33_gradients_absent",
            "candidate_files_present",
            "candidate_files_valid",
            "candidate_module_state",
            "candidate_initial_state",
            "cache_guard",
            "cuda_memory",
        },
        label="FAILURE postflight audit",
    )
    if not isinstance(audit, dict) or audit.get("audit_complete") is not True:
        raise PhaseBContractError("B-IPC1 FAILURE post-failure audit is incomplete")
    if audit["audit_errors"] != [] or audit["candidate_files_valid"] is not False:
        raise PhaseBContractError("B-IPC1 FAILURE audit error/candidate validity differs")
    expected_immutable_audit = _immutable_input_records(plan, require_match=False)
    if audit["immutable_inputs"] != expected_immutable_audit:
        raise PhaseBContractError("B-IPC1 FAILURE immutable input audit differs")
    provenance_failure = pair == FAILURE_STATUS_CLASSES[3] and receipt["error_type"] == "ProvenanceContractViolated"
    if audit["immutable_inputs_preserved"] is not True and not provenance_failure:
        raise PhaseBContractError("B-IPC1 FAILURE immutable inputs were not preserved")
    if audit["candidate_files_present"] != receipt["candidate_files_present"]:
        raise PhaseBContractError("B-IPC1 FAILURE candidate file evidence differs")
    if not isinstance(audit["candidate_module_state"], list) or not isinstance(
        receipt["candidate_files_present"], list
    ):
        raise PhaseBContractError("B-IPC1 FAILURE candidate state evidence differs")
    for record in audit["candidate_module_state"]:
        _require_exact_keys(record, {"name", "sha256"}, label="FAILURE candidate module state")
    if receipt.get("model_loaded") is True and not all(
        audit.get(key) is True
        for key in (
            "e33_tensor_preserved",
            "e33_disk_preserved",
            "metadata_preserved",
            "e33_gradients_absent",
        )
    ):
        raise PhaseBContractError("B-IPC1 FAILURE protected post-model audit differs")
    if receipt.get("model_loaded") is not True and any(
        audit[key] is not None
        for key in ("e33_tensor_preserved", "e33_disk_preserved", "metadata_preserved", "e33_gradients_absent")
    ):
        raise PhaseBContractError("B-IPC1 FAILURE claims model evidence before model load")


def _atomic_publish_bytes(directory: Path, name: str, payload: bytes) -> Path:
    if name not in {"SUCCESS.json", "FAILURE.json"}:
        raise PhaseBContractError("B-IPC1 terminal filename differs")
    if not directory.is_dir() or directory.is_symlink():
        raise PhaseBContractError("B-IPC1 terminal output directory is invalid")
    parsed = strict_json_loads(payload)
    if canonical_terminal_bytes(parsed) != payload:
        raise PhaseBContractError("B-IPC1 terminal payload is not canonical before publication")
    target = directory / name
    temporary = directory / f".{name}.{os.getpid()}.tmp"
    terminals = [directory / terminal for terminal in ("SUCCESS.json", "FAILURE.json")]
    if any(path.exists() or path.is_symlink() for path in terminals) or temporary.exists() or temporary.is_symlink():
        raise FileExistsError("B-IPC1 terminal namespace is not globally fresh")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        _fsync_directory(directory)
        _fsync_directory(directory.parent)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
            _fsync_directory(directory)
    if [path.name for path in terminals if path.is_file() and not path.is_symlink()] != [name]:
        raise PhaseBContractError("B-IPC1 terminal exclusivity failed after publication")
    return target


def _post_failure_audit(plan: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    errors = []
    result: dict[str, Any] = {
        "immutable_inputs_preserved": False,
        "immutable_inputs": [],
        "e33_tensor_preserved": None,
        "e33_disk_preserved": None,
        "metadata_preserved": None,
        "e33_gradients_absent": None,
        "candidate_files_present": [],
        "candidate_files_valid": False,
        "candidate_module_state": [],
        "candidate_initial_state": audit.get("module_pre"),
        "cache_guard": None,
        "cuda_memory": None,
    }
    try:
        result["immutable_inputs"] = _immutable_input_records(plan, require_match=False)
        result["immutable_inputs_preserved"] = all(
            record["expected_sha256"] == record["observed_sha256"] for record in result["immutable_inputs"]
        )
    except BaseException as error:
        errors.append(f"immutable:{type(error).__name__}:{error}")
    model = audit.get("model")
    smoke = audit.get("smoke")
    if model is not None and smoke is not None:
        try:
            current_tensor = smoke._module_tensor_sha256(model, audit["torch"])
            result["e33_tensor_preserved"] = current_tensor == audit.get("e33_tensor_pre")
            model_path = audit["model_path"]
            result["e33_disk_preserved"] = (
                file_sha256(smoke._model_file(model_path)) == plan["protected_model"]["weight_sha256"]
            )
            result["metadata_preserved"] = smoke._metadata_hashes(model_path) == plan["model_metadata_sha256"]
            result["e33_gradients_absent"] = all(parameter.grad is None for parameter in model.parameters())
        except BaseException as error:
            errors.append(f"e33:{type(error).__name__}:{error}")
    try:
        output = Path(audit["output_dir"])
        result["candidate_files_present"] = sorted(path.name for path in output.glob("*.pt")) if output.is_dir() else []
        modules = audit.get("modules") or []
        if modules and smoke is not None:
            result["candidate_module_state"] = [
                {"name": record["name"], "sha256": smoke._module_tensor_sha256(record["module"], audit["torch"])}
                for record in modules
            ]
    except BaseException as error:
        errors.append(f"candidate:{type(error).__name__}:{error}")
    guard = audit.get("cache_guard")
    if guard is not None:
        try:
            result["cache_guard"] = guard.evidence()
        except BaseException as error:
            errors.append(f"cache:{type(error).__name__}:{error}")
    result["cuda_memory"] = _memory(audit["torch"]) if audit.get("torch") is not None else None
    return {"audit_complete": not errors, "audit_errors": errors, **result}


def _failure_class(error: BaseException) -> tuple[str, str]:
    if isinstance(error, CacheContractViolated):
        return FAILURE_STATUS_CLASSES[1]
    if isinstance(error, MechanismRejected):
        return FAILURE_STATUS_CLASSES[0]
    if isinstance(error, (ResourceContractExceeded, ProvenanceContractViolated, TimeoutError, MemoryError)) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    ):
        return FAILURE_STATUS_CLASSES[3]
    if isinstance(error, PhaseBContractError):
        return FAILURE_STATUS_CLASSES[2]
    if isinstance(error, RuntimeError):
        return FAILURE_STATUS_CLASSES[2]
    return FAILURE_STATUS_CLASSES[2]


def main() -> int:
    args = parse_args()
    started = time.time()
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(COMPUTE_SECONDS)
    audit: dict[str, Any] = {"output_dir": str(args.output_dir)}
    plan: dict[str, Any] | None = None
    try:
        context = preflight(args)
        plan = context["plan"]
        if not args.preflight_only:
            args.output_dir.mkdir(mode=0o700)
        import pyarrow.parquet as parquet
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        context = tokenizer_preflight(context, parquet=parquet, AutoTokenizer=AutoTokenizer)
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "status": "preflight_passed",
                        "train_rows": len(context["rendered"]["train"]),
                        "validation_rows": len(context["rendered"]["validation"]),
                        "heldout_content_opened": False,
                        "model_loaded": False,
                        "cuda_initialized": context["smoke"]._cuda_initialized_if_torch_loaded(),
                    },
                    sort_keys=True,
                )
            )
            return 0
        import fla.models.utils  # noqa: F401
        import torch
        import transformers
        import transformers.cache_utils  # noqa: F401

        from prime_rl.latent.local_depth import LocalDepthCodec
        from prime_rl.latent.recurrent import (
            OneShotFeedForwardSidecar,
            TimestepFreeRecurrentSidecar,
            diagnose_recurrent_states,
        )

        receipt = execute(
            context,
            execution_commit=args.execution_commit,
            output=args.output_dir,
            torch=torch,
            transformers=transformers,
            AutoModelForImageTextToText=AutoModelForImageTextToText,
            LocalDepthCodec=LocalDepthCodec,
            OneShotFeedForwardSidecar=OneShotFeedForwardSidecar,
            TimestepFreeRecurrentSidecar=TimestepFreeRecurrentSidecar,
            diagnose_recurrent_states=diagnose_recurrent_states,
            parquet=parquet,
            audit=audit,
        )
        receipt["elapsed_seconds"] = time.time() - started
        receipt, payload, _file_hash = roundtrip_validate_terminal(
            receipt,
            validator=validate_success_receipt,
            validator_kwargs={
                "plan": plan,
                "execution_commit": args.execution_commit,
                "output_dir": args.output_dir,
                "torch": torch,
            },
        )
        signal.alarm(0)
        target = _atomic_publish_bytes(args.output_dir, "SUCCESS.json", payload)
        verify_published_terminal(
            target,
            payload,
            validator=validate_success_receipt,
            validator_kwargs={
                "plan": plan,
                "execution_commit": args.execution_commit,
                "output_dir": args.output_dir,
                "torch": torch,
            },
        )
        return 0
    except BaseException as error:
        signal.alarm(0)
        if plan is None:
            print(f"B-IPC1 preflight failed: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
        signal.alarm(AUDIT_SECONDS)
        status, failure_class = _failure_class(error)
        post = _post_failure_audit(plan, audit)
        failure = {
            "schema_version": "q35-2b-phase-b-ipc1-matched-learning-failure/v1",
            "terminal": "FAILURE",
            "status": status,
            "disposition": status,
            "failure_class": failure_class,
            "error_type": type(error).__name__,
            "error": str(error),
            "execution_commit": args.execution_commit,
            "plan_sha256": plan["_file_sha256"],
            "run_identity": plan["run_identity"],
            "model_loaded": audit.get("model") is not None,
            "candidate_files_valid": False,
            "candidate_files_present": sorted(path.name for path in args.output_dir.glob("*.pt")),
            "execution_breadcrumbs": {key: audit.get(key) for key in ("stage", "task_key", "arm", "call_index")},
            "post_failure_audit": post,
            "elapsed_seconds": time.time() - started,
        }
        try:
            if any(
                (args.output_dir / name).exists() or (args.output_dir / name).is_symlink()
                for name in ("SUCCESS.json", "FAILURE.json")
            ):
                print("B-IPC1 terminal already published; refusing a second terminal", file=sys.stderr)
                print(f"B-IPC1 failed after terminal publication: {type(error).__name__}: {error}", file=sys.stderr)
                return 1
            failure, payload, _file_hash = roundtrip_validate_terminal(
                failure,
                validator=validate_failure_receipt,
                validator_kwargs={"plan": plan, "execution_commit": args.execution_commit},
            )
            signal.alarm(0)
            target = _atomic_publish_bytes(args.output_dir, "FAILURE.json", payload)
            verify_published_terminal(
                target,
                payload,
                validator=validate_failure_receipt,
                validator_kwargs={"plan": plan, "execution_commit": args.execution_commit},
            )
        except BaseException as publication_error:
            print(f"B-IPC1 failure publication failed: {publication_error}", file=sys.stderr)
        print(f"B-IPC1 failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
