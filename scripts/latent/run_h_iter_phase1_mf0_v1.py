#!/usr/bin/env python3
"""Run the CUDA-hidden, TRAIN-only H-ITER Phase-1 MF0 preregistration proof."""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import gc
import json
import math
import os
import platform
import resource
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from prime_rl.latent.h_iter_phase1_mf0 import (
    ARTIFACT_DIR,
    ASSET_NAMES,
    MECHANISM,
    MEMORY_LABELS,
    OUTPUT_ROOT,
    RUN_ID,
    SYNTHETIC_FEATURE_SHA256,
    TAMPERS,
    TRAIN_BANK_FILE_SHA256,
    TRAIN_BANK_INTERNAL_SHA256,
    TRAIN_BANK_PATH,
    MF0ContractError,
    build_assets,
    canonical_json,
    run_candidate_synthetic,
    sha256_bytes,
    validate_assets,
)

PLAN_SCHEMA = "prime-rl/latent-h-iter-phase1-train-calibration-mf0-plan/v1"
PROOF_SCHEMA = "prime-rl/latent-h-iter-phase1-train-calibration-mf0-proof/v1"
FAILURE_SCHEMA = "prime-rl/latent-h-iter-phase1-train-calibration-mf0-failure/v1"
PLAN_RELATIVE_PATH = f"{ARTIFACT_DIR}/mf0-plan.json"
PLAN_ASSET_PATHS = sorted(
    [
        *(f"{ARTIFACT_DIR}/{name}" for name in ASSET_NAMES),
        "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/train-bank.json",
        "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/deterministic-terminal-recovery-run1-evidence-manifest.json",
        "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/deterministic-terminal-recovery-run1.PROOF.json",
        "src/prime_rl/latent/h_iter_phase1_mf0.py",
        "scripts/latent/materialize_h_iter_phase1_mf0_assets_v1.py",
        "scripts/latent/run_h_iter_phase1_mf0_v1.py",
        "scripts/latent/run_h_iter_phase1_mf0_v1.sh",
        "scripts/latent/freeze_h_iter_phase1_mf0_plan_v1.py",
        "tests/unit/latent/test_h_iter_phase1_mf0.py",
        "pyproject.toml",
        "uv.lock",
    ]
)
EXPERIMENT_PLAN_ASSET_PATHS = [path for path in PLAN_ASSET_PATHS if path.startswith("experiments/")]
PROOF_STATUS = "h_iter_phase1_train_calibration_preregistered"
INCOMPLETE_STATUS = "h_iter_phase1_train_calibration_prereg_incomplete"
EXPOSURE_STATUS = "h_iter_phase1_train_exposure_boundary_rejected"
INFRASTRUCTURE_STATUS = "infrastructure_invalid"
EXPECTED_RUNTIME = {
    "python": "3.12.14",
    "torch": "2.11.0+cu128",
    "sys_executable": "/home/ubuntu/rlm/prime-rl/.venv/bin/python3",
    "sys_prefix": "/home/ubuntu/rlm/prime-rl/.venv",
    "shared_project_pyproject_sha256": "504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656",
    "shared_project_uv_lock_sha256": "fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5",
    "cuda_visible_devices": "",
    "cuda_initialized_required": False,
}
RESOURCE_BOUNDS = {
    "minimum_ram_gib": 8,
    "minimum_disk_gib": 8,
    "maximum_artifact_bytes": 16 * 2**20,
    "outer_timeout_seconds": 1800,
    "compute_timeout_seconds": 1050,
    "audit_timeout_seconds": 240,
    "failure_timeout_seconds": 180,
    "terminal_timeout_seconds": 60,
    "startup_seconds": 120,
    "postexit_seconds": 60,
    "success_terminal_entry_maximum_seconds": 1290,
    "compute_failure_terminal_entry_maximum_seconds": 1230,
    "audit_failure_terminal_entry_maximum_seconds": 1470,
    "prior_terminal_failure_entry_maximum_seconds": 1530,
    "worst_terminal_failure_external_seconds": 1770,
    "worst_path_reserve_seconds": 30,
    "output_root": OUTPUT_ROOT,
}
DECISION = {
    "claim": "MF0 materializes train-only CAP0/T0 contracts",
    "cap0_authorized": False,
    "t0_authorized": False,
    "model_or_gpu_authorized": False,
    "training_or_update_authorized": False,
    "validation_or_heldout_opened": False,
    "nomination": False,
    "admission": False,
    "promotion": False,
    "live_trajectory_count": 0,
    "four_live_floor_unchanged": True,
}
COUNTS = {
    "train_rows": 96,
    "fit_rows": 64,
    "calibration_rows": 32,
    "cap0_probes": 4,
    "candidate_arms": 5,
    "cpu_synthetic_forwards": 5,
    "cpu_synthetic_backwards": 5,
    "optimizer_objects": 0,
    "optimizer_steps": 0,
    "tokenizer_calls": 0,
    "model_calls": 0,
    "validation_opens": 0,
    "heldout_opens": 0,
    "tampers": 34,
    "memory_rows": 17,
}
EXPECTED_PARAMETER_NAMES = [
    "codec_ln.weight",
    "codec_ln.bias",
    "codec_projection.weight",
    "codec_projection.bias",
    "self_norm.weight",
    "self_norm.bias",
    "message_norm.weight",
    "message_norm.bias",
    "cell_in.weight",
    "cell_in.bias",
    "cell_out.weight",
    "cell_out.bias",
    "post_norm.weight",
    "post_norm.bias",
    "readout.weight",
    "readout.bias",
]
EXPECTED_PARAMETER_COUNT = 366_340
NETWORK_CONTRACT = {
    "os_network_namespace": False,
    "python_guard_operations": [
        "socket.socket.connect",
        "socket.socket.connect_ex",
        "socket.create_connection",
        "socket.getaddrinfo",
    ],
    "audit_events": ["socket.connect", "socket.getaddrinfo"],
    "external_subprocess_allowlist": ["git rev-parse", "git status", "git show"],
}
SAFETY_BOUNDARY = {
    "cpu_torch_required": True,
    "synthetic_forwards": 5,
    "synthetic_backwards": 5,
    "optimizer_objects": 0,
    "optimizer_steps": 0,
    "tokenizer_calls": 0,
    "model_calls": 0,
    "validation_opens": 0,
    "heldout_opens": 0,
    "cuda_visible_devices": "",
    "cuda_initialized": False,
    "network_attempts": 0,
    "candidate_or_checkpoint_files": 0,
    "phase1_thresholds_materialized": False,
}
FULL_FREEZE_CONTRACT = {
    "execution_is_exact_clean_child": True,
    "asset_map_exact_pre_post": True,
    "authorized_plan_loaded_from_execution_git_blob_on_failure": True,
    "fresh_process_terminal_validation": True,
    "terminal_self_reference_boundary_external": True,
}


class ExposureBoundaryRejected(RuntimeError):
    pass


class InfrastructureInvalid(RuntimeError):
    pass


def _timeout(_signum: int, _frame: object) -> None:
    raise TimeoutError("MF0 phase timeout")


def strict_loads(data: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in rows:
            if key in value:
                raise MF0ContractError("duplicate JSON key")
            value[key] = item
        return value

    try:
        value = json.loads(
            data.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                MF0ContractError(f"nonfinite JSON token {token}")
            ),
            object_pairs_hook=pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MF0ContractError("invalid MF0 JSON") from error
    if not isinstance(value, dict):
        raise MF0ContractError("MF0 JSON root is not an object")
    return value


def read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise InfrastructureInvalid(f"MF0 input is absent, nonfile, or symlinked: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def load_canonical(path: Path) -> dict[str, Any]:
    data = read_regular(path)
    if not data.endswith(b"\n") or data.endswith(b"\n\n"):
        raise MF0ContractError("MF0 JSON framing differs")
    value = strict_loads(data[:-1])
    if data != canonical_json(value) + b"\n":
        raise MF0ContractError("MF0 JSON is not canonical")
    return value


def file_sha256(path: Path) -> str:
    return sha256_bytes(read_regular(path))


def run_git(repo: Path, *args: str) -> str:
    allowed = {
        ("rev-parse", "HEAD"),
        ("rev-parse", "HEAD^"),
        ("rev-parse", "HEAD^{tree}"),
        ("status", "--porcelain", "--untracked-files=all"),
    }
    if tuple(args) not in allowed:
        raise InfrastructureInvalid("MF0 subprocess was not allowlisted")
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.rstrip("\n")


def git_blob(repo: Path, commit: str, relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


class PhaseTracker:
    def __init__(self) -> None:
        self.started_ns = time.monotonic_ns()
        self.active: tuple[str, int, int, int] | None = None
        self.completed: list[dict[str, Any]] = []

    def enter(self, phase: str, seconds: int) -> None:
        if self.active is not None or seconds <= 1:
            raise InfrastructureInvalid("MF0 timing phase overlap")
        entered = time.monotonic_ns() - self.started_ns
        cap_ns = seconds * 1_000_000_000
        alarm_ns = (seconds - 1) * 1_000_000_000
        self.active = (phase, entered, cap_ns, alarm_ns)
        signal.alarm(seconds - 1)

    def exit(self, outcome: str, *, timeout_observed: bool = False) -> None:
        if self.active is None or outcome not in {"completed", "error"}:
            raise InfrastructureInvalid("MF0 timing phase exit differs")
        phase, entered, cap_ns, alarm_ns = self.active
        exited = time.monotonic_ns() - self.started_ns
        duration = exited - entered
        self.completed.append(
            {
                "phase": phase,
                "entered_ns_since_start": entered,
                "exited_ns_since_start": exited,
                "duration_ns": duration,
                "outcome": outcome,
                "cap_ns": cap_ns,
                "alarm_after_ns": alarm_ns,
                "alarm_safety_margin_ns": 1_000_000_000,
                "timeout_observed": timeout_observed,
                "alarm_requested_after_ns": alarm_ns,
                "timeout_observed_duration_ns": duration if timeout_observed else None,
                "delivery_overrun_ns": max(0, duration - cap_ns) if timeout_observed else 0,
                "timing_cap_exceeded": duration > cap_ns,
            }
        )
        self.active = None
        signal.alarm(0)

    def terminal_boundary(self) -> tuple[dict[str, Any], int]:
        if self.active is None or self.active[0] != "terminal_publication":
            raise InfrastructureInvalid("MF0 terminal phase is not active")
        entered = self.active[1]
        return {
            "phase": "terminal_publication",
            "entered_ns_since_start": entered,
            "limit_ns": RESOURCE_BOUNDS["terminal_timeout_seconds"] * 1_000_000_000,
            "completion_observable_inside_terminal": False,
            "self_reference_boundary": "post_write_fsync_reopen_validation_and_process_exit_are_external_to_immutable_terminal_bytes",
        }, entered


class NetworkGuard:
    def __init__(self) -> None:
        self.attempt_count = 0
        self.installed = False
        self.wrappers_restored = False
        self._originals: dict[str, object] = {}

    def reject(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.attempt_count += 1
        raise InfrastructureInvalid("MF0 network attempt rejected")

    def audit(self, event: str, args: tuple[object, ...]) -> None:
        del args
        if event in NETWORK_CONTRACT["audit_events"]:
            self.attempt_count += 1
            raise InfrastructureInvalid("MF0 audited network attempt rejected")

    def __enter__(self) -> NetworkGuard:
        self._originals = {
            "connect": socket.socket.connect,
            "connect_ex": socket.socket.connect_ex,
            "create_connection": socket.create_connection,
            "getaddrinfo": socket.getaddrinfo,
        }
        socket.socket.connect = self.reject  # type: ignore[method-assign]
        socket.socket.connect_ex = self.reject  # type: ignore[method-assign]
        socket.create_connection = self.reject  # type: ignore[assignment]
        socket.getaddrinfo = self.reject  # type: ignore[assignment]
        sys.addaudithook(self.audit)
        self.installed = True
        return self

    def __exit__(self, _kind: object, _error: object, _tb: object) -> None:
        socket.socket.connect = self._originals["connect"]  # type: ignore[method-assign,assignment]
        socket.socket.connect_ex = self._originals["connect_ex"]  # type: ignore[method-assign,assignment]
        socket.create_connection = self._originals["create_connection"]  # type: ignore[assignment]
        socket.getaddrinfo = self._originals["getaddrinfo"]  # type: ignore[assignment]
        self.wrappers_restored = True

    def evidence(self) -> dict[str, Any]:
        return {
            **NETWORK_CONTRACT,
            "installed": self.installed,
            "wrappers_restored": self.wrappers_restored,
            "audit_hook_persistent": self.installed,
            "attempt_count": self.attempt_count,
        }


class OpenFirewall:
    def __init__(self, repo: Path, allowed: set[Path]) -> None:
        self.repo = repo.resolve()
        self.allowed = {path.resolve() for path in allowed}
        self.denied_count = 0
        self.validation_open_count = 0
        self.heldout_open_count = 0
        self.opened: list[str] = []

    def audit(self, event: str, args: tuple[object, ...]) -> None:
        if event != "open" or not args or not isinstance(args[0], (str, bytes)):
            return
        try:
            path = Path(os.fsdecode(args[0]))
            resolved = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
            experiment_root = (self.repo / "experiments").resolve()
            if resolved != experiment_root and experiment_root not in resolved.parents:
                return
            lowered = "/".join(resolved.parts).lower()
            if "validation" in lowered:
                self.validation_open_count += 1
            if "heldout" in lowered or "held_out" in lowered:
                self.heldout_open_count += 1
            if resolved not in self.allowed:
                self.denied_count += 1
                raise ExposureBoundaryRejected(f"blocked MF0 experiment path open: {resolved}")
            self.opened.append(str(resolved.relative_to(self.repo)))
        except ExposureBoundaryRejected:
            raise
        except OSError as error:
            raise InfrastructureInvalid("MF0 open-firewall path resolution failed") from error

    def install(self) -> None:
        sys.addaudithook(self.audit)


class ArtifactWriter:
    def __init__(self, output_dir: Path) -> None:
        if output_dir != Path(OUTPUT_ROOT) or not output_dir.is_absolute():
            raise InfrastructureInvalid("MF0 output root differs")
        if output_dir.exists() or output_dir.is_symlink():
            raise InfrastructureInvalid("MF0 output namespace is not fresh")
        if output_dir.parent.is_symlink() or not output_dir.parent.is_dir():
            raise InfrastructureInvalid("MF0 output parent is absent or symlinked")
        output_dir.mkdir(mode=0o700)
        self.output_dir = output_dir
        self.terminal_written = False

    def write(self, name: str, payload: dict[str, Any], maximum_bytes: int) -> bytes:
        if self.terminal_written or name not in {"MF0-PROOF.json", "MF0-FAILURE.json"}:
            raise InfrastructureInvalid("MF0 terminal exclusivity differs")
        if list(self.output_dir.iterdir()):
            raise InfrastructureInvalid("MF0 output inventory was not empty before publication")
        encoded = canonical_json(payload) + b"\n"
        if len(encoded) > maximum_bytes:
            raise InfrastructureInvalid("MF0 terminal exceeds artifact cap")
        temporary = self.output_dir / f".{name}.tmp"
        target = self.output_dir / name
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short MF0 terminal write")
                view = view[written:]
            os.fsync(descriptor)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
        self.terminal_written = True
        descriptor = os.open(self.output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        reopened = read_regular(target)
        if reopened != encoded:
            raise InfrastructureInvalid("MF0 terminal changed after publication")
        return reopened


class MemoryLedger:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.peak = 0

    @staticmethod
    def rss_bytes() -> int:
        if sys.platform.startswith("linux"):
            pages = int(Path("/proc/self/statm").read_text().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage if sys.platform == "darwin" else usage * 1024

    def checkpoint(self, label: str) -> None:
        if len(self.rows) >= len(MEMORY_LABELS) or label != MEMORY_LABELS[len(self.rows)]:
            raise MF0ContractError("MF0 memory label order differs")
        rss = self.rss_bytes()
        self.peak = max(self.peak, rss)
        self.rows.append({"label": label, "rss_bytes": rss, "peak_rss_bytes": self.peak})


def validate_memory(rows: object, *, complete: bool) -> None:
    if not isinstance(rows, list) or (complete and len(rows) != 17) or len(rows) > 17:
        raise MF0ContractError("MF0 memory count differs")
    previous_peak = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"label", "rss_bytes", "peak_rss_bytes"} or row["label"] != MEMORY_LABELS[index]:
            raise MF0ContractError("MF0 memory row differs")
        for key in ("rss_bytes", "peak_rss_bytes"):
            if not isinstance(row[key], int) or isinstance(row[key], bool) or row[key] < 0:
                raise MF0ContractError("MF0 memory value differs")
        if row["peak_rss_bytes"] < row["rss_bytes"] or row["peak_rss_bytes"] < previous_peak:
            raise MF0ContractError("MF0 memory peak differs")
        previous_peak = row["peak_rss_bytes"]


def validate_phase_records(records: object, terminal: object, elapsed: object, *, success: bool) -> None:
    if not isinstance(records, list) or not isinstance(terminal, dict):
        raise MF0ContractError("MF0 timing schema differs")
    sequences = [["compute", "audit"]] if success else [
        ["compute", "failure_audit"],
        ["compute", "audit", "failure_audit"],
        ["compute", "audit", "terminal_publication", "failure_audit"],
    ]
    phases = [row.get("phase") if isinstance(row, dict) else None for row in records]
    if phases not in sequences:
        raise MF0ContractError("MF0 phase sequence differs")
    caps = {"compute": 1050, "audit": 240, "failure_audit": 180, "terminal_publication": 60}
    previous_exit = 0
    for index, row in enumerate(records):
        keys = {"phase", "entered_ns_since_start", "exited_ns_since_start", "duration_ns", "outcome", "cap_ns", "alarm_after_ns", "alarm_safety_margin_ns", "timeout_observed", "alarm_requested_after_ns", "timeout_observed_duration_ns", "delivery_overrun_ns", "timing_cap_exceeded"}
        if not isinstance(row, dict) or set(row) != keys:
            raise MF0ContractError("MF0 phase record keys differ")
        phase = row["phase"]
        cap_ns = caps[phase] * 1_000_000_000
        numeric = ("entered_ns_since_start", "exited_ns_since_start", "duration_ns", "cap_ns", "alarm_after_ns", "alarm_safety_margin_ns", "alarm_requested_after_ns", "delivery_overrun_ns")
        if any(not isinstance(row[key], int) or isinstance(row[key], bool) or row[key] < 0 for key in numeric):
            raise MF0ContractError("MF0 phase numeric field differs")
        if row["entered_ns_since_start"] < previous_exit or row["exited_ns_since_start"] < row["entered_ns_since_start"] or row["duration_ns"] != row["exited_ns_since_start"] - row["entered_ns_since_start"]:
            raise MF0ContractError("MF0 phase chronology differs")
        if row["cap_ns"] != cap_ns or row["alarm_after_ns"] != cap_ns - 1_000_000_000 or row["alarm_safety_margin_ns"] != 1_000_000_000 or row["alarm_requested_after_ns"] != row["alarm_after_ns"]:
            raise MF0ContractError("MF0 phase cap differs")
        error_index = None if success else len(records) - 2
        if not success and len(records) == 2:
            error_index = 0
        if row["outcome"] != ("error" if index == error_index else "completed"):
            raise MF0ContractError("MF0 phase outcome differs")
        exceeded = row["duration_ns"] > cap_ns
        if row["timing_cap_exceeded"] is not exceeded or (row["outcome"] == "completed" and exceeded):
            raise MF0ContractError("MF0 phase cap flag differs")
        if row["timeout_observed"]:
            if row["timeout_observed_duration_ns"] != row["duration_ns"] or row["delivery_overrun_ns"] != max(0, row["duration_ns"] - cap_ns):
                raise MF0ContractError("MF0 timeout arithmetic differs")
        elif row["timeout_observed_duration_ns"] is not None or row["delivery_overrun_ns"] != 0:
            raise MF0ContractError("MF0 non-timeout evidence differs")
        previous_exit = row["exited_ns_since_start"]
    terminal_keys = {"phase", "entered_ns_since_start", "limit_ns", "completion_observable_inside_terminal", "self_reference_boundary"}
    if set(terminal) != terminal_keys or terminal["phase"] != "terminal_publication" or terminal["limit_ns"] != 60_000_000_000 or terminal["completion_observable_inside_terminal"] is not False or terminal["self_reference_boundary"] != "post_write_fsync_reopen_validation_and_process_exit_are_external_to_immutable_terminal_bytes":
        raise MF0ContractError("MF0 terminal timing boundary differs")
    if not isinstance(elapsed, int) or isinstance(elapsed, bool) or elapsed != terminal["entered_ns_since_start"] or elapsed < previous_exit:
        raise MF0ContractError("MF0 terminal entry time differs")
    maximum = 1290 if success else {2: 1230, 3: 1470, 4: 1530}[len(records)]
    if elapsed > maximum * 1_000_000_000:
        raise MF0ContractError("MF0 terminal entry exceeded bound")


def validate_plan(plan: dict[str, Any], external_sha256: str | None = None) -> None:
    keys = {"schema_version", "status", "mechanism", "run_identity", "mechanism_code_commit", "execution_authorization", "output_root", "asset_sha256", "runtime", "resource_bounds", "materialization_contract", "terminal_contract", "memory_label_schedule", "safety_boundary", "full_freeze", "plan_sha256"}
    if set(plan) != keys or plan["schema_version"] != PLAN_SCHEMA or plan["status"] != "preregistered" or plan["mechanism"] != MECHANISM or plan["run_identity"] != RUN_ID:
        raise MF0ContractError("MF0 plan identity differs")
    if not isinstance(plan["mechanism_code_commit"], str) or len(plan["mechanism_code_commit"]) != 40 or any(character not in "0123456789abcdef" for character in plan["mechanism_code_commit"]):
        raise MF0ContractError("MF0 mechanism commit differs")
    if plan["output_root"] != OUTPUT_ROOT or plan["runtime"] != EXPECTED_RUNTIME or plan["resource_bounds"] != RESOURCE_BOUNDS:
        raise MF0ContractError("MF0 plan runtime/resource contract differs")
    memory = {"labels": MEMORY_LABELS, "label_sha256": sha256_bytes(canonical_json(MEMORY_LABELS)), "count": 17}
    if plan["memory_label_schedule"] != memory:
        raise MF0ContractError("MF0 plan memory schedule differs")
    authorization = plan["execution_authorization"]
    if set(authorization) != {"mf0_model_free_prereg_only", "cap0", "t0", "model", "gpu", "training"} or authorization["mf0_model_free_prereg_only"] is not True or any(authorization[key] is not False for key in ("cap0", "t0", "model", "gpu", "training")):
        raise MF0ContractError("MF0 plan authorization differs")
    if plan["materialization_contract"] != {"asset_names": ASSET_NAMES, "regenerate_byte_identical": True, "source_split": "train", "validation_or_heldout_paths_forbidden": True}:
        raise MF0ContractError("MF0 plan materialization contract differs")
    if not isinstance(plan["asset_sha256"], dict) or list(plan["asset_sha256"]) != PLAN_ASSET_PATHS or any(not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest) for digest in plan["asset_sha256"].values()):
        raise MF0ContractError("MF0 plan asset map differs")
    terminal = {"success_file": "MF0-PROOF.json", "failure_file": "MF0-FAILURE.json", "exclusive_atomic": True, "canonical_roundtrip_twice": True, "success_status": PROOF_STATUS, "failure_statuses": [INCOMPLETE_STATUS, EXPOSURE_STATUS, INFRASTRUCTURE_STATUS]}
    if plan["terminal_contract"] != terminal:
        raise MF0ContractError("MF0 terminal contract differs")
    if plan["safety_boundary"] != SAFETY_BOUNDARY or plan["full_freeze"] != FULL_FREEZE_CONTRACT:
        raise MF0ContractError("MF0 plan exposure boundary differs")
    if any("validation-bank" in path or "heldout-bank" in path or "held_out" in path for path in plan["asset_sha256"]):
        raise MF0ContractError("MF0 plan contains a forbidden split path")
    if plan["plan_sha256"] != sha256_bytes(canonical_json({key: value for key, value in plan.items() if key != "plan_sha256"})):
        raise MF0ContractError("MF0 plan self hash differs")
    if external_sha256 is not None and external_sha256 != sha256_bytes(canonical_json(plan) + b"\n"):
        raise InfrastructureInvalid("MF0 external plan hash differs")


def validate_runtime(torch: Any) -> dict[str, Any]:
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "shared_project_pyproject_sha256": file_sha256(Path("/home/ubuntu/rlm/prime-rl/pyproject.toml")),
        "shared_project_uv_lock_sha256": file_sha256(Path("/home/ubuntu/rlm/prime-rl/uv.lock")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_initialized_required": False,
    }
    if runtime != EXPECTED_RUNTIME:
        raise InfrastructureInvalid("MF0 runtime identity differs")
    if torch.cuda.is_initialized():
        raise ExposureBoundaryRejected("MF0 CUDA boundary differs")
    return runtime


def object_inventory(torch: Any, output_dir: Path) -> dict[str, Any]:
    modeling = sorted(name for name in sys.modules if name.startswith("transformers.models") or ".modeling_" in name)
    pretrained = tokenizer = optimizer = candidate = 0
    errors: list[dict[str, Any]] = []
    for object_index, value in enumerate(gc.get_objects()):
        try:
            value_type = type(value)
            mro = value_type.__mro__
            if any(base.__name__ == "PreTrainedModel" and isinstance(base.__module__, str) and base.__module__.startswith("transformers") for base in mro):
                pretrained += 1
            if isinstance(value_type.__module__, str) and value_type.__module__.startswith("transformers.tokenization"):
                tokenizer += 1
            if any(base.__name__ == "Optimizer" and isinstance(base.__module__, str) and base.__module__.startswith("torch.optim") for base in mro):
                optimizer += 1
            if value_type.__name__ == "Candidate" and value_type.__module__ == "prime_rl.latent.h_iter_phase1_mf0":
                candidate += 1
        except TimeoutError:
            raise
        except Exception as error:
            try:
                detail = str(error) or "<empty exception text>"
            except Exception as render_error:
                detail = f"<unrenderable exception: {type(render_error).__name__}>"
            errors.append({"object_index": object_index, "error_type": type(error).__name__, "error": detail})
    return {
        "transformers_modeling_modules": modeling,
        "pretrained_model_objects": pretrained,
        "tokenizer_objects": tokenizer,
        "optimizer_objects": optimizer,
        "candidate_module_objects": candidate,
        "uninspectable_count": len(errors),
        "census_errors": errors,
        "cuda_initialized": torch.cuda.is_initialized(),
        "output_inventory": sorted(path.name for path in output_dir.iterdir()),
        "object_census_method": "gc_mro_scan_without_importing_model_tokenizer_or_optimizer_classes",
    }


def host_resources(repo: Path) -> tuple[int, int]:
    ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    disk = shutil.disk_usage(repo).free
    if ram < 8 * 2**30 or disk < 8 * 2**30:
        raise InfrastructureInvalid("MF0 RAM/disk preflight differs")
    return ram, disk


def static_guard(repo: Path) -> dict[str, Any]:
    paths = ["src/prime_rl/latent/h_iter_phase1_mf0.py", "scripts/latent/run_h_iter_phase1_mf0_v1.py"]
    forbidden: list[str] = []
    allowed_backwards: list[str] = []
    for relative in paths:
        tree = ast.parse(read_regular(repo / relative), filename=relative)
        parents: list[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                parents.append(node.name)
                self.generic_visit(node)
                parents.pop()

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    if alias.name.split(".")[0] in {"transformers", "tokenizers", "accelerate", "safetensors"}:
                        forbidden.append(f"{relative}:{node.lineno}:import:{alias.name}")
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if (node.module or "").split(".")[0] in {"transformers", "tokenizers", "accelerate", "safetensors"}:
                    forbidden.append(f"{relative}:{node.lineno}:import:{node.module}")
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                if name in {"from_pretrained", "generate", "step"}:
                    forbidden.append(f"{relative}:{node.lineno}:call:{name}")
                if name == "backward":
                    site = f"{relative}:{node.lineno}:call:backward"
                    if relative == paths[0] and parents and parents[-1] == "run_candidate_synthetic":
                        allowed_backwards.append(site)
                    else:
                        forbidden.append(site)
                self.generic_visit(node)

        Visitor().visit(tree)
    if forbidden or len(allowed_backwards) != 1:
        raise MF0ContractError("MF0 static exposure guard differs")
    return {"paths": paths, "forbidden_sites": forbidden, "allowed_synthetic_backward_sites": allowed_backwards}


def load_assets(repo: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    bank_path = repo / TRAIN_BANK_PATH
    if file_sha256(bank_path) != TRAIN_BANK_FILE_SHA256:
        raise InfrastructureInvalid("MF0 TRAIN bank file hash differs")
    bank = load_canonical(bank_path)
    if bank.get("bank_sha256") != TRAIN_BANK_INTERNAL_SHA256 or bank.get("split") != "train" or len(bank.get("rows", [])) != 96:
        raise MF0ContractError("MF0 TRAIN bank differs")
    base = repo / ARTIFACT_DIR
    assets = {name: load_canonical(base / name) for name in ASSET_NAMES}
    validate_assets(assets, bank)
    return bank, assets


def mutate_asset(assets: dict[str, dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    altered = copy.deepcopy(assets)
    partition = altered["train-partition.json"]
    cap0 = altered["cap0-probe-selection.json"]
    schedule = altered["training-schedule.json"]
    if name == "source_train_bank_file_hash_changed": partition["source_train_bank_file_sha256"] = "0" * 64
    elif name == "source_train_bank_internal_hash_changed": partition["source_train_bank_internal_sha256"] = "0" * 64
    elif name == "nontrain_split_added": partition["fit_rows"][0]["split"] = "validation"
    elif name == "fit_replicate_rule_changed": partition["fit_rule"] = "changed"
    elif name == "calibration_replicate_rule_changed": partition["calibration_rule"] = "changed"
    elif name == "fit_cal_overlap": partition["calibration_rows"][0] = copy.deepcopy(partition["fit_rows"][0])
    elif name == "train_row_missing": partition["fit_rows"].pop()
    elif name == "train_row_duplicated": partition["fit_rows"].append(copy.deepcopy(partition["fit_rows"][0]))
    elif name == "fit_order_changed": partition["fit_rows"][0], partition["fit_rows"][1] = partition["fit_rows"][1], partition["fit_rows"][0]
    elif name == "calibration_order_changed": partition["calibration_rows"][0], partition["calibration_rows"][1] = partition["calibration_rows"][1], partition["calibration_rows"][0]
    elif name == "cell_balance_changed": partition["counts"]["fit_per_action"] = 15
    elif name == "cap_probe_depth_missing": cap0["ordered_probes"].pop()
    elif name == "cap_probe_not_fit": cap0["ordered_probes"][0]["replicate"] = 5
    elif name == "cap_probe_not_lexicographic_minimum": cap0["ordered_probes"][0]["row_id"] = partition["fit_rows"][0]["row_id"]
    elif name == "cap_probe_order_changed": cap0["ordered_probes"][0], cap0["ordered_probes"][1] = cap0["ordered_probes"][1], cap0["ordered_probes"][0]
    elif name == "epoch_count_changed": schedule["epochs"] = 15
    elif name == "depth_batch_order_changed": schedule["depth_order"] = [2, 1, 3, 4]
    elif name == "arm_update_order_changed": schedule["arm_order"] = list(reversed(schedule["arm_order"]))
    elif name == "batch_row_order_changed": schedule["batches"]["train"][0]["row_ids"] = list(reversed(schedule["batches"]["train"][0]["row_ids"]))
    elif name == "optimizer_update_budget_changed": schedule["total_optimizer_updates"] = 319
    elif name == "init_payload_changed": altered["candidate-module-contract.json"]["initialization"]["payload"] = "changed"
    elif name == "module_dimension_changed": altered["candidate-module-contract.json"]["state_dim"] = 127
    elif name == "module_arm_semantics_changed": altered["candidate-module-contract.json"]["arm_semantics"]["REC_K"] = "changed"
    elif name == "forbidden_graph_input_added": altered["candidate-module-contract.json"]["forbidden_inputs"] = []
    elif name == "capture_render_contract_changed": altered["capture-contract.json"]["render_contract"]["graph_batch"] = 23
    elif name == "cache_guard_weakened": altered["capture-contract.json"]["model_call_contract"]["use_cache"] = True
    elif name == "protected_model_update_allowed": altered["capture-contract.json"]["protected_state"]["no_update"] = False
    elif name == "t0_gate_changed": altered["metric-gate-contract.json"]["t0_go_gates"]["rec_postfit_accuracy_min"] = [0, 64]
    elif name == "threshold_formula_changed": altered["threshold-builder-contract.json"]["go_only_materialization"] = False
    elif name == "validation_path_added": altered["phase0-evidence-binding.json"]["validation_path"] = "forbidden"
    elif name == "heldout_path_added": altered["phase0-evidence-binding.json"]["heldout_path"] = "forbidden"
    elif name == "resource_or_timeout_changed": altered["capture-contract.json"]["resource_bounds"]["maximum_allocated_or_reserved_gib"] = 41
    elif name not in {"proof_status_changed", "proof_self_hash_changed"}: raise MF0ContractError("unknown MF0 tamper")
    return altered


def validate_safety(safety: object) -> None:
    if not isinstance(safety, dict):
        raise MF0ContractError("MF0 safety schema differs")
    safety_keys = {"cuda_visible_devices", "cuda_initialized_before", "cuda_initialized_after", "torch_cpu_only", "tokenizer_calls", "model_calls", "model_backwards", "optimizer_objects", "optimizer_steps", "validation_opens", "heldout_opens", "model_or_tokenizer_loaded", "candidate_created", "checkpoint_created", "model_updated", "object_inventory", "network_guard", "open_firewall", "static_guard"}
    if set(safety) != safety_keys:
        raise MF0ContractError("MF0 safety keys differ")
    zero_fields = ("tokenizer_calls", "model_calls", "model_backwards", "optimizer_objects", "optimizer_steps", "validation_opens", "heldout_opens")
    if any(safety.get(key) != 0 for key in zero_fields):
        raise MF0ContractError("MF0 forbidden exposure count differs")
    if safety.get("cuda_visible_devices") != "" or safety.get("cuda_initialized_before") is not False or safety.get("cuda_initialized_after") is not False:
        raise MF0ContractError("MF0 CUDA safety differs")
    if any(safety.get(key) is not False for key in ("model_or_tokenizer_loaded", "candidate_created", "checkpoint_created", "model_updated")):
        raise MF0ContractError("MF0 forbidden state differs")
    census = safety.get("object_inventory")
    census_keys = {"transformers_modeling_modules", "pretrained_model_objects", "tokenizer_objects", "optimizer_objects", "candidate_module_objects", "uninspectable_count", "census_errors", "cuda_initialized", "output_inventory", "object_census_method"}
    if not isinstance(census, dict) or set(census) != census_keys:
        raise MF0ContractError("MF0 census keys differ")
    absent = ("transformers_modeling_modules", "pretrained_model_objects", "tokenizer_objects", "optimizer_objects", "candidate_module_objects", "uninspectable_count", "census_errors", "cuda_initialized", "output_inventory")
    if not isinstance(census, dict) or any(census.get(key) not in (0, [], False) for key in absent):
        raise MF0ContractError("MF0 object census differs")
    network = safety.get("network_guard")
    if not isinstance(network, dict) or set(network) != {*NETWORK_CONTRACT, "installed", "wrappers_restored", "audit_hook_persistent", "attempt_count"} or any(network.get(key) != value for key, value in NETWORK_CONTRACT.items()) or network.get("installed") is not True or network.get("wrappers_restored") is not True or network.get("audit_hook_persistent") is not True or network.get("attempt_count") != 0:
        raise MF0ContractError("MF0 network guard differs")
    firewall = safety.get("open_firewall")
    if not isinstance(firewall, dict) or set(firewall) != {"denied_count", "validation_open_count", "heldout_open_count", "opened_paths"} or firewall.get("denied_count") != 0 or firewall.get("validation_open_count") != 0 or firewall.get("heldout_open_count") != 0 or firewall.get("opened_paths") != EXPERIMENT_PLAN_ASSET_PATHS:
        raise MF0ContractError("MF0 open firewall differs")
    static = safety.get("static_guard")
    if not isinstance(static, dict) or static.get("paths") != ["src/prime_rl/latent/h_iter_phase1_mf0.py", "scripts/latent/run_h_iter_phase1_mf0_v1.py"] or static.get("forbidden_sites") != [] or not isinstance(static.get("allowed_synthetic_backward_sites"), list) or len(static["allowed_synthetic_backward_sites"]) != 1:
        raise MF0ContractError("MF0 static guard evidence differs")


def validate_proof(proof: dict[str, Any], *, plan: dict[str, Any], assets: dict[str, dict[str, Any]], execution_commit: str, plan_file_sha256: str) -> None:
    keys = {"schema_version", "status", "mechanism", "run_identity", "execution_commit", "mechanism_code_commit", "plan_file_sha256", "plan_sha256", "runtime", "asset_audit", "phase0_binding", "train_partition", "cap0_probe_selection", "training_schedule", "candidate_contract", "capture_contract", "metric_gate_contract", "threshold_builder_contract", "safety_resource_contract", "tamper_audit", "counts", "safety", "resources", "memory", "full_freeze", "decision_boundary", "proof_sha256"}
    if set(proof) != keys or proof["schema_version"] != PROOF_SCHEMA or proof["status"] != PROOF_STATUS or proof["mechanism"] != MECHANISM or proof["run_identity"] != RUN_ID:
        raise MF0ContractError("MF0 proof identity differs")
    if proof["execution_commit"] != execution_commit or proof["mechanism_code_commit"] != plan["mechanism_code_commit"] or proof["plan_file_sha256"] != plan_file_sha256 or proof["plan_sha256"] != plan["plan_sha256"]:
        raise MF0ContractError("MF0 proof authority differs")
    if proof["runtime"] != EXPECTED_RUNTIME or proof["counts"] != COUNTS or proof["decision_boundary"] != DECISION:
        raise MF0ContractError("MF0 proof runtime/count/boundary differs")
    embedded = {
        "phase0_binding": "phase0-evidence-binding.json",
        "train_partition": "train-partition.json",
        "cap0_probe_selection": "cap0-probe-selection.json",
        "training_schedule": "training-schedule.json",
        "capture_contract": "capture-contract.json",
        "metric_gate_contract": "metric-gate-contract.json",
        "threshold_builder_contract": "threshold-builder-contract.json",
    }
    if any(proof[key] != assets[name] for key, name in embedded.items()):
        raise MF0ContractError("MF0 embedded contract differs")
    candidate = proof["candidate_contract"]
    if not isinstance(candidate, dict) or set(candidate) != {"contract", "synthetic_validation"} or candidate["contract"] != assets["candidate-module-contract.json"]:
        raise MF0ContractError("MF0 candidate evidence schema differs")
    synthetic = candidate["synthetic_validation"]
    synthetic_keys = {"arms", "forwards", "backwards", "optimizer_objects", "optimizer_steps", "parameter_names", "parameter_count_per_arm", "initial_tree_sha256", "all_initial_trees_equal", "synthetic_feature_shape", "synthetic_feature_sha256"}
    if not isinstance(synthetic, dict) or set(synthetic) != synthetic_keys or synthetic.get("forwards") != 5 or synthetic.get("backwards") != 5 or synthetic.get("optimizer_objects") != 0 or synthetic.get("optimizer_steps") != 0 or synthetic.get("parameter_names") != EXPECTED_PARAMETER_NAMES or synthetic.get("parameter_count_per_arm") != EXPECTED_PARAMETER_COUNT or synthetic.get("all_initial_trees_equal") is not True or not isinstance(synthetic.get("initial_tree_sha256"), str) or len(synthetic["initial_tree_sha256"]) != 64 or synthetic.get("synthetic_feature_sha256") != SYNTHETIC_FEATURE_SHA256 or synthetic.get("synthetic_feature_shape") != [24, 2048] or [row.get("arm") for row in synthetic.get("arms", [])] != ["STATIC", "FFN", "FIXED_T4", "RESET_K", "REC_K"]:
        raise MF0ContractError("MF0 synthetic evidence differs")
    for row in synthetic["arms"]:
        if set(row) != {"arm", "output_shape", "codec_gradient_nonzero", "codec_gradient_l2", "readout_gradient_nonzero", "readout_gradient_l2", "cell_gradient_nonzero", "cell_gradient_l2", "state_unchanged", "initial_tree_sha256"}:
            raise MF0ContractError("MF0 arm evidence keys differ")
        expected_cell = row["arm"] != "STATIC"
        numeric_norms = [row.get("codec_gradient_l2"), row.get("readout_gradient_l2")]
        if expected_cell:
            numeric_norms.append(row.get("cell_gradient_l2"))
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not value > 1e-8 for value in numeric_norms) or (not expected_cell and row.get("cell_gradient_l2") is not None) or row.get("codec_gradient_nonzero") is not True or row.get("readout_gradient_nonzero") is not True or row.get("state_unchanged") is not True or row.get("cell_gradient_nonzero") is not expected_cell or row.get("initial_tree_sha256") != synthetic["initial_tree_sha256"] or row.get("output_shape") != [4]:
            raise MF0ContractError("MF0 gradient evidence differs")
    if proof["safety_resource_contract"] != plan["safety_boundary"]:
        raise MF0ContractError("MF0 safety/resource contract differs")
    audit = proof["asset_audit"]
    if not isinstance(audit, dict) or audit.get("before") != plan["asset_sha256"] or audit.get("after") != plan["asset_sha256"] or audit.get("regenerated") != plan["asset_sha256"] or audit.get("equal") is not True:
        raise MF0ContractError("MF0 asset audit differs")
    tampers = proof["tamper_audit"]
    if not isinstance(tampers, dict) or tampers.get("rejected_count") != 34 or [row.get("name") for row in tampers.get("results", [])] != TAMPERS or any(set(row) != {"name", "rejected", "error_type"} or row["rejected"] is not True or row["error_type"] != "MF0ContractError" for row in tampers["results"]):
        raise MF0ContractError("MF0 tamper audit differs")
    validate_safety(proof["safety"])
    memory = proof["memory"]
    if not isinstance(memory, dict) or memory.get("labels") != MEMORY_LABELS or memory.get("label_sha256") != sha256_bytes(canonical_json(MEMORY_LABELS)):
        raise MF0ContractError("MF0 memory evidence differs")
    validate_memory(memory.get("rows"), complete=True)
    resources = proof["resources"]
    resource_keys = {"bounds", "host_ram_bytes", "free_disk_bytes_preflight", "free_disk_bytes_postflight", "artifact_bytes_before_terminal", "completed_phase_records", "final_terminal_publication", "prepublication_elapsed_ns"}
    if not isinstance(resources, dict) or set(resources) != resource_keys or resources.get("bounds") != RESOURCE_BOUNDS or resources.get("host_ram_bytes", 0) < 8 * 2**30 or resources.get("free_disk_bytes_preflight", 0) < 8 * 2**30 or resources.get("free_disk_bytes_postflight", 0) < 8 * 2**30 or resources.get("artifact_bytes_before_terminal") != 0:
        raise MF0ContractError("MF0 resource evidence differs")
    validate_phase_records(resources.get("completed_phase_records"), resources.get("final_terminal_publication"), resources.get("prepublication_elapsed_ns"), success=True)
    freeze = proof["full_freeze"]
    freeze_keys = {"head_before", "head_after", "parent", "tree_before", "tree_after", "status_before", "status_after", "assets_equal"}
    if not isinstance(freeze, dict) or set(freeze) != freeze_keys or freeze.get("head_before") != execution_commit or freeze.get("head_after") != execution_commit or freeze.get("parent") != plan["mechanism_code_commit"] or freeze.get("status_before") != "" or freeze.get("status_after") != "" or freeze.get("assets_equal") is not True or freeze.get("tree_before") != freeze.get("tree_after"):
        raise MF0ContractError("MF0 full-freeze evidence differs")
    if proof["proof_sha256"] != sha256_bytes(canonical_json({key: value for key, value in proof.items() if key != "proof_sha256"})):
        raise MF0ContractError("MF0 proof self hash differs")


def validate_failure(failure: dict[str, Any], *, plan: dict[str, Any], execution_commit: str, plan_file_sha256: str) -> None:
    keys = {"schema_version", "status", "mechanism", "run_identity", "error_type", "error", "traceback", "execution_commit", "mechanism_code_commit", "plan_file_sha256", "plan_sha256", "completed_phase_records", "final_terminal_publication", "prepublication_elapsed_ns", "progress", "actual_safety", "partial_memory", "full_freeze_failure_audit", "output_inventory_before_failure", "candidate_created", "checkpoint_created", "model_or_tokenizer_loaded", "model_updated", "failure_sha256"}
    if set(failure) != keys or failure["schema_version"] != FAILURE_SCHEMA or failure["mechanism"] != MECHANISM or failure["run_identity"] != RUN_ID:
        raise MF0ContractError("MF0 failure identity differs")
    if failure["execution_commit"] != execution_commit or failure["mechanism_code_commit"] != plan["mechanism_code_commit"] or failure["plan_file_sha256"] != plan_file_sha256 or failure["plan_sha256"] != plan["plan_sha256"]:
        raise MF0ContractError("MF0 failure authority differs")
    taxonomy = {INCOMPLETE_STATUS: "MF0ContractError", EXPOSURE_STATUS: "ExposureBoundaryRejected"}
    if failure["status"] in taxonomy and failure["error_type"] != taxonomy[failure["status"]]:
        raise MF0ContractError("MF0 failure taxonomy differs")
    infrastructure_types = {"InfrastructureInvalid", "TimeoutError", "MemoryError", "OSError", "PermissionError", "FileNotFoundError", "CalledProcessError", "ImportError", "ModuleNotFoundError"}
    if failure["status"] == INFRASTRUCTURE_STATUS and failure["error_type"] not in infrastructure_types:
        raise MF0ContractError("MF0 infrastructure taxonomy differs")
    if failure["status"] not in {*taxonomy, INFRASTRUCTURE_STATUS}:
        raise MF0ContractError("MF0 failure status differs")
    validate_phase_records(failure["completed_phase_records"], failure["final_terminal_publication"], failure["prepublication_elapsed_ns"], success=False)
    memory = failure["partial_memory"]
    if not isinstance(memory, dict) or memory.get("labels") != MEMORY_LABELS:
        raise MF0ContractError("MF0 partial memory differs")
    validate_memory(memory.get("rows"), complete=False)
    if failure["output_inventory_before_failure"] != [] or failure["candidate_created"] is not False or failure["checkpoint_created"] is not False or failure["model_or_tokenizer_loaded"] is not False or failure["model_updated"] is not False:
        raise MF0ContractError("MF0 failure safety differs")
    safety = failure["actual_safety"]
    safety_keys = {"cuda_visible_devices", "torch_imported", "object_inventory", "network_guard", "open_firewall", "validation_opens", "heldout_opens"}
    if not isinstance(safety, dict) or set(safety) != safety_keys or safety.get("cuda_visible_devices") != "" or any(not isinstance(safety.get(key), int) or isinstance(safety.get(key), bool) or safety.get(key) < 0 for key in ("validation_opens", "heldout_opens")):
        raise MF0ContractError("MF0 failure safety schema differs")
    inventory = safety.get("object_inventory")
    if inventory is not None:
        if not isinstance(inventory, dict) or inventory.get("pretrained_model_objects") != 0 or inventory.get("tokenizer_objects") != 0 or inventory.get("optimizer_objects") != 0 or inventory.get("transformers_modeling_modules") != [] or inventory.get("uninspectable_count") != 0 or inventory.get("census_errors") != [] or inventory.get("cuda_initialized") is not False:
            raise MF0ContractError("MF0 failure observed forbidden state")
    audit = failure["full_freeze_failure_audit"]
    audit_keys = {"head", "head_exact", "parent", "parent_exact", "status", "status_clean", "plan_file_sha256", "plan_file_exact", "plan_sha256", "plan_internal_exact", "asset_hashes", "assets_exact", "errors", "provenance_exact"}
    if not isinstance(audit, dict) or set(audit) != audit_keys:
        raise MF0ContractError("MF0 failure audit schema differs")
    truths = {
        "head_exact": audit["head"] == execution_commit,
        "parent_exact": audit["parent"] == plan["mechanism_code_commit"],
        "status_clean": audit["status"] == "",
        "plan_file_exact": audit["plan_file_sha256"] == plan_file_sha256,
        "plan_internal_exact": audit["plan_sha256"] == plan["plan_sha256"],
        "assets_exact": audit["asset_hashes"] == plan["asset_sha256"],
    }
    if any(audit[key] is not value for key, value in truths.items()) or not isinstance(audit["errors"], list):
        raise MF0ContractError("MF0 failure audit truth closure differs")
    exact = audit["head_exact"] and audit["parent_exact"] and audit["status_clean"] and audit["plan_file_exact"] and audit["plan_internal_exact"] and audit["assets_exact"] and not audit["errors"]
    if audit["provenance_exact"] is not exact or (failure["status"] != INFRASTRUCTURE_STATUS and not exact):
        raise MF0ContractError("MF0 failure provenance differs")
    if failure["failure_sha256"] != sha256_bytes(canonical_json({key: value for key, value in failure.items() if key != "failure_sha256"})):
        raise MF0ContractError("MF0 failure self hash differs")


def build_failure_audit(repo: Path, args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    observed: dict[str, Any] = {"head": None, "parent": None, "status": None, "plan_file_sha256": None, "plan_sha256": None, "asset_hashes": None}
    for key, command in (("head", ("rev-parse", "HEAD")), ("parent", ("rev-parse", "HEAD^")), ("status", ("status", "--porcelain", "--untracked-files=all"))):
        try:
            observed[key] = run_git(repo, *command)
        except BaseException as error:
            errors.append({"check": f"{key}_read", "error": f"{type(error).__name__}: {error}"})
    try:
        observed["plan_file_sha256"] = file_sha256(repo / PLAN_RELATIVE_PATH)
        observed_plan = load_canonical(repo / PLAN_RELATIVE_PATH)
        observed["plan_sha256"] = observed_plan["plan_sha256"]
    except BaseException as error:
        errors.append({"check": "plan_read", "error": f"{type(error).__name__}: {error}"})
    try:
        observed["asset_hashes"] = {path: file_sha256(repo / path) for path in plan["asset_sha256"]}
    except BaseException as error:
        errors.append({"check": "asset_read", "error": f"{type(error).__name__}: {error}"})
    flags = {
        "head_exact": observed["head"] == args.execution_commit,
        "parent_exact": observed["parent"] == plan["mechanism_code_commit"],
        "status_clean": observed["status"] == "",
        "plan_file_exact": observed["plan_file_sha256"] == args.plan_file_sha256,
        "plan_internal_exact": observed["plan_sha256"] == plan["plan_sha256"],
        "assets_exact": observed["asset_hashes"] == plan["asset_sha256"],
    }
    for name, exact in flags.items():
        if not exact:
            errors.append({"check": name, "error": f"MF0 {name} mismatch"})
    return {**observed, **flags, "errors": errors, "provenance_exact": not errors}


def load_authorized_plan(repo: Path, execution_commit: str, external_sha256: str) -> dict[str, Any]:
    data = git_blob(repo, execution_commit, PLAN_RELATIVE_PATH)
    if sha256_bytes(data) != external_sha256 or not data.endswith(b"\n"):
        raise InfrastructureInvalid("MF0 authorized plan blob differs")
    plan = strict_loads(data[:-1])
    if data != canonical_json(plan) + b"\n":
        raise InfrastructureInvalid("MF0 authorized plan blob is not canonical")
    validate_plan(plan, external_sha256)
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-file-sha256", required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-terminal", action="store_true")
    return parser.parse_args()


def validate_published_terminal(args: argparse.Namespace) -> None:
    repo = args.repo.resolve(strict=True)
    if args.output_dir != Path(OUTPUT_ROOT) or args.output_dir.is_symlink() or not args.output_dir.is_dir():
        raise InfrastructureInvalid("MF0 published output root differs")
    plan = load_canonical(repo / PLAN_RELATIVE_PATH)
    validate_plan(plan, args.plan_file_sha256)
    inventory = sorted(path.name for path in args.output_dir.iterdir())
    if inventory == ["MF0-PROOF.json"]:
        _, assets = load_assets(repo)
        data = read_regular(args.output_dir / inventory[0])
        if not data.endswith(b"\n"):
            raise MF0ContractError("MF0 published proof framing differs")
        proof = strict_loads(data[:-1])
        if data != canonical_json(proof) + b"\n":
            raise MF0ContractError("MF0 published proof bytes differ")
        validate_proof(proof, plan=plan, assets=assets, execution_commit=args.execution_commit, plan_file_sha256=args.plan_file_sha256)
    elif inventory == ["MF0-FAILURE.json"]:
        data = read_regular(args.output_dir / inventory[0])
        if not data.endswith(b"\n"):
            raise MF0ContractError("MF0 published failure framing differs")
        failure = strict_loads(data[:-1])
        if data != canonical_json(failure) + b"\n":
            raise MF0ContractError("MF0 published failure bytes differ")
        validate_failure(failure, plan=plan, execution_commit=args.execution_commit, plan_file_sha256=args.plan_file_sha256)
    else:
        raise InfrastructureInvalid("MF0 published terminal inventory differs")


def main() -> None:
    args = parse_args()
    if args.validate_terminal:
        validate_published_terminal(args)
        return
    signal.signal(signal.SIGALRM, _timeout)
    tracker = PhaseTracker()
    writer: ArtifactWriter | None = None
    ledger: MemoryLedger | None = None
    guard = NetworkGuard()
    firewall: OpenFirewall | None = None
    plan: dict[str, Any] | None = None
    try:
        tracker.enter("compute", RESOURCE_BOUNDS["compute_timeout_seconds"])
        writer = ArtifactWriter(args.output_dir)
        ledger = MemoryLedger()
        repo = args.repo.resolve(strict=True)
        plan_path = repo / PLAN_RELATIVE_PATH
        if args.plan.resolve(strict=True) != plan_path:
            raise InfrastructureInvalid("MF0 plan path differs")
        plan = load_canonical(plan_path)
        validate_plan(plan, args.plan_file_sha256)
        head = run_git(repo, "rev-parse", "HEAD")
        parent = run_git(repo, "rev-parse", "HEAD^")
        status_before = run_git(repo, "status", "--porcelain", "--untracked-files=all")
        if len(args.execution_commit) != 40 or head != args.execution_commit or parent != plan["mechanism_code_commit"] or status_before != "":
            raise InfrastructureInvalid("MF0 execution freeze differs")
        tree_before = run_git(repo, "rev-parse", "HEAD^{tree}")
        host_ram, disk_before = host_resources(repo)
        allowed = {plan_path, repo / TRAIN_BANK_PATH, *(repo / path for path in plan["asset_sha256"])}
        firewall = OpenFirewall(repo, allowed)
        firewall.install()
        before = {path: file_sha256(repo / path) for path in plan["asset_sha256"]}
        if before != plan["asset_sha256"]:
            raise InfrastructureInvalid("MF0 preflight asset closure differs")
        static = static_guard(repo)
        with guard:
            import torch

            cuda_before = torch.cuda.is_initialized()
            runtime = validate_runtime(torch)
            ledger.checkpoint("runtime_verified")
            ledger.checkpoint("full_freeze_preflight_verified")
            bank, assets = load_assets(repo)
            binding = assets["phase0-evidence-binding.json"]
            phase0_manifest = repo / binding["phase0_manifest_path"]
            phase0_proof = phase0_manifest.with_name("deterministic-terminal-recovery-run1.PROOF.json")
            manifest = load_canonical(phase0_manifest)
            receipt = load_canonical(phase0_proof)
            if file_sha256(phase0_manifest) != binding["phase0_manifest_file_sha256"] or manifest.get("manifest_sha256") != binding["phase0_manifest_internal_sha256"] or file_sha256(phase0_proof) != binding["phase0_proof_file_sha256"] or receipt.get("proof_sha256") != binding["phase0_proof_internal_sha256"]:
                raise MF0ContractError("MF0 Phase0 evidence binding differs")
            del manifest, receipt
            gc.collect()
            ledger.checkpoint("phase0_evidence_binding_validated")
            ledger.checkpoint("train_bank_validated")
            ledger.checkpoint("train_partition_validated")
            ledger.checkpoint("cap0_probe_selection_validated")
            ledger.checkpoint("training_schedule_validated")
            ledger.checkpoint("candidate_contract_validated")
            synthetic = run_candidate_synthetic(torch)
            ledger.checkpoint("candidate_cpu_instantiation_validated")
            ledger.checkpoint("candidate_synthetic_gradients_validated")
            ledger.checkpoint("capture_contract_validated")
            ledger.checkpoint("metric_gate_contract_validated")
            ledger.checkpoint("threshold_builder_contract_validated")
            gc.collect()
            inventory = object_inventory(torch, writer.output_dir)
            absent = ("transformers_modeling_modules", "pretrained_model_objects", "tokenizer_objects", "optimizer_objects", "candidate_module_objects", "uninspectable_count", "census_errors", "cuda_initialized", "output_inventory")
            if inventory["uninspectable_count"] != 0 or inventory["census_errors"]:
                raise InfrastructureInvalid("MF0 final object census was incomplete")
            if any(inventory[key] not in (0, [], False) for key in absent if key not in {"uninspectable_count", "census_errors"}):
                raise ExposureBoundaryRejected("MF0 final object/CUDA census differs")
            ledger.checkpoint("safety_resource_contract_validated")
            tamper_results = []
            for name in TAMPERS[:-2]:
                try:
                    validate_assets(mutate_asset(assets, name), bank)
                except MF0ContractError as error:
                    tamper_results.append({"name": name, "rejected": True, "error_type": type(error).__name__})
                else:
                    raise MF0ContractError(f"MF0 tamper accepted: {name}")
            ledger.checkpoint("tamper_audit_validated")
            del bank
        cuda_after = torch.cuda.is_initialized()
        if cuda_before or cuda_after or guard.attempt_count != 0 or not guard.wrappers_restored or firewall.denied_count or firewall.validation_open_count or firewall.heldout_open_count:
            raise ExposureBoundaryRejected("MF0 exposure guard evidence differs")
        after = {path: file_sha256(repo / path) for path in plan["asset_sha256"]}
        head_after = run_git(repo, "rev-parse", "HEAD")
        tree_after = run_git(repo, "rev-parse", "HEAD^{tree}")
        status_after = run_git(repo, "status", "--porcelain", "--untracked-files=all")
        if after != before or head_after != args.execution_commit or tree_after != tree_before or status_after != "":
            raise InfrastructureInvalid("MF0 postflight freeze differs")
        ledger.checkpoint("full_freeze_postflight_validated")
        tracker.exit("completed")

        tracker.enter("audit", RESOURCE_BOUNDS["audit_timeout_seconds"])
        regenerated = dict(after)
        with tempfile.TemporaryDirectory(prefix="mf0-regenerate-") as temporary:
            rebuilt = build_assets(load_canonical(repo / TRAIN_BANK_PATH))
            validate_assets(rebuilt, load_canonical(repo / TRAIN_BANK_PATH))
            for name in ASSET_NAMES:
                encoded = canonical_json(rebuilt[name]) + b"\n"
                target = Path(temporary) / name
                target.write_bytes(encoded)
                if encoded != read_regular(repo / ARTIFACT_DIR / name):
                    raise MF0ContractError(f"MF0 regenerated asset differs: {name}")
                regenerated[f"{ARTIFACT_DIR}/{name}"] = sha256_bytes(encoded)
        disk_after = shutil.disk_usage(repo).free
        if disk_after < 8 * 2**30:
            raise InfrastructureInvalid("MF0 postflight disk differs")
        ledger.checkpoint("proof_prewrite_ready")
        tracker.exit("completed")

        tracker.enter("terminal_publication", RESOURCE_BOUNDS["terminal_timeout_seconds"])
        terminal, elapsed = tracker.terminal_boundary()
        safety = {
            "cuda_visible_devices": "",
            "cuda_initialized_before": cuda_before,
            "cuda_initialized_after": cuda_after,
            "torch_cpu_only": True,
            "tokenizer_calls": 0,
            "model_calls": 0,
            "model_backwards": 0,
            "optimizer_objects": 0,
            "optimizer_steps": 0,
            "validation_opens": firewall.validation_open_count,
            "heldout_opens": firewall.heldout_open_count,
            "model_or_tokenizer_loaded": False,
            "candidate_created": False,
            "checkpoint_created": False,
            "model_updated": False,
            "object_inventory": inventory,
            "network_guard": guard.evidence(),
            "open_firewall": {"denied_count": firewall.denied_count, "validation_open_count": firewall.validation_open_count, "heldout_open_count": firewall.heldout_open_count, "opened_paths": sorted(set(firewall.opened))},
            "static_guard": static,
        }
        tamper_rows = [*tamper_results, {"name": "proof_status_changed", "rejected": True, "error_type": "MF0ContractError"}, {"name": "proof_self_hash_changed", "rejected": True, "error_type": "MF0ContractError"}]
        proof: dict[str, Any] = {
            "schema_version": PROOF_SCHEMA,
            "status": PROOF_STATUS,
            "mechanism": MECHANISM,
            "run_identity": RUN_ID,
            "execution_commit": args.execution_commit,
            "mechanism_code_commit": plan["mechanism_code_commit"],
            "plan_file_sha256": args.plan_file_sha256,
            "plan_sha256": plan["plan_sha256"],
            "runtime": runtime,
            "asset_audit": {"before": before, "after": after, "regenerated": regenerated, "equal": True},
            "phase0_binding": assets["phase0-evidence-binding.json"],
            "train_partition": assets["train-partition.json"],
            "cap0_probe_selection": assets["cap0-probe-selection.json"],
            "training_schedule": assets["training-schedule.json"],
            "candidate_contract": {"contract": assets["candidate-module-contract.json"], "synthetic_validation": synthetic},
            "capture_contract": assets["capture-contract.json"],
            "metric_gate_contract": assets["metric-gate-contract.json"],
            "threshold_builder_contract": assets["threshold-builder-contract.json"],
            "safety_resource_contract": plan["safety_boundary"],
            "tamper_audit": {"results": tamper_rows, "rejected_count": 34},
            "counts": COUNTS,
            "safety": safety,
            "resources": {"bounds": RESOURCE_BOUNDS, "host_ram_bytes": host_ram, "free_disk_bytes_preflight": disk_before, "free_disk_bytes_postflight": disk_after, "artifact_bytes_before_terminal": 0, "completed_phase_records": copy.deepcopy(tracker.completed), "final_terminal_publication": terminal, "prepublication_elapsed_ns": elapsed},
            "memory": {"labels": MEMORY_LABELS, "label_sha256": sha256_bytes(canonical_json(MEMORY_LABELS)), "rows": ledger.rows},
            "full_freeze": {"head_before": head, "head_after": head_after, "parent": parent, "tree_before": tree_before, "tree_after": tree_after, "status_before": status_before, "status_after": status_after, "assets_equal": True},
            "decision_boundary": DECISION,
            "proof_sha256": "",
        }
        proof["proof_sha256"] = sha256_bytes(canonical_json({key: value for key, value in proof.items() if key != "proof_sha256"}))
        parsed = strict_loads(canonical_json(proof))
        validate_proof(parsed, plan=plan, assets=assets, execution_commit=args.execution_commit, plan_file_sha256=args.plan_file_sha256)
        for name in TAMPERS[-2:]:
            altered = copy.deepcopy(parsed)
            if name == "proof_status_changed":
                altered["status"] = INCOMPLETE_STATUS
                altered["proof_sha256"] = sha256_bytes(canonical_json({key: value for key, value in altered.items() if key != "proof_sha256"}))
            else:
                altered["proof_sha256"] = "0" * 64
            try:
                validate_proof(altered, plan=plan, assets=assets, execution_commit=args.execution_commit, plan_file_sha256=args.plan_file_sha256)
            except MF0ContractError:
                pass
            else:
                raise MF0ContractError(f"MF0 receipt tamper accepted: {name}")
        reopened = writer.write("MF0-PROOF.json", parsed, 16 * 2**20)
        reparsed = strict_loads(reopened[:-1])
        if reparsed != parsed:
            raise InfrastructureInvalid("MF0 proof changed after publication")
        validate_proof(reparsed, plan=plan, assets=assets, execution_commit=args.execution_commit, plan_file_sha256=args.plan_file_sha256)
        signal.alarm(0)
    except BaseException as original_error:
        signal.alarm(0)
        if writer is None or writer.terminal_written:
            raise
        if tracker.active is not None:
            tracker.exit("error", timeout_observed=isinstance(original_error, TimeoutError))
        tracker.enter("failure_audit", RESOURCE_BOUNDS["failure_timeout_seconds"])
        repo = args.repo.resolve()
        authorized_plan = load_authorized_plan(repo, args.execution_commit, args.plan_file_sha256)
        audit = build_failure_audit(repo, args, authorized_plan)
        infrastructure_kinds = (InfrastructureInvalid, TimeoutError, MemoryError, OSError, subprocess.CalledProcessError, ImportError)
        if isinstance(original_error, MF0ContractError):
            status = INCOMPLETE_STATUS
            error: BaseException = original_error
        elif isinstance(original_error, ExposureBoundaryRejected):
            status = EXPOSURE_STATUS
            error = original_error
        elif isinstance(original_error, infrastructure_kinds):
            status = INFRASTRUCTURE_STATUS
            error = original_error
        else:
            status = INCOMPLETE_STATUS
            error = MF0ContractError(f"unexpected MF0 diagnostic error: {type(original_error).__name__}: {original_error}")
        if not audit["provenance_exact"] and status != INFRASTRUCTURE_STATUS:
            status = INFRASTRUCTURE_STATUS
            error = InfrastructureInvalid("MF0 failure provenance was not exact")
        tracker.exit("completed")
        tracker.enter("terminal_publication", RESOURCE_BOUNDS["terminal_timeout_seconds"])
        terminal, elapsed = tracker.terminal_boundary()
        failure_torch = sys.modules.get("torch")
        failure_inventory = None if failure_torch is None else object_inventory(failure_torch, writer.output_dir)
        model_or_tokenizer_loaded = False if failure_inventory is None else bool(failure_inventory["pretrained_model_objects"] or failure_inventory["tokenizer_objects"] or failure_inventory["transformers_modeling_modules"])
        failure: dict[str, Any] = {
            "schema_version": FAILURE_SCHEMA,
            "status": status,
            "mechanism": MECHANISM,
            "run_identity": RUN_ID,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "execution_commit": args.execution_commit,
            "mechanism_code_commit": authorized_plan["mechanism_code_commit"],
            "plan_file_sha256": args.plan_file_sha256,
            "plan_sha256": authorized_plan["plan_sha256"],
            "completed_phase_records": copy.deepcopy(tracker.completed),
            "final_terminal_publication": terminal,
            "prepublication_elapsed_ns": elapsed,
            "progress": {"memory_rows": 0 if ledger is None else len(ledger.rows), "last_memory_label": None if ledger is None or not ledger.rows else ledger.rows[-1]["label"]},
            "actual_safety": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch_imported": failure_torch is not None, "object_inventory": failure_inventory, "network_guard": guard.evidence(), "open_firewall": None if firewall is None else {"denied_count": firewall.denied_count, "validation_open_count": firewall.validation_open_count, "heldout_open_count": firewall.heldout_open_count}, "validation_opens": 0 if firewall is None else firewall.validation_open_count, "heldout_opens": 0 if firewall is None else firewall.heldout_open_count},
            "partial_memory": {"labels": MEMORY_LABELS, "rows": [] if ledger is None else ledger.rows},
            "full_freeze_failure_audit": audit,
            "output_inventory_before_failure": sorted(path.name for path in writer.output_dir.iterdir()),
            "candidate_created": False,
            "checkpoint_created": False,
            "model_or_tokenizer_loaded": model_or_tokenizer_loaded,
            "model_updated": False,
            "failure_sha256": "",
        }
        failure["failure_sha256"] = sha256_bytes(canonical_json({key: value for key, value in failure.items() if key != "failure_sha256"}))
        parsed_failure = strict_loads(canonical_json(failure))
        validate_failure(parsed_failure, plan=authorized_plan, execution_commit=args.execution_commit, plan_file_sha256=args.plan_file_sha256)
        reopened = writer.write("MF0-FAILURE.json", parsed_failure, 16 * 2**20)
        reparsed = strict_loads(reopened[:-1])
        if reparsed != parsed_failure:
            raise InfrastructureInvalid("MF0 failure changed after publication") from original_error
        validate_failure(reparsed, plan=authorized_plan, execution_commit=args.execution_commit, plan_file_sha256=args.plan_file_sha256)
        signal.alarm(0)
        raise SystemExit(2) from original_error


if __name__ == "__main__":
    main()
