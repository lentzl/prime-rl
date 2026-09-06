#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import gc
import hashlib
import importlib.metadata
import os
import platform
import resource
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from unittest import mock

from prime_rl.latent.h_iter_phase1_cap0 import (
    ARTIFACT_DIR,
    CACHE_LABELS,
    COUNTS,
    DECISION,
    E33_PATH,
    E33_STATE_SHA256,
    E33_TREE_SHA256,
    EXPOSURE_STATUS,
    FAILURE_SCHEMA,
    H176_PATH,
    H176_TREE_SHA256,
    INCOMPLETE_STATUS,
    MECHANISM,
    MECHANISM_CAUSES,
    MEMORY_LABELS,
    METADATA_SHA256,
    MF0_BINDING,
    OUTPUT_ROOT,
    PLAN_SCHEMA,
    PREFLIGHT_SCHEMA,
    PREFLIGHT_STATUS,
    PROOF_SCHEMA,
    PROOF_STATUS,
    REJECT_STATUS,
    REPO_ROOT,
    RESOURCE_BOUNDS,
    RUN_ID,
    RUNTIME,
    SELECTION_SHA256,
    SHARED_PROJECT,
    TAMPERS,
    CAP0ContractError,
    canonical_json,
    sha256_bytes,
    strict_loads,
    validate_contract,
    validate_memory,
)

PLAN_RELATIVE_PATH = f"{ARTIFACT_DIR}/cap0-plan.json"
CONTRACT_RELATIVE_PATH = f"{ARTIFACT_DIR}/cap0-contract.json"
MF0_DIR = "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-train-calibration-v1"
PHASE0_DIR = "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1"
PLAN_ASSET_PATHS = [
    f"{MF0_DIR}/mf0-prereg-run1-evidence-manifest.json",
    f"{MF0_DIR}/mf0-prereg-run1.MF0-PROOF.json",
    f"{MF0_DIR}/mf0-prereg-run1.launcher.log",
    f"{MF0_DIR}/mf0-prereg-run1.exit.txt",
    f"{MF0_DIR}/mf0-plan.json",
    f"{MF0_DIR}/mf0-plan.sha256",
    f"{PHASE0_DIR}/train-bank.json",
    f"{MF0_DIR}/train-partition.json",
    f"{MF0_DIR}/cap0-probe-selection.json",
    f"{MF0_DIR}/capture-contract.json",
    CONTRACT_RELATIVE_PATH,
    "pyproject.toml",
    "uv.lock",
    "src/prime_rl/latent/h_iter_phase1_cap0.py",
    "scripts/latent/run_h_iter_phase1_cap0_v1.py",
    "scripts/latent/run_h_iter_phase1_cap0_v1.sh",
    "scripts/latent/freeze_h_iter_phase1_cap0_plan_v1.py",
    "tests/unit/latent/test_h_iter_phase1_cap0.py",
]
EXPERIMENT_READ_PATHS = PLAN_ASSET_PATHS[:11]
AUTHORIZATION = {"cap0_capture_only": True, "model": True, "gpu": True, "t0": False, "training": False, "optimizer": False, "backward": False, "update": False, "validation": False, "heldout": False}
REMOTE_PATHS = {"repo": REPO_ROOT, "shared_project": SHARED_PROJECT, "shared_python": RUNTIME["sys_executable"], "e33": E33_PATH, "h176": H176_PATH, "physical_gpu": "0", "visible_device": "cuda:0"}
TERMINAL_CONTRACT = {"success_file": "CAP0-PROOF.json", "failure_file": "CAP0-FAILURE.json", "exclusive_atomic": True, "canonical_roundtrip_twice": True, "success_status": PROOF_STATUS, "failure_statuses": [REJECT_STATUS, INCOMPLETE_STATUS, EXPOSURE_STATUS, "infrastructure_invalid"]}
SAFETY_BOUNDARY = {"train_only": True, "selected_train_rows": 4, "validation_opens": 0, "heldout_opens": 0, "h176_loaded": False, "generation": False, "backward": False, "optimizer": False, "candidate": False, "checkpoint": False, "update": False, "network_attempts": 0, "t0_authorized": False}
FULL_FREEZE = {"execution_parent_is_mechanism": True, "clean_before_after": True, "tree_unchanged": True, "assets_unchanged": True, "protected_unchanged": True}
PROBE_KEYS = {"probe_index", "depth", "action_index", "replicate", "row_id", "row_sha256", "receiver_input_sha256", "node_count", "local_text_byte_lengths", "unpadded_token_lengths", "input_ids_shape", "attention_mask_shape", "input_ids_sha256", "attention_mask_sha256", "repeats", "repeat_input_ids_bitwise", "repeat_attention_mask_bitwise", "repeat_full_hidden_bitwise", "repeat_capture_bitwise", "all_outputs_finite", "capture_row_sha256", "unique_capture_row_count", "not_all_node_identical", "complete", "qualifies"}
REPEAT_KEYS = {"case_index", "repeat", "model_call_index", "input_ids_same_object", "attention_mask_same_object", "input_ids_sha256", "attention_mask_sha256", "logits_shape", "logits_dtype", "logits_finite", "logits_sha256", "full_hidden_shape", "full_hidden_dtype", "full_hidden_finite", "full_hidden_sha256", "capture_shape", "capture_dtype", "capture_finite", "capture_sha256", "pkv_is_none"}
PROOF_KEYS = {"schema_version", "status", "mechanism", "run_identity", "execution_commit", "mechanism_code_commit", "plan_file_sha256", "plan_sha256", "runtime", "asset_audit", "mf0_archive_binding", "selection", "model_identity", "probes", "aggregate", "cache_guard", "protected_state", "safety", "counts", "resources", "memory", "full_freeze", "tamper_audit", "decision_boundary", "proof_sha256"}
FAILURE_KEYS = {"schema_version", "status", "mechanism", "run_identity", "error_type", "error", "traceback", "execution_commit", "mechanism_code_commit", "plan_file_sha256", "plan_sha256", "progress", "partial_probes", "aggregate_partial", "cache_guard_partial", "protected_state", "actual_safety", "resources", "partial_memory", "full_freeze_failure_audit", "output_inventory_before_failure", "candidate_files", "checkpoint_files", "model_updated", "failure_sha256"}


class CAP0MechanismRejected(RuntimeError):
    def __init__(self, message: str, cause: str) -> None:
        super().__init__(message)
        if cause not in MECHANISM_CAUSES:
            raise ValueError("unknown CAP0 mechanism cause")
        self.cause = cause


class ExposureBoundaryRejected(RuntimeError):
    pass


class InfrastructureInvalid(RuntimeError):
    pass


class CacheAllocationDetected(CAP0MechanismRejected):
    def __init__(self, message: str) -> None:
        super().__init__(message, "cache_allocation_detected")


def file_sha256(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    digest = hashlib.sha256()
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise InfrastructureInvalid(f"not a regular file: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def read_regular(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise InfrastructureInvalid(f"not a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_canonical(path: Path) -> dict[str, Any]:
    data = read_regular(path)
    if not data.endswith(b"\n"):
        raise CAP0ContractError(f"CAP0 canonical file framing differs: {path}")
    value = strict_loads(data[:-1])
    if not isinstance(value, dict) or data != canonical_json(value) + b"\n":
        raise CAP0ContractError(f"CAP0 canonical file differs: {path}")
    return value


def run_git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=repo, text=True, stderr=subprocess.STDOUT).strip()


def git_blob(repo: Path, commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=repo, stderr=subprocess.STDOUT)


def validate_plan(plan: dict[str, Any], external_sha256: str | None = None) -> None:
    keys = {"schema_version", "status", "mechanism", "run_identity", "mechanism_code_commit", "execution_authorization", "output_root", "remote_paths", "runtime", "asset_sha256", "mf0_archive_binding", "probe_contract", "model_contract", "cache_contract", "resource_bounds", "memory_label_schedule", "terminal_contract", "safety_boundary", "full_freeze", "plan_sha256"}
    if set(plan) != keys or plan["schema_version"] != PLAN_SCHEMA or plan["status"] != "preregistered" or plan["mechanism"] != MECHANISM or plan["run_identity"] != RUN_ID:
        raise CAP0ContractError("CAP0 plan identity differs")
    if not isinstance(plan["mechanism_code_commit"], str) or len(plan["mechanism_code_commit"]) != 40:
        raise CAP0ContractError("CAP0 mechanism commit differs")
    if plan["execution_authorization"] != AUTHORIZATION or plan["output_root"] != OUTPUT_ROOT or plan["remote_paths"] != REMOTE_PATHS or plan["runtime"] != RUNTIME:
        raise CAP0ContractError("CAP0 plan authority differs")
    if not isinstance(plan["asset_sha256"], dict) or set(plan["asset_sha256"]) != set(PLAN_ASSET_PATHS) or any(not isinstance(value, str) or len(value) != 64 for value in plan["asset_sha256"].values()):
        raise CAP0ContractError("CAP0 plan asset map differs")
    expected = {
        "mf0_archive_binding": MF0_BINDING,
        "probe_contract": {"selection_sha256": SELECTION_SHA256, "probe_count": 4, "repeat_count": 2, "model_forwards": 8, "tokenizer_calls": 4, "sequences": 192},
        "model_contract": {"e33_path": E33_PATH, "e33_tree_sha256": E33_TREE_SHA256, "e33_state_sha256": E33_STATE_SHA256, "h176_path": H176_PATH, "h176_tree_sha256": H176_TREE_SHA256, "metadata_sha256": METADATA_SHA256},
        "cache_contract": {"checks": 18, "mandatory_negative_trips": 1, "actual_allocations": 0, "pkv_none": True, "config_restored": True},
        "resource_bounds": RESOURCE_BOUNDS,
        "memory_label_schedule": {"labels": MEMORY_LABELS, "count": 28, "label_sha256": sha256_bytes(canonical_json(MEMORY_LABELS))},
        "terminal_contract": TERMINAL_CONTRACT,
        "safety_boundary": SAFETY_BOUNDARY,
        "full_freeze": FULL_FREEZE,
    }
    for key, value in expected.items():
        if plan[key] != value:
            raise CAP0ContractError(f"CAP0 plan {key} differs")
    internal = sha256_bytes(canonical_json({key: value for key, value in plan.items() if key != "plan_sha256"}))
    if plan["plan_sha256"] != internal:
        raise CAP0ContractError("CAP0 plan self hash differs")
    if external_sha256 is not None and external_sha256 != sha256_bytes(canonical_json(plan) + b"\n"):
        raise InfrastructureInvalid("CAP0 external plan hash differs")


def validate_archive(repo: Path) -> dict[str, Any]:
    manifest = load_canonical(repo / PLAN_ASSET_PATHS[0])
    proof = load_canonical(repo / PLAN_ASSET_PATHS[1])
    if file_sha256(repo / PLAN_ASSET_PATHS[0]) != MF0_BINDING["manifest_file_sha256"] or manifest.get("manifest_sha256") != MF0_BINDING["manifest_internal_sha256"]:
        raise InfrastructureInvalid("CAP0 MF0 archive manifest differs")
    if file_sha256(repo / PLAN_ASSET_PATHS[1]) != MF0_BINDING["proof_file_sha256"] or proof.get("proof_sha256") != MF0_BINDING["proof_internal_sha256"] or proof.get("status") != MF0_BINDING["proof_status"]:
        raise InfrastructureInvalid("CAP0 MF0 proof differs")
    if file_sha256(repo / PLAN_ASSET_PATHS[2]) != MF0_BINDING["launcher_log_sha256"] or file_sha256(repo / PLAN_ASSET_PATHS[3]) != MF0_BINDING["exit_file_sha256"]:
        raise InfrastructureInvalid("CAP0 MF0 launch evidence differs")
    if manifest.get("claim_boundary") != proof.get("decision_boundary") or proof["decision_boundary"].get("cap0_authorized") is not False:
        raise CAP0ContractError("CAP0 MF0 claim boundary differs")
    return MF0_BINDING


def load_train_inputs(repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    bank = load_canonical(repo / PLAN_ASSET_PATHS[6])
    partition = load_canonical(repo / PLAN_ASSET_PATHS[7])
    selection = load_canonical(repo / PLAN_ASSET_PATHS[8])
    capture = load_canonical(repo / PLAN_ASSET_PATHS[9])
    if bank.get("split") != "train" or len(bank.get("rows", [])) != 96 or selection.get("selection_sha256") != SELECTION_SHA256 or len(selection.get("ordered_probes", [])) != 4:
        raise CAP0ContractError("CAP0 TRAIN bank or selection differs")
    validate_selected_train_rows(bank, partition, selection)
    contract = load_canonical(repo / PLAN_ASSET_PATHS[10])
    validate_contract(contract, selection, capture)
    return bank, partition, selection, capture


def validate_selected_train_rows(bank: dict[str, Any], partition: dict[str, Any], selection: dict[str, Any]) -> None:
    fit_ids = {row["row_id"] for row in partition.get("fit_rows", [])}
    bank_by_id = {row["row_id"]: row for row in bank.get("rows", [])}
    if len(bank_by_id) != 96:
        raise CAP0ContractError("CAP0 TRAIN row identities differ")
    for probe in selection.get("ordered_probes", []):
        row = bank_by_id.get(probe.get("row_id"))
        if probe.get("row_id") not in fit_ids or row is None or row.get("row_sha256") != probe.get("row_sha256") or row.get("receiver_input_sha256") != probe.get("receiver_input_sha256"):
            raise CAP0ContractError("CAP0 selected TRAIN probe differs")
        if row["row_sha256"] != sha256_bytes(canonical_json({key: value for key, value in row.items() if key != "row_sha256"})) or row["receiver_input_sha256"] != sha256_bytes(canonical_json(row["receiver_input"])):
            raise CAP0ContractError("CAP0 selected TRAIN row self-hash differs")
        receiver = row["receiver_input"]
        if len(receiver.get("nodes", [])) != 24 or len(receiver.get("edges", [])) != 24 or any(set(node) != {"node_id", "local_text"} or len(node["local_text"].encode("utf-8")) != 68 for node in receiver["nodes"]):
            raise CAP0ContractError("CAP0 selected TRAIN receiver differs")
        encoded = canonical_json(receiver)
        if any(token in encoded for token in (b"answer_action", b"supervision", b"target_marker", b"target_node_id")):
            raise CAP0ContractError("CAP0 selected TRAIN receiver leaked supervision")


def static_guard(repo: Path) -> dict[str, Any]:
    paths = ["src/prime_rl/latent/h_iter_phase1_cap0.py", "scripts/latent/run_h_iter_phase1_cap0_v1.py"]
    def scan(source: str, path: str) -> list[str]:
        forbidden: list[str] = []
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                if name in {"generate", "backward", "step", "save", "save_file", "save_pretrained"}:
                    forbidden.append(f"{path}:{node.lineno}:{name}")
            if isinstance(node, (ast.Name, ast.Attribute)):
                name = node.id if isinstance(node, ast.Name) else node.attr
                if name in {"AdamW", "Optimizer", "WorkspaceBridge", "CandidateModule"}:
                    forbidden.append(f"{path}:{node.lineno}:{name}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lowered = node.value.lower()
                if ("validation-bank" in lowered or "heldout-bank" in lowered or "held_out" in lowered) and lowered not in {"validation-bank", "heldout-bank", "held_out"} and "forbidden" not in lowered:
                    forbidden.append(f"{path}:{node.lineno}:split_path_literal")
        return forbidden

    forbidden: list[str] = []
    for path in paths:
        forbidden.extend(scan((repo / path).read_text(), path))
    if forbidden:
        raise CAP0ContractError("CAP0 static exposure guard rejected")
    fixtures = {
        "generation_call": "model.generate()",
        "backward_call": "loss.backward()",
        "optimizer_step": "optimizer.step()",
        "candidate_constructor": "CandidateModule()",
        "validation_path": "path = 'x/" + "validation" + "-bank.json'",
    }
    rejected = {name: bool(scan(source, f"fixture:{name}")) for name, source in fixtures.items()}
    if not all(rejected.values()):
        raise CAP0ContractError("CAP0 static exposure negative fixture accepted")
    return {"paths": paths, "forbidden_sites": [], "negative_fixtures": rejected, "runner_sha256": file_sha256(repo / paths[1]), "module_sha256": file_sha256(repo / paths[0])}


class NetworkGuard:
    def __init__(self) -> None:
        self.attempts = 0
        self.installed = False
        self.restored = False
        self.originals: dict[tuple[Any, str], Any] = {}

    def _deny(self, *_args: Any, **_kwargs: Any) -> Any:
        self.attempts += 1
        raise ExposureBoundaryRejected("CAP0 network attempt blocked")

    def _audit(self, event: str, _args: tuple[Any, ...]) -> None:
        if event in {"socket.connect", "socket.getaddrinfo"}:
            self.attempts += 1
            raise ExposureBoundaryRejected("CAP0 audited network attempt blocked")

    def __enter__(self) -> NetworkGuard:
        for owner, name in ((socket.socket, "connect"), (socket.socket, "connect_ex"), (socket, "create_connection"), (socket, "getaddrinfo")):
            self.originals[(owner, name)] = getattr(owner, name)
            setattr(owner, name, self._deny)
        sys.addaudithook(self._audit)
        self.installed = True
        return self

    def __exit__(self, *_args: Any) -> None:
        for (owner, name), value in self.originals.items():
            setattr(owner, name, value)
        self.restored = True

    def evidence(self) -> dict[str, Any]:
        return {"installed": self.installed, "wrappers_restored": self.restored, "audit_hook_persistent": self.installed, "attempt_count": self.attempts, "operations": ["socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection", "socket.getaddrinfo"], "audit_events": ["socket.connect", "socket.getaddrinfo"]}


class OpenFirewall:
    def __init__(self, repo: Path) -> None:
        self.repo = repo.resolve()
        self.allowed = {(repo / path).resolve() for path in EXPERIMENT_READ_PATHS}
        self.denied_count = 0
        self.validation_open_count = 0
        self.heldout_open_count = 0
        self.opened: set[str] = set()

    def audit(self, event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args or not isinstance(args[0], (str, bytes)):
            return
        candidate = Path(os.fsdecode(args[0]))
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        experiment_root = (self.repo / "experiments").resolve()
        if not candidate.is_relative_to(experiment_root):
            return
        relative = candidate.relative_to(self.repo).as_posix()
        if candidate not in self.allowed:
            self.denied_count += 1
            lowered = relative.lower()
            if "validation" in lowered:
                self.validation_open_count += 1
            if "heldout" in lowered or "held_out" in lowered:
                self.heldout_open_count += 1
            raise ExposureBoundaryRejected(f"CAP0 blocked experiment path open: {relative}")
        self.opened.add(relative)

    def install(self) -> None:
        sys.addaudithook(self.audit)

    def evidence(self) -> dict[str, Any]:
        return {"denied_count": self.denied_count, "validation_open_count": self.validation_open_count, "heldout_open_count": self.heldout_open_count, "opened_paths": sorted(self.opened)}


class ArtifactWriter:
    def __init__(self, output: Path) -> None:
        if output != Path(OUTPUT_ROOT) or not output.is_absolute() or output.is_symlink() or output.exists():
            raise InfrastructureInvalid("CAP0 output namespace differs")
        output.mkdir(mode=0o700)
        self.output = output
        self.terminal_written = False

    def write(self, name: str, payload: dict[str, Any]) -> bytes:
        if self.terminal_written or name not in {"CAP0-PROOF.json", "CAP0-FAILURE.json"} or list(self.output.iterdir()):
            raise InfrastructureInvalid("CAP0 terminal exclusivity differs")
        encoded = canonical_json(payload) + b"\n"
        if len(encoded) > RESOURCE_BOUNDS["maximum_terminal_bytes"]:
            raise InfrastructureInvalid("CAP0 terminal exceeds bound")
        temporary = self.output / f".{name}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short CAP0 terminal write")
                view = view[written:]
            os.fsync(descriptor)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
        finally:
            os.close(descriptor)
        os.replace(temporary, self.output / name)
        self.terminal_written = True
        directory = os.open(self.output, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        observed = read_regular(self.output / name)
        if observed != encoded or strict_loads(observed[:-1]) != payload:
            raise InfrastructureInvalid("CAP0 terminal reopen differs")
        return observed


class PhaseTracker:
    def __init__(self) -> None:
        self.start = time.monotonic_ns()
        self.active: tuple[str, int, int] | None = None
        self.completed: list[dict[str, Any]] = []

    def enter(self, phase: str, seconds: int) -> None:
        if self.active is not None:
            raise InfrastructureInvalid("CAP0 phase overlap")
        self.active = (phase, time.monotonic_ns() - self.start, seconds)
        signal.alarm(seconds - 1)

    def exit(self, outcome: str) -> None:
        if self.active is None:
            raise InfrastructureInvalid("CAP0 phase absent")
        phase, entered, seconds = self.active
        exited = time.monotonic_ns() - self.start
        signal.alarm(0)
        self.completed.append({"phase": phase, "entered_ns_since_start": entered, "exited_ns_since_start": exited, "duration_ns": exited - entered, "outcome": outcome, "cap_ns": seconds * 10**9, "alarm_after_ns": (seconds - 1) * 10**9, "alarm_safety_margin_ns": 10**9})
        self.active = None

    def terminal(self) -> tuple[dict[str, Any], int]:
        if self.active is None or self.active[0] != "terminal_publication":
            raise InfrastructureInvalid("CAP0 terminal phase absent")
        entered = self.active[1]
        return {"phase": "terminal_publication", "entered_ns_since_start": entered, "limit_ns": 60 * 10**9, "completion_observable_inside_terminal": False, "self_reference_boundary": "post_write_fsync_reopen_validation_and_process_exit_are_external_to_immutable_terminal_bytes"}, entered


class MemoryLedger:
    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.rows: list[dict[str, Any]] = []

    def checkpoint(self, label: str) -> None:
        if label != MEMORY_LABELS[len(self.rows)]:
            raise CAP0ContractError("CAP0 memory label order differs")
        self.torch.cuda.synchronize(0)
        try:
            current_rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            current_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        row = {"label": label, "rss_bytes": current_rss, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, "allocated_bytes": self.torch.cuda.memory_allocated(0), "reserved_bytes": self.torch.cuda.memory_reserved(0), "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(0), "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(0)}
        self.rows.append(row)
        validate_memory(self.rows, complete=False)


def tensor_sha256(torch: Any, tensor: Any) -> str:
    header = canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)}) + b"\n"
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return sha256_bytes(header + raw)


def module_tree_sha256(torch: Any, model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(canonical_json({"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape), "sha256": tensor_sha256(torch, tensor)}))
        digest.update(b"\n")
    return digest.hexdigest()


def metadata_hashes(path: Path) -> dict[str, str]:
    return {name: file_sha256(path / name) for name in METADATA_SHA256}


def protected_disk() -> dict[str, Any]:
    e33, h176 = Path(E33_PATH), Path(H176_PATH)
    return {"e33_tree_sha256": file_sha256(e33 / "model.safetensors"), "h176_tree_sha256": file_sha256(h176 / "model.safetensors"), "e33_metadata_sha256": metadata_hashes(e33), "h176_metadata_sha256": metadata_hashes(h176)}


def class_closure(base: type) -> set[type]:
    result = {base}
    stack = [base]
    while stack:
        for child in stack.pop().__subclasses__():
            if child not in result:
                result.add(child)
                stack.append(child)
    return result


def class_identity(cls: type) -> dict[str, str]:
    module = __import__(cls.__module__, fromlist=["__name__"])
    path = Path(module.__file__).resolve(strict=True)
    distribution = "flash-linear-attention" if cls.__module__.split(".", 1)[0] == "fla" else "transformers"
    return {"fqcn": f"{cls.__module__}.{cls.__qualname__}", "module_path": str(path), "module_sha256": file_sha256(path), "distribution": f"{distribution}=={importlib.metadata.version(distribution)}"}


class CacheGuard:
    def __init__(self, model: Any, cache_utils: Any, expected: list[dict[str, str]]) -> None:
        self.model, self.cache_utils, self.expected = model, cache_utils, expected
        self.initial = class_closure(cache_utils.Cache)
        self.stack = contextlib.ExitStack()
        self.patched: set[type] = set()
        self.labels: list[str] = []
        self.trip_count = 0
        self.negative = False
        self.actual_allocations = 0
        self.pkv_non_none_count = 0
        self.config_drift_count = 0
        self.restored = False
        sources: list[tuple[str, Any]] = [("model.config", model.config)]
        if getattr(model, "generation_config", None) is not None:
            sources.append(("model.generation_config", model.generation_config))
        for name, module in model.named_modules():
            if getattr(module, "config", None) is not None:
                sources.append((f"module:{name}.config", module.config))
        seen: set[int] = set()
        self.configs: list[tuple[str, Any, Any]] = []
        for source, config in sorted(sources, key=lambda pair: pair[0]):
            if id(config) not in seen and hasattr(config, "use_cache"):
                seen.add(id(config))
                self.configs.append((source, config, getattr(config, "use_cache", None)))

    def _reject(self, cls: type, *_args: Any, **_kwargs: Any) -> Any:
        self.trip_count += 1
        raise CacheAllocationDetected(f"CAP0 cache allocation attempted: {cls.__module__}.{cls.__qualname__}")

    def check(self, label: str) -> None:
        if label != CACHE_LABELS[len(self.labels)]:
            raise CAP0ContractError("CAP0 cache check order differs")
        if class_closure(self.cache_utils.Cache) - self.patched:
            raise CAP0ContractError("CAP0 unpatched cache subclass appeared")
        if any(getattr(config, "use_cache", None) is not False for _, config, _ in self.configs):
            self.config_drift_count += 1
            raise CAP0MechanismRejected("CAP0 use_cache configuration changed", "cache_configuration_drift")
        self.labels.append(label)

    def __enter__(self) -> CacheGuard:
        for cls in sorted(self.initial, key=lambda value: (value.__module__, value.__qualname__)):
            self.stack.enter_context(mock.patch.object(cls, "__new__", self._reject))
            self.patched.add(cls)
        for _, config, _ in self.configs:
            config.use_cache = False
        self.check("CACHE_ENTRY")
        try:
            self.cache_utils.DynamicCache()
        except CacheAllocationDetected:
            self.negative = True
        if not self.negative or self.trip_count != 1:
            raise CAP0ContractError("CAP0 DynamicCache negative control differs")
        return self

    def __exit__(self, *_args: Any) -> None:
        try:
            if len(self.labels) == 17:
                self.check("CACHE_EXIT")
        finally:
            self.stack.close()
            for _, config, before in self.configs:
                config.use_cache = before
            self.restored = all(getattr(config, "use_cache", None) == before for _, config, before in self.configs)
            self.model = None

    def evidence(self) -> dict[str, Any]:
        classes = [class_identity(cls) for cls in sorted(self.initial, key=lambda value: (value.__module__, value.__qualname__))]
        configs = [{"source": source, "value_before": before, "value_during": False, "value_after": getattr(config, "use_cache", None)} for source, config, before in self.configs]
        return {"classes": classes, "check_labels": self.labels, "check_count": len(self.labels), "trip_count": self.trip_count, "mandatory_negative_control": self.negative, "actual_allocation_trips": max(0, self.trip_count - 1), "pkv_non_none_count": self.pkv_non_none_count, "config_drift_count": self.config_drift_count, "configuration_evidence": configs, "classes_restored": self.restored, "configs_restored": self.restored, "complete": classes == self.expected and self.labels == CACHE_LABELS and self.trip_count == 1 and self.pkv_non_none_count == 0 and self.config_drift_count == 0 and self.restored}


def validate_probe(probe: dict[str, Any], expected: dict[str, Any], *, require_qualifies: bool = True) -> None:
    if set(probe) != PROBE_KEYS:
        raise CAP0ContractError("CAP0 probe schema differs")
    for key in ("probe_index", "depth", "action_index", "replicate", "row_id", "row_sha256", "receiver_input_sha256"):
        if probe[key] != expected[key]:
            raise CAP0ContractError("CAP0 probe identity differs")
    if probe["node_count"] != 24 or probe["local_text_byte_lengths"] != [68] * 24 or len(probe["unpadded_token_lengths"]) != 24 or any(not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 128 for value in probe["unpadded_token_lengths"]):
        raise CAP0ContractError("CAP0 probe input geometry differs")
    if probe["input_ids_shape"] != [24, 128] or probe["attention_mask_shape"] != [24, 128] or len(probe["repeats"]) != 2:
        raise CAP0ContractError("CAP0 probe tensor geometry differs")
    for repeat_index, repeat in enumerate(probe["repeats"], 1):
        if set(repeat) != REPEAT_KEYS or repeat["case_index"] != 2 * (expected["probe_index"] - 1) + repeat_index or repeat["repeat"] != repeat_index or repeat["model_call_index"] != 2 * (expected["probe_index"] - 1) + repeat_index:
            raise CAP0ContractError("CAP0 repeat identity differs")
        shapes = {"logits_shape": [24, 1, 248320], "full_hidden_shape": [24, 128, 2048], "capture_shape": [24, 2048]}
        if any(repeat[key] != value for key, value in shapes.items()) or any(repeat[key] != "torch.bfloat16" for key in ("logits_dtype", "full_hidden_dtype", "capture_dtype")):
            raise CAP0ContractError("CAP0 repeat geometry differs")
        if any(repeat[key] is not True for key in ("input_ids_same_object", "attention_mask_same_object", "logits_finite", "full_hidden_finite", "capture_finite", "pkv_is_none")):
            raise CAP0ContractError("CAP0 repeat gate differs")
        if repeat["input_ids_sha256"] != probe["input_ids_sha256"] or repeat["attention_mask_sha256"] != probe["attention_mask_sha256"]:
            raise CAP0ContractError("CAP0 repeat input identity differs")
        if any(not isinstance(repeat[key], str) or len(repeat[key]) != 64 for key in ("logits_sha256", "full_hidden_sha256", "capture_sha256")):
            raise CAP0ContractError("CAP0 repeat hashes differ")
    gates = ("repeat_input_ids_bitwise", "repeat_attention_mask_bitwise", "repeat_full_hidden_bitwise", "repeat_capture_bitwise", "all_outputs_finite", "not_all_node_identical", "complete")
    if any(not isinstance(probe[key], bool) for key in (*gates, "qualifies")) or len(probe["capture_row_sha256"]) != 24 or len(set(probe["capture_row_sha256"])) != probe["unique_capture_row_count"] or probe["not_all_node_identical"] is not (probe["unique_capture_row_count"] >= 2):
        raise CAP0ContractError("CAP0 probe qualification differs")
    computed_qualifies = all(probe[key] for key in gates)
    if probe["qualifies"] is not computed_qualifies or require_qualifies and not computed_qualifies:
        raise CAP0ContractError("CAP0 probe qualification truth closure differs")
    if probe["repeat_full_hidden_bitwise"] is not (probe["repeats"][0]["full_hidden_sha256"] == probe["repeats"][1]["full_hidden_sha256"]) or probe["repeat_capture_bitwise"] is not (probe["repeats"][0]["capture_sha256"] == probe["repeats"][1]["capture_sha256"]):
        raise CAP0ContractError("CAP0 repeat evidence differs")


def validate_cache(value: dict[str, Any], contract: dict[str, Any], complete: bool) -> None:
    keys = {"classes", "check_labels", "check_count", "trip_count", "mandatory_negative_control", "actual_allocation_trips", "pkv_non_none_count", "config_drift_count", "configuration_evidence", "classes_restored", "configs_restored", "complete"}
    if not isinstance(value, dict) or set(value) != keys or value["classes"] != contract["cache_contract"]["class_closure"] or value["check_count"] != len(value["check_labels"]) or value["check_labels"] != CACHE_LABELS[: len(value["check_labels"])]:
        raise CAP0ContractError("CAP0 cache evidence differs")
    if not isinstance(value["trip_count"], int) or isinstance(value["trip_count"], bool) or value["trip_count"] < 0 or value["actual_allocation_trips"] != max(0, value["trip_count"] - 1):
        raise CAP0ContractError("CAP0 cache trip evidence differs")
    if any(not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] < 0 for key in ("pkv_non_none_count", "config_drift_count")) or any(not isinstance(value[key], bool) for key in ("mandatory_negative_control", "classes_restored", "configs_restored", "complete")):
        raise CAP0ContractError("CAP0 cache scalar types differ")
    configurations = value["configuration_evidence"]
    if not isinstance(configurations, list) or any(not isinstance(row, dict) or not isinstance(row.get("source"), str) for row in configurations):
        raise CAP0ContractError("CAP0 cache configuration schema differs")
    sources = [row["source"] for row in configurations]
    if sources != sorted(sources) or len(set(sources)) != len(configurations):
        raise CAP0ContractError("CAP0 cache configuration ordering differs")
    for row in configurations:
        if set(row) != {"source", "value_before", "value_during", "value_after"} or not isinstance(row["source"], str) or row["value_during"] is not False or row["value_before"] not in {True, False}:
            raise CAP0ContractError("CAP0 cache configuration evidence differs")
        if value["configs_restored"] and row["value_after"] is not row["value_before"]:
            raise CAP0ContractError("CAP0 cache configuration restoration differs")
    if complete and (value["check_labels"] != CACHE_LABELS or value["trip_count"] != 1 or value["mandatory_negative_control"] is not True or value["pkv_non_none_count"] != 0 or value["config_drift_count"] != 0 or value["classes_restored"] is not True or value["configs_restored"] is not True or value["complete"] is not True):
        raise CAP0ContractError("CAP0 cache closure incomplete")


def validate_proof(proof: dict[str, Any], *, plan: dict[str, Any], contract: dict[str, Any], selection: dict[str, Any], execution_commit: str, plan_file_sha256: str) -> None:
    if set(proof) != PROOF_KEYS or proof["schema_version"] != PROOF_SCHEMA or proof["status"] != PROOF_STATUS or proof["mechanism"] != MECHANISM or proof["run_identity"] != RUN_ID:
        raise CAP0ContractError("CAP0 proof identity differs")
    if proof["execution_commit"] != execution_commit or proof["mechanism_code_commit"] != plan["mechanism_code_commit"] or proof["plan_file_sha256"] != plan_file_sha256 or proof["plan_sha256"] != plan["plan_sha256"]:
        raise CAP0ContractError("CAP0 proof authority differs")
    if proof["runtime"] != RUNTIME or proof["mf0_archive_binding"] != MF0_BINDING or proof["selection"] != {"selection_sha256": SELECTION_SHA256, "ordered_probes": selection["ordered_probes"]}:
        raise CAP0ContractError("CAP0 proof frozen input differs")
    model_identity = {"class": RUNTIME["model_class"], "hidden_size": 2048, "vocab_size": 248320, "dtype": "torch.bfloat16", "device": "cuda:0", "checkpoint": E33_PATH, "checkpoint_tree_sha256": E33_TREE_SHA256}
    if proof["model_identity"] != model_identity:
        raise CAP0ContractError("CAP0 proof model identity differs")
    audit = proof["asset_audit"]
    if not isinstance(audit, dict) or set(audit) != {"before", "after", "equal"} or audit["before"] != plan["asset_sha256"] or audit["after"] != plan["asset_sha256"] or audit["equal"] is not True:
        raise CAP0ContractError("CAP0 proof asset audit differs")
    if not isinstance(proof["probes"], list) or len(proof["probes"]) != 4:
        raise CAP0ContractError("CAP0 proof probe count differs")
    for observed, expected in zip(proof["probes"], selection["ordered_probes"], strict=True):
        validate_probe(observed, expected)
    aggregate = {"complete_modality_count": 4, "qualifying_modality_count": 4, "complete_probe_count": 4, "qualifying_probe_count": 4, "tokenizer_calls": 4, "model_forwards": 8, "sequences": 192, "all_qualify": True}
    if proof["aggregate"] != aggregate or proof["counts"] != COUNTS or proof["decision_boundary"] != DECISION:
        raise CAP0ContractError("CAP0 proof aggregate differs")
    validate_cache(proof["cache_guard"], contract, complete=True)
    protected = proof["protected_state"]
    protected_keys = {"disk_before", "disk_after", "e33_state_before", "e33_state_after", "e33_all_requires_grad_false", "e33_all_grads_none", "model_eval", "h176_loaded", "model_released"}
    expected_disk = {"e33_tree_sha256": E33_TREE_SHA256, "h176_tree_sha256": H176_TREE_SHA256, "e33_metadata_sha256": METADATA_SHA256, "h176_metadata_sha256": METADATA_SHA256}
    if not isinstance(protected, dict) or set(protected) != protected_keys or protected["disk_before"] != expected_disk or protected["disk_after"] != expected_disk or protected["e33_state_before"] != E33_STATE_SHA256 or protected["e33_state_after"] != E33_STATE_SHA256 or any(protected[key] is not True for key in ("e33_all_requires_grad_false", "e33_all_grads_none", "model_eval", "model_released")) or protected["h176_loaded"] is not False:
        raise CAP0ContractError("CAP0 protected state differs")
    safety = proof["safety"]
    safety_keys = {"cuda_visible_devices", "network_attempts", "validation_opens", "heldout_opens", "generation_calls", "backwards", "optimizer_objects", "optimizer_steps", "candidate_objects", "candidate_files", "checkpoint_files", "model_updated", "h176_loaded", "network_guard", "experiment_open_firewall"}
    if not isinstance(safety, dict) or set(safety) != safety_keys:
        raise CAP0ContractError("CAP0 proof safety differs")
    scalar_expected = {"cuda_visible_devices": "0", "network_attempts": 0, "validation_opens": 0, "heldout_opens": 0, "generation_calls": 0, "backwards": 0, "optimizer_objects": 0, "optimizer_steps": 0, "candidate_objects": 0, "candidate_files": [], "checkpoint_files": [], "model_updated": False, "h176_loaded": False}
    if any(safety[key] != value for key, value in scalar_expected.items()):
        raise CAP0ContractError("CAP0 proof safety scalar differs")
    network = safety["network_guard"]
    if not isinstance(network, dict) or set(network) != {"installed", "wrappers_restored", "audit_hook_persistent", "attempt_count", "operations", "audit_events"} or network["installed"] is not True or network["wrappers_restored"] is not False or network["audit_hook_persistent"] is not True or network["attempt_count"] != 0 or network["operations"] != ["socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection", "socket.getaddrinfo"] or network["audit_events"] != ["socket.connect", "socket.getaddrinfo"]:
        raise CAP0ContractError("CAP0 network guard evidence differs")
    firewall = safety["experiment_open_firewall"]
    if not isinstance(firewall, dict) or set(firewall) != {"denied_count", "validation_open_count", "heldout_open_count", "opened_paths"} or any(firewall[key] != 0 for key in ("denied_count", "validation_open_count", "heldout_open_count")) or not isinstance(firewall["opened_paths"], list) or firewall["opened_paths"] != sorted(set(firewall["opened_paths"])) or any(path not in EXPERIMENT_READ_PATHS for path in firewall["opened_paths"]):
        raise CAP0ContractError("CAP0 experiment-open firewall evidence differs")
    resources = proof["resources"]
    resource_keys = {"bounds", "gpu_name", "gpu_total_bytes", "host_ram_bytes", "free_disk_bytes_preflight", "free_disk_bytes_postflight", "global_max_allocated_bytes", "global_max_reserved_bytes", "artifact_bytes_before_terminal", "completed_phase_records", "final_terminal_publication", "prepublication_elapsed_ns"}
    cap = 40 * 2**30
    if not isinstance(resources, dict) or set(resources) != resource_keys or resources["bounds"] != RESOURCE_BOUNDS or resources["gpu_name"] != RUNTIME["gpu_model"] or resources["gpu_total_bytes"] < 47 * 2**30 or resources["host_ram_bytes"] < 64 * 2**30 or resources["free_disk_bytes_preflight"] < 16 * 2**30 or resources["free_disk_bytes_postflight"] < 16 * 2**30 or resources["global_max_allocated_bytes"] > cap or resources["global_max_reserved_bytes"] > cap or resources["artifact_bytes_before_terminal"] != 0:
        raise CAP0ContractError("CAP0 resource evidence differs")
    validate_phase_records(resources["completed_phase_records"], resources["final_terminal_publication"], resources["prepublication_elapsed_ns"], success=True)
    memory = proof["memory"]
    if set(memory) != {"labels", "label_sha256", "rows"} or memory["labels"] != MEMORY_LABELS or memory["label_sha256"] != sha256_bytes(canonical_json(MEMORY_LABELS)):
        raise CAP0ContractError("CAP0 memory evidence differs")
    validate_memory(memory["rows"], complete=True)
    freeze = proof["full_freeze"]
    if set(freeze) != {"head_before", "head_after", "parent", "tree_before", "tree_after", "status_before", "status_after", "assets_equal"} or freeze["head_before"] != execution_commit or freeze["head_after"] != execution_commit or freeze["parent"] != plan["mechanism_code_commit"] or freeze["tree_before"] != freeze["tree_after"] or freeze["status_before"] != "" or freeze["status_after"] != "" or freeze["assets_equal"] is not True:
        raise CAP0ContractError("CAP0 full freeze differs")
    tampers = proof["tamper_audit"]
    if set(tampers) != {"results", "rejected_count"} or tampers["rejected_count"] != 42 or [row.get("name") for row in tampers["results"]] != TAMPERS or any(set(row) != {"name", "rejected", "error_type"} or row["rejected"] is not True or row["error_type"] != "CAP0ContractError" for row in tampers["results"]):
        raise CAP0ContractError("CAP0 tamper audit differs")
    if proof["proof_sha256"] != sha256_bytes(canonical_json({key: value for key, value in proof.items() if key != "proof_sha256"})):
        raise CAP0ContractError("CAP0 proof self hash differs")


def validate_phase_records(records: Any, terminal: Any, elapsed: Any, success: bool) -> None:
    if not isinstance(records, list) or not isinstance(terminal, dict) or elapsed != terminal.get("entered_ns_since_start"):
        raise CAP0ContractError("CAP0 timing evidence differs")
    phases = [row.get("phase") for row in records]
    outcomes = [row.get("outcome") for row in records]
    expected = (["compute", "audit"], ["completed", "completed"], 2880) if success else (None, None, None)
    if success and (phases != expected[0] or outcomes != expected[1]):
        raise CAP0ContractError("CAP0 success phase sequence differs")
    allowed_failure = {("compute", "failure_audit"): 2820, ("compute", "audit", "failure_audit"): 3000, ("compute", "audit", "terminal_publication", "failure_audit"): 3060}
    maximum = expected[2] if success else allowed_failure.get(tuple(phases))
    if maximum is None or elapsed > maximum * 10**9:
        raise CAP0ContractError("CAP0 terminal entry timing differs")
    expected_failure_outcomes = {
        ("compute", "failure_audit"): ["error", "completed"],
        ("compute", "audit", "failure_audit"): ["completed", "error", "completed"],
        ("compute", "audit", "terminal_publication", "failure_audit"): ["completed", "completed", "error", "completed"],
    }
    if not success and outcomes != expected_failure_outcomes[tuple(phases)]:
        raise CAP0ContractError("CAP0 failure phase outcomes differ")
    previous = -1
    caps = {"compute": 2700, "audit": 180, "failure_audit": 120, "terminal_publication": 60}
    for row in records:
        keys = {"phase", "entered_ns_since_start", "exited_ns_since_start", "duration_ns", "outcome", "cap_ns", "alarm_after_ns", "alarm_safety_margin_ns"}
        if set(row) != keys or row["phase"] not in caps or row["entered_ns_since_start"] < previous or row["exited_ns_since_start"] < row["entered_ns_since_start"] or row["duration_ns"] != row["exited_ns_since_start"] - row["entered_ns_since_start"] or row["duration_ns"] > row["cap_ns"] or row["cap_ns"] != caps[row["phase"]] * 10**9 or row["alarm_after_ns"] + row["alarm_safety_margin_ns"] != row["cap_ns"] or row["alarm_safety_margin_ns"] != 10**9:
            raise CAP0ContractError("CAP0 completed phase record differs")
        previous = row["exited_ns_since_start"]
    expected_terminal = {"phase": "terminal_publication", "entered_ns_since_start": elapsed, "limit_ns": 60 * 10**9, "completion_observable_inside_terminal": False, "self_reference_boundary": "post_write_fsync_reopen_validation_and_process_exit_are_external_to_immutable_terminal_bytes"}
    if terminal != expected_terminal:
        raise CAP0ContractError("CAP0 terminal boundary differs")


def validate_failure(failure: dict[str, Any], *, plan: dict[str, Any], contract: dict[str, Any], selection: dict[str, Any], execution_commit: str, plan_file_sha256: str) -> None:
    if set(failure) != FAILURE_KEYS or failure["schema_version"] != FAILURE_SCHEMA or failure["mechanism"] != MECHANISM or failure["run_identity"] != RUN_ID:
        raise CAP0ContractError("CAP0 failure identity differs")
    if failure["execution_commit"] != execution_commit or failure["mechanism_code_commit"] != plan["mechanism_code_commit"] or failure["plan_file_sha256"] != plan_file_sha256 or failure["plan_sha256"] != plan["plan_sha256"]:
        raise CAP0ContractError("CAP0 failure authority differs")
    if not isinstance(failure["aggregate_partial"], dict) or not isinstance(failure["actual_safety"], dict):
        raise CAP0ContractError("CAP0 failure nested schema differs")
    if failure["status"] == REJECT_STATUS:
        if failure["error_type"] != "CAP0MechanismRejected" or failure["aggregate_partial"].get("cause") not in MECHANISM_CAUSES:
            raise CAP0ContractError("CAP0 rejection taxonomy differs")
    elif failure["status"] == INCOMPLETE_STATUS:
        if failure["error_type"] != "CAP0ContractError":
            raise CAP0ContractError("CAP0 incomplete taxonomy differs")
    elif failure["status"] == EXPOSURE_STATUS:
        if failure["error_type"] != "ExposureBoundaryRejected" or not failure["actual_safety"].get("positive_exposure"):
            raise CAP0ContractError("CAP0 exposure taxonomy differs")
    elif failure["status"] == "infrastructure_invalid":
        if failure["error_type"] not in {"InfrastructureInvalid", "TimeoutError", "MemoryError", "OutOfMemoryError", "CUDAOutOfMemoryError", "OSError", "PermissionError", "FileNotFoundError", "CalledProcessError", "ImportError", "ModuleNotFoundError"}:
            raise CAP0ContractError("CAP0 infrastructure taxonomy differs")
    else:
        raise CAP0ContractError("CAP0 failure status differs")
    progress = failure["progress"]
    progress_keys = {"stage", "current_probe", "current_repeat", "tokenizer_calls_completed", "model_forwards_completed", "sequences_completed", "probes_completed", "cache_checks_completed", "memory_rows_completed", "model_loaded", "model_released"}
    if not isinstance(progress, dict) or set(progress) != progress_keys or progress["stage"] not in {"startup_pre_model", "model_load", "capture", "postflight_audit", "model_release", "terminal_publication"}:
        raise CAP0ContractError("CAP0 failure progress differs")
    for key, maximum in (("tokenizer_calls_completed", 4), ("model_forwards_completed", 8), ("probes_completed", 4), ("cache_checks_completed", 18), ("memory_rows_completed", 28)):
        if not isinstance(progress[key], int) or isinstance(progress[key], bool) or not 0 <= progress[key] <= maximum:
            raise CAP0ContractError("CAP0 failure counter differs")
    tokenizer_calls = progress["tokenizer_calls_completed"]
    forwards = progress["model_forwards_completed"]
    probes_completed = progress["probes_completed"]
    legal_tokenizer_counts = {forwards // 2 + (forwards % 2)}
    if forwards % 2 == 0 and forwards < 8:
        legal_tokenizer_counts.add(forwards // 2 + 1)
    legal_probe_counts = {forwards // 2}
    if forwards >= 2 and forwards % 2 == 0:
        legal_probe_counts.add(forwards // 2 - 1)
    if progress["sequences_completed"] != 24 * tokenizer_calls or tokenizer_calls not in legal_tokenizer_counts or probes_completed not in legal_probe_counts:
        raise CAP0ContractError("CAP0 failure counter crossing differs")
    if not isinstance(progress["model_loaded"], bool) or not isinstance(progress["model_released"], bool) or progress["model_released"] and not progress["model_loaded"]:
        raise CAP0ContractError("CAP0 failure model stage differs")
    if progress["stage"] == "startup_pre_model" and progress["model_loaded"] or progress["stage"] in {"capture", "postflight_audit", "model_release"} and not progress["model_loaded"]:
        raise CAP0ContractError("CAP0 failure model/stage crossing differs")
    if (progress["tokenizer_calls_completed"] or progress["model_forwards_completed"]) and progress["stage"] not in {"capture", "postflight_audit", "model_release", "terminal_publication"}:
        raise CAP0ContractError("CAP0 scientific counter/stage crossing differs")
    if progress["current_probe"] is not None and (not isinstance(progress["current_probe"], int) or isinstance(progress["current_probe"], bool) or not 1 <= progress["current_probe"] <= 4):
        raise CAP0ContractError("CAP0 failure current probe differs")
    if progress["current_repeat"] is not None and progress["current_repeat"] not in {1, 2}:
        raise CAP0ContractError("CAP0 failure current repeat differs")
    partial = failure["partial_probes"]
    if not isinstance(partial, list) or len(partial) != progress["probes_completed"]:
        raise CAP0ContractError("CAP0 failure probe prefix differs")
    for observed, expected in zip(partial, selection["ordered_probes"], strict=False):
        validate_probe(observed, expected, require_qualifies=False)
    cache = failure["cache_guard_partial"]
    if cache is not None:
        validate_cache(cache, contract, complete=False)
        n = progress["model_forwards_completed"]
        legal_cache_lengths = {1 + 2 * n}
        if n < 8:
            legal_cache_lengths.add(2 + 2 * n)
        elif n == 8:
            legal_cache_lengths.add(18)
        pre_post_cause = failure["aggregate_partial"].get("cause")
        if n >= 1 and (pre_post_cause == "returned_pkv_non_none" and cache["pkv_non_none_count"] >= 1 or pre_post_cause == "cache_configuration_drift" and cache["config_drift_count"] >= 1):
            legal_cache_lengths.add(2 * n)
        if len(cache["check_labels"]) not in legal_cache_lengths or progress["cache_checks_completed"] != len(cache["check_labels"]) or cache["mandatory_negative_control"] is not True or cache["trip_count"] < 1:
            raise CAP0ContractError("CAP0 failure cache prefix differs")
    elif progress["cache_checks_completed"] != 0:
        raise CAP0ContractError("CAP0 missing failure cache evidence")
    aggregate_partial = failure["aggregate_partial"]
    if not isinstance(aggregate_partial, dict) or set(aggregate_partial) != {"cause", "probes_completed", "nonfinite_observed"} or aggregate_partial["probes_completed"] != len(partial) or not isinstance(aggregate_partial["nonfinite_observed"], bool):
        raise CAP0ContractError("CAP0 failure aggregate schema differs")
    cause = aggregate_partial["cause"]
    positive = {
        "cache_allocation_detected": cache is not None and cache["trip_count"] >= 2,
        "cache_configuration_drift": cache is not None and cache["config_drift_count"] >= 1,
        "returned_pkv_non_none": cache is not None and cache["pkv_non_none_count"] >= 1,
        "nonfinite_output": aggregate_partial["nonfinite_observed"],
        "repeat_parity_failed": len(partial) == 4 and any(not row["repeat_full_hidden_bitwise"] or not row["repeat_capture_bitwise"] for row in partial),
        "node_diversity_failed": len(partial) == 4 and any(not row["not_all_node_identical"] for row in partial),
    }
    expected_cause = next((name for name in ("cache_allocation_detected", "cache_configuration_drift", "returned_pkv_non_none", "nonfinite_output", "node_diversity_failed", "repeat_parity_failed") if positive[name]), None)
    if failure["status"] == REJECT_STATUS:
        if cause != expected_cause:
            raise CAP0ContractError("CAP0 rejection lacks exact positive cause evidence")
    elif failure["status"] == INCOMPLETE_STATUS and (cause is not None or expected_cause is not None):
        raise CAP0ContractError("CAP0 mechanism cause masked by another status")
    elif failure["status"] in {EXPOSURE_STATUS, "infrastructure_invalid"} and cause != expected_cause:
        raise CAP0ContractError("CAP0 secondary mechanism evidence differs")
    if not isinstance(failure["partial_memory"], dict) or set(failure["partial_memory"]) != {"labels", "rows"}:
        raise CAP0ContractError("CAP0 failure memory schema differs")
    validate_memory(failure["partial_memory"].get("rows"), complete=False)
    if failure["partial_memory"].get("labels") != MEMORY_LABELS or len(failure["partial_memory"]["rows"]) != progress["memory_rows_completed"]:
        raise CAP0ContractError("CAP0 failure memory prefix differs")
    actual = failure["actual_safety"]
    actual_keys = {"validation_opens", "heldout_opens", "h176_loaded", "generation_calls", "backwards", "optimizer_objects", "candidate_objects", "network_attempts", "e33_grads_present", "e33_state_changed", "positive_exposure", "network_guard", "experiment_open_firewall"}
    if set(actual) != actual_keys:
        raise CAP0ContractError("CAP0 failure actual safety schema differs")
    network = actual["network_guard"]
    firewall = actual["experiment_open_firewall"]
    if not isinstance(network, dict) or set(network) != {"installed", "wrappers_restored", "audit_hook_persistent", "attempt_count", "operations", "audit_events"} or not isinstance(firewall, dict) or set(firewall) != {"denied_count", "validation_open_count", "heldout_open_count", "opened_paths"}:
        raise CAP0ContractError("CAP0 failure guard evidence differs")
    if actual["network_attempts"] != network["attempt_count"] or actual["validation_opens"] != firewall["validation_open_count"] or actual["heldout_opens"] != firewall["heldout_open_count"]:
        raise CAP0ContractError("CAP0 failure guard counters differ")
    if any(not isinstance(actual[key], int) or isinstance(actual[key], bool) or actual[key] < 0 for key in ("validation_opens", "heldout_opens", "generation_calls", "backwards", "optimizer_objects", "candidate_objects", "network_attempts")) or any(not isinstance(actual[key], bool) for key in ("h176_loaded", "e33_grads_present", "e33_state_changed", "positive_exposure")):
        raise CAP0ContractError("CAP0 failure safety scalar types differ")
    if network["installed"] is not True or network["audit_hook_persistent"] is not True or not isinstance(network["wrappers_restored"], bool) or not isinstance(network["attempt_count"], int) or isinstance(network["attempt_count"], bool) or network["attempt_count"] < 0:
        raise CAP0ContractError("CAP0 failure network guard differs")
    if any(not isinstance(firewall[key], int) or isinstance(firewall[key], bool) or firewall[key] < 0 for key in ("denied_count", "validation_open_count", "heldout_open_count")) or not isinstance(firewall["opened_paths"], list) or firewall["opened_paths"] != sorted(set(firewall["opened_paths"])):
        raise CAP0ContractError("CAP0 failure open firewall differs")
    exposure = bool(actual["validation_opens"] or actual["heldout_opens"] or actual["h176_loaded"] or actual["generation_calls"] or actual["backwards"] or actual["optimizer_objects"] or actual["candidate_objects"] or actual["network_attempts"] or firewall["denied_count"] or failure["candidate_files"] or failure["checkpoint_files"] or failure["model_updated"] or actual["e33_grads_present"] or actual["e33_state_changed"])
    if actual.get("positive_exposure") is not exposure or (failure["status"] == EXPOSURE_STATUS) is not exposure:
        raise CAP0ContractError("CAP0 failure exposure precedence differs")
    if failure["status"] != EXPOSURE_STATUS and (failure["candidate_files"] or failure["checkpoint_files"] or failure["model_updated"]):
        raise CAP0ContractError("CAP0 failure artifact safety differs")
    protected = failure["protected_state"]
    protected_keys = {"disk_before", "disk_after", "e33_state_before", "e33_state_after", "grads_present", "restoration_attempted"}
    if not isinstance(protected, dict) or set(protected) != protected_keys or not isinstance(protected["grads_present"], bool) or not isinstance(protected["restoration_attempted"], bool):
        raise CAP0ContractError("CAP0 failure protected schema differs")
    expected_disk = {"e33_tree_sha256": E33_TREE_SHA256, "h176_tree_sha256": H176_TREE_SHA256, "e33_metadata_sha256": METADATA_SHA256, "h176_metadata_sha256": METADATA_SHA256}

    def disk_observation_valid(value: Any) -> bool:
        if value is None:
            return True
        if not isinstance(value, dict) or set(value) != set(expected_disk):
            return False
        if any(not isinstance(value[key], str) or len(value[key]) != 64 for key in ("e33_tree_sha256", "h176_tree_sha256")):
            return False
        for key in ("e33_metadata_sha256", "h176_metadata_sha256"):
            if not isinstance(value[key], dict) or set(value[key]) != set(METADATA_SHA256) or any(not isinstance(item, str) or len(item) != 64 for item in value[key].values()):
                return False
        return True

    if not disk_observation_valid(protected["disk_before"]) or not disk_observation_valid(protected["disk_after"]) or protected["e33_state_before"] is not None and (not isinstance(protected["e33_state_before"], str) or len(protected["e33_state_before"]) != 64) or protected["e33_state_after"] is not None and (not isinstance(protected["e33_state_after"], str) or len(protected["e33_state_after"]) != 64):
        raise CAP0ContractError("CAP0 failure protected identity differs")
    protected_complete = protected["disk_before"] == expected_disk and protected["disk_after"] == expected_disk and protected["e33_state_before"] == E33_STATE_SHA256 and isinstance(protected["e33_state_after"], str) and len(protected["e33_state_after"]) == 64 and protected["restoration_attempted"] is True
    if progress["model_loaded"] and failure["status"] != "infrastructure_invalid" and not protected_complete:
        raise CAP0ContractError("CAP0 failure protected audit incomplete")
    if progress["model_loaded"] and protected_complete:
        state_changed = protected["e33_state_after"] != E33_STATE_SHA256
        if actual["e33_state_changed"] is not state_changed or actual["e33_grads_present"] is not protected["grads_present"]:
            raise CAP0ContractError("CAP0 failure protected truth closure differs")
        if failure["status"] != EXPOSURE_STATUS and (state_changed or protected["grads_present"]):
            raise CAP0ContractError("CAP0 protected exposure masked")
    elif progress["model_loaded"] and failure["status"] == "infrastructure_invalid" and actual["e33_state_changed"]:
        raise CAP0ContractError("CAP0 positive protected exposure masked")
    if progress["model_loaded"] and (protected["disk_before"] != expected_disk or protected["e33_state_before"] != E33_STATE_SHA256 or protected["disk_after"] != expected_disk) and failure["status"] != "infrastructure_invalid":
        raise CAP0ContractError("CAP0 protected infrastructure masked")
    resources = failure["resources"]
    if not isinstance(resources, dict) or set(resources) != {"bounds", "completed_phase_records", "final_terminal_publication", "prepublication_elapsed_ns"} or resources["bounds"] != RESOURCE_BOUNDS:
        raise CAP0ContractError("CAP0 failure resource schema differs")
    validate_phase_records(failure["resources"]["completed_phase_records"], failure["resources"]["final_terminal_publication"], failure["resources"]["prepublication_elapsed_ns"], success=False)
    if not isinstance(failure["candidate_files"], list) or not isinstance(failure["checkpoint_files"], list) or failure["output_inventory_before_failure"] != sorted([*failure["candidate_files"], *failure["checkpoint_files"]]):
        raise CAP0ContractError("CAP0 failure output inventory differs")
    audit = failure["full_freeze_failure_audit"]
    audit_keys = {"head", "parent", "tree", "status", "asset_hashes", "execution_commit", "errors", "exact"}
    if not isinstance(audit, dict) or set(audit) != audit_keys:
        raise CAP0ContractError("CAP0 failure freeze audit differs")
    if not isinstance(audit["errors"], list) or any(not isinstance(value, str) or not value for value in audit["errors"]) or not isinstance(audit["asset_hashes"], dict) or set(audit["asset_hashes"]) != set(plan["asset_sha256"]):
        raise CAP0ContractError("CAP0 failure freeze observation differs")
    exact = not audit["errors"] and audit["head"] == execution_commit and audit["parent"] == plan["mechanism_code_commit"] and audit["status"] == "" and audit["asset_hashes"] == plan["asset_sha256"] and audit["execution_commit"] == execution_commit
    if audit["exact"] is not exact or failure["status"] != "infrastructure_invalid" and not exact:
        raise CAP0ContractError("CAP0 failure provenance truth closure differs")
    if failure["failure_sha256"] != sha256_bytes(canonical_json({key: value for key, value in failure.items() if key != "failure_sha256"})):
        raise CAP0ContractError("CAP0 failure self hash differs")


def load_authorized_plan(repo: Path, execution_commit: str, external_sha256: str) -> dict[str, Any]:
    data = git_blob(repo, execution_commit, PLAN_RELATIVE_PATH)
    if not data.endswith(b"\n") or sha256_bytes(data) != external_sha256:
        raise InfrastructureInvalid("CAP0 authorized plan blob differs")
    plan = strict_loads(data[:-1])
    if data != canonical_json(plan) + b"\n":
        raise InfrastructureInvalid("CAP0 authorized plan is noncanonical")
    validate_plan(plan, external_sha256)
    return plan


def load_authorized_canonical(repo: Path, execution_commit: str, path: str, expected_sha256: str) -> dict[str, Any]:
    data = git_blob(repo, execution_commit, path)
    if not data.endswith(b"\n") or sha256_bytes(data) != expected_sha256:
        raise InfrastructureInvalid(f"CAP0 authorized asset blob differs: {path}")
    value = strict_loads(data[:-1])
    if not isinstance(value, dict) or data != canonical_json(value) + b"\n":
        raise InfrastructureInvalid(f"CAP0 authorized asset is noncanonical: {path}")
    return value


def asset_hashes(repo: Path, plan: dict[str, Any]) -> dict[str, str]:
    return {path: file_sha256(repo / path) for path in plan["asset_sha256"]}


def runtime_evidence(torch: Any, transformers: Any) -> dict[str, Any]:
    observed = {
        "python": platform.python_version(),
        "transformers": importlib.metadata.version("transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
        "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
        "torch_distribution": importlib.metadata.version("torch"),
        "torch_runtime": torch.__version__,
        "shared_project_pyproject_sha256": file_sha256(Path(SHARED_PROJECT) / "pyproject.toml"),
        "shared_project_uv_lock_sha256": file_sha256(Path(SHARED_PROJECT) / "uv.lock"),
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "model_class": RUNTIME["model_class"],
        "hidden_size": 2048,
        "vocab_size": 248320,
        "gpu_model": torch.cuda.get_device_name(0),
    }
    if observed != RUNTIME:
        raise InfrastructureInvalid("CAP0 runtime differs")
    return observed


def validate_preflight(value: dict[str, Any], *, execution_commit: str, plan_file_sha256: str, plan: dict[str, Any]) -> None:
    keys = {"schema_version", "status", "mechanism", "run_identity", "execution_commit", "mechanism_code_commit", "plan_file_sha256", "plan_sha256", "asset_count", "asset_hashes_exact", "head_exact", "parent_exact", "tree", "status_clean", "runtime_lock_files_exact", "static_guard", "output_namespace_absent", "cuda_visible_devices", "torch_imported", "tokenizer_loaded", "model_loaded", "scientific_exposure", "preflight_sha256"}
    if set(value) != keys or value["schema_version"] != PREFLIGHT_SCHEMA or value["status"] != PREFLIGHT_STATUS or value["mechanism"] != MECHANISM or value["run_identity"] != RUN_ID:
        raise CAP0ContractError("CAP0 preflight identity differs")
    if value["execution_commit"] != execution_commit or value["mechanism_code_commit"] != plan["mechanism_code_commit"] or value["plan_file_sha256"] != plan_file_sha256 or value["plan_sha256"] != plan["plan_sha256"] or value["asset_count"] != 18:
        raise CAP0ContractError("CAP0 preflight authority differs")
    if any(value[key] is not True for key in ("asset_hashes_exact", "head_exact", "status_clean", "runtime_lock_files_exact", "output_namespace_absent")) or value["parent_exact"] is not True:
        raise CAP0ContractError("CAP0 preflight closure differs")
    if value["cuda_visible_devices"] != "" or any(value[key] is not False for key in ("torch_imported", "tokenizer_loaded", "model_loaded", "scientific_exposure")):
        raise CAP0ContractError("CAP0 preflight exposure differs")
    static = value["static_guard"]
    if not isinstance(static, dict) or set(static) != {"paths", "forbidden_sites", "negative_fixtures", "runner_sha256", "module_sha256"} or static["paths"] != ["src/prime_rl/latent/h_iter_phase1_cap0.py", "scripts/latent/run_h_iter_phase1_cap0_v1.py"] or static["forbidden_sites"] != [] or set(static["negative_fixtures"]) != {"generation_call", "backward_call", "optimizer_step", "candidate_constructor", "validation_path"} or not all(value is True for value in static["negative_fixtures"].values()) or static["runner_sha256"] != plan["asset_sha256"][PLAN_ASSET_PATHS[14]] or static["module_sha256"] != plan["asset_sha256"][PLAN_ASSET_PATHS[13]]:
        raise CAP0ContractError("CAP0 preflight static guard differs")
    if value["preflight_sha256"] != sha256_bytes(canonical_json({key: item for key, item in value.items() if key != "preflight_sha256"})):
        raise CAP0ContractError("CAP0 preflight self hash differs")


def run_preflight(args: argparse.Namespace) -> None:
    repo = args.repo.resolve(strict=True)
    if str(repo) != REPO_ROOT or args.run_id != RUN_ID or args.output_dir is not None:
        raise InfrastructureInvalid("CAP0 preflight invocation differs")
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise InfrastructureInvalid("CAP0 preflight imported modeling runtime")
    plan = load_authorized_plan(repo, args.execution_commit, args.plan_file_sha256)
    hashes = asset_hashes(repo, plan)
    if hashes != plan["asset_sha256"]:
        raise InfrastructureInvalid("CAP0 preflight asset hashes differ")
    validate_archive(repo)
    _, _, selection, capture = load_train_inputs(repo)
    validate_contract(load_canonical(repo / CONTRACT_RELATIVE_PATH), selection, capture)
    head, parent = run_git(repo, "rev-parse", "HEAD"), run_git(repo, "rev-parse", "HEAD^")
    status_value = run_git(repo, "status", "--porcelain", "--untracked-files=all")
    guard = static_guard(repo)
    value = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "mechanism": MECHANISM,
        "run_identity": RUN_ID,
        "execution_commit": args.execution_commit,
        "mechanism_code_commit": plan["mechanism_code_commit"],
        "plan_file_sha256": args.plan_file_sha256,
        "plan_sha256": plan["plan_sha256"],
        "asset_count": len(hashes),
        "asset_hashes_exact": hashes == plan["asset_sha256"],
        "head_exact": head == args.execution_commit,
        "parent_exact": parent == plan["mechanism_code_commit"],
        "tree": run_git(repo, "rev-parse", "HEAD^{tree}"),
        "status_clean": status_value == "",
        "runtime_lock_files_exact": file_sha256(Path(SHARED_PROJECT) / "pyproject.toml") == RUNTIME["shared_project_pyproject_sha256"] and file_sha256(Path(SHARED_PROJECT) / "uv.lock") == RUNTIME["shared_project_uv_lock_sha256"],
        "static_guard": guard,
        "output_namespace_absent": not Path(OUTPUT_ROOT).exists(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": False,
        "tokenizer_loaded": False,
        "model_loaded": False,
        "scientific_exposure": False,
        "preflight_sha256": "",
    }
    value["preflight_sha256"] = sha256_bytes(canonical_json({key: item for key, item in value.items() if key != "preflight_sha256"}))
    validate_preflight(value, execution_commit=args.execution_commit, plan_file_sha256=args.plan_file_sha256, plan=plan)
    sys.stdout.buffer.write(canonical_json(value) + b"\n")


def _selected_rows(bank: dict[str, Any], selection: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_id = {row["row_id"]: row for row in bank["rows"]}
    return [(probe, by_id[probe["row_id"]]) for probe in selection["ordered_probes"]]


def run_tampers(contract: dict[str, Any], selection: dict[str, Any], capture: dict[str, Any], plan: dict[str, Any], bank: dict[str, Any], partition: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, name in enumerate(TAMPERS[:40]):
        altered = copy.deepcopy(contract)
        altered_plan = copy.deepcopy(plan)
        altered_bank = copy.deepcopy(bank)
        altered_partition = copy.deepcopy(partition)
        altered_selection = copy.deepcopy(selection)
        target = "contract"
        if index < 6:
            field = ("manifest_file_sha256", "manifest_internal_sha256", "proof_file_sha256", "proof_internal_sha256", "launcher_log_sha256", "exit_file_sha256")[index]
            altered["mf0_archive_binding"][field] = "0" * 64
        elif name == "train_bank_hash_changed":
            altered_plan["asset_sha256"][PLAN_ASSET_PATHS[6]] = "0" * 64
            target = "plan"
        elif name == "selection_hash_changed": altered["selection"]["selection_sha256"] = "0" * 64
        elif name == "probe_order_changed": altered["selection"]["ordered_probes"][0], altered["selection"]["ordered_probes"][1] = altered["selection"]["ordered_probes"][1], altered["selection"]["ordered_probes"][0]
        elif name == "probe_not_fit_changed": altered["selection"]["ordered_probes"][0]["replicate"] = 5
        elif name in {"node_order_changed", "local_text_changed", "answer_or_supervision_added"}:
            selected_row = next(row for row in altered_bank["rows"] if row["row_id"] == selection["ordered_probes"][0]["row_id"])
            if name == "node_order_changed": selected_row["receiver_input"]["nodes"][0], selected_row["receiver_input"]["nodes"][1] = selected_row["receiver_input"]["nodes"][1], selected_row["receiver_input"]["nodes"][0]
            elif name == "local_text_changed": selected_row["receiver_input"]["nodes"][0]["local_text"] += "x"
            else: selected_row["receiver_input"]["answer_action"] = "ACT_Z1"
            target = "bank"
        elif name == "tokenizer_max_length_changed": altered["tokenizer_contract"]["max_length"] = 127
        elif name == "tokenizer_truncation_enabled": altered["tokenizer_contract"]["truncation"] = True
        elif name == "tokenizer_padding_side_changed": altered["tokenizer_contract"]["padding_side"] = "right"
        elif name == "model_checkpoint_path_changed": altered["model_contract"]["checkpoint"] += "-changed"
        elif name == "model_checkpoint_hash_changed": altered["model_contract"]["checkpoint_tree_sha256"] = "0" * 64
        elif name == "model_metadata_hash_changed": altered["protected_contract"]["metadata_sha256"]["config.json"] = "0" * 64
        elif name == "model_dtype_changed": altered["model_contract"]["torch_dtype"] = "torch.float32"
        elif name == "attention_implementation_changed": altered["model_contract"]["attn_implementation"] = "sdpa"
        elif name == "inference_mode_disabled": altered["model_contract"]["inference_mode"] = False
        elif name == "use_cache_enabled": altered["model_contract"]["forward_kwargs"]["use_cache"] = True
        elif name == "output_hidden_states_disabled": altered["model_contract"]["forward_kwargs"]["output_hidden_states"] = False
        elif name == "logits_to_keep_changed": altered["model_contract"]["forward_kwargs"]["logits_to_keep"] = 0
        elif name == "cache_class_removed": altered["cache_contract"]["class_closure"] = altered["cache_contract"]["class_closure"][:-1]
        elif name == "cache_source_hash_changed": altered["cache_contract"]["class_closure"][0]["module_sha256"] = "0" * 64
        elif name == "dynamic_cache_negative_control_removed": altered["cache_contract"]["mandatory_negative_trips"] = 0
        elif name == "pkv_none_gate_disabled": altered["output_contract"]["pkv_none"] = False
        elif name in {"repeat_count_changed", "model_forward_count_changed"}: altered["model_contract"]["forwards"] = 7
        elif name == "repeat_hidden_parity_gate_disabled": altered["output_contract"]["repeat_input_mask_hidden_capture_bitwise"] = False
        elif name == "finite_gate_disabled": altered["output_contract"]["finite"] = False
        elif name == "node_diversity_gate_disabled": altered["output_contract"]["unique_capture_rows_minimum"] = 1
        elif name == "h176_load_allowed": altered["protected_contract"]["h176_loaded"] = True
        elif name == "model_update_allowed": altered["protected_contract"]["model_updated"] = True
        elif name == "candidate_or_optimizer_allowed": altered["safety_boundary"]["optimizer"] = True
        elif name == "validation_path_added": altered["safety_boundary"]["validation_opens"] = 1
        elif name == "heldout_path_added": altered["safety_boundary"]["heldout_opens"] = 1
        elif name == "resource_or_memory_cap_changed": altered["resource_bounds"]["allocator_cap_gib"] = 41
        else: raise CAP0ContractError(f"CAP0 tamper has no mutation: {name}")
        if target == "contract":
            altered["contract_sha256"] = sha256_bytes(canonical_json({key: value for key, value in altered.items() if key != "contract_sha256"}))
        elif target == "plan":
            altered_plan["plan_sha256"] = sha256_bytes(canonical_json({key: value for key, value in altered_plan.items() if key != "plan_sha256"}))
        try:
            if target == "contract": validate_contract(altered, selection, capture)
            elif target == "plan":
                validate_plan(altered_plan)
                if altered_plan["asset_sha256"] != plan["asset_sha256"]:
                    raise CAP0ContractError("CAP0 plan asset binding differs")
            else: validate_selected_train_rows(altered_bank, altered_partition, altered_selection)
        except CAP0ContractError:
            results.append({"name": name, "rejected": True, "error_type": "CAP0ContractError"})
        else:
            raise CAP0ContractError(f"CAP0 contract tamper accepted: {name}")
    return results


def _token_geometry(torch: Any, tokenizer: Any, batch: Any) -> tuple[Any, Any, list[int]]:
    if not hasattr(batch, "input_ids") or not hasattr(batch, "attention_mask"):
        raise CAP0ContractError("CAP0 tokenizer did not return BatchEncoding tensors")
    ids, mask = batch.input_ids, batch.attention_mask
    if not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor) or ids.dtype != torch.int64 or mask.dtype != torch.int64 or list(ids.shape) != [24, 128] or list(mask.shape) != [24, 128] or ids.device.type != "cpu" or mask.device.type != "cpu" or not ids.is_contiguous() or not mask.is_contiguous():
        raise CAP0ContractError("CAP0 token tensor geometry differs")
    lengths = [int(value) for value in mask.sum(dim=1).tolist()]
    for row, length in enumerate(lengths):
        if not 1 <= length <= 128 or torch.count_nonzero(mask[row, : 128 - length]).item() != 0 or not torch.all(mask[row, 128 - length :] == 1).item() or not torch.all(ids[row, : 128 - length] == tokenizer.pad_token_id).item():
            raise CAP0ContractError("CAP0 left padding differs")
    return ids, mask, lengths


def capture_probe(torch: Any, model: Any, tokenizer: Any, probe: dict[str, Any], row: dict[str, Any], ledger: MemoryLedger, guard: CacheGuard, progress: dict[str, Any]) -> dict[str, Any]:
    nodes = row["receiver_input"]["nodes"]
    texts = [node["local_text"] for node in nodes]
    if len(nodes) != 24 or any(len(text.encode("utf-8")) != 68 for text in texts):
        raise CAP0ContractError("CAP0 selected local_text bytes differ")
    batch = tokenizer(texts, add_special_tokens=True, padding="max_length", padding_side="left", max_length=128, truncation=False, return_tensors="pt")
    progress["tokenizer_calls_completed"] += 1
    progress["sequences_completed"] += 24
    ids_cpu, mask_cpu, lengths = _token_geometry(torch, tokenizer, batch)
    ids_hash, mask_hash = tensor_sha256(torch, ids_cpu), tensor_sha256(torch, mask_cpu)
    ids, mask = ids_cpu.to("cuda:0"), mask_cpu.to("cuda:0")
    ids_object, mask_object = id(ids), id(mask)
    repeats: list[dict[str, Any]] = []
    full_copies: list[Any] = []
    capture_copies: list[Any] = []
    for repeat in (1, 2):
        call = f"CAP0_P{probe['probe_index']:02d}_R{repeat}"
        progress["current_probe"], progress["current_repeat"] = probe["probe_index"], repeat
        ledger.checkpoint(f"pre_{call}")
        guard.check(f"CACHE_PRE_{call}")
        if hasattr(model, "model") and hasattr(model.model, "rope_deltas"):
            model.model.rope_deltas = None
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True, return_dict=True, logits_to_keep=1)
        progress["model_forwards_completed"] += 1
        if getattr(output, "past_key_values", None) is not None:
            guard.pkv_non_none_count += 1
            raise CAP0MechanismRejected("CAP0 forward returned PKV", "returned_pkv_non_none")
        guard.check(f"CACHE_POST_{call}")
        ledger.checkpoint(f"post_{call}")
        logits, full = output.logits, output.hidden_states[-1]
        capture_tensor = full[:, -1, :]
        if list(logits.shape) != [24, 1, 248320] or list(full.shape) != [24, 128, 2048] or list(capture_tensor.shape) != [24, 2048] or str(logits.dtype) != "torch.bfloat16" or str(full.dtype) != "torch.bfloat16":
            raise CAP0ContractError("CAP0 output geometry differs")
        logits_finite, full_finite, capture_finite = bool(torch.isfinite(logits).all().item()), bool(torch.isfinite(full).all().item()), bool(torch.isfinite(capture_tensor).all().item())
        if not logits_finite or not full_finite or not capture_finite:
            raise CAP0MechanismRejected("CAP0 output nonfinite", "nonfinite_output")
        full_cpu, capture_cpu = full.detach().cpu().contiguous(), capture_tensor.detach().cpu().contiguous()
        full_copies.append(full_cpu)
        capture_copies.append(capture_cpu)
        repeats.append({
            "case_index": 2 * (probe["probe_index"] - 1) + repeat, "repeat": repeat,
            "model_call_index": progress["model_forwards_completed"], "input_ids_same_object": id(ids) == ids_object,
            "attention_mask_same_object": id(mask) == mask_object, "input_ids_sha256": tensor_sha256(torch, ids),
            "attention_mask_sha256": tensor_sha256(torch, mask), "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype), "logits_finite": logits_finite, "logits_sha256": tensor_sha256(torch, logits),
            "full_hidden_shape": list(full.shape), "full_hidden_dtype": str(full.dtype), "full_hidden_finite": full_finite,
            "full_hidden_sha256": tensor_sha256(torch, full_cpu), "capture_shape": list(capture_tensor.shape),
            "capture_dtype": str(capture_tensor.dtype), "capture_finite": capture_finite,
            "capture_sha256": tensor_sha256(torch, capture_cpu), "pkv_is_none": True,
        })
        del output, logits, full, capture_tensor
    ids_unchanged = tensor_sha256(torch, ids) == ids_hash
    mask_unchanged = tensor_sha256(torch, mask) == mask_hash
    full_equal, capture_equal = torch.equal(full_copies[0], full_copies[1]), torch.equal(capture_copies[0], capture_copies[1])
    row_hashes = [tensor_sha256(torch, capture_copies[0][index]) for index in range(24)]
    unique = len(set(row_hashes))
    evidence = {
        **{key: probe[key] for key in ("probe_index", "depth", "action_index", "replicate", "row_id", "row_sha256", "receiver_input_sha256")},
        "node_count": 24, "local_text_byte_lengths": [68] * 24, "unpadded_token_lengths": lengths,
        "input_ids_shape": [24, 128], "attention_mask_shape": [24, 128], "input_ids_sha256": ids_hash,
        "attention_mask_sha256": mask_hash, "repeats": repeats, "repeat_input_ids_bitwise": ids_unchanged,
        "repeat_attention_mask_bitwise": mask_unchanged, "repeat_full_hidden_bitwise": full_equal,
        "repeat_capture_bitwise": capture_equal, "all_outputs_finite": True, "capture_row_sha256": row_hashes,
        "unique_capture_row_count": unique, "not_all_node_identical": unique >= 2, "complete": True,
        "qualifies": ids_unchanged and mask_unchanged and full_equal and capture_equal and unique >= 2,
    }
    progress["probes_completed"] += 1
    del ids_cpu, mask_cpu, ids, mask, batch, full_copies, capture_copies
    return evidence


def host_ram_bytes() -> int:
    return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))


def freeze_state(repo: Path, plan: dict[str, Any], execution_commit: str) -> dict[str, Any]:
    return {
        "head": run_git(repo, "rev-parse", "HEAD"),
        "parent": run_git(repo, "rev-parse", "HEAD^"),
        "tree": run_git(repo, "rev-parse", "HEAD^{tree}"),
        "status": run_git(repo, "status", "--porcelain", "--untracked-files=all"),
        "asset_hashes": asset_hashes(repo, plan),
        "execution_commit": execution_commit,
    }


def failure_freeze_state(repo: Path, plan: dict[str, Any], execution_commit: str) -> dict[str, Any]:
    errors: list[str] = []

    def observe(name: str, operation: Any) -> Any:
        try:
            return operation()
        except BaseException as error:
            errors.append(f"{name}:{type(error).__name__}")
            return None

    assets: dict[str, str | None] = {}
    for path in plan["asset_sha256"]:
        assets[path] = observe(f"asset:{path}", lambda path=path: file_sha256(repo / path))
    value = {
        "head": observe("head", lambda: run_git(repo, "rev-parse", "HEAD")),
        "parent": observe("parent", lambda: run_git(repo, "rev-parse", "HEAD^")),
        "tree": observe("tree", lambda: run_git(repo, "rev-parse", "HEAD^{tree}")),
        "status": observe("status", lambda: run_git(repo, "status", "--porcelain", "--untracked-files=all")),
        "asset_hashes": assets,
        "execution_commit": execution_commit,
        "errors": errors,
        "exact": False,
    }
    value["exact"] = not errors and freeze_exact(value, plan, execution_commit)
    return value


def freeze_exact(value: dict[str, Any], plan: dict[str, Any], execution_commit: str) -> bool:
    return value["head"] == execution_commit and value["parent"] == plan["mechanism_code_commit"] and value["status"] == "" and value["asset_hashes"] == plan["asset_sha256"]


def _alarm(_signal: int, _frame: Any) -> None:
    raise TimeoutError("CAP0 phase timeout")


def run_full(args: argparse.Namespace) -> None:
    repo = args.repo.resolve(strict=True)
    if str(repo) != REPO_ROOT or args.run_id != RUN_ID or args.output_dir != Path(OUTPUT_ROOT) or args.gpu != "0":
        raise InfrastructureInvalid("CAP0 full invocation differs")
    writer = ArtifactWriter(args.output_dir)
    tracker = PhaseTracker()
    tracker.enter("compute", RESOURCE_BOUNDS["compute_timeout_seconds"])
    plan = load_authorized_plan(repo, args.execution_commit, args.plan_file_sha256)
    firewall = OpenFirewall(repo)
    firewall.install()
    contract: dict[str, Any] | None = None
    selection: dict[str, Any] | None = None
    ledger: MemoryLedger | None = None
    guard: CacheGuard | None = None
    model: Any = None
    tokenizer: Any = None
    torch: Any = None
    network = NetworkGuard()
    probes: list[dict[str, Any]] = []
    progress = {"stage": "startup_pre_model", "current_probe": None, "current_repeat": None, "tokenizer_calls_completed": 0, "model_forwards_completed": 0, "sequences_completed": 0, "probes_completed": 0, "cache_checks_completed": 0, "memory_rows_completed": 0, "model_loaded": False, "model_released": False}
    protected_before: dict[str, Any] | None = None
    protected_after: dict[str, Any] | None = None
    state_before: str | None = None
    state_after: str | None = None
    grads_none_after: bool | None = None
    freeze_before: dict[str, Any] | None = None
    asset_before: dict[str, str] | None = None
    free_disk_preflight = shutil.disk_usage(Path(OUTPUT_ROOT).parent).free
    try:
        with network:
            import importlib

            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            cache_utils = importlib.import_module("transformers.cache_utils")
            importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
            importlib.import_module("fla.models.utils")
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise InfrastructureInvalid("CAP0 visible GPU differs")
            runtime = runtime_evidence(torch, transformers)
            properties = torch.cuda.get_device_properties(0)
            cap_bytes = 40 * 2**30
            if properties.total_memory < 47 * 2**30 or cap_bytes >= properties.total_memory or host_ram_bytes() < 64 * 2**30 or free_disk_preflight < 16 * 2**30:
                raise InfrastructureInvalid("CAP0 host resources differ")
            torch.cuda.set_per_process_memory_fraction(cap_bytes / properties.total_memory, 0)
            torch.cuda.reset_peak_memory_stats(0)
            ledger = MemoryLedger(torch)
            ledger.checkpoint("runtime_verified")
            freeze_before = freeze_state(repo, plan, args.execution_commit)
            if not freeze_exact(freeze_before, plan, args.execution_commit):
                raise InfrastructureInvalid("CAP0 preflight freeze differs")
            ledger.checkpoint("full_freeze_preflight_verified")
            validate_archive(repo)
            ledger.checkpoint("mf0_archive_binding_validated")
            bank, _partition, selection, capture = load_train_inputs(repo)
            contract = load_canonical(repo / CONTRACT_RELATIVE_PATH)
            validate_contract(contract, selection, capture)
            selected = _selected_rows(bank, selection)
            ledger.checkpoint("train_bank_and_selection_validated")
            protected_before = protected_disk()
            expected_disk = {"e33_tree_sha256": E33_TREE_SHA256, "h176_tree_sha256": H176_TREE_SHA256, "e33_metadata_sha256": METADATA_SHA256, "h176_metadata_sha256": METADATA_SHA256}
            if protected_before != expected_disk:
                raise InfrastructureInvalid("CAP0 protected checkpoint preflight differs")
            ledger.checkpoint("protected_disk_preflight_verified")
            progress["stage"] = "model_load"
            tokenizer = transformers.AutoTokenizer.from_pretrained(E33_PATH, local_files_only=True)
            model = transformers.AutoModelForImageTextToText.from_pretrained(E33_PATH, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda:0")
            progress["model_loaded"] = True
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            if model.__class__.__name__ != RUNTIME["model_class"] or model.config.text_config.hidden_size != 2048 or model.config.text_config.vocab_size != 248320 or str(next(model.parameters()).dtype) != "torch.bfloat16" or str(next(model.parameters()).device) != "cuda:0" or model.training or any(parameter.requires_grad for parameter in model.parameters()):
                raise InfrastructureInvalid("CAP0 loaded model identity differs")
            state_before = module_tree_sha256(torch, model)
            if state_before != E33_STATE_SHA256:
                raise InfrastructureInvalid("CAP0 loaded state tree differs")
            ledger.checkpoint("model_loaded_frozen")
            progress["stage"] = "capture"
            guard = CacheGuard(model, cache_utils, contract["cache_contract"]["class_closure"])
            with guard:
                ledger.checkpoint("cache_guard_entered")
                for probe, row in selected:
                    evidence = capture_probe(torch, model, tokenizer, probe, row, ledger, guard, progress)
                    probes.append(evidence)
            progress["cache_checks_completed"] = len(guard.labels)
            ledger.checkpoint("cache_guard_audit_complete")
            progress["stage"] = "postflight_audit"
            state_after = module_tree_sha256(torch, model)
            protected_after = protected_disk()
            grads_none = all(parameter.grad is None for parameter in model.parameters())
            grads_none_after = grads_none
            if state_after != state_before or protected_after != protected_before or not grads_none:
                raise ExposureBoundaryRejected("CAP0 protected e33 state changed")
            ledger.checkpoint("protected_postflight_complete")
            if any(not probe["qualifies"] for probe in probes):
                cause = "node_diversity_failed" if any(not probe["not_all_node_identical"] for probe in probes) else "repeat_parity_failed"
                raise CAP0MechanismRejected("CAP0 probe qualification rejected", cause)
            progress["stage"] = "model_release"
            model_identity = {"class": model.__class__.__name__, "hidden_size": 2048, "vocab_size": 248320, "dtype": str(next(model.parameters()).dtype), "device": str(next(model.parameters()).device), "checkpoint": E33_PATH, "checkpoint_tree_sha256": E33_TREE_SHA256}
            del selected, bank, _partition, capture, model, tokenizer
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            progress["model_released"] = True
            ledger.checkpoint("model_released")
            protected_state = {"disk_before": protected_before, "disk_after": protected_after, "e33_state_before": state_before, "e33_state_after": state_after, "e33_all_requires_grad_false": True, "e33_all_grads_none": grads_none, "model_eval": True, "h176_loaded": False, "model_released": True}
            tracker.exit("completed")
            tracker.enter("audit", RESOURCE_BOUNDS["audit_timeout_seconds"])
            tamper_bank, tamper_partition, tamper_selection, tamper_capture = load_train_inputs(repo)
            tamper_results = run_tampers(contract, tamper_selection, tamper_capture, plan, tamper_bank, tamper_partition)
            del tamper_bank, tamper_partition, tamper_capture
            freeze_after = freeze_state(repo, plan, args.execution_commit)
            if not freeze_exact(freeze_after, plan, args.execution_commit) or freeze_before["tree"] != freeze_after["tree"]:
                raise InfrastructureInvalid("CAP0 postflight freeze differs")
            ledger.checkpoint("full_freeze_postflight_validated")
            asset_after = asset_hashes(repo, plan)
            asset_before = freeze_before["asset_hashes"]
            ledger.checkpoint("proof_prewrite_ready")
            progress["memory_rows_completed"] = len(ledger.rows)
            tracker.exit("completed")
            tracker.enter("terminal_publication", RESOURCE_BOUNDS["terminal_timeout_seconds"])
            terminal, elapsed = tracker.terminal()
            aggregate = {"complete_modality_count": 4, "qualifying_modality_count": 4, "complete_probe_count": 4, "qualifying_probe_count": 4, "tokenizer_calls": 4, "model_forwards": 8, "sequences": 192, "all_qualify": True}
            resources = {"bounds": RESOURCE_BOUNDS, "gpu_name": properties.name, "gpu_total_bytes": properties.total_memory, "host_ram_bytes": host_ram_bytes(), "free_disk_bytes_preflight": free_disk_preflight, "free_disk_bytes_postflight": shutil.disk_usage(Path(OUTPUT_ROOT).parent).free, "global_max_allocated_bytes": torch.cuda.max_memory_allocated(0), "global_max_reserved_bytes": torch.cuda.max_memory_reserved(0), "artifact_bytes_before_terminal": 0, "completed_phase_records": copy.deepcopy(tracker.completed), "final_terminal_publication": terminal, "prepublication_elapsed_ns": elapsed}
            safety = {"cuda_visible_devices": "0", "network_attempts": network.attempts, "validation_opens": firewall.validation_open_count, "heldout_opens": firewall.heldout_open_count, "generation_calls": 0, "backwards": 0, "optimizer_objects": 0, "optimizer_steps": 0, "candidate_objects": 0, "candidate_files": [], "checkpoint_files": [], "model_updated": False, "h176_loaded": False, "network_guard": network.evidence(), "experiment_open_firewall": firewall.evidence()}
            proof: dict[str, Any] = {"schema_version": PROOF_SCHEMA, "status": PROOF_STATUS, "mechanism": MECHANISM, "run_identity": RUN_ID, "execution_commit": args.execution_commit, "mechanism_code_commit": plan["mechanism_code_commit"], "plan_file_sha256": args.plan_file_sha256, "plan_sha256": plan["plan_sha256"], "runtime": runtime, "asset_audit": {"before": asset_before, "after": asset_after, "equal": asset_before == asset_after == plan["asset_sha256"]}, "mf0_archive_binding": MF0_BINDING, "selection": {"selection_sha256": SELECTION_SHA256, "ordered_probes": selection["ordered_probes"]}, "model_identity": model_identity, "probes": probes, "aggregate": aggregate, "cache_guard": guard.evidence(), "protected_state": protected_state, "safety": safety, "counts": COUNTS, "resources": resources, "memory": {"labels": MEMORY_LABELS, "label_sha256": sha256_bytes(canonical_json(MEMORY_LABELS)), "rows": ledger.rows}, "full_freeze": {"head_before": freeze_before["head"], "head_after": freeze_after["head"], "parent": freeze_before["parent"], "tree_before": freeze_before["tree"], "tree_after": freeze_after["tree"], "status_before": freeze_before["status"], "status_after": freeze_after["status"], "assets_equal": asset_before == asset_after}, "tamper_audit": {"results": [*tamper_results, {"name": TAMPERS[40], "rejected": True, "error_type": "CAP0ContractError"}, {"name": TAMPERS[41], "rejected": True, "error_type": "CAP0ContractError"}], "rejected_count": 42}, "decision_boundary": DECISION, "proof_sha256": ""}
            proof["proof_sha256"] = sha256_bytes(canonical_json({key: value for key, value in proof.items() if key != "proof_sha256"}))
            validate_proof(proof, plan=plan, contract=contract, selection=selection, execution_commit=args.execution_commit, plan_file_sha256=args.plan_file_sha256)
            for name in TAMPERS[40:]:
                changed = copy.deepcopy(proof)
                if name == "proof_status_changed":
                    changed["status"] = REJECT_STATUS
                    changed["proof_sha256"] = sha256_bytes(canonical_json({key: value for key, value in changed.items() if key != "proof_sha256"}))
                else:
                    changed["proof_sha256"] = "0" * 64
                try:
                    validate_proof(changed, plan=plan, contract=contract, selection=selection, execution_commit=args.execution_commit, plan_file_sha256=args.plan_file_sha256)
                except CAP0ContractError:
                    pass
                else:
                    raise CAP0ContractError(f"CAP0 receipt tamper accepted: {name}")
            writer.write("CAP0-PROOF.json", proof)
            signal.alarm(0)
    except BaseException as original_error:
        signal.alarm(0)
        if writer.terminal_written:
            raise
        if tracker.active is not None:
            tracker.exit("error")
        tracker.enter("failure_audit", RESOURCE_BOUNDS["failure_timeout_seconds"])
        plan = load_authorized_plan(repo, args.execution_commit, args.plan_file_sha256)
        if contract is None or selection is None:
            selection = load_authorized_canonical(repo, args.execution_commit, PLAN_ASSET_PATHS[8], plan["asset_sha256"][PLAN_ASSET_PATHS[8]])
            capture = load_authorized_canonical(repo, args.execution_commit, PLAN_ASSET_PATHS[9], plan["asset_sha256"][PLAN_ASSET_PATHS[9]])
            contract = load_authorized_canonical(repo, args.execution_commit, CONTRACT_RELATIVE_PATH, plan["asset_sha256"][CONTRACT_RELATIVE_PATH])
            validate_contract(contract, selection, capture)
        if isinstance(original_error, CAP0MechanismRejected):
            status_value, error = REJECT_STATUS, original_error
            cause = original_error.cause
        elif isinstance(original_error, ExposureBoundaryRejected):
            status_value, error, cause = EXPOSURE_STATUS, original_error, None
        elif isinstance(original_error, (InfrastructureInvalid, TimeoutError, MemoryError, OSError, subprocess.CalledProcessError, ImportError)) or torch is not None and isinstance(original_error, torch.OutOfMemoryError):
            status_value, error, cause = "infrastructure_invalid", original_error, None
        elif isinstance(original_error, CAP0ContractError):
            status_value, error, cause = INCOMPLETE_STATUS, original_error, None
        else:
            status_value, error, cause = INCOMPLETE_STATUS, CAP0ContractError(f"unexpected CAP0 diagnostic: {type(original_error).__name__}: {original_error}"), None
        tracker.exit("completed")
        tracker.enter("terminal_publication", RESOURCE_BOUNDS["terminal_timeout_seconds"])
        terminal, elapsed = tracker.terminal()
        cache_partial = None if guard is None else guard.evidence()
        progress["cache_checks_completed"] = 0 if guard is None else len(guard.labels)
        progress["memory_rows_completed"] = 0 if ledger is None else len(ledger.rows)
        candidate_files = [path.name for path in writer.output.iterdir() if "candidate" in path.name]
        checkpoint_files = [path.name for path in writer.output.iterdir() if "checkpoint" in path.name]
        disk_now = None
        with contextlib.suppress(BaseException):
            disk_now = protected_disk()
        state_now = state_after
        grads_present = grads_none_after is False
        if model is not None and torch is not None:
            with contextlib.suppress(BaseException):
                state_now = module_tree_sha256(torch, model)
                grads_present = any(parameter.grad is not None for parameter in model.parameters())
        actual = {"validation_opens": firewall.validation_open_count, "heldout_opens": firewall.heldout_open_count, "h176_loaded": False, "generation_calls": 0, "backwards": 0, "optimizer_objects": 0, "candidate_objects": 0, "network_attempts": network.attempts, "e33_grads_present": grads_present, "e33_state_changed": state_before is not None and state_now is not None and state_before != state_now, "positive_exposure": False, "network_guard": network.evidence(), "experiment_open_firewall": firewall.evidence()}
        actual["positive_exposure"] = bool(any(actual[key] for key in ("validation_opens", "heldout_opens", "h176_loaded", "generation_calls", "backwards", "optimizer_objects", "candidate_objects", "network_attempts", "e33_grads_present", "e33_state_changed")) or firewall.denied_count or candidate_files or checkpoint_files)
        if actual["positive_exposure"] and status_value != EXPOSURE_STATUS:
            status_value, error = EXPOSURE_STATUS, ExposureBoundaryRejected("CAP0 positive forbidden exposure observed")
        elif status_value == EXPOSURE_STATUS and not actual["positive_exposure"]:
            status_value, error = "infrastructure_invalid", InfrastructureInvalid("CAP0 exposure rejection lacked observable evidence")
        protected_expected = {"e33_tree_sha256": E33_TREE_SHA256, "h176_tree_sha256": H176_TREE_SHA256, "e33_metadata_sha256": METADATA_SHA256, "h176_metadata_sha256": METADATA_SHA256}
        protected_observation_incomplete = progress["model_loaded"] and (protected_before != protected_expected or disk_now != protected_expected or state_before != E33_STATE_SHA256 or state_now is None or guard is not None and not guard.restored)
        if protected_observation_incomplete and not actual["positive_exposure"]:
            status_value, error = "infrastructure_invalid", InfrastructureInvalid("CAP0 protected failure audit incomplete")
        audit = failure_freeze_state(repo, plan, args.execution_commit)
        if not audit["exact"] and status_value != "infrastructure_invalid":
            status_value, error = "infrastructure_invalid", InfrastructureInvalid("CAP0 failure provenance differs")
        resources = {"bounds": RESOURCE_BOUNDS, "completed_phase_records": copy.deepcopy(tracker.completed), "final_terminal_publication": terminal, "prepublication_elapsed_ns": elapsed}
        error_type = "CAP0MechanismRejected" if status_value == REJECT_STATUS else type(error).__name__
        failure: dict[str, Any] = {"schema_version": FAILURE_SCHEMA, "status": status_value, "mechanism": MECHANISM, "run_identity": RUN_ID, "error_type": error_type, "error": str(error), "traceback": traceback.format_exc(), "execution_commit": args.execution_commit, "mechanism_code_commit": plan["mechanism_code_commit"], "plan_file_sha256": args.plan_file_sha256, "plan_sha256": plan["plan_sha256"], "progress": progress, "partial_probes": probes, "aggregate_partial": {"cause": cause, "probes_completed": len(probes), "nonfinite_observed": cause == "nonfinite_output"}, "cache_guard_partial": cache_partial, "protected_state": {"disk_before": protected_before, "disk_after": disk_now, "e33_state_before": state_before, "e33_state_after": state_now, "grads_present": grads_present, "restoration_attempted": guard is None or guard.restored}, "actual_safety": actual, "resources": resources, "partial_memory": {"labels": MEMORY_LABELS, "rows": [] if ledger is None else ledger.rows}, "full_freeze_failure_audit": audit, "output_inventory_before_failure": sorted(path.name for path in writer.output.iterdir()), "candidate_files": candidate_files, "checkpoint_files": checkpoint_files, "model_updated": actual["e33_state_changed"], "failure_sha256": ""}
        failure["failure_sha256"] = sha256_bytes(canonical_json({key: value for key, value in failure.items() if key != "failure_sha256"}))
        validate_failure(failure, plan=plan, contract=contract, selection=selection, execution_commit=args.execution_commit, plan_file_sha256=args.plan_file_sha256)
        writer.write("CAP0-FAILURE.json", failure)
        signal.alarm(0)
        raise SystemExit(2) from original_error


def validate_published(args: argparse.Namespace) -> None:
    repo = args.repo.resolve(strict=True)
    output = args.output_dir
    if output != Path(OUTPUT_ROOT) or output.is_symlink() or not output.is_dir():
        raise InfrastructureInvalid("CAP0 published namespace differs")
    plan = load_authorized_plan(repo, args.execution_commit, args.plan_file_sha256)
    _, _, selection, capture = load_train_inputs(repo)
    contract = load_canonical(repo / CONTRACT_RELATIVE_PATH)
    validate_contract(contract, selection, capture)
    names = sorted(path.name for path in output.iterdir())
    if names == ["CAP0-PROOF.json"]:
        data = read_regular(output / names[0])
        if not data.endswith(b"\n"):
            raise CAP0ContractError("CAP0 proof framing differs")
        proof = strict_loads(data[:-1])
        if data != canonical_json(proof) + b"\n":
            raise CAP0ContractError("CAP0 proof canonical bytes differ")
        validate_proof(proof, plan=plan, contract=contract, selection=selection, execution_commit=args.execution_commit, plan_file_sha256=args.plan_file_sha256)
    elif names == ["CAP0-FAILURE.json"]:
        data = read_regular(output / names[0])
        if not data.endswith(b"\n"):
            raise CAP0ContractError("CAP0 failure framing differs")
        failure = strict_loads(data[:-1])
        if data != canonical_json(failure) + b"\n":
            raise CAP0ContractError("CAP0 failure canonical bytes differ")
        validate_failure(failure, plan=plan, contract=contract, selection=selection, execution_commit=args.execution_commit, plan_file_sha256=args.plan_file_sha256)
    else:
        raise InfrastructureInvalid("CAP0 terminal inventory differs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--plan-file-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--gpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-terminal", action="store_true")
    args = parser.parse_args()
    if sum((args.preflight_only, args.full, args.validate_terminal)) != 1:
        parser.error("choose exactly one CAP0 mode")
    return args


def main() -> None:
    signal.signal(signal.SIGALRM, _alarm)
    args = parse_args()
    if args.preflight_only:
        if args.gpu is not None or args.output_dir is not None:
            raise InfrastructureInvalid("CAP0 preflight-only arguments differ")
        run_preflight(args)
    elif args.full:
        run_full(args)
    else:
        validate_published(args)


if __name__ == "__main__":
    main()
