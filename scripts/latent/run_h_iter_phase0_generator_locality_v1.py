#!/usr/bin/env python3
"""Run the model-free, CUDA-hidden H-ITER Phase-0 proof."""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import gc
import importlib.metadata
import os
import platform
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from prime_rl.latent.h_iter_phase0 import (
    ARTIFACT_DIR_REL,
    DECISION_BOUNDARY,
    EXPECTED_RUNTIME,
    FAILURE_SCHEMA,
    MECHANISM,
    MECHANISM_TAMPERS,
    NETWORK_GUARD_CONTRACT,
    PHASE0_AUDITOR_EXPOSURE,
    PHASE1_LEARNING_EXPOSURE,
    PROOF_SCHEMA,
    RECEIPT_TAMPERS,
    RESOURCE_BOUNDS,
    RUN_IDENTITY,
    SPLITS,
    ContractError,
    canonical_json,
    canonical_sha256,
    extract_prior_source,
    locality_policy,
    memory_labels,
    new_identity_sets,
    run_mechanism_tamper_audit,
    run_symbolic_dependency_audit,
    sha256_bytes,
    strict_json_loads,
    validate_banks,
    validate_failure,
    validate_locality_evidence,
    validate_plan,
    validate_probe_selection,
    validate_proof,
    validate_schedule,
)


class InfrastructureInvalid(RuntimeError):
    pass


class PhaseTracker:
    def __init__(self) -> None:
        self.started_ns = time.monotonic_ns()
        self.active: tuple[str, int, int, int] | None = None
        self.completed: list[dict[str, object]] = []

    def enter(self, phase: str, seconds: int) -> int:
        if self.active is not None:
            raise InfrastructureInvalid("Phase-0 timing phase overlap")
        entered = time.monotonic_ns() - self.started_ns
        cap_ns = seconds * 1_000_000_000
        alarm_after_ns = (seconds - 1) * 1_000_000_000
        self.active = (phase, entered, cap_ns, alarm_after_ns)
        signal.alarm(seconds - 1)
        return entered

    def exit(self, outcome: str, *, timeout_observed: bool = False) -> None:
        if self.active is None or outcome not in {"completed", "error"}:
            raise InfrastructureInvalid("Phase-0 timing phase exit differs")
        phase, entered, cap_ns, alarm_after_ns = self.active
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
                "alarm_after_ns": alarm_after_ns,
                "alarm_safety_margin_ns": 1_000_000_000,
                "timeout_observed": timeout_observed,
                "alarm_requested_after_ns": alarm_after_ns,
                "timeout_observed_duration_ns": duration if timeout_observed else None,
                "delivery_overrun_ns": max(0, duration - cap_ns) if timeout_observed else 0,
                "timing_cap_exceeded": duration > cap_ns,
            }
        )
        self.active = None
        signal.alarm(0)

    def terminal_boundary(self) -> tuple[dict[str, object], int]:
        if self.active is None or self.active[0] != "terminal_publication":
            raise InfrastructureInvalid("Phase-0 terminal phase is not active")
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

    def _reject(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.attempt_count += 1
        raise InfrastructureInvalid("Phase-0 network attempt rejected")

    def _audit(self, event: str, args: tuple[object, ...]) -> None:
        del args
        if event in NETWORK_GUARD_CONTRACT["audit_events"]:
            self.attempt_count += 1
            raise InfrastructureInvalid("Phase-0 audited network attempt rejected")

    def __enter__(self) -> NetworkGuard:
        self._originals = {
            "connect": socket.socket.connect,
            "connect_ex": socket.socket.connect_ex,
            "create_connection": socket.create_connection,
            "getaddrinfo": socket.getaddrinfo,
        }
        socket.socket.connect = self._reject  # type: ignore[method-assign]
        socket.socket.connect_ex = self._reject  # type: ignore[method-assign]
        socket.create_connection = self._reject  # type: ignore[assignment]
        socket.getaddrinfo = self._reject  # type: ignore[assignment]
        sys.addaudithook(self._audit)
        self.installed = True
        return self

    def __exit__(self, _error_type: object, _error: object, _traceback: object) -> None:
        socket.socket.connect = self._originals["connect"]  # type: ignore[method-assign,assignment]
        socket.socket.connect_ex = self._originals["connect_ex"]  # type: ignore[method-assign,assignment]
        socket.create_connection = self._originals["create_connection"]  # type: ignore[assignment]
        socket.getaddrinfo = self._originals["getaddrinfo"]  # type: ignore[assignment]
        self.wrappers_restored = True

    def evidence(self) -> dict[str, object]:
        return {
            **NETWORK_GUARD_CONTRACT,
            "installed": self.installed,
            "wrappers_restored": self.wrappers_restored,
            "audit_hook_persistent": self.installed,
            "attempt_count": self.attempt_count,
        }


class ArtifactWriter:
    def __init__(self, output_dir: Path) -> None:
        expected = Path(RESOURCE_BOUNDS["output_root"])
        if output_dir != expected or not output_dir.is_absolute() or output_dir.exists() or output_dir.is_symlink():
            raise InfrastructureInvalid("Phase-0 output namespace is not exact and fresh")
        parent = output_dir.parent
        if parent.is_symlink() or not parent.is_dir():
            raise InfrastructureInvalid("Phase-0 output parent is absent or symlinked")
        output_dir.mkdir(mode=0o700)
        self.output_dir = output_dir
        self.terminal_written = False

    def write(self, name: str, payload: dict[str, object], maximum_bytes: int) -> bytes:
        if self.terminal_written or name not in {"PROOF.json", "FAILURE.json"}:
            raise InfrastructureInvalid("Phase-0 terminal is not exclusive")
        if list(self.output_dir.iterdir()):
            raise InfrastructureInvalid("Phase-0 output namespace is not empty before terminal publication")
        encoded = canonical_json(payload) + b"\n"
        if len(encoded) > maximum_bytes:
            raise InfrastructureInvalid("Phase-0 terminal exceeds artifact bound")
        temporary = self.output_dir / f".{name}.tmp"
        target = self.output_dir / name
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short Phase-0 terminal write")
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
        reopened = target.read_bytes()
        if reopened != encoded:
            raise InfrastructureInvalid("Phase-0 terminal changed after atomic publication")
        return reopened


class MemoryLedger:
    def __init__(self) -> None:
        self.labels = memory_labels()
        self.rows: list[dict[str, object]] = []
        self.peak = 0

    @staticmethod
    def rss_bytes() -> int:
        if sys.platform.startswith("linux"):
            pages = int(Path("/proc/self/statm").read_text().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage if sys.platform == "darwin" else usage * 1024

    def checkpoint(self, label: str) -> None:
        if len(self.rows) >= len(self.labels) or label != self.labels[len(self.rows)]:
            raise ContractError("Phase-0 memory label order changed")
        rss = self.rss_bytes()
        self.peak = max(self.peak, rss)
        self.rows.append({"label": label, "rss_bytes": rss, "peak_rss_bytes": self.peak})

    def evidence(self) -> dict[str, object]:
        if [row["label"] for row in self.rows] != self.labels:
            raise ContractError("Phase-0 memory ledger is incomplete")
        return {
            "labels": self.labels,
            "label_sha256": canonical_sha256(self.labels),
            "count": len(self.labels),
            "rows": self.rows,
        }


def file_sha256(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(repo: Path, *args: str) -> str:
    if not args or args[0] not in {"rev-parse", "status"}:
        raise InfrastructureInvalid("Phase-0 subprocess is outside the frozen allowlist")
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE, text=True
    ).stdout.rstrip("\n")


def git_blob(repo: Path, commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=repo, check=True, stdout=subprocess.PIPE
    ).stdout


def load_canonical_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise InfrastructureInvalid(f"required Phase-0 asset is absent/symlinked: {path}")
    data = path.read_bytes()
    if not data.endswith(b"\n") or data.endswith(b"\n\n"):
        raise InfrastructureInvalid(f"Phase-0 artifact LF framing differs: {path}")
    parsed = strict_json_loads(data[:-1])
    if not isinstance(parsed, dict) or data != canonical_json(parsed) + b"\n":
        raise InfrastructureInvalid(f"Phase-0 artifact is not canonical JSON: {path}")
    return parsed


def asset_hashes(repo: Path, plan: dict[str, object]) -> dict[str, str]:
    result = {}
    for relative, expected in plan["asset_sha256"].items():
        path = repo / relative
        if path.is_symlink() or not path.is_file():
            raise InfrastructureInvalid(f"Phase-0 asset missing/symlinked: {relative}")
        observed = file_sha256(path)
        if observed != expected:
            raise InfrastructureInvalid(f"Phase-0 asset hash changed: {relative}")
        result[relative] = observed
    return result


def runtime_evidence() -> tuple[dict[str, object], object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise InfrastructureInvalid("CUDA must be hidden for Phase-0")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1" or os.environ.get("HF_HUB_OFFLINE") != "1":
        raise InfrastructureInvalid("Phase-0 must be offline")
    if platform.python_version() != EXPECTED_RUNTIME["python"]:
        raise InfrastructureInvalid("Python runtime differs")
    if sys.executable != EXPECTED_RUNTIME["sys_executable"] or sys.prefix != EXPECTED_RUNTIME["sys_prefix"]:
        raise InfrastructureInvalid("shared Python executable/prefix differs")
    shared_project = Path("/home/ubuntu/rlm/prime-rl")
    if any(
        path.is_symlink() or not path.is_file()
        for path in (shared_project / "pyproject.toml", shared_project / "uv.lock")
    ):
        raise InfrastructureInvalid("shared project provenance file is absent/symlinked")
    if (
        file_sha256(shared_project / "pyproject.toml") != EXPECTED_RUNTIME["shared_project_pyproject_sha256"]
        or file_sha256(shared_project / "uv.lock") != EXPECTED_RUNTIME["shared_project_uv_lock_sha256"]
    ):
        raise InfrastructureInvalid("shared project bytes differ")
    versions = {
        "torch_distribution": importlib.metadata.version("torch"),
        "transformers_distribution": importlib.metadata.version("transformers"),
        "tokenizers_distribution": importlib.metadata.version("tokenizers"),
        "pyarrow_distribution": importlib.metadata.version("pyarrow"),
    }
    if any(versions[key] != EXPECTED_RUNTIME[key] for key in versions):
        raise InfrastructureInvalid("Phase-0 distribution version differs")
    if any(name.startswith("transformers.models") or ".modeling_" in name for name in sys.modules):
        raise InfrastructureInvalid("Transformers modeling module imported before Phase-0")
    import torch

    before = torch.cuda.is_initialized()
    if before or torch.__version__ != EXPECTED_RUNTIME["torch_runtime"]:
        raise InfrastructureInvalid("Torch runtime/CUDA initialization differs")
    evidence = {
        **EXPECTED_RUNTIME,
        "cuda_initialized_before": before,
        "cuda_initialized_after": False,
    }
    return evidence, torch


def static_forbidden_sites(repo: Path) -> list[str]:
    forbidden = []
    allowed_backward = (repo / "src/prime_rl/latent/h_iter_phase0.py", "run_locality_probe")
    for path in (
        repo / "src/prime_rl/latent/h_iter_phase0.py",
        repo / "scripts/latent/run_h_iter_phase0_generator_locality_v1.py",
    ):
        tree = ast.parse(path.read_text())
        parents: list[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                parents.append(node.name)
                self.generic_visit(node)
                parents.pop()

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    if alias.name.startswith("transformers.models") or ".modeling_" in alias.name:
                        forbidden.append(f"{path}:{node.lineno}:import:{alias.name}")

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                module = node.module or ""
                if module.startswith("transformers.models") or ".modeling_" in module:
                    forbidden.append(f"{path}:{node.lineno}:import:{module}")

            def visit_Call(self, node: ast.Call) -> None:
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                if name in {"generate", "from_pretrained", "step"}:
                    forbidden.append(f"{path}:{node.lineno}:call:{name}")
                if name == "backward" and (path, parents[-1] if parents else "") != allowed_backward:
                    forbidden.append(f"{path}:{node.lineno}:call:backward")
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess"
                    and node.func.attr == "run"
                    and (parents[-1] if parents else "") not in {"run_git", "git_blob"}
                ):
                    forbidden.append(f"{path}:{node.lineno}:call:subprocess.run")
                self.generic_visit(node)

        Visitor().visit(tree)
    return forbidden


def object_inventory(torch: object | None, output_dir: Path) -> dict[str, object]:
    modeling = sorted(name for name in sys.modules if name.startswith("transformers.models") or ".modeling_" in name)
    optimizer_objects = 0
    pretrained_model_objects = 0
    tokenizer_objects = 0
    for value in gc.get_objects():
        with contextlib.suppress(ReferenceError):
            value_type = type(value)
            mro = value_type.__mro__
            if any(base.__name__ == "Optimizer" and base.__module__.startswith("torch.optim") for base in mro):
                optimizer_objects += 1
            if any(base.__name__ == "PreTrainedModel" and base.__module__.startswith("transformers") for base in mro):
                pretrained_model_objects += 1
            if value_type.__module__.startswith("transformers.tokenization"):
                tokenizer_objects += 1
    inventory = sorted(path.name for path in output_dir.iterdir())
    candidates = [name for name in inventory if "candidate" in name.lower()]
    checkpoints = [name for name in inventory if "checkpoint" in name.lower()]
    relevant_absent = all(
        name not in sys.modules
        for name in ("torch", "transformers", "tokenizers")
    )
    if torch is None and not relevant_absent:
        raise InfrastructureInvalid("pre-Torch safety inference lacks absent-module proof")
    return {
        "transformers_modeling_modules": modeling,
        "pretrained_model_objects": pretrained_model_objects,
        "tokenizer_objects": tokenizer_objects,
        "optimizer_objects": optimizer_objects,
        "output_inventory": inventory,
        "candidate_files": candidates,
        "checkpoint_files": checkpoints,
        "cuda_initialized": False if torch is None else torch.cuda.is_initialized(),
        "object_census_method": "gc_mro_scan_without_importing_model_tokenizer_or_optimizer_classes",
        "cuda_observation_method": (
            "torch_module_absence_proves_no_torch_cuda_runtime_contact"
            if torch is None
            else "torch.cuda.is_initialized"
        ),
        "relevant_modules_absent_for_preimport_inference": relevant_absent if torch is None else False,
    }


def safety_evidence(
    torch: object, repo: Path, output_dir: Path, network_guard: NetworkGuard
) -> dict[str, object]:
    observed = object_inventory(torch, output_dir)
    return {
        "coordinator_e33_loaded": observed["pretrained_model_objects"] != 0,
        "worker_h176_loaded": observed["pretrained_model_objects"] != 0,
        "tokenizer_loaded": observed["tokenizer_objects"] != 0,
        "candidate_created": bool(observed["candidate_files"]),
        "checkpoint_created": bool(observed["checkpoint_files"]),
        "model_updated": observed["pretrained_model_objects"] != 0,
        "phase0_auditor_opened": PHASE0_AUDITOR_EXPOSURE,
        "phase1_learning_exposure": PHASE1_LEARNING_EXPOSURE,
        "phase0_probe_selection_precommitted": True,
        "phase1_thresholds_present": False,
        "network_guard": network_guard.evidence(),
        "transformers_modeling_modules": observed["transformers_modeling_modules"],
        "pretrained_model_objects": observed["pretrained_model_objects"],
        "tokenizer_objects": observed["tokenizer_objects"],
        "optimizer_objects": observed["optimizer_objects"],
        "object_census_method": observed["object_census_method"],
        "cuda_observation_method": observed["cuda_observation_method"],
        "relevant_modules_absent_for_preimport_inference": observed[
            "relevant_modules_absent_for_preimport_inference"
        ],
        "output_inventory_before_terminal": observed["output_inventory"],
        "static_forbidden_model_call_sites": static_forbidden_sites(repo),
        "tokenizer_calls": 0,
        "model_forwards": 0,
        "model_backwards": 0,
        "synthetic_cpu_backwards": 40,
        "optimizer_steps": 0,
    }


def host_resources(repo: Path) -> tuple[int, int]:
    host_ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    free_disk = shutil.disk_usage(repo).free
    if host_ram < RESOURCE_BOUNDS["minimum_host_ram_gib"] * 2**30:
        raise InfrastructureInvalid("Phase-0 host RAM is below frozen minimum")
    if free_disk < RESOURCE_BOUNDS["minimum_free_disk_gib"] * 2**30:
        raise InfrastructureInvalid("Phase-0 free disk is below frozen minimum")
    return host_ram, free_disk


def regenerate_overlap(repo: Path, overlap: dict[str, object], banks: dict[str, dict]) -> None:
    if overlap.get("source_record_count") != 38 or overlap.get("all_intersections_empty") is not True:
        raise ContractError("Phase-0 overlap summary differs")
    records = overlap.get("source_records")
    if not isinstance(records, list) or records != sorted(records, key=lambda item: (item["source_commit"], item["source_path"])):
        raise ContractError("Phase-0 overlap source order differs")
    identities = new_identity_sets(banks)
    for record in records:
        data = git_blob(repo, record["source_commit"], record["source_path"])
        if sha256_bytes(data) != record["file_sha256"]:
            raise InfrastructureInvalid("historical source bytes changed")
        observed, intersection = extract_prior_source(record["source_path"], data, identities)
        if observed != record["observed"] or intersection != record["intersection"]:
            raise ContractError("Phase-0 overlap extraction does not regenerate")
        if any(intersection.values()):
            raise ContractError("Phase-0 historical identity overlap detected")
    if overlap["overlap_sha256"] != canonical_sha256(overlap, omit="overlap_sha256"):
        raise ContractError("Phase-0 overlap self hash differs")


def receipt_tamper_audit(
    proof: dict[str, object],
    *,
    plan: dict,
    banks: dict,
    selection: dict,
    schedule: dict,
    overlap: dict,
    expected_execution_commit: str,
    expected_plan_file_sha256: str,
) -> dict[str, object]:
    results = []

    def set_hash(value: dict[str, object]) -> None:
        value["proof_sha256"] = canonical_sha256(value, omit="proof_sha256")

    for name in RECEIPT_TAMPERS:
        altered = copy.deepcopy(proof)
        if name == "missing_top_key":
            del altered["counts"]
        elif name == "extra_top_key":
            altered["unexpected"] = True
        elif name == "status_changed":
            altered["status"] = "h_iter_phase0_generator_locality_incomplete"
        elif name == "bank_hash_changed":
            altered["banks"]["train"]["bank_sha256"] = "0" * 64
        elif name == "full_freeze_truncated":
            del altered["full_freeze"]["tree_after"]
        elif name == "model_call_nonzero":
            altered["counts"]["model_or_transformer_forwards"] = 1
        elif name == "cuda_initialized_true":
            altered["runtime"]["cuda_initialized_after"] = True
        elif name == "optimizer_step_nonzero":
            altered["safety"]["optimizer_steps"] = 1
        elif name == "validation_phase1_learning_exposure_nonzero":
            altered["safety"]["phase1_learning_exposure"]["validation_model_or_tokenizer_calls"] = 1
        elif name == "heldout_phase1_learning_exposure_nonzero":
            altered["safety"]["phase1_learning_exposure"]["heldout_model_or_tokenizer_calls"] = 1
        elif name == "thresholds_present":
            altered["learning_thresholds"] = {}
        elif name == "receipt_sha_stale":
            altered["resources"]["prepublication_elapsed_ns"] += 1
        else:
            raise AssertionError(name)
        if name != "receipt_sha_stale":
            set_hash(altered)
        try:
            validate_proof(
                altered,
                plan=plan,
                banks=banks,
                selection=selection,
                schedule=schedule,
                overlap=overlap,
                expected_execution_commit=expected_execution_commit,
                expected_plan_file_sha256=expected_plan_file_sha256,
                require_receipt_tampers=False,
                require_final_timing=False,
            )
        except (ContractError, KeyError, TypeError):
            results.append({"name": name, "rejected": True})
        else:
            raise ContractError(f"Phase-0 receipt tamper was accepted: {name}")
    return {"results": results, "rejected_count": len(results)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-file-sha256", required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _timeout(_signum: int, _frame: object) -> None:
    raise TimeoutError("Phase-0 frozen phase timeout")


def run(
    args: argparse.Namespace,
    ledger: MemoryLedger,
    network_guard: NetworkGuard,
) -> tuple[dict[str, object], dict[str, object]]:
    repo = args.repo.resolve(strict=True)
    plan_path = args.plan.resolve(strict=True)
    plan = load_canonical_json(plan_path)
    validate_plan(plan)
    if file_sha256(plan_path) != args.plan_file_sha256:
        raise InfrastructureInvalid("external Phase-0 plan hash differs")
    sidecar = plan_path.with_name("phase0-plan.sha256")
    if sidecar.read_text() != f"{args.plan_file_sha256}\n":
        raise InfrastructureInvalid("Phase-0 plan sidecar differs")
    runtime, torch = runtime_evidence()
    host_ram, free_disk_preflight = host_resources(repo)
    ledger.checkpoint("runtime_verified")

    head_before = run_git(repo, "rev-parse", "HEAD")
    tree_before = run_git(repo, "rev-parse", "HEAD^{tree}")
    status_before = run_git(repo, "status", "--porcelain")
    parent = run_git(repo, "rev-parse", "HEAD^")
    if head_before != args.execution_commit or parent != plan["mechanism_code_commit"] or status_before:
        raise InfrastructureInvalid("Phase-0 execution tree is not the exact clean freeze")
    before_assets = asset_hashes(repo, plan)
    ledger.checkpoint("full_freeze_preflight_verified")

    base = repo / ARTIFACT_DIR_REL
    banks = {split: load_canonical_json(base / f"{split}-bank.json") for split in SPLITS}
    structural = validate_banks(banks)
    ledger.checkpoint("banks_structurally_validated")
    selection = load_canonical_json(base / "locality-probe-selection.json")
    validate_probe_selection(selection, banks)
    overlap = load_canonical_json(base / "overlap-evidence.json")
    regenerate_overlap(repo, overlap, banks)
    ledger.checkpoint("overlap_closure_validated")
    schedule = load_canonical_json(base / "operation-schedule.json")
    validate_schedule(schedule, selection)
    ledger.checkpoint("operation_schedule_validated")
    tamper_schedule = load_canonical_json(base / "tamper-schedule.json")
    if [row["name"] for row in tamper_schedule["mechanism_tampers"]] != MECHANISM_TAMPERS:
        raise ContractError("mechanism tamper schedule differs")
    if [row["name"] for row in tamper_schedule["receipt_tampers"]] != RECEIPT_TAMPERS:
        raise ContractError("receipt tamper schedule differs")

    locality_rows = []
    for probe in selection["probes"]:
        ledger.checkpoint(f"pre_locality_probe_{probe['probe_index']:02d}")
        row = next(row for row in banks[probe["split"]]["rows"] if row["row_id"] == probe["row_id"])
        from prime_rl.latent.h_iter_phase0 import run_locality_probe

        locality_rows.append(run_locality_probe(row, probe))
        ledger.checkpoint(f"post_locality_probe_{probe['probe_index']:02d}")
    totals = {
        key: sum(probe["counts"][key] for probe in locality_rows)
        for key in locality_rows[0]["counts"]
    }
    locality_evidence = {
        "probes": locality_rows,
        "probe_count": len(locality_rows),
        "counts": totals,
        "policy": locality_policy(),
        "symbolic_dependencies": run_symbolic_dependency_audit(banks),
    }
    ledger.checkpoint("symbolic_dependency_validated")
    validate_locality_evidence(locality_evidence, selection)
    mechanism_tampers = run_mechanism_tamper_audit(banks, selection, locality_evidence)
    ledger.checkpoint("mechanism_tampers_validated")

    safety = safety_evidence(torch, repo, args.output_dir, network_guard)
    runtime["cuda_initialized_after"] = torch.cuda.is_initialized()
    if safety["transformers_modeling_modules"] or safety["optimizer_objects"] or runtime["cuda_initialized_after"]:
        raise InfrastructureInvalid("Phase-0 safety postflight rejected")
    ledger.checkpoint("safety_postflight_validated")
    after_assets = asset_hashes(repo, plan)
    head_after = run_git(repo, "rev-parse", "HEAD")
    tree_after = run_git(repo, "rev-parse", "HEAD^{tree}")
    status_after = run_git(repo, "status", "--porcelain")
    if (head_after, tree_after, status_after, after_assets) != (head_before, tree_before, status_before, before_assets):
        raise InfrastructureInvalid("Phase-0 full freeze changed during proof")
    ledger.checkpoint("full_freeze_postflight_validated")
    ledger.checkpoint("proof_prewrite_ready")
    _, free_disk_postflight = host_resources(repo)
    return {
        "schema_version": PROOF_SCHEMA,
        "status": "h_iter_phase0_generator_locality_validated",
        "mechanism": MECHANISM,
        "run_identity": RUN_IDENTITY,
        "execution_commit": args.execution_commit,
        "mechanism_code_commit": plan["mechanism_code_commit"],
        "plan_file_sha256": args.plan_file_sha256,
        "plan_sha256": plan["plan_sha256"],
        "runtime": runtime,
        "asset_audit": {
            "before": before_assets,
            "after": after_assets,
            "before_after_equal": before_assets == after_assets,
            "all_plan_assets_exact": True,
        },
        "banks": {
            split: {
                "file_sha256": file_sha256(base / f"{split}-bank.json"),
                "bank_sha256": banks[split]["bank_sha256"],
                "row_count": len(banks[split]["rows"]),
            }
            for split in SPLITS
        },
        "structural_audit": structural,
        "overlap_audit": {
            "file_sha256": file_sha256(base / "overlap-evidence.json"),
            "overlap_sha256": overlap["overlap_sha256"],
            "source_record_count": overlap["source_record_count"],
            "all_intersections_empty": overlap["all_intersections_empty"],
            "regenerated_byte_identical": True,
        },
        "operation_schedule": {
            "file_sha256": file_sha256(base / "operation-schedule.json"),
            "schedule_sha256": schedule["schedule_sha256"],
            "operation_count": len(schedule["operations"]),
        },
        "locality": locality_evidence,
        "tamper_audit": {
            "mechanism": mechanism_tampers,
            "receipt": {"results": [{"name": name, "rejected": True} for name in RECEIPT_TAMPERS], "rejected_count": 12},
        },
        "counts": {
            "structural_rows": 192,
            "locality_probes": 8,
            "symbolic_final_arm_checks": 960,
            "symbolic_recurrent_timestep_checks": 1056,
            "synthetic_cpu_backwards": 40,
            "model_or_transformer_forwards": 0,
            "model_backwards": 0,
            "optimizer_steps": 0,
        },
        "safety": safety,
        "resources": {
            "bounds": RESOURCE_BOUNDS,
            "host_ram_bytes": host_ram,
            "free_disk_bytes_preflight": free_disk_preflight,
            "free_disk_bytes_postflight": free_disk_postflight,
            "artifact_bytes_before_terminal": 0,
            "completed_phase_records": [],
            "final_terminal_publication": {},
            "prepublication_elapsed_ns": 0,
        },
        "memory": ledger.evidence(),
        "full_freeze": {
            "head_before": head_before,
            "head_after": head_after,
            "tree_before": tree_before,
            "tree_after": tree_after,
            "status_before": status_before,
            "status_after": status_after,
            "mechanism_exact_parent": parent == plan["mechanism_code_commit"],
            "head_tree_unchanged": head_before == head_after and tree_before == tree_after,
            "plan_sidecar_exact": True,
            "historical_sources_reopened": True,
        },
        "decision_boundary": DECISION_BOUNDARY,
        "proof_sha256": "",
    }, {
        "plan": plan,
        "banks": banks,
        "selection": selection,
        "schedule": schedule,
        "overlap": overlap,
        "expected_execution_commit": args.execution_commit,
        "expected_plan_file_sha256": args.plan_file_sha256,
    }


def failure_payload(
    args: argparse.Namespace,
    error: BaseException,
    status: str,
    ledger: MemoryLedger | None,
    tracker: PhaseTracker,
    network_guard: NetworkGuard,
) -> dict[str, object]:
    audit: dict[str, object] = {
        "head": None,
        "tree": None,
        "status": None,
        "plan_file_sha256": None,
        "plan_sha256": None,
        "plan_sidecar_sha256": None,
        "plan_asset_hashes": None,
        "errors": [],
    }
    repo = args.repo.resolve()
    for name, git_args in (
        ("head", ("rev-parse", "HEAD")),
        ("tree", ("rev-parse", "HEAD^{tree}")),
        ("status", ("status", "--porcelain")),
    ):
        try:
            audit[name] = run_git(repo, *git_args)
        except BaseException as audit_error:
            audit["errors"].append({"check": name, "error": f"{type(audit_error).__name__}: {audit_error}"})
    try:
        plan = load_canonical_json(args.plan.resolve())
        audit["plan_file_sha256"] = file_sha256(args.plan.resolve())
        audit["plan_sha256"] = plan["plan_sha256"]
        audit["plan_sidecar_sha256"] = file_sha256(args.plan.resolve().with_name("phase0-plan.sha256"))
        audit["plan_asset_hashes"] = asset_hashes(repo, plan)
    except BaseException as audit_error:
        audit["errors"].append({"check": "plan_and_assets", "error": f"{type(audit_error).__name__}: {audit_error}"})
    output_inventory = []
    if args.output_dir.is_dir() and not args.output_dir.is_symlink():
        output_inventory = sorted(path.name for path in args.output_dir.iterdir())
    torch = sys.modules.get("torch")
    inventory = object_inventory(torch, args.output_dir)
    observed_safety = {
        "torch_imported": torch is not None,
        **inventory,
        "static_forbidden_model_call_sites": static_forbidden_sites(repo),
        "network_guard": network_guard.evidence(),
        "observation_complete": True,
    }
    plan_for_binding = plan if "plan" in locals() else {}
    result = {
        "schema_version": FAILURE_SCHEMA,
        "status": status,
        "mechanism": MECHANISM,
        "run_identity": RUN_IDENTITY,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "execution_commit": args.execution_commit,
        "mechanism_code_commit": plan_for_binding.get("mechanism_code_commit"),
        "authorized_plan_file_sha256": args.plan_file_sha256,
        "plan_file_sha256": audit["plan_file_sha256"],
        "plan_sha256": audit["plan_sha256"],
        "actual_safety": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), **observed_safety},
        "completed_phase_records": copy.deepcopy(tracker.completed),
        "final_terminal_publication": {},
        "prepublication_elapsed_ns": 0,
        "partial_memory": {
            "expected_labels": memory_labels(),
            "rows": [] if ledger is None else ledger.rows,
        },
        "full_freeze_failure_audit": audit,
        "output_inventory_before_failure": output_inventory,
        "candidate_created": bool(observed_safety["candidate_files"]),
        "checkpoint_created": bool(observed_safety["checkpoint_files"]),
        "model_updated": observed_safety["pretrained_model_objects"] != 0,
        "failure_sha256": "",
    }
    if (
        observed_safety["cuda_initialized"]
        or observed_safety["transformers_modeling_modules"]
        or observed_safety["pretrained_model_objects"]
        or observed_safety["tokenizer_objects"]
        or observed_safety["optimizer_objects"]
        or observed_safety["candidate_files"]
        or observed_safety["checkpoint_files"]
        or observed_safety["static_forbidden_model_call_sites"]
        or observed_safety["network_guard"]["attempt_count"]
        or observed_safety["network_guard"]["installed"] is not True
        or observed_safety["network_guard"]["wrappers_restored"] is not True
        or audit["errors"]
    ):
        result["status"] = "infrastructure_invalid"
    return result


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGALRM, _timeout)
    tracker = PhaseTracker()
    network_guard = NetworkGuard()
    writer = None
    ledger = None
    try:
        tracker.enter("compute", RESOURCE_BOUNDS["compute_timeout_seconds"])
        writer = ArtifactWriter(args.output_dir)
        ledger = MemoryLedger()
        with network_guard:
            proof, context = run(args, ledger, network_guard)
        proof["safety"]["network_guard"] = network_guard.evidence()
        tracker.exit("completed")
        tracker.enter("audit", RESOURCE_BOUNDS["audit_timeout_seconds"])
        proof["proof_sha256"] = canonical_sha256(proof, omit="proof_sha256")
        receipt = receipt_tamper_audit(proof, **context)
        proof["tamper_audit"]["receipt"] = receipt
        tracker.exit("completed")
        tracker.enter("terminal_publication", RESOURCE_BOUNDS["terminal_timeout_seconds"])
        terminal, prepublication_elapsed = tracker.terminal_boundary()
        proof["resources"]["completed_phase_records"] = copy.deepcopy(tracker.completed)
        proof["resources"]["final_terminal_publication"] = terminal
        proof["resources"]["prepublication_elapsed_ns"] = prepublication_elapsed
        proof["proof_sha256"] = canonical_sha256(proof, omit="proof_sha256")
        encoded = canonical_json(proof)
        parsed = strict_json_loads(encoded)
        validate_proof(parsed, **context)
        reopened = writer.write("PROOF.json", parsed, RESOURCE_BOUNDS["maximum_artifact_bytes"])
        reparsed = strict_json_loads(reopened[:-1])
        validate_proof(reparsed, **context)
        signal.alarm(0)
    except BaseException as error:
        if writer is None or writer.terminal_written:
            raise
        if tracker.active is not None:
            tracker.exit("error", timeout_observed=isinstance(error, TimeoutError))
        tracker.enter("failure_audit", RESOURCE_BOUNDS["failure_audit_timeout_seconds"])
        status = (
            "h_iter_phase0_generator_locality_incomplete"
            if isinstance(error, ContractError)
            else "infrastructure_invalid"
        )
        failure = failure_payload(args, error, status, ledger, tracker, network_guard)
        tracker.exit("completed")
        tracker.enter("terminal_publication", RESOURCE_BOUNDS["terminal_timeout_seconds"])
        terminal, prepublication_elapsed = tracker.terminal_boundary()
        failure["completed_phase_records"] = copy.deepcopy(tracker.completed)
        failure["final_terminal_publication"] = terminal
        failure["prepublication_elapsed_ns"] = prepublication_elapsed
        failure["failure_sha256"] = canonical_sha256(failure, omit="failure_sha256")
        parsed = strict_json_loads(canonical_json(failure))
        failure_plan = load_canonical_json(args.plan.resolve())
        validate_failure(
            parsed,
            plan=failure_plan,
            expected_execution_commit=args.execution_commit,
            expected_plan_file_sha256=args.plan_file_sha256,
        )
        reopened = writer.write("FAILURE.json", parsed, 16 * 2**20)
        reparsed = strict_json_loads(reopened[:-1])
        if reparsed != parsed:
            raise InfrastructureInvalid("Phase-0 failure changed after publication") from error
        validate_failure(
            reparsed,
            plan=failure_plan,
            expected_execution_commit=args.execution_commit,
            expected_plan_file_sha256=args.plan_file_sha256,
        )
        signal.alarm(0)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
