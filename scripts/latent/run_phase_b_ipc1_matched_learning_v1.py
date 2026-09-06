#!/usr/bin/env python3
"""Run the nomination-only B-IPC1 matched in-place-carrier learning screen."""

from __future__ import annotations

import argparse
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
    if expected is None or not str(path).endswith(expected[1]) or identity["module_sha256"] != expected[2] or identity[
        "distribution"
    ] != expected[3]:
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
                _cache_class_identity(cls, hic0=self.hic0)
                for cls in self.hic0.ordered_subclass_closure(self.base)
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
    for name in ("UV_PROJECT_ENVIRONMENT", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
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
        or hic0.get("success_file_sha256")
        != "26dfb2c8942b767c0ca8697cc66eea9c0e0931123aa4ffd9c7426440323245c8"
        or hic0.get("internal_receipt_sha256")
        != "108342e04a3255afedbbf60d6dc8ccf86c5f0d2736b549634edbdeb8febbafc3"
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
    paths = {
        "bank_manifest": Path(plan["bank_manifest"]["path"]),
        "overlap_closure": Path(plan["overlap_closure"]["path"]),
        "runtime_source": Path(plan["runtime_source"]["path"]),
        "generator_source": Path(plan["generator_source"]["path"]),
        "taskset_source": Path(plan["taskset_source"]["path"]),
    }
    expected = {name: plan[name]["sha256"] for name in paths}
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
    if manifest.get("freshness", {}).get("all_zero") is not True or closure.get("overlap_evidence", {}).get(
        "all_zero"
    ) is not True:
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
        value = torch.zeros_like(residual) if zero and residual is not None else torch.zeros_like(
            base_embeddings[:, target, :]
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


def _metric(output: Any, example: dict[str, Any], predictor: int, inputs: dict[str, Any], *, audit: dict[str, Any]) -> dict[str, Any]:
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


def _initialize_modules(torch: Any, LocalDepthCodec: Any, OneShotFeedForwardSidecar: Any, TimestepFreeRecurrentSidecar: Any, smoke: ModuleType):
    torch.manual_seed(INITIALIZATION_SEED)
    torch.cuda.manual_seed_all(INITIALIZATION_SEED)
    template = LocalDepthCodec(2048, 256, SLOTS, initial_receiver_gate=0.001).to(
        device="cuda:0", dtype=torch.bfloat16
    )
    template_state = {name: tensor.detach().clone() for name, tensor in template.state_dict().items()}
    codecs = {
        arm: LocalDepthCodec(2048, 256, SLOTS, initial_receiver_gate=0.001).to(
            device="cuda:0", dtype=torch.bfloat16
        )
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
    if any(float(torch.tanh(codec.receiver_gate.detach()).cpu()) != float(torch.tensor(0.001, dtype=codec.receiver_gate.dtype)) for codec in codecs.values()):
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
    values = [parameter.grad for name, parameter in parameters if name.startswith(prefixes) and parameter.grad is not None]
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
                            parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for _name, parameter in named
                        ),
                        "e33_gradients_absent": all(parameter.grad is None for parameter in model.parameters()),
                    }
                    if not all((gradient["residual"]["finite"], gradient["residual"]["nonzero"], codec_group["finite"], codec_group["nonzero"], gradient["all_named_present_gradients_finite"], gradient["e33_gradients_absent"])) or (
                        sidecar_group is not None and not all((sidecar_group["finite"], sidecar_group["nonzero"]))
                    ):
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
            if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
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
    snapshots: dict[str, Any], *, torch: Any, LocalDepthCodec: Any, OneShotFeedForwardSidecar: Any, TimestepFreeRecurrentSidecar: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    codecs = {
        arm: LocalDepthCodec(2048, 256, SLOTS, initial_receiver_gate=0.001).to(
            device="cuda:0", dtype=torch.bfloat16
        )
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
            metrics["BASE"].append({"task_key": example["task_key"], "action": example["action"], **_reporting_metric(base_metric)})
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
                            "finite_nonzero": bool(torch.isfinite(residual).all()) and bool(torch.count_nonzero(residual)),
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
        payload = {
            "schema_version": "q35-2b-phase-b-ipc1-candidate/v1",
            "arm": arm,
            "codec": b1._cpu_state_dict(codecs[arm]),
            "sidecar": None if arm == "STATIC" else b1._cpu_state_dict(sidecars[arm]),
        }
        sha = b1._exclusive_torch_save(output, name, payload, torch=torch)
        _memory_checkpoint(torch, audit, f"candidate:{arm}:after_write")
        records.append({"name": name, "arm": arm, "sha256": sha, "valid_only_with_terminal": "SUCCESS.json"})
    if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > ARTIFACT_CAP_BYTES:
        raise ResourceContractExceeded("B-IPC1 artifacts exceed 512 MiB")
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
                "sidecar": None
                if arm == "STATIC"
                else smoke._module_tensor_sha256(pre_sidecars[arm], torch),
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
            "restored_pre": [
                {"name": arm, **restored_pre_hashes[arm]} for arm in TRAINING_ARMS
            ],
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
        "render_proofs": [{"name": split, "rows": proofs} for split, proofs in context["render_proofs"].items()],
        "bank_bindings": [
            {"name": bank["split"], "selection_sha256": bank["selection_sha256"], "parquet_sha256": bank["parquet_sha256"]}
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


def validate_success_receipt(
    receipt: dict[str, Any], *, plan: dict[str, Any], execution_commit: str, output_dir: Path
) -> None:
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
        or receipt.get("optimizer_steps") != 12
        or receipt.get("backward_calls") != 147
    ):
        raise PhaseBContractError("B-IPC1 SUCCESS top-level contract differs")
    heldout_open = receipt.get("heldout_opened") is True
    expected_counts = (796, 96, 700) if heldout_open else (532, 72, 460)
    if (receipt.get("model_forwards"), receipt.get("source_forwards"), receipt.get("receiver_forwards")) != expected_counts:
        raise PhaseBContractError("B-IPC1 SUCCESS model-call counts differ")
    training = receipt.get("training")
    evidence = receipt.get("training_evidence")
    if (
        not isinstance(training, list)
        or [record.get("name") for record in training] != list(TRAINING_ARMS)
        or any(len(record.get("updates", [])) != 4 for record in training)
        or any(len(update.get("rows", [])) != 12 for record in training for update in record["updates"])
        or not isinstance(evidence, list)
        or [record.get("name") for record in evidence] != list(TRAINING_ARMS)
        or any(record.get("value", {}).get("optimizer_destroyed") is not True for record in evidence)
    ):
        raise PhaseBContractError("B-IPC1 SUCCESS training evidence differs")
    expected_train_keys = [record["task_key"] for record in load_json_file(Path(_bank_record(plan, "train")["selection_path"]))["selected"]]
    for arm in training:
        observed = [row["task_key"] for update in arm["updates"] for row in update["rows"]]
        if observed != expected_train_keys:
            raise PhaseBContractError(f"B-IPC1 SUCCESS {arm['name']} training order differs")
    mechanism = receipt.get("pre_update_mechanism_gate")
    if not isinstance(mechanism, dict) or [probe.get("selection_index") for probe in mechanism.get("probes", [])] != [
        0,
        1,
        2,
        5,
    ]:
        raise PhaseBContractError("B-IPC1 SUCCESS mechanism probes differ")
    for probe in mechanism["probes"]:
        if probe.get("base_equals_direct_zero") is not True or [record.get("name") for record in probe.get("arms", [])] != list(
            TRAINING_ARMS
        ):
            raise PhaseBContractError("B-IPC1 SUCCESS zero-identity evidence differs")
        if any(record.get("inplace_zero_identity") is not True for record in probe["arms"]):
            raise PhaseBContractError("B-IPC1 SUCCESS arm zero-identity evidence differs")
    if mechanism.get("pre_tensor_hashes") != mechanism.get("post_tensor_hashes"):
        raise PhaseBContractError("B-IPC1 SUCCESS preprobe candidate state changed")
    module_hashes = receipt.get("module_hashes")
    if not isinstance(module_hashes, dict):
        raise PhaseBContractError("B-IPC1 SUCCESS module hashes are absent")
    deltas = module_hashes.get("delta_groups")
    gates = module_hashes.get("receiver_gates")
    if (
        not isinstance(deltas, list)
        or [record.get("name") for record in deltas] != list(TRAINING_ARMS)
        or not isinstance(gates, list)
        or [record.get("name") for record in gates] != list(TRAINING_ARMS)
    ):
        raise PhaseBContractError("B-IPC1 SUCCESS module safety records differ")
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
    for evaluation in (validation,) if heldout is None else (validation, heldout):
        recomputed = _recompute_evaluation(evaluation, module_safety)
        if recomputed["common"] != evaluation.get("common_arm_gates") or recomputed["recurrent"] != evaluation.get(
            "recurrent_gates"
        ):
            raise PhaseBContractError("B-IPC1 SUCCESS evaluation aggregates do not recompute")
    status, nominated_arms = _status_for_evaluations(validation, heldout)
    nomination = receipt.get("nomination")
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
    memory = receipt.get("cuda_memory")
    ledger = memory.get("ledger", []) if isinstance(memory, dict) else []
    expected_memory = build_memory_checkpoint_labels(
        build_model_call_schedule(
            expected_train_keys,
            [record["task_key"] for record in load_json_file(Path(_bank_record(plan, "validation")["selection_path"]))["selected"]],
            [record["task_key"] for record in load_json_file(Path(_bank_record(plan, "heldout")["selection_path"]))["selected"]],
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
    protection = receipt.get("protection")
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
    candidates = receipt.get("candidates")
    if not isinstance(candidates, list) or [record.get("arm") for record in candidates] != list(TRAINING_ARMS):
        raise PhaseBContractError("B-IPC1 SUCCESS candidate records differ")
    for record in candidates:
        if file_sha256(output_dir / record["name"]) != record["sha256"]:
            raise PhaseBContractError(f"B-IPC1 candidate hash differs: {record['name']}")
    expected_banks = [
        {"name": bank["split"], "selection_sha256": bank["selection_sha256"], "parquet_sha256": bank["parquet_sha256"]}
        for bank in plan["banks"]
    ]
    expected_antecedents = [
        {"name": item["name"], "binding_sha256": item["binding_sha256"]} for item in plan["antecedents"]
    ]
    if receipt.get("bank_bindings") != expected_banks or receipt.get("antecedent_bindings") != expected_antecedents:
        raise PhaseBContractError("B-IPC1 SUCCESS bank or antecedent binding differs")
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
        or receipt.get("candidate_files_valid") is not False
    ):
        raise PhaseBContractError("B-IPC1 FAILURE top-level contract differs")
    audit = receipt.get("post_failure_audit")
    if not isinstance(audit, dict) or audit.get("audit_complete") is not True:
        raise PhaseBContractError("B-IPC1 FAILURE post-failure audit is incomplete")
    if receipt.get("model_loaded") is True and not all(
        audit.get(key) is True
        for key in ("e33_tensor_preserved", "e33_disk_preserved", "metadata_preserved", "immutable_inputs_preserved")
    ):
        raise PhaseBContractError("B-IPC1 FAILURE protected post-model audit differs")


def _atomic_publish_bytes(directory: Path, name: str, payload: bytes) -> Path:
    target = directory / name
    temporary = directory / f".{name}.{os.getpid()}.tmp"
    if target.exists() or temporary.exists():
        raise FileExistsError(f"B-IPC1 terminal path already exists: {target}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, target)
    temporary.unlink()
    return target


def _post_failure_audit(plan: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    errors = []
    result: dict[str, Any] = {}
    try:
        expected_inputs = {
            "plan": plan["_file_sha256"],
            "bank_manifest": plan["bank_manifest"]["sha256"],
            "overlap_closure": plan["overlap_closure"]["sha256"],
            **{f"{bank['split']}_selection": bank["selection_sha256"] for bank in plan["banks"]},
            **{f"{bank['split']}_parquet": bank["parquet_sha256"] for bank in plan["banks"]},
        }
        observed_inputs = {
            "plan": file_sha256(Path(plan["_path"])),
            "bank_manifest": file_sha256(Path(plan["bank_manifest"]["path"])),
            "overlap_closure": file_sha256(Path(plan["overlap_closure"]["path"])),
            **{f"{bank['split']}_selection": file_sha256(Path(bank["selection_path"])) for bank in plan["banks"]},
            **{f"{bank['split']}_parquet": file_sha256(Path(bank["parquet_path"])) for bank in plan["banks"]},
        }
        result["immutable_inputs_preserved"] = observed_inputs == expected_inputs
        result["immutable_input_hashes"] = observed_inputs
    except BaseException as error:
        errors.append(f"immutable:{type(error).__name__}:{error}")
    model = audit.get("model")
    smoke = audit.get("smoke")
    if model is not None and smoke is not None:
        try:
            current_tensor = smoke._module_tensor_sha256(model, audit["torch"])
            result["e33_tensor_preserved"] = current_tensor == audit.get("e33_tensor_pre")
            model_path = audit["model_path"]
            result["e33_disk_preserved"] = file_sha256(smoke._model_file(model_path)) == plan["protected_model"][
                "weight_sha256"
            ]
            result["metadata_preserved"] = smoke._metadata_hashes(model_path) == plan["model_metadata_sha256"]
        except BaseException as error:
            errors.append(f"e33:{type(error).__name__}:{error}")
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
        return FAILURE_STATUS_CLASSES[3]
    return FAILURE_STATUS_CLASSES[2]


def main() -> int:
    args = parse_args()
    started = time.time()
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(COMPUTE_SECONDS)
    audit: dict[str, Any] = {}
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
            validator_kwargs={"plan": plan, "execution_commit": args.execution_commit, "output_dir": args.output_dir},
        )
        signal.alarm(0)
        target = _atomic_publish_bytes(args.output_dir, "SUCCESS.json", payload)
        verify_published_terminal(
            target,
            payload,
            validator=validate_success_receipt,
            validator_kwargs={"plan": plan, "execution_commit": args.execution_commit, "output_dir": args.output_dir},
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
            "execution_breadcrumbs": {
                key: audit.get(key) for key in ("stage", "task_key", "arm", "call_index")
            },
            "post_failure_audit": post,
            "elapsed_seconds": time.time() - started,
        }
        try:
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
