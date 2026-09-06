#!/usr/bin/env python3
"""Run the prospective B-HIC0 zero-update identity-carrier diagnostic."""

from __future__ import annotations

import argparse
import contextlib
import gc
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
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

from prime_rl.phase_b_contract import (
    PhaseBContractError,
    atomic_exclusive_json,
    canonical_json_sha256,
    file_sha256,
    load_json_file,
)
from prime_rl.phase_b_identity_carrier import (
    ARMS,
    CACHE_LABEL_SHA256,
    EXPECTED_CACHE_CLASSES,
    IDENTITY_FIELDS,
    SLOTS,
    aligned_suffix_geometry,
    build_cache_guard_labels,
    evaluate_hic0,
    normalized_rms_difference,
    ordered_subclass_closure,
    recursive_subclass_closure,
    validate_hic0_selection,
    validate_hic0_terminal_receipt,
    validate_suffix_target_ids,
)
from prime_rl.phase_b_value_screen import action_margin_from_logits

WORKTREE = Path("/home/ubuntu/rlm/worktrees/q35-2b-recurrent-sidecar-v1")
EXPERIMENT = WORKTREE / "experiments/qwen35-2b-latent-coordinator-v1"
DEFAULT_PLAN = EXPERIMENT / "phase-b-hic0-identity-carrier-run1-plan.json"
BR5_RUNNER = WORKTREE / "scripts/latent/run_phase_b_fixed_depth_smoke_v1.py"
B1_RUNNER = WORKTREE / "scripts/latent/run_phase_b_teacher_forced_value_screen_v1.py"
EXPECTED_ENV = Path("/home/ubuntu/rlm/prime-rl/.venv")
EXPECTED_PYTHONPATH = (
    "/home/ubuntu/rlm/worktrees/q35-2b-recurrent-sidecar-v1/src:"
    "/home/ubuntu/rlm/worktrees/q35-2b-recurrent-sidecar-v1/packages/prime-rl-configs/src"
)
ARTIFACT_CAP = 256 * 1024**2
MINIMUM_FREE_BYTES = 60 * 1024**3
CUDA_MEMORY_CAP_BYTES = 32 * 1024**3
OUTER_SECONDS = 14_400
COMPUTE_SECONDS = 14_040
AUDIT_SECONDS = 300
TERMINAL_SECONDS = 60
CODEC_SEED = 262_502_387
MINIMUM_RAM_BYTES = 64 * 1024**3


class CacheContractViolated(PhaseBContractError):
    """Raised when a receiver or source call constructs or returns a cache."""


class ResourceContractExceeded(PhaseBContractError):
    """Raised when HIC0 exceeds its frozen allocator contract."""


class ProvenanceContractViolated(PhaseBContractError):
    """Raised when an immutable input or protected source has changed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--authorized-plan-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PhaseBContractError(f"cannot load frozen helper {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timeout_handler(_signum: int, _frame: Any) -> None:
    raise TimeoutError("B-HIC0 internal compute wall-clock limit reached")


def _canonical_plan_hash(plan: dict[str, Any]) -> str:
    payload = deepcopy(plan)
    payload["plan_sha256"] = None
    return canonical_json_sha256(payload)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WORKTREE, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_parent(commit: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", f"{commit}^"], cwd=WORKTREE, check=True, capture_output=True, text=True
    ).stdout.strip()


def _available_ram_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        raise ResourceContractExceeded("B-HIC0 requires Linux /proc/meminfo RAM evidence")
    fields = {}
    for line in meminfo.read_text(encoding="ascii").splitlines():
        name, value = line.split(":", 1)
        fields[name] = value.strip()
    available = fields.get("MemAvailable")
    if available is None or not available.endswith(" kB"):
        raise ResourceContractExceeded("B-HIC0 MemAvailable evidence is malformed")
    return int(available.removesuffix(" kB")) * 1024


def _validate_plan(plan: dict[str, Any], args: argparse.Namespace) -> None:
    if plan.get("schema_version") != "q35-2b-phase-b-hic0-identity-carrier/v1":
        raise PhaseBContractError("B-HIC0 plan schema differs")
    if plan.get("status") != "frozen_pending_independent_review":
        raise PhaseBContractError("B-HIC0 plan is not frozen")
    if plan.get("plan_sha256") != _canonical_plan_hash(plan):
        raise PhaseBContractError("B-HIC0 internal plan hash differs")
    if file_sha256(args.plan) != args.authorized_plan_sha256:
        raise PhaseBContractError("B-HIC0 external authorized plan hash differs")
    if args.execution_commit != _git_head():
        raise PhaseBContractError("B-HIC0 execution commit differs")
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=WORKTREE,
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise PhaseBContractError("B-HIC0 requires a clean deployed worktree")
    if plan.get("mechanism_code_commit") != _git_parent(args.execution_commit):
        raise PhaseBContractError("B-HIC0 exact-parent mechanism binding differs")
    if plan.get("execution_environment") != {
        "uv": "/home/ubuntu/.local/bin/uv",
        "uv_project": "/home/ubuntu/rlm/prime-rl",
        "UV_PROJECT_ENVIRONMENT": str(EXPECTED_ENV),
        "PYTHONPATH": EXPECTED_PYTHONPATH,
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }:
        raise PhaseBContractError("B-HIC0 execution environment contract differs")
    if Path(os.environ.get("UV_PROJECT_ENVIRONMENT", "")) != EXPECTED_ENV:
        raise PhaseBContractError("B-HIC0 UV_PROJECT_ENVIRONMENT differs")
    if Path(sys.prefix).resolve() != EXPECTED_ENV.resolve(strict=True):
        raise PhaseBContractError("B-HIC0 runner is outside the frozen shared environment")
    if os.environ.get("PYTHONPATH") != EXPECTED_PYTHONPATH:
        raise PhaseBContractError("B-HIC0 PYTHONPATH differs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise PhaseBContractError("B-HIC0 requires exactly physical GPU 0")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != plan["execution_environment"][name] or os.environ.get(name) != "1":
            raise PhaseBContractError(f"B-HIC0 offline environment differs: {name}")
    if args.output_dir != Path(plan["outputs"]["directory"]):
        raise PhaseBContractError("B-HIC0 output namespace differs")
    if args.output_dir.exists():
        raise PhaseBContractError("B-HIC0 output namespace is not fresh")
    if not args.output_dir.parent.is_dir() or args.output_dir.parent.is_symlink():
        raise PhaseBContractError("B-HIC0 output parent is absent or symlinked")
    if shutil.disk_usage(args.output_dir.parent).free < MINIMUM_FREE_BYTES:
        raise ResourceContractExceeded("B-HIC0 has less than 60 GiB free disk")
    resources = plan.get("resources", {})
    if (
        resources.get("outer_wall_clock_seconds"),
        resources.get("compute_limit_seconds"),
        resources.get("failure_audit_limit_seconds"),
        resources.get("terminal_publication_headroom_seconds"),
        resources.get("cuda_memory_cap_bytes"),
        resources.get("minimum_host_ram_bytes"),
        resources.get("minimum_free_disk_bytes"),
        resources.get("artifact_cap_bytes"),
        resources.get("gpus"),
        resources.get("gpu"),
        resources.get("network"),
    ) != (
        OUTER_SECONDS,
        COMPUTE_SECONDS,
        AUDIT_SECONDS,
        TERMINAL_SECONDS,
        CUDA_MEMORY_CAP_BYTES,
        MINIMUM_RAM_BYTES,
        MINIMUM_FREE_BYTES,
        ARTIFACT_CAP,
        1,
        "NVIDIA RTX A6000 physical GPU 0",
        False,
    ):
        raise PhaseBContractError("B-HIC0 resource contract differs")
    if _available_ram_bytes() < MINIMUM_RAM_BYTES:
        raise ResourceContractExceeded("B-HIC0 has less than 64 GiB available RAM")
    mechanism = plan.get("mechanism", {})
    if (
        mechanism.get("arms"),
        mechanism.get("slots"),
        mechanism.get("receiver_gate"),
        mechanism.get("codec_seed"),
        mechanism.get("cache_guard_label_sha256"),
        mechanism.get("source_forwards"),
        mechanism.get("receiver_forwards"),
    ) != (list(ARMS), 8, 0.001, CODEC_SEED, CACHE_LABEL_SHA256, 12, 60):
        raise PhaseBContractError("B-HIC0 mechanism constants differ")
    if mechanism.get("cache_classes") != [item[0] for item in EXPECTED_CACHE_CLASSES]:
        raise PhaseBContractError("B-HIC0 pinned cache class list differs")
    if mechanism.get("cache_class_sources") != {
        "fla.models.utils": {
            "relative_path": EXPECTED_CACHE_CLASSES[0][1],
            "sha256": EXPECTED_CACHE_CLASSES[0][2],
            "distribution": EXPECTED_CACHE_CLASSES[0][3],
        },
        "transformers.cache_utils": {
            "relative_path": EXPECTED_CACHE_CLASSES[3][1],
            "sha256": EXPECTED_CACHE_CLASSES[3][2],
            "distribution": EXPECTED_CACHE_CLASSES[3][3],
        },
    }:
        raise PhaseBContractError("B-HIC0 pinned cache source provenance differs")
    forbidden = plan.get("boundaries", {})
    if (
        forbidden.get("optimizer_updates"),
        forbidden.get("generation"),
        forbidden.get("cache"),
        forbidden.get("H176_loaded"),
        forbidden.get("strand_a_combined"),
        forbidden.get("live_trajectory_count"),
        forbidden.get("live_promotion_floor"),
    ) != (0, False, False, False, False, 0, 4):
        raise PhaseBContractError("B-HIC0 boundary contract differs")
    statuses = plan.get("terminal_statuses", {})
    if (
        statuses.get("top_level_status_is_literal"),
        statuses.get("internal_receipt_sha256_omits_only_itself"),
        statuses.get("pass"),
        statuses.get("complete_threshold_miss"),
        statuses.get("cache_or_pkv_failure"),
        statuses.get("nonfinite_schema_render_alignment_gradient_missing"),
        statuses.get("oom_cap_timeout_runtime_provenance"),
    ) != (
        True,
        True,
        "b_hic0_inplace_carrier_nominated",
        "b_hic0_inplace_carrier_not_nominated",
        "b_hic0_nocache_rejected",
        "b_hic0_incomplete",
        "infrastructure_invalid",
    ):
        raise PhaseBContractError("B-HIC0 terminal status contract differs")


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if not args.plan.is_file() or not args.selection.is_file():
        raise PhaseBContractError("B-HIC0 plan or selection is absent")
    plan = load_json_file(args.plan)
    _validate_plan(plan, args)
    plan["_preflight_resources"] = {
        "available_host_ram_bytes": _available_ram_bytes(),
        "free_disk_bytes": shutil.disk_usage(args.output_dir.parent).free,
        "minimum_host_ram_bytes": MINIMUM_RAM_BYTES,
        "minimum_free_disk_bytes": MINIMUM_FREE_BYTES,
        "offline_environment": {
            name: os.environ[name] for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        },
    }
    selection = load_json_file(args.selection)
    validate_hic0_selection(selection)
    bank = plan["diagnostic_bank"]
    if args.selection != Path(bank["selection_path"]):
        raise PhaseBContractError("B-HIC0 selection path differs")
    immutable = {
        "selection": args.selection,
        "parquet": Path(bank["parquet_path"]),
        "manifest": Path(bank["manifest_path"]),
        "runtime": Path(bank["runtime_source_path"]),
        "generator": Path(bank["generator_path"]),
        "taskset": Path(bank["taskset_source_path"]),
        "b1r_binding": Path(plan["b1r_dependency"]["binding_path"]),
    }
    expected = {
        "selection": bank["selection_sha256"],
        "parquet": bank["parquet_sha256"],
        "manifest": bank["manifest_sha256"],
        "runtime": bank["runtime_source_sha256"],
        "generator": bank["generator_sha256"],
        "taskset": bank["taskset_source_sha256"],
        "b1r_binding": plan["b1r_dependency"]["binding_sha256"],
    }
    observed = {key: file_sha256(path) for key, path in immutable.items()}
    if observed != expected:
        raise PhaseBContractError("B-HIC0 immutable artifact closure differs")
    manifest = load_json_file(immutable["manifest"])
    if manifest.get("row_list_canonical_sha256") != bank["row_list_canonical_sha256"]:
        raise PhaseBContractError("B-HIC0 bank row-list hash differs")
    if manifest.get("freshness", {}).get("diagnostic_rows_forbidden_future_training") is not True:
        raise PhaseBContractError("B-HIC0 future-training exclusion is absent")
    prior = load_json_file(immutable["b1r_binding"])
    if (
        prior.get("status"),
        prior.get("disposition"),
        prior.get("execution_commit"),
        prior.get("plan_file_sha256"),
        prior.get("success_receipt_sha256"),
        prior.get("snapshot_manifest_sha256"),
    ) != (
        "SUCCESS",
        "b1_not_nominated",
        "ff4e3fd8695a05945187702d7ff8bc6412711ad9",
        "8909f30d82559203018282a1006ce8cbe4bdbf418728a0e06ca2399e4992d5a6",
        "4cb2bf1e7e884f24c297381dc30698bce4dac2586e3f4522f231600ad76ef761",
        "7ea9c294c5db0450b9c67f800153df57a17605f7bb58df2000996a6fba26f038",
    ):
        raise PhaseBContractError("B-HIC0 B1R negative dependency differs")
    return plan | {"_path": str(args.plan), "_file_sha256": observed.get("plan", args.authorized_plan_sha256)}, selection


def tokenizer_preflight(plan: dict[str, Any], selection: dict[str, Any], *, parquet: Any, AutoTokenizer: Any) -> dict[str, Any]:
    smoke = _load_module(BR5_RUNNER, "phase_b_hic0_br5_runtime")
    b1 = _load_module(B1_RUNNER, "phase_b_hic0_b1_runtime")
    model_path = Path(plan["protected_model"]["path"])
    if file_sha256(smoke._model_file(model_path)) != plan["protected_model"]["weight_sha256"]:
        raise PhaseBContractError("B-HIC0 e33 weight file differs")
    if smoke._metadata_hashes(model_path) != plan["model_metadata_sha256"]:
        raise PhaseBContractError("B-HIC0 e33 metadata differs")
    rows = parquet.read_table(Path(plan["diagnostic_bank"]["parquet_path"])).to_pylist()
    pairs = validate_hic0_selection(selection)
    by_key = {row["task_key"]: row for row in rows}
    if len(by_key) != 12 or any(pair["task_key"] not in by_key for pair in pairs):
        raise PhaseBContractError("B-HIC0 selected rows differ")
    ordered = [by_key[pair["task_key"]] for pair in pairs]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    rendered, proofs = b1._render_rows(ordered, tokenizer=tokenizer, smoke=smoke)
    if [row["task_key"] for row in rendered] != selection["task_keys"]:
        raise PhaseBContractError("B-HIC0 render order differs")
    for row in rendered:
        geometry = aligned_suffix_geometry(
            total=len(row["full_ids"]), supervised_start=len(row["open_ids"]), insertion_index=len(row["plain_ids"])
        )
        row["geometry"] = geometry
        if max(branch["logit_offset"] for branch in row["action_trie"]["branches"]) >= geometry["K"] - 1:
            raise PhaseBContractError("B-HIC0 action trie leaves the supervised suffix")
    if smoke._cuda_initialized_if_torch_loaded():
        raise PhaseBContractError("CUDA initialized during B-HIC0 tokenizer-only preflight")
    return {"rows": rendered, "proofs": proofs, "smoke": smoke}


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
        raise ResourceContractExceeded(f"B-HIC0 CUDA memory cap exceeded at {label}")


def _cache_class_identity(cls: type) -> dict[str, str]:
    module = __import__(cls.__module__, fromlist=["__name__"])
    path = Path(module.__file__).resolve(strict=True)
    if not path.is_relative_to(EXPECTED_ENV.resolve(strict=True)):
        raise CacheContractViolated(f"B-HIC0 cache class outside frozen environment: {cls}")
    distribution = "flash-linear-attention" if cls.__module__.split(".", 1)[0] == "fla" else "transformers"
    return {
        "fqcn": f"{cls.__module__}.{cls.__qualname__}",
        "module_path": str(path),
        "module_sha256": file_sha256(path),
        "distribution": f"{distribution}=={importlib.metadata.version(distribution)}",
    }


class _CacheGuard:
    def __init__(self, model: Any, *, transformers: Any) -> None:
        self.model = model
        self.transformers = transformers
        self.expected = build_cache_guard_labels()
        self.labels: list[str] = []
        self.base = transformers.cache_utils.Cache
        self.initial_classes = recursive_subclass_closure(self.base)
        self.patched_classes: set[type] = set()
        self.stack = contextlib.ExitStack()
        self.configs: dict[int, tuple[Any, bool]] = {}
        self.calls = 0
        self.trips = 0
        self.closure_checks = 0
        self.restored = False

    @staticmethod
    def _forbidden(cls: type, *_args: Any, **_kwargs: Any) -> None:
        del cls
        raise CacheContractViolated("B-HIC0 Cache subclass construction attempted")

    def _trip(self, cls: type, *_args: Any, **_kwargs: Any) -> None:
        self.trips += 1
        raise CacheContractViolated(f"B-HIC0 cache allocation attempted: {cls.__module__}.{cls.__qualname__}")

    def _verify_closure(self) -> None:
        new_classes = recursive_subclass_closure(self.base) - self.patched_classes
        if new_classes:
            names = sorted(f"{cls.__module__}.{cls.__qualname__}" for cls in new_classes)
            raise CacheContractViolated(f"B-HIC0 new unpatched cache subclasses loaded: {names}")
        if any(config.use_cache is not False for config, _value in self.configs.values()):
            raise CacheContractViolated("B-HIC0 recursive use_cache closure reopened")
        self.closure_checks += 1

    def _restore(self) -> None:
        stack_error = None
        try:
            self.stack.close()
        except BaseException as error:
            stack_error = error
        for config, value in self.configs.values():
            config.use_cache = value
        if any(config.use_cache is not value for config, value in self.configs.values()):
            raise CacheContractViolated("B-HIC0 use_cache originals were not restored exactly")
        self.restored = True
        if stack_error is not None:
            raise stack_error

    def _append(self, label: str) -> None:
        self._verify_closure()
        self.labels.append(label)

    def __enter__(self) -> "_CacheGuard":
        try:
            config_candidates = [getattr(module, "config", None) for module in self.model.modules()]
            config_candidates.append(getattr(self.model, "generation_config", None))
            for config in config_candidates:
                if config is not None and hasattr(config, "use_cache"):
                    self.configs.setdefault(id(config), (config, config.use_cache))
                    config.use_cache = False
            classes = ordered_subclass_closure(self.base)
            if set(classes) != self.initial_classes:
                raise CacheContractViolated("B-HIC0 cache census changed before patching")
            identities = tuple(
                (
                    item["fqcn"],
                    str(Path(item["module_path"]).relative_to(EXPECTED_ENV)),
                    item["module_sha256"],
                    item["distribution"],
                )
                for item in map(_cache_class_identity, classes)
            )
            if identities != EXPECTED_CACHE_CLASSES:
                raise CacheContractViolated("B-HIC0 pinned eight-class cache closure differs")
            for cls in classes:
                replacement = self._trip if cls is self.transformers.cache_utils.DynamicCache else self._forbidden
                self.stack.enter_context(mock.patch.object(cls, "__new__", replacement))
                self.patched_classes.add(cls)
            try:
                self.transformers.cache_utils.DynamicCache()
            except CacheContractViolated:
                pass
            else:
                raise CacheContractViolated("B-HIC0 DynamicCache negative control did not trip")
            if self.trips != 1:
                raise CacheContractViolated("B-HIC0 DynamicCache trip count differs")
            self._append("CACHE_GUARD_ENTRY")
        except BaseException:
            self._restore()
            raise
        return self

    def call(self, *, row_index: int, operation: str, **kwargs: Any) -> Any:
        expected_pre = f"CACHE_GUARD_PRE_HIC0_R{row_index:02d}_{operation}"
        expected_post = f"CACHE_GUARD_POST_HIC0_R{row_index:02d}_{operation}"
        self.model.model.rope_deltas = None
        if self.model.config.use_cache is not False:
            raise CacheContractViolated("B-HIC0 model use_cache default reopened")
        self._append(expected_pre)
        output = self.model(past_key_values=None, use_cache=False, return_dict=True, **kwargs)
        if getattr(output, "past_key_values", None) is not None or self.model.model.rope_deltas is not None:
            raise CacheContractViolated(f"B-HIC0 cache/rope closure failed at {row_index}:{operation}")
        self.calls += 1
        self._append(expected_post)
        return output

    def final(self) -> None:
        if self.calls != 72 or self.trips != 1:
            raise CacheContractViolated("B-HIC0 cache guard call/trip count differs")
        self._append("CACHE_GUARD_FINAL")

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        try:
            self._append("CACHE_GUARD_EXIT")
        finally:
            self._restore()

    def evidence(self) -> dict[str, Any]:
        complete = self.labels == self.expected
        prefix_without_exit = self.labels[:-1] if self.labels and self.labels[-1] == "CACHE_GUARD_EXIT" else self.labels
        return {
            "complete": complete,
            "labels": self.labels,
            "label_count": len(self.labels),
            "canonical_label_sha256": canonical_json_sha256(self.labels),
            "expected_label_sha256": CACHE_LABEL_SHA256,
            "exact_prefix_before_exit": self.expected[: len(prefix_without_exit)] == prefix_without_exit,
            "exit_recorded": bool(self.labels and self.labels[-1] == "CACHE_GUARD_EXIT"),
            "dynamic_cache_trip_count": self.trips,
            "closure_check_count": self.closure_checks,
            "closure_checked_at_every_recorded_label": self.closure_checks == len(self.labels),
            "classes": [
                _cache_class_identity(cls)
                for cls in ordered_subclass_closure(self.base)
            ],
            "recursively_closed_config_count": len(self.configs),
            "restored_in_finally": self.restored,
            "model_calls": self.calls,
        }


def _tensor_hash(tensor: Any, *, smoke: ModuleType, torch: Any) -> str:
    return smoke._tensor_bytes_sha256(tensor, torch)


def _suffix_nll_float64(logits: Any, labels: Any, *, torch: Any) -> float:
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise PhaseBContractError("B-HIC0 suffix logits/labels shapes differ")
    if int(labels[0, 0]) != -100 or logits.shape[1] < 2:
        raise PhaseBContractError("B-HIC0 suffix labels lack their causal masked predictor")
    targets = labels.detach()[0, 1:].to(device="cpu", dtype=torch.long)
    if targets.numel() != logits.shape[1] - 1:
        raise PhaseBContractError("B-HIC0 suffix target labels are not contiguous and valid")
    validate_suffix_target_ids([int(value) for value in targets.tolist()], vocabulary_size=logits.shape[2])
    values = logits.detach()[0, : targets.numel(), :].to(device="cpu", dtype=torch.float64)
    per_token = torch.logsumexp(values, dim=-1) - values.gather(1, targets[:, None]).squeeze(1)
    as_list = [float(value) for value in per_token.tolist()]
    if not as_list or not all(math.isfinite(value) for value in as_list):
        raise PhaseBContractError("B-HIC0 float64 suffix NLL is non-finite")
    return math.fsum(as_list) / len(as_list)


def _metric(output: Any, row: dict[str, Any], predictor: int, inputs: dict[str, Any], *, smoke: ModuleType, torch: Any) -> dict[str, Any]:
    if output.loss is None or not bool(torch.isfinite(output.loss)):
        raise PhaseBContractError("B-HIC0 NLL is absent or non-finite")
    if output.logits.shape[1] != row["geometry"]["K"]:
        raise PhaseBContractError("B-HIC0 aligned suffix logit length differs")
    logits_cpu64 = output.logits.detach().to(device="cpu", dtype=torch.float64)
    loss_labels = inputs["labels"]
    nll = _suffix_nll_float64(output.logits, loss_labels, torch=torch)
    margin, branches = action_margin_from_logits(
        row["action_trie"], lambda offset, token: float(logits_cpu64[0, offset, token])
    )
    hidden = output.hidden_states[-1][:, predictor : predictor + 1, :]
    first_logits = output.logits[:, 0:1, :]
    return {
        "nll": nll,
        "native_output_loss_descriptive": float(output.loss),
        "native_minus_float64_nll_descriptive": float(output.loss) - nll,
        "margin": margin,
        "finite": bool(torch.isfinite(hidden).all() and torch.isfinite(first_logits).all()),
        "branch_metrics": branches,
        "final_hidden": hidden,
        "first_suffix_logits": first_logits,
        "hashes": {
            "inputs_embeds": _tensor_hash(inputs["inputs_embeds"], smoke=smoke, torch=torch),
            "attention_mask": _tensor_hash(inputs["attention_mask"], smoke=smoke, torch=torch),
            "position_ids": _tensor_hash(inputs["position_ids"], smoke=smoke, torch=torch),
            "labels": _tensor_hash(inputs["full_labels"], smoke=smoke, torch=torch),
            "final_hidden": _tensor_hash(hidden, smoke=smoke, torch=torch),
            "first_suffix_logits": _tensor_hash(first_logits, smoke=smoke, torch=torch),
        },
    }


def _receiver_inputs(row: dict[str, Any], arm: str, base_embeddings: Any, residual: Any, *, torch: Any) -> tuple[dict[str, Any], int]:
    g = row["geometry"]
    total = g["TQ"] if arm.startswith("INSERT") else g["T"]
    predictor = g["BQ"] if arm.startswith("INSERT") else g["B"]
    full_ids = row["full_tensor"]
    base_mask = torch.ones_like(full_ids)
    base_positions = torch.arange(g["T"], device="cuda:0")[None]
    base_labels = full_ids.clone()
    base_labels[:, : g["S"]] = -100
    if arm.startswith("INSERT"):
        value = torch.zeros_like(residual) if arm == "INSERT_ZERO" else residual
        prefix = slice(0, g["I"])
        suffix = slice(g["I"], None)
        embeddings = torch.cat((base_embeddings[:, prefix], value, base_embeddings[:, suffix]), dim=1)
        mask = torch.cat((base_mask[:, prefix], torch.ones((1, SLOTS), dtype=base_mask.dtype, device="cuda:0"), base_mask[:, suffix]), dim=1)
        positions = torch.cat((base_positions[:, prefix], torch.arange(g["I"], g["I"] + SLOTS, device="cuda:0")[None], base_positions[:, suffix] + SLOTS), dim=1)
        labels = torch.cat((base_labels[:, prefix], torch.full((1, SLOTS), -100, dtype=base_labels.dtype, device="cuda:0"), base_labels[:, suffix]), dim=1)
    else:
        embeddings = base_embeddings.clone()
        target = slice(g["I"] - SLOTS, g["I"])
        if arm == "INPLACE_EPS":
            embeddings[:, target, :] = base_embeddings[:, target, :] + residual
        elif arm == "INPLACE_ZERO":
            embeddings[:, target, :] = base_embeddings[:, target, :] + torch.zeros_like(residual)
        elif arm == "BASE":
            embeddings = base_embeddings
        else:
            raise PhaseBContractError(f"unknown B-HIC0 arm {arm}")
        mask, positions, labels = base_mask, base_positions, base_labels
    if embeddings.shape[1] != total or labels.shape[1] != total:
        raise PhaseBContractError("B-HIC0 arm geometry differs")
    loss_labels = labels[:, predictor:]
    if loss_labels.shape[1] != g["K"] or int(loss_labels[0, 0]) != -100:
        raise PhaseBContractError("B-HIC0 aligned suffix labels differ")
    return {
        "inputs_embeds": embeddings,
        "attention_mask": mask,
        "position_ids": positions,
        "labels": loss_labels,
        "full_labels": labels,
        "logits_to_keep": g["K"],
        "output_hidden_states": True,
    }, predictor


def _cosine(left: Any, right: Any, *, torch: Any) -> float:
    left = left.detach().cpu().double().reshape(-1)
    right = right.detach().cpu().double().reshape(-1)
    if float(torch.linalg.vector_norm(left)) == 0.0 or float(torch.linalg.vector_norm(right)) == 0.0:
        raise PhaseBContractError("B-HIC0 cosine input has zero norm")
    value = float(torch.nn.functional.cosine_similarity(left, right, dim=0))
    if not math.isfinite(value):
        raise PhaseBContractError("B-HIC0 cosine is non-finite")
    return value


def execute(plan: dict[str, Any], context: dict[str, Any], *, execution_commit: str, torch: Any, transformers: Any, AutoModelForImageTextToText: Any, LocalDepthCodec: Any, audit: dict[str, Any]) -> dict[str, Any]:
    smoke = context["smoke"]
    smoke._validate_torch_runtime(plan, torch=torch)
    smoke._validate_transformers_runtime(plan, transformers=transformers)
    smoke._require_gpu0_idle()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ResourceContractExceeded("B-HIC0 requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    if torch.cuda.get_device_name(0) != "NVIDIA RTX A6000":
        raise ResourceContractExceeded("B-HIC0 requires one NVIDIA RTX A6000")
    total_device_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    fraction = CUDA_MEMORY_CAP_BYTES / total_device_bytes
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    torch.cuda.reset_peak_memory_stats(0)
    audit["allocator"] = {"device_total_bytes": total_device_bytes, "cap_bytes": CUDA_MEMORY_CAP_BYTES, "requested_fraction": fraction, "observed_fraction": float(torch.cuda.get_per_process_memory_fraction(0))}
    model_path = Path(plan["protected_model"]["path"])
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).to("cuda:0")
    model.eval()
    if model.__class__.__name__ != "Qwen3_5ForConditionalGeneration":
        raise PhaseBContractError("B-HIC0 loaded model architecture differs")
    if int(model.config.text_config.hidden_size) != 2048:
        raise PhaseBContractError("B-HIC0 e33 hidden size differs")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    importlib.import_module("fla.models.utils")
    audit.update({"model": model, "stage": "model_loaded"})
    _memory_checkpoint(torch, audit, "after_model_load")
    e33_pre = smoke._module_tensor_sha256(model, torch)
    file_pre = file_sha256(smoke._model_file(model_path))
    metadata_pre = smoke._metadata_hashes(model_path)
    audit.update({"e33_pre": e33_pre, "file_pre": file_pre, "metadata_pre": metadata_pre})
    torch.manual_seed(CODEC_SEED)
    torch.cuda.manual_seed_all(CODEC_SEED)
    codec = LocalDepthCodec(2048, 256, 8, initial_receiver_gate=0.001).to(device="cuda:0", dtype=torch.bfloat16)
    codec_pre = smoke._module_tensor_sha256(codec, torch)
    audit.update({"codec": codec, "codec_pre": codec_pre})
    _memory_checkpoint(torch, audit, "after_codec_construction")
    shell = smoke._mean_embedding_norm(model.get_input_embeddings().weight, torch)
    first_norm = model.model.language_model.layers[0].input_layernorm
    rows_evidence: list[dict[str, Any]] = []
    backward: dict[str, Any] | None = None
    backward_row = "document_adaptive_d2-v4-i35100"
    guard = _CacheGuard(model, transformers=transformers)
    audit["cache_guard_object"] = guard
    with guard:
        for row_index, rendered in enumerate(context["rows"], start=1):
            audit.update({"stage": "source_capture", "task_key": rendered["task_key"], "arm": None})
            plain = torch.tensor([rendered["plain_ids"]], dtype=torch.long, device="cuda:0")
            full = torch.tensor([rendered["full_ids"]], dtype=torch.long, device="cuda:0")
            with torch.no_grad():
                capture_output = guard.call(
                    row_index=row_index,
                    operation="SOURCE_CAPTURE",
                    input_ids=plain,
                    attention_mask=torch.ones_like(plain),
                    position_ids=torch.arange(plain.shape[1], device="cuda:0")[None],
                    output_hidden_states=True,
                    logits_to_keep=1,
                )
            captured = capture_output.hidden_states[-1][:, -SLOTS:, :].detach()
            if captured.shape != (1, SLOTS, 2048) or not bool(torch.isfinite(captured).all()):
                raise PhaseBContractError("B-HIC0 source capture differs")
            del capture_output
            _memory_checkpoint(torch, audit, f"capture:r{row_index:02d}")
            gradient_row = rendered["task_key"] == backward_row
            with torch.enable_grad() if gradient_row else torch.no_grad():
                anchor = codec.encode(
                    captured, torch.ones(captured.shape[:2], dtype=torch.long, device="cuda:0")
                )
                residual = codec.decode(anchor, shell)
            if residual.shape != (1, SLOTS, 2048) or not bool(torch.isfinite(residual).all()):
                raise PhaseBContractError("B-HIC0 residual differs")
            residual_hash = _tensor_hash(residual, smoke=smoke, torch=torch)
            base_embeddings = model.get_input_embeddings()(full).detach()
            g = aligned_suffix_geometry(total=full.shape[1], supervised_start=len(rendered["open_ids"]), insertion_index=plain.shape[1])
            if g != rendered["geometry"]:
                raise PhaseBContractError("B-HIC0 runtime geometry differs from tokenizer preflight")
            row = rendered | {"geometry": g, "full_tensor": full}
            norm_r = first_norm(residual)
            norm_x = first_norm(base_embeddings[:, g["I"] - SLOTS : g["I"], :])
            norm_xr = first_norm(base_embeddings[:, g["I"] - SLOTS : g["I"], :] + residual)
            residual_cpu64 = residual.detach().cpu().double()
            norm_r_cpu64 = norm_r.detach().cpu().double()
            norm_x_cpu64 = norm_x.detach().cpu().double()
            norm_xr_cpu64 = norm_xr.detach().cpu().double()
            residual_norms = torch.linalg.vector_norm(residual_cpu64, dim=-1)
            if bool(torch.any(residual_norms == 0)):
                raise PhaseBContractError("B-HIC0 residual has a zero-norm slot")
            a_insert = torch.linalg.vector_norm(norm_r_cpu64, dim=-1) / residual_norms
            inplace_delta = norm_xr_cpu64 - norm_x_cpu64
            a_inplace = torch.linalg.vector_norm(inplace_delta, dim=-1) / residual_norms
            amplification = {
                "A_insert": [float(value) for value in a_insert.reshape(-1)],
                "A_inplace": [float(value) for value in a_inplace.reshape(-1)],
                "insert_norm_residual_cosine": [
                    _cosine(norm_r_cpu64[:, slot], residual_cpu64[:, slot], torch=torch)
                    for slot in range(SLOTS)
                ],
                "inplace_norm_cosine": [
                    _cosine(norm_x_cpu64[:, slot], norm_xr_cpu64[:, slot], torch=torch)
                    for slot in range(SLOTS)
                ],
                "rmsnorm_module": "model.model.language_model.layers[0].input_layernorm",
            }
            arm_runtime: dict[str, dict[str, Any]] = {}
            for arm in ARMS:
                audit.update({"stage": "receiver", "task_key": rendered["task_key"], "arm": arm})
                inputs, predictor = _receiver_inputs(row, arm, base_embeddings, residual, torch=torch)
                model_inputs = {key: value for key, value in inputs.items() if key != "full_labels"}
                if gradient_row and arm == "INPLACE_EPS":
                    residual.retain_grad()
                    output = guard.call(row_index=row_index, operation=arm, **model_inputs)
                else:
                    with torch.no_grad():
                        output = guard.call(row_index=row_index, operation=arm, **model_inputs)
                metric = _metric(output, row, predictor, inputs, smoke=smoke, torch=torch)
                if arm in ("INSERT_EPS", "INPLACE_EPS"):
                    metric["residual_bfloat16_sha256"] = residual_hash
                arm_runtime[arm] = metric
                metric["final_hidden"] = metric["final_hidden"].detach().cpu()
                metric["first_suffix_logits"] = metric["first_suffix_logits"].detach().cpu()
                if not (gradient_row and arm == "INPLACE_EPS"):
                    del output
                _memory_checkpoint(torch, audit, f"receiver:r{row_index:02d}:{arm}")
            base = arm_runtime["BASE"]
            inplace_zero = arm_runtime["INPLACE_ZERO"]
            identity = {
                **{field: base["hashes"][field] == inplace_zero["hashes"][field] for field in IDENTITY_FIELDS[:-2]},
                "nll": base["nll"] == inplace_zero["nll"],
                "margin": base["margin"] == inplace_zero["margin"],
            }
            if tuple(identity) != IDENTITY_FIELDS:
                raise PhaseBContractError("B-HIC0 identity field order differs")
            insert_hidden = normalized_rms_difference(arm_runtime["INSERT_EPS"]["final_hidden"], base["final_hidden"], torch=torch)
            inplace_hidden = normalized_rms_difference(arm_runtime["INPLACE_EPS"]["final_hidden"], base["final_hidden"], torch=torch)
            insert_logit = normalized_rms_difference(arm_runtime["INSERT_EPS"]["first_suffix_logits"], base["first_suffix_logits"], torch=torch)
            inplace_logit = normalized_rms_difference(arm_runtime["INPLACE_EPS"]["first_suffix_logits"], base["first_suffix_logits"], torch=torch)
            rows_evidence.append({
                "task_key": rendered["task_key"],
                "action": rendered["action"],
                "geometry": g,
                "residual_bfloat16_sha256": residual_hash,
                "same_residual_bytes_insert_and_inplace": arm_runtime["INSERT_EPS"]
                ["residual_bfloat16_sha256"]
                == arm_runtime["INPLACE_EPS"]["residual_bfloat16_sha256"]
                == residual_hash,
                "arms": {arm: {key: value for key, value in arm_runtime[arm].items() if key not in ("final_hidden", "first_suffix_logits")} for arm in ARMS},
                "inplace_zero_identity": identity,
                "rmsnorm_amplification": amplification,
                "drift": {
                    "hidden_insert_eps_nrms": insert_hidden,
                    "hidden_inplace_eps_nrms": inplace_hidden,
                    "logit_insert_eps_nrms": insert_logit,
                    "logit_inplace_eps_nrms": inplace_logit,
                },
            })
            if gradient_row:
                audit.update({"stage": "backward", "task_key": backward_row, "arm": "INPLACE_EPS"})
                output.loss.backward()
                residual_grad = residual.grad
                named = {name: parameter.grad for name, parameter in codec.named_parameters()}
                encoder_names = ("source_norm.", "source_projection.")
                receiver_names = ("workspace_norm.", "receiver_projection.", "receiver_gate")

                def group_evidence(
                    named_gradients: dict[str, Any], prefixes: tuple[str, ...]
                ) -> dict[str, Any]:
                    values = [
                        grad
                        for name, grad in named_gradients.items()
                        if name.startswith(prefixes) and grad is not None
                    ]
                    return {
                        "tensor_count": len(values),
                        "finite": bool(values)
                        and all(bool(torch.isfinite(value).all()) for value in values),
                        "nonzero": bool(values)
                        and any(bool(torch.count_nonzero(value)) for value in values),
                    }

                backward = {
                    "row": backward_row,
                    "receiver_forward_reused": True,
                    "extra_receiver_forwards": 0,
                    "residual_gradient": {
                        "finite": residual_grad is not None
                        and bool(torch.isfinite(residual_grad).all()),
                        "nonzero": residual_grad is not None and bool(torch.count_nonzero(residual_grad)),
                    },
                    "encoder_group": group_evidence(named, encoder_names),
                    "receiver_group": group_evidence(named, receiver_names),
                    "all_named_gradients_finite": all(
                        grad is None or bool(torch.isfinite(grad).all()) for grad in named.values()
                    ),
                    "e33_gradients_absent": all(parameter.grad is None for parameter in model.parameters()),
                }
                for parameter in codec.parameters():
                    parameter.grad = None
                del output, residual_grad, named
                _memory_checkpoint(torch, audit, "after_backward")
            for metric in arm_runtime.values():
                metric.pop("final_hidden", None)
                metric.pop("first_suffix_logits", None)
            del plain, full, captured, anchor, base_embeddings, norm_r, norm_x, norm_xr
            del residual_cpu64, norm_r_cpu64, norm_x_cpu64, norm_xr_cpu64, inplace_delta, a_insert, a_inplace
            del residual
            gc.collect()
            torch.cuda.empty_cache()
        if backward is None:
            raise PhaseBContractError("B-HIC0 predesignated backward forward is absent")
        guard.final()
    cache_evidence = guard.evidence()
    audit["cache_guard"] = cache_evidence
    codec_post = smoke._module_tensor_sha256(codec, torch)
    e33_post = smoke._module_tensor_sha256(model, torch)
    file_post = file_sha256(smoke._model_file(model_path))
    metadata_post = smoke._metadata_hashes(model_path)
    immutable_post = {
        "plan": file_sha256(Path(plan["_path"])),
        "selection": file_sha256(Path(plan["diagnostic_bank"]["selection_path"])),
        "parquet": file_sha256(Path(plan["diagnostic_bank"]["parquet_path"])),
        "manifest": file_sha256(Path(plan["diagnostic_bank"]["manifest_path"])),
        "runtime": file_sha256(Path(plan["diagnostic_bank"]["runtime_source_path"])),
        "generator": file_sha256(Path(plan["diagnostic_bank"]["generator_path"])),
        "taskset": file_sha256(Path(plan["diagnostic_bank"]["taskset_source_path"])),
        "b1r_binding": file_sha256(Path(plan["b1r_dependency"]["binding_path"])),
    }
    protected = e33_pre == e33_post and file_pre == file_post and metadata_pre == metadata_post
    expected_immutable = {"plan": plan["_file_sha256"], **plan["immutable_input_hashes"]}
    provenance = immutable_post == expected_immutable
    resource = all(value <= CUDA_MEMORY_CAP_BYTES for item in audit["memory_ledger"] for key, value in item.items() if key.endswith("_bytes"))
    safety = {
        "backward": all((backward["residual_gradient"]["finite"], backward["residual_gradient"]["nonzero"], backward["encoder_group"]["finite"], backward["encoder_group"]["nonzero"], backward["receiver_group"]["finite"], backward["receiver_group"]["nonzero"], backward["all_named_gradients_finite"], backward["e33_gradients_absent"])),
        "provenance": provenance,
        "cache": cache_evidence["complete"]
        and cache_evidence["restored_in_finally"]
        and cache_evidence["closure_check_count"] == 147
        and cache_evidence["closure_checked_at_every_recorded_label"]
        and len(cache_evidence["classes"]) == 8,
        "protection": protected and codec_pre == codec_post,
        "resource": resource,
    }
    if not safety["backward"] or not safety["protection"]:
        raise PhaseBContractError("B-HIC0 gradient or protected-state evidence is incomplete")
    if not safety["provenance"]:
        raise ProvenanceContractViolated("B-HIC0 immutable input provenance changed")
    if not safety["cache"]:
        raise CacheContractViolated("B-HIC0 cache closure is incomplete")
    if not safety["resource"]:
        raise ResourceContractExceeded("B-HIC0 resource evidence is incomplete")
    nomination = evaluate_hic0(rows_evidence, safety=safety)
    if nomination["summaries"]["zero_drift_denominators"]:
        raise PhaseBContractError("B-HIC0 insertion drift denominator is zero")
    if not nomination["gates"]["1_complete_finite_safe"]:
        raise PhaseBContractError("B-HIC0 finite descriptive evidence is incomplete")
    _memory_checkpoint(torch, audit, "before_success")
    return {
        "schema_version": "q35-2b-phase-b-hic0-identity-carrier-success/v1",
        "status": nomination["disposition"],
        "terminal": "SUCCESS",
        "disposition": nomination["disposition"],
        "claim_class": "zero_update_identity_carrier_causal_diagnostic_nomination_only",
        "execution_commit": execution_commit,
        "saved_model_state": False,
        "B1R_candidates_reused": False,
        "optimizer": None,
        "optimizer_updates": 0,
        "generation": False,
        "cache": False,
        "worker_loaded": False,
        "H176_loaded": False,
        "strand_a_combined": False,
        "source_forwards": 12,
        "receiver_forwards": 60,
        "backward_forwards_reused": 1,
        "rows": rows_evidence,
        "backward": backward,
        "cache_guard": cache_evidence,
        "nomination": nomination,
        "protection": {"e33_tensor_pre": e33_pre, "e33_tensor_post": e33_post, "e33_file_pre": file_pre, "e33_file_post": file_post, "metadata_pre": metadata_pre, "metadata_post": metadata_post, "codec_pre": codec_pre, "codec_post": codec_post},
        "immutable_input_hashes": immutable_post,
        "allocator": audit["allocator"],
        "preflight_resources": plan["_preflight_resources"],
        "cuda_memory": _memory(torch),
        "cuda_memory_ledger": audit["memory_ledger"],
        "promotion": {"admitted": False, "diagnostic_rows_count_as_live_trajectories": False, "complete_live_trajectory_count": 0, "minimum_complete_live_trajectories_unchanged": 4},
    }


def _failure_class(error: BaseException) -> tuple[str, str]:
    if isinstance(error, CacheContractViolated):
        return "b_hic0_nocache_rejected", "scientific_cache_rejection"
    if isinstance(error, (ResourceContractExceeded, ProvenanceContractViolated, TimeoutError, OSError)):
        return "infrastructure_invalid", "infrastructure"
    if isinstance(error, PhaseBContractError):
        return "b_hic0_incomplete", "contract_or_evidence_incomplete"
    if isinstance(error, RuntimeError) or "out of memory" in str(error).lower():
        return "infrastructure_invalid", "infrastructure"
    return "infrastructure_invalid", "infrastructure"


def _failure_audit(plan: dict[str, Any], args: argparse.Namespace, audit: dict[str, Any]) -> dict[str, Any]:
    errors = []
    evidence: dict[str, Any] = {}
    try:
        smoke = _load_module(BR5_RUNNER, "phase_b_hic0_failure_br5")
        model = audit.get("model")
        if model is not None:
            evidence["e33_tensor_post"] = smoke._module_tensor_sha256(model, audit["torch"])
            evidence["e33_tensor_pre"] = audit.get("e33_pre")
            evidence["e33_tensor_reference_available"] = audit.get("e33_pre") is not None
            evidence["e33_tensor_preserved"] = (
                evidence["e33_tensor_post"] == audit["e33_pre"]
                if evidence["e33_tensor_reference_available"]
                else None
            )
        codec = audit.get("codec")
        if codec is not None:
            evidence["codec_tensor_post"] = smoke._module_tensor_sha256(codec, audit["torch"])
            evidence["codec_tensor_pre"] = audit.get("codec_pre")
            evidence["codec_tensor_preserved"] = evidence["codec_tensor_post"] == audit.get("codec_pre")
        model_path = Path(plan["protected_model"]["path"])
        evidence["e33_file_post"] = file_sha256(smoke._model_file(model_path))
        evidence["metadata_post"] = smoke._metadata_hashes(model_path)
        evidence["e33_file_pre"] = audit.get("file_pre")
        evidence["metadata_pre"] = audit.get("metadata_pre")
        evidence["e33_disk_and_metadata_preserved"] = (
            evidence["e33_file_post"],
            evidence["metadata_post"],
        ) == (audit.get("file_pre"), audit.get("metadata_pre"))
        evidence["e33_disk_and_metadata_exact"] = (
            evidence["e33_file_post"] == plan["protected_model"]["weight_sha256"]
            and evidence["metadata_post"] == plan["model_metadata_sha256"]
        )
        if model is not None and evidence["e33_tensor_reference_available"] and not evidence["e33_tensor_preserved"]:
            errors.append("e33_tensor_preserved: false")
        if codec is not None and not evidence["codec_tensor_preserved"]:
            errors.append("codec_tensor_preserved: false")
        if audit.get("file_pre") is not None and not evidence["e33_disk_and_metadata_preserved"]:
            errors.append("e33_disk_and_metadata_preserved: false")
        if not evidence["e33_disk_and_metadata_exact"]:
            errors.append("e33_disk_and_metadata_exact: false")
    except BaseException as error:
        errors.append(f"protected_hash: {type(error).__name__}: {error}")
    try:
        evidence["immutable_input_hashes"] = {
            "plan": file_sha256(args.plan),
            "selection": file_sha256(args.selection),
            "parquet": file_sha256(Path(plan["diagnostic_bank"]["parquet_path"])),
            "manifest": file_sha256(Path(plan["diagnostic_bank"]["manifest_path"])),
            "runtime": file_sha256(Path(plan["diagnostic_bank"]["runtime_source_path"])),
            "generator": file_sha256(Path(plan["diagnostic_bank"]["generator_path"])),
            "taskset": file_sha256(Path(plan["diagnostic_bank"]["taskset_source_path"])),
            "b1r_binding": file_sha256(Path(plan["b1r_dependency"]["binding_path"])),
        }
        evidence["immutable_input_hashes_match"] = evidence["immutable_input_hashes"] == {
            "plan": plan["_file_sha256"],
            **plan["immutable_input_hashes"],
        }
        if not evidence["immutable_input_hashes_match"]:
            errors.append("immutable_input_hashes_match: false")
    except BaseException as error:
        errors.append(f"immutable_hash: {type(error).__name__}: {error}")
    guard = audit.get("cache_guard_object")
    if guard is not None:
        try:
            evidence["cache_guard"] = guard.evidence()
        except BaseException as error:
            errors.append(f"cache_guard: {type(error).__name__}: {error}")
    torch = audit.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        try:
            evidence["cuda_memory"] = _memory(torch)
        except BaseException as error:
            errors.append(f"cuda_memory: {type(error).__name__}: {error}")
    return {"audit_complete": not errors, "hash_probe_error": "; ".join(errors) or None, **evidence}


def main() -> int:
    args = parse_args()
    started = time.time()
    audit: dict[str, Any] = {}
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(COMPUTE_SECONDS)
    try:
        plan, selection = preflight(args)
    except BaseException as error:
        signal.alarm(0)
        print(f"B-HIC0 preflight refusal: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    import pyarrow.parquet as parquet
    import transformers
    from transformers import AutoTokenizer
    try:
        context = tokenizer_preflight(plan, selection, parquet=parquet, AutoTokenizer=AutoTokenizer)
        if args.preflight_only:
            signal.alarm(0)
            print(json.dumps({"status": "B_HIC0_TOKENIZER_PREFLIGHT_ONLY_SUCCESS", "rows": len(context["rows"]), "model_loaded": False, "cuda_initialized": False, "output_created": False}, sort_keys=True))
            return 0
        args.output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        import torch
        from transformers import AutoModelForImageTextToText

        from prime_rl.latent.local_depth import LocalDepthCodec
        audit["torch"] = torch
        receipt = execute(plan, context, execution_commit=args.execution_commit, torch=torch, transformers=transformers, AutoModelForImageTextToText=AutoModelForImageTextToText, LocalDepthCodec=LocalDepthCodec, audit=audit)
        receipt.update({"model_loaded": True, "elapsed_seconds": time.time() - started, "plan_sha256": args.authorized_plan_sha256, "selection_sha256": file_sha256(args.selection), "wall_clock_contract": {"outer_seconds": OUTER_SECONDS, "compute_seconds": COMPUTE_SECONDS, "failure_audit_seconds": AUDIT_SECONDS, "terminal_publication_headroom_seconds": TERMINAL_SECONDS}})
        receipt["receipt_sha256"] = canonical_json_sha256(receipt, omitted_fields=("receipt_sha256",))
        validate_hic0_terminal_receipt(
            receipt,
            success_file=True,
            plan=plan,
            execution_commit=args.execution_commit,
        )
        signal.alarm(0)
        atomic_exclusive_json(args.output_dir, "SUCCESS.json", receipt, maximum_directory_bytes=ARTIFACT_CAP)
        return 0
    except BaseException as error:
        disposition, failure_class = _failure_class(error)
        if args.output_dir.is_dir():
            signal.alarm(AUDIT_SECONDS)
            post = _failure_audit(plan, args, audit)
            failure = {"schema_version": "q35-2b-phase-b-hic0-identity-carrier-failure/v1", "status": disposition, "terminal": "FAILURE", "disposition": disposition, "failure_class": failure_class, "error_type": type(error).__name__, "error": str(error), "elapsed_seconds": time.time() - started, "plan_sha256": args.authorized_plan_sha256, "execution_commit": args.execution_commit, "selection_sha256": file_sha256(args.selection), "model_loaded": audit.get("model") is not None, "saved_model_state": False, "B1R_candidates_reused": False, "optimizer": None, "optimizer_updates": 0, "generation": False, "cache": False, "worker_loaded": False, "H176_loaded": False, "strand_a_combined": False, "preflight_resources": plan["_preflight_resources"], "post_failure_hash_audit": post, "wall_clock_contract": {"outer_seconds": OUTER_SECONDS, "compute_seconds": COMPUTE_SECONDS, "failure_audit_seconds": AUDIT_SECONDS, "terminal_publication_headroom_seconds": TERMINAL_SECONDS}}
            failure["receipt_sha256"] = canonical_json_sha256(
                failure, omitted_fields=("receipt_sha256",)
            )
            validate_hic0_terminal_receipt(
                failure,
                success_file=False,
                plan=plan,
                execution_commit=args.execution_commit,
            )
            signal.alarm(0)
            atomic_exclusive_json(args.output_dir, "FAILURE.json", failure, maximum_directory_bytes=ARTIFACT_CAP)
        print(f"B-HIC0 failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
