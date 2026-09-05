#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import contextlib
import gc
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import time
from pathlib import Path
from unittest import mock

import torch
import transformers.cache_utils
from transformers import AutoModelForImageTextToText, AutoTokenizer

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a0nc import recursive_subclass_closure
from prime_rl.latent.a1cap768_flag0 import (
    AUTHORIZED_RUN_ID,
    COMPARISON_SCHEDULE,
    COMPARISON_SCHEDULE_SHA256,
    DECISION_BOUNDARY,
    FAILURE_SCHEMA,
    FIXTURE,
    FIXTURE_SHA256,
    FLAG_NAMES,
    FLAG_NAMES_SHA256,
    INTERPRETATION,
    OPERATION_SCHEDULE,
    OPERATION_SCHEDULE_SHA256,
    RECEIPT_SCHEMA,
    RESOURCE_BOUNDS,
    RUN4_FLAG_NAMES,
    RUN4_FLAG_NAMES_SHA256,
    RUN4_REJECTION_EVIDENCE,
    TRAIN_BANK_SHA256,
    DiagnosticIncomplete,
    NoCacheRejected,
    ResourceFitRejected,
    causal_interpretation,
    classify_failure,
    load_plan,
    memory_labels,
    validate_receipt,
)
from prime_rl.latent.a1nc0 import (
    _CACHE_CLASS_CLOSURE,
    module_state_tree_sha256,
    tensor_bytes_sha256,
    validate_bank_artifact,
)
from prime_rl.latent.policy_adapter import HiddenStateCaptureSpec, capture_parent_features

_CAP_RUNNER_PATH = Path(__file__).with_name("run_a1_nc0_cap768_v1.py")
_SPEC = importlib.util.spec_from_file_location("a1_nc0_cap768_frozen_runner", _CAP_RUNNER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("frozen CAP768 runner unavailable")
cap = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cap)
base = cap.base

OUTPUT_ROOT = Path(RESOURCE_BOUNDS["output_root"])


class ArtifactWriter:
    def __init__(self, output_dir: Path) -> None:
        if (
            not output_dir.is_absolute()
            or OUTPUT_ROOT.is_symlink()
            or not OUTPUT_ROOT.is_dir()
            or output_dir.parent.resolve(strict=True) != OUTPUT_ROOT.resolve(strict=True)
            or output_dir.name != AUTHORIZED_RUN_ID
            or output_dir.exists()
            or output_dir.is_symlink()
        ):
            raise ValueError("FLAG0 output namespace changed")
        output_dir.mkdir(mode=0o700)
        self.output_dir = output_dir
        self.terminal_written = False

    def write_terminal(self, name: str, payload: dict[str, object], maximum_bytes: int) -> None:
        if self.terminal_written or name not in {"receipt.json", "failure.json"}:
            raise ValueError("FLAG0 terminal is not exclusive")
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if len(encoded) > maximum_bytes:
            raise ValueError("FLAG0 terminal exceeds frozen bound")
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
                if written == 0:
                    raise OSError("short FLAG0 terminal write")
                view = view[written:]
            os.fsync(descriptor)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
        directory_fd = os.open(self.output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self.terminal_written = True


class MemoryLedger:
    def __init__(self) -> None:
        self.labels = memory_labels()
        self.rows: list[dict[str, object]] = []
        self.cap = RESOURCE_BOUNDS["allocator_cap_gib"] * 2**30

    def checkpoint(self, label: str) -> None:
        if len(self.rows) >= len(self.labels) or label != self.labels[len(self.rows)]:
            raise DiagnosticIncomplete("FLAG0 memory-label order changed")
        torch.cuda.synchronize(0)
        row = {
            "label": label,
            "allocated_bytes": torch.cuda.memory_allocated(0),
            "reserved_bytes": torch.cuda.memory_reserved(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        }
        self.rows.append(row)
        if any(row[key] > self.cap for key in row if key.endswith("_bytes")):
            raise ResourceFitRejected("FLAG0 exceeded 40GiB allocator cap")

    def validate_complete(self) -> None:
        if [row["label"] for row in self.rows] != self.labels:
            raise DiagnosticIncomplete("FLAG0 memory ledger incomplete")


class CacheGuard:
    def __init__(self) -> None:
        self.base = transformers.cache_utils.Cache
        self.initial = recursive_subclass_closure(self.base)
        self.patched: set[type] = set()
        self.stack = contextlib.ExitStack()
        self.negative = False
        self.checks = 0
        self.restored = False

    @staticmethod
    def _reject(cls, *_args, **_kwargs):
        raise NoCacheRejected(f"FLAG0 cache allocation attempted: {cls.__module__}.{cls.__qualname__}")

    def __enter__(self):
        try:
            for cls in sorted(self.initial, key=lambda item: (item.__module__, item.__qualname__)):
                self.stack.enter_context(mock.patch.object(cls, "__new__", self._reject))
                self.patched.add(cls)
            try:
                transformers.cache_utils.DynamicCache()
            except NoCacheRejected:
                self.negative = True
            if not self.negative:
                raise DiagnosticIncomplete("FLAG0 cache negative control did not trip")
            self.verify()
        except BaseException:
            self.stack.close()
            self.restored = True
            raise
        return self

    def verify(self) -> None:
        if recursive_subclass_closure(self.base) - self.patched:
            raise DiagnosticIncomplete("FLAG0 new unpatched cache subclass loaded")
        self.checks += 1

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.verify()
        finally:
            self.stack.close()
            self.restored = True
        return False

    def evidence(self) -> dict[str, object]:
        observed = [
            base.class_identity(cls)
            for cls in sorted(self.initial, key=lambda item: (item.__module__, item.__qualname__))
        ]
        if observed != _CACHE_CLASS_CLOSURE:
            raise ValueError("FLAG0 cache class provenance changed")
        return {
            "classes": observed,
            "negative_control_dynamic_cache_tripped": self.negative,
            "closure_check_count": self.checks,
            "restored_in_finally": self.restored,
        }


def _timed_cuda(call):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start.record()
    value = call()
    end.record()
    end.synchronize()
    return value, start.elapsed_time(end) / 1000.0, time.perf_counter() - wall_started


def _operation_timed(name: str, kind: str, call, ledger: MemoryLedger):
    ledger.checkpoint(f"pre_{name}")
    value, cuda_seconds, wall_seconds = _timed_cuda(call)
    ledger.checkpoint(f"post_{name}")
    return value, {
        "operation_index": next(item["operation_index"] for item in OPERATION_SCHEDULE if item["name"] == name),
        "kind": kind,
        "name": name,
        "cuda_event_seconds": cuda_seconds,
        "wall_seconds": wall_seconds,
    }


def _forward(model, *, input_ids, inputs_embeds, mask, positions, logits_to_keep: int, arm: str, ledger, guard):
    ledger.checkpoint(f"pre_{arm}")
    guard.verify()
    if hasattr(model, "model") and hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None
        if model.model.rope_deltas is not None:
            raise NoCacheRejected("FLAG0 RoPE recurrence state did not reset")

    def invoke():
        with torch.inference_mode():
            return model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=None,
                use_cache=False,
                output_hidden_states=True,
                logits_to_keep=logits_to_keep,
                return_dict=True,
            )

    output, cuda_seconds, wall_seconds = _timed_cuda(invoke)
    if getattr(output, "past_key_values", None) is not None:
        raise NoCacheRejected("FLAG0 forward returned past_key_values")
    guard.verify()
    ledger.checkpoint(f"post_{arm}")
    return output, {
        "operation_index": next(item["operation_index"] for item in OPERATION_SCHEDULE if item["name"] == arm),
        "kind": "model_forward",
        "name": arm,
        "cuda_event_seconds": cuda_seconds,
        "wall_seconds": wall_seconds,
    }


def _capture(output, mask):
    hidden = output.hidden_states[-1]
    captured = capture_parent_features(hidden, mask, HiddenStateCaptureSpec())
    expected_indices = list(range(640, 768))
    if (
        hidden.shape != (1, 768, 2048)
        or captured.hidden_states.shape != (1, 128, 2048)
        or captured.token_indices.tolist() != expected_indices
        or not torch.equal(
            captured.attention_mask,
            torch.ones((1, 128), dtype=mask.dtype, device=mask.device),
        )
        or not torch.equal(captured.hidden_states, hidden[:, 640:768, :])
    ):
        raise DiagnosticIncomplete("FLAG0 hidden capture geometry/content changed")
    return captured


def _tensor_evidence(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": tensor_bytes_sha256(tensor),
    }


def _comparison(spec: dict[str, object], lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, object]:
    if lhs.shape != rhs.shape:
        raise DiagnosticIncomplete(f"FLAG0 comparison shape changed: {spec['name']}")
    left = lhs.detach().cpu().contiguous()
    right = rhs.detach().cpu().contiguous()
    mismatch_tensor = left != right
    mismatch_indices = torch.nonzero(mismatch_tensor.reshape(-1), as_tuple=False)
    mismatch_count = int(mismatch_indices.shape[0])
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if finite and left.numel() > 0:
        difference = left.to(torch.float64) - right.to(torch.float64)
        max_abs = float(difference.abs().max().item())
        rms_diff = float(torch.sqrt(torch.mean(difference.square())).item())
        rhs_rms = float(torch.sqrt(torch.mean(right.to(torch.float64).square())).item())
        normalized_rms = rms_diff / max(rhs_rms, 1e-12)
        if not all(math.isfinite(item) for item in (max_abs, rms_diff, rhs_rms, normalized_rms)):
            raise DiagnosticIncomplete("FLAG0 finite comparison produced nonfinite metric")
    else:
        max_abs = rms_diff = rhs_rms = normalized_rms = None
    return {
        **spec,
        "lhs_dtype": str(left.dtype),
        "rhs_dtype": str(right.dtype),
        "lhs_shape": list(left.shape),
        "rhs_shape": list(right.shape),
        "lhs_sha256": tensor_bytes_sha256(left),
        "rhs_sha256": tensor_bytes_sha256(right),
        "torch_equal": torch.equal(left, right),
        "element_count": left.numel(),
        "mismatch_count": mismatch_count,
        "count_nonzero": int(torch.count_nonzero(mismatch_tensor).item()),
        "first_flat_mismatch": None if mismatch_count == 0 else int(mismatch_indices[0, 0].item()),
        "metrics_defined": finite and left.numel() > 0,
        "max_abs": max_abs,
        "rms_diff": rms_diff,
        "rhs_rms": rhs_rms,
        "normalized_rms": normalized_rms,
    }


def _find_fixture(bank: dict[str, object]):
    record = next(
        (item for item in bank["records"] if item["evidence_id"] == FIXTURE["evidence_id"]),
        None,
    )
    query = (
        None
        if record is None
        else next(
            (item for item in record["queries"] if item["query_id"] == FIXTURE["query_id"]),
            None,
        )
    )
    if record is None or query is None or record["family"] != FIXTURE["family"]:
        raise DiagnosticIncomplete("FLAG0 frozen fixture changed")
    return record, query


def _static_guard(path: Path) -> dict[str, object]:
    source = path.read_text()
    tree = ast.parse(source)
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id
                if isinstance(node.func, ast.Name)
                else ""
            )
            if name in {"generate", "backward", "step"}:
                forbidden.append(name)
    if forbidden or "WorkspaceBridge" in source or "AdamW" in source:
        raise DiagnosticIncomplete("FLAG0 forbidden training/generation source appeared")
    return {"runner_sha256": file_sha256(path), "forbidden_calls": forbidden}


def _physical_gpu_audit() -> dict[str, object]:
    names = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    memory = [
        int(value)
        for value in subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
    ]
    uuids = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    evidence = {"names": names, "uuids": uuids, "memory_used_mib": memory, "compute_apps": apps}
    if (
        names != ["NVIDIA RTX A6000", "NVIDIA RTX A6000"]
        or len(memory) != 2
        or len(uuids) != 2
        or memory[1] > 512
        or any(line.strip().startswith(uuids[1]) for line in apps)
    ):
        raise ValueError("FLAG0 physical GPU inventory/GPU1 idle evidence changed")
    return evidence


def _run_case(model, tokenizer, record, ledger: MemoryLedger, guard: CacheGuard, stage):
    stage["operation_timings"] = []
    try:
        ids = base.render_ids(
            tokenizer,
            base.parent_messages(record["parent_evidence"]),
            generation_prompt=False,
            tools=base.PARENT_TOOLS,
        )
    except base.ExperimentIncomplete as error:
        raise DiagnosticIncomplete(f"FLAG0 operational render failed: {error}") from error
    if ids.device.type != "cuda" or ids.dtype != torch.int64 or ids.shape != (1, 517) or not ids.is_contiguous():
        raise DiagnosticIncomplete("FLAG0 selected transcript tensor changed")
    pad = 768 - ids.shape[1]
    padded = torch.nn.functional.pad(ids, (pad, 0), value=248046)
    mask = torch.nn.functional.pad(torch.ones_like(ids), (pad, 0), value=0)
    positions = torch.arange(768, device="cuda:0").unsqueeze(0)
    if (
        pad < 0
        or not torch.all(padded[:, :pad] == 248046).item()
        or not torch.equal(padded[:, pad:], ids)
        or torch.count_nonzero(mask[:, :pad]).item() != 0
        or not torch.all(mask[:, pad:] == 1).item()
        or not torch.equal(positions, torch.arange(768, device="cuda:0").unsqueeze(0))
    ):
        raise DiagnosticIncomplete("FLAG0 fixed left-pad geometry changed")
    prefix = "CAP768_FLAG_P01_PARENT"
    embed_name = f"{prefix}_EXACT_EMBED_LOOKUP"
    ledger.checkpoint(f"pre_{embed_name}")
    embedding_call = lambda: model.get_input_embeddings()(padded)
    exact, embed_cuda, embed_wall = _timed_cuda(embedding_call)
    ledger.checkpoint(f"post_{embed_name}")
    embed_timing = {
        "operation_index": 1,
        "kind": "embedding_lookup",
        "name": embed_name,
        "cuda_event_seconds": embed_cuda,
        "wall_seconds": embed_wall,
    }
    stage["operation_timings"].append(embed_timing)
    if not torch.isfinite(exact).all() or exact.requires_grad:
        raise DiagnosticIncomplete("FLAG0 exact embedding lookup produced nonfinite values")
    exact_before = tensor_bytes_sha256(exact)
    arms = [f"{prefix}_{name}" for name in ("L_ID_KEEP1", "L_E_KEEP1", "L_E_REPEAT_KEEP1", "L_ID_KEEP0_CONTROL")]
    id1, id1_timing = _forward(
        model,
        input_ids=padded,
        inputs_embeds=None,
        mask=mask,
        positions=positions,
        logits_to_keep=1,
        arm=arms[0],
        ledger=ledger,
        guard=guard,
    )
    stage["operation_timings"].append(id1_timing)
    e1, e1_timing = _forward(
        model,
        input_ids=None,
        inputs_embeds=exact,
        mask=mask,
        positions=positions,
        logits_to_keep=1,
        arm=arms[1],
        ledger=ledger,
        guard=guard,
    )
    stage["operation_timings"].append(e1_timing)
    repeated_object_id = id(exact)
    e2, e2_timing = _forward(
        model,
        input_ids=None,
        inputs_embeds=exact,
        mask=mask,
        positions=positions,
        logits_to_keep=1,
        arm=arms[2],
        ledger=ledger,
        guard=guard,
    )
    stage["operation_timings"].append(e2_timing)
    id0, id0_timing = _forward(
        model,
        input_ids=padded,
        inputs_embeds=None,
        mask=mask,
        positions=positions,
        logits_to_keep=0,
        arm=arms[3],
        ledger=ledger,
        guard=guard,
    )
    stage["operation_timings"].append(id0_timing)
    outputs = (id1, e1, e2, id0)
    captures = tuple(_capture(output, mask) for output in outputs)
    if (
        id1.logits.shape != (1, 1, model.config.text_config.vocab_size)
        or e1.logits.shape != id1.logits.shape
        or e2.logits.shape != id1.logits.shape
        or id0.logits.shape != (1, 768, model.config.text_config.vocab_size)
        or not all(torch.isfinite(output.logits).all() for output in outputs)
    ):
        raise DiagnosticIncomplete("FLAG0 logits shape/finiteness changed")
    flags = {
        "left_padding_exact": True,
        "attention_mask_exact": True,
        "position_ids_exact": True,
        "no_truncation": True,
        "id_embed_keep1_logits_bitwise": torch.equal(id1.logits, e1.logits),
        "id_embed_keep1_full_hidden_bitwise": torch.equal(id1.hidden_states[-1], e1.hidden_states[-1]),
        "id_embed_keep1_capture_bitwise": torch.equal(captures[0].hidden_states, captures[1].hidden_states),
        "repeat_same_embedding_object": id(exact) == repeated_object_id,
        "repeat_embedding_unchanged": tensor_bytes_sha256(exact) == exact_before,
        "repeat_logits_bitwise": torch.equal(e1.logits, e2.logits),
        "repeat_full_hidden_bitwise": torch.equal(e1.hidden_states[-1], e2.hidden_states[-1]),
        "repeat_capture_bitwise": torch.equal(captures[1].hidden_states, captures[2].hidden_states),
        "keep0_keep1_full_hidden_bitwise": torch.equal(id1.hidden_states[-1], id0.hidden_states[-1]),
        "keep0_keep1_capture_bitwise": torch.equal(captures[0].hidden_states, captures[3].hidden_states),
        "keep0_last_logits_keep1_bitwise": torch.equal(id1.logits, id0.logits[:, -1:]),
        "all_outputs_finite": all(torch.isfinite(output.hidden_states[-1]).all() for output in outputs),
    }
    projection_timings = []
    projection_tensors = []
    for suffix, hidden in (
        ("PROJ_ID1_LAST", id1.hidden_states[-1][:, -1:, :]),
        ("PROJ_ID0_LAST", id0.hidden_states[-1][:, -1:, :]),
    ):
        name = f"{prefix}_{suffix}"

        def project(hidden=hidden):
            with torch.inference_mode():
                return model.lm_head(hidden)

        projected, timing = _operation_timed(
            name,
            "lm_head_projection",
            project,
            ledger,
        )
        projection_tensors.append(projected)
        projection_timings.append(timing)
        stage["operation_timings"].append(timing)
    proj_id1, proj_id0 = projection_tensors
    flags.update(
        {
            "all_output_logits_finite": all(torch.isfinite(output.logits).all() for output in outputs)
            and torch.isfinite(proj_id1).all().item()
            and torch.isfinite(proj_id0).all().item(),
            "all_output_full_hidden_finite": all(torch.isfinite(output.hidden_states[-1]).all() for output in outputs),
            "all_capture_finite": all(torch.isfinite(item.hidden_states).all() for item in captures),
            "exact_embeddings_finite": torch.isfinite(exact).all().item(),
            "exact_embeddings_requires_grad_false": not exact.requires_grad,
            "proj_id1_matches_id1_logits_bitwise": torch.equal(proj_id1, id1.logits),
            "proj_id0_matches_id0_last_logits_bitwise": torch.equal(proj_id0, id0.logits[:, -1:]),
            "proj_id1_proj_id0_bitwise": torch.equal(proj_id1, proj_id0),
            "id1_logits_proj_id0_bitwise": torch.equal(id1.logits, proj_id0),
        }
    )
    if list(flags) != FLAG_NAMES or any(not isinstance(value, bool) for value in flags.values()):
        raise DiagnosticIncomplete("FLAG0 boolean flag schema changed")
    stage["flags"] = flags
    tensors = {
        "exact_embeddings": exact,
        "L_ID_KEEP1.logits": id1.logits,
        "L_ID_KEEP1.hidden": id1.hidden_states[-1],
        "L_ID_KEEP1.capture": captures[0].hidden_states,
        "L_E_KEEP1.logits": e1.logits,
        "L_E_KEEP1.hidden": e1.hidden_states[-1],
        "L_E_KEEP1.capture": captures[1].hidden_states,
        "L_E_REPEAT_KEEP1.logits": e2.logits,
        "L_E_REPEAT_KEEP1.hidden": e2.hidden_states[-1],
        "L_E_REPEAT_KEEP1.capture": captures[2].hidden_states,
        "L_ID_KEEP0_CONTROL.logits": id0.logits,
        "L_ID_KEEP0_CONTROL.last_logits": id0.logits[:, -1:],
        "L_ID_KEEP0_CONTROL.hidden": id0.hidden_states[-1],
        "L_ID_KEEP0_CONTROL.capture": captures[3].hidden_states,
        "PROJ_ID1_LAST.logits": proj_id1,
        "PROJ_ID0_LAST.logits": proj_id0,
    }
    comparisons = []
    stage["comparisons"] = comparisons
    for spec in COMPARISON_SCHEDULE:
        comparisons.append(_comparison(spec, tensors[spec["lhs"]], tensors[spec["rhs"]]))
    tensor_evidence = {name: _tensor_evidence(tensor) for name, tensor in tensors.items()}
    operation_timings = [embed_timing, id1_timing, e1_timing, e2_timing, id0_timing, *projection_timings]
    if [item["name"] for item in operation_timings] != [item["name"] for item in OPERATION_SCHEDULE]:
        raise DiagnosticIncomplete("FLAG0 operation order changed")
    input_evidence = {
        "rendered_ids_shape": list(ids.shape),
        "rendered_ids_dtype": str(ids.dtype),
        "rendered_ids_contiguous": ids.is_contiguous(),
        "rendered_ids_sha256": tensor_bytes_sha256(ids),
        "padded_ids_sha256": tensor_bytes_sha256(padded),
        "attention_mask_sha256": tensor_bytes_sha256(mask),
        "position_ids_sha256": tensor_bytes_sha256(positions),
        "capture_mask_sha256": tensor_bytes_sha256(captures[0].attention_mask),
    }
    return flags, comparisons, tensor_evidence, input_evidence, operation_timings


def run(args, plan, stage):
    if not args.owner_approved:
        raise ValueError("FLAG0 requires root approval")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    parent = subprocess.run(
        ["git", "rev-parse", f"{head}^"], cwd=args.repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=args.repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if head != args.execution_commit or parent != plan["mechanism_code_commit"] or status:
        raise ValueError("FLAG0 execution tree changed")
    if file_sha256(args.train_bank) != TRAIN_BANK_SHA256:
        raise DiagnosticIncomplete("FLAG0 train bank bytes changed")
    try:
        bank = validate_bank_artifact(args.train_bank, "train")["bank"]
        record, _query = _find_fixture(bank)
    except Exception as error:
        raise DiagnosticIncomplete(f"FLAG0 bank/fixture validation failed: {error}") from error
    versions = {
        "python": platform.python_version(),
        "transformers": importlib.metadata.version("transformers"),
        "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
        "torch_distribution": importlib.metadata.version("torch"),
        "torch_runtime": str(torch.__version__),
    }
    if versions != {key: plan["runtime"][key] for key in versions}:
        raise ValueError("FLAG0 runtime changed")
    runtime_sources = base.source_hashes()
    if {name: item["sha256"] for name, item in runtime_sources.items()} != plan["runtime"][
        "transformers_source_sha256"
    ]:
        raise ValueError("FLAG0 runtime source identity changed")
    weights = {
        "coordinator_e33": base.model_weight(args.coordinator),
        "worker_h176": base.model_weight(args.worker),
    }
    protected_before = {name: file_sha256(path) for name, path in weights.items()}
    metadata_before = {
        name: base.metadata_hashes(path)
        for name, path in (("coordinator_e33", args.coordinator), ("worker_h176", args.worker))
    }
    if protected_before != plan["protected_checkpoints"] or any(
        value != plan["runtime"]["checkpoint_metadata_sha256"] for value in metadata_before.values()
    ):
        raise ValueError("FLAG0 protected preflight changed")
    stage.update({"weights": weights, "protected_before": protected_before, "metadata_before": metadata_before})
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != RESOURCE_BOUNDS["gpu_model"]
    ):
        raise ValueError("FLAG0 visible GPU changed")
    properties = torch.cuda.get_device_properties(0)
    free_disk = shutil.disk_usage(OUTPUT_ROOT).free
    host_ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    if (
        properties.total_memory < RESOURCE_BOUNDS["minimum_gpu_memory_gib"] * 2**30
        or free_disk < RESOURCE_BOUNDS["minimum_free_disk_gib"] * 2**30
        or host_ram < RESOURCE_BOUNDS["minimum_host_ram_gib"] * 2**30
    ):
        raise ValueError("FLAG0 host resources changed")
    physical_before = _physical_gpu_audit()
    cap_bytes = RESOURCE_BOUNDS["allocator_cap_gib"] * 2**30
    torch.cuda.set_per_process_memory_fraction(cap_bytes / properties.total_memory, 0)
    torch.cuda.reset_peak_memory_stats(0)
    tokenizer_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.coordinator, local_files_only=True)
    tokenizer_seconds = time.perf_counter() - tokenizer_started
    model_load_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.coordinator,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda:0")
    model.eval()
    model.config.use_cache = False
    model.generation_config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_load_seconds = time.perf_counter() - model_load_started
    stage["model"] = model
    state_before = module_state_tree_sha256(model)
    stage["state_before"] = state_before
    ledger = MemoryLedger()
    stage["ledger"] = ledger
    ledger.checkpoint("model_loaded_frozen")
    guard = CacheGuard()
    stage["guard"] = guard
    try:
        with guard:
            flags, comparisons, tensor_evidence, input_evidence, operation_timings = _run_case(
                model, tokenizer, record, ledger, guard, stage
            )
            guard.verify()
    except (NoCacheRejected, ResourceFitRejected, TimeoutError, torch.cuda.OutOfMemoryError):
        raise
    except DiagnosticIncomplete:
        raise
    except Exception as error:
        raise DiagnosticIncomplete(f"FLAG0 diagnostic operation failed: {error}") from error
    compute_seconds = time.perf_counter() - stage["compute_started"]
    if compute_seconds > RESOURCE_BOUNDS["compute_seconds"]:
        raise ResourceFitRejected("FLAG0 compute timeout")
    signal.signal(signal.SIGALRM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError("FLAG0 audit timeout")))
    signal.alarm(RESOURCE_BOUNDS["audit_seconds"])
    audit_started = time.perf_counter()
    ledger.checkpoint("cache_guard_audit_complete")
    cache_evidence = guard.evidence()
    state_after = module_state_tree_sha256(model)
    protected_after = {name: file_sha256(path) for name, path in weights.items()}
    metadata_after = {
        name: base.metadata_hashes(path)
        for name, path in (("coordinator_e33", args.coordinator), ("worker_h176", args.worker))
    }
    if (
        state_after != state_before
        or protected_after != protected_before
        or metadata_after != metadata_before
        or any(parameter.grad is not None for parameter in model.parameters())
    ):
        raise ValueError("FLAG0 protected model changed")
    ledger.checkpoint("protected_postflight_complete")
    ledger.validate_complete()
    physical_after = _physical_gpu_audit()
    audit_seconds = time.perf_counter() - audit_started
    if audit_seconds > RESOURCE_BOUNDS["audit_seconds"]:
        raise TimeoutError("FLAG0 audit timeout")
    run4_reproduced = not all(flags[name] for name in RUN4_FLAG_NAMES)
    completed_status = (
        "capture768_flag_isolation_complete" if run4_reproduced else "capture768_flag_isolation_nonreproduced"
    )
    cuda_times = [float(item["cuda_event_seconds"]) for item in operation_timings]
    wall_times = [float(item["wall_seconds"]) for item in operation_timings]
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": completed_status,
        "plan_sha256": plan["plan_sha256"],
        "mechanism_code_commit": plan["mechanism_code_commit"],
        "execution_commit": args.execution_commit,
        "asset_sha256": plan["asset_sha256"],
        "run_id": args.output_dir.name,
        "fixture": FIXTURE,
        "fixture_sha256": FIXTURE_SHA256,
        "train_bank_sha256": TRAIN_BANK_SHA256,
        "operation_schedule": OPERATION_SCHEDULE,
        "operation_schedule_sha256": OPERATION_SCHEDULE_SHA256,
        "operation_counts": {
            "embedding_lookup": 1,
            "e33_forward": 4,
            "lm_head_projection": 2,
            "capture": 4,
            "generation": 0,
            "h176_forward": 0,
            "bridge": 0,
            "optimizer": 0,
            "backward": 0,
            "step": 0,
            "checkpoint": 0,
            "candidate": 0,
        },
        "flag_names": FLAG_NAMES,
        "flag_names_sha256": FLAG_NAMES_SHA256,
        "run4_flag_names": RUN4_FLAG_NAMES,
        "run4_flag_names_sha256": RUN4_FLAG_NAMES_SHA256,
        "flags": flags,
        "run4_aggregate_reproduced": run4_reproduced,
        "comparison_schedule": COMPARISON_SCHEDULE,
        "comparison_schedule_sha256": COMPARISON_SCHEDULE_SHA256,
        "comparisons": comparisons,
        "tensor_evidence": tensor_evidence,
        "input_evidence": input_evidence,
        "run4_rejection_evidence": RUN4_REJECTION_EVIDENCE,
        "versions": versions,
        "runtime_sources": runtime_sources,
        "static_guard": _static_guard(Path(__file__)),
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "checkpoint_metadata_before": metadata_before,
        "checkpoint_metadata_after": metadata_after,
        "e33_state_tree_before": state_before,
        "e33_state_tree_after": state_after,
        "e33_parameters_frozen_no_grad": True,
        "worker_h176_loaded": False,
        "model_runtime": {
            "class": model.__class__.__name__,
            "hidden_size": model.config.text_config.hidden_size,
            "vocab_size": model.config.text_config.vocab_size,
            "dtype": str(next(model.parameters()).dtype),
            "device": str(next(model.parameters()).device),
        },
        "no_cache_contract": {
            "calls": 4,
            "use_cache_false": True,
            "pkv_input_none": True,
            "pkv_output_none": True,
            "rope_reset_every_call": True,
            "model_config_use_cache": model.config.use_cache,
            "generation_config_use_cache": model.generation_config.use_cache,
        },
        "cache_guard": cache_evidence,
        "memory_ledger": ledger.rows,
        "memory_labels_sha256": canonical_json_hash([row["label"] for row in ledger.rows]),
        "resources": {
            "gpu_name": torch.cuda.get_device_name(0),
            "total_gpu_memory_bytes": properties.total_memory,
            "allocator_cap_bytes": cap_bytes,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "host_ram_bytes": host_ram,
            "free_disk_bytes_preflight": free_disk,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "network_disabled": {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            },
            "physical_gpu_before": physical_before,
            "physical_gpu_after": physical_after,
            "physical_gpu1_unused_before_after": True,
        },
        "timings": {
            "operations": operation_timings,
            "operation_cuda_event_seconds_sum": math.fsum(cuda_times),
            "operation_wall_seconds_sum": math.fsum(wall_times),
            "tokenizer_load_seconds": tokenizer_seconds,
            "model_load_seconds": model_load_seconds,
            "compute_seconds": compute_seconds,
            "audit_seconds": audit_seconds,
            "total_seconds": time.perf_counter() - stage["compute_started"],
        },
        "decision_boundary": {
            **DECISION_BOUNDARY,
            "causal_interpretation": causal_interpretation(flags),
        },
        "interpretation_boundary": INTERPRETATION,
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    validate_receipt(receipt, plan=plan)
    return receipt


def failure_record(args, error, plan, stage):
    status, category = classify_failure(error)
    model = stage.get("model")
    audit_errors = []
    protected_after = {}
    metadata_after = {}
    for name, path in stage.get("weights", {}).items():
        try:
            protected_after[name] = file_sha256(path)
        except BaseException as audit_error:
            audit_errors.append(f"weight:{name}:{type(audit_error).__name__}:{audit_error}")
    for name, path in (("coordinator_e33", args.coordinator), ("worker_h176", args.worker)):
        try:
            metadata_after[name] = base.metadata_hashes(path)
        except BaseException as audit_error:
            audit_errors.append(f"metadata:{name}:{type(audit_error).__name__}:{audit_error}")
    state_after = None
    gradients_absent = None
    if model is not None:
        try:
            state_after = module_state_tree_sha256(model)
            gradients_absent = all(parameter.grad is None for parameter in model.parameters())
        except BaseException as audit_error:
            audit_errors.append(f"model:{type(audit_error).__name__}:{audit_error}")
    guard = stage.get("guard")
    cache_partial = None
    if isinstance(guard, CacheGuard):
        try:
            cache_partial = guard.evidence()
        except BaseException as audit_error:
            audit_errors.append(f"cache:{type(audit_error).__name__}:{audit_error}")
    asset_probe = {}
    if plan is not None:
        for relative in plan.get("asset_sha256", {}):
            try:
                asset_probe[relative] = file_sha256(args.repo / relative)
            except BaseException as audit_error:
                audit_errors.append(f"asset:{relative}:{type(audit_error).__name__}:{audit_error}")
    ledger = stage.get("ledger")
    failure = {
        "schema_version": FAILURE_SCHEMA,
        "status": status,
        "failure_category": category,
        "error_type": type(error).__name__,
        "error": str(error),
        "execution_commit": args.execution_commit,
        "mechanism_code_commit": None if plan is None else plan.get("mechanism_code_commit"),
        "plan_sha256": None if plan is None else plan.get("plan_sha256"),
        "run_id": args.output_dir.name,
        "model_loaded": model is not None,
        "model_update_attempted": False,
        "bridge_created": False,
        "optimizer_created": False,
        "backward_used": False,
        "checkpoint_created": False,
        "candidate_created": False,
        "worker_h176_loaded": False,
        "run4_rejection_evidence": RUN4_REJECTION_EVIDENCE,
        "decision_boundary": DECISION_BOUNDARY,
        "flags_partial": stage.get("flags"),
        "comparisons_partial": stage.get("comparisons"),
        "operation_timings_partial": stage.get("operation_timings"),
        "protected_hashes_before": stage.get("protected_before"),
        "protected_hash_probe_after_failure": protected_after,
        "checkpoint_metadata_before": stage.get("metadata_before"),
        "checkpoint_metadata_probe_after_failure": metadata_after,
        "e33_state_tree_before": stage.get("state_before"),
        "e33_state_tree_failure_audit": state_after,
        "e33_gradients_absent_failure_audit": gradients_absent,
        "cache_guard_partial": cache_partial,
        "memory_ledger_partial": [] if not isinstance(ledger, MemoryLedger) else ledger.rows,
        "asset_hash_probe_after_failure": asset_probe,
        "asset_hashes_match_plan": plan is not None and asset_probe == plan.get("asset_sha256"),
        "failure_audit_errors": audit_errors,
        "failure_audit_bounded_seconds": RESOURCE_BOUNDS["failure_audit_seconds"],
        "failure_sha256": "",
    }
    failure["failure_sha256"] = canonical_json_hash(failure, omitted_fields=("failure_sha256",))
    return failure


def main():
    parser = argparse.ArgumentParser()
    for name in ("repo", "plan", "coordinator", "worker", "train_bank", "output_dir"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    writer = ArtifactWriter(args.output_dir)
    plan = None
    stage = {"compute_started": time.perf_counter()}
    signal.signal(
        signal.SIGALRM,
        lambda _s, _f: (_ for _ in ()).throw(ResourceFitRejected("FLAG0 compute timeout")),
    )
    signal.alarm(RESOURCE_BOUNDS["compute_seconds"])
    try:
        plan = load_plan(args.plan, args.repo)
        receipt = run(args, plan, stage)
        signal.alarm(RESOURCE_BOUNDS["terminal_seconds"])
        writer.write_terminal("receipt.json", receipt, RESOURCE_BOUNDS["maximum_receipt_bytes"])
    except torch.cuda.OutOfMemoryError as error:
        signal.alarm(RESOURCE_BOUNDS["failure_audit_seconds"])
        wrapped = ResourceFitRejected(f"FLAG0 CUDA OOM: {error}")
        failure = failure_record(args, wrapped, plan, stage)
        signal.alarm(RESOURCE_BOUNDS["terminal_seconds"])
        writer.write_terminal("failure.json", failure, RESOURCE_BOUNDS["maximum_failure_bytes"])
        raise wrapped from error
    except BaseException as error:
        signal.alarm(RESOURCE_BOUNDS["failure_audit_seconds"])
        failure = failure_record(args, error, plan, stage)
        signal.alarm(RESOURCE_BOUNDS["terminal_seconds"])
        writer.write_terminal("failure.json", failure, RESOURCE_BOUNDS["maximum_failure_bytes"])
        raise
    finally:
        signal.alarm(0)
        gc.collect()


if __name__ == "__main__":
    main()
