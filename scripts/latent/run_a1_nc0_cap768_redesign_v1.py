#!/usr/bin/env python3
"""Prospective no-update CAP768 carrier redesign diagnostic."""

from __future__ import annotations

import argparse
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
from prime_rl.latent.a1cap768_redesign import (
    _CACHE_CLASS_CLOSURE,
    AUTHORIZED_RUN_ID,
    CASE_SCHEDULE,
    CASE_SCHEDULE_SHA256,
    COMPARISON_SCHEDULE_SHA256,
    DECISION_BOUNDARY,
    DESCRIPTIVE_FLAG_NAMES,
    DESCRIPTIVE_FLAG_NAMES_SHA256,
    FAILURE_SCHEMA,
    FLAG0_INCOMPLETE_EVIDENCE,
    FLAG_NAMES,
    FLAG_NAMES_SHA256,
    GATING_FLAG_NAMES,
    GATING_FLAG_NAMES_SHA256,
    INTERPRETATION,
    OPERATION_SCHEDULE_SHA256,
    RECEIPT_SCHEMA,
    RESOURCE_BOUNDS,
    SELECTION,
    SELECTION_SHA256,
    DiagnosticIncomplete,
    NoCacheRejected,
    ResourceFitRejected,
    build_comparison_schedule,
    build_operation_schedule,
    classification,
    classify_failure,
    load_plan,
    memory_labels,
    validate_receipt,
)
from prime_rl.latent.a1nc0 import module_state_tree_sha256, tensor_bytes_sha256, validate_bank_artifact
from prime_rl.latent.cap768_redesign_invariants import inspect_no_training_runner
from prime_rl.latent.policy_adapter import HiddenStateCaptureSpec, capture_parent_features

_CAP_PATH = Path(__file__).with_name("run_a1_nc0_cap768_v1.py")
_SPEC = importlib.util.spec_from_file_location("cap768_frozen_runner_for_redesign", _CAP_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("frozen CAP768 runner unavailable")
cap = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cap)
base = cap.base

OUTPUT_ROOT = Path(RESOURCE_BOUNDS["output_root"])
OPERATIONS = build_operation_schedule()
COMPARISONS = build_comparison_schedule()


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
            raise ValueError("CAP768R output namespace changed")
        output_dir.mkdir(mode=0o700)
        self.output_dir = output_dir
        self.terminal_written = False

    def write_terminal(self, name: str, payload: dict[str, object], maximum_bytes: int) -> None:
        if self.terminal_written or name not in {"receipt.json", "failure.json"}:
            raise ValueError("CAP768R terminal is not exclusive")
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if len(encoded) > maximum_bytes:
            raise ValueError("CAP768R terminal exceeds frozen bound")
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
                    raise OSError("short CAP768R terminal write")
                view = view[written:]
            os.fsync(descriptor)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
        descriptor = os.open(self.output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.terminal_written = True


class MemoryLedger:
    def __init__(self) -> None:
        self.labels = memory_labels()
        self.rows: list[dict[str, object]] = []
        self.cap = RESOURCE_BOUNDS["allocator_cap_gib"] * 2**30

    def checkpoint(self, label: str) -> None:
        if len(self.rows) >= len(self.labels) or label != self.labels[len(self.rows)]:
            raise DiagnosticIncomplete("CAP768R memory-label order changed")
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
            raise ResourceFitRejected("CAP768R exceeded 40GiB allocator cap")


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
        raise NoCacheRejected(f"CAP768R cache allocation: {cls.__module__}.{cls.__qualname__}")

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
                raise DiagnosticIncomplete("CAP768R cache negative control did not trip")
            self.verify()
        except BaseException:
            self.stack.close()
            self.restored = True
            raise
        return self

    def verify(self) -> None:
        if recursive_subclass_closure(self.base) - self.patched:
            raise DiagnosticIncomplete("CAP768R unpatched cache subclass loaded")
        self.checks += 1

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.verify()
        finally:
            self.stack.close()
            self.restored = True
        return False

    def evidence(self) -> dict[str, object]:
        classes = [
            base.class_identity(cls)
            for cls in sorted(self.initial, key=lambda item: (item.__module__, item.__qualname__))
        ]
        if classes != _CACHE_CLASS_CLOSURE:
            raise ValueError("CAP768R cache provenance changed")
        return {
            "classes": classes,
            "negative_control_dynamic_cache_tripped": self.negative,
            "closure_check_count": self.checks,
            "restored_in_finally": self.restored,
        }


def _timed(call):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    value = call()
    end.record()
    end.synchronize()
    return value, start.elapsed_time(end) / 1000.0, time.perf_counter() - wall_start


def _operation(spec, call, ledger):
    ledger.checkpoint(f"pre_{spec['name']}")
    value, cuda_seconds, wall_seconds = _timed(call)
    ledger.checkpoint(f"post_{spec['name']}")
    return value, {**spec, "cuda_event_seconds": cuda_seconds, "wall_seconds": wall_seconds}


def _project(model, hidden):
    with torch.inference_mode():
        return model.lm_head(hidden)


def _forward(model, spec, *, input_ids, inputs_embeds, mask, positions, keep, ledger, guard):
    ledger.checkpoint(f"pre_{spec['name']}")
    guard.verify()
    if hasattr(model, "model") and hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None
        if model.model.rope_deltas is not None:
            raise NoCacheRejected("CAP768R RoPE state did not reset")

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
                logits_to_keep=keep,
                return_dict=True,
            )

    output, cuda_seconds, wall_seconds = _timed(invoke)
    if getattr(output, "past_key_values", None) is not None:
        raise NoCacheRejected("CAP768R forward returned past_key_values")
    guard.verify()
    ledger.checkpoint(f"post_{spec['name']}")
    return output, {**spec, "cuda_event_seconds": cuda_seconds, "wall_seconds": wall_seconds}


def _capture(output, mask):
    hidden = output.hidden_states[-1]
    captured = capture_parent_features(hidden, mask, HiddenStateCaptureSpec())
    if (
        hidden.shape != (1, 768, 2048)
        or captured.hidden_states.shape != (1, 128, 2048)
        or captured.token_indices.tolist() != list(range(640, 768))
        or not torch.equal(captured.attention_mask, torch.ones_like(captured.attention_mask))
        or not torch.equal(captured.hidden_states, hidden[:, 640:768, :])
    ):
        raise DiagnosticIncomplete("CAP768R capture geometry changed")
    return captured


def _tensor_evidence(tensor):
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "sha256": tensor_bytes_sha256(tensor)}


def _comparison(spec, lhs, rhs):
    if lhs.shape != rhs.shape:
        raise DiagnosticIncomplete(f"CAP768R comparison shape changed: {spec['name']}")
    left, right = lhs.detach().cpu().contiguous(), rhs.detach().cpu().contiguous()
    mismatch = left != right
    indices = torch.nonzero(mismatch.reshape(-1), as_tuple=False)
    mismatch_count = int(indices.shape[0])
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if finite:
        delta = left.to(torch.float64) - right.to(torch.float64)
        max_abs = float(delta.abs().max())
        rms_diff = float(torch.sqrt(torch.mean(delta.square())))
        rhs_rms = float(torch.sqrt(torch.mean(right.to(torch.float64).square())))
        normalized_rms = rms_diff / max(rhs_rms, 1e-12)
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
        "count_nonzero": int(torch.count_nonzero(mismatch)),
        "first_flat_mismatch": None if mismatch_count == 0 else int(indices[0, 0]),
        "metrics_defined": finite,
        "max_abs": max_abs,
        "rms_diff": rms_diff,
        "rhs_rms": rhs_rms,
        "normalized_rms": normalized_rms,
    }


def _find_selected(artifacts):
    selected = []
    for item in SELECTION:
        split = item["evidence_id"].split("-", 1)[0]
        record = next(
            (row for row in artifacts[split]["bank"]["records"] if row["evidence_id"] == item["evidence_id"]), None
        )
        query = (
            None
            if record is None
            else next((row for row in record["queries"] if row["query_id"] == item["query_id"]), None)
        )
        if record is None or query is None or record["family"] != item["family"]:
            raise DiagnosticIncomplete("CAP768R selection changed")
        selected.append((record, query))
    return selected


def _case(model, tokenizer, record, query, selection, case, operations, ledger, guard):
    modality = case["modality"]
    messages = (
        base.parent_messages(record["parent_evidence"])
        if modality == "PARENT"
        else base.self_messages(query["child_query"])
    )
    expected = selection["parent_unpadded_tokens"] if modality == "PARENT" else selection["mself_unpadded_tokens"]
    try:
        ids = base.render_ids(tokenizer, messages, generation_prompt=False, tools=base.PARENT_TOOLS)
    except base.ExperimentIncomplete as error:
        raise DiagnosticIncomplete(f"CAP768R render failed: {error}") from error
    if ids.shape != (1, expected) or ids.dtype != torch.int64 or not ids.is_contiguous():
        raise DiagnosticIncomplete("CAP768R rendered token geometry changed")
    pad = 768 - expected
    padded = torch.nn.functional.pad(ids, (pad, 0), value=248046)
    mask = torch.nn.functional.pad(torch.ones_like(ids), (pad, 0), value=0)
    positions = torch.arange(768, device="cuda:0").unsqueeze(0)
    geometry = {
        "left_padding_exact": pad >= 0
        and bool(torch.all(padded[:, :pad] == 248046))
        and torch.equal(padded[:, pad:], ids),
        "attention_mask_exact": bool(torch.count_nonzero(mask[:, :pad]) == 0 and torch.all(mask[:, pad:] == 1)),
        "position_ids_exact": torch.equal(positions, torch.arange(768, device="cuda:0").unsqueeze(0)),
        "no_truncation": padded.shape == (1, 768) and expected + pad == 768,
    }
    embed_spec, *forward_specs, proj1_spec, proj0_spec = operations
    exact, embed_timing = _operation(embed_spec, lambda: model.get_input_embeddings()(padded), ledger)
    exact_hash = tensor_bytes_sha256(exact)
    object_id = id(exact)
    id1, id1_timing = _forward(
        model,
        forward_specs[0],
        input_ids=padded,
        inputs_embeds=None,
        mask=mask,
        positions=positions,
        keep=1,
        ledger=ledger,
        guard=guard,
    )
    e1, e1_timing = _forward(
        model,
        forward_specs[1],
        input_ids=None,
        inputs_embeds=exact,
        mask=mask,
        positions=positions,
        keep=1,
        ledger=ledger,
        guard=guard,
    )
    e2, e2_timing = _forward(
        model,
        forward_specs[2],
        input_ids=None,
        inputs_embeds=exact,
        mask=mask,
        positions=positions,
        keep=1,
        ledger=ledger,
        guard=guard,
    )
    id0, id0_timing = _forward(
        model,
        forward_specs[3],
        input_ids=padded,
        inputs_embeds=None,
        mask=mask,
        positions=positions,
        keep=0,
        ledger=ledger,
        guard=guard,
    )
    outputs = (id1, e1, e2, id0)
    captures = tuple(_capture(output, mask) for output in outputs)
    id1_last_hidden = id1.hidden_states[-1][:, -1:, :]
    id0_last_hidden = id0.hidden_states[-1][:, -1:, :]
    proj1, proj1_timing = _operation(proj1_spec, lambda: _project(model, id1_last_hidden), ledger)
    proj0, proj0_timing = _operation(proj0_spec, lambda: _project(model, id0_last_hidden), ledger)
    flags = {
        **geometry,
        "id_embed_keep1_logits_bitwise": torch.equal(id1.logits, e1.logits),
        "id_embed_keep1_full_hidden_bitwise": torch.equal(id1.hidden_states[-1], e1.hidden_states[-1]),
        "id_embed_keep1_capture_bitwise": torch.equal(captures[0].hidden_states, captures[1].hidden_states),
        "repeat_same_embedding_object": id(exact) == object_id,
        "repeat_embedding_unchanged": tensor_bytes_sha256(exact) == exact_hash,
        "repeat_logits_bitwise": torch.equal(e1.logits, e2.logits),
        "repeat_full_hidden_bitwise": torch.equal(e1.hidden_states[-1], e2.hidden_states[-1]),
        "repeat_capture_bitwise": torch.equal(captures[1].hidden_states, captures[2].hidden_states),
        "keep0_keep1_full_hidden_bitwise": torch.equal(id1.hidden_states[-1], id0.hidden_states[-1]),
        "keep0_keep1_capture_bitwise": torch.equal(captures[0].hidden_states, captures[3].hidden_states),
        "keep0_last_logits_keep1_bitwise": torch.equal(id1.logits, id0.logits[:, -1:]),
        "all_outputs_finite": all(bool(torch.isfinite(output.hidden_states[-1]).all()) for output in outputs),
        "all_output_logits_finite": all(bool(torch.isfinite(output.logits).all()) for output in outputs)
        and bool(torch.isfinite(proj1).all())
        and bool(torch.isfinite(proj0).all()),
        "all_output_full_hidden_finite": all(
            bool(torch.isfinite(output.hidden_states[-1]).all()) for output in outputs
        ),
        "all_capture_finite": all(bool(torch.isfinite(item.hidden_states).all()) for item in captures),
        "exact_embeddings_finite": bool(torch.isfinite(exact).all()),
        "exact_embeddings_requires_grad_false": not exact.requires_grad,
        "proj_id1_matches_id1_logits_bitwise": torch.equal(proj1, id1.logits),
        "proj_id0_matches_id0_last_logits_bitwise": torch.equal(proj0, id0.logits[:, -1:]),
        "proj_id1_proj_id0_bitwise": torch.equal(proj1, proj0),
        "id1_logits_proj_id0_bitwise": torch.equal(id1.logits, proj0),
    }
    if list(flags) != FLAG_NAMES or any(not isinstance(value, bool) for value in flags.values()):
        raise DiagnosticIncomplete("CAP768R flag schema changed")
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
        "PROJ_ID1_LAST.logits": proj1,
        "PROJ_ID0_LAST.logits": proj0,
    }
    comparison_specs = [
        row for row in COMPARISONS if row["probe_index"] == case["probe_index"] and row["modality"] == modality
    ]
    comparisons = [_comparison(spec, tensors[spec["lhs"]], tensors[spec["rhs"]]) for spec in comparison_specs]
    timings = [embed_timing, id1_timing, e1_timing, e2_timing, id0_timing, proj1_timing, proj0_timing]
    input_evidence = {
        "rendered_ids_shape": list(ids.shape),
        "rendered_ids_dtype": str(ids.dtype),
        "rendered_ids_contiguous": ids.is_contiguous(),
        "rendered_ids_sha256": tensor_bytes_sha256(ids),
        "padded_ids_sha256": tensor_bytes_sha256(padded),
        "attention_mask_sha256": tensor_bytes_sha256(mask),
        "position_ids_sha256": tensor_bytes_sha256(positions),
        "capture_mask_sha256": tensor_bytes_sha256(captures[0].attention_mask),
        "capture_indices": list(range(640, 768)),
        "capture_shape": [1, 128, 2048],
    }
    evidence = {
        **case,
        "evidence_id": record["evidence_id"],
        "query_id": query["query_id"],
        "unpadded_tokens": expected,
        "padded_tokens": 768,
        "left_pad_tokens": pad,
        "input_evidence": input_evidence,
        "tensor_evidence": {name: _tensor_evidence(tensor) for name, tensor in tensors.items()},
        "flags": flags,
        "gating_flags_all_true": all(flags[name] for name in GATING_FLAG_NAMES),
        "qualifies": all(flags[name] for name in GATING_FLAG_NAMES),
        "comparisons": comparisons,
        "operation_timings": timings,
        "released": True,
    }
    del exact, id1, e1, e2, id0, outputs, captures, proj1, proj0, tensors, id1_last_hidden, id0_last_hidden
    gc.collect()
    torch.cuda.empty_cache()
    ledger.checkpoint(f"post_CAP768R_P{case['probe_index']:02d}_{modality}_RELEASE")
    return evidence


def _physical_gpu_audit():
    names = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=True, text=True, capture_output=True
    ).stdout.splitlines()
    used = [
        int(value)
        for value in subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
    ]
    uuids = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], check=True, text=True, capture_output=True
    ).stdout.splitlines()
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    evidence = {"names": names, "uuids": uuids, "memory_used_mib": used, "compute_apps": apps}
    if (
        names != [RESOURCE_BOUNDS["gpu_model"]] * 2
        or len(used) != 2
        or len(uuids) != 2
        or used[1] > 512
        or any(line.startswith(uuids[1]) for line in apps)
    ):
        raise ValueError("CAP768R physical GPU inventory changed")
    return evidence


def run(args, plan, stage):
    if not args.owner_approved:
        raise ValueError("CAP768R requires root approval")
    static_guard = inspect_no_training_runner(Path(__file__)).as_dict()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    parent = subprocess.run(
        ["git", "rev-parse", f"{head}^"], cwd=args.repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=args.repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if head != args.execution_commit or parent != plan["mechanism_code_commit"] or dirty:
        raise ValueError("CAP768R execution tree changed")
    try:
        artifacts = {
            name: validate_bank_artifact(path, name)
            for name, path in (
                ("train", args.train_bank),
                ("validation", args.validation_bank),
                ("held_out", args.held_out_bank),
            )
        }
        selected = _find_selected(artifacts)
    except Exception as error:
        raise DiagnosticIncomplete(f"CAP768R bank/selection failed: {error}") from error
    versions = {
        "python": platform.python_version(),
        "transformers": importlib.metadata.version("transformers"),
        "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
        "torch_distribution": importlib.metadata.version("torch"),
        "torch_runtime": str(torch.__version__),
    }
    if versions != {key: plan["runtime"][key] for key in versions}:
        raise ValueError("CAP768R runtime changed")
    runtime_sources = base.source_hashes()
    if {key: value["sha256"] for key, value in runtime_sources.items()} != plan["runtime"][
        "transformers_source_sha256"
    ]:
        raise ValueError("CAP768R runtime source changed")
    weights = {"coordinator_e33": base.model_weight(args.coordinator), "worker_h176": base.model_weight(args.worker)}
    before = {name: file_sha256(path) for name, path in weights.items()}
    metadata_before = {
        name: base.metadata_hashes(path)
        for name, path in (("coordinator_e33", args.coordinator), ("worker_h176", args.worker))
    }
    if before != plan["protected_checkpoints"] or any(
        value != plan["runtime"]["checkpoint_metadata_sha256"] for value in metadata_before.values()
    ):
        raise ValueError("CAP768R protected preflight changed")
    stage.update({"weights": weights, "protected_before": before, "metadata_before": metadata_before})
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != RESOURCE_BOUNDS["gpu_model"]
    ):
        raise ValueError("CAP768R visible GPU changed")
    properties = torch.cuda.get_device_properties(0)
    free_disk = shutil.disk_usage(OUTPUT_ROOT).free
    host_ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    if (
        properties.total_memory < RESOURCE_BOUNDS["minimum_gpu_memory_gib"] * 2**30
        or free_disk < RESOURCE_BOUNDS["minimum_free_disk_gib"] * 2**30
        or host_ram < RESOURCE_BOUNDS["minimum_host_ram_gib"] * 2**30
    ):
        raise ValueError("CAP768R resources changed")
    physical_before = _physical_gpu_audit()
    cap_bytes = RESOURCE_BOUNDS["allocator_cap_gib"] * 2**30
    torch.cuda.set_per_process_memory_fraction(cap_bytes / properties.total_memory, 0)
    torch.cuda.reset_peak_memory_stats(0)
    tokenizer_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.coordinator, local_files_only=True)
    try:
        full_preflight = base.validate_rendering_preflight(tokenizer, artifacts, 248046, feature_token_budget=768)
    except Exception as error:
        raise DiagnosticIncomplete(f"CAP768R rendering preflight failed: {error}") from error
    rendering_preflight = {key: value for key, value in full_preflight.items() if key != "label_alignment"}
    tokenizer_seconds = time.perf_counter() - tokenizer_started
    model_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.coordinator, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda:0")
    model.eval()
    model.config.use_cache = False
    model.generation_config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_seconds = time.perf_counter() - model_started
    stage["model"] = model
    state_before = module_state_tree_sha256(model)
    stage["state_before"] = state_before
    ledger = MemoryLedger()
    stage["ledger"] = ledger
    ledger.checkpoint("model_loaded_frozen")
    cases = []
    guard = CacheGuard()
    stage["guard"] = guard
    try:
        with guard:
            offset = 0
            for case in CASE_SCHEDULE:
                index = int(case["probe_index"]) - 1
                record, query = selected[index]
                case_operations = OPERATIONS[offset : offset + 7]
                offset += 7
                evidence = _case(
                    model, tokenizer, record, query, SELECTION[index], case, case_operations, ledger, guard
                )
                cases.append(evidence)
                stage["cases"] = cases
            guard.verify()
    except (NoCacheRejected, ResourceFitRejected, TimeoutError, torch.cuda.OutOfMemoryError):
        raise
    except DiagnosticIncomplete:
        raise
    except Exception as error:
        raise DiagnosticIncomplete(f"CAP768R operation failed: {error}") from error
    compute_seconds = time.perf_counter() - stage["compute_started"]
    if compute_seconds > RESOURCE_BOUNDS["compute_seconds"]:
        raise ResourceFitRejected("CAP768R compute timeout")
    signal.signal(signal.SIGALRM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError("CAP768R audit timeout")))
    signal.alarm(RESOURCE_BOUNDS["audit_seconds"])
    audit_started = time.perf_counter()
    ledger.checkpoint("cache_guard_audit_complete")
    cache_evidence = guard.evidence()
    after = {name: file_sha256(path) for name, path in weights.items()}
    metadata_after = {
        name: base.metadata_hashes(path)
        for name, path in (("coordinator_e33", args.coordinator), ("worker_h176", args.worker))
    }
    state_after = module_state_tree_sha256(model)
    if (
        after != before
        or metadata_after != metadata_before
        or state_after != state_before
        or any(parameter.grad is not None for parameter in model.parameters())
    ):
        raise ValueError("CAP768R protected state changed")
    ledger.checkpoint("protected_postflight_complete")
    if [row["label"] for row in ledger.rows] != memory_labels():
        raise DiagnosticIncomplete("CAP768R memory ledger incomplete")
    physical_after = _physical_gpu_audit()
    audit_seconds = time.perf_counter() - audit_started
    operation_timings = [timing for case in cases for timing in case["operation_timings"]]
    probes = []
    for index, selection in enumerate(SELECTION, 1):
        matched = [case for case in cases if case["probe_index"] == index]
        probes.append(
            {
                "probe_index": index,
                "family": selection["family"],
                "modalities_complete": len(matched) == 2,
                "qualifies": len(matched) == 2 and all(case["qualifies"] for case in matched),
            }
        )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": classification(all(item["qualifies"] for item in probes)),
        "plan_sha256": plan["plan_sha256"],
        "mechanism_code_commit": plan["mechanism_code_commit"],
        "execution_commit": args.execution_commit,
        "asset_sha256": plan["asset_sha256"],
        "run_id": args.output_dir.name,
        "selection": SELECTION,
        "selection_sha256": SELECTION_SHA256,
        "case_schedule": CASE_SCHEDULE,
        "case_schedule_sha256": CASE_SCHEDULE_SHA256,
        "operation_schedule": OPERATIONS,
        "operation_schedule_sha256": OPERATION_SCHEDULE_SHA256,
        "operation_counts": {
            "embedding_lookup": 8,
            "e33_forward": 32,
            "lm_head_projection": 16,
            "capture": 32,
            "comparison": 104,
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
        "gating_flag_names": GATING_FLAG_NAMES,
        "gating_flag_names_sha256": GATING_FLAG_NAMES_SHA256,
        "descriptive_flag_names": DESCRIPTIVE_FLAG_NAMES,
        "descriptive_flag_names_sha256": DESCRIPTIVE_FLAG_NAMES_SHA256,
        "comparison_schedule": COMPARISONS,
        "comparison_schedule_sha256": COMPARISON_SCHEDULE_SHA256,
        "cases": cases,
        "probes": probes,
        "flag0_incomplete_evidence": FLAG0_INCOMPLETE_EVIDENCE,
        "versions": versions,
        "runtime_sources": runtime_sources,
        "static_guard": static_guard,
        "rendering_preflight": rendering_preflight,
        "protected_hashes_before": before,
        "protected_hashes_after": after,
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
            "calls": 32,
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
            "operation_cuda_event_seconds_sum": math.fsum(item["cuda_event_seconds"] for item in operation_timings),
            "operation_wall_seconds_sum": math.fsum(item["wall_seconds"] for item in operation_timings),
            "tokenizer_load_seconds": tokenizer_seconds,
            "model_load_seconds": model_seconds,
            "compute_seconds": compute_seconds,
            "audit_seconds": audit_seconds,
            "total_seconds": time.perf_counter() - stage["compute_started"],
        },
        "decision_boundary": DECISION_BOUNDARY,
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
    protected_after, metadata_after = {}, {}
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
    state_after, gradients_absent = None, None
    if model is not None:
        try:
            state_after = module_state_tree_sha256(model)
            gradients_absent = all(parameter.grad is None for parameter in model.parameters())
        except BaseException as audit_error:
            audit_errors.append(f"model:{type(audit_error).__name__}:{audit_error}")
    guard = stage.get("guard")
    try:
        cache_partial = guard.evidence() if isinstance(guard, CacheGuard) else None
    except BaseException as audit_error:
        cache_partial = None
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
        "flag0_incomplete_evidence": FLAG0_INCOMPLETE_EVIDENCE,
        "decision_boundary": DECISION_BOUNDARY,
        "cases_partial": stage.get("cases", []),
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
    for name in (
        "repo",
        "plan",
        "coordinator",
        "worker",
        "train_bank",
        "validation_bank",
        "held_out_bank",
        "output_dir",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    writer = ArtifactWriter(args.output_dir)
    plan = None
    stage = {"compute_started": time.perf_counter()}
    signal.signal(signal.SIGALRM, lambda _s, _f: (_ for _ in ()).throw(ResourceFitRejected("CAP768R compute timeout")))
    signal.alarm(RESOURCE_BOUNDS["compute_seconds"])
    try:
        plan = load_plan(args.plan, args.repo)
        receipt = run(args, plan, stage)
        signal.alarm(RESOURCE_BOUNDS["terminal_seconds"])
        writer.write_terminal("receipt.json", receipt, RESOURCE_BOUNDS["maximum_receipt_bytes"])
    except torch.cuda.OutOfMemoryError as error:
        signal.alarm(RESOURCE_BOUNDS["failure_audit_seconds"])
        wrapped = ResourceFitRejected(f"CAP768R CUDA OOM: {error}")
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
