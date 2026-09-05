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
from prime_rl.latent.a1cap768 import (
    FAILURE_SCHEMA,
    INTERPRETATION,
    RECEIPT_SCHEMA,
    RESOURCE_BOUNDS,
    SCHEDULE_SHA256,
    SELECTION,
    SELECTION_SHA256,
    CaptureMechanismRejected,
    DiagnosticIncomplete,
    ResourceFitRejected,
    build_schedule,
    classify_failure,
    load_plan,
    memory_labels,
    validate_bank_artifact,
    validate_receipt,
)
from prime_rl.latent.a1nc0 import _CACHE_CLASS_CLOSURE, module_state_tree_sha256, tensor_bytes_sha256
from prime_rl.latent.policy_adapter import HiddenStateCaptureSpec, capture_parent_features

_A1_RUNNER_PATH = Path(__file__).with_name("run_a1_nc0_nomination_v1.py")
_SPEC = importlib.util.spec_from_file_location("a1_nc0_repaired_runner", _A1_RUNNER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("repaired A1-NC0 runner unavailable")
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)

OUTPUT_ROOT = Path(RESOURCE_BOUNDS["output_root"])
SHARED_ENVIRONMENT = Path("/home/ubuntu/rlm/prime-rl/.venv")


class ArtifactWriter:
    def __init__(self, output_dir: Path) -> None:
        if (
            not output_dir.is_absolute()
            or OUTPUT_ROOT.is_symlink()
            or not OUTPUT_ROOT.is_dir()
            or output_dir.parent.resolve(strict=True) != OUTPUT_ROOT.resolve(strict=True)
            or not output_dir.name.startswith("a1-nc0-cap768-")
            or output_dir.exists()
            or output_dir.is_symlink()
        ):
            raise ValueError("CAP768 output namespace changed")
        output_dir.mkdir(mode=0o700)
        self.output_dir = output_dir
        self.terminal_written = False

    def write_terminal(self, name: str, payload: dict[str, object], maximum_bytes: int) -> None:
        if self.terminal_written or name not in {"receipt.json", "failure.json"}:
            raise ValueError("CAP768 terminal is not exclusive")
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if len(encoded) > maximum_bytes:
            raise ValueError("CAP768 terminal exceeds frozen bound")
        temporary = self.output_dir / f".{name}.tmp"
        target = self.output_dir / name
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written == 0:
                    raise OSError("short CAP768 terminal write")
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
            raise DiagnosticIncomplete("CAP768 memory-label order changed")
        row = {
            "label": label,
            "allocated_bytes": torch.cuda.memory_allocated(0),
            "reserved_bytes": torch.cuda.memory_reserved(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        }
        self.rows.append(row)
        if any(row[key] > self.cap for key in row if key.endswith("_bytes")):
            raise ResourceFitRejected("CAP768 exceeded 40GiB allocator cap")

    def validate_complete(self) -> None:
        if [row["label"] for row in self.rows] != self.labels:
            raise DiagnosticIncomplete("CAP768 memory ledger incomplete")


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
        raise CaptureMechanismRejected(f"CAP768 cache allocation attempted: {cls.__module__}.{cls.__qualname__}")

    def __enter__(self):
        try:
            for cls in sorted(self.initial, key=lambda item: (item.__module__, item.__qualname__)):
                self.stack.enter_context(mock.patch.object(cls, "__new__", self._reject))
                self.patched.add(cls)
            try:
                transformers.cache_utils.DynamicCache()
            except CaptureMechanismRejected:
                self.negative = True
            if not self.negative:
                raise DiagnosticIncomplete("CAP768 cache negative control did not trip")
            self.verify()
        except BaseException:
            self.stack.close()
            self.restored = True
            raise
        return self

    def verify(self) -> None:
        if recursive_subclass_closure(self.base) - self.patched:
            raise DiagnosticIncomplete("CAP768 new unpatched cache subclass loaded")
        self.checks += 1

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.verify()
        finally:
            self.stack.close()
            self.restored = True
        return False

    def evidence(self) -> dict[str, object]:
        observed = [base.class_identity(cls) for cls in sorted(self.initial, key=lambda x: (x.__module__, x.__qualname__))]
        if observed != _CACHE_CLASS_CLOSURE:
            raise ValueError("CAP768 cache class provenance changed")
        return {
            "classes": observed,
            "negative_control_dynamic_cache_tripped": self.negative,
            "closure_check_count": self.checks,
            "restored_in_finally": self.restored,
        }


def _cuda_timed(call):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    value = call()
    end.record()
    end.synchronize()
    return value, start.elapsed_time(end) / 1000.0, time.perf_counter() - wall


def _forward(model, *, input_ids, inputs_embeds, mask, positions, logits_to_keep: int, arm: str, ledger, guard):
    ledger.checkpoint(f"pre_{arm}")
    guard.verify()
    if hasattr(model, "model") and hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None
        if model.model.rope_deltas is not None:
            raise CaptureMechanismRejected("CAP768 RoPE recurrence state did not reset")

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

    output, cuda_seconds, wall_seconds = _cuda_timed(invoke)
    if getattr(output, "past_key_values", None) is not None:
        raise CaptureMechanismRejected("CAP768 forward returned past_key_values")
    guard.verify()
    ledger.checkpoint(f"post_{arm}")
    return output, cuda_seconds, wall_seconds


def _capture(output, mask):
    hidden = output.hidden_states[-1]
    captured = capture_parent_features(hidden, mask, HiddenStateCaptureSpec())
    expected_indices = list(range(640, 768))
    if (
        hidden.shape != (1, 768, 2048)
        or not torch.isfinite(hidden).all()
        or captured.hidden_states.shape != (1, 128, 2048)
        or captured.token_indices.tolist() != expected_indices
        or not torch.equal(captured.attention_mask, torch.ones((1, 128), dtype=mask.dtype, device=mask.device))
        or not torch.equal(captured.hidden_states, hidden[:, 640:768, :])
    ):
        raise CaptureMechanismRejected("CAP768 hidden capture geometry/content changed")
    return captured


def _case(model, tokenizer, messages, *, expected_unpadded: int, prefix: str, ledger, guard):
    tools = base.PARENT_TOOLS
    ids = base.render_ids(tokenizer, messages, generation_prompt=False, tools=tools)
    if ids.shape != (1, expected_unpadded):
        raise DiagnosticIncomplete("CAP768 selected transcript token length changed")
    pad = 768 - expected_unpadded
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
        raise CaptureMechanismRejected("CAP768 fixed left-pad geometry changed")
    embedding_call = lambda: model.get_input_embeddings()(padded)
    exact, embed_cuda, embed_wall = _cuda_timed(embedding_call)
    if not torch.isfinite(exact).all() or exact.requires_grad:
        raise CaptureMechanismRejected("CAP768 exact embedding lookup produced nonfinite values")
    exact_before = tensor_bytes_sha256(exact)
    arms = [f"{prefix}_{name}" for name in ("L_ID_KEEP1", "L_E_KEEP1", "L_E_REPEAT_KEEP1", "L_ID_KEEP0_CONTROL")]
    id1, id1_cuda, id1_wall = _forward(
        model, input_ids=padded, inputs_embeds=None, mask=mask, positions=positions, logits_to_keep=1,
        arm=arms[0], ledger=ledger, guard=guard,
    )
    e1, e1_cuda, e1_wall = _forward(
        model, input_ids=None, inputs_embeds=exact, mask=mask, positions=positions, logits_to_keep=1,
        arm=arms[1], ledger=ledger, guard=guard,
    )
    repeated_object_id = id(exact)
    e2, e2_cuda, e2_wall = _forward(
        model, input_ids=None, inputs_embeds=exact, mask=mask, positions=positions, logits_to_keep=1,
        arm=arms[2], ledger=ledger, guard=guard,
    )
    id0, id0_cuda, id0_wall = _forward(
        model, input_ids=padded, inputs_embeds=None, mask=mask, positions=positions, logits_to_keep=0,
        arm=arms[3], ledger=ledger, guard=guard,
    )
    outputs = (id1, e1, e2, id0)
    captures = tuple(_capture(output, mask) for output in outputs)
    if (
        id1.logits.shape != (1, 1, model.config.text_config.vocab_size)
        or e1.logits.shape != id1.logits.shape
        or e2.logits.shape != id1.logits.shape
        or id0.logits.shape != (1, 768, model.config.text_config.vocab_size)
        or not all(torch.isfinite(output.logits).all() for output in outputs)
    ):
        raise CaptureMechanismRejected("CAP768 logits shape/finiteness changed")
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
    if not all(flags.values()):
        raise CaptureMechanismRejected("CAP768 parity/repeat/capture predicate rejected")
    call_times = []
    for arm, output, cuda_seconds, wall_seconds in zip(
        arms, outputs, (id1_cuda, e1_cuda, e2_cuda, id0_cuda), (id1_wall, e1_wall, e2_wall, id0_wall), strict=True
    ):
        call_times.append(
            {
                "arm": arm,
                "unpadded_tokens": expected_unpadded,
                "padded_tokens": 768,
                "logits_to_keep": 0 if arm.endswith("KEEP0_CONTROL") else 1,
                "cuda_event_seconds": cuda_seconds,
                "wall_seconds": wall_seconds,
                "logits_sha256": tensor_bytes_sha256(output.logits[:, -1:]),
            }
        )
    evidence = {
        "unpadded_tokens": expected_unpadded,
        "padded_tokens": 768,
        "padding_tokens": pad,
        "capture_indices": list(range(640, 768)),
        "capture_shape": [1, 128, 2048],
        "input_ids_sha256": tensor_bytes_sha256(padded),
        "attention_mask_sha256": tensor_bytes_sha256(mask),
        "captured_mask_sha256": tensor_bytes_sha256(captures[0].attention_mask),
        "position_ids_sha256": tensor_bytes_sha256(positions),
        "exact_embeddings_sha256": exact_before,
        "full_hidden_sha256": tensor_bytes_sha256(id1.hidden_states[-1]),
        "capture_sha256": tensor_bytes_sha256(captures[0].hidden_states),
        "keep1_logits_sha256": tensor_bytes_sha256(id1.logits),
        "operation_hashes": {
            arm: {
                "last_logits_sha256": tensor_bytes_sha256(output.logits[:, -1:]),
                "full_hidden_sha256": tensor_bytes_sha256(output.hidden_states[-1]),
                "capture_sha256": tensor_bytes_sha256(capture.hidden_states),
            }
            for arm, output, capture in zip(arms, outputs, captures, strict=True)
        },
        "exact_embeddings_finite": True,
        "exact_embeddings_requires_grad_false": True,
        **flags,
        "embedding_lookup_cuda_event_seconds": embed_cuda,
        "embedding_lookup_wall_seconds": embed_wall,
        "four_call_cuda_event_seconds": math.fsum(item["cuda_event_seconds"] for item in call_times),
        "four_call_wall_seconds": math.fsum(item["wall_seconds"] for item in call_times),
    }
    del exact, outputs, captures, id1, e1, e2, id0
    return evidence, call_times


def _find_selected(artifacts):
    found = []
    for selected in SELECTION:
        split = selected["evidence_id"].split("-", 1)[0]
        record = next((item for item in artifacts[split]["bank"]["records"] if item["evidence_id"] == selected["evidence_id"]), None)
        query = None if record is None else next((item for item in record["queries"] if item["query_id"] == selected["query_id"]), None)
        if record is None or query is None or record["family"] != selected["family"]:
            raise DiagnosticIncomplete("CAP768 selected bank identity changed")
        found.append((record, query))
    return found


def _static_guard(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in {"generate", "backward", "step"}:
                forbidden.append(name)
    if forbidden or "WorkspaceBridge" in path.read_text() or "AdamW" in path.read_text():
        raise DiagnosticIncomplete("CAP768 forbidden training/generation source appeared")
    return {"runner_sha256": file_sha256(path), "forbidden_calls": forbidden}


def _physical_gpu_audit() -> dict[str, object]:
    names = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=True, text=True, capture_output=True
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
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], check=True, text=True, capture_output=True
    ).stdout.splitlines()
    app_lines = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    apps = [
        {"gpu_uuid": parts[0].strip(), "pid": int(parts[1].strip())}
        for line in app_lines
        if line.strip()
        for parts in [line.split(",", 1)]
    ]
    evidence = {"names": names, "uuids": uuids, "memory_used_mib": memory, "compute_apps": apps}
    if (
        names != ["NVIDIA RTX A6000", "NVIDIA RTX A6000"]
        or len(memory) != 2
        or len(uuids) != 2
        or memory[1] > 512
        or any(app["gpu_uuid"] == uuids[1] for app in apps)
    ):
        raise ValueError("CAP768 physical GPU inventory/GPU1 idle evidence changed")
    return evidence


def run(args, plan, writer, stage):
    if not args.owner_approved:
        raise ValueError("CAP768 requires root approval")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.repo, check=True, text=True, capture_output=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=args.repo, check=True, text=True, capture_output=True).stdout
    parent = subprocess.run(["git", "rev-parse", f"{head}^"], cwd=args.repo, check=True, text=True, capture_output=True).stdout.strip()
    if head != args.execution_commit or status or parent != plan["mechanism_code_commit"]:
        raise ValueError("CAP768 execution tree changed")
    try:
        artifacts = {
            "train": validate_bank_artifact(args.train_bank, "train"),
            "validation": validate_bank_artifact(args.validation_bank, "validation"),
            "held_out": validate_bank_artifact(args.held_out_bank, "held_out"),
        }
        selected = _find_selected(artifacts)
    except Exception as error:
        raise DiagnosticIncomplete(f"CAP768 bank/selection validation failed: {error}") from error
    versions = {
        "python": platform.python_version(), "transformers": importlib.metadata.version("transformers"),
        "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
        "torch_distribution": importlib.metadata.version("torch"), "torch_runtime": str(torch.__version__),
    }
    if versions != {key: plan["runtime"][key] for key in versions}:
        raise ValueError("CAP768 runtime changed")
    runtime_sources = base.source_hashes()
    if {name: item["sha256"] for name, item in runtime_sources.items()} != plan["runtime"]["transformers_source_sha256"]:
        raise ValueError("CAP768 runtime source identity changed")
    weights = {"coordinator_e33": base.model_weight(args.coordinator), "worker_h176": base.model_weight(args.worker)}
    before = {name: file_sha256(path) for name, path in weights.items()}
    metadata_before = {name: base.metadata_hashes(path) for name, path in (("coordinator_e33", args.coordinator), ("worker_h176", args.worker))}
    if before != plan["protected_checkpoints"] or any(value != plan["runtime"]["checkpoint_metadata_sha256"] for value in metadata_before.values()):
        raise ValueError("CAP768 protected preflight changed")
    stage.update({"weights": weights, "protected_before": before, "metadata_before": metadata_before})
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != RESOURCE_BOUNDS["gpu_model"] or torch.cuda.device_count() != 1:
        raise ValueError("CAP768 visible GPU changed")
    properties = torch.cuda.get_device_properties(0)
    free_disk = shutil.disk_usage(OUTPUT_ROOT).free
    host_ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    if properties.total_memory < RESOURCE_BOUNDS["minimum_gpu_memory_gib"] * 2**30 or free_disk < RESOURCE_BOUNDS["minimum_free_disk_gib"] * 2**30 or host_ram < RESOURCE_BOUNDS["minimum_host_ram_gib"] * 2**30:
        raise ValueError("CAP768 host resources changed")
    cap_bytes = RESOURCE_BOUNDS["allocator_cap_gib"] * 2**30
    physical_before = _physical_gpu_audit()
    torch.cuda.set_per_process_memory_fraction(cap_bytes / properties.total_memory, 0)
    torch.cuda.reset_peak_memory_stats(0)
    # One frozen compute allowance begins before plan/provenance validation in
    # main. Protected hashing, tokenizer/model load, and probes share it.
    started = float(stage["compute_started"])
    tokenizer_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.coordinator, local_files_only=True)
    try:
        full_preflight = base.validate_rendering_preflight(
            tokenizer, artifacts, 248046, feature_token_budget=768
        )
    except Exception as error:
        raise DiagnosticIncomplete(f"CAP768 all-bank render preflight failed: {error}") from error
    preflight = {key: value for key, value in full_preflight.items() if key != "label_alignment"}
    tokenizer_seconds = time.perf_counter() - tokenizer_started
    if preflight["maximum_unpadded_feature_tokens"] != 644 or "label_alignment" in preflight:
        raise DiagnosticIncomplete("CAP768 all-bank maximum changed")
    model_load_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.coordinator, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
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
    calls = []
    probes = []
    guard = CacheGuard()
    stage["guard"] = guard
    with guard:
        for probe_index, ((record, query), selection) in enumerate(zip(selected, SELECTION, strict=True), 1):
            modalities = {}
            for modality, messages, expected in (
                ("PARENT", base.parent_messages(record["parent_evidence"]), selection["parent_unpadded_tokens"]),
                ("MSELF", base.self_messages(query["child_query"]), selection["mself_unpadded_tokens"]),
            ):
                evidence, case_calls = _case(
                    model, tokenizer, messages, expected_unpadded=expected,
                    prefix=f"CAP768_P{probe_index:02d}_{modality}", ledger=ledger, guard=guard,
                )
                modalities[modality] = evidence
                for item in case_calls:
                    item.update({"probe_index": probe_index, "family": selection["family"], "modality": modality})
                calls.extend(case_calls)
            probes.append({"selection": selection, "modalities": modalities})
        guard.verify()
    for observed, expected_call in zip(calls, build_schedule(), strict=True):
        if observed["arm"] != expected_call["arm"]:
            raise DiagnosticIncomplete("CAP768 observed call schedule changed")
        observed["call_index"] = expected_call["call_index"]
    compute_seconds = time.perf_counter() - started
    if compute_seconds > RESOURCE_BOUNDS["compute_seconds"]:
        raise ResourceFitRejected("CAP768 compute timeout")
    signal.signal(signal.SIGALRM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError("CAP768 audit timeout")))
    signal.alarm(RESOURCE_BOUNDS["audit_seconds"])
    audit_started = time.perf_counter()
    ledger.checkpoint("cache_guard_audit_complete")
    cache_evidence = guard.evidence()
    state_after = module_state_tree_sha256(model)
    after = {name: file_sha256(path) for name, path in weights.items()}
    metadata_after = {name: base.metadata_hashes(path) for name, path in (("coordinator_e33", args.coordinator), ("worker_h176", args.worker))}
    if state_after != state_before or after != before or metadata_after != metadata_before or any(p.grad is not None for p in model.parameters()):
        raise ValueError("CAP768 protected model changed")
    ledger.checkpoint("protected_postflight_complete")
    ledger.validate_complete()
    physical_after = _physical_gpu_audit()
    audit_seconds = time.perf_counter() - audit_started
    if audit_seconds > RESOURCE_BOUNDS["audit_seconds"]:
        raise TimeoutError("CAP768 audit timeout")
    call_cuda = [float(item["cuda_event_seconds"]) for item in calls]
    call_wall = [float(item["wall_seconds"]) for item in calls]
    per_probe_timings = []
    for probe_index, probe in enumerate(probes, 1):
        probe_calls = [item for item in calls if item["probe_index"] == probe_index]
        per_probe_timings.append(
            {
                "probe_index": probe_index,
                "embedding_cuda_event_seconds": math.fsum(
                    probe["modalities"][name]["embedding_lookup_cuda_event_seconds"]
                    for name in ("PARENT", "MSELF")
                ),
                "embedding_wall_seconds": math.fsum(
                    probe["modalities"][name]["embedding_lookup_wall_seconds"]
                    for name in ("PARENT", "MSELF")
                ),
                "call_cuda_event_seconds": math.fsum(float(item["cuda_event_seconds"]) for item in probe_calls),
                "call_wall_seconds": math.fsum(float(item["wall_seconds"]) for item in probe_calls),
            }
        )
    receipt = {
        "schema_version": RECEIPT_SCHEMA, "status": "capture768_mechanism_validated",
        "plan_sha256": plan["plan_sha256"], "mechanism_code_commit": plan["mechanism_code_commit"],
        "execution_commit": args.execution_commit, "asset_sha256": plan["asset_sha256"],
        "selection": SELECTION, "selection_sha256": SELECTION_SHA256,
        "call_schedule": build_schedule(), "call_schedule_sha256": SCHEDULE_SHA256,
        "prior_evidence": plan["prior_evidence"], "versions": versions, "runtime_sources": runtime_sources,
        "static_guard": _static_guard(Path(__file__)), "render_preflight": preflight,
        "protected_hashes_before": before, "protected_hashes_after": after,
        "checkpoint_metadata_before": metadata_before, "checkpoint_metadata_after": metadata_after,
        "e33_state_tree_before": state_before, "e33_state_tree_after": state_after,
        "e33_parameters_frozen_no_grad": True, "worker_h176_loaded": False,
        "model_runtime": {"class": model.__class__.__name__, "hidden_size": model.config.text_config.hidden_size,
                          "vocab_size": model.config.text_config.vocab_size,
                          "dtype": str(next(model.parameters()).dtype), "device": str(next(model.parameters()).device)},
        "probes": probes, "calls": calls,
        "no_cache_contract": {"calls": 32, "use_cache_false": True, "pkv_input_none": True,
                              "pkv_output_none": True, "rope_reset_every_call": True, "embedding_lookups": 8,
                              "model_config_use_cache": model.config.use_cache,
                              "generation_config_use_cache": model.generation_config.use_cache},
        "cache_guard": cache_evidence, "memory_ledger": ledger.rows,
        "memory_labels_sha256": canonical_json_hash([row["label"] for row in ledger.rows]),
        "resources": {"gpu_name": torch.cuda.get_device_name(0), "total_gpu_memory_bytes": properties.total_memory,
                      "allocator_cap_bytes": cap_bytes, "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(0), "host_ram_bytes": host_ram,
                      "free_disk_bytes_before": free_disk, "visible_cuda_devices": 1,
                      "physical_gpu1_unused": physical_before["memory_used_mib"][1] <= 512
                      and physical_after["memory_used_mib"][1] <= 512
                      and not any(app["gpu_uuid"] == physical_before["uuids"][1]
                                  for app in physical_before["compute_apps"])
                      and not any(app["gpu_uuid"] == physical_after["uuids"][1]
                                  for app in physical_after["compute_apps"]),
                      "physical_gpu_audit_before": physical_before,
                      "physical_gpu_audit_after": physical_after,
                      "network_used": False},
        "timings": {"tokenizer_seconds": tokenizer_seconds, "model_load_seconds": model_load_seconds,
                    "compute_seconds": compute_seconds, "audit_seconds": audit_seconds,
                    "call_cuda_event_seconds_sum": math.fsum(call_cuda), "call_wall_seconds_sum": math.fsum(call_wall),
                    "embedding_cuda_event_seconds_sum": math.fsum(
                        probe["modalities"][modality]["embedding_lookup_cuda_event_seconds"]
                        for probe in probes for modality in ("PARENT", "MSELF")
                    ),
                    "embedding_wall_seconds_sum": math.fsum(
                        probe["modalities"][modality]["embedding_lookup_wall_seconds"]
                        for probe in probes for modality in ("PARENT", "MSELF")
                    ),
                    "per_probe": per_probe_timings,
                    "total_seconds": time.perf_counter() - started},
        "claim": "capture768_geometry_and_resource_fit_only", "training_authorized": False,
        "bridge_created": False, "optimizer_created": False, "backward_used": False,
        "checkpoint_created": False, "candidate_created": False, "generation_used": False,
        "model_update_attempted": False, "semantic_heldout_output": False,
        "reusable_hidden_persisted": False, "interpretation_boundary": INTERPRETATION, "receipt_sha256": "",
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
        "schema_version": FAILURE_SCHEMA, "status": status, "failure_category": category,
        "error_type": type(error).__name__, "error": str(error), "execution_commit": args.execution_commit,
        "mechanism_code_commit": None if plan is None else plan.get("mechanism_code_commit"),
        "plan_sha256": None if plan is None else plan.get("plan_sha256"),
        "model_loaded": model is not None, "model_update_attempted": False, "bridge_created": False,
        "optimizer_created": False, "backward_used": False, "checkpoint_created": False,
        "candidate_created": False, "worker_h176_loaded": False, "failure_sha256": "",
        "prior_evidence": None if plan is None else plan.get("prior_evidence"),
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
    }
    failure["failure_sha256"] = canonical_json_hash(failure, omitted_fields=("failure_sha256",))
    return failure


def main():
    parser = argparse.ArgumentParser()
    for name in ("repo", "plan", "coordinator", "worker", "train_bank", "validation_bank", "held_out_bank", "output_dir"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    writer = ArtifactWriter(args.output_dir)
    plan = None
    stage = {}
    signal.signal(
        signal.SIGALRM,
        lambda _s, _f: (_ for _ in ()).throw(ResourceFitRejected("CAP768 compute timeout")),
    )
    stage["compute_started"] = time.perf_counter()
    signal.alarm(RESOURCE_BOUNDS["compute_seconds"])
    try:
        plan = load_plan(args.plan, args.repo)
        receipt = run(args, plan, writer, stage)
        signal.alarm(RESOURCE_BOUNDS["terminal_seconds"])
        writer.write_terminal("receipt.json", receipt, RESOURCE_BOUNDS["maximum_receipt_bytes"])
    except torch.cuda.OutOfMemoryError as error:
        signal.alarm(RESOURCE_BOUNDS["failure_audit_seconds"])
        wrapped = ResourceFitRejected(f"CAP768 CUDA OOM: {error}")
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
