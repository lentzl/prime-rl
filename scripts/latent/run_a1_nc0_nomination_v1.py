from __future__ import annotations

import argparse
import ast
import collections
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import signal
import stat
import subprocess
import time
from pathlib import Path
from unittest import mock

import torch
import transformers.cache_utils
import transformers.generation.utils
import transformers.loss.loss_utils
import transformers.models.qwen3_5.modeling_qwen3_5
from safetensors.torch import save as save_safetensors
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a0nc import recursive_subclass_closure
from prime_rl.latent.a0nc import validate_receipt as validate_a0nc_receipt
from prime_rl.latent.a1nc0 import (
    CacheAllocationDetected,
    ExperimentIncomplete,
    MechanismRejected,
    build_arm_orders,
    build_memory_ledger_paths,
    classify_failure,
    fixed_feature_inputs,
    load_plan,
    module_state_tree_sha256,
    nomination_gate_passes,
    norm_matched_noise,
    summarize_arm_results,
    tensor_bytes_sha256,
    validate_a1nc0_r1_evidence,
    validate_finite_metrics,
    validate_receipt,
    validation_gate_passes,
)
from prime_rl.latent.bridge import WorkspaceBridge, WorkspaceBridgeConfig
from prime_rl.latent.policy_adapter import HiddenStateCaptureSpec, capture_parent_features, compose_receiver_inputs

OUTPUT_ROOT = Path("/home/ubuntu/rlm/outputs/latent-a1-nc0-r1-nomination-v1")
SHARED_ENVIRONMENT = Path("/home/ubuntu/rlm/prime-rl/.venv")
FIXED_CONTINUATION_TEXT = " Acknowledged and continuing safely."
FIXED_CONTINUATION_IDS = [49265, 48338, 3438, 321]
LENGTH_CONTROL_IDS = [40, 4021, 2528, 8976, 35139, 635, 524, 599]

PARENT_SYSTEM = (
    "You are the root coordinator. Preserve the private source packet in your internal state and delegate one "
    "coordinator child without copying the packet into the tool call."
)
PARENT_REASONING = (
    "The child will receive a separate downstream query. I should delegate without copying the private source packet."
)
PARENT_CODE = (
    'child = await rlm("Answer the downstream query supplied in your own child context. Return only the requested '
    'value.", name="a1-nc0-child")'
)
PARENT_TOOL_RESPONSE = (
    '{"model":"e33","name":"a1-nc0-child","rlm_child_id":"a1-nc0-fixture-child",'
    '"session_dir":"/synthetic/a1-nc0-child"}'
)
PARENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ipython",
            "description": "Execute Python code in the coordinator's persistent IPython kernel.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    }
]
CHILD_SYSTEM = (
    "You are a coordinator child. Answer the user's downstream query. If an authorized latent workspace is present, "
    "use it. Return only the requested value. Do not call tools."
)
TERMINAL_TOKEN_IDS = [248046, 198]
CHILD_OPENING_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def tensor_sha256(tensor: torch.Tensor) -> str:
    return tensor_bytes_sha256(tensor)


def bridge_parameter_sha256(bridge: WorkspaceBridge) -> str:
    return module_state_tree_sha256(bridge)


class ArtifactWriter:
    def __init__(self, output_dir: Path) -> None:
        if not output_dir.is_absolute() or OUTPUT_ROOT.is_symlink() or not OUTPUT_ROOT.is_dir():
            raise ValueError("A1-NC0 output root must be absolute, existing, and non-symlinked")
        if output_dir.parent.resolve(strict=True) != OUTPUT_ROOT.resolve(strict=True):
            raise ValueError("A1-NC0 output directory must be a direct child of its frozen root")
        if not output_dir.name.startswith("a1-nc0-r1-") or "/" in output_dir.name:
            raise ValueError("A1-NC0 output directory is outside the frozen namespace")
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        self.root_fd = os.open(OUTPUT_ROOT, os.O_RDONLY | os.O_DIRECTORY | nofollow)
        os.mkdir(output_dir.name, mode=0o700, dir_fd=self.root_fd)
        self.dir_fd = os.open(output_dir.name, os.O_RDONLY | os.O_DIRECTORY | nofollow, dir_fd=self.root_fd)
        self.candidate_name: str | None = None
        self.terminal_name: str | None = None

    def _write_bytes(self, name: str, encoded: bytes, maximum_bytes: int) -> dict[str, object]:
        if name.startswith(".") or "/" in name or name in os.listdir(self.dir_fd):
            raise ValueError("A1-NC0 artifact name is invalid or already exists")
        if len(encoded) > maximum_bytes:
            raise ValueError("A1-NC0 artifact exceeds its frozen bound")
        temporary = f".{name}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=self.dir_fd,
        )
        try:
            try:
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if written == 0:
                        raise OSError("short A1-NC0 artifact write")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException:
            try:
                os.unlink(temporary, dir_fd=self.dir_fd)
            except OSError:
                pass
            raise
        os.rename(temporary, name, src_dir_fd=self.dir_fd, dst_dir_fd=self.dir_fd)
        os.fsync(self.dir_fd)
        artifact = os.stat(name, dir_fd=self.dir_fd, follow_symlinks=False)
        if not stat.S_ISREG(artifact.st_mode):
            raise RuntimeError("A1-NC0 artifact postflight failed")
        return {
            "name": name,
            "bytes": artifact.st_size,
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }

    def write_candidate(self, state: dict[str, torch.Tensor], maximum_bytes: int) -> dict[str, object]:
        if self.candidate_name is not None or self.terminal_name is not None or os.listdir(self.dir_fd):
            raise ValueError("A1-NC0 candidate must be the first and only nonterminal artifact")
        payload = {name: tensor.detach().cpu().contiguous() for name, tensor in sorted(state.items())}
        encoded = save_safetensors(payload, metadata={"schema": "prime-rl/latent-a1-nc0-candidate/v1"})
        evidence = self._write_bytes("bridge-candidate.safetensors", encoded, maximum_bytes)
        self.candidate_name = evidence["name"]
        return evidence

    def write_terminal(self, name: str, value: dict[str, object], maximum_bytes: int) -> dict[str, object]:
        if name not in {"receipt.json", "failure.json"} or self.terminal_name is not None:
            raise ValueError("A1-NC0 terminal artifact contract changed")
        existing = set(os.listdir(self.dir_fd))
        permitted = set() if self.candidate_name is None else {self.candidate_name}
        if existing != permitted:
            raise ValueError("A1-NC0 output contains unexpected files before terminal write")
        encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        evidence = self._write_bytes(name, encoded, maximum_bytes)
        self.terminal_name = name
        expected = permitted | {name}
        if set(os.listdir(self.dir_fd)) != expected:
            raise RuntimeError("A1-NC0 terminal output inventory changed")
        os.fsync(self.root_fd)
        return evidence

    def inventory(self) -> list[dict[str, object]]:
        result = []
        for name in sorted(os.listdir(self.dir_fd)):
            info = os.stat(name, dir_fd=self.dir_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError("A1-NC0 output contains a nonregular artifact")
            descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=self.dir_fd)
            try:
                digest = hashlib.sha256()
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            finally:
                os.close(descriptor)
            result.append({"name": name, "bytes": info.st_size, "sha256": digest.hexdigest()})
        return result

    def close(self) -> None:
        os.close(self.dir_fd)
        os.close(self.root_fd)


def model_weight(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or (path / "STABLE").is_symlink() or not (path / "STABLE").is_file():
        raise ValueError(f"protected model is not an absolute stable checkpoint: {path}")
    weight = path / "model.safetensors"
    if weight.is_symlink() or not weight.is_file():
        raise ValueError(f"protected model has no direct dense weight file: {weight}")
    return weight


def metadata_hashes(path: Path) -> dict[str, str]:
    names = (
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    if any((path / name).is_symlink() or not (path / name).is_file() for name in names):
        raise ValueError("checkpoint metadata is incomplete or symlinked")
    return {name: file_sha256(path / name) for name in names}


def source_hashes() -> dict[str, dict[str, str]]:
    modules = {
        "transformers.cache_utils": transformers.cache_utils,
        "transformers.generation.utils": transformers.generation.utils,
        "transformers.loss.loss_utils": transformers.loss.loss_utils,
        "transformers.models.qwen3_5.modeling_qwen3_5": transformers.models.qwen3_5.modeling_qwen3_5,
    }
    environment = SHARED_ENVIRONMENT.resolve(strict=True)
    result = {}
    for name, module in modules.items():
        path = Path(module.__file__)
        resolved = path.resolve(strict=True)
        if path.is_symlink() or not resolved.is_relative_to(environment):
            raise ValueError(f"runtime source outside frozen environment: {name}")
        result[name] = {"path": str(resolved), "sha256": file_sha256(resolved)}
    return result


def validate_no_generation_source(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    forbidden = []
    cpu_seed_calls = 0
    cuda_seed_calls = 0
    compose_gates = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr
            in {
                "generate",
                "prepare_inputs_for_generation",
            }
        ):
            forbidden.append({"line": node.lineno, "attribute": node.func.attr})
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr == "manual_seed"
            ):
                cpu_seed_calls += 1
            if (
                isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "torch"
                and node.func.value.attr == "cuda"
                and node.func.attr == "manual_seed_all"
            ):
                cuda_seed_calls += 1
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "compose_receiver_inputs":
            keyword = next((item for item in node.keywords if item.arg == "gate"), None)
            compose_gates.append(keyword.value.value if keyword and isinstance(keyword.value, ast.Constant) else None)
    if forbidden or cpu_seed_calls != 1 or cuda_seed_calls != 1 or compose_gates != [1.0]:
        raise ValueError("A1-NC0 runner contains a forbidden generation/cache-preparation call")
    return {
        "runner_sha256": file_sha256(path),
        "forbidden_calls": forbidden,
        "generate_used": False,
        "prepare_inputs_for_generation_used": False,
        "torch_manual_seed_call_count": cpu_seed_calls,
        "torch_cuda_manual_seed_all_call_count": cuda_seed_calls,
        "compose_receiver_inputs_gate_values": compose_gates,
        "receiver_gate_applied_by_bridge_then_compose_gate_one": True,
    }


def class_identity(cls: type) -> dict[str, str]:
    module = __import__(cls.__module__, fromlist=["__name__"])
    path = Path(module.__file__).resolve(strict=True)
    if not path.is_relative_to(SHARED_ENVIRONMENT.resolve(strict=True)):
        raise ExperimentIncomplete(f"cache class outside frozen environment: {cls.__module__}.{cls.__qualname__}")
    distribution = "flash-linear-attention" if cls.__module__.split(".", 1)[0] == "fla" else "transformers"
    return {
        "fqcn": f"{cls.__module__}.{cls.__qualname__}",
        "module_path": str(path),
        "module_sha256": file_sha256(path),
        "distribution": f"{distribution}=={importlib.metadata.version(distribution)}",
    }


class CacheGuard:
    def __init__(self) -> None:
        self.base = transformers.cache_utils.Cache
        self.initial_classes = recursive_subclass_closure(self.base)
        self.patched_classes: set[type] = set()
        self.stack = contextlib.ExitStack()
        self.negative_control = False
        self.checks = 0
        self.restored = False

    @staticmethod
    def _reject(cls, *_args, **_kwargs):
        raise CacheAllocationDetected(f"cache allocation attempted: {cls.__module__}.{cls.__qualname__}")

    def __enter__(self):
        try:
            for cls in sorted(self.initial_classes, key=lambda item: (item.__module__, item.__qualname__)):
                self.stack.enter_context(mock.patch.object(cls, "__new__", self._reject))
                self.patched_classes.add(cls)
            try:
                transformers.cache_utils.DynamicCache()
            except CacheAllocationDetected:
                self.negative_control = True
            if not self.negative_control:
                raise ExperimentIncomplete("cache negative control did not trip")
            self.verify_closure()
        except BaseException:
            self.stack.close()
            self.restored = True
            raise
        return self

    def verify_closure(self) -> None:
        new = recursive_subclass_closure(self.base) - self.patched_classes
        if new:
            raise ExperimentIncomplete(
                f"new unpatched cache subclasses loaded: {sorted(f'{c.__module__}.{c.__qualname__}' for c in new)}"
            )
        self.checks += 1

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.verify_closure()
        finally:
            self.stack.close()
            self.restored = True
        return False

    def evidence(self) -> dict[str, object]:
        return {
            "classes": [
                class_identity(cls)
                for cls in sorted(self.initial_classes, key=lambda item: (item.__module__, item.__qualname__))
            ],
            "negative_control_dynamic_cache_tripped": self.negative_control,
            "closure_check_count": self.checks,
            "restored_in_finally": self.restored,
        }


def no_cache_forward(model, call_log: list[dict[str, object]], *, arm: str, **kwargs):
    if (
        kwargs.get("past_key_values", None) is not None
        or kwargs.get("use_cache") is not False
        or model.config.use_cache is not False
        or (getattr(model, "generation_config", None) is not None and model.generation_config.use_cache is not False)
    ):
        raise CacheAllocationDetected("A1-NC0 forward contract supplied cache state")
    entry = {
        "arm": arm,
        "input_ids_is_none": kwargs.get("input_ids") is None,
        "inputs_embeds_is_none": kwargs.get("inputs_embeds") is None,
        "inputs_embeds_object_id": None if kwargs.get("inputs_embeds") is None else id(kwargs["inputs_embeds"]),
        "past_key_values_input_is_none": True,
        "use_cache": False,
        "past_key_values_output_is_none": False,
        "rope_deltas_reset_before_call": False,
    }
    call_log.append(entry)
    reset_rope_state(model)
    if hasattr(model, "model") and hasattr(model.model, "rope_deltas") and model.model.rope_deltas is not None:
        raise CacheAllocationDetected("A1-NC0 RoPE recurrence state did not reset")
    entry["rope_deltas_reset_before_call"] = True
    output = model(return_dict=True, **kwargs)
    if getattr(output, "past_key_values", None) is not None:
        raise CacheAllocationDetected("A1-NC0 forward returned past_key_values")
    entry["past_key_values_output_is_none"] = True
    if hasattr(call_log, "after_forward"):
        call_log.after_forward(arm)
    return output


def render_ids(tokenizer, messages: list[dict[str, object]], *, generation_prompt: bool, tools) -> torch.Tensor:
    ids = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=generation_prompt,
        enable_thinking=False,
        return_tensors="pt",
    )
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
        raise ExperimentIncomplete("chat template did not return one token sequence")
    return ids.to("cuda:0")


def preflight_template_input_ids(
    encoded: object,
    *,
    label: str,
    extraction_counts: collections.Counter[str] | None = None,
) -> list[int]:
    """Extract one flat token sequence from the pinned chat-template container."""
    if not isinstance(encoded, BatchEncoding):
        raise ExperimentIncomplete(f"A1-NC0 {label} template did not return BatchEncoding")
    input_ids = encoded.input_ids
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in input_ids)
    ):
        raise ExperimentIncomplete(f"A1-NC0 {label} template input_ids are not one flat nonempty integer list")
    if extraction_counts is not None:
        extraction_counts[label] += 1
    return list(input_ids)


def parent_messages(evidence: str) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": PARENT_SYSTEM},
        {"role": "user", "content": f"Private source packet:\n{evidence}\n\nCreate the coordinator-child handoff now."},
        {
            "role": "assistant",
            "reasoning_content": PARENT_REASONING,
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "ipython", "arguments": {"code": PARENT_CODE}},
                }
            ],
        },
        {"role": "tool", "name": "ipython", "content": PARENT_TOOL_RESPONSE},
    ]


def self_messages(query: str) -> list[dict[str, object]]:
    return parent_messages(query)


def child_messages(query: str) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": CHILD_SYSTEM},
        {"role": "user", "content": f"Downstream query:\n{query}\nReturn only the requested value."},
    ]


def validate_parent_fixture(evidence: str) -> None:
    messages = parent_messages(evidence)
    assistant = messages[2]
    calls = assistant.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ExperimentIncomplete("A1-NC0 parent fixture must contain one tool call")
    function = calls[0].get("function")
    arguments = function.get("arguments") if isinstance(function, dict) else None
    if function.get("name") != "ipython" or arguments != {"code": PARENT_CODE}:
        raise ExperimentIncomplete("A1-NC0 parent fixture tool call changed")
    tree = ast.parse(PARENT_CODE)
    if (
        len(tree.body) != 1
        or not isinstance(tree.body[0], ast.Assign)
        or len(tree.body[0].targets) != 1
        or not isinstance(tree.body[0].targets[0], ast.Name)
        or tree.body[0].targets[0].id != "child"
        or not isinstance(tree.body[0].value, ast.Await)
        or not isinstance(tree.body[0].value.value, ast.Call)
        or not isinstance(tree.body[0].value.value.func, ast.Name)
        or tree.body[0].value.value.func.id != "rlm"
        or len(tree.body[0].value.value.args) != 1
        or not isinstance(tree.body[0].value.value.args[0], ast.Constant)
        or tree.body[0].value.value.args[0].value
        != "Answer the downstream query supplied in your own child context. Return only the requested value."
        or [(keyword.arg, keyword.value.value) for keyword in tree.body[0].value.value.keywords]
        != [("name", "a1-nc0-child")]
    ):
        raise ExperimentIncomplete("A1-NC0 parent awaited rlm AST changed")
    expected_response = {
        "model": "e33",
        "name": "a1-nc0-child",
        "rlm_child_id": "a1-nc0-fixture-child",
        "session_dir": "/synthetic/a1-nc0-child",
    }
    if (
        json.loads(PARENT_TOOL_RESPONSE) != expected_response
        or json.dumps(expected_response, sort_keys=True, separators=(",", ":")) != PARENT_TOOL_RESPONSE
    ):
        raise ExperimentIncomplete("A1-NC0 tool response bytes changed")
    if evidence in PARENT_CODE or evidence in PARENT_TOOL_RESPONSE:
        raise ExperimentIncomplete("A1-NC0 source packet leaked into delegation call/result")


def synchronize() -> None:
    torch.cuda.synchronize(0)


def reset_rope_state(model) -> None:
    if hasattr(model, "model") and hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None


def cuda_timer_start() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return start, end


def cuda_timer_finish(timer: tuple[torch.cuda.Event, torch.cuda.Event]) -> float:
    start, end = timer
    end.record()
    end.synchronize()
    seconds = float(start.elapsed_time(end)) / 1000.0
    if not math.isfinite(seconds) or seconds < 0:
        raise RuntimeError("A1-NC0 CUDA event timing is invalid")
    return seconds


def feature_capture(
    model,
    tokenizer,
    messages: list[dict[str, object]],
    call_log: list[dict[str, object]],
    *,
    arm: str,
    full_logits_control: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    tools = PARENT_TOOLS if len(messages) == 4 else None
    if tools is not None:
        content = messages[1]["content"]
        prefix = "Private source packet:\n"
        suffix = "\n\nCreate the coordinator-child handoff now."
        if not isinstance(content, str) or not content.startswith(prefix) or not content.endswith(suffix):
            raise ExperimentIncomplete("A1-NC0 parent source wrapper changed")
        validate_parent_fixture(content[len(prefix) : -len(suffix)])
    unpadded = render_ids(tokenizer, messages, generation_prompt=False, tools=tools)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise ExperimentIncomplete("A1-NC0 tokenizer has no pad/eos token")
    padded, mask = fixed_feature_inputs(unpadded, pad_token_id=pad_id, budget=256)
    positions = torch.arange(256, device="cuda:0").unsqueeze(0)
    timer = cuda_timer_start()
    reset_rope_state(model)
    with torch.inference_mode():
        output = no_cache_forward(
            model,
            call_log,
            arm=arm,
            input_ids=padded,
            inputs_embeds=None,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=None,
            use_cache=False,
            output_hidden_states=True,
            logits_to_keep=1,
        )
    elapsed = cuda_timer_finish(timer)
    captured = capture_parent_features(output.hidden_states[-1], mask, HiddenStateCaptureSpec())
    visible = min(unpadded.shape[1], 128)
    expected_indices = list(range(256 - visible, 256))
    expected_mask = torch.cat(
        (
            torch.zeros((1, 128 - visible), dtype=mask.dtype, device="cuda:0"),
            torch.ones((1, visible), dtype=mask.dtype, device="cuda:0"),
        ),
        dim=1,
    )
    if (
        captured.hidden_states.shape != (1, 128, 2048)
        or output.hidden_states[-1].shape != (1, 256, 2048)
        or output.logits.shape != (1, 1, model.config.text_config.vocab_size)
        or not torch.isfinite(output.logits).all()
        or not torch.isfinite(output.hidden_states[-1]).all()
        or captured.attention_mask.shape != (1, 128)
        or not torch.isfinite(captured.hidden_states).all()
        or captured.hidden_states.requires_grad
        or captured.token_indices[-visible:].tolist() != expected_indices
        or any(index != -1 for index in captured.token_indices[:-visible].tolist())
        or not torch.equal(captured.attention_mask, expected_mask)
        or torch.count_nonzero(captured.hidden_states[:, : 128 - visible]).item() != 0
        or not torch.equal(captured.hidden_states[:, -visible:], output.hidden_states[-1][:, expected_indices, :])
    ):
        raise MechanismRejected("A1-NC0 capture contract failed", reason="capture_numeric_or_content_rejected")
    control_evidence = None
    if full_logits_control:
        reset_rope_state(model)
        with torch.inference_mode():
            control = no_cache_forward(
                model,
                call_log,
                arm=f"{arm}_KEEP0_CONTROL",
                input_ids=padded,
                inputs_embeds=None,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=None,
                use_cache=False,
                output_hidden_states=True,
                logits_to_keep=0,
            )
        control_capture = capture_parent_features(control.hidden_states[-1], mask, HiddenStateCaptureSpec())
        if (
            control.logits.shape != (1, 256, model.config.text_config.vocab_size)
            or not torch.isfinite(control.logits).all()
            or not torch.equal(control.logits[:, -1:], output.logits)
            or not torch.equal(control.hidden_states[-1], output.hidden_states[-1])
            or not torch.equal(control_capture.hidden_states, captured.hidden_states)
        ):
            raise MechanismRejected(
                "A1-NC0 capture keep0 control changed hidden state", reason="capture_parity_rejected"
            )
        control_evidence = {
            "full_hidden_bitwise_equal": True,
            "selected_hidden_bitwise_equal": True,
            "keep0_full_hidden_sha256": tensor_sha256(control.hidden_states[-1]),
            "keep0_last_logits_sha256": tensor_sha256(control.logits[:, -1:]),
            "keep1_logits_sha256": tensor_sha256(output.logits),
            "last_logits_bitwise_equal": True,
        }
        del control
    return (
        captured.hidden_states,
        captured.attention_mask,
        {
            "unpadded_tokens": unpadded.shape[1],
            "padded_tokens": padded.shape[1],
            "tokens_truncated": 0,
            "input_ids_sha256": tensor_sha256(padded),
            "attention_mask_sha256": tensor_sha256(mask),
            "captured_hidden_sha256": tensor_sha256(captured.hidden_states),
            "full_final_hidden_sha256": tensor_sha256(output.hidden_states[-1]),
            "keep1_logits_sha256": tensor_sha256(output.logits),
            "captured_mask_sha256": tensor_sha256(captured.attention_mask),
            "captured_token_indices": captured.token_indices.tolist(),
            "captured_visible_tokens": visible,
            "captured_zero_left_padding": True,
            "captured_suffix_matches_final_hidden_bitwise": True,
            "capture_spec_sha256": captured.capture_spec_hash,
            "gpu_seconds": elapsed,
            "keep0_control": control_evidence,
        },
    )


def bridge_slots(
    bridge: WorkspaceBridge,
    hidden: torch.Tensor,
    mask: torch.Tensor,
    embedding_shell_norm: torch.Tensor,
    call_log: list[dict[str, object]],
    *,
    arm: str,
) -> tuple[torch.Tensor, dict[str, object]]:
    timer = cuda_timer_start()
    workspace_f32 = bridge.encoder(hidden.float(), mask)
    receiver_f32 = bridge.decoder(workspace_f32, embedding_shell_norm=embedding_shell_norm)
    slots = receiver_f32.to(dtype=torch.bfloat16).contiguous()
    elapsed = cuda_timer_finish(timer)
    if hasattr(call_log, "after_bridge"):
        call_log.after_bridge(arm)
    if (
        workspace_f32.dtype != torch.float32
        or receiver_f32.dtype != torch.float32
        or slots.shape != (1, 8, 2048)
        or not torch.isfinite(slots.float()).all()
    ):
        raise MechanismRejected("A1-NC0 bridge output contract failed", reason="bridge_numeric_contract_rejected")
    return slots, {
        "encoder_workspace_float32_sha256": tensor_sha256(workspace_f32),
        "receiver_precast_float32_sha256": tensor_sha256(receiver_f32),
        "receiver_final_bfloat16_sha256": tensor_sha256(slots),
        "receiver_gate_applied_exactly_once": True,
        "gpu_seconds": elapsed,
    }


def receiver_inputs(
    model,
    tokenizer,
    query: str,
    answer: str | None,
    workspace_slots: torch.Tensor | None,
) -> dict[str, torch.Tensor | int]:
    messages = child_messages(query)
    prefix_without_opening = render_ids(tokenizer, messages, generation_prompt=False, tools=None)
    prompt = render_ids(tokenizer, messages, generation_prompt=True, tools=None)
    injection = prefix_without_opening.shape[1]
    if not torch.equal(prompt[:, :injection], prefix_without_opening):
        raise ExperimentIncomplete("A1-NC0 assistant-opening boundary changed")
    answer_ids = None
    full_ids = prompt
    if answer is not None:
        full_messages = [*messages, {"role": "assistant", "reasoning_content": "", "content": answer}]
        plain_text = tokenizer.apply_chat_template(
            messages, tools=None, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        opening_text = tokenizer.apply_chat_template(
            messages, tools=None, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full_text = tokenizer.apply_chat_template(
            full_messages, tools=None, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        if full_text != opening_text + answer + "<|im_end|>\n" or opening_text != plain_text + CHILD_OPENING_SUFFIX:
            raise ExperimentIncomplete("A1-NC0 child rendered text boundary changed")
        if not opening_text.endswith(CHILD_OPENING_SUFFIX):
            raise ExperimentIncomplete("A1-NC0 child assistant opening suffix changed")
        encoded = tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda:0")
        if encoded.ndim != 2 or encoded.shape[0] != 1 or not 1 <= encoded.shape[1] <= 12:
            raise ExperimentIncomplete("A1-NC0 answer-token span is outside the frozen bound")
        answer_ids = encoded
        rendered_full = render_ids(tokenizer, full_messages, generation_prompt=False, tools=None)
        if (
            rendered_full.shape[1] < 2
            or rendered_full[:, -2:].flatten().tolist() != TERMINAL_TOKEN_IDS
            or not torch.equal(rendered_full, torch.cat((prompt, answer_ids, rendered_full[:, -2:]), dim=1))
        ):
            raise ExperimentIncomplete("A1-NC0 child full token boundary changed")
        full_ids = rendered_full[:, :-2]
    mask = torch.ones_like(full_ids)
    positions = torch.arange(full_ids.shape[1], device="cuda:0").unsqueeze(0)
    labels = None
    if answer_ids is not None:
        labels = torch.full_like(full_ids, -100)
        labels[:, prompt.shape[1] : prompt.shape[1] + answer_ids.shape[1]] = answer_ids
    if workspace_slots is None:
        return {
            "input_ids": full_ids,
            "attention_mask": mask,
            "position_ids": positions,
            "labels": labels,
            "answer_tokens": 0 if answer_ids is None else answer_ids.shape[1],
            "injection_index": injection,
            "label_alignment": None,
        }
    token_embeddings = model.get_input_embeddings()(full_ids)
    composed = compose_receiver_inputs(
        token_embeddings,
        mask,
        workspace_slots,
        injection_index=injection,
        gate=1.0,
        position_ids=positions,
        labels=labels,
    )
    if (
        composed.workspace_span != (injection, injection + 8)
        or composed.inputs_embeds.shape[1] != full_ids.shape[1] + 8
        or not torch.equal(composed.attention_mask, torch.ones_like(composed.attention_mask))
        or not torch.equal(
            composed.position_ids,
            torch.arange(full_ids.shape[1] + 8, device="cuda:0").unsqueeze(0),
        )
        or not torch.equal(composed.inputs_embeds[:, injection : injection + 8], workspace_slots)
        or not torch.equal(composed.inputs_embeds[:, :injection], token_embeddings[:, :injection])
        or not torch.equal(composed.inputs_embeds[:, injection + 8 :], token_embeddings[:, injection:])
        or (labels is not None and torch.count_nonzero(composed.labels[:, injection : injection + 8] != -100).item())
    ):
        raise MechanismRejected("A1-NC0 receiver injection contract failed", reason="receiver_geometry_rejected")
    label_alignment = None
    if answer_ids is not None:
        active = torch.nonzero(composed.labels[0] != -100, as_tuple=False).flatten().tolist()
        expected_active = list(range(prompt.shape[1] + 8, full_ids.shape[1] + 8))
        active_values = composed.labels[0, active].tolist()
        if (
            active != expected_active
            or active_values != answer_ids.flatten().tolist()
            or torch.count_nonzero(composed.labels != -100).item() != answer_ids.shape[1]
        ):
            raise MechanismRejected("A1-NC0 shifted loss span changed", reason="receiver_geometry_rejected")
        label_alignment = {
            "active_label_positions": active,
            "active_logit_positions": [position - 1 for position in active],
            "raw_answer_token_ids": answer_ids.flatten().tolist(),
            "terminal_token_ids": TERMINAL_TOKEN_IDS,
            "terminal_ids_excluded_from_teacher_input": True,
            "all_other_labels_masked": True,
        }
    return {
        "inputs_embeds": composed.inputs_embeds,
        "attention_mask": composed.attention_mask,
        "position_ids": composed.position_ids,
        "labels": composed.labels,
        "answer_tokens": 0 if answer_ids is None else answer_ids.shape[1],
        "injection_index": injection,
        "label_alignment": label_alignment,
    }


def answer_token_loss(
    model,
    batch: dict[str, torch.Tensor | int],
    call_log: list[dict[str, object]],
    *,
    arm: str,
    full_logits_control: bool = False,
) -> tuple[torch.Tensor, dict[str, object]]:
    kwargs = {
        key: value
        for key, value in batch.items()
        if key in {"input_ids", "inputs_embeds", "attention_mask", "position_ids", "labels"}
    }
    labels = kwargs.pop("labels")
    values = labels[0].detach().cpu().tolist()
    active = [index for index, value in enumerate(values) if value != -100]
    if (
        not active
        or active[0] == 0
        or active != list(range(active[0], active[-1] + 1))
        or active[-1] != len(values) - 1
        or any(value != -100 for value in values[: active[0]])
    ):
        raise MechanismRejected("A1-NC0 supervised answer span changed", reason="receiver_geometry_rejected")
    start = active[0] - 1
    keep = len(values) - start
    suffix_labels = labels[:, start:]
    original_pairs = [(index - 1, values[index]) for index in active]
    suffix_values = suffix_labels[0].detach().cpu().tolist()
    suffix_pairs = [(start + index - 1, value) for index, value in enumerate(suffix_values) if value != -100]
    if original_pairs != suffix_pairs or suffix_labels.shape[1] != keep:
        raise MechanismRejected("A1-NC0 suffix objective changed causal pairs", reason="receiver_geometry_rejected")
    output = no_cache_forward(
        model,
        call_log,
        arm=arm,
        past_key_values=None,
        use_cache=False,
        labels=suffix_labels,
        logits_to_keep=keep,
        **kwargs,
    )
    loss = output.loss
    if loss is None or not torch.isfinite(loss):
        raise MechanismRejected("A1-NC0 answer loss is nonfinite", reason="receiver_numeric_rejected")
    if output.logits.shape[1] != keep or not torch.isfinite(output.logits).all():
        raise MechanismRejected("A1-NC0 suffix logits changed", reason="receiver_numeric_rejected")
    full_control = None
    if full_logits_control:
        with torch.inference_mode():
            control = no_cache_forward(
                model,
                call_log,
                arm=f"{arm}_FULL_LOGITS_CONTROL",
                past_key_values=None,
                use_cache=False,
                labels=labels,
                logits_to_keep=0,
                **kwargs,
            )
        if (
            control.loss is None
            or not torch.isfinite(control.loss)
            or control.logits.shape[1] != labels.shape[1]
            or not torch.isfinite(control.logits).all()
            or not torch.equal(control.logits[:, -keep:], output.logits)
            or not torch.equal(control.loss, loss.detach())
        ):
            raise MechanismRejected(
                "A1-NC0 full-vs-suffix loss control changed",
                reason="receiver_suffix_objective_rejected",
            )
        full_control = {
            "last_k_logits_bitwise_equal": True,
            "loss_bitwise_equal": True,
            "full_logits_sha256": tensor_sha256(control.logits),
            "suffix_logits_sha256": tensor_sha256(output.logits),
            "full_loss_sha256": tensor_sha256(control.loss),
            "suffix_loss_sha256": tensor_sha256(loss.detach()),
        }
    return loss, {
        "first_active_label_index": active[0],
        "logit_suffix_start": start,
        "logits_to_keep": keep,
        "active_label_count": len(active),
        "active_causal_pairs_sha256": canonical_json_hash(original_pairs),
        "active_causal_pairs_unchanged": True,
        "terminal_ids_excluded_from_teacher_input": True,
        "full_logits_control": full_control,
    }


def greedy_full_recompute(
    model,
    tokenizer,
    query: str,
    answer: str,
    workspace_slots: torch.Tensor | None,
    call_log: list[dict[str, object]],
    *,
    arm: str,
) -> tuple[list[int], str, float, int, list[dict[str, object]], float]:
    batch = receiver_inputs(model, tokenizer, query, None, workspace_slots)
    input_ids = batch.get("input_ids")
    inputs_embeds = batch.get("inputs_embeds")
    mask = batch["attention_mask"]
    positions = batch["position_ids"]
    gold = tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids.flatten().tolist()
    if not 1 <= len(gold) <= 12 or 248046 in gold or 198 in gold:
        raise ExperimentIncomplete("A1-NC0 gold answer-token count is outside 1..12")
    terminal = 248046
    generated: list[int] = []
    evidence = []
    nlls = []
    eos_seen = False
    decode_timer = cuda_timer_start()
    for step in range(12):
        output = no_cache_forward(
            model,
            call_log,
            arm=f"{arm}_DECODE",
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=None,
            use_cache=False,
            logits_to_keep=1,
        )
        logits = output.logits[:, -1].float()
        if not torch.isfinite(logits).all():
            raise MechanismRejected("A1-NC0 generation logits are nonfinite", reason="receiver_numeric_rejected")
        argmax = None if eos_seen else int(torch.argmax(logits, dim=-1).item())
        token = terminal if eos_seen else argmax
        gold_id = gold[step] if step < len(gold) else None
        step_nll = None
        if gold_id is not None:
            step_nll = float((-torch.log_softmax(logits, dim=-1)[0, gold_id]).item())
            if not math.isfinite(step_nll) or step_nll < 0:
                raise MechanismRejected("A1-NC0 rollout NLL is nonfinite", reason="receiver_numeric_rejected")
            nlls.append(step_nll)
        prefix = input_ids if input_ids is not None else inputs_embeds
        generated.append(token)
        evidence.append(
            {
                "step": step + 1,
                "prefix_length": mask.shape[1],
                "argmax_token_id": argmax,
                "appended_token_id": token,
                "forced_after_eos": eos_seen,
                "terminal_selected": argmax == terminal if argmax is not None else False,
                "gold_token_id": gold_id,
                "gold_token_nll": step_nll,
                "prefix_sha256": tensor_sha256(prefix),
                "attention_mask_sha256": tensor_sha256(mask),
                "position_ids_sha256": tensor_sha256(positions),
                "logits_sha256": tensor_sha256(logits),
            }
        )
        eos_seen = eos_seen or token == terminal
        token_tensor = torch.tensor([[token]], device="cuda:0")
        if input_ids is not None:
            input_ids = torch.cat((input_ids, token_tensor), dim=1)
        else:
            token_embedding = model.get_input_embeddings()(token_tensor)
            inputs_embeds = torch.cat((inputs_embeds, token_embedding), dim=1)
        mask = torch.cat((mask, torch.ones_like(token_tensor)), dim=1)
        positions = torch.cat((positions, positions[:, -1:] + 1), dim=1)
    visible = generated[: generated.index(terminal)] if terminal in generated else generated
    text = tokenizer.decode(visible, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip(" \t\r\n")
    decode_cuda_seconds = cuda_timer_finish(decode_timer)
    return generated, text, math.fsum(nlls) / len(gold), len(gold), evidence, decode_cuda_seconds


def _selected_records(train_artifact: dict[str, object], schedule: dict[str, object]):
    records = {record["evidence_id"]: record for record in train_artifact["bank"]["records"]}
    result = []
    for selected in schedule["a0nc_repeat_selection"]:
        record = records.get(selected["evidence_id"])
        if record is None or record["family"] != selected["family"]:
            raise ExperimentIncomplete("A1-NC0 repeat selection does not resolve")
        queries = {query["query_id"]: query for query in record["queries"]}
        if selected["query_id"] not in queries:
            raise ExperimentIncomplete("A1-NC0 repeat query does not resolve")
        result.append((record, queries[selected["query_id"]]))
    return result


def _gradient_group_prefixes() -> dict[str, tuple[str, ...]]:
    return {
        "source_norm": ("encoder.source_norm.",),
        "source_projection": ("encoder.source_projection.",),
        "learned_queries": ("encoder.learned_queries",),
        "resampler_in_proj": ("encoder.resampler.in_proj_",),
        "resampler_out_proj": ("encoder.resampler.out_proj.",),
        "output_norm": ("encoder.output_norm.",),
        "decoder_workspace_norm": ("decoder.workspace_norm.",),
        "decoder_projection": ("decoder.projection.",),
        "receiver_gate": ("decoder.receiver_gate",),
    }


def _gradient_groups(bridge: WorkspaceBridge, *, require_nonzero: bool = True) -> dict[str, float]:
    prefixes = _gradient_group_prefixes()
    norms = {}
    named = dict(bridge.named_parameters())
    for group, choices in prefixes.items():
        selected = [(name, parameter) for name, parameter in sorted(named.items()) if name.startswith(choices)]
        if not selected or any(parameter.grad is None for _name, parameter in selected):
            raise MechanismRejected("A1-NC0 gradient group missing", reason="gradient_connectivity_rejected")
        tensors = [(name, parameter.grad.detach().double()) for name, parameter in selected]
        if any(not torch.isfinite(tensor).all() for _name, tensor in tensors):
            raise MechanismRejected("A1-NC0 gradient group nonfinite", reason="gradient_connectivity_rejected")
        square = math.fsum(float(torch.sum(tensor * tensor).item()) for _name, tensor in tensors)
        norm = math.sqrt(square)
        if not math.isfinite(norm) or (require_nonzero and norm <= 0):
            raise MechanismRejected("A1-NC0 gradient group nonfinite or zero", reason="gradient_connectivity_rejected")
        norms[group] = norm
    return norms


def run_a0nc_repeat(
    model,
    tokenizer,
    bridge: WorkspaceBridge,
    shell_norm: torch.Tensor,
    train_artifact: dict[str, object],
    schedule: dict[str, object],
    call_log: list[dict[str, object]],
) -> dict[str, object]:
    observed_continuation = (
        tokenizer(FIXED_CONTINUATION_TEXT, add_special_tokens=False, return_tensors="pt").input_ids[:, :4].to("cuda:0")
    )
    if observed_continuation.flatten().tolist() != FIXED_CONTINUATION_IDS:
        raise MechanismRejected("A1-NC0 fixed continuation changed", reason="pretraining_parity_rejected")
    initial_hash = bridge_parameter_sha256(bridge)
    bridge.zero_grad(set_to_none=True)
    probes = []
    call_start = len(call_log)
    bridge_start = getattr(call_log, "bridge_forwards", 0)
    for probe_index, (record, query) in enumerate(_selected_records(train_artifact, schedule)):
        hidden1, mask1, capture1 = feature_capture(
            model,
            tokenizer,
            parent_messages(record["parent_evidence"]),
            call_log,
            arm="A0NC_REPEAT_CAPTURE",
            full_logits_control=probe_index == 0,
        )
        hidden2, mask2, capture2 = feature_capture(
            model,
            tokenizer,
            parent_messages(record["parent_evidence"]),
            call_log,
            arm="A0NC_REPEAT_CAPTURE_REPEAT",
        )
        if not torch.equal(hidden1, hidden2) or not torch.equal(mask1, mask2):
            raise MechanismRejected("A1-NC0 repeated capture changed", reason="pretraining_parity_rejected")
        slots1, bridge1 = bridge_slots(bridge, hidden1, mask1, shell_norm, call_log, arm="A0NC_REPEAT_BRIDGE")
        slots2, bridge2 = bridge_slots(bridge, hidden2, mask2, shell_norm, call_log, arm="A0NC_REPEAT_BRIDGE_REPEAT")
        if not torch.equal(slots1, slots2):
            raise MechanismRejected("A1-NC0 repeated bridge changed", reason="pretraining_parity_rejected")

        messages = child_messages(query["child_query"])
        child_prefix = render_ids(tokenizer, messages, generation_prompt=False, tools=None)
        child_ids = render_ids(tokenizer, messages, generation_prompt=True, tools=None)
        injection = child_prefix.shape[1]
        fixed = torch.tensor([LENGTH_CONTROL_IDS], device="cuda:0")
        length_ids = torch.cat((child_ids[:, :injection], fixed, child_ids[:, injection:]), dim=1)
        exact = model.get_input_embeddings()(length_ids)
        soft_batch = receiver_inputs(model, tokenizer, query["child_query"], None, slots1)
        soft = soft_batch["inputs_embeds"]
        mask = torch.ones_like(length_ids)
        positions = torch.arange(length_ids.shape[1], device="cuda:0").unsqueeze(0)
        soft_span = soft[:, injection : injection + 8]
        hard_span = exact[:, injection : injection + 8]
        outside_exact = torch.equal(soft[:, :injection], exact[:, :injection]) and torch.equal(
            soft[:, injection + 8 :], exact[:, injection + 8 :]
        )
        if (
            soft.shape != exact.shape
            or not torch.equal(soft_batch["attention_mask"], mask)
            or torch.count_nonzero(soft_span).item() == 0
            or torch.equal(soft_span, hard_span)
            or not outside_exact
        ):
            raise MechanismRejected("A1-NC0 repeat geometry changed", reason="pretraining_parity_rejected")
        steps = []
        ids = length_ids
        exact_embeds = exact
        soft_embeds = soft
        for index, continuation_id in enumerate(FIXED_CONTINUATION_IDS, start=1):
            kwargs = {
                "attention_mask": mask,
                "position_ids": positions,
                "past_key_values": None,
                "use_cache": False,
                "logits_to_keep": 1,
            }
            input_hashes = {
                "l_id_input_ids_sha256": tensor_sha256(ids),
                "l_e_inputs_embeds_sha256": tensor_sha256(exact_embeds),
                "shared_soft_inputs_embeds_sha256": tensor_sha256(soft_embeds),
                "attention_mask_sha256": tensor_sha256(mask),
                "position_ids_sha256": tensor_sha256(positions),
            }
            soft_identity = id(soft_embeds)
            calls_before = len(call_log)
            with torch.inference_mode():
                l_id = no_cache_forward(model, call_log, arm="A0NC_REPEAT_L_ID", input_ids=ids, **kwargs)
                l_e = no_cache_forward(model, call_log, arm="A0NC_REPEAT_L_E", inputs_embeds=exact_embeds, **kwargs)
                s1 = no_cache_forward(model, call_log, arm="A0NC_REPEAT_S", inputs_embeds=soft_embeds, **kwargs)
                s2 = no_cache_forward(model, call_log, arm="A0NC_REPEAT_S_REPEAT", inputs_embeds=soft_embeds, **kwargs)
            logits = [output.logits[:, -1].float() for output in (l_id, l_e, s1, s2)]
            step_calls = call_log[calls_before:]
            actual_soft_ids = [
                entry["inputs_embeds_object_id"]
                for entry in step_calls
                if entry["arm"] in {"A0NC_REPEAT_S", "A0NC_REPEAT_S_REPEAT"}
            ]
            if (
                not all(torch.isfinite(item).all() for item in logits)
                or not torch.equal(logits[0], logits[1])
                or not torch.equal(logits[2], logits[3])
                or actual_soft_ids != [soft_identity, soft_identity]
            ):
                raise MechanismRejected("A1-NC0 repeat logit parity failed", reason="pretraining_parity_rejected")
            steps.append(
                {
                    "step": index,
                    "continuation_token_id": continuation_id,
                    "l_id_l_e_bitwise_equal": True,
                    "soft_repeat_bitwise_equal": True,
                    "l_id_logits_sha256": tensor_sha256(logits[0]),
                    "l_e_logits_sha256": tensor_sha256(logits[1]),
                    "soft_logits_sha256": tensor_sha256(logits[2]),
                    "soft_repeat_logits_sha256": tensor_sha256(logits[3]),
                    "soft_same_tensor_object_for_repeat": actual_soft_ids == [soft_identity, soft_identity],
                    "soft_input_unchanged_after_forwards": tensor_sha256(soft_embeds)
                    == input_hashes["shared_soft_inputs_embeds_sha256"],
                    **input_hashes,
                }
            )
            token = torch.tensor([[continuation_id]], device="cuda:0")
            token_embedding = model.get_input_embeddings()(token)
            ids = torch.cat((ids, token), dim=1)
            exact_embeds = torch.cat((exact_embeds, token_embedding), dim=1)
            soft_embeds = torch.cat((soft_embeds, token_embedding), dim=1)
            mask = torch.cat((mask, torch.ones_like(token)), dim=1)
            positions = torch.cat((positions, positions[:, -1:] + 1), dim=1)

        gradient_batch = receiver_inputs(model, tokenizer, query["child_query"], query["answer"], slots1)
        loss, suffix_objective = answer_token_loss(
            model,
            gradient_batch,
            call_log,
            arm="A0NC_REPEAT_GRADIENT",
            full_logits_control=probe_index == 0,
        )
        (loss / 4).backward()
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise MechanismRejected(
                "A1-NC0 e33 gradient appeared during preprobe",
                reason="protected_mutation_rejected",
            )
        probes.append(
            {
                "family": record["family"],
                "evidence_id": record["evidence_id"],
                "query_id": query["query_id"],
                "capture_repeat_bitwise": True,
                "capture": [capture1, capture2],
                "bridge_repeat_bitwise": True,
                "bridge": [bridge1, bridge2],
                "soft_span_active": True,
                "soft_span_differs_from_hard": True,
                "outside_soft_span_exact": True,
                "steps": steps,
                "answer_token_count": gradient_batch["answer_tokens"],
                "suffix_objective": suffix_objective,
                "loss": float(loss.detach().item()),
            }
        )
    gradient_norms = _gradient_groups(bridge)
    final_hash = bridge_parameter_sha256(bridge)
    if initial_hash != final_hash or any(parameter.grad is not None for parameter in model.parameters()):
        raise MechanismRejected("A1-NC0 preprobe mutated protected parameters", reason="protected_mutation_rejected")
    bridge.zero_grad(set_to_none=True)
    if len(call_log) - call_start != 78:
        raise ExperimentIncomplete("A1-NC0 pretraining repeat did not make exactly 78 e33 forwards")
    histogram = {}
    for entry in call_log[call_start:]:
        histogram[entry["arm"]] = histogram.get(entry["arm"], 0) + 1
    if histogram != {
        "A0NC_REPEAT_CAPTURE": 4,
        "A0NC_REPEAT_CAPTURE_REPEAT": 4,
        "A0NC_REPEAT_L_ID": 16,
        "A0NC_REPEAT_L_E": 16,
        "A0NC_REPEAT_S": 16,
        "A0NC_REPEAT_S_REPEAT": 16,
        "A0NC_REPEAT_GRADIENT": 4,
        "A0NC_REPEAT_CAPTURE_KEEP0_CONTROL": 1,
        "A0NC_REPEAT_GRADIENT_FULL_LOGITS_CONTROL": 1,
    }:
        raise ExperimentIncomplete("A1-NC0 pretraining repeat call histogram changed")
    bridge_histogram: dict[str, int] = {}
    for bridge_arm in getattr(call_log, "bridge_log", [])[bridge_start:]:
        bridge_histogram[bridge_arm] = bridge_histogram.get(bridge_arm, 0) + 1
    if bridge_histogram != {"A0NC_REPEAT_BRIDGE": 4, "A0NC_REPEAT_BRIDGE_REPEAT": 4}:
        raise ExperimentIncomplete("A1-NC0 pretraining repeat bridge histogram changed")
    return {
        "selection": schedule["a0nc_repeat_selection"],
        "selection_sha256": schedule["a0nc_repeat_selection_sha256"],
        "fixed_continuation_text": FIXED_CONTINUATION_TEXT,
        "fixed_continuation_token_ids": FIXED_CONTINUATION_IDS,
        "length_control_token_ids": LENGTH_CONTROL_IDS,
        "probes": probes,
        "gradient_group_l2": gradient_norms,
        "bridge_parameter_sha256_before": initial_hash,
        "bridge_parameter_sha256_after": final_hash,
        "base_model_gradients_absent": True,
        "optimizer_step": False,
        "e33_forward_count": 78,
        "e33_call_histogram": histogram,
        "bridge_forward_count": 8,
        "bridge_call_histogram": bridge_histogram,
    }


def build_train_feature_cache(
    model,
    tokenizer,
    train_artifact: dict[str, object],
    call_log: list[dict[str, object]],
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], list[dict[str, object]]]:
    cache = {}
    evidence = []
    for record in train_artifact["bank"]["records"]:
        hidden, mask, capture = feature_capture(
            model,
            tokenizer,
            parent_messages(record["parent_evidence"]),
            call_log,
            arm="TRAIN_PARENT_FEATURE",
        )
        host_hidden = hidden.detach().to(device="cpu").contiguous().clone()
        host_mask = mask.detach().to(device="cpu").contiguous().clone()
        if host_hidden.device.type != "cpu" or host_mask.device.type != "cpu":
            raise MechanismRejected("A1-NC0 feature cache is not host-resident", reason="feature_cache_rejected")
        cache[record["evidence_id"]] = (host_hidden, host_mask)
        evidence.append(
            {
                "evidence_id": record["evidence_id"],
                "host_hidden_sha256": tensor_sha256(host_hidden),
                "host_mask_sha256": tensor_sha256(host_mask),
                "detached": not host_hidden.requires_grad,
                "device": "cpu",
                **capture,
            }
        )
    if len(cache) != 64 or any(item["tokens_truncated"] != 0 for item in evidence):
        raise ExperimentIncomplete("A1-NC0 train feature cache is incomplete or truncated")
    return cache, evidence


def train_bridge(
    model,
    tokenizer,
    bridge: WorkspaceBridge,
    shell_norm: torch.Tensor,
    train_artifact: dict[str, object],
    schedule: dict[str, object],
    feature_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    call_log: list[dict[str, object]],
    memory_checkpoint,
) -> tuple[list[dict[str, object]], torch.optim.Optimizer, dict[str, object]]:
    records = train_artifact["bank"]["records"]
    query_lookup = {query["query_id"]: (record, query) for record in records for query in record["queries"]}
    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=0.0001,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    bridge.train()
    updates = []
    feature_cache_before = {
        evidence_id: (tensor_sha256(hidden), tensor_sha256(mask))
        for evidence_id, (hidden, mask) in feature_cache.items()
    }
    epoch_positive_groups: dict[int, set[str]] = {epoch: set() for epoch in range(1, 5)}
    for update in schedule["train_updates"]:
        optimizer.zero_grad(set_to_none=True)
        losses = []
        objectives = []
        workspace_hashes: dict[str, list[str]] = {}
        for query_id in update["query_ids"]:
            record, query = query_lookup[query_id]
            host_hidden, host_parent_mask = feature_cache[record["evidence_id"]]
            hidden = host_hidden.to(device="cuda:0")
            parent_mask = host_parent_mask.to(device="cuda:0")
            slots, _ = bridge_slots(bridge, hidden, parent_mask, shell_norm, call_log, arm="TRAIN_MCUR")
            workspace_hashes.setdefault(record["evidence_id"], []).append(tensor_sha256(slots))
            batch = receiver_inputs(model, tokenizer, query["child_query"], query["answer"], slots)
            loss, suffix_objective = answer_token_loss(model, batch, call_log, arm="TRAIN_MCUR")
            (loss / 12).backward()
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise MechanismRejected(
                    "A1-NC0 base-model gradient appeared after row backward",
                    reason="protected_mutation_rejected",
                )
            memory_checkpoint(f"train_update_{update['update_index']:02d}_row_{len(losses) + 1:02d}_post_backward")
            losses.append(float(loss.detach().item()))
            if suffix_objective["active_label_count"] != batch["answer_tokens"]:
                raise MechanismRejected("A1-NC0 train suffix count changed", reason="receiver_geometry_rejected")
            objectives.append({"query_id": query_id, **suffix_objective})
            del loss, batch, slots, hidden, parent_mask
        if len(workspace_hashes) != 4 or any(
            len(hashes) != 3 or len(set(hashes)) != 1 for hashes in workspace_hashes.values()
        ):
            raise MechanismRejected(
                "A1-NC0 q0/q1/q2 workspace changed within update", reason="workspace_reuse_rejected"
            )
        group_norms = _gradient_groups(bridge, require_nonzero=False)
        gate_norm = group_norms["receiver_gate"]
        if gate_norm <= 0:
            raise MechanismRejected(
                "A1-NC0 receiver gate gradient is zero",
                reason="gradient_connectivity_rejected",
            )
        epoch_positive_groups[int(update["epoch"])].update(group for group, value in group_norms.items() if value > 0)
        norm = torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        memory_checkpoint(f"train_update_{update['update_index']:02d}_clip")
        if not torch.isfinite(norm):
            raise MechanismRejected("A1-NC0 train gradient norm is nonfinite", reason="gradient_connectivity_rejected")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise MechanismRejected("A1-NC0 base-model gradient appeared", reason="protected_mutation_rejected")
        optimizer.step()
        memory_checkpoint(f"train_update_{update['update_index']:02d}_optimizer_step")
        if not all(torch.isfinite(parameter).all() for parameter in bridge.parameters()):
            raise MechanismRejected(
                "A1-NC0 bridge parameter became nonfinite", reason="bridge_numeric_contract_rejected"
            )
        updates.append(
            {
                "epoch": update["epoch"],
                "update_index": update["update_index"],
                "query_ids_sha256": canonical_json_hash(update["query_ids"]),
                "query_exposures": 12,
                "mean_loss": math.fsum(losses) / 12,
                "suffix_objectives": objectives,
                "base_model_gradients_absent_after_each_row": True,
                "preclip_gradient_l2": float(norm.item()),
                "gradient_group_l2": group_norms,
                "receiver_gate_gradient_finite_nonzero": True,
                "within_update_evidence_workspace_sha256": {
                    evidence_id: hashes[0] for evidence_id, hashes in workspace_hashes.items()
                },
                "bridge_parameter_sha256_after": bridge_parameter_sha256(bridge),
            }
        )
        if update["update_index"] % 16 == 0:
            if epoch_positive_groups[int(update["epoch"])] != set(_gradient_group_prefixes()):
                raise MechanismRejected(
                    "A1-NC0 bridge gradient group was never nonzero in epoch",
                    reason="gradient_connectivity_rejected",
                )
            memory_checkpoint(f"train_epoch_{update['epoch']}_complete")
    if len(updates) != 64 or sum(item["query_exposures"] for item in updates) != 768:
        raise ExperimentIncomplete("A1-NC0 frozen training budget did not complete")
    bridge.eval()
    feature_cache_after = {
        evidence_id: {
            "hidden_sha256": tensor_sha256(hidden),
            "mask_sha256": tensor_sha256(mask),
            "device": hidden.device.type,
        }
        for evidence_id, (hidden, mask) in feature_cache.items()
    }
    if any(
        feature_cache_before[evidence_id] != (evidence["hidden_sha256"], evidence["mask_sha256"])
        or evidence["device"] != "cpu"
        for evidence_id, evidence in feature_cache_after.items()
    ):
        raise MechanismRejected(
            "A1-NC0 immutable host feature cache changed",
            reason="feature_cache_rejected",
        )
    return (
        updates,
        optimizer,
        {
            "feature_cache_after": feature_cache_after,
            "all_64_host_features_unchanged_after_768_exposures": True,
            "epoch_positive_gradient_groups": {
                str(epoch): sorted(groups) for epoch, groups in epoch_positive_groups.items()
            },
        },
    )


def _record_lookup(artifact: dict[str, object]) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    records = {record["evidence_id"]: record for record in artifact["bank"]["records"]}
    queries = {
        query["query_id"]: {"record": record, "query": query}
        for record in records.values()
        for query in record["queries"]
    }
    return records, queries


def _timed_workspace(
    model,
    tokenizer,
    bridge,
    shell_norm,
    messages,
    call_log,
    *,
    arm,
):
    timer = cuda_timer_start()
    wall_started = time.perf_counter()
    hidden, mask, capture = feature_capture(model, tokenizer, messages, call_log, arm=f"{arm}_FEATURE")
    with torch.inference_mode():
        slots, bridge_evidence = bridge_slots(bridge, hidden, mask, shell_norm, call_log, arm=f"{arm}_BRIDGE")
    elapsed = cuda_timer_finish(timer)
    return slots, {
        "feature": capture,
        "bridge": bridge_evidence,
        "feature_bridge_cuda_event_seconds": elapsed,
        "feature_bridge_wall_seconds": time.perf_counter() - wall_started,
    }


def _workspace_evidence_matches(observed: dict[str, object], expected: dict[str, object]) -> bool:
    capture_keys = (
        "captured_hidden_sha256",
        "captured_mask_sha256",
        "captured_token_indices",
        "full_final_hidden_sha256",
    )
    bridge_keys = (
        "encoder_workspace_float32_sha256",
        "receiver_precast_float32_sha256",
        "receiver_final_bfloat16_sha256",
    )
    return all(observed["feature"].get(key) == expected["feature"].get(key) for key in capture_keys) and all(
        observed["bridge"].get(key) == expected["bridge"].get(key) for key in bridge_keys
    )


def evaluate_split(
    model,
    tokenizer,
    bridge: WorkspaceBridge,
    shell_norm: torch.Tensor,
    artifact: dict[str, object],
    schedule: dict[str, object],
    split: str,
    call_log: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    call_start = len(call_log)
    bridge_start = getattr(call_log, "bridge_forwards", 0)
    records, queries = _record_lookup(artifact)
    donors = artifact["moth_donors"]
    arm_orders = {item["query_id"]: item["arms"] for item in schedule["arm_orders"][split]}
    canonical_slots = {}
    moth_slots = {}
    noise_slots = {}
    setup_by_id = {}
    for record in records.values():
        current, current_evidence = _timed_workspace(
            model,
            tokenizer,
            bridge,
            shell_norm,
            parent_messages(record["parent_evidence"]),
            call_log,
            arm=f"{split}_MCUR_CANONICAL_SETUP",
        )
        canonical_slots[record["evidence_id"]] = current.detach().contiguous().clone()
        setup_by_id[record["evidence_id"]] = current_evidence
    canonical_hashes = {evidence_id: tensor_sha256(slots) for evidence_id, slots in canonical_slots.items()}
    setup = []
    for record in records.values():
        donor = records[donors[record["evidence_id"]]]
        current = canonical_slots[record["evidence_id"]]
        other = canonical_slots[donor["evidence_id"]]
        noise = norm_matched_noise(current[0].detach().cpu(), split=split, evidence_id=record["evidence_id"])
        moth_slots[record["evidence_id"]] = other
        noise_slots[record["evidence_id"]] = noise.tensor.unsqueeze(0).detach().contiguous().to("cuda:0").contiguous()
        setup.append(
            {
                "evidence_id": record["evidence_id"],
                "moth_donor_evidence_id": donor["evidence_id"],
                "mcur": setup_by_id[record["evidence_id"]],
                "moth_canonical_source_sha256": tensor_sha256(other),
                "noise": noise.evidence,
            }
        )

    rows = []
    reuse_hashes = {record_id: {"MCUR": [], "MOTH": [], "NOISE": []} for record_id in records}
    for expected in build_arm_orders(artifact, split):
        query_id = expected["query_id"]
        record = queries[query_id]["record"]
        query = queries[query_id]["query"]
        if arm_orders[query_id] != expected["arms"]:
            raise ExperimentIncomplete("A1-NC0 materialized arm order changed")
        results = {}
        arm_costs = {}
        arm_decodes = {}
        for arm in expected["arms"]:
            timer = cuda_timer_start()
            arm_wall_started = time.perf_counter()
            feature_bridge = None
            if arm == "MCUR":
                slots, feature_bridge = _timed_workspace(
                    model,
                    tokenizer,
                    bridge,
                    shell_norm,
                    parent_messages(record["parent_evidence"]),
                    call_log,
                    arm=f"{split}_MCUR",
                )
                expected_slots = canonical_slots[record["evidence_id"]]
                expected_witness = setup_by_id[record["evidence_id"]]
                if not torch.equal(slots, expected_slots) or not _workspace_evidence_matches(
                    feature_bridge, expected_witness
                ):
                    raise MechanismRejected("A1-NC0 MCUR reuse changed", reason="workspace_reuse_rejected")
                slots = expected_slots.clone()
            elif arm == "MOTH":
                donor = records[donors[record["evidence_id"]]]
                slots, feature_bridge = _timed_workspace(
                    model,
                    tokenizer,
                    bridge,
                    shell_norm,
                    parent_messages(donor["parent_evidence"]),
                    call_log,
                    arm=f"{split}_MOTH",
                )
                expected_slots = moth_slots[record["evidence_id"]]
                expected_witness = setup_by_id[donor["evidence_id"]]
                if not torch.equal(slots, expected_slots) or not _workspace_evidence_matches(
                    feature_bridge, expected_witness
                ):
                    raise MechanismRejected("A1-NC0 MOTH reuse changed", reason="workspace_reuse_rejected")
                slots = expected_slots.clone()
            elif arm == "MSELF":
                slots, feature_bridge = _timed_workspace(
                    model,
                    tokenizer,
                    bridge,
                    shell_norm,
                    self_messages(query["child_query"]),
                    call_log,
                    arm=f"{split}_MSELF",
                )
                slots = slots.clone()
            elif arm == "ZERO":
                slots = torch.zeros((1, 8, 2048), dtype=torch.bfloat16, device="cuda:0")
            elif arm == "NOISE":
                slots = noise_slots[record["evidence_id"]].clone()
            else:
                slots = None
            if arm in reuse_hashes[record["evidence_id"]]:
                reuse_hashes[record["evidence_id"]][arm].append(tensor_sha256(slots))
            with torch.inference_mode():
                generated, text, nll, answer_token_count, decode, decode_cuda_seconds = greedy_full_recompute(
                    model,
                    tokenizer,
                    query["child_query"],
                    query["answer"],
                    slots,
                    call_log,
                    arm=f"{split}_{arm}",
                )
            elapsed = cuda_timer_finish(timer)
            results[arm] = {
                "exact_match": text == query["answer"],
                "generated_text": text,
                "expected_answer_sha256": hashlib.sha256(query["answer"].encode()).hexdigest(),
                "answer_token_nll": nll,
                "answer_token_count": answer_token_count,
                "generated_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "generated_token_ids": generated,
                "fixed_decode_steps": len(decode),
            }
            arm_costs[arm] = {
                "cuda_event_gpu_seconds": elapsed,
                "wall_seconds": time.perf_counter() - arm_wall_started,
                "feature_forwards": 1 if feature_bridge is not None else 0,
                "bridge_forwards": 1 if feature_bridge is not None else 0,
                "receiver_forwards": 12,
                "receiver_cuda_event_seconds": decode_cuda_seconds,
                "feature_bridge": feature_bridge,
            }
            arm_decodes[arm] = decode
        rows.append(
            {
                "query_id": query_id,
                "evidence_id": record["evidence_id"],
                "family": record["family"],
                "arm_order": expected["arms"],
                "arms": results,
                "costs": arm_costs,
                "decode_evidence": arm_decodes,
            }
        )
    for values in reuse_hashes.values():
        if any(len(hashes) != 3 or len(set(hashes)) != 1 for hashes in values.values()):
            raise MechanismRejected("A1-NC0 three-query workspace reuse changed", reason="workspace_reuse_rejected")
    if any(
        not slots.is_contiguous() or slots.requires_grad or tensor_sha256(slots) != canonical_hashes[evidence_id]
        for evidence_id, slots in canonical_slots.items()
    ):
        raise MechanismRejected(
            "A1-NC0 canonical evaluation workspace mutated",
            reason="workspace_reuse_rejected",
        )
    expected_queries = [
        (query["query_id"], record["family"]) for record in artifact["bank"]["records"] for query in record["queries"]
    ]
    summary = summarize_arm_results(rows, expected_queries=expected_queries)
    mcur_cost = math.fsum(row["costs"]["MCUR"]["cuda_event_gpu_seconds"] for row in rows)
    mself_cost = math.fsum(row["costs"]["MSELF"]["cuda_event_gpu_seconds"] for row in rows)
    denominator = mcur_cost + mself_cost
    if (
        not math.isfinite(mcur_cost)
        or not math.isfinite(mself_cost)
        or mcur_cost < 0
        or mself_cost < 0
        or not math.isfinite(denominator)
        or denominator <= 0
    ):
        raise RuntimeError("A1-NC0 compute-match CUDA-event totals are invalid")
    summary["mself_compute_match"] = {
        "mcur_feature_forwards": 48,
        "mself_feature_forwards": 48,
        "mcur_bridge_forwards": 48,
        "mself_bridge_forwards": 48,
        "mcur_feature_input_tokens": 48 * 256,
        "mself_feature_input_tokens": 48 * 256,
        "receiver_forwards_each": 48 * 12,
        "mcur_cuda_event_gpu_seconds": mcur_cost,
        "mself_cuda_event_gpu_seconds": mself_cost,
        "relative_gpu_seconds_difference": 2 * abs(mcur_cost - mself_cost) / denominator,
    }
    if summary["mself_compute_match"]["relative_gpu_seconds_difference"] > 0.10:
        raise RuntimeError("A1-NC0 MCUR/MSELF aggregate GPU-seconds exceed frozen 0.10 symmetry tolerance")
    summary["workspace_reuse"] = reuse_hashes
    split_calls = call_log[call_start:]
    call_histogram: dict[str, int] = {}
    for entry in split_calls:
        call_histogram[entry["arm"]] = call_histogram.get(entry["arm"], 0) + 1
    expected_histogram = {
        f"{split}_MCUR_CANONICAL_SETUP_FEATURE": 16,
        f"{split}_MCUR_FEATURE": 48,
        f"{split}_MOTH_FEATURE": 48,
        f"{split}_MSELF_FEATURE": 48,
        **{f"{split}_{arm}_DECODE": 48 * 12 for arm in ("M0", "MOTH", "MSELF", "MCUR", "ZERO", "NOISE")},
    }
    if (
        len(split_calls) != 3616
        or getattr(call_log, "bridge_forwards", 0) - bridge_start != 160
        or call_histogram != expected_histogram
    ):
        raise ExperimentIncomplete("A1-NC0 split capture/bridge/receiver call schedule changed")
    summary["operation_counts"] = {
        "captures": 160,
        "bridges": 160,
        "receiver_forwards": 3456,
        "canonical_captures_and_bridges": 16,
        "mcur_query_captures_and_bridges": 48,
        "moth_query_captures_and_bridges": 48,
        "mself_query_captures_and_bridges": 48,
        "e33_call_histogram": call_histogram,
    }
    if hasattr(call_log, "ledger"):
        call_log.ledger.checkpoint(f"{split}_split_audit_complete")
    return rows, {"setup": setup, "summary": summary}


def validate_rendering_preflight(
    tokenizer, artifacts: dict[str, dict[str, object]], expected_eos: int
) -> dict[str, object]:
    extraction_counts: collections.Counter[str] = collections.Counter()
    if tokenizer.eos_token_id != expected_eos or tokenizer.pad_token_id != expected_eos:
        raise ValueError("A1-NC0 tokenizer EOS identity changed")
    terminal = tokenizer("<|im_end|>\n", add_special_tokens=False, return_tensors="pt").input_ids.flatten().tolist()
    if terminal != TERMINAL_TOKEN_IDS:
        raise ValueError("A1-NC0 terminal marker token IDs changed")
    continuation = tokenizer(FIXED_CONTINUATION_TEXT, add_special_tokens=False).input_ids[:4]
    if continuation != FIXED_CONTINUATION_IDS or set(LENGTH_CONTROL_IDS) & set(tokenizer.all_special_ids):
        raise ExperimentIncomplete("A1-NC0 preprobe continuation or ordinary length-control IDs changed")
    hashes = {}
    label_alignment = {}
    maximum_feature_tokens = 0
    materialized_queries = 0
    for split, artifact in artifacts.items():
        for record in artifact["bank"]["records"]:
            validate_parent_fixture(record["parent_evidence"])
            parent_text = tokenizer.apply_chat_template(
                parent_messages(record["parent_evidence"]),
                tools=PARENT_TOOLS,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            parent_ids = preflight_template_input_ids(
                tokenizer.apply_chat_template(
                    parent_messages(record["parent_evidence"]),
                    tools=PARENT_TOOLS,
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=False,
                ),
                label="parent",
                extraction_counts=extraction_counts,
            )
            maximum_feature_tokens = max(maximum_feature_tokens, len(parent_ids))
            if len(parent_ids) > 256:
                raise ExperimentIncomplete("A1-NC0 parent fixture exceeds 256 tokens; truncation forbidden")
            for query in record["queries"]:
                messages = child_messages(query["child_query"])
                full_messages = [*messages, {"role": "assistant", "reasoning_content": "", "content": query["answer"]}]
                plain_text = tokenizer.apply_chat_template(
                    messages, tools=None, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
                opening_text = tokenizer.apply_chat_template(
                    messages, tools=None, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                full_text = tokenizer.apply_chat_template(
                    full_messages, tools=None, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
                plain_ids = preflight_template_input_ids(
                    tokenizer.apply_chat_template(
                        messages, tools=None, tokenize=True, add_generation_prompt=False, enable_thinking=False
                    ),
                    label="child_plain",
                    extraction_counts=extraction_counts,
                )
                opening_ids = preflight_template_input_ids(
                    tokenizer.apply_chat_template(
                        messages, tools=None, tokenize=True, add_generation_prompt=True, enable_thinking=False
                    ),
                    label="child_opening",
                    extraction_counts=extraction_counts,
                )
                full_ids = preflight_template_input_ids(
                    tokenizer.apply_chat_template(
                        full_messages, tools=None, tokenize=True, add_generation_prompt=False, enable_thinking=False
                    ),
                    label="child_full",
                    extraction_counts=extraction_counts,
                )
                answer_ids = tokenizer(query["answer"], add_special_tokens=False).input_ids
                self_ids = preflight_template_input_ids(
                    tokenizer.apply_chat_template(
                        self_messages(query["child_query"]),
                        tools=PARENT_TOOLS,
                        tokenize=True,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    ),
                    label="mself_parent",
                    extraction_counts=extraction_counts,
                )
                validate_parent_fixture(query["child_query"])
                self_text = tokenizer.apply_chat_template(
                    self_messages(query["child_query"]),
                    tools=PARENT_TOOLS,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                maximum_feature_tokens = max(maximum_feature_tokens, len(self_ids))
                if (
                    len(self_ids) > 256
                    or opening_text != plain_text + CHILD_OPENING_SUFFIX
                    or full_text != opening_text + query["answer"] + "<|im_end|>\n"
                    or opening_ids[: len(plain_ids)] != plain_ids
                    or full_ids != opening_ids + answer_ids + TERMINAL_TOKEN_IDS
                    or not 1 <= len(answer_ids) <= 12
                    or record["parent_evidence"] in plain_text
                    or query["child_query"] in parent_text
                ):
                    raise ExperimentIncomplete("A1-NC0 materialized render boundary changed")
                identity = f"{split}:{query['query_id']}"
                hashes[identity] = canonical_json_hash(
                    {
                        "parent_text": parent_text,
                        "plain_text": plain_text,
                        "opening_text": opening_text,
                        "full_text": full_text,
                        "mself_parent_text": self_text,
                        "parent_ids": parent_ids,
                        "plain_ids": plain_ids,
                        "opening_ids": opening_ids,
                        "full_ids": full_ids,
                        "mself_parent_ids": self_ids,
                    }
                )
                active = list(range(len(opening_ids) + 8, len(full_ids) - 2 + 8))
                label_alignment[identity] = {
                    "active_label_positions": active,
                    "active_logit_positions": [position - 1 for position in active],
                    "raw_answer_token_ids": answer_ids,
                    "terminal_token_ids": TERMINAL_TOKEN_IDS,
                    "all_other_labels_masked": True,
                }
                materialized_queries += 1
    if materialized_queries != 288:
        raise ExperimentIncomplete("A1-NC0 materialized query count changed")
    expected_extractions = {
        "parent": 96,
        "child_plain": 288,
        "child_opening": 288,
        "child_full": 288,
        "mself_parent": 288,
    }
    if dict(extraction_counts) != expected_extractions:
        raise ExperimentIncomplete("A1-NC0 BatchEncoding extraction schedule changed")
    return {
        "enable_thinking": False,
        "tools_none_for_child": True,
        "parent_fixture_messages": 4,
        "child_base_messages": 2,
        "terminal_token_ids": TERMINAL_TOKEN_IDS,
        "fixed_continuation_token_ids": FIXED_CONTINUATION_IDS,
        "length_control_token_ids": LENGTH_CONTROL_IDS,
        "length_control_tokens_non_special": True,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "maximum_unpadded_feature_tokens": maximum_feature_tokens,
        "feature_sequences_truncated": 0,
        "materialized_queries": materialized_queries,
        "tokenized_template_container": "transformers.tokenization_utils_base.BatchEncoding",
        "preflight_input_ids_extracted_from_batch_encoding": True,
        "batch_encoding_extraction_counts": expected_extractions,
        "answer_key_interpolation_scope": "teacher_target_and_scoring_only",
        "answer_key_not_interpolated_into_parent_or_child_opening": True,
        "render_hashes_sha256": canonical_json_hash(hashes),
        "label_alignment": label_alignment,
        "label_alignment_sha256": canonical_json_hash(label_alignment),
    }


def verify_execution_tree(args: argparse.Namespace, plan: dict[str, object]) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=args.repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if head != args.execution_commit or len(head) != 40 or status:
        raise ValueError("A1-NC0 execution tree must be exact and clean")
    parent = subprocess.run(
        ["git", "rev-parse", f"{head}^"], cwd=args.repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    mechanism_parent = subprocess.run(
        ["git", "rev-parse", f"{parent}^"], cwd=args.repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    if parent != plan["evidence_commit"] or mechanism_parent != plan["mechanism_code_commit"]:
        raise ValueError("A1-NC0-R1 freeze/evidence/mechanism lineage changed")
    for relative, expected in plan["asset_sha256"].items():
        path = args.repo / relative
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"A1-NC0 execution asset changed: {relative}")


def host_ram_bytes() -> int:
    pages = os.sysconf("SC_PHYS_PAGES")
    size = os.sysconf("SC_PAGE_SIZE")
    return pages * size


class MemoryLedger:
    def __init__(self, paths: dict[str, list[str]], cap_bytes: int) -> None:
        self.paths = paths
        self.possible_paths = set(paths)
        self.selected_path: str | None = None
        self.cap_bytes = cap_bytes
        self.rows = []
        self.guard: CacheGuard | None = None

    def checkpoint(self, label: str) -> None:
        index = len(self.rows)
        compatible = {
            path for path in self.possible_paths if index < len(self.paths[path]) and self.paths[path][index] == label
        }
        if not compatible:
            raise ExperimentIncomplete("A1-NC0 memory-ledger order changed")
        self.possible_paths = compatible
        if self.guard is not None:
            self.guard.verify_closure()
        synchronize()
        row = {
            "label": label,
            "allocated_bytes": torch.cuda.memory_allocated(0),
            "reserved_bytes": torch.cuda.memory_reserved(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        }
        self.rows.append(row)
        if any(
            row[key] > self.cap_bytes
            for key in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes")
        ):
            raise torch.cuda.OutOfMemoryError("A1-NC0 exceeded frozen 40 GiB CUDA memory cap")

    def choose_path(self, path: str) -> None:
        if path not in self.possible_paths:
            raise ExperimentIncomplete("A1-NC0 memory-ledger branch changed")
        self.possible_paths = {path}
        self.selected_path = path

    def validate_complete(self) -> None:
        if len(self.possible_paths) != 1:
            raise ExperimentIncomplete("A1-NC0 memory-ledger path stayed ambiguous")
        path = next(iter(self.possible_paths))
        self.selected_path = path
        if [row["label"] for row in self.rows] != self.paths[path]:
            raise ExperimentIncomplete("A1-NC0 memory ledger is incomplete")


class TrackedCallLog(list[dict[str, object]]):
    def __init__(self, ledger: MemoryLedger) -> None:
        super().__init__()
        self.ledger = ledger
        self.bridge_forwards = 0
        self.bridge_log: list[str] = []

    def after_forward(self, arm: str) -> None:
        self.ledger.checkpoint(f"e33_forward_{len(self):04d}_{arm}")

    def after_bridge(self, arm: str) -> None:
        self.bridge_forwards += 1
        self.bridge_log.append(arm)
        self.ledger.checkpoint(f"bridge_forward_{self.bridge_forwards:04d}_{arm}")


def validate_a0nc_binding(repo: Path, plan: dict[str, object]) -> dict[str, object]:
    experiment = repo / "experiments/qwen35-2b-latent-workspace-v1"
    prior_plan_path = experiment / "a0-nocache-plan-v1.json"
    receipt_path = experiment / "a0nc-success-receipt.json"
    if any(path.is_symlink() or not path.is_file() for path in (prior_plan_path, receipt_path)):
        raise ValueError("A1-NC0 A0NC binding assets absent or symlinked")
    prior_plan = json.loads(prior_plan_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    validate_a0nc_receipt(receipt, plan=prior_plan)
    evidence = plan["a0nc_success_evidence"]
    observed = {
        "status": receipt["status"],
        "claim": receipt["claim"],
        "receipt_file_sha256": file_sha256(receipt_path),
        "receipt_internal_sha256": receipt["receipt_sha256"],
        "plan_sha256": receipt["plan_sha256"],
        "bank_sha256": receipt["bank_sha256"],
        "mechanism_code_commit": receipt["mechanism_code_commit"],
        "execution_commit": receipt["execution_commit"],
        "complete_distinct_probes": receipt["complete_distinct_probes"],
        "protected_hashes_before": receipt["protected_hashes_before"],
        "protected_hashes_after": receipt["protected_hashes_after"],
        "model_update_attempted": receipt["model_update_attempted"],
        "optimizer_created": receipt["optimizer_created"],
        "checkpoint_created": receipt["checkpoint_created"],
        "prior_cache_rejection": receipt["prior_cache_rejection"],
    }
    if observed != evidence:
        raise ValueError("A1-NC0 A0NC success binding changed")
    return observed


def set_stage(stage: dict[str, object], name: str) -> None:
    stage["name"] = name
    stage.setdefault("breadcrumbs", []).append(name)


def run(args, plan, artifacts, schedule, disjointness, stage, writer: ArtifactWriter):
    if not args.owner_approved:
        raise ValueError("A1-NC0 requires root approval after immutable review")
    verify_execution_tree(args, plan)
    if file_sha256(args.plan) != args.authorized_plan_sha256:
        raise ValueError("A1-NC0 plan file is not the root-authorized immutable plan")
    expected_memory_paths = build_memory_ledger_paths(schedule)
    frozen_memory_paths = {name: value["labels"] for name, value in schedule["memory_ledger_paths"].items()}
    if frozen_memory_paths != expected_memory_paths:
        raise ValueError("A1-NC0 runner memory labels differ from frozen schedule")
    a0nc_evidence = validate_a0nc_binding(args.repo, plan)
    stage["a0nc_success_evidence_observed"] = a0nc_evidence
    r1_evidence = validate_a1nc0_r1_evidence(args.repo, plan)
    stage["a1nc0_r1_evidence_observed"] = r1_evidence
    stage["frozen_asset_paths"] = {
        relative: str((args.repo / relative).resolve(strict=True)) for relative in plan["asset_sha256"]
    }
    source_guard = validate_no_generation_source(Path(__file__).resolve(strict=True))
    if os.environ.get("UV_PROJECT_ENVIRONMENT") != str(SHARED_ENVIRONMENT) or os.environ.get("PYTHONPATH") != str(
        args.repo / "src"
    ):
        raise ValueError("A1-NC0 shared environment changed")
    versions = {
        "python": platform.python_version(),
        "transformers": importlib.metadata.version("transformers"),
        "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
        "torch_distribution": importlib.metadata.version("torch"),
        "torch_runtime": str(torch.__version__),
    }
    expected_runtime = plan["runtime"]
    sources = source_hashes()
    if (
        versions["python"] != expected_runtime["python"]
        or versions["transformers"] != expected_runtime["transformers"]
        or versions["flash_linear_attention"] != expected_runtime["flash_linear_attention"]
        or versions["torch_distribution"] != expected_runtime["torch_distribution"]
        or versions["torch_runtime"] != expected_runtime["torch_runtime"]
        or {name: item["sha256"] for name, item in sources.items()} != expected_runtime["transformers_source_sha256"]
        or args.coordinator.resolve() != Path(plan["remote_paths"]["coordinator_e33"])
        or args.worker.resolve() != Path(plan["remote_paths"]["worker_h176"])
    ):
        raise ValueError("A1-NC0 runtime source or protected path changed")
    weights = {
        "coordinator_e33": model_weight(args.coordinator),
        "worker_h176": model_weight(args.worker),
    }
    before = {name: file_sha256(path) for name, path in weights.items()}
    metadata_before = {
        "coordinator_e33": metadata_hashes(args.coordinator),
        "worker_h176": metadata_hashes(args.worker),
    }
    stage["protected_hashes_before"] = before
    stage["checkpoint_metadata_before"] = metadata_before
    if before != plan["protected_checkpoints"] or any(
        value != expected_runtime["checkpoint_metadata_sha256"] for value in metadata_before.values()
    ):
        raise ValueError("A1-NC0 protected preflight changed")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != plan["resource_bounds"]["gpu_model"]
    ):
        raise RuntimeError("A1-NC0 GPU identity changed")
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / 2**30
    cap_bytes = plan["resource_bounds"]["allocator_cap_gib"] * 2**30
    if total_gib < plan["resource_bounds"]["minimum_gpu_memory_gib"] or cap_bytes >= properties.total_memory:
        raise RuntimeError("A1-NC0 GPU capacity is below frozen requirement")
    ram = host_ram_bytes()
    free_disk = shutil.disk_usage(plan["resource_bounds"]["output_root"]).free
    if (
        ram < plan["resource_bounds"]["minimum_host_ram_gib"] * 2**30
        or free_disk < plan["resource_bounds"]["minimum_free_disk_gib"] * 2**30
    ):
        raise RuntimeError("A1-NC0 host resources are below freeze")
    torch.cuda.set_per_process_memory_fraction(cap_bytes / properties.total_memory, 0)
    torch.cuda.reset_peak_memory_stats(0)
    tokenizer = AutoTokenizer.from_pretrained(args.coordinator, local_files_only=True)
    render_preflight = validate_rendering_preflight(tokenizer, artifacts, plan["runtime"]["tokenizer_eos_token_id"])
    set_stage(stage, "render_preflight_complete_no_model")
    started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.coordinator,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda:0")
    stage["_model"] = model
    set_stage(stage, "e33_loaded_no_bridge")
    model.eval()
    if (
        model.__class__.__name__ != expected_runtime["model_class"]
        or model.config.text_config.hidden_size != 2048
        or model.config.text_config.vocab_size != expected_runtime["vocab_size"]
        or str(next(model.parameters()).dtype) != "torch.bfloat16"
        or str(next(model.parameters()).device) != "cuda:0"
    ):
        raise RuntimeError("A1-NC0 loaded model runtime changed")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is None:
        raise RuntimeError("A1-NC0 model generation configuration is absent")
    model.generation_config.use_cache = False
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise MechanismRejected("A1-NC0 e33 freeze failed", reason="protected_mutation_rejected")
    ledger = MemoryLedger(frozen_memory_paths, cap_bytes)
    stage["_ledger"] = ledger
    ledger.checkpoint("model_loaded_frozen")
    call_log: list[dict[str, object]] = TrackedCallLog(ledger)
    e33_parameter_sha_before = module_state_tree_sha256(model)
    stage["e33_parameter_tree_sha256_before"] = e33_parameter_sha_before
    embedding_weights = model.get_input_embeddings().weight.detach()
    shell_norm = torch.linalg.vector_norm(embedding_weights.float(), dim=-1).mean()
    if not torch.isfinite(shell_norm) or shell_norm <= 0:
        raise MechanismRejected("A1-NC0 embedding shell norm invalid", reason="bridge_numeric_contract_rejected")
    guard = CacheGuard()
    stage["_cache_guard"] = guard
    bridge = None
    optimizer = None
    held_rows = None
    held_evaluation = None
    with guard:
        ledger.guard = guard
        feature_cache, train_capture_evidence = build_train_feature_cache(
            model, tokenizer, artifacts["train"], call_log
        )
        set_stage(stage, "train_feature_cache_complete")
        ledger.checkpoint("train_feature_cache_host_complete")
        torch.manual_seed(plan["seeds"]["bridge_init"])
        torch.cuda.manual_seed_all(plan["seeds"]["bridge_init"])
        bridge = WorkspaceBridge(WorkspaceBridgeConfig()).to(device="cuda:0", dtype=torch.float32)
        if (
            bridge.trainable_parameter_count() != plan["bridge"]["trainable_parameter_count"]
            or sorted(bridge.state_dict()) != plan["bridge"]["candidate_tensor_names"]
        ):
            raise MechanismRejected("A1-NC0 bridge parameter count changed", reason="bridge_structure_rejected")
        bridge_initial_sha = bridge_parameter_sha256(bridge)
        ledger.checkpoint("bridge_initialized")
        pretraining_repeat = run_a0nc_repeat(
            model, tokenizer, bridge, shell_norm, artifacts["train"], schedule, call_log
        )
        set_stage(stage, "pretraining_repeat_complete")
        ledger.checkpoint("a0nc_repeat_gradient_complete")
        set_stage(stage, "training_started")
        updates, optimizer, training_invariants = train_bridge(
            model,
            tokenizer,
            bridge,
            shell_norm,
            artifacts["train"],
            schedule,
            feature_cache,
            call_log,
            ledger.checkpoint,
        )
        set_stage(stage, "training_complete")
        bridge_final_sha = bridge_parameter_sha256(bridge)
        if bridge_final_sha == bridge_initial_sha:
            raise MechanismRejected("A1-NC0 optimizer made no bridge update", reason="bridge_update_rejected")
        optimizer_created = True
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        optimizer = None
        gc.collect()
        torch.cuda.empty_cache()
        ledger.checkpoint("optimizer_destroyed_before_evaluation")
        validation_rows, validation_evaluation = evaluate_split(
            model, tokenizer, bridge, shell_norm, artifacts["validation"], schedule, "validation", call_log
        )
        set_stage(stage, "validation_complete")
        validation_passed = validation_gate_passes(validation_evaluation["summary"])
        ledger.checkpoint("validation_evaluation_complete")
        if validation_passed:
            ledger.choose_path("full_evaluation")
            held_rows, held_evaluation = evaluate_split(
                model, tokenizer, bridge, shell_norm, artifacts["held_out"], schedule, "held_out", call_log
            )
            set_stage(stage, "held_out_complete")
            ledger.checkpoint("held_out_evaluation_complete")
        else:
            ledger.choose_path("validation_stop")
            ledger.checkpoint("held_out_skipped_no_model_exposure")
        guard.verify_closure()
    compute_seconds = time.perf_counter() - started
    if compute_seconds > plan["resource_bounds"]["compute_seconds"]:
        raise TimeoutError("A1-NC0 compute phase exceeded its frozen budget")
    signal.alarm(240)
    set_stage(stage, "compute_complete_audit_window")
    audit_started = time.perf_counter()
    ledger.guard = None
    cache_guard = guard.evidence()
    ledger.checkpoint("cache_guard_audit_complete")
    e33_parameter_sha_after = module_state_tree_sha256(model)
    stage["e33_parameter_tree_sha256_after"] = e33_parameter_sha_after
    if e33_parameter_sha_after != e33_parameter_sha_before:
        raise MechanismRejected("A1-NC0 in-memory e33 tree changed", reason="protected_mutation_rejected")
    ledger.checkpoint("e33_in_memory_post_hash_complete")
    after = {name: file_sha256(path) for name, path in weights.items()}
    metadata_after = {
        "coordinator_e33": metadata_hashes(args.coordinator),
        "worker_h176": metadata_hashes(args.worker),
    }
    stage["protected_hashes_after"] = after
    stage["checkpoint_metadata_after"] = metadata_after
    if (
        after != before
        or metadata_after != metadata_before
        or any(parameter.grad is not None for parameter in model.parameters())
    ):
        raise MechanismRejected("A1-NC0 protected model changed", reason="protected_mutation_rejected")
    ledger.checkpoint("protected_disk_postflight_complete")
    nominated = held_evaluation is not None and nomination_gate_passes(held_evaluation["summary"])
    status = (
        "a1_nc0_nominated"
        if nominated
        else "valid_not_nominated"
        if held_evaluation is not None
        else "valid_not_nominated_validation"
    )
    expected_calls = 4526 + (3616 if held_evaluation is not None else 0)
    if (
        len(call_log) != expected_calls
        or not all(item["use_cache"] is False for item in call_log)
        or not all(item["past_key_values_input_is_none"] for item in call_log)
        or not all(item["past_key_values_output_is_none"] for item in call_log)
        or not all(item["rope_deltas_reset_before_call"] for item in call_log)
        or model.config.use_cache is not False
        or (getattr(model, "generation_config", None) is not None and model.generation_config.use_cache is not False)
    ):
        raise CacheAllocationDetected("A1-NC0 no-cache call ledger changed")
    set_stage(stage, "audits_complete_before_candidate")
    ledger.checkpoint("candidate_write_preflight")
    candidate = writer.write_candidate(bridge.state_dict(), plan["resource_bounds"]["maximum_candidate_bytes"])
    stage["candidate_inventory"] = writer.inventory()
    set_stage(stage, "candidate_written_pending_terminal")
    ledger.checkpoint("candidate_write_complete")
    ledger.checkpoint("preterminal_receipt_audit_complete")
    ledger.validate_complete()
    audit_seconds = time.perf_counter() - audit_started
    if audit_seconds > plan["resource_bounds"]["audit_seconds"]:
        raise TimeoutError("A1-NC0 audit phase exceeded its frozen budget")
    receipt = {
        "schema_version": "prime-rl/latent-a1-nc0-r1-receipt/v1",
        "status": status,
        "plan_sha256": plan["plan_sha256"],
        "mechanism_code_commit": plan["mechanism_code_commit"],
        "evidence_commit": plan["evidence_commit"],
        "execution_commit": args.execution_commit,
        "execution_commit_is_exact_child_of_evidence": True,
        "asset_sha256": plan["asset_sha256"],
        "a0nc_success_evidence": a0nc_evidence,
        "a1nc0_r1_evidence": r1_evidence,
        "bank_disjointness": {
            "file_sha256": plan["bank_disjointness"]["file_sha256"],
            "report_sha256": disjointness["report_sha256"],
            "all_pairwise_intersections_zero": True,
        },
        "static_no_generation_guard": source_guard,
        "versions": versions,
        "runtime_sources": sources,
        "render_preflight": render_preflight,
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "checkpoint_metadata_before": metadata_before,
        "checkpoint_metadata_after": metadata_after,
        "model_runtime": {
            "class": model.__class__.__name__,
            "hidden_size": model.config.text_config.hidden_size,
            "vocab_size": model.config.text_config.vocab_size,
            "dtype": str(next(model.parameters()).dtype),
            "device": str(next(model.parameters()).device),
        },
        "bridge": {
            "config": WorkspaceBridgeConfig().__dict__
            if hasattr(WorkspaceBridgeConfig(), "__dict__")
            else {name: getattr(WorkspaceBridgeConfig(), name) for name in WorkspaceBridgeConfig.__dataclass_fields__},
            "trainable_parameter_count": bridge.trainable_parameter_count(),
            "initialization_seed": plan["seeds"]["bridge_init"],
            "torch_manual_seed_calls": 1,
            "torch_cuda_manual_seed_all_calls": 1,
            "parameter_sha256_initial": bridge_initial_sha,
            "parameter_sha256_final": bridge_final_sha,
            "parameter_tree_hash_schema": "sorted_state_dict_name_dtype_shape_tensor_sha256_lines/v1",
            "optimizer_created": optimizer_created,
            "optimizer_updates": len(updates),
            "optimizer_destroyed_before_evaluation": optimizer is None,
            "optimizer_state_persisted": False,
            "base_model_checkpoint_created": False,
            "candidate": {
                **candidate,
                "contains_bridge_and_receiver_gate_only": True,
                "valid_only_with_exact_terminal_receipt": True,
                "promotion_authorized": False,
            },
        },
        "train_feature_cache": train_capture_evidence,
        "pretraining_a0nc_repeat": pretraining_repeat,
        "training_updates": updates,
        "training_invariants": training_invariants,
        "validation": {"rows": validation_rows, **validation_evaluation, "proceed_gate_passed": validation_passed},
        "held_out": None
        if held_evaluation is None
        else {"rows": held_rows, **held_evaluation, "nomination_gate_passed": nominated},
        "no_cache_call_contract": {
            "total_e33_forwards": len(call_log),
            "expected_e33_forwards": expected_calls,
            "use_cache_false_every_call": True,
            "past_key_values_input_none_every_call": True,
            "past_key_values_output_none_every_call": True,
            "generate_used": False,
            "prepare_inputs_for_generation_used": False,
            "rope_deltas_reset_before_every_call": True,
            "model_config_use_cache": model.config.use_cache,
            "generation_config_use_cache": model.generation_config.use_cache,
        },
        "cache_guard": cache_guard,
        "memory_ledger": ledger.rows,
        "memory_ledger_path": ledger.selected_path,
        "memory_ledger_labels_sha256": canonical_json_hash([row["label"] for row in ledger.rows]),
        "resources": {
            "wall_seconds": time.perf_counter() - started,
            "compute_seconds": compute_seconds,
            "audit_seconds_before_receipt_materialization": audit_seconds,
            "gpu_name": torch.cuda.get_device_name(0),
            "total_gpu_memory_bytes": properties.total_memory,
            "allocator_cap_bytes": cap_bytes,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "host_ram_bytes": ram,
            "free_disk_bytes_before": free_disk,
            "visible_cuda_device_count": torch.cuda.device_count(),
            "launcher_verified_two_a6000_idle_before_gpu0_exposure": True,
            "physical_gpu1_unused": True,
            "network_used": False,
        },
        "claim": "A1-NC0 nomination-only no-cache bridge learnability",
        "bound_a0nc_dependency_valid_for_B_only": True,
        "e33_parameter_tree_sha256_before": e33_parameter_sha_before,
        "e33_parameter_tree_sha256_after": e33_parameter_sha_after,
        "e33_tensor_tree_hash_schema": "sorted_state_dict_name_dtype_shape_tensor_sha256_lines/v1",
        "e33_parameters_require_grad_false": True,
        "e33_gradients_absent": True,
        "worker_h176_loaded": False,
        "live_trajectory_count": 0,
        "a_plus_b_combined": False,
        "resume_used": False,
        "candidate_valid": True,
        "candidate_valid_only_with_this_exact_terminal_receipt": True,
        "a1_admission": False,
        "live_harness_authorized": False,
        "a2_authorized": False,
        "model_promotion_authorized": False,
        "interpretation_boundary": plan["interpretation_boundary"],
        "receipt_sha256": "",
    }
    validate_finite_metrics(receipt)
    receipt["receipt_sha256"] = canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    validate_receipt(
        receipt,
        plan=plan,
        schedule=schedule,
        artifacts=artifacts,
        tokenizer=tokenizer,
        candidate_path=args.output_dir / "bridge-candidate.safetensors",
    )
    signal.alarm(60)
    set_stage(stage, "terminal_write_window")
    return receipt


def failure_record(
    args, error: BaseException, stage: dict[str, object], plan: dict[str, object] | None
) -> dict[str, object]:
    status, category = classify_failure(error)
    protected = {}
    metadata = {}
    audit_errors = []
    for name, path in (("coordinator_e33", args.coordinator), ("worker_h176", args.worker)):
        try:
            protected[name] = file_sha256(model_weight(path))
        except BaseException as hash_error:
            protected[name] = f"unavailable:{type(hash_error).__name__}:{hash_error}"
            audit_errors.append(f"protected_weight:{name}:{type(hash_error).__name__}:{hash_error}")
        try:
            metadata[name] = metadata_hashes(path)
        except BaseException as metadata_error:
            metadata[name] = {"unavailable": f"{type(metadata_error).__name__}:{metadata_error}"}
            audit_errors.append(f"protected_metadata:{name}:{type(metadata_error).__name__}:{metadata_error}")
    asset_hashes = {}
    if plan is not None:
        for relative in plan.get("asset_sha256", {}):
            try:
                path = args.repo / relative
                asset_hashes[relative] = file_sha256(path)
            except BaseException as asset_error:
                asset_hashes[relative] = f"unavailable:{type(asset_error).__name__}:{asset_error}"
                audit_errors.append(f"asset:{relative}:{type(asset_error).__name__}:{asset_error}")
    model = stage.get("_model")
    in_memory_sha = None
    e33_gradients_absent = None
    if isinstance(model, torch.nn.Module):
        try:
            in_memory_sha = module_state_tree_sha256(model)
        except BaseException as model_error:
            in_memory_sha = f"unavailable:{type(model_error).__name__}:{model_error}"
            audit_errors.append(f"e33_state_tree:{type(model_error).__name__}:{model_error}")
        try:
            e33_gradients_absent = all(parameter.grad is None for parameter in model.parameters())
        except BaseException as grad_error:
            audit_errors.append(f"e33_gradients:{type(grad_error).__name__}:{grad_error}")
    guard = stage.get("_cache_guard")
    guard_evidence = None
    if isinstance(guard, CacheGuard):
        try:
            guard_evidence = guard.evidence()
        except BaseException as guard_error:
            guard_evidence = {"unavailable": f"{type(guard_error).__name__}:{guard_error}"}
    failure = {
        "schema_version": "prime-rl/latent-a1-nc0-r1-failure/v1",
        "status": status,
        "failure_category": category,
        "stage": stage["name"],
        "error_type": type(error).__name__,
        "error": str(error),
        "execution_commit": args.execution_commit,
        "mechanism_code_commit": None if plan is None else plan.get("mechanism_code_commit"),
        "plan_sha256": None if plan is None else plan.get("plan_sha256"),
        "protected_hash_probe_after_failure": protected,
        "protected_metadata_probe_after_failure": metadata,
        "frozen_asset_hash_probe_after_failure": asset_hashes,
        "frozen_asset_hashes_match_plan": plan is not None and asset_hashes == plan.get("asset_sha256"),
        "a0nc_success_evidence_observed": stage.get("a0nc_success_evidence_observed"),
        "a0nc_success_evidence_matches_plan": plan is not None
        and stage.get("a0nc_success_evidence_observed") == plan.get("a0nc_success_evidence"),
        "protected_hashes_before": stage.get("protected_hashes_before"),
        "protected_hashes_after_completed_audit": stage.get("protected_hashes_after"),
        "checkpoint_metadata_before": stage.get("checkpoint_metadata_before"),
        "checkpoint_metadata_after_completed_audit": stage.get("checkpoint_metadata_after"),
        "e33_parameter_tree_sha256_before": stage.get("e33_parameter_tree_sha256_before"),
        "e33_parameter_tree_sha256_after_completed_audit": stage.get("e33_parameter_tree_sha256_after"),
        "e33_parameter_tree_sha256_failure_audit": in_memory_sha,
        "e33_parameter_tree_matches_preflight": isinstance(in_memory_sha, str)
        and in_memory_sha == stage.get("e33_parameter_tree_sha256_before"),
        "e33_gradients_absent_failure_audit": e33_gradients_absent,
        "base_model_update_attempted": False,
        "bridge_update_attempted": "training_started" in stage.get("breadcrumbs", []),
        "checkpoint_created": False,
        "memory_ledger_partial": stage.get("ledger_rows", []),
        "memory_maxima": stage.get("memory_maxima", {}),
        "candidate_inventory": stage.get("candidate_inventory", []),
        "candidate_valid": False,
        "cache_guard_partial": guard_evidence,
        "failure_audit_errors": audit_errors,
        "failure_audit_bounded_seconds": 180,
        "stage_breadcrumbs": stage.get("breadcrumbs", []),
        "run_id_reusable": False,
        "failure_sha256": "",
    }
    failure["failure_sha256"] = canonical_json_hash(failure, omitted_fields=("failure_sha256",))
    return failure


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "plan",
        "schedule",
        "disjointness",
        "train_bank",
        "validation_bank",
        "held_out_bank",
        "coordinator",
        "worker",
        "repo",
        "output_dir",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--authorized-plan-sha256", required=True)
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    if (
        not isinstance(args.authorized_plan_sha256, str)
        or len(args.authorized_plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in args.authorized_plan_sha256)
    ):
        raise ValueError("A1-NC0 authorized plan SHA-256 is malformed")
    writer = ArtifactWriter(args.output_dir)
    stage = {"name": "artifact_namespace_created", "breadcrumbs": ["artifact_namespace_created"]}
    plan = None
    signal.signal(signal.SIGALRM, lambda _signum, _frame: (_ for _ in ()).throw(TimeoutError("A1-NC0 timeout")))
    signal.alarm(28_260)
    try:
        plan, artifacts, schedule, disjointness = load_plan(
            args.plan,
            {"train": args.train_bank, "validation": args.validation_bank, "held_out": args.held_out_bank},
            args.schedule,
            args.disjointness,
        )
        receipt = run(args, plan, artifacts, schedule, disjointness, stage, writer)
        writer.write_terminal("receipt.json", receipt, plan["resource_bounds"]["maximum_receipt_bytes"])
    except BaseException as error:
        signal.alarm(180)
        ledger = stage.get("_ledger")
        if isinstance(ledger, MemoryLedger):
            stage["ledger_rows"] = ledger.rows
            maxima = {}
            for key in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes"):
                maxima[key] = max((int(row[key]) for row in ledger.rows), default=0)
            stage["memory_maxima"] = maxima
        stage["candidate_inventory"] = writer.inventory()
        maximum_failure_bytes = 16 * 1024 * 1024 if plan is None else plan["resource_bounds"]["maximum_failure_bytes"]
        failure = failure_record(args, error, stage, plan)
        signal.alarm(60)
        writer.write_terminal("failure.json", failure, maximum_failure_bytes)
        raise
    finally:
        signal.alarm(0)
        writer.close()


if __name__ == "__main__":
    main()
