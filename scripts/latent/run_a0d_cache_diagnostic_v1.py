from __future__ import annotations

import argparse
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

import torch
import transformers.cache_utils
import transformers.generation.utils
import transformers.models.qwen3_5.modeling_qwen3_5
from transformers import AutoModelForImageTextToText, AutoTokenizer

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a0d import load_and_validate_a0d_plan, validate_a0d_receipt
from prime_rl.latent.policy_adapter import compose_receiver_inputs

OUTPUT_ROOT = Path("/home/ubuntu/rlm/outputs/latent-a0d-cache-diagnostic-v1")
SHARED_ENVIRONMENT = Path("/home/ubuntu/rlm/prime-rl/.venv")


class DiagnosticIncomplete(RuntimeError):
    pass


class ArtifactWriter:
    def __init__(self, output_dir: Path) -> None:
        if not output_dir.is_absolute() or OUTPUT_ROOT.is_symlink() or not OUTPUT_ROOT.is_dir():
            raise ValueError("A0D output root must be absolute, existing, and non-symlinked")
        if output_dir.parent.resolve(strict=True) != OUTPUT_ROOT.resolve(strict=True):
            raise ValueError("A0D output directory must be a direct child of its frozen root")
        if not output_dir.name.startswith("a0d-") or "/" in output_dir.name:
            raise ValueError("A0D output directory is outside the frozen namespace")
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        self.root_fd = os.open(OUTPUT_ROOT, os.O_RDONLY | os.O_DIRECTORY | nofollow)
        os.mkdir(output_dir.name, mode=0o700, dir_fd=self.root_fd)
        self.dir_fd = os.open(output_dir.name, os.O_RDONLY | os.O_DIRECTORY | nofollow, dir_fd=self.root_fd)

    def write_json(self, name: str, value: dict[str, object], maximum_bytes: int) -> None:
        if name not in {"receipt.json", "failure.json"} or os.listdir(self.dir_fd):
            raise ValueError("A0D artifact name is unknown or namespace is not empty")
        encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        if len(encoded) > maximum_bytes:
            raise ValueError("A0D artifact exceeds its frozen bound")
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
                        raise OSError("short A0D artifact write")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException:
            try:
                os.unlink(temporary, dir_fd=self.dir_fd)
            except FileNotFoundError:
                pass
            raise
        published = False
        try:
            os.rename(temporary, name, src_dir_fd=self.dir_fd, dst_dir_fd=self.dir_fd)
            published = True
        finally:
            if not published:
                try:
                    os.unlink(temporary, dir_fd=self.dir_fd)
                except FileNotFoundError:
                    pass
        os.fsync(self.dir_fd)
        artifact = os.stat(name, dir_fd=self.dir_fd, follow_symlinks=False)
        if os.listdir(self.dir_fd) != [name] or not stat.S_ISREG(artifact.st_mode) or artifact.st_size > maximum_bytes:
            raise RuntimeError("A0D artifact postflight failed")

    def close(self) -> None:
        os.close(self.dir_fd)
        os.close(self.root_fd)


def model_weight(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or not (path / "STABLE").is_file():
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


def require_version(distribution: str, expected: str) -> str:
    actual = importlib.metadata.version(distribution)
    if actual != expected:
        raise ValueError(f"{distribution} version {actual!r} differs from frozen {expected!r}")
    return actual


def source_hashes() -> dict[str, dict[str, str]]:
    modules = {
        "transformers.cache_utils": transformers.cache_utils,
        "transformers.generation.utils": transformers.generation.utils,
        "transformers.models.qwen3_5.modeling_qwen3_5": transformers.models.qwen3_5.modeling_qwen3_5,
    }
    environment = SHARED_ENVIRONMENT.resolve(strict=True)
    observed = {}
    for name, module in modules.items():
        source_path = Path(module.__file__)
        source = source_path.resolve(strict=True)
        if source_path.is_symlink() or not source.is_relative_to(environment):
            raise ValueError(f"A0D runtime source is outside the shared environment: {name}")
        observed[name] = {"path": str(source), "sha256": file_sha256(source)}
    return observed


def tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def cache_length(cache) -> int:
    getter = getattr(cache, "get_seq_length", None)
    if not callable(getter):
        raise DiagnosticIncomplete("cache does not expose get_seq_length")
    value = getter()
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int):
        raise DiagnosticIncomplete("cache sequence length is not an integer")
    return value


def normalized_rms(cached: torch.Tensor, full: torch.Tensor) -> float:
    delta = torch.sqrt(torch.mean((cached.float() - full.float()) ** 2))
    scale = torch.sqrt(torch.mean(full.float() ** 2)).clamp_min(1e-12)
    return float((delta / scale).item())


def rope_summary(model) -> dict[str, object]:
    rope = model.model.rope_deltas
    if rope is None:
        return {"state": "none"}
    detached = rope.detach()
    summary: dict[str, object] = {
        "state": "tensor",
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "sha256": tensor_sha256(detached),
    }
    if detached.numel() <= 16:
        summary["values"] = detached.cpu().reshape(-1).tolist()
    return summary


def prepared_summary(prepared: dict[str, object]) -> dict[str, object]:
    values: dict[str, object] = {}
    for key in sorted(prepared):
        value = prepared[key]
        if isinstance(value, torch.Tensor):
            item: dict[str, object] = {
                "kind": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": tensor_sha256(value),
            }
            if value.numel() <= 16:
                item["values"] = value.detach().cpu().reshape(-1).tolist()
            values[key] = item
        elif key == "past_key_values":
            values[key] = {"kind": type(value).__name__, "sequence_length": cache_length(value)}
        elif value is None or isinstance(value, (bool, int, float, str)):
            values[key] = value
        else:
            values[key] = {"kind": type(value).__name__}
    return {"keys": sorted(prepared), "values": values}


def render_ids(tokenizer, messages: list[dict[str, object]], *, generation_prompt: bool) -> torch.Tensor:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
    return tokenizer(rendered, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda:0")


def build_representations(model, tokenizer, example: dict[str, object]) -> tuple[dict[str, object], torch.Tensor]:
    parent_ids = render_ids(tokenizer, example["parent_messages"], generation_prompt=False)
    child_prefix_ids = render_ids(tokenizer, example["child_messages"], generation_prompt=False)
    child_ids = render_ids(tokenizer, example["child_messages"], generation_prompt=True)
    if hashlib.sha256(example["continuation_text"].encode()).hexdigest() != (
        "d2a9291c35fc42fadedff20c365f38da2813504f980dd6ba6bdda413a79bd6e0"
    ):
        raise DiagnosticIncomplete("frozen continuation text changed")
    continuation_ids = (
        tokenizer(example["continuation_text"], add_special_tokens=False, return_tensors="pt")
        .input_ids[:, :4]
        .to("cuda:0")
    )
    if (
        continuation_ids.shape[1] != 4
        or child_ids.shape[1] <= child_prefix_ids.shape[1]
        or not torch.equal(child_ids[:, : child_prefix_ids.shape[1]], child_prefix_ids)
    ):
        raise DiagnosticIncomplete("frozen fixture no longer renders the expected prompt and four-token continuation")
    mask = torch.ones_like(child_ids)
    positions = torch.arange(child_ids.shape[1], device="cuda:0").unsqueeze(0)
    exact_embeddings = model.get_input_embeddings()(child_ids)
    with torch.inference_mode():
        parent = model(
            input_ids=parent_ids,
            attention_mask=torch.ones_like(parent_ids),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    if parent.hidden_states is None or not bool(torch.isfinite(parent.hidden_states[-1]).all()):
        raise DiagnosticIncomplete("parent final hidden state is absent or non-finite")
    source = parent.hidden_states[-1][:, -8:, :].detach()
    embedding_norm = exact_embeddings.detach().float().norm(dim=-1).mean()
    source_norm = source.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    workspace = source / source_norm.to(source.dtype) * embedding_norm.to(source.dtype)
    soft = compose_receiver_inputs(
        exact_embeddings,
        mask,
        workspace,
        injection_index=child_prefix_ids.shape[1],
        gate=0.125,
        position_ids=positions,
    )
    if soft.workspace_span != (child_prefix_ids.shape[1], child_prefix_ids.shape[1] + 8):
        raise DiagnosticIncomplete("soft workspace span changed")
    representations: dict[str, object] = {
        "D": {"input_ids": child_ids, "attention_mask": mask, "position_ids": None},
        "E": {"inputs_embeds": exact_embeddings, "attention_mask": mask, "position_ids": positions},
        "S": {
            "inputs_embeds": soft.inputs_embeds,
            "attention_mask": soft.attention_mask,
            "position_ids": soft.position_ids,
        },
        "fixture": {
            "example_id": example["example_id"],
            "parent_token_count": parent_ids.shape[1],
            "child_token_count": child_ids.shape[1],
            "soft_prompt_length": soft.inputs_embeds.shape[1],
            "child_input_ids_sha256": tensor_sha256(child_ids),
            "continuation_input_ids_sha256": tensor_sha256(continuation_ids),
            "continuation_token_ids": continuation_ids.flatten().tolist(),
            "workspace_source_sha256": tensor_sha256(source),
            "soft_prompt_sha256": tensor_sha256(soft.inputs_embeds),
        },
    }
    return representations, continuation_ids


def run_arm_branch(
    model,
    arm_name: str,
    representation: dict[str, torch.Tensor | None],
    continuation_ids: torch.Tensor,
    branch: str,
) -> dict[str, object]:
    model.model.rope_deltas = None
    rope_states: dict[str, object] = {"before_prefill": rope_summary(model)}
    attention_mask = representation["attention_mask"]
    position_ids = representation["position_ids"]
    if arm_name == "D":
        full_ids = representation["input_ids"]
        full_embeddings = None
        prefill_kwargs = {"input_ids": full_ids, "attention_mask": attention_mask}
    else:
        full_ids = None
        full_embeddings = representation["inputs_embeds"]
        prefill_kwargs = {
            "inputs_embeds": full_embeddings,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
    with torch.inference_mode():
        cached = model(**prefill_kwargs, use_cache=True, return_dict=True)
    initial_length = cache_length(cached.past_key_values)
    expected_initial = attention_mask.shape[1]
    if initial_length != expected_initial or not bool(torch.isfinite(cached.logits).all()):
        raise DiagnosticIncomplete(f"{arm_name}/{branch} prefill cache or logits invalid")
    prefill_cache_type = type(cached.past_key_values).__name__
    prefill_last_logits_sha256 = tensor_sha256(cached.logits[:, -1].float())
    rope_states["after_prefill"] = rope_summary(model)
    steps = []
    for index in range(4):
        token = continuation_ids[:, index : index + 1]
        attention_mask = torch.cat((attention_mask, torch.ones_like(token)), dim=1)
        if arm_name == "D":
            full_ids = torch.cat((full_ids, token), dim=1)
            full_kwargs = {"input_ids": full_ids, "attention_mask": attention_mask}
        else:
            full_embeddings = torch.cat((full_embeddings, model.get_input_embeddings()(token)), dim=1)
            position_ids = torch.cat((position_ids, position_ids[:, -1:] + 1), dim=1)
            full_kwargs = {
                "inputs_embeds": full_embeddings,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        prepare_kwargs: dict[str, object] = {
            "past_key_values": cached.past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }
        if branch == "explicit_next_position":
            next_position = torch.tensor([[initial_length + index]], dtype=torch.long, device="cuda:0")
            prepare_kwargs["position_ids"] = next_position
            prepare_kwargs["cache_position"] = next_position.flatten()
        rope_before_prepare = rope_summary(model)
        prepared = model.prepare_inputs_for_generation(token, **prepare_kwargs)
        if branch == "auto_position" and ("position_ids" in prepared or "cache_position" in prepared):
            raise DiagnosticIncomplete("auto-position branch unexpectedly prepared an explicit position")
        if branch == "explicit_next_position" and not {"position_ids", "cache_position"}.issubset(prepared):
            raise DiagnosticIncomplete("explicit-position branch dropped a frozen position input")
        rope_after_prepare = rope_summary(model)
        with torch.inference_mode():
            full = model(**full_kwargs, use_cache=False, return_dict=True)
            cached = model(**prepared, return_dict=True)
        cached_logits = cached.logits[:, -1].float()
        full_logits = full.logits[:, -1].float()
        max_abs = float((cached_logits - full_logits).abs().max().item())
        rms = normalized_rms(cached_logits, full_logits)
        observed_length = cache_length(cached.past_key_values)
        if (
            observed_length != initial_length + index + 1
            or not bool(torch.isfinite(cached_logits).all())
            or not bool(torch.isfinite(full_logits).all())
            or not math.isfinite(max_abs)
            or not math.isfinite(rms)
        ):
            raise DiagnosticIncomplete(f"{arm_name}/{branch} step {index + 1} is incomplete or non-finite")
        steps.append(
            {
                "step": index + 1,
                "cache_sequence_length": observed_length,
                "maximum_absolute_logit_difference": max_abs,
                "normalized_rms": rms,
                "greedy_equal": bool(torch.equal(cached_logits.argmax(-1), full_logits.argmax(-1))),
                "cached_logits_sha256": tensor_sha256(cached_logits),
                "full_logits_sha256": tensor_sha256(full_logits),
                "prepared": prepared_summary(prepared),
                "rope_state": {
                    "before_prepare": rope_before_prepare,
                    "after_prepare": rope_after_prepare,
                    "after_decode": rope_summary(model),
                },
            }
        )
    return {
        "arm": arm_name,
        "position_branch": branch,
        "fresh_cache": True,
        "initial_cache_sequence_length": initial_length,
        "initial_logits_finite": True,
        "prefill_cache_type": prefill_cache_type,
        "prefill_last_logits_sha256": prefill_last_logits_sha256,
        "rope_state": rope_states,
        "steps": steps,
    }


def verify_execution_tree(args: argparse.Namespace, plan: dict[str, object]) -> None:
    if (
        not args.repo.is_absolute()
        or args.repo.is_symlink()
        or args.repo.resolve() != args.plan.resolve(strict=True).parents[2]
    ):
        raise ValueError("A0D repository differs from plan location")
    if len(args.execution_commit) != 40 or any(char not in "0123456789abcdef" for char in args.execution_commit):
        raise ValueError("A0D execution commit is malformed")
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=args.repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if actual != args.execution_commit or dirty:
        raise ValueError("A0D requires its exact clean execution commit")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", plan["mechanism_code_commit"], args.execution_commit],
        cwd=args.repo,
        check=True,
    )
    for relative, expected in plan["asset_sha256"].items():
        asset = args.repo / relative
        if asset.is_symlink() or not asset.is_file() or file_sha256(asset) != expected:
            raise ValueError(f"A0D asset differs from freeze: {relative}")


def host_ram_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def run(
    args: argparse.Namespace, plan: dict[str, object], bank: dict[str, object], stage: dict[str, str]
) -> dict[str, object]:
    if not args.owner_approved:
        raise ValueError("A0D requires root approval after freeze review")
    verify_execution_tree(args, plan)
    stage["name"] = "execution_tree_verified"
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise ValueError("A0D requires Python 3.12")
    if os.environ.get("UV_PROJECT_ENVIRONMENT") != str(SHARED_ENVIRONMENT):
        raise ValueError("A0D requires the shared project environment")
    if os.environ.get("PYTHONPATH") != str(args.repo / "src"):
        raise ValueError("A0D requires the exact checked-out source path")
    versions = {
        "python": platform.python_version(),
        "transformers": require_version("transformers", plan["runtime"]["transformers"]),
        "torch_distribution": require_version("torch", plan["runtime"]["torch_distribution"]),
        "torch_runtime": str(torch.__version__),
    }
    if versions["torch_runtime"] != plan["runtime"]["torch_runtime"]:
        raise ValueError("A0D Torch runtime differs from freeze")
    runtime_sources = source_hashes()
    if {name: value["sha256"] for name, value in runtime_sources.items()} != plan["runtime"][
        "transformers_source_sha256"
    ]:
        raise ValueError("A0D Transformers sources differ from freeze")
    stage["name"] = "runtime_preflight_verified"
    if args.coordinator.resolve() != Path(plan["remote_paths"]["coordinator_e33"]) or args.worker.resolve() != Path(
        plan["remote_paths"]["worker_h176"]
    ):
        raise ValueError("A0D protected paths differ from freeze")
    weights = {"coordinator_e33": model_weight(args.coordinator), "worker_h176": model_weight(args.worker)}
    hashes_before = {name: file_sha256(weight) for name, weight in weights.items()}
    metadata_before = {
        "coordinator_e33": metadata_hashes(args.coordinator),
        "worker_h176": metadata_hashes(args.worker),
    }
    if hashes_before != plan["protected_checkpoints"] or any(
        value != plan["runtime"]["checkpoint_metadata_sha256"] for value in metadata_before.values()
    ):
        raise ValueError("A0D protected checkpoint preflight differs from freeze")
    stage["name"] = "protected_preflight_verified"
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != plan["resource_bounds"]["gpu_model"]:
        raise RuntimeError("A0D GPU differs from freeze")
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    free_disk = shutil.disk_usage(plan["resource_bounds"]["output_root"]).free
    ram = host_ram_bytes()
    if (
        total_gib < plan["resource_bounds"]["minimum_gpu_memory_gib"]
        or free_disk < plan["resource_bounds"]["minimum_free_disk_gib"] * 2**30
        or ram < plan["resource_bounds"]["minimum_host_ram_gib"] * 2**30
    ):
        raise RuntimeError("A0D host falls below a frozen resource bound")
    torch.manual_seed(20260905)
    torch.cuda.manual_seed_all(20260905)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.coordinator, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.coordinator,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda:0")
    model.eval()
    if (
        model.__class__.__name__ != plan["runtime"]["model_class"]
        or model.config.text_config.hidden_size != plan["runtime"]["hidden_size"]
    ):
        raise TypeError("A0D model runtime differs from freeze")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stage["name"] = "model_loaded_frozen"
    examples = [example for example in bank["examples"] if example["example_id"] == plan["diagnostic"]["example_id"]]
    if len(examples) != 1:
        raise ValueError("A0D frozen example is absent or duplicated")
    representations, continuation = build_representations(model, tokenizer, examples[0])
    fixture = representations.pop("fixture")
    arms = []
    for arm_name in ("D", "E", "S"):
        for branch in ("auto_position", "explicit_next_position"):
            stage["name"] = f"arm_{arm_name}_{branch}"
            try:
                arms.append(run_arm_branch(model, arm_name, representations[arm_name], continuation, branch))
            except (torch.cuda.OutOfMemoryError, TimeoutError):
                raise
            except DiagnosticIncomplete:
                raise
            except (RuntimeError, TypeError, ValueError) as error:
                raise DiagnosticIncomplete(
                    f"{arm_name}/{branch} model-interface diagnostic failed: {type(error).__name__}: {error}"
                ) from error
    hashes_after = {name: file_sha256(weight) for name, weight in weights.items()}
    metadata_after = {"coordinator_e33": metadata_hashes(args.coordinator), "worker_h176": metadata_hashes(args.worker)}
    if hashes_after != hashes_before or metadata_after != metadata_before:
        raise RuntimeError("A0D protected checkpoints changed")
    stage["name"] = "protected_postflight_verified"
    receipt: dict[str, object] = {
        "schema_version": "prime-rl/latent-a0d-cache-diagnostic-receipt/v1",
        "status": "diagnostic_complete",
        "claim": "non-promotional cache causal measurements only",
        "plan_sha256": plan["plan_sha256"],
        "bank_sha256": plan["bank_sha256"],
        "mechanism_code_commit": plan["mechanism_code_commit"],
        "execution_commit": args.execution_commit,
        "asset_sha256": plan["asset_sha256"],
        "versions": versions,
        "transformers_runtime_sources": runtime_sources,
        "model_runtime": {
            "class": model.__class__.__name__,
            "hidden_size": model.config.text_config.hidden_size,
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        },
        "gpu": {"name": torch.cuda.get_device_name(0), "total_memory_gib": total_gib},
        "host": {"ram_bytes": ram, "free_disk_bytes_before": free_disk},
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
        "checkpoint_metadata_before": metadata_before,
        "checkpoint_metadata_after": metadata_after,
        "fixture": fixture,
        "reference_normalized_rms": 0.01,
        "reference_is_promotion_gate": False,
        "arms": arms,
        "optimizer_created": False,
        "checkpoint_created": False,
        "model_update_attempted": False,
        "resources": {
            "wall_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(0),
        },
        "interpretation_boundary": plan["interpretation_boundary"],
    }
    receipt["receipt_sha256"] = canonical_json_hash(receipt)
    try:
        validate_a0d_receipt(receipt, plan=plan)
    except ValueError as error:
        raise DiagnosticIncomplete(f"A0D receipt contract failed: {error}") from error
    return receipt


def failure_record(
    args: argparse.Namespace, error: BaseException, stage: str, plan: dict[str, object] | None
) -> dict[str, object]:
    protected = {}
    for name, checkpoint in (("coordinator_e33", args.coordinator), ("worker_h176", args.worker)):
        try:
            protected[name] = {
                "model_sha256": file_sha256(model_weight(checkpoint)),
                "metadata_sha256": metadata_hashes(checkpoint),
            }
        except (OSError, ValueError) as hash_error:
            protected[name] = {"hash_probe_error": f"{type(hash_error).__name__}: {hash_error}"}
    failure: dict[str, object] = {
        "schema_version": "prime-rl/latent-a0d-cache-diagnostic-failure/v1",
        "status": "diagnostic_incomplete" if isinstance(error, DiagnosticIncomplete) else "infrastructure_invalid",
        "failure_category": (
            "diagnostic_execution_or_finiteness_failure"
            if isinstance(error, DiagnosticIncomplete)
            else "environment_provenance_timeout_or_oom"
        ),
        "error_type": type(error).__name__,
        "error": str(error),
        "stage": stage,
        "plan_sha256": None if plan is None else plan.get("plan_sha256"),
        "mechanism_code_commit": None if plan is None else plan.get("mechanism_code_commit"),
        "execution_commit": args.execution_commit,
        "protected_hash_probe_after_failure": protected,
        "model_update_attempted": False,
    }
    failure["failure_sha256"] = canonical_json_hash(failure)
    return failure


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the no-update A0D cache causal diagnostic.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--coordinator", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    writer = ArtifactWriter(args.output_dir)
    stage = {"name": "artifact_namespace_created"}
    plan = None

    def timeout_handler(_signum, _frame) -> None:
        raise TimeoutError("A0D exceeded its frozen wall-time bound")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(28 * 60)
    try:
        plan, bank = load_and_validate_a0d_plan(args.plan, args.bank)
        stage["name"] = "plan_bank_and_evidence_validated"
        receipt = run(args, plan, bank, stage)
        writer.write_json("receipt.json", receipt, plan["resource_bounds"]["maximum_output_bytes"])
    except BaseException as error:
        writer.write_json("failure.json", failure_record(args, error, stage["name"], plan), 16 * 1024 * 1024)
        raise
    finally:
        signal.alarm(0)
        writer.close()


if __name__ == "__main__":
    main()
