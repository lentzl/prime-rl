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

from prime_rl.latent.a0 import canonical_json_hash, file_sha256, load_and_validate_a0_plan
from prime_rl.latent.policy_adapter import CapturedFeatures, compose_receiver_inputs

A0_OUTPUT_ROOT = Path("/home/ubuntu/rlm/outputs/latent-a0-mechanism-v1")
A0_SHARED_ENVIRONMENT = Path("/home/ubuntu/rlm/prime-rl/.venv")


class MechanismRejected(RuntimeError):
    pass


class ArtifactWriter:
    def __init__(self, output_dir: Path, expected_root: Path) -> None:
        if not output_dir.is_absolute() or expected_root.is_symlink() or not expected_root.is_dir():
            raise ValueError("A0 output root must be an absolute existing non-symlink directory")
        if output_dir.parent.resolve(strict=True) != expected_root.resolve(strict=True):
            raise ValueError("A0 output directory must be a direct child of the frozen output root")
        if not output_dir.name.startswith("a0-") or "/" in output_dir.name:
            raise ValueError("A0 output directory name is outside the frozen namespace")
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        self.root_fd = os.open(expected_root, os.O_RDONLY | os.O_DIRECTORY | nofollow)
        os.mkdir(output_dir.name, mode=0o700, dir_fd=self.root_fd)
        self.dir_fd = os.open(output_dir.name, os.O_RDONLY | os.O_DIRECTORY | nofollow, dir_fd=self.root_fd)

    def write_json(self, name: str, value: dict[str, object], *, maximum_directory_bytes: int) -> None:
        if name not in {"receipt.json", "failure.json"}:
            raise ValueError("unknown A0 artifact name")
        if os.listdir(self.dir_fd):
            raise FileExistsError("A0 artifact namespace is not empty")
        encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        if len(encoded) > maximum_directory_bytes:
            raise ValueError("A0 artifact exceeds the output-directory bound")
        temporary = f".{name}.tmp"
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
            0o600,
            dir_fd=self.dir_fd,
        )
        try:
            try:
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if written == 0:
                        raise OSError("short write while persisting A0 artifact")
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
        entries = os.listdir(self.dir_fd)
        artifact = os.stat(name, dir_fd=self.dir_fd, follow_symlinks=False)
        if entries != [name] or not stat.S_ISREG(artifact.st_mode) or artifact.st_size > maximum_directory_bytes:
            raise RuntimeError("A0 artifact namespace postflight failed")

    def close(self) -> None:
        os.close(self.dir_fd)
        os.close(self.root_fd)


def model_weight(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or not (path / "STABLE").is_file():
        raise ValueError(f"protected model is not an absolute stable non-symlink checkpoint: {path}")
    weight = path / "model.safetensors"
    if weight.is_symlink() or not weight.is_file():
        raise ValueError(f"protected model has no direct dense weight file: {weight}")
    return weight


def metadata_hashes(path: Path) -> dict[str, str]:
    required = (
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    if any((path / name).is_symlink() or not (path / name).is_file() for name in required):
        raise ValueError("checkpoint metadata is incomplete or symlinked")
    return {name: file_sha256(path / name) for name in required}


def render_ids(tokenizer, messages: list[dict[str, object]], *, generation_prompt: bool) -> torch.Tensor:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
    return tokenizer(rendered, add_special_tokens=False, return_tensors="pt").input_ids


def finite_nonzero(tensor: torch.Tensor | None) -> bool:
    return tensor is not None and bool(torch.isfinite(tensor).all()) and bool(torch.count_nonzero(tensor))


def require_version(distribution: str, expected: str) -> str:
    actual = importlib.metadata.version(distribution)
    if actual != expected:
        raise ValueError(f"{distribution} version {actual!r} differs from frozen {expected!r}")
    return actual


def transformers_source_hashes() -> dict[str, dict[str, str]]:
    modules = {
        "transformers.cache_utils": transformers.cache_utils,
        "transformers.generation.utils": transformers.generation.utils,
        "transformers.models.qwen3_5.modeling_qwen3_5": transformers.models.qwen3_5.modeling_qwen3_5,
    }
    environment = A0_SHARED_ENVIRONMENT.resolve(strict=True)
    result: dict[str, dict[str, str]] = {}
    for name, module in modules.items():
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise ValueError(f"A0 runtime module has no source path: {name}")
        path = Path(module_file)
        resolved = path.resolve(strict=True)
        if path.is_symlink() or not resolved.is_relative_to(environment):
            raise ValueError(f"A0 runtime module is outside the frozen shared environment: {name}")
        result[name] = {"path": str(resolved), "sha256": file_sha256(resolved)}
    return result


def capture_transcript_end(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> CapturedFeatures:
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1 or hidden_states.shape[-1] != 2048:
        raise MechanismRejected("A0 transcript capture did not expose one 2048-wide hidden sequence")
    indices = torch.nonzero(attention_mask[0].bool(), as_tuple=False).flatten()[-128:]
    if not indices.numel():
        raise MechanismRejected("A0 transcript capture is empty")
    captured = torch.zeros((1, 128, 2048), dtype=hidden_states.dtype, device=hidden_states.device)
    captured_mask = torch.zeros((1, 128), dtype=attention_mask.dtype, device=attention_mask.device)
    padded_indices = torch.full((128,), -1, dtype=torch.long)
    selected = hidden_states[:, indices, :]
    captured[:, -selected.shape[1] :, :] = selected
    captured_mask[:, -selected.shape[1] :] = 1
    padded_indices[-indices.shape[0] :] = indices.detach().cpu()
    spec_hash = canonical_json_hash(
        {
            "schema_version": "prime-rl/a0-rendered-transcript-capture/v1",
            "layer": -1,
            "maximum_non_padding_tokens": 128,
            "boundary": "rendered_transcript_end_not_harness_acceptance",
            "detach": True,
        }
    )
    return CapturedFeatures(captured.detach(), captured_mask, padded_indices, spec_hash)


def cache_length(cache) -> int:
    getter = getattr(cache, "get_seq_length", None)
    if not callable(getter):
        raise MechanismRejected("Qwen cache does not expose get_seq_length()")
    length = getter()
    if isinstance(length, torch.Tensor):
        length = length.item()
    if isinstance(length, bool) or not isinstance(length, int):
        raise MechanismRejected("Qwen cache sequence length is not an integer")
    return length


def normalized_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    difference_rms = torch.sqrt(torch.mean((left.float() - right.float()) ** 2))
    reference_rms = torch.sqrt(torch.mean(right.float() ** 2)).clamp_min(1e-12)
    return float((difference_rms / reference_rms).item())


def tensor_bytes_sha256(tensor: torch.Tensor) -> str:
    raw_bytes = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


def probe_example(model, tokenizer, example: dict[str, object], mechanism: dict[str, object]) -> dict[str, object]:
    parent_ids = render_ids(tokenizer, example["parent_messages"], generation_prompt=False).to("cuda:0")
    child_prefix_ids = render_ids(tokenizer, example["child_messages"], generation_prompt=False).to("cuda:0")
    child_ids = render_ids(tokenizer, example["child_messages"], generation_prompt=True).to("cuda:0")
    continuation_ids = (
        tokenizer(example["continuation_text"], add_special_tokens=False, return_tensors="pt")
        .input_ids[:, :4]
        .to("cuda:0")
    )
    if continuation_ids.shape[1] != 4:
        raise ValueError("A0 continuation fixture does not contain four tokens")
    if child_ids.shape[1] <= child_prefix_ids.shape[1] or not torch.equal(
        child_ids[:, : child_prefix_ids.shape[1]], child_prefix_ids
    ):
        raise ValueError("chat template does not expose a strict assistant-generation opening suffix")
    parent_mask = torch.ones_like(parent_ids)
    child_mask = torch.ones_like(child_ids)
    child_labels = child_ids.clone()
    child_positions = torch.arange(child_ids.shape[1], device="cuda:0").unsqueeze(0)
    embeddings = model.get_input_embeddings()(child_ids)
    placeholder = torch.zeros((8, 2048), dtype=embeddings.dtype, device=embeddings.device)

    with torch.inference_mode():
        standard = model(
            input_ids=child_ids,
            attention_mask=child_mask,
            labels=child_labels,
            use_cache=False,
            return_dict=True,
        )
        bypass_batch = compose_receiver_inputs(
            embeddings,
            child_mask,
            placeholder,
            injection_index=child_prefix_ids.shape[1],
            gate=0.0,
            position_ids=child_positions,
            labels=child_labels,
        )
        bypass = model(
            inputs_embeds=bypass_batch.inputs_embeds,
            attention_mask=bypass_batch.attention_mask,
            position_ids=bypass_batch.position_ids,
            labels=bypass_batch.labels,
            use_cache=False,
            return_dict=True,
        )
        parent = model(
            input_ids=parent_ids,
            attention_mask=parent_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hard_equal = torch.equal(standard.logits, bypass.logits)
    hard_max_abs = float((standard.logits.float() - bypass.logits.float()).abs().max().item())
    hard_logits_finite = (
        bool(torch.isfinite(standard.logits).all())
        and bool(torch.isfinite(bypass.logits).all())
        and math.isfinite(hard_max_abs)
        and hard_max_abs == 0.0
    )
    hard_contract_exact = (
        torch.equal(bypass_batch.inputs_embeds, embeddings)
        and torch.equal(bypass_batch.attention_mask, child_mask)
        and torch.equal(bypass_batch.position_ids, child_positions)
        and torch.equal(bypass_batch.labels, child_labels)
    )
    if not hard_equal or not hard_logits_finite or not hard_contract_exact or bypass_batch.workspace_span is not None:
        raise MechanismRejected(f"hard bypass differs from input_ids path; max_abs={hard_max_abs}")
    if parent.hidden_states is None:
        raise MechanismRejected("e33 wrapper did not return hidden states")
    captured = capture_transcript_end(parent.hidden_states[-1], parent_mask)
    repeated_capture = capture_transcript_end(parent.hidden_states[-1], parent_mask)
    if captured.hidden_states.requires_grad or not bool(torch.isfinite(captured.hidden_states).all()):
        raise MechanismRejected("final transcript hidden capture is not detached and finite")
    repeat_equal = (
        torch.equal(captured.hidden_states, repeated_capture.hidden_states)
        and torch.equal(captured.attention_mask, repeated_capture.attention_mask)
        and torch.equal(captured.token_indices, repeated_capture.token_indices)
        and captured.capture_spec_hash == repeated_capture.capture_spec_hash
    )
    visible_indices = torch.nonzero(parent_mask[0].bool(), as_tuple=False).flatten()[-128:]
    expected_mask = torch.zeros_like(captured.attention_mask)
    expected_mask[:, -visible_indices.shape[0] :] = 1
    expected_indices = torch.full_like(captured.token_indices, -1)
    expected_indices[-visible_indices.shape[0] :] = visible_indices.detach().cpu()
    expected_hidden = torch.zeros_like(captured.hidden_states)
    expected_hidden[:, -visible_indices.shape[0] :, :] = parent.hidden_states[-1][:, visible_indices, :]
    mask_and_indices_exact = torch.equal(captured.attention_mask, expected_mask) and torch.equal(
        captured.token_indices, expected_indices
    )
    hidden_padding_content_exact = torch.equal(captured.hidden_states, expected_hidden)
    if not repeat_equal or not mask_and_indices_exact or not hidden_padding_content_exact:
        raise MechanismRejected("final transcript hidden capture is not repeatable or exactly indexed")
    capture_tensor_sha256 = tensor_bytes_sha256(captured.hidden_states)

    with torch.inference_mode():
        ids_path = child_ids
        embeds_path = embeddings
        mask = child_mask
        positions = child_positions
        ids_tokens: list[int] = []
        embeds_tokens: list[int] = []
        for _ in range(4):
            ids_logits = model(input_ids=ids_path, attention_mask=mask, use_cache=False, return_dict=True).logits[:, -1]
            embeds_logits = model(
                inputs_embeds=embeds_path,
                attention_mask=mask,
                position_ids=positions,
                use_cache=False,
                return_dict=True,
            ).logits[:, -1]
            if not bool(torch.isfinite(ids_logits).all()) or not bool(torch.isfinite(embeds_logits).all()):
                raise MechanismRejected("hard-path greedy logits are not finite")
            ids_token = ids_logits.argmax(dim=-1, keepdim=True)
            embeds_token = embeds_logits.argmax(dim=-1, keepdim=True)
            ids_tokens.append(int(ids_token.item()))
            embeds_tokens.append(int(embeds_token.item()))
            ids_path = torch.cat((ids_path, ids_token), dim=1)
            embeds_path = torch.cat((embeds_path, model.get_input_embeddings()(embeds_token)), dim=1)
            mask = torch.cat((mask, torch.ones_like(ids_token)), dim=1)
            positions = torch.cat((positions, positions[:, -1:] + 1), dim=1)
    if ids_tokens != embeds_tokens:
        raise MechanismRejected("four-token hard-bypass greedy continuation differs from input_ids")

    workspace_source = captured.hidden_states[:, -8:, :].clone().requires_grad_(True)
    embedding_norm = embeddings.detach().float().norm(dim=-1).mean()
    source_norm = workspace_source.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    workspace = workspace_source / source_norm.to(workspace_source.dtype) * embedding_norm.to(workspace_source.dtype)
    gate = torch.tensor(mechanism["soft_probe_gate"], dtype=embeddings.dtype, device="cuda:0", requires_grad=True)
    soft_batch = compose_receiver_inputs(
        embeddings.detach(),
        child_mask,
        workspace,
        injection_index=child_prefix_ids.shape[1],
        gate=gate,
        position_ids=child_positions,
        labels=child_labels,
    )
    injection_index = child_prefix_ids.shape[1]
    soft_mask_expected = torch.cat(
        (
            child_mask[:, :injection_index],
            torch.ones((1, 8), device="cuda:0", dtype=child_mask.dtype),
            child_mask[:, injection_index:],
        ),
        dim=1,
    )
    soft_labels_expected = torch.cat(
        (
            child_labels[:, :injection_index],
            torch.full((1, 8), -100, device="cuda:0", dtype=child_labels.dtype),
            child_labels[:, injection_index:],
        ),
        dim=1,
    )
    soft_positions_expected = torch.arange(soft_batch.inputs_embeds.shape[1], device="cuda:0").unsqueeze(0)
    original_embeddings = torch.cat(
        (
            soft_batch.inputs_embeds[:, :injection_index],
            soft_batch.inputs_embeds[:, injection_index + 8 :],
        ),
        dim=1,
    )
    original_tokens_mask_labels_preserved = (
        torch.equal(original_embeddings, embeddings)
        and torch.equal(soft_batch.attention_mask, soft_mask_expected)
        and torch.equal(soft_batch.labels, soft_labels_expected)
    )
    positions_sequential_shifted = torch.equal(soft_batch.position_ids, soft_positions_expected)
    expected_loss_mask = torch.zeros_like(soft_labels_expected, dtype=torch.bool)
    expected_loss_mask[:, injection_index : injection_index + 8] = True
    loss_mask_exact = torch.equal(soft_batch.labels == -100, expected_loss_mask)
    soft = model(
        inputs_embeds=soft_batch.inputs_embeds,
        attention_mask=soft_batch.attention_mask,
        position_ids=soft_batch.position_ids,
        labels=soft_batch.labels,
        use_cache=False,
        return_dict=True,
    )
    expected_span = (injection_index, injection_index + 8)
    if (
        soft_batch.workspace_span != expected_span
        or not original_tokens_mask_labels_preserved
        or not positions_sequential_shifted
        or not loss_mask_exact
        or not bool(torch.isfinite(soft.logits).all())
        or soft.loss is None
        or not bool(torch.isfinite(soft.loss))
    ):
        raise MechanismRejected("soft workspace insertion, masking, positions, or outputs are invalid")
    soft.loss.float().backward()
    workspace_gradient_ok = finite_nonzero(workspace_source.grad)
    gate_gradient_ok = finite_nonzero(gate.grad)
    if (
        not workspace_gradient_ok
        or not gate_gradient_ok
        or any(parameter.grad is not None for parameter in model.parameters())
    ):
        raise MechanismRejected("soft workspace autograd isolation failed")

    cache_steps: list[dict[str, object]] = []
    with torch.inference_mode():
        full_embeddings = soft_batch.inputs_embeds.detach()
        full_mask = soft_batch.attention_mask
        full_positions = soft_batch.position_ids
        cached = model(
            inputs_embeds=full_embeddings,
            attention_mask=full_mask,
            position_ids=full_positions,
            use_cache=True,
            return_dict=True,
        )
        initial_cache_length = cache_length(cached.past_key_values)
        if initial_cache_length != full_embeddings.shape[1] or not bool(torch.isfinite(cached.logits).all()):
            raise MechanismRejected("soft cache prefill length or logits are invalid")
        for index in range(4):
            token = continuation_ids[:, index : index + 1]
            full_embeddings = torch.cat((full_embeddings, model.get_input_embeddings()(token)), dim=1)
            full_mask = torch.cat((full_mask, torch.ones_like(token)), dim=1)
            full_positions = torch.cat((full_positions, full_positions[:, -1:] + 1), dim=1)
            full = model(
                inputs_embeds=full_embeddings,
                attention_mask=full_mask,
                position_ids=full_positions,
                use_cache=False,
                return_dict=True,
            )
            prepared = model.prepare_inputs_for_generation(
                token,
                past_key_values=cached.past_key_values,
                attention_mask=full_mask,
                use_cache=True,
            )
            cached = model(**prepared, return_dict=True)
            observed_length = cache_length(cached.past_key_values)
            cached_logits = cached.logits[:, -1].float()
            full_logits = full.logits[:, -1].float()
            max_abs = float((cached_logits - full_logits).abs().max().item())
            rms = normalized_rms(cached_logits, full_logits)
            greedy_equal = torch.equal(cached_logits.argmax(dim=-1), full_logits.argmax(dim=-1))
            if (
                observed_length != initial_cache_length + index + 1
                or not bool(torch.isfinite(cached_logits).all())
                or not bool(torch.isfinite(full_logits).all())
                or not math.isfinite(max_abs)
                or not math.isfinite(rms)
                or max_abs > mechanism["cache_full_recompute_max_abs"]
                or rms > mechanism["cache_full_recompute_normalized_rms"]
                or not greedy_equal
            ):
                raise MechanismRejected(
                    f"cached/full disagreement at step {index + 1}: length={observed_length}, "
                    f"max_abs={max_abs}, normalized_rms={rms}, greedy_equal={greedy_equal}"
                )
            cache_steps.append(
                {
                    "step": index + 1,
                    "cache_sequence_length": observed_length,
                    "maximum_absolute_logit_difference": max_abs,
                    "normalized_rms": rms,
                    "greedy_token_equal": greedy_equal,
                }
            )

    return {
        "example_id": example["example_id"],
        "status": "complete",
        "prompt_tokens": {
            "parent": parent_ids.shape[1],
            "child_without_assistant_opening": child_prefix_ids.shape[1],
            "child_with_assistant_opening": child_ids.shape[1],
        },
        "hard_bypass": {
            "bitwise_equal": hard_equal,
            "logits_finite": hard_logits_finite,
            "maximum_absolute_logit_difference": hard_max_abs,
            "additional_positions": 0,
            "labels_mask_positions_preserved": hard_contract_exact,
            "four_token_greedy_continuation_equal": True,
            "four_token_greedy_logits_finite": True,
            "greedy_token_ids": ids_tokens,
        },
        "capture": {
            "claim_boundary": "rendered_transcript_end_not_harness_acceptance",
            "layer": -1,
            "shape": list(captured.hidden_states.shape),
            "finite": True,
            "detached": True,
            "repeat_bitwise_equal": repeat_equal,
            "mask_and_indices_exact": mask_and_indices_exact,
            "hidden_padding_content_exact": hidden_padding_content_exact,
            "tensor_bytes_sha256": capture_tensor_sha256,
            "capture_spec_sha256": captured.capture_spec_hash,
        },
        "soft_insertion": {
            "claim": "carrier_and_autograd_connectivity_only",
            "workspace_span": list(soft_batch.workspace_span),
            "positions": 8,
            "logits_finite": True,
            "loss_finite": True,
            "original_tokens_mask_labels_preserved": original_tokens_mask_labels_preserved,
            "positions_sequential_shifted": positions_sequential_shifted,
            "inserted_attention_mask_ones": 8,
            "inserted_loss_mask_negative_100": 8,
            "no_other_loss_masking": loss_mask_exact,
            "workspace_gradient_finite_nonzero": workspace_gradient_ok,
            "gate_gradient_finite_nonzero": gate_gradient_ok,
            "base_parameter_gradients": 0,
        },
        "cache_probe": {
            "initial_sequence_length": initial_cache_length,
            "teacher_forced_token_ids": continuation_ids.flatten().tolist(),
            "decode_steps": cache_steps,
        },
    }


def host_ram_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def verify_execution_tree(args: argparse.Namespace, plan: dict[str, object]) -> None:
    if (
        not args.repo.is_absolute()
        or args.repo.is_symlink()
        or args.repo.resolve() != args.plan.resolve(strict=True).parents[2]
    ):
        raise ValueError("A0 repository path differs from the plan location")
    if len(args.execution_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.execution_commit
    ):
        raise ValueError("A0 execution commit must be an exact lowercase 40-character commit")
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != args.execution_commit:
        raise ValueError("reported A0 execution commit differs from checked-out HEAD")
    worktree_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=args.repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if worktree_status:
        raise ValueError("A0 execution requires a clean worktree")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", plan["mechanism_code_commit"], args.execution_commit],
        cwd=args.repo,
        check=True,
    )
    for relative, expected in plan["asset_sha256"].items():
        path = args.repo / relative
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"A0 executable asset differs from the freeze: {relative}")


def run(
    args: argparse.Namespace,
    plan: dict[str, object],
    bank: dict[str, object],
    stage: dict[str, str],
) -> dict[str, object]:
    if not args.owner_approved:
        raise ValueError("A0 launch requires --owner-approved after root freeze review")
    verify_execution_tree(args, plan)
    stage["name"] = "execution_tree_verified"
    paths = plan["remote_paths"]
    if args.coordinator.resolve() != Path(paths["coordinator_e33"]):
        raise ValueError("coordinator path differs from frozen A0 plan")
    if args.worker.resolve() != Path(paths["worker_h176"]):
        raise ValueError("worker path differs from frozen A0 plan")
    weights = {"coordinator_e33": model_weight(args.coordinator), "worker_h176": model_weight(args.worker)}
    pre_hashes = {name: file_sha256(path) for name, path in weights.items()}
    if pre_hashes != plan["protected_checkpoints"]:
        raise ValueError("protected checkpoint preflight hashes differ from the A0 plan")
    metadata_before = {
        "coordinator_e33": metadata_hashes(args.coordinator),
        "worker_h176": metadata_hashes(args.worker),
    }
    if metadata_before["coordinator_e33"] != plan["runtime"]["checkpoint_metadata_sha256"]:
        raise ValueError("e33 checkpoint metadata differs from the frozen A0 plan")
    stage["name"] = "protected_preflight_verified"
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise ValueError("A0 requires Python 3.12")
    if os.environ.get("UV_PROJECT_ENVIRONMENT") != str(A0_SHARED_ENVIRONMENT):
        raise ValueError("A0 requires the frozen shared project environment")
    if os.environ.get("PYTHONPATH") != str(args.repo / "src"):
        raise ValueError("A0 requires the exact checked-out source path")
    versions = {
        "python": platform.python_version(),
        "transformers": require_version("transformers", "5.6.2"),
        "torch": require_version("torch", "2.11.0"),
    }
    runtime_sources = transformers_source_hashes()
    observed_source_hashes = {name: item["sha256"] for name, item in runtime_sources.items()}
    if observed_source_hashes != plan["runtime"]["transformers_source_sha256"]:
        raise ValueError("Transformers runtime sources differ from the frozen A0 plan")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA RTX A6000":
        raise RuntimeError("A0 cuda:0 must be an NVIDIA RTX A6000")
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    output_root = Path(plan["resource_bounds"]["output_root"])
    free_disk = shutil.disk_usage(output_root).free
    ram = host_ram_bytes()
    if (
        total_gib < plan["resource_bounds"]["minimum_gpu_memory_gib"]
        or free_disk < plan["resource_bounds"]["minimum_free_disk_gib"] * 2**30
        or ram < plan["resource_bounds"]["minimum_host_ram_gib"] * 2**30
    ):
        raise RuntimeError("A0 host falls below a frozen resource bound")

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
    if model.__class__.__name__ != "Qwen3_5ForConditionalGeneration":
        raise TypeError(f"unexpected e33 model class: {model.__class__.__name__}")
    if getattr(model.config.text_config, "hidden_size", None) != 2048:
        raise ValueError("e33 text hidden size differs from 2048")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stage["name"] = "model_loaded_frozen"

    probes = []
    for example in bank["examples"]:
        stage["name"] = f"probe_{example['example_id']}"
        probes.append(probe_example(model, tokenizer, example, plan["mechanism"]))
    if len(probes) != 4 or any(probe["status"] != "complete" for probe in probes):
        raise MechanismRejected("fewer than four A0 probes completed")
    post_hashes = {name: file_sha256(path) for name, path in weights.items()}
    metadata_after = {
        "coordinator_e33": metadata_hashes(args.coordinator),
        "worker_h176": metadata_hashes(args.worker),
    }
    if post_hashes != pre_hashes or metadata_after != metadata_before:
        raise RuntimeError("protected checkpoint weights or metadata changed during A0")
    stage["name"] = "protected_postflight_verified"
    receipt: dict[str, object] = {
        "schema_version": "prime-rl/latent-a0-mechanism-receipt/v1",
        "status": "rendered_transcript_carrier_mechanism_validated",
        "claim": "four-probe carrier/autograd/cache validation; not harness acceptance/timing or model admission",
        "plan_sha256": plan["plan_sha256"],
        "bank_sha256": plan["bank_sha256"],
        "mechanism_code_commit": plan["mechanism_code_commit"],
        "execution_commit": args.execution_commit,
        "asset_sha256": plan["asset_sha256"],
        "versions": versions,
        "transformers_runtime_sources": runtime_sources,
        "gpu": {"name": torch.cuda.get_device_name(0), "total_memory_gib": total_gib},
        "host": {"ram_bytes": ram, "free_disk_bytes_before": free_disk},
        "checkpoint_metadata_before": metadata_before,
        "checkpoint_metadata_after": metadata_after,
        "protected_hashes_before": pre_hashes,
        "protected_hashes_after": post_hashes,
        "complete_probes": 4,
        "probes": probes,
        "resources": {
            "wall_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(0),
        },
        "optimizer_created": False,
        "checkpoint_created": False,
        "artifact_contract": {
            "expected_files": ["receipt.json"],
            "maximum_directory_bytes": plan["resource_bounds"]["maximum_output_directory_bytes"],
        },
        "a1_blocker": "live typed-harness action acceptance and capture timing remain unvalidated",
        "promotion_boundary": plan["promotion_boundary"],
    }
    receipt["receipt_sha256"] = canonical_json_hash(receipt)
    return receipt


def failure_record(
    args: argparse.Namespace,
    error: BaseException,
    stage: str,
    plan: dict[str, object] | None,
) -> dict[str, object]:
    protected: dict[str, object] = {}
    for name, path in (("coordinator_e33", args.coordinator), ("worker_h176", args.worker)):
        try:
            protected[name] = {
                "model_sha256": file_sha256(model_weight(path)),
                "metadata_sha256": metadata_hashes(path),
            }
        except (OSError, ValueError) as hash_error:
            protected[name] = {"hash_probe_error": f"{type(hash_error).__name__}: {hash_error}"}
    failure: dict[str, object] = {
        "schema_version": "prime-rl/latent-a0-mechanism-failure/v1",
        "status": "mechanism_rejected" if isinstance(error, MechanismRejected) else "infrastructure_invalid",
        "failure_category": (
            "mechanism_predicate_failure"
            if isinstance(error, MechanismRejected)
            else "environment_provenance_timeout_or_oom"
        ),
        "error_type": type(error).__name__,
        "error": str(error),
        "plan_path": str(args.plan),
        "bank_path": str(args.bank),
        "mechanism_code_commit": None if plan is None else plan.get("mechanism_code_commit"),
        "plan_sha256": None if plan is None else plan.get("plan_sha256"),
        "execution_commit": args.execution_commit,
        "stage": stage,
        "protected_hash_probe_after_failure": protected,
        "model_update_attempted": False,
    }
    failure["failure_sha256"] = canonical_json_hash(failure)
    return failure


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the no-update e33 latent A0 mechanism probe.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--coordinator", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    writer = ArtifactWriter(args.output_dir, A0_OUTPUT_ROOT)
    stage = {"name": "artifact_namespace_created"}
    plan: dict[str, object] | None = None

    def timeout_handler(_signum, _frame) -> None:
        raise TimeoutError("A0 exceeded its frozen wall-time bound")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm((30 - 2) * 60)
    try:
        plan, bank = load_and_validate_a0_plan(args.plan, args.bank)
        stage["name"] = "plan_and_bank_validated"
        receipt = run(args, plan, bank, stage)
        encoded_size = len((json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
        if encoded_size > plan["resource_bounds"]["maximum_output_bytes"]:
            raise RuntimeError("A0 receipt exceeds the frozen output bound")
        writer.write_json(
            "receipt.json",
            receipt,
            maximum_directory_bytes=plan["resource_bounds"]["maximum_output_directory_bytes"],
        )
    except BaseException as error:
        writer.write_json(
            "failure.json",
            failure_record(args, error, stage["name"], plan),
            maximum_directory_bytes=16 * 1024 * 1024,
        )
        raise
    finally:
        signal.alarm(0)
        writer.close()


if __name__ == "__main__":
    main()
