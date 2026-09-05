from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import signal
import time
from pathlib import Path
from unittest import mock

import torch
import transformers.cache_utils
import transformers.generation.utils
import transformers.models.qwen3_5.modeling_qwen3_5

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a0nc import (
    CacheAllocationDetected,
    DiagnosticIncomplete,
    classify_failure,
    load_plan,
    recursive_subclass_closure,
    validate_receipt,
)
from prime_rl.latent.policy_adapter import compose_receiver_inputs

OUTPUT_ROOT = Path("/home/ubuntu/rlm/outputs/latent-a0-nocache-receiver-v1")
SHARED_ENVIRONMENT = Path("/home/ubuntu/rlm/prime-rl/.venv")
_BASE_PATH = Path(__file__).with_name("run_a0dr_cache_diagnostic_v1.py")
_SPEC = importlib.util.spec_from_file_location("prime_rl_a0dr_frozen_runner", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("frozen A0DR runner unavailable")
_base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_base)
_base.OUTPUT_ROOT = OUTPUT_ROOT


def class_identity(cls: type) -> dict[str, str]:
    module = __import__(cls.__module__, fromlist=["__name__"])
    module_path = Path(module.__file__).resolve(strict=True)
    if not module_path.is_relative_to(SHARED_ENVIRONMENT.resolve(strict=True)):
        raise DiagnosticIncomplete(
            f"cache class source outside frozen environment: {cls.__module__}.{cls.__qualname__}"
        )
    return {
        "fqcn": f"{cls.__module__}.{cls.__qualname__}",
        "module_path": str(module_path),
        "module_sha256": file_sha256(module_path),
        "package": f"transformers=={importlib.metadata.version('transformers')}",
    }


class CacheGuard:
    def __init__(self) -> None:
        self.base = transformers.cache_utils.Cache
        self.initial_classes = recursive_subclass_closure(self.base)
        self.patched_classes: set[type] = set()
        self.stack = contextlib.ExitStack()
        self.negative_control_dynamic_cache_tripped = False
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
                self.negative_control_dynamic_cache_tripped = True
            if not self.negative_control_dynamic_cache_tripped:
                raise DiagnosticIncomplete("cache allocation negative control did not trip")
            self.verify_closure()
        except BaseException:
            self.stack.close()
            self.restored = True
            raise
        return self

    def verify_closure(self) -> None:
        current = recursive_subclass_closure(self.base)
        new_classes = current - self.patched_classes
        if new_classes:
            names = sorted(f"{cls.__module__}.{cls.__qualname__}" for cls in new_classes)
            raise DiagnosticIncomplete(f"new unpatched cache subclasses loaded: {names}")
        self.checks += 1

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.verify_closure()
        finally:
            self.stack.close()
            self.restored = True
        return False

    def evidence(self, expected_probe_checks: int) -> dict[str, object]:
        return {
            "classes": [
                class_identity(cls)
                for cls in sorted(self.initial_classes, key=lambda item: (item.__module__, item.__qualname__))
            ],
            "negative_control_dynamic_cache_tripped": self.negative_control_dynamic_cache_tripped,
            "closure_rechecked_after_each_probe_and_finally": self.checks >= expected_probe_checks + 2,
            "closure_check_count": self.checks,
            "restored_in_finally": self.restored,
        }


def no_cache_forward(
    model, *, input_ids=None, inputs_embeds=None, attention_mask, position_ids, call_log, arm, **kwargs
):
    if (input_ids is None) == (inputs_embeds is None):
        raise DiagnosticIncomplete("exactly one receiver input form is required")
    observed = {
        "arm": arm,
        "input_ids_is_none": input_ids is None,
        "inputs_embeds_is_none": inputs_embeds is None,
        "use_cache": False,
        "past_key_values_input_is_none": True,
        "past_key_values_output_is_none": False,
    }
    call_log.append(observed)
    output = model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        return_dict=True,
        **kwargs,
    )
    if getattr(output, "past_key_values", None) is not None:
        raise CacheAllocationDetected("no-cache forward returned past_key_values")
    observed["past_key_values_output_is_none"] = True
    return output


def render_probe(
    model,
    tokenizer,
    example: dict[str, object],
    continuation_text: str,
    continuation_ids: list[int],
    call_log: list[dict[str, object]],
):
    parent_ids = _base.render_ids(tokenizer, example["parent_messages"], generation_prompt=False)
    child_prefix = _base.render_ids(tokenizer, example["child_messages"], generation_prompt=False)
    child_ids = _base.render_ids(tokenizer, example["child_messages"], generation_prompt=True)
    observed_continuation = (
        tokenizer(continuation_text, add_special_tokens=False, return_tensors="pt").input_ids[:, :4].to("cuda:0")
    )
    if observed_continuation.flatten().tolist() != continuation_ids or not torch.equal(
        child_ids[:, : child_prefix.shape[1]], child_prefix
    ):
        raise DiagnosticIncomplete("A0NC exact continuation or child rendering changed")
    fixed_ids = torch.tensor([[40, 4021, 2528, 8976, 35139, 635, 524, 599]], device="cuda:0")
    injection = child_prefix.shape[1]
    length_ids = torch.cat((child_ids[:, :injection], fixed_ids, child_ids[:, injection:]), dim=1)
    positions = torch.arange(length_ids.shape[1], device="cuda:0").unsqueeze(0)
    mask = torch.ones_like(length_ids)
    child_embeddings = model.get_input_embeddings()(child_ids)
    length_embeddings = model.get_input_embeddings()(length_ids)
    with torch.inference_mode():
        parent = no_cache_forward(
            model,
            input_ids=parent_ids,
            attention_mask=torch.ones_like(parent_ids),
            position_ids=torch.arange(parent_ids.shape[1], device="cuda:0").unsqueeze(0),
            call_log=call_log,
            arm="PARENT_CAPTURE",
            output_hidden_states=True,
        )
    source = parent.hidden_states[-1][:, -8:, :].detach()
    scale = child_embeddings.detach().float().norm(dim=-1).mean()
    workspace = source / source.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(source.dtype)
    workspace = workspace * scale.to(source.dtype)
    soft = compose_receiver_inputs(
        child_embeddings,
        torch.ones_like(child_ids),
        workspace,
        injection_index=injection,
        gate=0.125,
        position_ids=torch.arange(child_ids.shape[1], device="cuda:0").unsqueeze(0),
    )
    if not torch.equal(mask, soft.attention_mask) or not torch.equal(positions, soft.position_ids):
        raise DiagnosticIncomplete("A0NC matched geometry changed")
    soft_span = soft.inputs_embeds[:, injection : injection + 8]
    hard_span = length_embeddings[:, injection : injection + 8]
    outside_exact = torch.equal(soft.inputs_embeds[:, :injection], child_embeddings[:, :injection]) and torch.equal(
        soft.inputs_embeds[:, injection + 8 :], child_embeddings[:, injection:]
    )
    if torch.count_nonzero(soft_span).item() == 0 or torch.equal(soft_span, hard_span) or not outside_exact:
        raise DiagnosticIncomplete("A0NC soft-span fixture activity changed")
    return (
        length_ids,
        length_embeddings,
        soft.inputs_embeds,
        mask,
        positions,
        observed_continuation,
        {
            "parent_token_count": parent_ids.shape[1],
            "child_token_count": child_ids.shape[1],
            "injection_index": injection,
            "matched_prompt_length": length_ids.shape[1],
            "child_ids_sha256": _base.tensor_sha256(child_ids),
            "length_ids_sha256": _base.tensor_sha256(length_ids),
            "soft_prompt_sha256": _base.tensor_sha256(soft.inputs_embeds),
            "soft_span_start": injection,
            "soft_span_end_exclusive": injection + 8,
            "soft_span_active": bool(torch.count_nonzero(soft_span).item() > 0),
            "soft_span_differs_from_hard_span": not torch.equal(soft_span, hard_span),
            "outside_soft_span_exact": outside_exact,
            "mask_positions_exact": True,
            "soft_used_inputs_embeds_without_input_ids": False,
        },
    )


def run_probe(model, tokenizer, example, continuation_text, continuation_ids, call_log):
    call_start = len(call_log)
    ids, exact, soft, mask, positions, continuation, fixture = render_probe(
        model, tokenizer, example, continuation_text, continuation_ids, call_log
    )
    steps = []
    past_outputs = 0
    for index in range(4):
        expected_prefix = continuation_ids[:index]
        expected_length = fixture["matched_prompt_length"] + index
        if ids.shape[1] != expected_length or mask.shape[1] != expected_length or positions.shape[1] != expected_length:
            raise DiagnosticIncomplete("A0NC full-prefix geometry changed")
        if expected_prefix and ids[:, -index:].flatten().tolist() != expected_prefix:
            raise DiagnosticIncomplete("A0NC continuation prefix changed")
        soft_before = _base.tensor_sha256(soft)
        model.model.rope_deltas = None
        with torch.inference_mode():
            l_id = no_cache_forward(
                model,
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                call_log=call_log,
                arm="L_ID",
            )
        model.model.rope_deltas = None
        with torch.inference_mode():
            l_e = no_cache_forward(
                model,
                inputs_embeds=exact,
                attention_mask=mask,
                position_ids=positions,
                call_log=call_log,
                arm="L_E",
            )
        model.model.rope_deltas = None
        with torch.inference_mode():
            s1 = no_cache_forward(
                model,
                inputs_embeds=soft,
                attention_mask=mask,
                position_ids=positions,
                call_log=call_log,
                arm="S",
            )
        model.model.rope_deltas = None
        with torch.inference_mode():
            s2 = no_cache_forward(
                model,
                inputs_embeds=soft,
                attention_mask=mask,
                position_ids=positions,
                call_log=call_log,
                arm="S_REPEAT",
            )
        outputs = (l_id, l_e, s1, s2)
        past_outputs += sum(getattr(output, "past_key_values", None) is not None for output in outputs)
        id_logits, e_logits, soft_logits, soft_repeat = (output.logits[:, -1].float() for output in outputs)
        if not all(torch.isfinite(logits).all() for logits in (id_logits, e_logits, soft_logits, soft_repeat)):
            raise DiagnosticIncomplete("A0NC nonfinite diagnostic logits")
        if not torch.equal(id_logits, e_logits) or not torch.equal(soft_logits, soft_repeat):
            raise DiagnosticIncomplete("A0NC deterministic interface parity changed")
        steps.append(
            {
                "step": index + 1,
                "continuation_token_id": continuation_ids[index],
                "l_id_l_e_bitwise_equal": bool(torch.equal(id_logits, e_logits)),
                "l_id_l_e_finite": bool(torch.isfinite(id_logits).all() and torch.isfinite(e_logits).all()),
                "soft_finite": bool(torch.isfinite(soft_logits).all()),
                "soft_repeat_bitwise_equal": bool(torch.equal(soft_logits, soft_repeat)),
                "l_id_logits_sha256": _base.tensor_sha256(id_logits),
                "l_e_logits_sha256": _base.tensor_sha256(e_logits),
                "soft_logits_sha256": _base.tensor_sha256(soft_logits),
                "soft_repeat_logits_sha256": _base.tensor_sha256(soft_repeat),
                "prefix_length": expected_length,
                "attention_mask_exact_all_visible": bool(torch.equal(mask, torch.ones_like(mask))),
                "position_ids_exact_sequential": bool(
                    torch.equal(positions, torch.arange(expected_length, device="cuda:0").unsqueeze(0))
                ),
                "attention_mask_sha256": _base.tensor_sha256(mask),
                "position_ids_sha256": _base.tensor_sha256(positions),
                "continuation_prefix_token_ids": expected_prefix,
                "continuation_prefix_sha256": canonical_json_hash(expected_prefix),
                "l_id_prefix_sha256": _base.tensor_sha256(ids),
                "l_e_prefix_sha256": _base.tensor_sha256(exact),
                "soft_prefix_sha256": soft_before,
                "soft_repeat_input_sha256": _base.tensor_sha256(soft),
            }
        )
        token = continuation[:, index : index + 1]
        ids = torch.cat((ids, token), dim=1)
        token_embedding = model.get_input_embeddings()(token)
        exact = torch.cat((exact, token_embedding), dim=1)
        soft = torch.cat((soft, token_embedding), dim=1)
        mask = torch.cat((mask, torch.ones_like(token)), dim=1)
        positions = torch.cat((positions, positions[:, -1:] + 1), dim=1)
    probe_calls = call_log[call_start:]
    fixture["soft_used_inputs_embeds_without_input_ids"] = (
        all(
            entry["input_ids_is_none"] and not entry["inputs_embeds_is_none"]
            for entry in probe_calls
            if entry["arm"] in {"S", "S_REPEAT"}
        )
        and sum(entry["arm"] in {"S", "S_REPEAT"} for entry in probe_calls) == 8
    )
    return {
        "example_id": example["example_id"],
        "complete": True,
        "cache_allocations": 0,
        "past_key_values_outputs": past_outputs,
        "fixture": fixture,
        "steps": steps,
    }


def runtime_disjointness(tokenizer, bank_path: Path, fresh_bank: dict[str, object]) -> dict[str, object]:
    prior_bank_path = bank_path.with_name("a0-mechanism-bank-v1.json")
    reference_path = bank_path.with_name("a0-nocache-disjointness-v1.json")
    prior_bank = json.loads(prior_bank_path.read_text())
    rendered: dict[str, str] = {}
    for label, payload in (("prior", prior_bank), ("fresh", fresh_bank)):
        for example in payload["examples"]:
            parent = _base.render_ids(tokenizer, example["parent_messages"], generation_prompt=False)
            child = _base.render_ids(tokenizer, example["child_messages"], generation_prompt=True)
            rendered[f"{label}:{example['example_id']}:parent"] = _base.tensor_sha256(parent)
            rendered[f"{label}:{example['example_id']}:child"] = _base.tensor_sha256(child)
    all_unique = len(rendered) == 16 and len(set(rendered.values())) == 16
    if not all_unique:
        raise DiagnosticIncomplete("A0NC rendered prior/fresh token tensors overlap")
    return {
        "reference_sha256": file_sha256(reference_path),
        "prior_bank_sha256": file_sha256(prior_bank_path),
        "fresh_bank_sha256": file_sha256(bank_path),
        "rendered_token_sha256": rendered,
        "all_parent_child_token_hashes_unique": all_unique,
    }


def run(args, plan, bank, stage):
    if not args.owner_approved:
        raise ValueError("A0NC requires root approval after immutable review")
    _base.verify_execution_tree(args, plan)
    if os.environ.get("UV_PROJECT_ENVIRONMENT") != str(SHARED_ENVIRONMENT) or os.environ.get("PYTHONPATH") != str(
        args.repo / "src"
    ):
        raise ValueError("A0NC environment changed")
    versions = {
        "python": _base.platform.python_version(),
        "transformers": _base.require_version("transformers", plan["runtime"]["transformers"]),
        "torch_distribution": _base.require_version("torch", plan["runtime"]["torch_distribution"]),
        "torch_runtime": str(torch.__version__),
    }
    sources = _base.source_hashes()
    if (
        versions["torch_runtime"] != plan["runtime"]["torch_runtime"]
        or {name: value["sha256"] for name, value in sources.items()} != plan["runtime"]["transformers_source_sha256"]
        or args.coordinator.resolve() != Path(plan["remote_paths"]["coordinator_e33"])
        or args.worker.resolve() != Path(plan["remote_paths"]["worker_h176"])
    ):
        raise ValueError("A0NC runtime sources or protected paths changed")
    weights = {"coordinator_e33": _base.model_weight(args.coordinator), "worker_h176": _base.model_weight(args.worker)}
    before = {name: file_sha256(path) for name, path in weights.items()}
    metadata_before = {
        "coordinator_e33": _base.metadata_hashes(args.coordinator),
        "worker_h176": _base.metadata_hashes(args.worker),
    }
    expected_metadata = plan["runtime"]["checkpoint_metadata_sha256"]
    if before != plan["protected_checkpoints"] or any(value != expected_metadata for value in metadata_before.values()):
        raise ValueError("A0NC protected/runtime preflight changed")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != plan["resource_bounds"]["gpu_model"]:
        raise RuntimeError("A0NC GPU changed")
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    free_disk = shutil.disk_usage(plan["resource_bounds"]["output_root"]).free
    ram = _base.host_ram_bytes()
    if (
        total_gib < plan["resource_bounds"]["minimum_gpu_memory_gib"]
        or free_disk < plan["resource_bounds"]["minimum_free_disk_gib"] * 2**30
        or ram < plan["resource_bounds"]["minimum_host_ram_gib"] * 2**30
    ):
        raise RuntimeError("A0NC host resources below freeze")
    torch.manual_seed(20260905)
    torch.cuda.manual_seed_all(20260905)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    tokenizer = _base.AutoTokenizer.from_pretrained(args.coordinator, local_files_only=True)
    disjointness = runtime_disjointness(tokenizer, args.bank, bank)
    model = _base.AutoModelForImageTextToText.from_pretrained(
        args.coordinator, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda:0")
    model.eval()
    if (
        model.__class__.__name__ != plan["runtime"]["model_class"]
        or model.config.text_config.hidden_size != plan["runtime"]["hidden_size"]
        or str(next(model.parameters()).device) != plan["runtime"]["device"]
        or str(next(model.parameters()).dtype).removeprefix("torch.") != plan["runtime"]["dtype"]
    ):
        raise TypeError("A0NC model runtime changed")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stage["name"] = "model_loaded_frozen"
    cache_guard = CacheGuard()
    call_log: list[dict[str, object]] = []
    with cache_guard:
        probes = []
        for example in bank["examples"]:
            probes.append(
                run_probe(
                    model,
                    tokenizer,
                    example,
                    bank["continuation_text"],
                    bank["continuation_token_ids"],
                    call_log,
                )
            )
            cache_guard.verify_closure()
    cache_guard_evidence = cache_guard.evidence(expected_probe_checks=len(probes))
    after = {name: file_sha256(path) for name, path in weights.items()}
    metadata_after = {
        "coordinator_e33": _base.metadata_hashes(args.coordinator),
        "worker_h176": _base.metadata_hashes(args.worker),
    }
    qualifies = all(
        probe["past_key_values_outputs"] == 0
        and all(
            probe["fixture"][field]
            for field in (
                "soft_span_active",
                "soft_span_differs_from_hard_span",
                "outside_soft_span_exact",
                "mask_positions_exact",
                "soft_used_inputs_embeds_without_input_ids",
            )
        )
        and all(
            step["l_id_l_e_bitwise_equal"]
            and step["l_id_l_e_finite"]
            and step["soft_finite"]
            and step["soft_repeat_bitwise_equal"]
            and step["attention_mask_exact_all_visible"]
            and step["position_ids_exact_sequential"]
            for step in probe["steps"]
        )
        for probe in probes
    )
    if not qualifies:
        raise DiagnosticIncomplete("A0NC diagnostic fixture or qualification contract failed")
    receipt = {
        "schema_version": "prime-rl/latent-a0-nocache-receipt/v1",
        "status": "nocache_receiver_mechanism_validated",
        "plan_sha256": plan["plan_sha256"],
        "bank_sha256": plan["bank_sha256"],
        "mechanism_code_commit": plan["mechanism_code_commit"],
        "execution_commit": args.execution_commit,
        "asset_sha256": plan["asset_sha256"],
        "versions": versions,
        "transformers_runtime_sources": sources,
        "model_runtime": {
            "class": model.__class__.__name__,
            "hidden_size": model.config.text_config.hidden_size,
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        },
        "gpu": {"name": torch.cuda.get_device_name(0), "total_memory_gib": total_gib},
        "host": {"ram_bytes": ram, "free_disk_bytes_before": free_disk},
        "cache_guard": cache_guard_evidence,
        "no_cache_call_contract": {
            "use_cache_false_every_call": all(entry["use_cache"] is False for entry in call_log),
            "past_key_values_input_none_every_call": all(
                entry["past_key_values_input_is_none"] is True for entry in call_log
            ),
            "past_key_values_output_none_every_call": all(
                entry["past_key_values_output_is_none"] is True for entry in call_log
            ),
            "generate_used": False,
            "prepare_inputs_for_generation_used": False,
            "cached_decode_used": False,
            "feedback_used": False,
            "observed_forward_calls": len(call_log),
            "observed_soft_inputs_embeds_calls": sum(entry["arm"] in {"S", "S_REPEAT"} for entry in call_log),
        },
        "disjointness": disjointness,
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "checkpoint_metadata_before": metadata_before,
        "checkpoint_metadata_after": metadata_after,
        "probes": probes,
        "complete_distinct_probes": len(probes),
        "prior_cache_rejection": plan["prior_cache_rejection"],
        "claim": plan["mechanism"]["claim"],
        "optimizer_created": False,
        "checkpoint_created": False,
        "model_update_attempted": False,
        "tensor_persistence": False,
        "resources": {
            "wall_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(0),
        },
        "interpretation_boundary": plan["interpretation_boundary"],
        "receipt_sha256": "",
    }
    if after != before or metadata_after != metadata_before:
        raise RuntimeError("A0NC protected checkpoints changed")
    receipt["receipt_sha256"] = canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    try:
        validate_receipt(receipt, plan=plan)
    except ValueError as error:
        raise DiagnosticIncomplete(f"A0NC receipt contract failed: {error}") from error
    return receipt


def failure_record(args, error: BaseException, stage: str, plan: dict[str, object] | None) -> dict[str, object]:
    failure = _base.failure_record(args, error, stage, plan)
    failure["schema_version"] = "prime-rl/latent-a0-nocache-failure/v1"
    failure["status"], failure["failure_category"] = classify_failure(error)
    failure["optimizer_created"] = False
    failure["checkpoint_created"] = False
    failure["tensor_persistence"] = False
    failure["failure_sha256"] = canonical_json_hash(failure, omitted_fields=("failure_sha256",))
    return failure


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "bank", "coordinator", "worker", "repo", "output_dir"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    writer = _base.ArtifactWriter(args.output_dir)
    stage = {"name": "artifact_namespace_created"}
    plan = None
    signal.signal(signal.SIGALRM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError("A0NC timeout")))
    signal.alarm(28 * 60)
    try:
        plan, bank = load_plan(args.plan, args.bank)
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
