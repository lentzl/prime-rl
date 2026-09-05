from __future__ import annotations

import argparse
import importlib.util
import math
import signal
from pathlib import Path

import torch

from prime_rl.latent.a0 import canonical_json_hash
from prime_rl.latent.a0dr2 import (
    invoke_after_predecode_snapshot,
    load_and_validate_a0dr2_plan,
    validate_a0dr2_receipt,
)

OUTPUT_ROOT = Path("/home/ubuntu/rlm/outputs/latent-a0dr2-cache-diagnostic-v1")
_BASE_PATH = Path(__file__).with_name("run_a0dr_cache_diagnostic_v1.py")
_SPEC = importlib.util.spec_from_file_location("prime_rl_a0dr_frozen_runner", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load frozen A0DR runner from {_BASE_PATH}")
_base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_base)
_base.OUTPUT_ROOT = OUTPUT_ROOT


def run_arm_branch(
    model,
    arm_name: str,
    representation: dict[str, torch.Tensor | None],
    continuation_ids: torch.Tensor,
    branch: str,
) -> dict[str, object]:
    model.model.rope_deltas = None
    rope_states: dict[str, object] = {"before_prefill": _base.rope_summary(model)}
    attention_mask = representation["attention_mask"]
    position_ids = representation["position_ids"]
    if "input_ids" in representation:
        full_ids = representation["input_ids"]
        full_embeddings = None
        prefill_kwargs = {"input_ids": full_ids, "attention_mask": attention_mask}
        if position_ids is not None:
            prefill_kwargs["position_ids"] = position_ids
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
    initial_length = _base.cache_length(cached.past_key_values)
    expected_initial = attention_mask.shape[1]
    if initial_length != expected_initial or not bool(torch.isfinite(cached.logits).all()):
        raise _base.DiagnosticIncomplete(f"{arm_name}/{branch} prefill cache or logits invalid")
    prefill_cache_type = type(cached.past_key_values).__name__
    prefill_last_logits_sha256 = _base.tensor_sha256(cached.logits[:, -1].float())
    rope_states["after_prefill"] = _base.rope_summary(model)
    steps = []
    for index in range(4):
        token = continuation_ids[:, index : index + 1]
        attention_mask = torch.cat((attention_mask, torch.ones_like(token)), dim=1)
        if full_ids is not None:
            full_ids = torch.cat((full_ids, token), dim=1)
            full_kwargs = {"input_ids": full_ids, "attention_mask": attention_mask}
            if position_ids is not None:
                position_ids = torch.cat((position_ids, position_ids[:, -1:] + 1), dim=1)
                full_kwargs["position_ids"] = position_ids
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
        rope_before_prepare = _base.rope_summary(model)
        prepared = model.prepare_inputs_for_generation(token, **prepare_kwargs)
        if branch == "auto_position" and ("position_ids" in prepared or "cache_position" in prepared):
            raise _base.DiagnosticIncomplete("auto-position branch unexpectedly prepared an explicit position")
        if branch == "explicit_next_position" and not {"position_ids", "cache_position"}.issubset(prepared):
            raise _base.DiagnosticIncomplete("explicit-position branch dropped a frozen position input")
        rope_after_prepare = _base.rope_summary(model)

        def decode_pair():
            with torch.inference_mode():
                full_result = model(**full_kwargs, use_cache=False, return_dict=True)
                cached_result = model(**prepared, return_dict=True)
            return full_result, cached_result

        (full, cached), prepared_evidence = invoke_after_predecode_snapshot(
            prepared,
            summarize=_base.prepared_summary,
            invoke=decode_pair,
        )
        cached_logits = cached.logits[:, -1].float()
        full_logits = full.logits[:, -1].float()
        max_abs = float((cached_logits - full_logits).abs().max().item())
        rms = _base.normalized_rms(cached_logits, full_logits)
        observed_length = _base.cache_length(cached.past_key_values)
        if (
            observed_length != initial_length + index + 1
            or not bool(torch.isfinite(cached_logits).all())
            or not bool(torch.isfinite(full_logits).all())
            or not math.isfinite(max_abs)
            or not math.isfinite(rms)
        ):
            raise _base.DiagnosticIncomplete(f"{arm_name}/{branch} step {index + 1} is incomplete or non-finite")
        steps.append(
            {
                "step": index + 1,
                "cache_sequence_length": observed_length,
                "maximum_absolute_logit_difference": max_abs,
                "normalized_rms": rms,
                "greedy_equal": bool(torch.equal(cached_logits.argmax(-1), full_logits.argmax(-1))),
                "cached_logits_sha256": _base.tensor_sha256(cached_logits),
                "full_logits_sha256": _base.tensor_sha256(full_logits),
                "prepared": prepared_evidence,
                "rope_state": {
                    "before_prepare": rope_before_prepare,
                    "after_prepare": rope_after_prepare,
                    "after_decode": _base.rope_summary(model),
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


def adapt_and_validate_receipt(receipt: dict[str, object], *, plan: dict[str, object]) -> None:
    receipt["schema_version"] = "prime-rl/latent-a0dr2-cache-diagnostic-receipt/v1"
    receipt["supersedes_failed_run"] = plan["supersedes_failed_run"]
    receipt["evidence_capture_timing"] = (
        "immediately_after_prepare_inputs_for_generation_before_full_or_cached_decode"
    )
    receipt["receipt_sha256"] = canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    validate_a0dr2_receipt(receipt, plan=plan)


def failure_record(
    args: argparse.Namespace, error: BaseException, stage: str, plan: dict[str, object] | None
) -> dict[str, object]:
    failure = _base.failure_record(args, error, stage, plan)
    failure["schema_version"] = "prime-rl/latent-a0dr2-cache-diagnostic-failure/v1"
    failure["supersedes_failed_run"] = None if plan is None else plan.get("supersedes_failed_run")
    failure["failure_sha256"] = canonical_json_hash(failure, omitted_fields=("failure_sha256",))
    return failure


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the no-update A0DR2 receipt-instrumentation repair.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--coordinator", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    writer = _base.ArtifactWriter(args.output_dir)
    stage = {"name": "artifact_namespace_created"}
    plan = None

    def timeout_handler(_signum, _frame) -> None:
        raise TimeoutError("A0DR2 exceeded its frozen wall-time bound")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(28 * 60)
    _base.run_arm_branch = run_arm_branch
    _base.validate_a0dr_receipt = adapt_and_validate_receipt
    try:
        plan, bank = load_and_validate_a0dr2_plan(args.plan, args.bank)
        stage["name"] = "plan_bank_and_failed_evidence_validated"
        receipt = _base.run(args, plan, bank, stage)
        writer.write_json("receipt.json", receipt, plan["resource_bounds"]["maximum_output_bytes"])
    except BaseException as error:
        writer.write_json("failure.json", failure_record(args, error, stage["name"], plan), 16 * 1024 * 1024)
        raise
    finally:
        signal.alarm(0)
        writer.close()


if __name__ == "__main__":
    main()
