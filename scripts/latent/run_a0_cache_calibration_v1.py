from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import signal
import time
from pathlib import Path

import torch

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a0cal import calculate_probe_criterion, load_and_validate_a0cal_plan, validate_a0cal_receipt
from prime_rl.latent.policy_adapter import compose_receiver_inputs

OUTPUT_ROOT = Path("/home/ubuntu/rlm/outputs/latent-a0-cache-calibration-v1")
SHARED_ENVIRONMENT = Path("/home/ubuntu/rlm/prime-rl/.venv")
_RUNNER_PATH = Path(__file__).with_name("run_a0dr2_cache_diagnostic_v1.py")
_SPEC = importlib.util.spec_from_file_location("prime_rl_a0dr2_frozen_runner", _RUNNER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load frozen A0DR2 runner from {_RUNNER_PATH}")
_a0dr2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_a0dr2)
_base = _a0dr2._base
_base.OUTPUT_ROOT = OUTPUT_ROOT

_EXPECTED = {
    "a0-mechanism-0001": {"parent": 93, "child": 47, "injection": 40, "matched": 55},
    "a0-mechanism-0002": {"parent": 84, "child": 42, "injection": 35, "matched": 50},
    "a0-mechanism-0003": {"parent": 77, "child": 35, "injection": 28, "matched": 43},
    "a0-mechanism-0004": {"parent": 74, "child": 40, "injection": 33, "matched": 48},
}
_FIXED_IDS = [40, 4021, 2528, 8976, 35139, 635, 524, 599]
_FIXED_IDS_SHA256 = "e86e01e61315008783cc217a5bb83a1b3aced0daaecbc920b8d3b45ab4b205d8"
_CONTINUATION_SHA256 = "d2a9291c35fc42fadedff20c365f38da2813504f980dd6ba6bdda413a79bd6e0"


def build_probe(
    model, tokenizer, example: dict[str, object]
) -> tuple[dict[str, object], torch.Tensor, dict[str, object]]:
    expected = _EXPECTED[example["example_id"]]
    parent_ids = _base.render_ids(tokenizer, example["parent_messages"], generation_prompt=False)
    child_prefix_ids = _base.render_ids(tokenizer, example["child_messages"], generation_prompt=False)
    child_ids = _base.render_ids(tokenizer, example["child_messages"], generation_prompt=True)
    if hashlib.sha256(example["continuation_text"].encode()).hexdigest() != _CONTINUATION_SHA256:
        raise _base.DiagnosticIncomplete("frozen continuation changed")
    continuation = (
        tokenizer(example["continuation_text"], add_special_tokens=False, return_tensors="pt")
        .input_ids[:, :4]
        .to("cuda:0")
    )
    fixed_ids = torch.tensor([_FIXED_IDS], dtype=torch.long, device="cuda:0")
    injection = child_prefix_ids.shape[1]
    if (
        continuation.shape[1] != 4
        or parent_ids.shape[1] != expected["parent"]
        or child_ids.shape[1] != expected["child"]
        or injection != expected["injection"]
        or not torch.equal(child_ids[:, :injection], child_prefix_ids)
        or _base.tensor_sha256(fixed_ids) != _FIXED_IDS_SHA256
        or any(token in set(tokenizer.all_special_ids) for token in _FIXED_IDS)
    ):
        raise _base.DiagnosticIncomplete("frozen probe rendering or length-control identity changed")
    length_ids = torch.cat((child_ids[:, :injection], fixed_ids, child_ids[:, injection:]), dim=1)
    mask = torch.ones_like(child_ids)
    length_mask = torch.ones_like(length_ids)
    positions = torch.arange(child_ids.shape[1], device="cuda:0").unsqueeze(0)
    length_positions = torch.arange(length_ids.shape[1], device="cuda:0").unsqueeze(0)
    exact_embeddings = model.get_input_embeddings()(child_ids)
    length_embeddings = model.get_input_embeddings()(length_ids)
    with torch.inference_mode():
        parent = model(
            input_ids=parent_ids,
            attention_mask=torch.ones_like(parent_ids),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    if parent.hidden_states is None or not bool(torch.isfinite(parent.hidden_states[-1]).all()):
        raise _base.DiagnosticIncomplete("parent hidden state absent or non-finite")
    source = parent.hidden_states[-1][:, -8:, :].detach()
    embedding_norm = exact_embeddings.detach().float().norm(dim=-1).mean()
    workspace = source / source.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(source.dtype)
    workspace = workspace * embedding_norm.to(source.dtype)
    soft = compose_receiver_inputs(
        exact_embeddings,
        mask,
        workspace,
        injection_index=injection,
        gate=0.125,
        position_ids=positions,
    )
    if (
        length_ids.shape[1] != expected["matched"]
        or soft.inputs_embeds.shape[1] != expected["matched"]
        or not torch.equal(length_mask, soft.attention_mask)
        or not torch.equal(length_positions, soft.position_ids)
    ):
        raise _base.DiagnosticIncomplete("matched discrete/soft geometry changed")
    representations = {
        "L_ID": {"input_ids": length_ids, "attention_mask": length_mask, "position_ids": length_positions},
        "L_E": {
            "inputs_embeds": length_embeddings,
            "attention_mask": length_mask,
            "position_ids": length_positions,
        },
        "S": {
            "inputs_embeds": soft.inputs_embeds,
            "attention_mask": soft.attention_mask,
            "position_ids": soft.position_ids,
        },
    }
    fixture = {
        "parent_token_count": parent_ids.shape[1],
        "child_token_count": child_ids.shape[1],
        "injection_index": injection,
        "matched_prompt_length": length_ids.shape[1],
        "length_control_token_ids": _FIXED_IDS,
        "length_control_token_ids_sha256": _base.tensor_sha256(fixed_ids),
        "length_control_tokens_non_special": True,
        "matched_mask_and_positions_exact": True,
        "child_input_ids_sha256": _base.tensor_sha256(child_ids),
        "length_control_input_ids_sha256": _base.tensor_sha256(length_ids),
        "continuation_input_ids_sha256": _base.tensor_sha256(continuation),
        "continuation_token_ids": continuation.flatten().tolist(),
        "workspace_source_sha256": _base.tensor_sha256(source),
        "soft_prompt_sha256": _base.tensor_sha256(soft.inputs_embeds),
    }
    return representations, continuation, fixture


def run(
    args: argparse.Namespace, plan: dict[str, object], bank: dict[str, object], stage: dict[str, str]
) -> dict[str, object]:
    if not args.owner_approved:
        raise ValueError("A0 cache calibration requires root/evaluator approval")
    _base.verify_execution_tree(args, plan)
    stage["name"] = "execution_tree_verified"
    if _base.platform.python_version_tuple()[:2] != ("3", "12"):
        raise ValueError("A0 cache calibration requires Python 3.12")
    if os.environ.get("UV_PROJECT_ENVIRONMENT") != str(SHARED_ENVIRONMENT) or os.environ.get("PYTHONPATH") != str(
        args.repo / "src"
    ):
        raise ValueError("A0 cache-calibration environment changed")
    versions = {
        "python": _base.platform.python_version(),
        "transformers": _base.require_version("transformers", plan["runtime"]["transformers"]),
        "torch_distribution": _base.require_version("torch", plan["runtime"]["torch_distribution"]),
        "torch_runtime": str(torch.__version__),
    }
    if versions["torch_runtime"] != plan["runtime"]["torch_runtime"]:
        raise ValueError("A0 cache-calibration torch runtime changed")
    sources = _base.source_hashes()
    if {name: value["sha256"] for name, value in sources.items()} != plan["runtime"]["transformers_source_sha256"]:
        raise ValueError("A0 cache-calibration transformers sources changed")
    if args.coordinator.resolve() != Path(plan["remote_paths"]["coordinator_e33"]) or args.worker.resolve() != Path(
        plan["remote_paths"]["worker_h176"]
    ):
        raise ValueError("A0 cache-calibration protected paths changed")
    weights = {
        "coordinator_e33": _base.model_weight(args.coordinator),
        "worker_h176": _base.model_weight(args.worker),
    }
    hashes_before = {name: file_sha256(weight) for name, weight in weights.items()}
    metadata_before = {
        "coordinator_e33": _base.metadata_hashes(args.coordinator),
        "worker_h176": _base.metadata_hashes(args.worker),
    }
    if hashes_before != plan["protected_checkpoints"] or any(
        value != plan["runtime"]["checkpoint_metadata_sha256"] for value in metadata_before.values()
    ):
        raise ValueError("A0 cache-calibration protected preflight changed")
    stage["name"] = "protected_preflight_verified"
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != plan["resource_bounds"]["gpu_model"]:
        raise RuntimeError("A0 cache-calibration GPU changed")
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    free_disk = shutil.disk_usage(plan["resource_bounds"]["output_root"]).free
    ram = _base.host_ram_bytes()
    if (
        total_gib < plan["resource_bounds"]["minimum_gpu_memory_gib"]
        or free_disk < plan["resource_bounds"]["minimum_free_disk_gib"] * 2**30
        or ram < plan["resource_bounds"]["minimum_host_ram_gib"] * 2**30
    ):
        raise RuntimeError("A0 cache-calibration host below frozen bounds")
    torch.manual_seed(20260905)
    torch.cuda.manual_seed_all(20260905)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    tokenizer = _base.AutoTokenizer.from_pretrained(args.coordinator, local_files_only=True)
    model = _base.AutoModelForImageTextToText.from_pretrained(
        args.coordinator,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda:0")
    model.eval()
    if (
        model.__class__.__name__ != plan["runtime"]["model_class"]
        or model.config.text_config.hidden_size != plan["runtime"]["hidden_size"]
        or str(next(model.parameters()).device) != plan["runtime"]["device"]
        or str(next(model.parameters()).dtype).removeprefix("torch.") != plan["runtime"]["dtype"]
    ):
        raise TypeError("A0 cache-calibration model runtime changed")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stage["name"] = "model_loaded_frozen"
    by_id = {example["example_id"]: example for example in bank["examples"]}
    probes = []
    for example_id in plan["criterion"]["required_probe_ids"]:
        stage["name"] = f"probe_{example_id}"
        representations, continuation, fixture = build_probe(model, tokenizer, by_id[example_id])
        arms = [
            _a0dr2.run_arm_branch(model, name, representations[name], continuation, "explicit_next_position")
            for name in ("L_ID", "L_E", "S")
        ]
        arm_map = {arm["arm"]: arm for arm in arms}
        probes.append(
            {
                "example_id": example_id,
                "fixture": fixture,
                "arms": arms,
                "criterion_result": calculate_probe_criterion(arm_map),
            }
        )
    hashes_after = {name: file_sha256(weight) for name, weight in weights.items()}
    metadata_after = {
        "coordinator_e33": _base.metadata_hashes(args.coordinator),
        "worker_h176": _base.metadata_hashes(args.worker),
    }
    if hashes_after != hashes_before or metadata_after != metadata_before:
        raise RuntimeError("A0 cache-calibration protected checkpoints changed")
    qualified = all(probe["criterion_result"]["qualifies"] for probe in probes)
    receipt = {
        "schema_version": "prime-rl/latent-a0-cache-calibration-receipt/v1",
        "status": "relative_cache_calibrated" if qualified else "relative_cache_rejected",
        "claim": "relative soft-vs-discrete cache calibration only",
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
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
        "checkpoint_metadata_before": metadata_before,
        "checkpoint_metadata_after": metadata_after,
        "criterion": plan["criterion"],
        "probes": probes,
        "complete_distinct_probes": len(probes),
        "optimizer_created": False,
        "checkpoint_created": False,
        "model_update_attempted": False,
        "resources": {
            "wall_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(0),
        },
        "interpretation_boundary": plan["interpretation_boundary"],
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    validate_a0cal_receipt(receipt, plan=plan)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prospective four-probe A0 relative cache calibration.")
    for name in ("plan", "bank", "coordinator", "worker", "repo", "output_dir"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    writer = _base.ArtifactWriter(args.output_dir)
    stage = {"name": "artifact_namespace_created"}
    plan = None

    def timeout_handler(_signum, _frame) -> None:
        raise TimeoutError("A0 cache calibration exceeded frozen wall time")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(28 * 60)
    try:
        plan, bank = load_and_validate_a0cal_plan(args.plan, args.bank)
        stage["name"] = "plan_bank_and_evidence_validated"
        receipt = run(args, plan, bank, stage)
        writer.write_json("receipt.json", receipt, plan["resource_bounds"]["maximum_output_bytes"])
    except BaseException as error:
        writer.write_json("failure.json", _a0dr2.failure_record(args, error, stage["name"], plan), 16 * 1024 * 1024)
        raise
    finally:
        signal.alarm(0)
        writer.close()


if __name__ == "__main__":
    main()
