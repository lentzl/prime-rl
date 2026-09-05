#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
from pathlib import Path
from unittest import mock

import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a1cap768 import SELECTION
from prime_rl.latent.a1nc0 import tensor_bytes_sha256, validate_bank_artifact

OUTPUT_ROOT = Path("/home/ubuntu/rlm/outputs/latent-a1-nc0-cap768-r3-render-proof-v1")
RUN_ID = "a1-nc0-cap768-r3-render-proof-run2"
EXPECTED_CAP_RUNNER_SHA = "5de4f57deb324451efd1e3c11576c046fc9bfd31b4f3a3218bd5a90c963bf061"
EXPECTED_BASE_RUNNER_SHA = "3ad4949d70edc467e30eeb2b512292a09dfa5d66f411253bf31045d6047034d9"
EXPECTED_RUNTIME = {
    "python": "3.12.14",
    "transformers": "5.6.2",
    "flash_linear_attention": "0.5.2",
    "torch_distribution": "2.11.0+cu128",
    "torch_runtime": "2.11.0+cu128",
}
EXPECTED_METADATA = {
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "config.json": "22949388ed61c1100b20a3cae55bb22122554c74e06fc23f1be50cca1fec3b8c",
    "generation_config.json": "93f19a5ed0fb9f9e8e65dafae7a9bc4c6a32b3e37f6278980d05d3f4ca29f17b",
    "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "747ba36a06ba5428bb74e984d75136b37cf5dafe97b8dd315f701b361a9f417f",
}


def _load_cap(repo: Path):
    path = repo / "scripts/latent/run_a1_nc0_cap768_v1.py"
    if file_sha256(path) != EXPECTED_CAP_RUNNER_SHA:
        raise RuntimeError("R3 proof CAP runner identity changed")
    spec = importlib.util.spec_from_file_location("a1_nc0_cap768_r3_render_proof", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("R3 CAP operational render module unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    payload["receipt_sha256"] = canonical_json_hash(payload, omitted_fields=("receipt_sha256",))
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except BaseException:
            pass
        raise


def _selected_cases(artifacts: dict[str, dict[str, object]]):
    cases = []
    for selection in SELECTION:
        split = selection["evidence_id"].split("-", 1)[0]
        records = artifacts[split]["bank"]["records"]
        record = next(item for item in records if item["evidence_id"] == selection["evidence_id"])
        query = next(item for item in record["queries"] if item["query_id"] == selection["query_id"])
        cases.append((selection, record, query))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--coordinator", type=Path, required=True)
    parser.add_argument("--train-bank", type=Path, required=True)
    parser.add_argument("--held-out-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    args = parser.parse_args()
    if (
        args.output_dir.parent != OUTPUT_ROOT
        or args.output_dir.name != RUN_ID
        or args.output_dir.is_symlink()
        or not args.output_dir.is_dir()
        or [path.name for path in args.output_dir.iterdir()] != ["proof.log"]
        or (args.output_dir / "proof.log").stat().st_size != 0
    ):
        raise RuntimeError("R3 proof namespace changed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise RuntimeError("R3 proof requires CUDA hidden and uninitialized")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("R3 proof requires offline mode")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=args.repo, text=True
    )
    if head != args.execution_commit or dirty or len(head) != 40:
        raise RuntimeError("R3 proof execution tree changed")
    runtime = {
        "python": platform.python_version(),
        "transformers": importlib.metadata.version("transformers"),
        "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
        "torch_distribution": importlib.metadata.version("torch"),
        "torch_runtime": str(torch.__version__),
    }
    if runtime != EXPECTED_RUNTIME:
        raise RuntimeError("R3 proof runtime changed")
    cap, cap_path = _load_cap(args.repo)
    base = cap.base
    base_path = Path(base.__file__).resolve()
    expected_base_path = (args.repo / "scripts/latent/run_a1_nc0_nomination_v1.py").resolve()
    if (
        base_path != expected_base_path
        or file_sha256(base_path) != EXPECTED_BASE_RUNNER_SHA
        or cap.validate_bank_artifact is not validate_bank_artifact
        or validate_bank_artifact.__module__ != "prime_rl.latent.a1nc0"
        or cap.base.operational_template_input_ids is not base.operational_template_input_ids
    ):
        raise RuntimeError("R3 CAP/base/validator import identity changed")
    if base.metadata_hashes(args.coordinator) != EXPECTED_METADATA:
        raise RuntimeError("R3 proof tokenizer identity changed")
    artifacts = {
        "train": validate_bank_artifact(args.train_bank, "train"),
        "held_out": validate_bank_artifact(args.held_out_bank, "held_out"),
    }
    cap_root = Path("/home/ubuntu/rlm/outputs/latent-a1-nc0-cap768-v1")
    before = sorted(path.name for path in cap_root.iterdir())
    if "a1-nc0-cap768-run4" in before:
        raise RuntimeError("R3 proof found nonfresh operational run4")
    cases = []
    tokenizer_load_calls = 0
    with mock.patch.object(
        base.AutoModelForImageTextToText,
        "from_pretrained",
        side_effect=AssertionError("model loading forbidden in R3 render proof"),
    ) as model_loader:
        tokenizer = AutoTokenizer.from_pretrained(args.coordinator, local_files_only=True)
        tokenizer_load_calls += 1
        for probe_index, (selection, record, query) in enumerate(_selected_cases(artifacts), 1):
            for modality, messages, expected in (
                ("PARENT", base.parent_messages(record["parent_evidence"]), selection["parent_unpadded_tokens"]),
                ("MSELF", base.self_messages(query["child_query"]), selection["mself_unpadded_tokens"]),
            ):
                base.validate_parent_fixture(
                    record["parent_evidence"] if modality == "PARENT" else query["child_query"]
                )
                encoded = tokenizer.apply_chat_template(
                    messages,
                    tools=base.PARENT_TOOLS,
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=False,
                    return_tensors="pt",
                )
                ids = base.operational_template_input_ids(encoded)
                if (
                    type(encoded) is not BatchEncoding
                    or ids.shape != (1, expected)
                    or ids.device.type != "cpu"
                    or ids.dtype != torch.long
                    or not ids.is_contiguous()
                    or ids.numel() <= 0
                ):
                    raise RuntimeError("R3 operational render proof predicate rejected")
                cases.append(
                    {
                        "probe_index": probe_index,
                        "family": selection["family"],
                        "evidence_id": selection["evidence_id"],
                        "query_id": selection["query_id"],
                        "modality": modality,
                        "container_fqcn": f"{type(encoded).__module__}.{type(encoded).__qualname__}",
                        "input_ids_shape": list(ids.shape),
                        "input_ids_dtype": str(ids.dtype),
                        "input_ids_device": ids.device.type,
                        "input_ids_contiguous": ids.is_contiguous(),
                        "input_ids_sha256": tensor_bytes_sha256(ids),
                    }
                )
        if model_loader.call_count != 0:
            raise RuntimeError("R3 proof model loader was called")
    after = sorted(path.name for path in cap_root.iterdir())
    if before != after or "a1-nc0-cap768-run4" in after or torch.cuda.is_initialized():
        raise RuntimeError("R3 proof changed CUDA or CAP output state")
    expected_lengths = [517, 475, 599, 471, 616, 476, 644, 470]
    if (
        tokenizer_load_calls != 1
        or len(cases) != 8
        or [row["input_ids_shape"][1] for row in cases] != expected_lengths
    ):
        raise RuntimeError("R3 proof call or length schedule changed")
    receipt = {
        "schema_version": "prime-rl/latent-a1-nc0-cap768-r3-render-proof/v1",
        "status": "operational_render_mechanism_validated",
        "execution_commit": args.execution_commit,
        "cap_runner_path": str(cap_path),
        "cap_runner_sha256": file_sha256(cap_path),
        "runner_path": str(base_path),
        "runner_sha256": file_sha256(base_path),
        "proof_script_sha256": file_sha256(Path(__file__)),
        "runtime": runtime,
        "coordinator_metadata_sha256": EXPECTED_METADATA,
        "cuda_visible_devices": "",
        "cuda_uninitialized_before": True,
        "cuda_uninitialized_after": True,
        "offline": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "tokenizer_load_calls": 1,
        "model_load_calls": 0,
        "cap_runner_import_succeeded": True,
        "cap_base_module_identity": True,
        "validator_object_identity": True,
        "validator_defining_module": "prime_rl.latent.a1nc0",
        "main_called": False,
        "cases": cases,
        "case_count": 8,
        "case_order": ["PARENT", "MSELF"] * 4,
        "expected_lengths": expected_lengths,
        "cap_output_inventory_before": before,
        "cap_output_inventory_after": after,
        "run4_namespace_absent": True,
        "scientific_exposure": False,
        "model_loaded": False,
        "model_update_attempted": False,
        "receipt_sha256": "",
    }
    _atomic_json(args.output_dir / "receipt.json", receipt)


if __name__ == "__main__":
    main()
