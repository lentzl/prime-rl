#!/usr/bin/env python3
"""Tokenizer-only proof for the prospective A1-NC0-R1 render repair."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from unittest import mock

import run_a1_nc0_nomination_v1 as runner
import torch
import transformers
from transformers import AutoTokenizer

from prime_rl.latent.a0 import canonical_json_hash, file_sha256
from prime_rl.latent.a1nc0 import validate_bank_artifact

OUTPUT_ROOT = Path("/home/ubuntu/rlm/outputs/latent-a1-nc0-r1-tokenizer-proof-v1")
TOKENIZER_HASHES = {
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "747ba36a06ba5428bb74e984d75136b37cf5dafe97b8dd315f701b361a9f417f",
}
EXPECTED_VERSIONS = {
    "python": "3.12.14",
    "transformers": "5.6.2",
    "flash-linear-attention": "0.5.2",
    "torch": "2.11.0+cu128",
}


def _write_exclusive(directory: Path, name: str, payload: bytes) -> dict[str, object]:
    temporary = directory / f".{name}.tmp"
    target = directory / name
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written == 0:
                raise OSError("short tokenizer-proof write")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)
    os.replace(temporary, target)
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {"name": name, "bytes": len(payload), "sha256": file_sha256(target)}


def _verify_tree(repo: Path, commit: str) -> None:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if head != commit or status or len(commit) != 40:
        raise ValueError("tokenizer proof requires the exact clean mechanism commit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train-bank", type=Path, required=True)
    parser.add_argument("--validation-bank", type=Path, required=True)
    parser.add_argument("--held-out-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mechanism-commit", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise RuntimeError("tokenizer proof requires CUDA hidden and uninitialized")
    if args.output_dir.parent != OUTPUT_ROOT or not args.output_dir.name.startswith("a1-nc0-r1-tokenizer-proof-"):
        raise ValueError("tokenizer proof output namespace changed")
    if OUTPUT_ROOT.is_symlink() or args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("tokenizer proof output must be fresh and symlink-safe")
    _verify_tree(args.repo, args.mechanism_commit)

    versions = {
        "python": platform.python_version(),
        "transformers": importlib.metadata.version("transformers"),
        "flash-linear-attention": importlib.metadata.version("flash-linear-attention"),
        "torch": importlib.metadata.version("torch"),
    }
    if versions != EXPECTED_VERSIONS or transformers.__version__ != EXPECTED_VERSIONS["transformers"]:
        raise RuntimeError("tokenizer proof runtime versions changed")
    tokenizer_hashes = {name: file_sha256(args.tokenizer / name) for name in TOKENIZER_HASHES}
    if tokenizer_hashes != TOKENIZER_HASHES:
        raise RuntimeError("tokenizer proof e33 tokenizer identity changed")
    artifacts = {
        "train": validate_bank_artifact(args.train_bank, "train"),
        "validation": validate_bank_artifact(args.validation_bank, "validation"),
        "held_out": validate_bank_artifact(args.held_out_bank, "held_out"),
    }
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    cuda_before = torch.cuda.is_initialized()
    with mock.patch.object(
        runner.AutoModelForImageTextToText,
        "from_pretrained",
        side_effect=AssertionError("model loading is forbidden in tokenizer proof"),
    ) as model_loader:
        first = runner.validate_rendering_preflight(tokenizer, artifacts, 248046)
        second = runner.validate_rendering_preflight(tokenizer, artifacts, 248046)
    cuda_after = torch.cuda.is_initialized()
    if first != second or cuda_before or cuda_after or model_loader.call_count != 0:
        raise RuntimeError("tokenizer proof determinism, CUDA, or no-model boundary changed")

    OUTPUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output_dir.mkdir(mode=0o700)
    log_payload = (
        json.dumps(
            {
                "event": "tokenizer_only_preflight_complete",
                "materialized_queries": first["materialized_queries"],
                "render_hashes_sha256": first["render_hashes_sha256"],
                "label_alignment_sha256": first["label_alignment_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    log = _write_exclusive(args.output_dir, "proof.log", log_payload)
    receipt = {
        "schema_version": "prime-rl/latent-a1-nc0-r1-tokenizer-proof/v1",
        "status": "tokenizer_render_mechanism_validated",
        "mechanism_commit": args.mechanism_commit,
        "versions": versions,
        "transformers_runtime_version": transformers.__version__,
        "tokenizer_class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
        "tokenizer_asset_sha256": tokenizer_hashes,
        "bank_file_sha256": {
            "train": file_sha256(args.train_bank),
            "validation": file_sha256(args.validation_bank),
            "held_out": file_sha256(args.held_out_bank),
        },
        "preflight": first,
        "repeat_preflight_bitwise_canonical_equal": True,
        "repeat_render_hashes_sha256": second["render_hashes_sha256"],
        "repeat_label_alignment_sha256": second["label_alignment_sha256"],
        "cuda_visible_devices": "",
        "torch_cuda_initialized_before": cuda_before,
        "torch_cuda_initialized_after": cuda_after,
        "model_from_pretrained_calls": model_loader.call_count,
        "model_loaded": False,
        "optimizer_created": False,
        "model_update_attempted": False,
        "proof_log": log,
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    encoded = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _write_exclusive(args.output_dir, "receipt.json", encoded)


if __name__ == "__main__":
    main()
