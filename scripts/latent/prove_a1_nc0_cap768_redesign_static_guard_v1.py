#!/usr/bin/env python3
"""CUDA-hidden exact-host proof for the REDESIGN0 AST-only static guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

from prime_rl.latent.cap768_redesign_invariants import InvariantViolation, inspect_no_training_runner


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise RuntimeError("static-guard proof must begin CUDA-hidden and uninitialized")
    runner = args.repo / "scripts/latent/run_a1_nc0_cap768_redesign_v1.py"
    positive = inspect_no_training_runner(runner).as_dict()
    negatives = {}
    fixtures = {
        "generate_call": "model.generate()\n",
        "backward_call": "loss.backward()\n",
        "optimizer_step_call": "optimizer.step()\n",
        "workspace_bridge_name": "WorkspaceBridge\n",
        "adamw_attribute": "optimizer = torch.optim.AdamW([])\n",
    }
    with tempfile.TemporaryDirectory(prefix="cap768r-static-guard-") as temporary:
        root = Path(temporary)
        for name, source in fixtures.items():
            fixture = root / f"{name}.py"
            fixture.write_text(source)
            try:
                inspect_no_training_runner(fixture)
            except InvariantViolation:
                negatives[name] = True
            else:
                negatives[name] = False
    if not all(negatives.values()) or torch.cuda.is_initialized():
        raise RuntimeError("static-guard proof failed")
    payload = {
        "schema_version": "prime-rl/latent-a1-nc0-cap768-redesign-static-guard-proof/v1",
        "status": "static_guard_validated_cuda_hidden",
        "execution_commit": args.execution_commit,
        "runner_sha256": sha256(runner),
        "guard_module_sha256": sha256(args.repo / "src/prime_rl/latent/cap768_redesign_invariants.py"),
        "positive": positive,
        "negative_fixtures": negatives,
        "cuda_visible_devices": "",
        "cuda_initialized_before_after": True,
        "model_loaded": False,
        "model_forward_count": 0,
        "scientific_exposure": False,
        "model_update_attempted": False,
        "proof_sha256": "",
    }
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    payload["proof_sha256"] = hashlib.sha256(canonical({**payload, "proof_sha256": ""})).hexdigest()
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("proof output must be fresh")
    encoded = canonical(payload) + b"\n"
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    main()
