#!/usr/bin/env python3
"""Create the exact-parent H-ITER Phase-1 MF0 plan and sidecar."""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
from pathlib import Path

from run_h_iter_phase1_mf0_v1 import (
    EXPECTED_RUNTIME,
    EXPOSURE_STATUS,
    FULL_FREEZE_CONTRACT,
    INCOMPLETE_STATUS,
    INFRASTRUCTURE_STATUS,
    MEMORY_LABELS,
    PLAN_ASSET_PATHS,
    PLAN_SCHEMA,
    PROOF_STATUS,
    RESOURCE_BOUNDS,
    SAFETY_BOUNDARY,
    validate_plan,
)

from prime_rl.latent.h_iter_phase1_mf0 import (
    ARTIFACT_DIR,
    ASSET_NAMES,
    MECHANISM,
    OUTPUT_ROOT,
    RUN_ID,
    canonical_json,
    sha256_bytes,
)

PLAN_RELATIVE_PATH = f"{ARTIFACT_DIR}/mf0-plan.json"
SIDECAR_RELATIVE_PATH = f"{ARTIFACT_DIR}/mf0-plan.sha256"
def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short MF0 plan write")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--mechanism-commit", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    if git(repo, "rev-parse", "HEAD") != args.mechanism_commit or git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("MF0 mechanism tree is not exact and clean")
    asset_sha256 = {path: sha256_bytes((repo / path).read_bytes()) for path in PLAN_ASSET_PATHS}
    plan = {
        "schema_version": PLAN_SCHEMA,
        "status": "preregistered",
        "mechanism": MECHANISM,
        "run_identity": RUN_ID,
        "mechanism_code_commit": args.mechanism_commit,
        "execution_authorization": {"mf0_model_free_prereg_only": True, "cap0": False, "t0": False, "model": False, "gpu": False, "training": False},
        "output_root": OUTPUT_ROOT,
        "asset_sha256": asset_sha256,
        "runtime": EXPECTED_RUNTIME,
        "resource_bounds": RESOURCE_BOUNDS,
        "materialization_contract": {"asset_names": ASSET_NAMES, "regenerate_byte_identical": True, "source_split": "train", "validation_or_heldout_paths_forbidden": True},
        "terminal_contract": {"success_file": "MF0-PROOF.json", "failure_file": "MF0-FAILURE.json", "exclusive_atomic": True, "canonical_roundtrip_twice": True, "success_status": PROOF_STATUS, "failure_statuses": [INCOMPLETE_STATUS, EXPOSURE_STATUS, INFRASTRUCTURE_STATUS]},
        "memory_label_schedule": {"labels": MEMORY_LABELS, "label_sha256": sha256_bytes(canonical_json(MEMORY_LABELS)), "count": 17},
        "safety_boundary": SAFETY_BOUNDARY,
        "full_freeze": FULL_FREEZE_CONTRACT,
        "plan_sha256": "",
    }
    plan["plan_sha256"] = sha256_bytes(canonical_json({key: value for key, value in plan.items() if key != "plan_sha256"}))
    validate_plan(plan)
    plan_bytes = canonical_json(plan) + b"\n"
    plan_sha256 = sha256_bytes(plan_bytes)
    base = repo / ARTIFACT_DIR
    write_atomic(base / "mf0-plan.json", plan_bytes)
    write_atomic(base / "mf0-plan.sha256", f"{plan_sha256}\n".encode())
    descriptor = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(plan_sha256)
    print(plan["plan_sha256"])


if __name__ == "__main__":
    main()
