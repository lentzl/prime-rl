#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import subprocess
from pathlib import Path

from prime_rl.latent.h_iter_phase1_cap0 import (
    ARTIFACT_DIR,
    E33_PATH,
    E33_STATE_SHA256,
    E33_TREE_SHA256,
    H176_PATH,
    H176_TREE_SHA256,
    MECHANISM,
    MEMORY_LABELS,
    METADATA_SHA256,
    MF0_BINDING,
    OUTPUT_ROOT,
    PLAN_SCHEMA,
    RESOURCE_BOUNDS,
    RUN_ID,
    RUNTIME,
    SELECTION_SHA256,
    canonical_json,
    sha256_bytes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--mechanism-commit", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != args.mechanism_commit or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo, text=True).strip():
        raise RuntimeError("CAP0 mechanism tree is not exact and clean")
    runner_path = repo / "scripts/latent/run_h_iter_phase1_cap0_v1.py"
    spec = importlib.util.spec_from_file_location("cap0_runner_freeze", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("CAP0 runner unavailable")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assets = {path: hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in runner.PLAN_ASSET_PATHS}
    plan = {
        "schema_version": PLAN_SCHEMA,
        "status": "preregistered",
        "mechanism": MECHANISM,
        "run_identity": RUN_ID,
        "mechanism_code_commit": args.mechanism_commit,
        "execution_authorization": runner.AUTHORIZATION,
        "output_root": OUTPUT_ROOT,
        "remote_paths": runner.REMOTE_PATHS,
        "runtime": RUNTIME,
        "asset_sha256": assets,
        "mf0_archive_binding": MF0_BINDING,
        "probe_contract": {"selection_sha256": SELECTION_SHA256, "probe_count": 4, "repeat_count": 2, "model_forwards": 8, "tokenizer_calls": 4, "sequences": 192},
        "model_contract": {"e33_path": E33_PATH, "e33_tree_sha256": E33_TREE_SHA256, "e33_state_sha256": E33_STATE_SHA256, "h176_path": H176_PATH, "h176_tree_sha256": H176_TREE_SHA256, "metadata_sha256": METADATA_SHA256},
        "cache_contract": {"checks": 18, "mandatory_negative_trips": 1, "actual_allocations": 0, "pkv_none": True, "config_restored": True},
        "resource_bounds": RESOURCE_BOUNDS,
        "memory_label_schedule": {"labels": MEMORY_LABELS, "count": 28, "label_sha256": sha256_bytes(canonical_json(MEMORY_LABELS))},
        "terminal_contract": runner.TERMINAL_CONTRACT,
        "safety_boundary": runner.SAFETY_BOUNDARY,
        "full_freeze": runner.FULL_FREEZE,
        "plan_sha256": "",
    }
    plan["plan_sha256"] = sha256_bytes(canonical_json({key: value for key, value in plan.items() if key != "plan_sha256"}))
    runner.validate_plan(plan)
    encoded = canonical_json(plan) + b"\n"
    output = repo / ARTIFACT_DIR
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "cap0-plan.json"
    sidecar = output / "cap0-plan.sha256"
    plan_path.write_bytes(encoded)
    external = hashlib.sha256(encoded).hexdigest()
    sidecar.write_text(external + "\n")
    print(external)
    print(plan["plan_sha256"])


if __name__ == "__main__":
    main()
