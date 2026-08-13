#!/usr/bin/env python3
"""Back up stable LoRA checkpoints from a live Prime Agent run to one HF repo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

STEP = re.compile(r"step_(\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def stable_adapters(output: Path) -> list[tuple[int, Path]]:
    found = []
    for checkpoint in (output / "weights").glob("step_*"):
        match = STEP.fullmatch(checkpoint.name)
        adapter = checkpoint / "lora_adapters"
        if (
            match
            and (checkpoint / "STABLE").is_file()
            and (adapter / "adapter_config.json").is_file()
            and (adapter / "adapter_model.safetensors").is_file()
        ):
            found.append((int(match.group(1)), adapter))
    return sorted(found)


def backup_step(
    api: HfApi,
    repo_id: str,
    run_id: str,
    step: int,
    adapter: Path,
    root: Path,
    training_config: Path | None,
) -> None:
    prefix = f"runs/{run_id}/step_{step}"
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=adapter,
        path_in_repo=f"{prefix}/lora_adapters",
        commit_message=f"Back up {run_id} adapter step {step}",
    )
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "step": step,
        "prime_rl_revision": git_revision(root),
        "verifiers_revision": git_revision(root / "deps" / "verifiers"),
        "training_config_sha256": sha256(training_config) if training_config else None,
        "files": {
            path.name: {"size": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(adapter.iterdir())
            if path.is_file()
        },
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "backup.json"
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=path,
            path_in_repo=f"{prefix}/backup.json",
            commit_message=f"Record {run_id} adapter step {step}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("run_id")
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_KEY")
    if not token:
        raise SystemExit("HF_TOKEN or HF_KEY is required")
    root = Path(__file__).resolve().parents[1]
    api = HfApi(token=token)
    api.create_repo(args.repo_id, private=True, exist_ok=True, repo_type="model")
    remote_files = set(api.list_repo_files(args.repo_id, repo_type="model"))
    config_target = f"runs/{args.run_id}/training_config.toml"
    if args.training_config and config_target not in remote_files:
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=args.training_config,
            path_in_repo=config_target,
            commit_message=f"Record {args.run_id} training config",
        )
        remote_files.add(config_target)

    while True:
        for step, adapter in stable_adapters(args.output):
            marker = f"runs/{args.run_id}/step_{step}/backup.json"
            if marker in remote_files:
                continue
            backup_step(
                api,
                args.repo_id,
                args.run_id,
                step,
                adapter,
                root,
                args.training_config,
            )
            remote_files.add(marker)
            print(f"backed up {args.run_id} step {step}", flush=True)
        if args.once or (
            args.max_step is not None
            and f"runs/{args.run_id}/step_{args.max_step}/backup.json" in remote_files
        ):
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
