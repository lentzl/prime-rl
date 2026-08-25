#!/usr/bin/env python3
"""Mirror the latest completed dual-dense role frontiers into rolling HF repos."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

MODEL_FILENAME = "model.safetensors"
REQUIRED_CHECKPOINT_FILES = (
    "STABLE",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    MODEL_FILENAME,
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
ROLES = ("coordinator", "child")


def _run(command: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def _ssh(args: argparse.Namespace, remote_command: str) -> str:
    return _run(
        [
            "ssh",
            "-i",
            str(args.ssh_key),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            args.remote,
            remote_command,
        ],
        capture=True,
    )


def _load_events(args: argparse.Namespace) -> list[dict[str, Any]]:
    events_path = shlex.quote(f"{args.remote_state_dir}/events.jsonl")
    payload = _ssh(args, f"cat {events_path}")
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


def _latest_completed_frontiers(
    events: list[dict[str, Any]],
) -> dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    starts: dict[tuple[int, str], dict[str, Any]] = {}
    updates: dict[tuple[int, str], dict[str, Any]] = {}
    evaluations: dict[tuple[int, str], dict[str, Any]] = {}
    for event in events:
        role = event.get("role")
        cycle = event.get("cycle")
        if role not in ROLES or not isinstance(cycle, int):
            continue
        key = (cycle, role)
        if event.get("kind") == "train_started":
            starts[key] = event
        elif event.get("kind") == "train_completed":
            updates[key] = event
        elif event.get("kind") == "evaluation_completed":
            evaluations[key] = event

    frontiers: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for role in ROLES:
        complete_cycles = [
            cycle
            for cycle, event_role in updates
            if event_role == role
            and (cycle, role) in starts
            and (cycle, role) in evaluations
        ]
        if not complete_cycles:
            continue
        cycle = max(complete_cycles)
        key = (cycle, role)
        frontiers[role] = starts[key], updates[key], evaluations[key]
    return frontiers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_model_hash(api: HfApi, repo_id: str) -> str | None:
    info = api.model_info(repo_id, files_metadata=True)
    for sibling in info.siblings or []:
        if sibling.rfilename != MODEL_FILENAME or sibling.lfs is None:
            continue
        return str(sibling.lfs["sha256"])
    return None


def _model_card(
    role: str,
    start: dict[str, Any],
    update: dict[str, Any],
    evaluation: dict[str, Any],
    prime_revision: str,
    verifier_revision: str,
) -> str:
    cycle = update["cycle"]
    model_hash = update["output_candidate"]["model_sha256"]
    source_hash = start["source"]["model_sha256"]
    anchor_hash = start["anchor"]["model_sha256"]
    qualifying = evaluation["qualifying"]
    episodes = evaluation["episodes"]
    minimum = evaluation["promotion_minimum"]
    admitted = evaluation["admitted"]
    status = "admitted" if admitted else "retained training frontier; not admitted"
    return f"""---
library_name: transformers
base_model: Qwen/Qwen3.5-2B
tags:
  - qwen3.5
  - reinforcement-learning
  - grpo
  - prime-agent
  - spade
---

# Qwen3.5 2B Prime Agent SPADE {role} — cycle {cycle}

Rolling latest-only full-weight {role} recovery checkpoint from the dual-dense
autonomous Prime Agent harness experiment.

- Status: {status}; leak-free admission {qualifying}/{episodes}, with unchanged
  promotion minimum {minimum}.
- Training: one full dense optimizer update over a GRPO group of eight.
- Phase: `{update['phase']}`.
- Model SHA-256: `{model_hash}`.
- Source {role} SHA-256: `{source_hash}`.
- Counterpart anchor SHA-256: `{anchor_hash}`.
- Prime-RL source commit: `{prime_revision}`.
- Verifier source commit: `{verifier_revision}`.

Training scaffolds are excluded from admission. This repository is a rolling
latest-only recovery slot; dense revision history is squashed after each
verified replacement. Load it as a normal Transformers model.
"""


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _validate_remote_checkpoint(
    args: argparse.Namespace, model_path: str, expected_hash: str
) -> None:
    quoted_path = shlex.quote(model_path)
    checks = " && ".join(
        f"test -{'e' if filename == 'STABLE' else 'f'} {quoted_path}/{shlex.quote(filename)}"
        for filename in REQUIRED_CHECKPOINT_FILES
    )
    output = _ssh(
        args,
        f"{checks} && sha256sum {quoted_path}/{MODEL_FILENAME}",
    )
    remote_hash = output.split()[0]
    if remote_hash != expected_hash:
        raise RuntimeError(
            f"remote checkpoint hash mismatch: expected {expected_hash}, got {remote_hash}"
        )


def _rsync_checkpoint(
    args: argparse.Namespace, model_path: str, checkpoint_dir: Path
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "rsync",
            "-a",
            "--delete",
            "--partial",
            "-e",
            f"ssh -i {args.ssh_key} -o BatchMode=yes -o ConnectTimeout=8",
            f"{args.remote}:{model_path}/",
            f"{checkpoint_dir}/",
        ]
    )


def _squash_and_verify(
    api: HfApi, repo_id: str, expected_hash: str, message: str
) -> str:
    commits = api.list_repo_commits(repo_id, repo_type="model")
    if len(commits) != 1:
        api.super_squash_history(
            repo_id,
            branch="main",
            repo_type="model",
            commit_message=message,
        )
    commits = api.list_repo_commits(repo_id, repo_type="model")
    if len(commits) != 1:
        raise RuntimeError(f"{repo_id} still has {len(commits)} commits after squash")
    hub_hash = _repo_model_hash(api, repo_id)
    if hub_hash != expected_hash:
        raise RuntimeError(
            f"Hub checkpoint hash mismatch: expected {expected_hash}, got {hub_hash}"
        )
    return commits[0].commit_id


def _sync_role(
    args: argparse.Namespace,
    api: HfApi,
    role: str,
    records: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    prime_revision: str,
    verifier_revision: str,
) -> dict[str, Any]:
    start, update, evaluation = records
    candidate = update["output_candidate"]
    expected_hash = candidate["model_sha256"]
    model_path = candidate["model_path"]
    repo_id = args.coordinator_repo if role == "coordinator" else args.child_repo
    manifest_path = args.local_state_dir / f"latest-hf-{role}.json"
    checkpoint_dir = args.checkpoint_root / f"q35-2b-grpo-auto-latest-{role}"

    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    if _repo_model_hash(api, repo_id) != expected_hash:
        _validate_remote_checkpoint(args, model_path, expected_hash)
        _rsync_checkpoint(args, model_path, checkpoint_dir)
        local_hash = _sha256(checkpoint_dir / MODEL_FILENAME)
        if local_hash != expected_hash:
            raise RuntimeError(
                f"local checkpoint hash mismatch: expected {expected_hash}, got {local_hash}"
            )
        (checkpoint_dir / "README.md").write_text(
            _model_card(
                role,
                start,
                update,
                evaluation,
                prime_revision,
                verifier_revision,
            ),
            encoding="utf-8",
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=checkpoint_dir,
            commit_message=f"Replace latest {role} frontier with cycle {update['cycle']}",
        )

    head = _squash_and_verify(
        api,
        repo_id,
        expected_hash,
        f"Latest-only recovery checkpoint: {role} cycle {update['cycle']}",
    )
    manifest = {
        "admitted": evaluation["admitted"],
        "cycle": update["cycle"],
        "event_sequence": evaluation["sequence"],
        "hf_commit": head,
        "model_sha256": expected_hash,
        "qualifying": evaluation["qualifying"],
        "repo_id": repo_id,
        "role": role,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--remote-state-dir", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--local-state-dir", type=Path, required=True)
    parser.add_argument("--coordinator-repo", required=True)
    parser.add_argument("--child-repo", required=True)
    parser.add_argument("--prime-revision", required=True)
    parser.add_argument("--verifier-revision", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_KEY")
    if not token:
        raise RuntimeError("HF_TOKEN or HF_KEY is required")
    args.ssh_key = args.ssh_key.expanduser().resolve()
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    args.local_state_dir = args.local_state_dir.expanduser().resolve()
    args.local_state_dir.mkdir(parents=True, exist_ok=True)

    with (args.local_state_dir / "sync.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        events = _load_events(args)
        frontiers = _latest_completed_frontiers(events)
        if set(frontiers) != set(ROLES):
            raise RuntimeError(f"missing completed role frontiers: {sorted(frontiers)}")
        api = HfApi(token=token)
        results = [
            _sync_role(
                args,
                api,
                role,
                frontiers[role],
                args.prime_revision,
                args.verifier_revision,
            )
            for role in ROLES
        ]
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
