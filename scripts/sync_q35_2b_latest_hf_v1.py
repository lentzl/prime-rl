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
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
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
SSH_KEEPALIVE_OPTIONS = (
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=12",
    "-o",
    "TCPKeepAlive=yes",
)


def _run(
    command: list[str], *, capture: bool = False, input_text: str | None = None
) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        input=input_text,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def _ssh(
    args: argparse.Namespace, remote_command: str, *, input_text: str | None = None
) -> str:
    return _run(
        [
            "ssh",
            "-i",
            str(args.ssh_key),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            *SSH_KEEPALIVE_OPTIONS,
            args.remote,
            remote_command,
        ],
        capture=True,
        input_text=input_text,
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


def _append_jsonl_durable(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


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
            "--append",
            "-e",
            (
                f"ssh -i {shlex.quote(str(args.ssh_key))}"
                " -o BatchMode=yes -o ConnectTimeout=8"
                " -o ServerAliveInterval=15 -o ServerAliveCountMax=12"
                " -o TCPKeepAlive=yes"
            ),
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


def _upload_latest_only(
    api: HfApi,
    repo_id: str,
    checkpoint_dir: Path,
    commit_message: str,
    remote_upload: tuple[argparse.Namespace, str, str, str] | None = None,
) -> None:
    # This is a recovery slot, not a checkpoint archive. Recreate it for every
    # replacement so the old dense blob and commit history cannot accumulate
    # or temporarily double private-storage use. The caller has already
    # validated the immutable remote source and its complete local copy; the
    # remote source is pruned only after this upload is independently verified.
    api.delete_repo(repo_id, repo_type="model")
    api.create_repo(repo_id, repo_type="model", private=True)
    if remote_upload is None:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=checkpoint_dir,
            commit_message=commit_message,
        )
        return

    args, model_path, token, model_card = remote_upload
    _upload_remote_checkpoint(
        args,
        repo_id=repo_id,
        model_path=model_path,
        commit_message=commit_message,
        token=token,
        model_card=model_card,
    )


def _upload_remote_checkpoint(
    args: argparse.Namespace,
    *,
    repo_id: str,
    model_path: str,
    commit_message: str,
    token: str,
    model_card: str,
) -> None:
    if not args.remote_upload_helper or not args.remote_uv_bin:
        raise RuntimeError("remote upload requires helper and uv paths")
    command = " ".join(
        [
            "PYTHONDONTWRITEBYTECODE=1",
            shlex.quote(args.remote_uv_bin),
            "run",
            "--no-project",
            "--with",
            shlex.quote("huggingface-hub>=0.34"),
            "python",
            shlex.quote(args.remote_upload_helper),
            "--repo-id",
            shlex.quote(repo_id),
            "--checkpoint-dir",
            shlex.quote(model_path),
            "--commit-message",
            shlex.quote(commit_message),
        ]
    )
    _ssh(
        args,
        command,
        input_text=json.dumps({"model_card": model_card, "token": token}) + "\n",
    )


def _sync_role(
    args: argparse.Namespace,
    api: HfApi,
    token: str,
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
            # macOS openrsync supports resumable --append but not GNU
            # --append-verify. If the resumed prefix was not valid, discard
            # only the invalid local model blob and fetch it once from the
            # already-validated immutable remote checkpoint.
            (checkpoint_dir / MODEL_FILENAME).unlink()
            _rsync_checkpoint(args, model_path, checkpoint_dir)
            local_hash = _sha256(checkpoint_dir / MODEL_FILENAME)
            if local_hash != expected_hash:
                raise RuntimeError(
                    "local checkpoint hash mismatch after clean retry: "
                    f"expected {expected_hash}, got {local_hash}"
                )
        model_card = _model_card(
            role,
            start,
            update,
            evaluation,
            prime_revision,
            verifier_revision,
        )
        (checkpoint_dir / "README.md").write_text(model_card, encoding="utf-8")
        remote_upload = None
        if args.remote_upload_helper:
            remote_upload = (args, model_path, token, model_card)
        _upload_latest_only(
            api,
            repo_id,
            checkpoint_dir,
            f"Replace latest {role} frontier with cycle {update['cycle']}",
            remote_upload,
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


def _prune_superseded_remote_weights(
    args: argparse.Namespace,
    events: list[dict[str, Any]],
    frontiers: dict[
        str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
    ],
) -> list[dict[str, Any]]:
    """Remove only completed, superseded autonomous weight payloads.

    This runs only after both current role frontiers have been mirrored and
    verified on the Hub.  It deliberately leaves each run directory and all
    results, receipts, logs, artifacts, and lifecycle events intact.  A run
    without both train_completed and evaluation_completed events can never
    become a deletion candidate, so an in-flight admission or recoverable
    interrupted update is protected as well.
    """
    protected_model_paths = {
        records[1]["output_candidate"]["model_path"]
        for records in frontiers.values()
    }
    protected_run_dirs = {
        PurePosixPath(model_path).parents[1]
        for model_path in protected_model_paths
    }
    if len(protected_run_dirs) != len(ROLES):
        raise RuntimeError("latest role frontiers do not resolve to two run directories")
    output_roots = {run_dir.parent for run_dir in protected_run_dirs}
    if len(output_roots) != 1:
        raise RuntimeError("latest role frontiers do not share one output root")
    output_root = next(iter(output_roots))

    # A dense update is not superseded merely because the last *evaluated*
    # frontier still points at an older checkpoint.  Admission can take many
    # minutes, during which the newer update is the aggressive training
    # frontier and the only copy may still live on the trainer.  Only updates
    # with their own completed evaluation may be considered for deletion, and
    # only when an even newer/equal evaluated checkpoint for that same role is
    # protected on the Hub.
    evaluated = {
        (event.get("cycle"), event.get("role"))
        for event in events
        if event.get("kind") == "evaluation_completed"
    }
    protected_cycles = {
        role: records[1]["cycle"] for role, records in frontiers.items()
    }

    candidates: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("kind") != "train_completed":
            continue
        role = event.get("role")
        cycle = event.get("cycle")
        if (
            role not in ROLES
            or not isinstance(cycle, int)
            or (cycle, role) not in evaluated
            or cycle >= protected_cycles[role]
        ):
            continue
        candidate = event.get("output_candidate")
        if not isinstance(candidate, dict):
            continue
        model_path = candidate.get("model_path")
        if not isinstance(model_path, str) or model_path in protected_model_paths:
            continue
        path = PurePosixPath(model_path)
        if len(path.parents) < 2:
            continue
        run_dir = path.parents[1]
        if (
            run_dir.parent != output_root
            or not run_dir.name.startswith("grpo-auto-")
            or path.name != "step_1"
            or path.parent.name != "weights"
        ):
            continue
        candidates[model_path] = {
            "cycle": cycle,
            "model_path": model_path,
            "model_sha256": candidate.get("model_sha256"),
            "role": role,
        }

    ordered = sorted(candidates)
    if ordered:
        quoted = " ".join(shlex.quote(path) for path in ordered)
        removed = _ssh(
            args,
            "for path in "
            f"{quoted}; do "
            'if test -e "$path"; then printf \'%s\\n\' "$path"; '
            'rm -rf -- "$path"; fi; done',
        )
        return [candidates[line] for line in removed.splitlines() if line]
    return []


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
    parser.add_argument("--remote-upload-helper")
    parser.add_argument("--remote-uv-bin")
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
                token,
                role,
                frontiers[role],
                args.prime_revision,
                args.verifier_revision,
            )
            for role in ROLES
        ]
        pruned = _prune_superseded_remote_weights(args, events, frontiers)
        if pruned:
            _append_jsonl_durable(
                args.local_state_dir / "remote-weight-retention.jsonl",
                {
                    "mirrors": results,
                    "recorded_at_utc": datetime.now(timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "remote": args.remote,
                    "remote_state_dir": args.remote_state_dir,
                    "removed": pruned,
                    "schema_version": "q35-2b-hf-latest-retention/v1",
                },
            )
    print(
        json.dumps(
            {"mirrors": results, "pruned_remote_weights": pruned},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
