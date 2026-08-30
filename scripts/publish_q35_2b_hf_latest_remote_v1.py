#!/usr/bin/env python3
"""Publish one validated dense checkpoint to an already-created HF repo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

_PRESERVED_REPO_FILES = {".gitattributes"}


def _read_payload() -> tuple[str, str]:
    payload: Any = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise TypeError("upload payload must be an object")
    token = payload.get("token")
    model_card = payload.get("model_card")
    if not isinstance(token, str) or not token:
        raise ValueError("upload payload requires a token")
    if not isinstance(model_card, str) or not model_card:
        raise ValueError("upload payload requires a model card")
    return token, model_card


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--commit-message", required=True)
    return parser.parse_args()


def _local_files(checkpoint_dir: Path) -> set[str]:
    return {path.relative_to(checkpoint_dir).as_posix() for path in checkpoint_dir.rglob("*") if path.is_file()}


def _stale_repo_files(api: HfApi, *, repo_id: str, checkpoint_dir: Path) -> list[str]:
    local_files = _local_files(checkpoint_dir)
    return sorted(
        path
        for path in api.list_repo_files(repo_id=repo_id, repo_type="model")
        if path not in local_files and path not in _PRESERVED_REPO_FILES
    )


def remote_checkpoint_sha256(*, token: str, repo_id: str, revision: str | None = None) -> str | None:
    """Return the SHA-256 of the dense checkpoint currently exposed by a repo ref."""
    info = HfApi(token=token).model_info(
        repo_id=repo_id,
        revision=revision,
        files_metadata=True,
    )
    for sibling in info.siblings:
        if sibling.rfilename != "model.safetensors" or sibling.lfs is None:
            continue
        sha256 = sibling.lfs.sha256
        return sha256 if isinstance(sha256, str) and sha256 else None
    return None


def publish_checkpoint(
    *,
    token: str,
    repo_id: str,
    checkpoint_dir: Path,
    model_card: str,
    commit_message: str,
) -> str:
    """Replace a model repository with one validated dense checkpoint."""
    checkpoint_dir = checkpoint_dir.resolve()
    if not (checkpoint_dir / "model.safetensors").is_file():
        raise FileNotFoundError(checkpoint_dir / "model.safetensors")
    (checkpoint_dir / "README.md").write_text(model_card, encoding="utf-8")
    api = HfApi(token=token)
    stale_files = _stale_repo_files(api, repo_id=repo_id, checkpoint_dir=checkpoint_dir)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=checkpoint_dir,
        commit_message=commit_message,
        delete_patterns=stale_files or None,
    )
    revision = getattr(commit, "oid", None)
    return revision if isinstance(revision, str) else "unknown"


def main() -> None:
    args = _parse_args()
    token, model_card = _read_payload()
    publish_checkpoint(
        token=token,
        repo_id=args.repo_id,
        checkpoint_dir=args.checkpoint_dir,
        model_card=model_card,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
