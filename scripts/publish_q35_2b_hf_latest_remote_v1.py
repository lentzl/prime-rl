#!/usr/bin/env python3
"""Publish one validated dense checkpoint to an already-created HF repo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


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


def main() -> None:
    args = _parse_args()
    token, model_card = _read_payload()
    checkpoint_dir = args.checkpoint_dir.resolve()
    if not (checkpoint_dir / "model.safetensors").is_file():
        raise FileNotFoundError(checkpoint_dir / "model.safetensors")
    (checkpoint_dir / "README.md").write_text(model_card, encoding="utf-8")
    HfApi(token=token).upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=checkpoint_dir,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
