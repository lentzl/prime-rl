#!/usr/bin/env python3
"""Validate and publish a selected Prime Agent checkpoint to Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def eos_fields(value: Any, path: tuple[str, ...] = ()) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = (*path, key)
            if key == "eos_token_id":
                values = item if isinstance(item, list) else [item]
                for index, token_id in enumerate(values):
                    if isinstance(token_id, int) and not isinstance(token_id, bool):
                        suffix = f"[{index}]" if isinstance(item, list) else ""
                        found.append((f"{'.'.join(child_path)}{suffix}", token_id))
            found.extend(eos_fields(item, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(eos_fields(item, (*path, f"[{index}]")))
    return found


def validate_checkpoint(checkpoint: Path) -> dict[str, Any]:
    if not (checkpoint / "STABLE").is_file():
        raise ValueError(f"checkpoint is not marked stable: {checkpoint}")
    if not any(checkpoint.glob("model*.safetensors")):
        raise ValueError("checkpoint does not contain merged full-model safetensors")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(im_end_ids) != 1:
        raise ValueError(f"<|im_end|> must encode to one token, got {im_end_ids}")
    expected_eos = im_end_ids[0]
    if tokenizer.eos_token_id != expected_eos:
        raise ValueError(f"tokenizer eos_token_id={tokenizer.eos_token_id}, expected {expected_eos}")

    metadata_files = [checkpoint / "config.json", checkpoint / "generation_config.json"]
    missing = [str(path) for path in metadata_files if not path.is_file()]
    if missing:
        raise ValueError(f"checkpoint metadata missing: {', '.join(missing)}")
    fields: list[tuple[str, int]] = []
    for path in metadata_files:
        fields.extend(
            (f"{path.name}:{field}", token_id)
            for field, token_id in eos_fields(json.loads(path.read_text()))
        )
    if not fields:
        raise ValueError("checkpoint metadata has no numeric eos_token_id fields")
    mismatches = [f"{field}={token_id}" for field, token_id in fields if token_id != expected_eos]
    if mismatches:
        raise ValueError(f"EOS metadata does not match <|im_end|>={expected_eos}: {', '.join(mismatches)}")
    return {"im_end_token_id": expected_eos, "validated_eos_fields": dict(fields)}


def validate_remote_file_list(filenames: set[str]) -> None:
    required = {"config.json", "generation_config.json", "tokenizer_config.json", "prime_agent_bundle.json"}
    missing = sorted(required - filenames)
    if missing:
        raise RuntimeError(f"published checkpoint is missing: {', '.join(missing)}")
    if not any(Path(name).name.startswith("model") and name.endswith(".safetensors") for name in filenames):
        raise RuntimeError("published checkpoint has no full-model safetensors")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--eval-report", action="append", required=True, type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--commit-message", default="Publish verified Prime Agent checkpoint")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    dataset_manifest = args.dataset_manifest.resolve()
    eval_reports = [path.resolve() for path in args.eval_report]
    for path in [dataset_manifest, *eval_reports]:
        if not path.is_file():
            raise SystemExit(f"required provenance file does not exist: {path}")

    validation = validate_checkpoint(checkpoint)
    root = Path(__file__).resolve().parents[1]
    bundle = {
        "schema_version": 1,
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "prime_rl_revision": git_revision(root),
        "verifiers_revision": git_revision(root / "deps" / "verifiers"),
        "checkpoint": str(checkpoint),
        "dataset_manifest": {
            "name": dataset_manifest.name,
            "sha256": sha256(dataset_manifest),
        },
        "eval_reports": [
            {"name": path.name, "sha256": sha256(path)}
            for path in eval_reports
        ],
        **validation,
    }

    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_KEY")
    if not token:
        raise SystemExit("HF_TOKEN or HF_KEY is required")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, private=not args.public, exist_ok=True, repo_type="model")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=checkpoint,
        commit_message=args.commit_message,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        bundle_path = Path(temp_dir) / "prime_agent_bundle.json"
        bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=bundle_path,
            path_in_repo=bundle_path.name,
            commit_message="Attach Prime Agent training provenance",
        )

    info = api.model_info(args.repo_id)
    validate_remote_file_list({sibling.rfilename for sibling in info.siblings})
    remote_bundle = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename="prime_agent_bundle.json",
            revision=info.sha,
            token=token,
        )
    )
    if json.loads(remote_bundle.read_text()) != bundle:
        raise RuntimeError("published provenance did not round-trip exactly")
    for filename in ("config.json", "generation_config.json", "tokenizer_config.json"):
        remote_path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                revision=info.sha,
                token=token,
            )
        )
        if sha256(remote_path) != sha256(checkpoint / filename):
            raise RuntimeError(f"published {filename} did not round-trip exactly")
    print(json.dumps({"repo_id": args.repo_id, "revision": info.sha, **validation}, sort_keys=True))


if __name__ == "__main__":
    main()
