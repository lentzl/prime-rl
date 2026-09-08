"""Stage a pinned upstream reference without training or modifying its configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--weight-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise ValueError("reference parent directory must already exist")
    repo = "Qwen/Qwen3.5-2B"
    info = HfApi().model_info(repo, revision=args.revision, files_metadata=True)
    if info.sha != args.revision:
        raise ValueError("an exact upstream commit is required")
    weights = [file for file in info.siblings if file.rfilename.endswith(".safetensors")]
    if len(weights) != 1 or weights[0].lfs.sha256 != args.weight_sha256:
        raise ValueError("expected one weight file with the independently recorded SHA-256")
    files = [file.rfilename for file in info.siblings if (
        file.rfilename.endswith((".json", ".jinja", ".txt", ".safetensors"))
        or file.rfilename in {"LICENSE", "README.md"}
    ) and file.rfilename != "model.safetensors.index.json"]
    required = sum(file.size for file in info.siblings if file.rfilename in files) + 512 * 1024**2
    if shutil.disk_usage(output.parent).free < required:
        raise ValueError(f"need {required} free bytes before staging the reference")
    manifest = {
        "repository": repo, "revision": args.revision,
        "upstream_weight_filename": weights[0].rfilename,
        "weight_sha256": args.weight_sha256,
        "optimizer_updates": 0,
        "purpose": "upstream reference only; not a replacement or promotion of the acquired lineage",
        "exact_ancestor_identity": "not established",
        "configuration_modified": False,
        "layout": "single original weight file renamed to model.safetensors; redundant shard index omitted",
    }
    if args.dry_run:
        print(json.dumps({**manifest, "required_free_bytes": required, "files": files}, indent=2))
        return
    snapshot_download(repo, revision=args.revision, local_dir=output, allow_patterns=files)
    weight = output / weights[0].rfilename
    if sha256(weight) != args.weight_sha256:
        raise ValueError("downloaded upstream weight hash mismatch; reference is not stable")
    weight.rename(output / "model.safetensors")
    manifest["files"] = {file.name: sha256(file) for file in output.iterdir() if file.is_file()}
    (output / "REFERENCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "STABLE").write_text("Verified upstream download; zero optimizer updates.\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
