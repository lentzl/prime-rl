#!/usr/bin/env python3
"""Merge one audited Qwen3.5 LoRA adapter into a dense HF checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_WEIGHT = "model.safetensors"
ADAPTER_WEIGHT = "adapter_model.safetensors"
ADAPTER_PREFIX = "base_model.model."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_targets(adapter_path: Path) -> dict[str, dict[str, torch.Tensor]]:
    targets: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(adapter_path, framework="pt", device="cpu") as adapter:
        for key in adapter.keys():
            if not key.startswith(ADAPTER_PREFIX):
                raise ValueError(f"unsupported adapter tensor prefix: {key}")
            stripped = key.removeprefix(ADAPTER_PREFIX)
            if stripped.endswith(".lora_A.weight"):
                target = stripped.removesuffix(".lora_A.weight") + ".weight"
                factor = "A"
            elif stripped.endswith(".lora_B.weight"):
                target = stripped.removesuffix(".lora_B.weight") + ".weight"
                factor = "B"
            else:
                raise ValueError(f"unsupported adapter tensor: {key}")
            if factor in targets.setdefault(target, {}):
                raise ValueError(f"duplicate LoRA factor for {target}")
            targets[target][factor] = adapter.get_tensor(key)
    malformed = [target for target, factors in targets.items() if set(factors) != {"A", "B"}]
    if malformed:
        raise ValueError(f"LoRA targets lack paired A/B factors: {malformed[:3]}")
    return targets


def merge(
    *,
    base_model: Path,
    adapter_dir: Path,
    output_dir: Path,
    expected_base_sha256: str | None = None,
    expected_adapter_sha256: str | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite dense checkpoint: {output_dir}")
    base_weight = base_model / MODEL_WEIGHT
    adapter_weight = adapter_dir / ADAPTER_WEIGHT
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not (base_model / "STABLE").is_file() or not base_weight.is_file():
        raise ValueError("base model is incomplete or not marked STABLE")
    if not adapter_weight.is_file() or not adapter_config_path.is_file():
        raise ValueError("LoRA adapter is incomplete")
    base_sha256 = sha256_file(base_weight)
    adapter_sha256 = sha256_file(adapter_weight)
    if expected_base_sha256 is not None and base_sha256 != expected_base_sha256:
        raise ValueError("base model SHA-256 mismatch")
    if expected_adapter_sha256 is not None and adapter_sha256 != expected_adapter_sha256:
        raise ValueError("adapter SHA-256 mismatch")

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    rank = adapter_config.get("r")
    alpha = adapter_config.get("lora_alpha")
    if adapter_config.get("peft_type") != "LORA" or not isinstance(rank, int) or rank < 1:
        raise ValueError("unsupported LoRA adapter configuration")
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError("invalid LoRA alpha")
    scale = float(alpha) / rank
    targets = _adapter_targets(adapter_weight)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for source in base_model.iterdir():
            if source.name in {MODEL_WEIGHT, "STABLE"}:
                continue
            destination = temporary / source.name
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)

        tensors: dict[str, torch.Tensor] = {}
        with safe_open(base_weight, framework="pt", device="cpu") as base:
            base_keys = set(base.keys())
            missing = sorted(set(targets) - base_keys)
            if missing:
                raise ValueError(f"adapter targets are absent from dense base: {missing[:3]}")
            metadata = base.metadata()
            for key in base.keys():
                tensor = base.get_tensor(key)
                factors = targets.get(key)
                if factors is not None:
                    a = factors["A"]
                    b = factors["B"]
                    if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
                        raise ValueError(f"invalid LoRA factor shapes for {key}")
                    if (b.shape[0], a.shape[1]) != tuple(tensor.shape):
                        raise ValueError(f"LoRA delta shape does not match {key}")
                    delta = torch.matmul(b.float(), a.float()).mul_(scale)
                    tensor = tensor.add(delta.to(dtype=tensor.dtype))
                tensors[key] = tensor.contiguous()
        save_file(tensors, temporary / MODEL_WEIGHT, metadata=metadata)
        output_sha256 = sha256_file(temporary / MODEL_WEIGHT)
        if output_sha256 == base_sha256:
            raise ValueError("merged dense checkpoint is identical to its base")
        manifest = {
            "schema_version": "qwen35-2b-lora-dense-merge/v1",
            "base_model_path": str(base_model.resolve()),
            "base_model_sha256": base_sha256,
            "adapter_path": str(adapter_dir.resolve()),
            "adapter_sha256": adapter_sha256,
            "adapter_rank": rank,
            "adapter_alpha": float(alpha),
            "adapter_scale": scale,
            "merged_target_count": len(targets),
            "output_model_sha256": output_sha256,
        }
        (temporary / "MERGE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "STABLE").touch()
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-base-sha256")
    parser.add_argument("--expected-adapter-sha256")
    args = parser.parse_args()
    manifest = merge(
        base_model=args.base_model.resolve(),
        adapter_dir=args.adapter_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        expected_base_sha256=args.expected_base_sha256,
        expected_adapter_sha256=args.expected_adapter_sha256,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
