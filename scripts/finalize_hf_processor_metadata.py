"""Complete a trained Hugging Face checkpoint with processor metadata."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

PROCESSOR_FILES = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
)


class MetadataFailure(ValueError):
    """The source or destination cannot form a complete model artifact."""


def _read_config(directory: Path) -> dict[str, Any]:
    path = directory / "config.json"
    if not path.is_file():
        raise MetadataFailure(f"missing model config: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise MetadataFailure(f"expected a JSON object in {path}")
    return value


def finalize(source: Path, destination: Path) -> list[str]:
    """Copy immutable processor files while preserving trained model metadata."""
    if not source.is_dir() or not destination.is_dir():
        raise MetadataFailure("source and destination must be directories")
    source_config = _read_config(source)
    destination_config = _read_config(destination)
    for key in ("model_type", "architectures"):
        if source_config.get(key) != destination_config.get(key):
            raise MetadataFailure(f"source and destination disagree on {key}")

    is_multimodal = bool(destination_config.get("vision_config"))
    copied: list[str] = []
    for name in PROCESSOR_FILES:
        source_path = source / name
        destination_path = destination / name
        if not source_path.is_file():
            continue
        if destination_path.is_file() and destination_path.read_bytes() != source_path.read_bytes():
            raise MetadataFailure(f"existing processor metadata differs from source: {name}")
        if not destination_path.exists():
            shutil.copy2(source_path, destination_path)
        copied.append(name)

    if is_multimodal and not copied:
        raise MetadataFailure("multimodal checkpoint has no processor metadata")
    return copied


def validate_auto_processor(destination: Path) -> None:
    """Exercise the same local processor construction required by vLLM."""
    from transformers import AutoProcessor

    AutoProcessor.from_pretrained(
        destination,
        trust_remote_code=False,
        local_files_only=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        files = finalize(args.source, args.destination)
        validate_auto_processor(args.destination)
    except (MetadataFailure, json.JSONDecodeError, OSError, ValueError) as error:
        raise SystemExit(f"checkpoint processor finalization failed: {error}") from error
    print(json.dumps({"destination": str(args.destination), "processor_files": files}))


if __name__ == "__main__":
    main()
