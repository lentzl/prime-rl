#!/usr/bin/env python3
"""Verify that the frozen 2B/4B capacity comparison has not drifted."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 development machines
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "configs"
    / "debug"
    / "subagent-communication"
    / "frozen-capacity-battery-v1.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def comparison_semantics(path: Path, allowed_overrides: set[str]) -> dict[str, Any]:
    config = load_toml(path)
    for key in allowed_overrides:
        config.pop(key, None)
    return config


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    failures: list[str] = []

    for relative_path, expected in manifest["files"].items():
        path = ROOT / relative_path
        if not path.is_file():
            failures.append(f"missing frozen file: {relative_path}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(
                f"frozen file changed: {relative_path}\n"
                f"  expected {expected}\n"
                f"  actual   {actual}"
            )

    allowed = set(manifest["allowed_comparison_overrides"])
    for pair in manifest["paired_configs"]:
        historical = comparison_semantics(ROOT / pair["historical"], allowed)
        comparison = comparison_semantics(ROOT / pair["comparison"], allowed)
        if historical != comparison:
            failures.append(
                f"comparison semantics drifted for {pair['name']}: only "
                f"{sorted(allowed)} may differ"
            )

    if failures:
        raise SystemExit("\n".join(failures))

    print(f"{manifest['battery_id']}: {len(manifest['files'])} files verified")
    print(f"{len(manifest['paired_configs'])} model-only config pairs verified")


if __name__ == "__main__":
    main()
