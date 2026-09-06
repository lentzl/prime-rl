#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path

from prime_rl.latent.h_iter_phase1_mf0 import (
    ARTIFACT_DIR,
    ASSET_NAMES,
    TRAIN_BANK_FILE_SHA256,
    TRAIN_BANK_INTERNAL_SHA256,
    TRAIN_BANK_PATH,
    build_assets,
    canonical_json,
    validate_assets,
)


def strict_load(path: Path) -> dict:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != TRAIN_BANK_FILE_SHA256:
        raise RuntimeError("MF0 train bank file hash differs")
    value = json.loads(data)
    if value.get("split") != "train" or value.get("bank_sha256") != TRAIN_BANK_INTERNAL_SHA256:
        raise RuntimeError("MF0 train bank identity differs")
    if canonical_json(value) + b"\n" != data:
        raise RuntimeError("MF0 train bank is not canonical")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    bank = strict_load(repo / TRAIN_BANK_PATH)
    assets = build_assets(bank)
    validate_assets(assets, bank)
    output = repo / ARTIFACT_DIR
    output.mkdir(parents=True, exist_ok=True)
    for name in ASSET_NAMES:
        target = output / name
        encoded = canonical_json(assets[name]) + b"\n"
        temporary = output / f".{name}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short MF0 asset write")
                view = view[written:]
            os.fsync(descriptor)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
    descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    regenerated = build_assets(strict_load(repo / TRAIN_BANK_PATH))
    validate_assets(regenerated, strict_load(repo / TRAIN_BANK_PATH))
    for name in ASSET_NAMES:
        if (output / name).read_bytes() != canonical_json(regenerated[name]) + b"\n":
            raise RuntimeError(f"MF0 asset changed after publication: {name}")


if __name__ == "__main__":
    main()
