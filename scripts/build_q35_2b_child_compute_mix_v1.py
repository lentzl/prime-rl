#!/usr/bin/env python3
"""Mix operation-grounded child targets with previously verified child replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "q35-2b-child-compute-mix/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_rows(corpus: Path, *, source: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import Dataset

    manifest_path = corpus / "MANIFEST.json"
    parquet_path = corpus / "train.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = (manifest.get("dataset") or {}).get("sha256")
    actual = sha256_file(parquet_path)
    if manifest.get("status") != "complete" or expected != actual:
        raise ValueError(f"{source} corpus is incomplete or has a checksum mismatch")
    rows = [
        dict(row)
        for row in Dataset.from_parquet(str(parquet_path))
        if row.get("role") == "coordinator_nonroot"
    ]
    if not rows:
        raise ValueError(f"{source} corpus has no non-root child rows")
    return rows, {
        "path": str(corpus),
        "manifest_sha256": sha256_file(manifest_path),
        "parquet_sha256": actual,
    }


def interleave(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(max(len(left), len(right))):
        if index < len(left):
            rows.append(left[index])
        if index < len(right):
            rows.append(right[index])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-corpus", type=Path, required=True)
    parser.add_argument("--replay-corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite corpus: {args.output_dir}")

    from datasets import Dataset

    compute, compute_source = verified_rows(args.compute_corpus, source="compute")
    replay, replay_source = verified_rows(args.replay_corpus, source="replay")
    rows = interleave(compute, replay)
    args.output_dir.mkdir(parents=True)
    parquet = args.output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "objective": "operation_grounded_child_compute_with_natural_and_forced_replay",
        "row_count": len(rows),
        "compute_rows": len(compute),
        "replay_rows": len(replay),
        "root_rows": sum(row.get("role") == "coordinator_root" for row in rows),
        "resource_family_counts": dict(
            sorted(Counter(row.get("resource_family") for row in compute).items())
        ),
        "sources": {"compute": compute_source, "replay": replay_source},
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows: {len(compute)} compute + {len(replay)} replay")


if __name__ == "__main__":
    main()
