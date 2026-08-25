#!/usr/bin/env python3
"""Combine balanced admitted E0c-child and E0d-orchestrator SFT corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets
from export_prime_agent_role_sft_v1 import sha256_file

SCHEMA_VERSION = "qwen35-2b-split-frontier-corpus/v1"
SOURCE_SCHEMA_VERSION = "qwen35-2b-interaction-joint-corpus/v2"
BASELINE_SCHEMA_VERSION = "qwen35-2b-split-frontier-pretraining-baseline/v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _source(corpus: Path, *, phase: str, role: str) -> dict[str, Any]:
    manifest_path = corpus / "MANIFEST.json"
    baseline_path = corpus / "PRETRAINING-BASELINE.json"
    parquet_path = corpus / "train.parquet"
    manifest = _read_json(manifest_path)
    baseline = _read_json(baseline_path)
    if manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported source corpus: {corpus}")
    if manifest.get("selected_roles") != [role] or manifest.get("rows_by_role") != {role: 4}:
        raise ValueError(f"source corpus does not contain four {role} rows: {corpus}")
    if baseline.get("admission", {}).get("phase") != phase:
        raise ValueError(f"source corpus phase mismatch: {corpus}")
    if sha256_file(parquet_path) != manifest.get("dataset", {}).get("sha256"):
        raise ValueError(f"source parquet SHA-256 mismatch: {corpus}")
    if sha256_file(baseline_path) != manifest.get("pretraining_baseline", {}).get("sha256"):
        raise ValueError(f"source baseline SHA-256 mismatch: {corpus}")
    return {
        "corpus": corpus,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "baseline": baseline,
        "baseline_path": baseline_path,
        "parquet_path": parquet_path,
    }


def combine(
    *,
    child_corpus: Path,
    yield_corpus: Path,
    output_dir: Path,
    child_phase: str = "e0c_natural_child",
    yield_phase: str = "e0d_guided_yield",
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite combined corpus: {output_dir}")
    child = _source(child_corpus, phase=child_phase, role="child")
    yielded = _source(
        yield_corpus,
        phase=yield_phase,
        role="orchestrator",
    )
    if child["baseline"].get("student") != yielded["baseline"].get("student"):
        raise ValueError("split-frontier sources have different immutable students")
    if child["baseline"].get("initial_adapter") != yielded["baseline"].get("initial_adapter"):
        raise ValueError("split-frontier sources have different initial adapters")

    datasets = [
        Dataset.from_parquet(str(child["parquet_path"])),
        Dataset.from_parquet(str(yielded["parquet_path"])),
    ]
    if [len(dataset) for dataset in datasets] != [4, 4]:
        raise ValueError("split-frontier sources must contain four rows each")
    combined = concatenate_datasets(datasets)
    if len(combined) != 8:
        raise ValueError("combined split-frontier corpus must contain eight rows")

    output_dir.mkdir(parents=True)
    parquet_path = output_dir / "train.parquet"
    combined.to_parquet(str(parquet_path))
    sources = []
    for source in (child, yielded):
        sources.append(
            {
                "manifest_path": str(source["manifest_path"].resolve()),
                "manifest_sha256": sha256_file(source["manifest_path"]),
                "baseline_path": str(source["baseline_path"].resolve()),
                "baseline_sha256": sha256_file(source["baseline_path"]),
                "parquet_path": str(source["parquet_path"].resolve()),
                "parquet_sha256": sha256_file(source["parquet_path"]),
            }
        )
    baseline = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "student": child["baseline"]["student"],
        "initial_adapter": child["baseline"].get("initial_adapter"),
        "admission": {
            "phases": [child_phase, yield_phase],
            "selected_qualifying_trajectories_per_phase": 4,
            "acceptance_floor_relaxed": False,
            "sources": sources,
        },
    }
    baseline_path = output_dir / "PRETRAINING-BASELINE.json"
    baseline_path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "student": baseline["student"],
        "initial_adapter": baseline["initial_adapter"],
        "rows": 8,
        "rows_by_phase_and_role": {
            f"{child_phase}:child": 4,
            f"{yield_phase}:orchestrator": 4,
        },
        "sources": sources,
        "dataset": {"path": parquet_path.name, "sha256": sha256_file(parquet_path)},
        "pretraining_baseline": {
            "path": baseline_path.name,
            "sha256": sha256_file(baseline_path),
        },
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-corpus", type=Path, required=True)
    parser.add_argument("--yield-corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--child-phase", default="e0c_natural_child")
    parser.add_argument("--yield-phase", default="e0d_guided_yield")
    args = parser.parse_args()
    print(
        json.dumps(
            combine(
                child_corpus=args.child_corpus,
                yield_corpus=args.yield_corpus,
                output_dir=args.output_dir,
                child_phase=args.child_phase,
                yield_phase=args.yield_phase,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
