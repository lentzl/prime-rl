#!/usr/bin/env python3
"""Materialize the frozen H-ITER Phase-0 banks and structural schedules."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from prime_rl.latent.h_iter_phase0 import (
    MECHANISM,
    OVERLAP_SCHEMA,
    build_operation_schedule,
    build_probe_selection,
    build_tamper_schedule,
    canonical_json,
    canonical_sha256,
    extract_prior_source,
    generate_bank,
    new_identity_sets,
    sha256_bytes,
    strict_json_loads,
    validate_banks,
    validate_schedule,
)

A_SOURCE_COMMIT = "a8f347c9a5fdf1c2d532c6527ce169cff0000a07"
B_SOURCE_COMMIT = "4ae0308094a71d13520554da40cfe6375438b610"

A_PATHS = [
    "experiments/qwen35-2b-latent-workspace-v1/a0-cache-calibration-rejected-manifest.sha256",
    "experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0-nocache-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a0-nocache-disjointness-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-census-manifest.sha256",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-cap768-census.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-disjointness-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-held_out-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-render-rejection-manifest.sha256",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-schedule-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-train-bank-v1.json",
    "experiments/qwen35-2b-latent-workspace-v1/a1-nc0-validation-bank-v1.json",
]

B_PATHS = [
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-b1-bank-v1/MANIFEST.json",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-b1-bank-v1/heldout-selection.json",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-b1-bank-v1/heldout.parquet",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-b1-bank-v1/training-selection.json",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-b1r-negative-binding-v1.json",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-fixed-depth-smoke-a0c-br1-preflight-MANIFEST.json",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-fixed-depth-smoke-a0c-br2-failure-MANIFEST.json",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-fixed-depth-smoke-a0c-br3-failure-MANIFEST.json",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-fixed-depth-smoke-a0c-br4-failure-MANIFEST.sha256",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-fixed-depth-smoke-a0c-br5-success-MANIFEST.sha256",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-fixed-depth-smoke-v1-selection.json",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-hic0-identity-carrier-bank-v1/MANIFEST.json",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-hic0-identity-carrier-bank-v1/heldout.parquet",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-hic0-identity-carrier-bank-v1/selection.json",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-ipc1-bank-v1/MANIFEST.json",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-ipc1-bank-v1/heldout-selection.json",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-ipc1-bank-v1/heldout.parquet",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-ipc1-bank-v1/prior-overlap-closure-v1.json",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-ipc1-bank-v1/train-selection.json",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-ipc1-bank-v1/train.parquet",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-ipc1-bank-v1/validation-selection.json",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-ipc1-bank-v1/validation.parquet",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-ipc1-r1-late-incomplete-v1/MANIFEST.sha256",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-ipc1-render-proof-hygiene-v1/MANIFEST.json",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-ipc1-run2-incomplete-v1/MANIFEST.sha256",
    "experiments/qwen35-2b-latent-coordinator-v1/phase-b-ipc1-terminal-proof-v1/MANIFEST.json",
]

SCHEMAS = {
    "a0-mechanism-bank-v1.json": "prime-rl/latent-a0-mechanism-bank/v1",
    "a0-nocache-bank-v1.json": "prime-rl/latent-a0-nocache-bank/v1",
    "a0-nocache-disjointness-v1.json": "prime-rl/latent-a0-nocache-disjointness/v1",
    "a1-nc0-disjointness-v1.json": "prime-rl/latent-a1-nc0-disjointness/v1",
    "a1-nc0-schedule-v1.json": "prime-rl/latent-a1-nc0-schedule/v1",
    "a1-nc0-cap768-census.json": "a1-nc0-cap768-runtime-census/v1",
    "training-selection.json": "selection-schema-bound-by-source-path",
    "heldout-selection.json": "selection-schema-bound-by-source-path",
    "validation-selection.json": "q35-2b-b-ipc1-selection/v1",
    "selection.json": "selection-schema-bound-by-source-path",
    "MANIFEST.json": "manifest-schema-bound-by-source-path",
    "prior-overlap-closure-v1.json": "q35-2b-b-ipc1-prior-overlap-closure/v1",
}

def git_blob(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def parser_for(path: str) -> str:
    if path.endswith(".parquet"):
        return "pyarrow24_recursive_scalar_and_row_parser/v1"
    if path.endswith(".json"):
        return "strict-json-recursive-string-parser/v1"
    if path.endswith(".sha256"):
        return "sha256-manifest-line-parser/v1"
    raise ValueError(f"no frozen parser for {path}")


def source_schema(path: str, data: bytes) -> str:
    if path.endswith(".parquet"):
        return "parquet-schema-and-rows-bound-by-companion-manifest/v1"
    if path.endswith(".sha256"):
        return "sha256-manifest/v1"
    parsed = strict_json_loads(data)
    if isinstance(parsed, dict):
        for key in ("schema_version", "schema"):
            if isinstance(parsed.get(key), str):
                return parsed[key]
    return SCHEMAS.get(Path(path).name, "strict-json-source-path-bound/v1")


def build_overlap(banks: dict[str, dict]) -> dict:
    identities = new_identity_sets(banks)
    records = []
    for source_commit, paths in ((A_SOURCE_COMMIT, A_PATHS), (B_SOURCE_COMMIT, B_PATHS)):
        for path in paths:
            data = git_blob(source_commit, path)
            observed, intersection = extract_prior_source(path, data, identities)
            records.append(
                {
                    "source_commit": source_commit,
                    "source_path": path,
                    "file_sha256": sha256_bytes(data),
                    "parser": parser_for(path),
                    "schema": source_schema(path, data),
                    "observed": observed,
                    "intersection": intersection,
                }
            )
    records.sort(key=lambda item: (item["source_commit"], item["source_path"]))
    result = {
        "schema_version": OVERLAP_SCHEMA,
        "mechanism": MECHANISM,
        "source_records": records,
        "source_record_count": len(records),
        "new_identity_counts": {key: len(value) for key, value in identities.items()},
        "cross_split_intersections": {
            "row_ids": [],
            "node_ids": [],
            "nonces": [],
            "receiver_input_sha256": [],
            "complete_local_text": [],
        },
        "all_intersections_empty": all(
            not values
            for record in records
            for values in record["intersection"].values()
        ),
        "overlap_sha256": "",
    }
    result["overlap_sha256"] = canonical_sha256(result, omit="overlap_sha256")
    return result


def write_json(path: Path, value: object) -> None:
    data = canonical_json(value) + b"\n"
    path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    banks = {split: generate_bank(split) for split in ("train", "validation", "heldout")}
    validate_banks(banks)
    selection = build_probe_selection(banks)
    schedule = build_operation_schedule(selection)
    validate_schedule(schedule, selection)
    tamper = build_tamper_schedule()
    overlap = build_overlap(banks)
    for split, bank in banks.items():
        write_json(args.output_dir / f"{split}-bank.json", bank)
    write_json(args.output_dir / "locality-probe-selection.json", selection)
    write_json(args.output_dir / "operation-schedule.json", schedule)
    write_json(args.output_dir / "tamper-schedule.json", tamper)
    write_json(args.output_dir / "overlap-evidence.json", overlap)


if __name__ == "__main__":
    main()
