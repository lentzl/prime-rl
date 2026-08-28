"""Freeze a completed untouched Qwen3.5-2B baseline for the distillation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "q38-to-q35-2b-untouched-baseline/v1"
EXPECTED_STUDENT_SHA = "c75915dd41cd4fc9b1a1ef5582c6fd14913fc6f9971a58feca3b72b4bfcad406"
REQUIRED_AXES = {
    "natural_direct_control": 16,
    "natural_n1a": 16,
    "natural_n1a_local": 16,
    "natural_n1b": 16,
}
RUNTIME_PROVENANCE_KEYS = (
    "vllm_version",
    "vllm_distribution_url",
    "uv_lock_sha256",
    "inference_config_sha256",
)
REQUIRED_HARNESS_SHA256 = {
    "experiments/qwen38-27b-prime-harness-qualification-v1/qualification-template.toml": (
        "ddf1a871696f0f639983c88f77d2071c75386e43dbfa6fe2b8d048c150ef3778"
    ),
    "deps/verifiers/verifiers/v1/dialects/chat.py": (
        "33f147fb9d0d2a20b2c25f8a70a75c759983000998bcd0470e58a433c5fb3dd0"
    ),
    "deps/verifiers/datasets/procedural_harness_master_v1/generate.py": (
        "62a7717fc4100697e8810a29ab67830f116d2404114a1b67a0349134a34bcb30"
    ),
    "deps/verifiers/environments/procedural_harness_master_v1/procedural_harness_master_v1/taskset.py": (
        "bbf0b1d2ea8ed77b83ff27f1aaef98e49acadc4ed6a02d1be30e036440ffa417"
    ),
    "deps/verifiers/environments/procedural_harness_master_v1/"
    "procedural_harness_master_v1/causal_context_boundary.py": (
        "ad1e73d6432e4d741226e969525b0ab98826e7fdd769765e7e443ac90544f50a"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_versions(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    values: dict[str, str] = {}
    recorded_hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            values[key] = value
            continue
        digest, separator, recorded_path = line.partition("  ")
        if not separator or len(digest) != 64:
            continue
        try:
            int(digest, 16)
        except ValueError:
            continue
        for required_path in REQUIRED_HARNESS_SHA256:
            if recorded_path.endswith(required_path):
                recorded_hashes[required_path] = digest
    return values, recorded_hashes


def validate_harness_hashes(versions_path: Path, recorded: dict[str, str]) -> None:
    missing = sorted(set(REQUIRED_HARNESS_SHA256) - set(recorded))
    if missing:
        raise ValueError(f"{versions_path} lacks frozen harness hashes for {', '.join(missing)}")
    mismatched = sorted(path for path, expected in REQUIRED_HARNESS_SHA256.items() if recorded[path] != expected)
    if mismatched:
        raise ValueError(f"{versions_path} has non-frozen harness hashes for {', '.join(mismatched)}")


def same_model_reference(recorded: str | None, snapshot: Path) -> bool:
    if not recorded:
        return False
    if recorded == str(snapshot):
        return True
    return recorded.startswith("/") and Path(recorded).resolve() == snapshot.resolve()


def qualification_metadata(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("runtime qualification JSON must be an object")
        return payload
    return dict(line.partition("=")[::2] for line in text.splitlines() if "=" in line and line.partition("=")[0])


def build_manifest(baseline_run: Path, student_snapshot: Path, runtime_qualification: Path) -> dict[str, Any]:
    baseline_run = baseline_run.resolve()
    student_snapshot = student_snapshot.resolve()
    runtime_qualification = runtime_qualification.resolve()
    weight_path = student_snapshot / "model.safetensors"
    if not (student_snapshot / "STABLE").is_file() or not weight_path.is_file():
        raise ValueError(f"student snapshot is not a stable dense export: {student_snapshot}")
    weight_sha = sha256_file(weight_path)
    if weight_sha != EXPECTED_STUDENT_SHA:
        raise ValueError(f"student weight hash mismatch: {weight_sha}")
    if not runtime_qualification.is_file():
        raise ValueError(f"runtime qualification manifest is missing: {runtime_qualification}")
    qualification = qualification_metadata(runtime_qualification)
    if qualification.get("status") != "qualified":
        raise ValueError("runtime qualification manifest is not qualified")
    if not qualification.get("vllm_version"):
        raise ValueError("runtime qualification manifest has no vLLM version")

    versions_path = baseline_run / "VERSIONS.txt"
    inference_path = baseline_run / "inference.toml"
    if not versions_path.is_file() or not inference_path.is_file():
        raise ValueError(f"baseline run lacks VERSIONS.txt or inference.toml: {baseline_run}")
    versions, harness_hashes = parse_versions(versions_path)
    validate_harness_hashes(versions_path, harness_hashes)
    if not same_model_reference(versions.get("model"), student_snapshot):
        raise ValueError("baseline VERSIONS.txt does not identify the immutable student snapshot")
    missing_runtime = sorted(key for key in RUNTIME_PROVENANCE_KEYS if not versions.get(key))
    if missing_runtime:
        raise ValueError(f"baseline runtime provenance is missing {', '.join(missing_runtime)}")
    if versions["inference_config_sha256"] != sha256_file(inference_path):
        raise ValueError("baseline inference config hash does not match inference.toml")
    if versions["vllm_version"] != qualification["vllm_version"]:
        raise ValueError("baseline vLLM version does not match the qualified runtime")

    accepted_trace_ids: set[str] = set()
    accepted_task_keys: set[str] = set()
    axis_records: dict[str, Any] = {}
    for axis, required_count in REQUIRED_AXES.items():
        trace_path = baseline_run / axis / "traces.jsonl"
        if not trace_path.is_file():
            raise ValueError(f"baseline trace file is missing: {trace_path}")
        trace_ids: list[str] = []
        score_counts: Counter[str] = Counter()
        with trace_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                episode = json.loads(line)
                traces = episode.get("traces")
                if not isinstance(traces, list) or len(traces) != 1:
                    raise ValueError(f"{trace_path}:{line_number} must contain one trace")
                trace = traces[0]
                trace_id = trace.get("id")
                task_key = (trace.get("task") or {}).get("key")
                if not isinstance(trace_id, str) or not isinstance(task_key, str):
                    raise ValueError(f"{trace_path}:{line_number} lacks trace/task identity")
                if trace_id in accepted_trace_ids or task_key in accepted_task_keys:
                    raise ValueError(f"duplicate baseline trace/task identity: {trace_id}/{task_key}")
                if f"-{axis}-" not in task_key:
                    raise ValueError(f"baseline axis/task mismatch: {axis}/{task_key}")
                if not same_model_reference(
                    ((trace.get("agent") or {}).get("config") or {}).get("model"),
                    student_snapshot,
                ):
                    raise ValueError(f"trace {trace_id} was not sampled from the immutable student")
                if trace.get("ok") is not True or trace.get("errors") or trace.get("is_completed") is not True:
                    raise ValueError(f"baseline trace {trace_id} is not infrastructure-complete")
                accepted_trace_ids.add(trace_id)
                accepted_task_keys.add(task_key)
                trace_ids.append(trace_id)
                reward = (trace.get("rewards") or {}).get("harness_score") or {}
                score_counts[str(float(reward.get("score", reward.get("value", 0.0))))] += 1
        if len(trace_ids) != required_count:
            raise ValueError(f"baseline axis {axis} has {len(trace_ids)} complete traces; requires {required_count}")
        axis_records[axis] = {
            "trace_path": str(trace_path),
            "sha256": sha256_file(trace_path),
            "trace_count": len(trace_ids),
            "trace_ids": trace_ids,
            "harness_score_counts": dict(sorted(score_counts.items())),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "student": {
            "snapshot": str(student_snapshot),
            "weight_path": str(weight_path),
            "weight_sha256": weight_sha,
        },
        "runtime_qualification": {
            "path": str(runtime_qualification),
            "sha256": sha256_file(runtime_qualification),
            "status": "qualified",
            "vllm_version": qualification["vllm_version"],
        },
        "runtime_provenance": {key: versions[key] for key in RUNTIME_PROVENANCE_KEYS},
        "baseline_run": str(baseline_run),
        "versions": {"path": str(versions_path), "sha256": sha256_file(versions_path)},
        "inference_config": {
            "path": str(inference_path),
            "sha256": sha256_file(inference_path),
        },
        "required_axes": REQUIRED_AXES,
        "all_infrastructure_complete": True,
        "axes": axis_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--student-snapshot", type=Path, required=True)
    parser.add_argument("--runtime-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite baseline manifest: {args.output}")
    manifest = build_manifest(args.baseline_run, args.student_snapshot, args.runtime_qualification)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({axis: record["trace_count"] for axis, record in manifest["axes"].items()}))


if __name__ == "__main__":
    main()
