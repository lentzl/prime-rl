"""Build a fail-closed admission manifest for the Q38 role-distillation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "q38-to-q35-2b-teacher-admission/v1"
TEACHER_MODEL = "Qwen/Qwen3.8-27B"
TEACHER_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
REQUIRED_AXES = {
    "natural_direct_control": 12,
    "natural_n1a": 4,
    "natural_n1a_local": 4,
    "natural_n1b": 4,
}
DIRECT_RESOURCE_MARKERS = (
    "completion_gate",
    "/workspace/",
    "pathlib",
    "subprocess",
    "os.",
    "open(",
)
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


def hard_success(trace: dict[str, Any]) -> bool:
    reward = (trace.get("rewards") or {}).get("harness_score") or {}
    score = reward.get("score", reward.get("value"))
    return (
        trace.get("ok") is True
        and not trace.get("errors")
        and float(score or 0.0) == 1.0
        and float((trace.get("metrics") or {}).get("final_answer_exact", 0.0)) == 1.0
    )


def direct_rejection_reason(trace: dict[str, Any]) -> str | None:
    if trace.get("ok") is not True or trace.get("errors"):
        return "errored"
    if float((trace.get("metrics") or {}).get("final_answer_exact", 0.0)) != 1.0:
        return "inexact_final_answer"
    roots = [node for node in trace.get("nodes", []) if node.get("parent") is None]
    if len(roots) != 1:
        return "delegated_or_multiroot"

    tool_calls: list[dict[str, Any]] = []
    for node in trace.get("nodes", []):
        if node.get("sampled") is not True:
            continue
        tool_calls.extend((node.get("message") or {}).get("tool_calls") or [])
    if any(call.get("name") != "ipython" for call in tool_calls):
        return "nonlocal_tool"
    if len(tool_calls) > 1:
        return "repeated_local_computation"
    if tool_calls:
        arguments = str(tool_calls[0].get("arguments", "")).lower()
        if any(marker in arguments for marker in DIRECT_RESOURCE_MARKERS):
            return "unrelated_resource_access"
    return None


def parse_source(value: str) -> tuple[str, Path]:
    axis, separator, path = value.partition("=")
    if not separator or axis not in REQUIRED_AXES or not path:
        raise argparse.ArgumentTypeError("source must be REQUIRED_AXIS=RELATIVE_TRACE_PATH")
    return axis, Path(path)


def validate_harness_provenance(versions_path: Path, lines: list[str]) -> None:
    recorded: dict[str, str] = {}
    for line in lines:
        digest, separator, recorded_path = line.partition("  ")
        if not separator or len(digest) != 64:
            continue
        try:
            int(digest, 16)
        except ValueError:
            continue
        for required_path in REQUIRED_HARNESS_SHA256:
            if recorded_path.endswith(required_path):
                recorded[required_path] = digest

    missing = sorted(set(REQUIRED_HARNESS_SHA256) - set(recorded))
    if missing:
        raise ValueError(f"{versions_path} lacks frozen harness hashes for {', '.join(missing)}")
    mismatched = sorted(path for path, expected in REQUIRED_HARNESS_SHA256.items() if recorded[path] != expected)
    if mismatched:
        raise ValueError(f"{versions_path} has non-frozen harness hashes for {', '.join(mismatched)}")


def source_versions(source_root: Path, trace_path: Path) -> tuple[Path, dict[str, str]]:
    try:
        relative_trace = trace_path.resolve().relative_to(source_root.resolve())
    except ValueError as error:
        raise ValueError(f"trace path escapes source root: {trace_path}") from error

    for parent in (trace_path.parent, *trace_path.parents):
        try:
            parent.resolve().relative_to(source_root.resolve())
        except ValueError:
            break
        versions_path = parent / "VERSIONS.txt"
        if not versions_path.is_file():
            continue
        values: dict[str, str] = {}
        lines = versions_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            key, separator, value = line.partition("=")
            if separator and key:
                values[key] = value
        if values.get("model") != TEACHER_MODEL:
            raise ValueError(f"{versions_path} does not pin teacher model {TEACHER_MODEL}")
        if values.get("model_revision") != TEACHER_REVISION:
            raise ValueError(f"{versions_path} does not pin teacher revision {TEACHER_REVISION}")
        validate_harness_provenance(versions_path, lines)
        return versions_path.relative_to(source_root), values
    raise ValueError(f"no VERSIONS.txt provenance found for {relative_trace}")


def runtime_provenance(values: dict[str, str]) -> dict[str, str]:
    present = [key for key in RUNTIME_PROVENANCE_KEYS if values.get(key)]
    if not present:
        return {"status": "legacy_unrecorded"}
    if len(present) != len(RUNTIME_PROVENANCE_KEYS):
        missing = sorted(set(RUNTIME_PROVENANCE_KEYS) - set(present))
        raise ValueError(f"partial runtime provenance; missing {', '.join(missing)}")
    return {"status": "recorded", **{key: values[key] for key in RUNTIME_PROVENANCE_KEYS}}


def build_manifest(source_root: Path, sources: list[tuple[str, Path]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    accepted_task_keys: set[str] = set()
    accepted_trace_ids: set[str] = set()
    source_records: list[dict[str, Any]] = []

    for axis, relative_path in sources:
        trace_path = source_root / relative_path
        versions_path, version_values = source_versions(source_root, trace_path)
        accepted: list[str] = []
        rejected: dict[str, str] = {}
        with trace_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                episode = json.loads(line)
                traces = episode.get("traces")
                if not isinstance(traces, list) or len(traces) != 1:
                    raise ValueError(f"{trace_path}:{line_number} must contain one teacher trace")
                trace = traces[0]
                trace_id = trace.get("id")
                if not isinstance(trace_id, str):
                    raise ValueError(f"{trace_path}:{line_number} has no trace ID")
                if trace.get("agent", {}).get("config", {}).get("model") != TEACHER_MODEL:
                    raise ValueError(f"trace {trace_id} was not sampled from {TEACHER_MODEL}")
                reason = (
                    direct_rejection_reason(trace)
                    if axis == "natural_direct_control"
                    else None
                    if hard_success(trace)
                    else "not_complete_hard_success"
                )
                if reason is not None:
                    rejected[trace_id] = reason
                    continue
                task_key = trace.get("task", {}).get("key")
                if not isinstance(task_key, str):
                    raise ValueError(f"admitted trace {trace_id} has no task key")
                if trace_id in accepted_trace_ids or task_key in accepted_task_keys:
                    raise ValueError(f"duplicate admitted trace/task identity: {trace_id}/{task_key}")
                accepted_trace_ids.add(trace_id)
                accepted_task_keys.add(task_key)
                accepted.append(trace_id)
                counts[axis] += 1
        source_records.append(
            {
                "axis": axis,
                "trace_path": relative_path.as_posix(),
                "sha256": sha256_file(trace_path),
                "versions_path": versions_path.as_posix(),
                "versions_sha256": sha256_file(source_root / versions_path),
                "runtime_provenance": runtime_provenance(version_values),
                "accepted_trace_ids": accepted,
                "rejected_trace_ids": rejected,
            }
        )

    for axis, minimum in REQUIRED_AXES.items():
        if counts[axis] < minimum:
            raise ValueError(f"axis {axis} has {counts[axis]} admitted trajectories; requires {minimum}")
    return {
        "schema_version": SCHEMA_VERSION,
        "teacher": {"model": TEACHER_MODEL, "revision": TEACHER_REVISION},
        "required_axes": REQUIRED_AXES,
        "admitted_trajectories_by_axis": dict(sorted(counts.items())),
        "direct_policy": (
            "clean in-context and one clean coordinator-local IPython calculation are valid; "
            "delegation, repeated work, and unrelated resource access are rejected"
        ),
        "sources": source_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite admission manifest: {args.output}")
    manifest = build_manifest(args.source_root, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["admitted_trajectories_by_axis"], sort_keys=True))


if __name__ == "__main__":
    main()
