"""Freeze a paired unmerged-LoRA evaluation against the untouched 2B baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "q38-to-q35-2b-role-evaluation/v1"
BASELINE_SCHEMA_VERSION = "q38-to-q35-2b-untouched-baseline/v1"
EXPECTED_STUDENT_SHA = "c75915dd41cd4fc9b1a1ef5582c6fd14913fc6f9971a58feca3b72b4bfcad406"
EXPECTED_TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
REQUIRED_AXES = {
    "natural_direct_control": 16,
    "natural_n1a": 16,
    "natural_n1a_local": 16,
    "natural_n1b": 16,
}
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


def same_path(recorded: str | None, expected: Path) -> bool:
    return isinstance(recorded, str) and recorded.startswith("/") and Path(recorded).resolve() == expected.resolve()


def parse_versions(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    values: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            values[key] = value
            continue
        digest, separator, recorded_path = line.partition("  ")
        if separator and len(digest) == 64:
            try:
                int(digest, 16)
            except ValueError:
                continue
            hashes[recorded_path] = digest
    return values, hashes


def find_recorded_hash(hashes: dict[str, str], path: Path) -> str | None:
    expected = path.resolve()
    for recorded, digest in hashes.items():
        if Path(recorded).resolve() == expected:
            return digest
    return None


def validate_harness_hashes(versions_path: Path, hashes: dict[str, str]) -> None:
    for suffix, expected in REQUIRED_HARNESS_SHA256.items():
        matches = [digest for path, digest in hashes.items() if path.endswith(suffix)]
        if matches != [expected]:
            raise ValueError(f"{versions_path} lacks the frozen hash for {suffix}")


def validate_artifact(artifact: dict[str, Any], description: str) -> Path:
    path = Path(artifact.get("path", ""))
    if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
        raise ValueError(f"{description} hash mismatch: {path}")
    return path


def load_axis_traces(trace_path: Path, expected_model: str, axis: str) -> dict[str, dict[str, Any]]:
    traces_by_task: dict[str, dict[str, Any]] = {}
    with trace_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            episode = json.loads(line)
            traces = episode.get("traces")
            if not isinstance(traces, list) or len(traces) != 1:
                raise ValueError(f"{trace_path}:{line_number} must contain one trace")
            trace = traces[0]
            task_key = (trace.get("task") or {}).get("key")
            model = ((trace.get("agent") or {}).get("config") or {}).get("model")
            if not isinstance(task_key, str) or f"-{axis}-" not in task_key:
                raise ValueError(f"{trace_path}:{line_number} has the wrong task axis")
            if task_key in traces_by_task:
                raise ValueError(f"duplicate task key in {trace_path}: {task_key}")
            if model != expected_model:
                raise ValueError(f"trace for {task_key} used model {model!r}, not {expected_model!r}")
            if trace.get("ok") is not True or trace.get("errors") or trace.get("is_completed") is not True:
                raise ValueError(f"trace for {task_key} is not infrastructure-complete")
            reward = (trace.get("rewards") or {}).get("harness_score") or {}
            score = float(reward.get("score", reward.get("value", 0.0)))
            traces_by_task[task_key] = {
                "trace_id": trace.get("id"),
                "score": score,
                "strict_success": score >= 1.0,
            }
    return traces_by_task


def validate_adapter(adapter: Path, student: Path) -> dict[str, Any]:
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    if not (adapter.parent / "STABLE").is_file() or not config_path.is_file() or not weights_path.is_file():
        raise ValueError(f"adapter is not a complete stable Prime-RL checkpoint: {adapter}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("adapter is not a causal-LM LoRA")
    if config.get("r") != 16:
        raise ValueError("adapter is not rank 16")
    if not same_path(config.get("base_model_name_or_path"), student):
        raise ValueError("adapter identifies a different dense base")
    if set(config.get("target_modules") or ()) != EXPECTED_TARGET_MODULES:
        raise ValueError("adapter target modules do not match the preregistered topology")
    if config.get("modules_to_save") not in (None, []):
        raise ValueError("adapter contains non-LoRA trainable modules")
    return {
        "path": str(adapter.resolve()),
        "rank": 16,
        "config": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)},
        "weights": {"path": str(weights_path.resolve()), "sha256": sha256_file(weights_path)},
    }


def build_manifest(role: str, candidate_run: Path, baseline_path: Path, adapter_path: Path) -> dict[str, Any]:
    if role not in {"orchestrator", "child"}:
        raise ValueError(f"unsupported role: {role}")
    candidate_run = candidate_run.resolve()
    baseline_path = baseline_path.resolve()
    adapter_path = adapter_path.resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError("unsupported untouched-baseline manifest")
    if baseline.get("student", {}).get("weight_sha256") != EXPECTED_STUDENT_SHA:
        raise ValueError("baseline does not identify the immutable student")
    if baseline.get("all_infrastructure_complete") is not True:
        raise ValueError("untouched baseline is not infrastructure-complete")
    student = Path(baseline["student"]["snapshot"]).resolve()
    if sha256_file(student / "model.safetensors") != EXPECTED_STUDENT_SHA:
        raise ValueError("immutable student weight file has changed")
    validate_artifact(baseline["runtime_qualification"], "runtime qualification")
    validate_artifact(baseline["versions"], "baseline versions")
    validate_artifact(baseline["inference_config"], "baseline inference config")
    adapter = validate_adapter(adapter_path, student)

    versions_path = candidate_run / "VERSIONS.txt"
    inference_path = candidate_run / "inference.toml"
    if not versions_path.is_file() or not inference_path.is_file():
        raise ValueError("candidate run lacks VERSIONS.txt or inference.toml")
    versions, recorded_hashes = parse_versions(versions_path)
    validate_harness_hashes(versions_path, recorded_hashes)
    lora_name = versions.get("lora_name")
    if not lora_name or versions.get("model") != lora_name:
        raise ValueError("candidate traces were not addressed through the recorded LoRA name")
    if not same_path(versions.get("base_model"), student):
        raise ValueError("candidate run used a different dense base")
    if not same_path(versions.get("lora_path"), adapter_path) or versions.get("lora_rank") != "16":
        raise ValueError("candidate run used a different adapter or rank")
    for artifact in (adapter["config"], adapter["weights"]):
        path = Path(artifact["path"])
        if find_recorded_hash(recorded_hashes, path) != artifact["sha256"]:
            raise ValueError(f"candidate VERSIONS.txt has the wrong adapter hash for {path}")
    for key in ("vllm_version", "vllm_distribution_url", "uv_lock_sha256"):
        if versions.get(key) != baseline.get("runtime_provenance", {}).get(key):
            raise ValueError(f"candidate {key} does not match the untouched baseline")
    if versions.get("inference_config_sha256") != sha256_file(inference_path):
        raise ValueError("candidate inference config hash mismatch")

    axes: dict[str, Any] = {}
    totals: Counter[str] = Counter()
    for axis, expected_count in REQUIRED_AXES.items():
        baseline_trace_path = validate_artifact(baseline["axes"][axis], f"baseline {axis} traces")
        candidate_trace_path = candidate_run / axis / "traces.jsonl"
        if not candidate_trace_path.is_file():
            raise ValueError(f"candidate trace file is missing: {candidate_trace_path}")
        baseline_traces = load_axis_traces(baseline_trace_path, str(student), axis)
        candidate_traces = load_axis_traces(candidate_trace_path, lora_name, axis)
        if len(baseline_traces) != expected_count or len(candidate_traces) != expected_count:
            raise ValueError(f"axis {axis} does not contain exactly {expected_count} paired traces")
        if baseline_traces.keys() != candidate_traces.keys():
            raise ValueError(f"candidate axis {axis} does not use the untouched baseline task keys")
        paired = Counter[str]()
        for task_key in baseline_traces:
            before = baseline_traces[task_key]["strict_success"]
            after = candidate_traces[task_key]["strict_success"]
            outcome = "unchanged_pass" if before and after else "unchanged_fail"
            if not before and after:
                outcome = "gain"
            elif before and not after:
                outcome = "loss"
            paired[outcome] += 1
            totals[outcome] += 1
        axes[axis] = {
            "trace_path": str(candidate_trace_path),
            "sha256": sha256_file(candidate_trace_path),
            "trace_count": len(candidate_traces),
            "task_keys": sorted(candidate_traces),
            "paired_strict_outcomes": dict(sorted(paired.items())),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "role": role,
        "student": baseline["student"],
        "adapter": adapter,
        "lora_name": lora_name,
        "baseline_manifest": {"path": str(baseline_path), "sha256": sha256_file(baseline_path)},
        "candidate_run": str(candidate_run),
        "versions": {"path": str(versions_path), "sha256": sha256_file(versions_path)},
        "inference_config": {"path": str(inference_path), "sha256": sha256_file(inference_path)},
        "all_infrastructure_complete": True,
        "all_tasks_paired": True,
        "paired_strict_outcomes": dict(sorted(totals.items())),
        "axes": axes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("orchestrator", "child"), required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite role evaluation manifest: {args.output}")
    manifest = build_manifest(args.role, args.candidate_run, args.baseline_manifest, args.adapter_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["paired_strict_outcomes"], sort_keys=True))


if __name__ == "__main__":
    main()
