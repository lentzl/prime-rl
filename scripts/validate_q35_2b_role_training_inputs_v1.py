"""Validate the complete immutable input chain for one Q38-to-Q35-2B role SFT run."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROLE_CORPUS_SCHEMA = "q38-to-q35-2b-role-corpus/v1"
ADMISSION_SCHEMA = "q38-to-q35-2b-teacher-admission/v1"
BASELINE_SCHEMA = "q38-to-q35-2b-untouched-baseline/v1"
EXPECTED_STUDENT_SHA = "c75915dd41cd4fc9b1a1ef5582c6fd14913fc6f9971a58feca3b72b4bfcad406"
EXPECTED_TEACHER_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
REQUIRED_TEACHER_AXES = {
    "natural_direct_control": 12,
    "natural_n1a": 4,
    "natural_n1a_local": 4,
    "natural_n1b": 4,
}
REQUIRED_BASELINE_AXES = {axis: 16 for axis in REQUIRED_TEACHER_AXES}
EXPECTED_TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_hash(path: Path, expected: str | None, description: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{description} hash mismatch: {path}")


def same_path(recorded: str | None, expected: Path) -> bool:
    return isinstance(recorded, str) and recorded.startswith("/") and Path(recorded).resolve() == expected.resolve()


def validate_template(template_path: Path, role: str) -> dict[str, Any]:
    with template_path.open("rb") as handle:
        template = tomllib.load(handle)
    expected_run = f"q38-to-q35-2b-{role}-lora-r1"
    if template.get("max_steps") != 1:
        raise ValueError("role SFT template is not a one-step probe")
    if template.get("run", {}).get("name") != expected_run or template.get("run", {}).get("dir") != expected_run:
        raise ValueError("role SFT template has the wrong run identity")
    model = template.get("model", {})
    if model.get("name") != "__STUDENT_SNAPSHOT__":
        raise ValueError("role SFT template does not retain the student placeholder")
    if model.get("optimization_dtype") != "bfloat16" or model.get("reduce_dtype") != "bfloat16":
        raise ValueError("role SFT template changed the frozen dtypes")
    lora = model.get("lora", {})
    if lora.get("rank") != 16 or float(lora.get("alpha", 0.0)) != 32.0 or float(lora.get("dropout", -1.0)) != 0.0:
        raise ValueError("role SFT template changed the rank-16 LoRA configuration")
    if set(lora.get("target_modules") or ()) != EXPECTED_TARGET_MODULES:
        raise ValueError("role SFT template changed the LoRA target modules")
    if template.get("optim", {}).get("lr") != 1e-5:
        raise ValueError("role SFT template changed the preregistered learning rate")
    if template.get("data", {}).get("name") != "__ROLE_DATASET__":
        raise ValueError("role SFT template does not retain the role-dataset placeholder")
    if template.get("data", {}).get("loss_mask") != {
        "system": False,
        "user": False,
        "assistant": True,
        "tool": False,
    }:
        raise ValueError("role SFT template is not assistant-only")
    weights = template.get("ckpt", {}).get("weights", {})
    if weights.get("save_adapter_separately") is not True:
        raise ValueError("role SFT template does not preserve an unmerged adapter")
    return {"path": str(template_path.resolve()), "sha256": sha256_file(template_path)}


def validate_admission_chain(corpus: dict[str, Any], role: str) -> tuple[dict[str, Any], dict[str, set[str]]]:
    artifact = corpus.get("admission_manifest") or {}
    admission_path = Path(artifact.get("path", ""))
    validate_hash(admission_path, artifact.get("sha256"), "teacher admission manifest")
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if admission.get("schema_version") != ADMISSION_SCHEMA:
        raise ValueError("unsupported teacher admission manifest")
    if admission.get("teacher", {}).get("revision") != EXPECTED_TEACHER_REVISION:
        raise ValueError("teacher admission revision mismatch")
    if admission.get("required_axes") != REQUIRED_TEACHER_AXES:
        raise ValueError("teacher admission floors changed")
    source_root = Path(corpus.get("source_root", ""))
    if not source_root.is_absolute() or not source_root.is_dir():
        raise ValueError("role corpus has no absolute teacher source root")

    admitted_ids: dict[str, set[str]] = defaultdict(set)
    expected_sources: list[dict[str, Any]] = []
    for source in admission.get("sources", []):
        axis = source.get("axis")
        if axis not in REQUIRED_TEACHER_AXES:
            raise ValueError(f"unexpected teacher axis: {axis}")
        trace_path = source_root / source["trace_path"]
        versions_path = source_root / source["versions_path"]
        validate_hash(trace_path, source.get("sha256"), f"teacher {axis} traces")
        validate_hash(versions_path, source.get("versions_sha256"), f"teacher {axis} versions")
        accepted = source.get("accepted_trace_ids")
        if not isinstance(accepted, list) or len(accepted) != len(set(accepted)):
            raise ValueError(f"teacher source {axis} has invalid accepted trace IDs")
        if admitted_ids[axis].intersection(accepted):
            raise ValueError(f"teacher source {axis} repeats accepted trace IDs")
        admitted_ids[axis].update(accepted)
        expected_sources.append(
            {
                "axis": axis,
                "trace_path": source["trace_path"],
                "sha256": source["sha256"],
                "versions_path": source["versions_path"],
                "versions_sha256": source["versions_sha256"],
                "accepted_trace_ids": accepted,
            }
        )
    for axis, minimum in REQUIRED_TEACHER_AXES.items():
        if len(admitted_ids[axis]) < minimum:
            raise ValueError(f"teacher axis {axis} is below its frozen floor")
    if corpus.get("sources") != expected_sources:
        raise ValueError("role corpus source audit does not match the teacher admission manifest")
    expected_counts = {axis: len(admitted_ids[axis]) for axis in sorted(admitted_ids)}
    if corpus.get("admitted_trajectories_by_axis") != expected_counts:
        raise ValueError("role corpus admitted counts do not match the teacher admission manifest")
    return {"path": str(admission_path.resolve()), "sha256": sha256_file(admission_path)}, admitted_ids


def validate_parquet(
    dataset_path: Path,
    corpus: dict[str, Any],
    role: str,
    admitted_ids: dict[str, set[str]],
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(dataset_path, columns=["axis", "trace_id", "role", "branch_index"])
    rows = table.to_pylist()
    if len(rows) != corpus.get("row_count") or not rows:
        raise ValueError("role corpus row count does not match its Parquet dataset")
    row_counts: Counter[str] = Counter()
    seen_branches: set[tuple[str, int]] = set()
    represented: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        axis = row.get("axis")
        trace_id = row.get("trace_id")
        branch_index = row.get("branch_index")
        if row.get("role") != role:
            raise ValueError("role corpus Parquet contains the wrong role")
        if axis not in REQUIRED_TEACHER_AXES or trace_id not in admitted_ids[axis]:
            raise ValueError("role corpus Parquet contains a non-admitted trace")
        if not isinstance(branch_index, int) or (trace_id, branch_index) in seen_branches:
            raise ValueError("role corpus Parquet has invalid or duplicate branches")
        seen_branches.add((trace_id, branch_index))
        row_counts[axis] += 1
        represented[axis].add(trace_id)
    expected_row_counts = {axis: row_counts[axis] for axis in sorted(row_counts)}
    if corpus.get("rows_by_axis") != expected_row_counts:
        raise ValueError("role corpus rows_by_axis does not match its Parquet dataset")
    for axis in REQUIRED_TEACHER_AXES:
        expected = set() if role == "child" and axis == "natural_direct_control" else admitted_ids[axis]
        if represented[axis] != expected:
            raise ValueError(f"role corpus does not represent every expected {axis} trajectory")
    return {"path": str(dataset_path.resolve()), "sha256": sha256_file(dataset_path), "row_count": len(rows)}


def validate_baseline(baseline_path: Path, student: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != BASELINE_SCHEMA:
        raise ValueError("unsupported untouched-student baseline manifest")
    if not same_path(baseline.get("student", {}).get("snapshot"), student):
        raise ValueError("baseline identifies a different student snapshot")
    if baseline.get("student", {}).get("weight_sha256") != EXPECTED_STUDENT_SHA:
        raise ValueError("baseline student weight hash mismatch")
    if baseline.get("runtime_qualification", {}).get("status") != "qualified":
        raise ValueError("baseline runtime was not qualified")
    if baseline.get("all_infrastructure_complete") is not True:
        raise ValueError("untouched-student baseline is not infrastructure-complete")
    for axis, count in REQUIRED_BASELINE_AXES.items():
        if baseline.get("axes", {}).get(axis, {}).get("trace_count") != count:
            raise ValueError(f"untouched baseline axis {axis} is incomplete")
    artifacts = [
        baseline["runtime_qualification"],
        baseline["versions"],
        baseline["inference_config"],
        *(baseline["axes"][axis] for axis in REQUIRED_BASELINE_AXES),
    ]
    for artifact in artifacts:
        validate_hash(Path(artifact["path"]), artifact.get("sha256"), "baseline artifact")
    return {"path": str(baseline_path.resolve()), "sha256": sha256_file(baseline_path)}


def validate_inputs(
    role: str,
    role_dataset: Path,
    baseline_path: Path,
    student: Path,
    template_path: Path,
) -> dict[str, Any]:
    if role not in {"orchestrator", "child"}:
        raise ValueError(f"unsupported role: {role}")
    role_dataset = role_dataset.resolve()
    student = student.resolve()
    if not (student / "STABLE").is_file() or not (student / "model.safetensors").is_file():
        raise ValueError("student snapshot is not a stable dense export")
    if sha256_file(student / "model.safetensors") != EXPECTED_STUDENT_SHA:
        raise ValueError("student weight hash mismatch")
    manifest_path = role_dataset / "MANIFEST.json"
    dataset_path = role_dataset / "train.parquet"
    if not manifest_path.is_file() or not dataset_path.is_file():
        raise ValueError("role dataset is incomplete")
    corpus = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corpus.get("schema_version") != ROLE_CORPUS_SCHEMA or corpus.get("role") != role:
        raise ValueError("role corpus schema or role mismatch")
    if corpus.get("teacher", {}).get("revision") != EXPECTED_TEACHER_REVISION:
        raise ValueError("role corpus teacher revision mismatch")
    if corpus.get("dataset", {}).get("path") != "train.parquet":
        raise ValueError("role corpus points to an unexpected dataset path")
    validate_hash(dataset_path, corpus.get("dataset", {}).get("sha256"), "role dataset")
    admission, admitted_ids = validate_admission_chain(corpus, role)
    parquet = validate_parquet(dataset_path, corpus, role, admitted_ids)
    baseline = validate_baseline(baseline_path.resolve(), student)
    template = validate_template(template_path.resolve(), role)
    return {
        "role": role,
        "student": {
            "path": str(student),
            "weight_sha256": EXPECTED_STUDENT_SHA,
        },
        "role_corpus_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "dataset": parquet,
        "teacher_admission": admission,
        "untouched_baseline": baseline,
        "training_template": template,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("orchestrator", "child"), required=True)
    parser.add_argument("--role-dataset", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--student-snapshot", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    args = parser.parse_args()
    result = validate_inputs(
        args.role,
        args.role_dataset,
        args.baseline_manifest,
        args.student_snapshot,
        args.template,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
