import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _module():
    path = Path(__file__).parents[2] / "scripts" / "validate_q35_2b_role_training_inputs_v1.py"
    spec = importlib.util.spec_from_file_location("validate_q35_2b_role_training_inputs_v1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(module, tmp_path: Path):
    student = tmp_path / "student"
    student.mkdir()
    (student / "STABLE").touch()
    student_weights = student / "model.safetensors"
    student_weights.write_bytes(b"immutable student")
    module.EXPECTED_STUDENT_SHA = _sha(student_weights)

    source_root = tmp_path / "teacher"
    source_root.mkdir()
    admission_sources = []
    corpus_sources = []
    admitted_counts = {}
    parquet_rows = []
    for axis, count in module.REQUIRED_TEACHER_AXES.items():
        trace_path = source_root / f"{axis}.jsonl"
        versions_path = source_root / f"{axis}-VERSIONS.txt"
        trace_path.write_text(f"frozen {axis} traces\n")
        versions_path.write_text(f"frozen {axis} versions\n")
        accepted = [f"{axis}-trace-{index}" for index in range(count)]
        source = {
            "axis": axis,
            "trace_path": trace_path.name,
            "sha256": _sha(trace_path),
            "versions_path": versions_path.name,
            "versions_sha256": _sha(versions_path),
            "accepted_trace_ids": accepted,
        }
        admission_sources.append(source | {"runtime_provenance": {"status": "recorded"}})
        corpus_sources.append(source)
        admitted_counts[axis] = count
        parquet_rows.extend(
            {
                "axis": axis,
                "trace_id": trace_id,
                "role": "orchestrator",
                "branch_index": 0,
            }
            for trace_id in accepted
        )
    admission_path = tmp_path / "ADMISSION.json"
    admission_path.write_text(
        json.dumps(
            {
                "schema_version": module.ADMISSION_SCHEMA,
                "teacher": {"model": "Qwen/Qwen3.8-27B", "revision": module.EXPECTED_TEACHER_REVISION},
                "required_axes": module.REQUIRED_TEACHER_AXES,
                "admitted_trajectories_by_axis": admitted_counts,
                "sources": admission_sources,
            }
        )
    )

    role_dataset = tmp_path / "role-dataset"
    role_dataset.mkdir()
    parquet_path = role_dataset / "train.parquet"
    pq.write_table(pa.Table.from_pylist(parquet_rows), parquet_path)
    corpus_manifest = role_dataset / "MANIFEST.json"
    corpus_manifest.write_text(
        json.dumps(
            {
                "schema_version": module.ROLE_CORPUS_SCHEMA,
                "role": "orchestrator",
                "teacher": {"model": "Qwen/Qwen3.8-27B", "revision": module.EXPECTED_TEACHER_REVISION},
                "admitted_trajectories_by_axis": admitted_counts,
                "rows_by_axis": admitted_counts,
                "row_count": len(parquet_rows),
                "sources": corpus_sources,
                "admission_manifest": {"path": str(admission_path), "sha256": _sha(admission_path)},
                "source_root": str(source_root),
                "dataset": {"path": "train.parquet", "sha256": _sha(parquet_path)},
            }
        )
    )

    qualification = tmp_path / "RUNTIME-QUALIFIED.txt"
    qualification.write_text("status=qualified\n")
    baseline_versions = tmp_path / "BASELINE-VERSIONS.txt"
    baseline_versions.write_text("baseline versions\n")
    baseline_inference = tmp_path / "baseline-inference.toml"
    baseline_inference.write_text("backend_port = 8100\n")
    baseline_axes = {}
    for axis, count in module.REQUIRED_BASELINE_AXES.items():
        path = tmp_path / f"baseline-{axis}.jsonl"
        path.write_text(f"{axis} baseline\n")
        baseline_axes[axis] = {"path": str(path), "sha256": _sha(path), "trace_count": count}
    baseline_path = tmp_path / "BASELINE.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": module.BASELINE_SCHEMA,
                "student": {"snapshot": str(student), "weight_sha256": _sha(student_weights)},
                "runtime_qualification": {
                    "path": str(qualification),
                    "sha256": _sha(qualification),
                    "status": "qualified",
                },
                "versions": {"path": str(baseline_versions), "sha256": _sha(baseline_versions)},
                "inference_config": {"path": str(baseline_inference), "sha256": _sha(baseline_inference)},
                "all_infrastructure_complete": True,
                "axes": baseline_axes,
            }
        )
    )
    template = (
        Path(__file__).parents[2] / "experiments/qwen38-to-qwen35-2b-role-distillation-v1/orchestrator-sft-lora.toml"
    )
    return role_dataset, baseline_path, student, template


def test_training_input_validation_binds_corpus_to_admission_baseline_and_template(tmp_path: Path):
    module = _module()
    role_dataset, baseline, student, template = _write_fixture(module, tmp_path)
    result = module.validate_inputs("orchestrator", role_dataset, baseline, student, template)
    assert result["role"] == "orchestrator"
    assert result["dataset"]["row_count"] == 24

    manifest_path = role_dataset / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["admitted_trajectories_by_axis"]["natural_n1b"] = 5
    manifest_path.write_text(json.dumps(manifest))
    try:
        module.validate_inputs("orchestrator", role_dataset, baseline, student, template)
    except ValueError as error:
        assert "admitted counts do not match" in str(error)
    else:
        raise AssertionError("tampered teacher admission counts should be rejected")
