import hashlib
import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[2] / "scripts" / "build_q35_2b_role_eval_manifest_v1.py"
    spec = importlib.util.spec_from_file_location("build_q35_2b_role_eval_manifest_v1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace(task_key: str, trace_id: str, model: str, score: float) -> dict:
    return {
        "traces": [
            {
                "id": trace_id,
                "ok": True,
                "errors": [],
                "is_completed": True,
                "task": {"key": task_key},
                "agent": {"config": {"model": model}},
                "rewards": {"harness_score": {"score": score}},
            }
        ]
    }


def _write_fixture(module, tmp_path: Path):
    student = tmp_path / "student"
    student.mkdir()
    (student / "STABLE").touch()
    weights = student / "model.safetensors"
    weights.write_bytes(b"immutable student")
    module.EXPECTED_STUDENT_SHA = _sha(weights)

    qualification = tmp_path / "RUNTIME-QUALIFIED.txt"
    qualification.write_text("status=qualified\nvllm_version=0.27.2+cu129\n")
    baseline_run = tmp_path / "baseline"
    baseline_run.mkdir()
    baseline_versions = baseline_run / "VERSIONS.txt"
    baseline_versions.write_text("baseline\n")
    baseline_inference = baseline_run / "inference.toml"
    baseline_inference.write_text("backend_port = 8100\n")
    baseline_axes = {}
    for axis, count in module.REQUIRED_AXES.items():
        trace_dir = baseline_run / axis
        trace_dir.mkdir()
        trace_path = trace_dir / "traces.jsonl"
        rows = [
            _trace(f"train_gen-{axis}-{index:08d}-test", f"base-{axis}-{index}", str(student), index % 2)
            for index in range(count)
        ]
        trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        baseline_axes[axis] = {"path": str(trace_path), "sha256": _sha(trace_path), "trace_count": count}
    baseline_manifest = tmp_path / "BASELINE.json"
    baseline_manifest.write_text(
        json.dumps(
            {
                "schema_version": module.BASELINE_SCHEMA_VERSION,
                "student": {
                    "snapshot": str(student),
                    "weight_path": str(weights),
                    "weight_sha256": _sha(weights),
                },
                "runtime_qualification": {"path": str(qualification), "sha256": _sha(qualification)},
                "runtime_provenance": {
                    "vllm_version": "0.27.2+cu129",
                    "vllm_distribution_url": "https://example.invalid/vllm.whl",
                    "uv_lock_sha256": "a" * 64,
                },
                "versions": {"path": str(baseline_versions), "sha256": _sha(baseline_versions)},
                "inference_config": {"path": str(baseline_inference), "sha256": _sha(baseline_inference)},
                "all_infrastructure_complete": True,
                "axes": baseline_axes,
            }
        )
    )

    step = tmp_path / "weights" / "step_1"
    adapter = step / "lora_adapters"
    adapter.mkdir(parents=True)
    (step / "STABLE").touch()
    adapter_config = adapter / "adapter_config.json"
    adapter_config.write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": str(student),
                "r": 16,
                "target_modules": sorted(module.EXPECTED_TARGET_MODULES),
                "modules_to_save": None,
            }
        )
    )
    adapter_weights = adapter / "adapter_model.safetensors"
    adapter_weights.write_bytes(b"rank16 adapter")
    lora_name = f"q35-2b-orchestrator-r16-{_sha(adapter_weights)[:12]}"

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    inference = candidate / "inference.toml"
    inference.write_text("backend_port = 8100\nenable_lora = true\n")
    harness_lines = "".join(f"{digest}  /checkout/{path}\n" for path, digest in module.REQUIRED_HARNESS_SHA256.items())
    (candidate / "VERSIONS.txt").write_text(
        f"model={lora_name}\n"
        f"base_model={student}\n"
        f"lora_name={lora_name}\n"
        f"lora_path={adapter}\n"
        "lora_rank=16\n"
        "vllm_version=0.27.2+cu129\n"
        "vllm_distribution_url=https://example.invalid/vllm.whl\n"
        f"uv_lock_sha256={'a' * 64}\n"
        f"inference_config_sha256={_sha(inference)}\n"
        f"{_sha(adapter_config)}  {adapter_config}\n"
        f"{_sha(adapter_weights)}  {adapter_weights}\n"
        f"{harness_lines}"
    )
    for axis, count in module.REQUIRED_AXES.items():
        trace_dir = candidate / axis
        trace_dir.mkdir()
        rows = [
            _trace(
                f"train_gen-{axis}-{index:08d}-test",
                f"candidate-{axis}-{index}",
                lora_name,
                (index + 1) % 2,
            )
            for index in range(count)
        ]
        (trace_dir / "traces.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return candidate, baseline_manifest, adapter


def test_role_eval_manifest_requires_same_tasks_runtime_and_unmerged_adapter(tmp_path: Path):
    module = _module()
    candidate, baseline, adapter = _write_fixture(module, tmp_path)
    manifest = module.build_manifest("orchestrator", candidate, baseline, adapter)
    assert manifest["all_infrastructure_complete"] is True
    assert manifest["all_tasks_paired"] is True
    assert manifest["adapter"]["rank"] == 16
    assert manifest["paired_strict_outcomes"] == {"gain": 32, "loss": 32}

    trace_path = candidate / "natural_n1b" / "traces.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    rows[0]["traces"][0]["task"]["key"] = "train_gen-natural_n1b-99999999-test"
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    try:
        module.build_manifest("orchestrator", candidate, baseline, adapter)
    except ValueError as error:
        assert "does not use the untouched baseline task keys" in str(error)
    else:
        raise AssertionError("an unpaired candidate task set should be rejected")
