import hashlib
import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[2] / "scripts" / "build_q35_2b_baseline_manifest_v1.py"
    spec = importlib.util.spec_from_file_location("build_q35_2b_baseline_manifest_v1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_baseline(module, tmp_path):
    snapshot = tmp_path / "student"
    snapshot.mkdir()
    (snapshot / "STABLE").touch()
    weight_path = snapshot / "model.safetensors"
    weight_path.write_bytes(b"immutable student")
    module.EXPECTED_STUDENT_SHA = hashlib.sha256(weight_path.read_bytes()).hexdigest()

    qualification = tmp_path / "RUNTIME-QUALIFIED.txt"
    qualification.write_text("status=qualified\nvllm_version=0.27.2+cu129\n", encoding="utf-8")
    run = tmp_path / "baseline"
    run.mkdir()
    inference = run / "inference.toml"
    inference.write_text("backend_port = 8100\n", encoding="utf-8")
    harness_lines = "".join(f"{digest}  /checkout/{path}\n" for path, digest in module.REQUIRED_HARNESS_SHA256.items())
    (run / "VERSIONS.txt").write_text(
        f"model={snapshot}\n"
        "model_revision=local\n"
        "vllm_version=0.27.2+cu129\n"
        "vllm_distribution_url=https://example.invalid/vllm.whl\n"
        f"uv_lock_sha256={'a' * 64}\n"
        f"inference_config_sha256={hashlib.sha256(inference.read_bytes()).hexdigest()}\n"
        f"{harness_lines}",
        encoding="utf-8",
    )
    for axis, count in module.REQUIRED_AXES.items():
        trace_dir = run / axis
        trace_dir.mkdir()
        lines = []
        for index in range(count):
            trace = {
                "id": f"{axis}-{index}",
                "ok": True,
                "errors": [],
                "is_completed": True,
                "task": {"key": f"train_gen-{axis}-{index:08d}-test"},
                "agent": {"config": {"model": str(snapshot)}},
                "rewards": {"harness_score": {"score": float(index % 2)}},
            }
            lines.append(json.dumps({"traces": [trace]}))
        (trace_dir / "traces.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run, snapshot, qualification


def test_baseline_manifest_requires_complete_frozen_baseline(tmp_path):
    module = _module()
    run, snapshot, qualification = _write_baseline(module, tmp_path)
    manifest = module.build_manifest(run, snapshot, qualification)
    assert manifest["all_infrastructure_complete"] is True
    assert {axis: record["trace_count"] for axis, record in manifest["axes"].items()} == (module.REQUIRED_AXES)

    qualification.write_text("status=qualified\nvllm_version=other-runtime\n", encoding="utf-8")
    try:
        module.build_manifest(run, snapshot, qualification)
    except ValueError as error:
        assert "does not match the qualified runtime" in str(error)
    else:
        raise AssertionError("baseline from a different runtime should be rejected")
    qualification.write_text("status=qualified\nvllm_version=0.27.2+cu129\n", encoding="utf-8")

    trace_path = run / "natural_n1b" / "traces.jsonl"
    episodes = [json.loads(line) for line in trace_path.read_text().splitlines()]
    episodes[0]["traces"][0]["ok"] = False
    trace_path.write_text("\n".join(json.dumps(row) for row in episodes) + "\n", encoding="utf-8")
    try:
        module.build_manifest(run, snapshot, qualification)
    except ValueError as error:
        assert "not infrastructure-complete" in str(error)
    else:
        raise AssertionError("incomplete untouched baseline should be rejected")
