import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[2] / "scripts" / "build_q38_teacher_admission_v1.py"
    spec = importlib.util.spec_from_file_location("build_q38_teacher_admission_v1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _direct_trace(tool_calls=None):
    return {
        "ok": True,
        "errors": [],
        "metrics": {"final_answer_exact": 1.0},
        "nodes": [
            {"parent": None, "sampled": False, "message": {"role": "user", "content": "task"}},
            {
                "parent": 0,
                "sampled": True,
                "message": {"role": "assistant", "content": "answer", "tool_calls": tool_calls},
            },
        ],
    }


def test_direct_policy_accepts_mental_and_one_clean_ipython_call():
    module = _module()
    assert module.direct_rejection_reason(_direct_trace()) is None
    assert (
        module.direct_rejection_reason(_direct_trace([{"name": "ipython", "arguments": '{"code":"print(7 * 8)"}'}]))
        is None
    )


def test_direct_policy_rejects_repeated_work_and_gate_access():
    module = _module()
    assert (
        module.direct_rejection_reason(
            _direct_trace(
                [
                    {"name": "ipython", "arguments": '{"code":"print(7 * 8)"}'},
                    {"name": "ipython", "arguments": '{"code":"print(56)"}'},
                ]
            )
        )
        == "repeated_local_computation"
    )
    assert (
        module.direct_rejection_reason(
            _direct_trace(
                [
                    {
                        "name": "ipython",
                        "arguments": '{"code":"print(open(\'/workspace/completion_gate.py\').read())"}',
                    }
                ]
            )
        )
        == "unrelated_resource_access"
    )


def test_source_versions_requires_exact_teacher_revision(tmp_path):
    module = _module()
    run_dir = tmp_path / "teacher-run"
    trace_path = run_dir / "natural_n1a" / "traces.jsonl"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text("", encoding="utf-8")
    versions_path = run_dir / "VERSIONS.txt"
    harness_lines = "".join(f"{digest}  /checkout/{path}\n" for path, digest in module.REQUIRED_HARNESS_SHA256.items())
    versions_path.write_text(
        f"model={module.TEACHER_MODEL}\nmodel_revision={module.TEACHER_REVISION}\n{harness_lines}",
        encoding="utf-8",
    )

    relative, values = module.source_versions(tmp_path, trace_path)
    assert relative == Path("teacher-run/VERSIONS.txt")
    assert values["model_revision"] == module.TEACHER_REVISION

    versions_path.write_text(
        f"model={module.TEACHER_MODEL}\nmodel_revision=mutable-main\n{harness_lines}",
        encoding="utf-8",
    )
    try:
        module.source_versions(tmp_path, trace_path)
    except ValueError as error:
        assert "does not pin teacher revision" in str(error)
    else:
        raise AssertionError("mutable teacher revision should be rejected")

    wrong_harness_lines = harness_lines.replace(next(iter(module.REQUIRED_HARNESS_SHA256.values())), "0" * 64)
    versions_path.write_text(
        f"model={module.TEACHER_MODEL}\nmodel_revision={module.TEACHER_REVISION}\n{wrong_harness_lines}",
        encoding="utf-8",
    )
    try:
        module.source_versions(tmp_path, trace_path)
    except ValueError as error:
        assert "non-frozen harness hashes" in str(error)
    else:
        raise AssertionError("non-frozen harness source should be rejected")


def test_runtime_provenance_distinguishes_legacy_recorded_and_partial_runs():
    module = _module()
    assert module.runtime_provenance({}) == {"status": "legacy_unrecorded"}

    values = {
        "vllm_version": "0.27.1+cu129",
        "vllm_distribution_url": "https://example.invalid/vllm.whl",
        "uv_lock_sha256": "a" * 64,
        "inference_config_sha256": "b" * 64,
    }
    assert module.runtime_provenance(values) == {"status": "recorded", **values}

    values.pop("inference_config_sha256")
    try:
        module.runtime_provenance(values)
    except ValueError as error:
        assert "partial runtime provenance" in str(error)
        assert "inference_config_sha256" in str(error)
    else:
        raise AssertionError("partial runtime provenance should be rejected")
