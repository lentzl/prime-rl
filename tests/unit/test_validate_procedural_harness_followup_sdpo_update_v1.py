import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "validate_procedural_harness_followup_sdpo_update_v1.py"
SPEC = importlib.util.spec_from_file_location("validate_procedural_harness_followup_sdpo_update_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _make_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "followup-update"
    _write_json(
        run_dir / "configs" / "trainer.json",
        {
            "max_steps": 1,
            "model": {"lora": None, "optimization_dtype": "bfloat16"},
            "optim": {"lr": 1.25e-7},
            "enable_token_export": True,
        },
    )
    _write_json(
        run_dir / "configs" / "orchestrator.json",
        {
            "max_steps": 1,
            "batch_size": 1,
            "train": {
                "source": [
                    {
                        "name": MODULE.ENV_NAME,
                        "algo": {
                            "type": "sdpo",
                            "required_feedback_contract_schema": MODULE.FEEDBACK_SCHEMA_VERSION,
                            "filter": {
                                "import_path": (
                                    "procedural_harness_master_v1.taskset."
                                    "keep_followup_feedback_response"
                                )
                            },
                            "multi_turn_replay": True,
                            "environment_feedback_only_without_solution": True,
                        },
                    }
                ]
            },
        },
    )
    metrics = {
        "loss_tokens/rl": 0,
        "loss_tokens/ce": 0,
        "loss_tokens/ref_kl": 0,
        "loss_tokens/sdpo": 3,
        "optim/grad_norm": 0.25,
        "optim/update_succeeded": 1,
    }
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics) + "\n")

    export_dir = run_dir / "token_exports" / "step_1"
    export_dir.mkdir(parents=True)
    (export_dir / "STABLE").touch()
    export = {
        "env_name": MODULE.ENV_NAME,
        "token_ids": [1, 2, 3, 4],
        "loss_mask": [True, True, True, False],
        "rl_weights": [0.0] * 4,
        "ce_weights": [0.0] * 4,
        "ref_kl_weights": [0.0] * 4,
        "sdpo_weights": [1.0, 1.0, 1.0, 0.0],
    }
    (export_dir / "rank_0.jsonl").write_text(json.dumps(export) + "\n")

    weights = run_dir / "weights" / "step_1"
    weights.mkdir(parents=True)
    (weights / "STABLE").touch()
    (weights / "model.safetensors").write_bytes(b"weights")
    return run_dir


def test_validator_accepts_exact_failure_local_component_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _make_run(tmp_path)
    monkeypatch.setattr(MODULE, "_validate_traces", lambda _run_dir, _count: {"count": 1, "selected_tokens": 3})

    report = MODULE.validate(run_dir)

    assert report["verdict"] == "pass"
    assert report["token_routing"]["sdpo_tokens"] == 3
    assert report["metrics"]["grad_norm"] == 0.25


def test_validator_rejects_trainer_trace_token_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _make_run(tmp_path)
    monkeypatch.setattr(MODULE, "_validate_traces", lambda _run_dir, _count: {"count": 1, "selected_tokens": 4})

    with pytest.raises(MODULE.UpdateValidationFailure, match="trainer received 3 SDPO tokens"):
        MODULE.validate(run_dir)


def test_validator_rejects_cross_component_contamination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _make_run(tmp_path)
    export_path = run_dir / "token_exports" / "step_1" / "rank_0.jsonl"
    export = json.loads(export_path.read_text())
    export["rl_weights"][0] = 1.0
    export_path.write_text(json.dumps(export) + "\n")
    monkeypatch.setattr(MODULE, "_validate_traces", lambda _run_dir, _count: {"count": 1, "selected_tokens": 3})

    with pytest.raises(MODULE.UpdateValidationFailure, match="non-SDPO component"):
        MODULE.validate(run_dir)


def test_followup_launcher_finalizes_and_validates_the_update() -> None:
    launcher = (
        ROOT / "scripts" / "run_qwen35_27b_procedural_harness_followup_sdpo_v1.sh"
    ).read_text()

    assert "finalize_hf_processor_metadata.py" in launcher
    assert "validate_procedural_harness_followup_sdpo_update_v1.py" in launcher
    assert '--output "$run_dir/UPDATE.json"' in launcher
