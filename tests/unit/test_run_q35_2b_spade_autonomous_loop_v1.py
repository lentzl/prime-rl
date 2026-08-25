import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _module():
    scripts = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "run_q35_2b_spade_autonomous_loop_v1.py"
        spec = importlib.util.spec_from_file_location("run_q35_2b_spade_autonomous_loop_v1", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _config(module, tmp_path, **overrides):
    values = {
        "repo_root": tmp_path,
        "events": tmp_path / "events.jsonl",
        "base_model": tmp_path / "base",
        "initial_adapter_path": tmp_path / "adapter",
        "artifacts_root": tmp_path / "artifacts",
        "results_root": tmp_path / "results",
        "output_root": tmp_path / "outputs",
        "experiment_dir": tmp_path / "experiments",
        "journal": tmp_path / "journal.jsonl",
        "stop_file": tmp_path / "STOP",
        "uv_bin": "uv",
        "max_evaluations": 1,
        "max_updates": 0,
        "max_actions": 1,
        "dry_run": False,
    }
    values.update(overrides)
    return module.RunnerConfig(**values)


def _collecting_status():
    return {
        "event_head_sha256": "a" * 64,
        "candidate": {"label": "R6Y5"},
        "next_action": {
            "kind": "collect",
            "arms": [
                {
                    "track": "child",
                    "phase": "e0c25_inline_evidence",
                    "start_index": 4_009_100,
                    "tasks": 6,
                }
            ],
        },
    }


def test_runner_executes_planned_collection_then_stops_at_budget(tmp_path) -> None:
    module = _module()

    class FakeRunner(module.AutonomousRunner):
        def __init__(self, config):
            super().__init__(config)
            self.collected = []

        def status(self):
            return _collecting_status()

        def collect(self, status, arm):
            self.collected.append((status["event_head_sha256"], arm["start_index"]))

    runner = FakeRunner(_config(module, tmp_path))
    report = runner.run()

    assert runner.collected == [("a" * 64, 4_009_100)]
    assert report["actions_completed"] == 1
    assert report["evaluations_completed"] == 1
    assert report["updates_completed"] == 0
    assert report["stop_reason"] == "action_budget_exhausted"


def test_training_refuses_without_exact_controller_authorization(tmp_path) -> None:
    module = _module()
    runner = module.AutonomousRunner(_config(module, tmp_path))

    with pytest.raises(ValueError, match="without one-step authorization"):
        runner.train(_collecting_status())


def test_restart_skips_partial_training_attempt_and_reuses_complete_one(tmp_path) -> None:
    module = _module()
    config = _config(module, tmp_path)
    runner = module.AutonomousRunner(config)
    partial = config.output_root / "cycle"
    partial.mkdir(parents=True)
    complete = config.output_root / "cycle-attempt2"
    adapter = complete / "weights/step_1/lora_adapters"
    adapter.mkdir(parents=True)
    (adapter / module.ADAPTER_WEIGHT_NAME).write_bytes(b"adapter")
    (complete / "metrics.jsonl").write_text("{}\n")

    name, output = runner._training_attempt("cycle")

    assert name == "cycle-attempt2"
    assert output == complete


def test_candidate_adapter_hash_is_fail_closed(tmp_path) -> None:
    module = _module()
    config = _config(module, tmp_path)
    config.initial_adapter_path.mkdir(parents=True)
    weight = config.initial_adapter_path / module.ADAPTER_WEIGHT_NAME
    weight.write_bytes(b"adapter")
    runner = module.AutonomousRunner(config)

    with pytest.raises(ValueError, match="does not match the event log"):
        runner._adapter_path({"adapter_sha256": "0" * 64})

    assert runner._adapter_path(
        {"adapter_sha256": hashlib.sha256(b"adapter").hexdigest()}
    ) == config.initial_adapter_path


def test_generated_training_config_preserves_bounded_bf16_update(tmp_path) -> None:
    module = _module()
    runner = module.AutonomousRunner(_config(module, tmp_path))

    rendered = runner._training_config(
        run_name="r6y6",
        adapter_path=tmp_path / "adapter",
        adapter_sha256="a" * 64,
        corpus=tmp_path / "corpus",
    )

    assert "max_steps = 1" in rendered
    assert 'optimization_dtype = "bfloat16"' in rendered
    assert 'reduce_dtype = "bfloat16"' in rendered
    assert 'initial_adapter_sha256 = "' + "a" * 64 + '"' in rendered
    assert "batch_size = 8" in rendered


def test_existing_bootstrap_must_match_planned_bank(tmp_path) -> None:
    module = _module()
    config = _config(module, tmp_path)
    runner = module.AutonomousRunner(config)
    candidate = {"label": "R6Y5"}
    arm = _collecting_status()["next_action"]["arms"][0]
    path = runner._bootstrap_path(candidate, arm)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "axes": [{"name": "natural_n1a", "start_index": 1}],
                "tasks_per_axis": 6,
                "gradient_updates": 0,
                "records": [{"final_answer_in_context": False}] * 6,
            }
        )
    )

    with pytest.raises(ValueError, match="does not match planned arm"):
        runner._ensure_bootstrap(candidate, arm)
