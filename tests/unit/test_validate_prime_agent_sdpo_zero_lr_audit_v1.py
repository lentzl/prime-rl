import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "validate_prime_agent_sdpo_zero_lr_audit_v1.py"
SPEC = importlib.util.spec_from_file_location("validate_prime_agent_sdpo_zero_lr_audit_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _make_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "zero-lr-audit"
    revision = MODULE.DEFAULT_REVISION
    snapshot = f"/cache/Qwen3.5-27B/snapshots/{revision}"
    _write_json(
        run_dir / "configs" / "trainer.json",
        {
            "max_steps": 1,
            "model": {"name": snapshot},
            "optim": {"lr": 0},
            "ckpt": {"interval": None},
        },
    )
    _write_json(
        run_dir / "configs" / "orchestrator.json",
        {
            "max_steps": 1,
            "model": {"name": snapshot},
            "ckpt": {"interval": None},
            "train": {"sampling": {"reasoning_effort": "high"}},
        },
    )
    _write_json(
        run_dir / "configs" / "inference.json",
        {"vllm": {"model": snapshot}},
    )
    metrics = [
        {
            "step": 1,
            "progress/rollouts": 6,
            "progress/tasks": 6,
            "time/save_ckpt": 0,
            "train/agg/effective/agent/is_trainable/mean": 1,
            "train/agg/effective/agent/is_filtered/mean": 0,
        },
        {
            "step": 1,
            "loss_tokens/rl": 0,
            "loss_tokens/ce": 0,
            "loss_tokens/ref_kl": 0,
            "loss_tokens/sdpo": 128,
        },
        {
            "step": 1,
            "optim/lr": 0,
            "optim/update_succeeded": 1,
            "optim/grad_norm": 0.25,
        },
        {"step": 1, "loss/mean": 0.1, "sdpo/mean": 0.2},
        {"step": 1, "time/save_ckpt": 0},
    ]
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("".join(json.dumps(record) + "\n" for record in metrics))

    trace_path = run_dir / "rollouts" / "step_1" / "train" / "effective" / "traces.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    traces = []
    for index in range(6):
        feedback = "Delegate the child-owned resource and retain the returned handle."
        traces.append(
            {
                "run": {"type": "train", "step": 1},
                "info": {
                    "env_name": "ownership-child-natural-failures",
                    "feedback": feedback,
                    "feedback_contract": {
                        "schema_version": MODULE.FEEDBACK_SCHEMA,
                        "code": "required_delegation_missing",
                        "category": "routing",
                        "ownership": "child",
                        "turn_index": 0,
                        "answer_free": True,
                        "retryable": True,
                        "message": feedback,
                    },
                },
                "metrics": {"strict_success": 0},
                "task": {"data": {"idx": index}},
            }
        )
    trace_path.write_text("".join(json.dumps(trace) + "\n" for trace in traces))
    return run_dir


def test_validator_accepts_complete_zero_lr_mechanism_audit(tmp_path: Path) -> None:
    report = MODULE.validate(_make_run(tmp_path))

    assert report["verdict"] == "pass"
    assert report["metrics"]["sdpo_tokens"] == 128
    assert report["metrics"]["grad_norm"] == 0.25
    assert report["traces"]["count"] == 6
    assert report["model_artifacts_written"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nonzero_lr", "learning rate is not zero"),
        ("no_sdpo_tokens", "SDPO token mass must be positive"),
        ("competing_loss", "expected loss_tokens/rl=0"),
        ("zero_gradient", "gradient norm must be positive"),
        ("bad_contract", "invalid feedback contract"),
        ("checkpoint", "forbidden model artifacts"),
    ],
)
def test_validator_fails_closed(tmp_path: Path, mutation: str, message: str) -> None:
    run_dir = _make_run(tmp_path)
    if mutation == "nonzero_lr":
        path = run_dir / "configs" / "trainer.json"
        config = json.loads(path.read_text())
        config["optim"]["lr"] = 1e-7
        _write_json(path, config)
    elif mutation in {"no_sdpo_tokens", "competing_loss", "zero_gradient"}:
        path = run_dir / "metrics.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        key, value = {
            "no_sdpo_tokens": ("loss_tokens/sdpo", 0),
            "competing_loss": ("loss_tokens/rl", 1),
            "zero_gradient": ("optim/grad_norm", 0),
        }[mutation]
        next(record for record in records if key in record)[key] = value
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
    elif mutation == "bad_contract":
        path = run_dir / "rollouts" / "step_1" / "train" / "effective" / "traces.jsonl"
        traces = [json.loads(line) for line in path.read_text().splitlines()]
        traces[0]["info"]["feedback_contract"]["answer_free"] = False
        path.write_text("".join(json.dumps(trace) + "\n" for trace in traces))
    else:
        artifact = run_dir / "weights" / "step_1" / "model.safetensors"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"weights")

    with pytest.raises(MODULE.AuditFailure, match=message):
        MODULE.validate(run_dir)
