import importlib.util
import json
from pathlib import Path

import pytest

from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import cli

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "experiments" / "qwen35-27b-prime-agent-sdpo-v1" / "zero-lr-audit.toml"
SCRIPT = ROOT / "scripts" / "validate_prime_agent_sdpo_minimum_update_v1.py"
SPEC = importlib.util.spec_from_file_location("validate_prime_agent_sdpo_minimum_update_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _make_update_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "minimum-update"
    revision = MODULE.DEFAULT_REVISION
    snapshot = f"/cache/Qwen3.5-27B/snapshots/{revision}"
    sources = [
        {
            "name": MODULE.ZERO.DIAGNOSTIC_ENV,
            "group_size": 1,
            "ratio": MODULE.ZERO.EXPECTED_RATIOS[MODULE.ZERO.DIAGNOSTIC_ENV],
            "algo": {"type": "sdpo"},
        },
        *[
            {
                "name": name,
                "group_size": 2,
                "ratio": MODULE.ZERO.EXPECTED_RATIOS[name],
                "algo": {"type": "grpo"},
            }
            for name in sorted(MODULE.ZERO.RETENTION_ENVS)
        ],
    ]
    _write_json(
        run_dir / "configs" / "trainer.json",
        {
            "max_steps": 1,
            "model": {"name": snapshot},
            "optim": {"lr": MODULE.EXPECTED_LR},
            "ckpt": {"interval": 1, "keep_last": 1, "weights_only": True},
            "enable_token_export": True,
        },
    )
    _write_json(
        run_dir / "configs" / "orchestrator.json",
        {
            "max_steps": 1,
            "model": {"name": snapshot},
            "ckpt": {"interval": 1, "keep_last": 1},
            "train": {"source": sources},
        },
    )
    _write_json(run_dir / "configs" / "inference.json", {"vllm": {"model": snapshot}})
    metrics = [
        {
            "step": 1,
            "loss_tokens/rl": 256,
            "loss_tokens/sdpo": 128,
            "loss_tokens/ce": 0,
            "loss_tokens/ref_kl": 0,
        },
        {
            "step": 1,
            "optim/lr": MODULE.EXPECTED_LR,
            "optim/update_succeeded": 1,
            "optim/grad_norm": 0.25,
        },
    ]
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("".join(json.dumps(record) + "\n" for record in metrics))
    weights = run_dir / "weights" / "step_1"
    weights.mkdir(parents=True)
    (weights / "STABLE").touch()
    (weights / "model.safetensors").write_bytes(b"weights")
    audit_report = tmp_path / "AUDIT.json"
    _write_json(
        audit_report,
        {
            "verdict": "pass",
            "mechanism": "mixed-feedback-conditioned-sdpo-grpo-zero-lr",
            "expected_revision": revision,
            "model_artifacts_written": False,
        },
    )
    return run_dir, audit_report


def test_minimum_update_overrides_only_update_and_checkpoint_policy() -> None:
    config = cli(
        RLConfig,
        args=[
            "@",
            str(CONFIG),
            "--trainer.optim.lr",
            "1e-7",
            "--trainer.ckpt.weights-only",
            "true",
            "--trainer.ckpt.interval",
            "1",
            "--trainer.ckpt.keep-last",
            "1",
            "--orchestrator.ckpt.interval",
            "1",
            "--orchestrator.ckpt.keep-last",
            "1",
            "--dry-run",
        ],
    )

    assert config.max_steps == 1
    assert config.trainer.optim.lr == MODULE.EXPECTED_LR
    assert config.trainer.ckpt is not None
    assert config.trainer.ckpt.weights_only is True
    assert config.trainer.ckpt.interval == 1
    assert config.orchestrator.ckpt is not None
    assert config.orchestrator.ckpt.interval == 1
    assert config.trainer.enable_token_export is True
    assert config.orchestrator.batch_size == MODULE.ZERO.EXPECTED_BATCH_SIZE


def test_minimum_update_launcher_fails_closed() -> None:
    launcher = (
        ROOT / "scripts" / "run_qwen35_27b_prime_agent_sdpo_minimum_update_v1.sh"
    ).read_text()

    assert "refusing to launch while another GPU process is active" in launcher
    assert "matching passing zero-LR audit" in launcher
    assert "--trainer.optim.lr 1e-7" in launcher
    assert "--trainer.ckpt.weights-only true" in launcher
    assert "SDPO_MINIMUM_UPDATE_DRY_RUN" in launcher
    assert "validate_prime_agent_sdpo_minimum_update_v1.py" in launcher
    assert '--output "$run_dir/UPDATE.json"' in launcher


def test_validator_accepts_minimum_update_artifacts(tmp_path: Path, monkeypatch) -> None:
    run_dir, audit_report = _make_update_run(tmp_path)
    monkeypatch.setattr(MODULE.ZERO, "_validate_traces", lambda _: ({"count": 32}, []))
    monkeypatch.setattr(
        MODULE.ZERO,
        "_validate_token_routing",
        lambda _run_dir, _traces: {"matched_exports": 32},
    )

    report = MODULE.validate(run_dir, audit_report=audit_report)

    assert report["verdict"] == "pass"
    assert report["metrics"]["lr"] == MODULE.EXPECTED_LR
    assert report["weights"].endswith("weights/step_1")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("optimizer_state", "trainer optimizer state"),
        ("missing_stable", "no stable weight snapshot"),
        ("wrong_lr", "learning rate must be"),
    ],
)
def test_validator_rejects_invalid_update_artifacts(
    tmp_path: Path, mutation: str, message: str
) -> None:
    run_dir, _ = _make_update_run(tmp_path)
    if mutation == "optimizer_state":
        state = run_dir / "checkpoints" / "step_1" / "trainer" / "optimizer.pt"
        state.parent.mkdir(parents=True)
        state.write_bytes(b"state")
    elif mutation == "missing_stable":
        (run_dir / "weights" / "step_1" / "STABLE").unlink()
    else:
        trainer_path = run_dir / "configs" / "trainer.json"
        trainer = json.loads(trainer_path.read_text())
        trainer["optim"]["lr"] = 0
        _write_json(trainer_path, trainer)

    with pytest.raises(MODULE.UpdateFailure, match=message):
        if mutation == "wrong_lr":
            MODULE._validate_configs(run_dir, MODULE.DEFAULT_REVISION)
        else:
            MODULE._validate_weights(run_dir)
