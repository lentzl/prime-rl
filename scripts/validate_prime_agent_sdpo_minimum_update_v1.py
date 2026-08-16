"""Validate the one-step Qwen3.5 27B mixed SDPO/GRPO minimum update."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

AUDIT_REPORT = Path(
    "/ephemeral/outputs/qwen35-27b-prime-agent-sdpo-v1/zero-lr-audit/AUDIT.json"
)
DEFAULT_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
EXPECTED_LR = 1e-7

_ZERO_PATH = Path(__file__).with_name("validate_prime_agent_sdpo_zero_lr_audit_v1.py")
_ZERO_SPEC = importlib.util.spec_from_file_location("_zero_lr_audit_validator", _ZERO_PATH)
assert _ZERO_SPEC is not None and _ZERO_SPEC.loader is not None
ZERO = importlib.util.module_from_spec(_ZERO_SPEC)
_ZERO_SPEC.loader.exec_module(ZERO)


class UpdateFailure(ValueError):
    """The completed run does not prove a valid minimum update."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise UpdateFailure(f"missing required file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise UpdateFailure(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise UpdateFailure(f"missing required file: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records or not all(isinstance(record, dict) for record in records):
        raise UpdateFailure(f"no valid records found in {path}")
    return records


def _metric(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if key in record]
    if not values or not all(math.isfinite(value) for value in values):
        raise UpdateFailure(f"missing finite metric: {key}")
    return values[-1]


def _validate_audit(report_path: Path, expected_revision: str) -> None:
    report = _read_json(report_path)
    if (
        report.get("verdict") != "pass"
        or report.get("mechanism") != "mixed-feedback-conditioned-sdpo-grpo-zero-lr"
        or report.get("expected_revision") != expected_revision
        or report.get("model_artifacts_written") is not False
    ):
        raise UpdateFailure("zero-LR prerequisite audit is not a matching passing report")


def _validate_configs(run_dir: Path, expected_revision: str) -> dict[str, str]:
    trainer = _read_json(run_dir / "configs" / "trainer.json")
    orchestrator = _read_json(run_dir / "configs" / "orchestrator.json")
    inference = _read_json(run_dir / "configs" / "inference.json")
    if trainer.get("max_steps") != 1 or orchestrator.get("max_steps") != 1:
        raise UpdateFailure("minimum update must run exactly one step")
    configured_lr = trainer.get("optim", {}).get("lr", -1)
    if not math.isclose(configured_lr, EXPECTED_LR, rel_tol=0, abs_tol=1e-15):
        raise UpdateFailure(f"minimum update learning rate must be {EXPECTED_LR:g}")
    if not trainer.get("enable_token_export"):
        raise UpdateFailure("minimum update must enable token export")
    trainer_ckpt = trainer.get("ckpt")
    orchestrator_ckpt = orchestrator.get("ckpt")
    if not isinstance(trainer_ckpt, dict) or trainer_ckpt.get("weights_only") is not True:
        raise UpdateFailure("minimum update must save a weights-only trainer checkpoint")
    if not isinstance(orchestrator_ckpt, dict):
        raise UpdateFailure("minimum update must save orchestrator progress")
    sources = orchestrator.get("train", {}).get("source", [])
    routing = {
        source.get("name"): (
            source.get("algo", {}).get("type"),
            source.get("group_size"),
            source.get("ratio"),
        )
        for source in sources
    }
    expected_routing = {
        ZERO.DIAGNOSTIC_ENV: (
            "sdpo",
            1,
            ZERO.EXPECTED_RATIOS[ZERO.DIAGNOSTIC_ENV],
        ),
        **{
            name: ("grpo", 2, ZERO.EXPECTED_RATIOS[name])
            for name in ZERO.RETENTION_ENVS
        },
    }
    if routing != expected_routing:
        raise UpdateFailure(f"minimum update source routing changed: {routing}")

    model_paths = {
        "trainer": trainer.get("model", {}).get("name"),
        "orchestrator": orchestrator.get("model", {}).get("name"),
        "inference": inference.get("vllm", {}).get("model"),
    }
    if not all(isinstance(path, str) for path in model_paths.values()):
        raise UpdateFailure(f"resolved model paths are incomplete: {model_paths}")
    revisions = {name: Path(path).name for name, path in model_paths.items()}
    if set(revisions.values()) != {expected_revision}:
        raise UpdateFailure(f"resolved model revisions do not match {expected_revision}: {revisions}")
    return revisions


def _validate_metrics(run_dir: Path) -> dict[str, float]:
    records = _read_jsonl(run_dir / "metrics.jsonl")
    if {record.get("step") for record in records if "step" in record} != {1}:
        raise UpdateFailure("minimum update metrics must contain exactly step 1")
    rl_tokens = _metric(records, "loss_tokens/rl")
    sdpo_tokens = _metric(records, "loss_tokens/sdpo")
    ce_tokens = _metric(records, "loss_tokens/ce")
    ref_kl_tokens = _metric(records, "loss_tokens/ref_kl")
    lr = _metric(records, "optim/lr")
    update_succeeded = _metric(records, "optim/update_succeeded")
    grad_norm = _metric(records, "optim/grad_norm")
    if rl_tokens <= 0 or sdpo_tokens <= 0 or ce_tokens != 0 or ref_kl_tokens != 0:
        raise UpdateFailure("minimum update has invalid component token mass")
    if not math.isclose(lr, EXPECTED_LR, rel_tol=0, abs_tol=1e-15):
        raise UpdateFailure(f"optimizer metric has the wrong learning rate: {lr:g}")
    if update_succeeded != 1 or grad_norm <= 0:
        raise UpdateFailure("minimum optimizer update did not succeed with a positive gradient")
    return {
        "rl_tokens": rl_tokens,
        "sdpo_tokens": sdpo_tokens,
        "ce_tokens": ce_tokens,
        "ref_kl_tokens": ref_kl_tokens,
        "lr": lr,
        "grad_norm": grad_norm,
    }


def _validate_weights(run_dir: Path) -> Path:
    weights = run_dir / "weights" / "step_1"
    if not (weights / "STABLE").is_file():
        raise UpdateFailure(f"minimum update has no stable weight snapshot: {weights}")
    safetensors = [path for path in weights.rglob("*.safetensors") if path.is_file()]
    if not safetensors:
        raise UpdateFailure(f"minimum update weight snapshot has no safetensors: {weights}")
    config = _read_json(weights / "config.json")
    if config.get("vision_config"):
        for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
            if not (weights / name).is_file():
                raise UpdateFailure(f"multimodal weight snapshot lacks {name}: {weights}")
    checkpoint_files = (
        [path for path in (run_dir / "checkpoints").rglob("*") if path.is_file()]
        if (run_dir / "checkpoints").exists()
        else []
    )
    trainer_state = [path for path in checkpoint_files if "trainer" in path.parts]
    if trainer_state:
        raise UpdateFailure("weights-only update unexpectedly wrote trainer optimizer state")
    return weights


def validate(
    run_dir: Path,
    expected_revision: str = DEFAULT_REVISION,
    audit_report: Path = AUDIT_REPORT,
) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise UpdateFailure(f"run directory does not exist: {run_dir}")
    _validate_audit(audit_report, expected_revision)
    revisions = _validate_configs(run_dir, expected_revision)
    metrics = _validate_metrics(run_dir)
    try:
        trace_report, traces = ZERO._validate_traces(run_dir)
        token_routing = ZERO._validate_token_routing(run_dir, traces)
    except ZERO.AuditFailure as error:
        raise UpdateFailure(f"minimum update failed branch-routing validation: {error}") from error
    weights = _validate_weights(run_dir)
    return {
        "verdict": "pass",
        "mechanism": "mixed-feedback-conditioned-sdpo-grpo-minimum-update",
        "expected_revision": expected_revision,
        "resolved_revisions": revisions,
        "metrics": metrics,
        "traces": trace_report,
        "token_routing": token_routing,
        "weights": str(weights),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-revision", default=DEFAULT_REVISION)
    parser.add_argument("--audit-report", type=Path, default=AUDIT_REPORT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate(args.run_dir, args.expected_revision, args.audit_report)
    except (UpdateFailure, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"minimum SDPO update validation failed: {error}") from error
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
