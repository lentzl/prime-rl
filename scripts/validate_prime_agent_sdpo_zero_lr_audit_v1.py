"""Validate the one-step Qwen3.5 27B Prime Agent SDPO mechanism audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

FEEDBACK_SCHEMA = "prime-agent/ownership-decision-feedback/v1"
DEFAULT_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
MIN_EFFECTIVE_TRACES = 6


class AuditFailure(ValueError):
    """The completed run does not prove the intended SDPO mechanism."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditFailure(f"missing required file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AuditFailure(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AuditFailure(f"missing required file: {path}")
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AuditFailure(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    if not records:
        raise AuditFailure(f"no records found in {path}")
    return records


def _metric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values = [record[key] for record in records if key in record]
    if not values or not all(isinstance(value, (int, float)) for value in values):
        raise AuditFailure(f"missing numeric metric: {key}")
    return [float(value) for value in values]


def _require_all(records: list[dict[str, Any]], key: str, expected: float) -> None:
    values = _metric_values(records, key)
    if not all(math.isclose(value, expected, abs_tol=1e-12) for value in values):
        raise AuditFailure(f"expected {key}={expected:g}, found {values}")


def _require_finite(records: list[dict[str, Any]], key: str) -> float:
    values = _metric_values(records, key)
    if not all(math.isfinite(value) for value in values):
        raise AuditFailure(f"non-finite {key}: {values}")
    return values[-1]


def _validate_configs(run_dir: Path, expected_revision: str) -> dict[str, str]:
    trainer = _read_json(run_dir / "configs" / "trainer.json")
    orchestrator = _read_json(run_dir / "configs" / "orchestrator.json")
    inference = _read_json(run_dir / "configs" / "inference.json")

    if trainer.get("max_steps") != 1 or orchestrator.get("max_steps") != 1:
        raise AuditFailure("resolved trainer and orchestrator must run exactly one step")
    if trainer.get("optim", {}).get("lr") != 0:
        raise AuditFailure("resolved trainer learning rate is not zero")
    if trainer.get("ckpt", {}).get("interval") is not None:
        raise AuditFailure("resolved trainer unexpectedly enables checkpointing")
    if orchestrator.get("ckpt", {}).get("interval") is not None:
        raise AuditFailure("resolved orchestrator unexpectedly enables checkpointing")
    if orchestrator.get("train", {}).get("sampling", {}).get("reasoning_effort") != "high":
        raise AuditFailure("resolved train sampling does not use high reasoning effort")

    model_paths = {
        "trainer": trainer.get("model", {}).get("name"),
        "orchestrator": orchestrator.get("model", {}).get("name"),
        "inference": inference.get("vllm", {}).get("model"),
    }
    if not all(isinstance(path, str) for path in model_paths.values()):
        raise AuditFailure(f"resolved model paths are incomplete: {model_paths}")
    revisions = {name: Path(path).name for name, path in model_paths.items()}
    if set(revisions.values()) != {expected_revision}:
        raise AuditFailure(f"resolved model revisions do not match {expected_revision}: {revisions}")
    return revisions


def _validate_metrics(run_dir: Path) -> dict[str, float]:
    records = _read_jsonl(run_dir / "metrics.jsonl")
    steps = {record.get("step") for record in records if "step" in record}
    if steps != {1}:
        raise AuditFailure(f"expected metrics for exactly step 1, found {sorted(steps)}")

    for key in ("loss_tokens/rl", "loss_tokens/ce", "loss_tokens/ref_kl"):
        _require_all(records, key, 0.0)
    sdpo_tokens = _require_finite(records, "loss_tokens/sdpo")
    if sdpo_tokens <= 0:
        raise AuditFailure(f"SDPO token mass must be positive, found {sdpo_tokens:g}")

    _require_all(records, "optim/lr", 0.0)
    _require_all(records, "optim/update_succeeded", 1.0)
    grad_norm = _require_finite(records, "optim/grad_norm")
    if grad_norm <= 0:
        raise AuditFailure(f"gradient norm must be positive, found {grad_norm:g}")
    loss = _require_finite(records, "loss/mean")
    sdpo_loss = _require_finite(records, "sdpo/mean")
    _require_all(records, "time/save_ckpt", 0.0)
    _require_all(records, "train/agg/effective/agent/is_trainable/mean", 1.0)
    _require_all(records, "train/agg/effective/agent/is_filtered/mean", 0.0)

    rollouts = _require_finite(records, "progress/rollouts")
    tasks = _require_finite(records, "progress/tasks")
    if rollouts < MIN_EFFECTIVE_TRACES or tasks < MIN_EFFECTIVE_TRACES:
        raise AuditFailure(f"expected at least {MIN_EFFECTIVE_TRACES} rollout/tasks, found {rollouts:g}/{tasks:g}")
    return {
        "sdpo_tokens": sdpo_tokens,
        "grad_norm": grad_norm,
        "loss": loss,
        "sdpo_loss": sdpo_loss,
        "rollouts": rollouts,
        "tasks": tasks,
    }


def _validate_traces(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "rollouts" / "step_1" / "train" / "effective" / "traces.jsonl"
    traces = _read_jsonl(path)
    if len(traces) < MIN_EFFECTIVE_TRACES:
        raise AuditFailure(f"expected at least {MIN_EFFECTIVE_TRACES} effective traces, found {len(traces)}")

    categories: dict[str, int] = {}
    codes: dict[str, int] = {}
    for index, trace in enumerate(traces):
        if trace.get("run", {}).get("type") != "train" or trace.get("run", {}).get("step") != 1:
            raise AuditFailure(f"effective trace {index} is not from train step 1")
        info = trace.get("info")
        if not isinstance(info, dict):
            raise AuditFailure(f"effective trace {index} has no info object")
        if info.get("env_name") != "ownership-child-natural-failures":
            raise AuditFailure(f"effective trace {index} came from an unexpected environment")
        feedback = info.get("feedback")
        contract = info.get("feedback_contract")
        if not isinstance(feedback, str) or not feedback.strip() or not isinstance(contract, dict):
            raise AuditFailure(f"effective trace {index} has no explicit typed feedback")
        required = {
            "schema_version": FEEDBACK_SCHEMA,
            "answer_free": True,
            "retryable": True,
            "turn_index": 0,
            "ownership": "child",
            "message": feedback,
        }
        if any(contract.get(key) != value for key, value in required.items()):
            raise AuditFailure(f"effective trace {index} has an invalid feedback contract")
        code = contract.get("code")
        category = contract.get("category")
        if not isinstance(code, str) or not code or not isinstance(category, str) or not category:
            raise AuditFailure(f"effective trace {index} lacks a stable feedback code/category")
        if trace.get("metrics", {}).get("strict_success") != 0:
            raise AuditFailure(f"effective trace {index} is not a diagnosed failure")
        codes[code] = codes.get(code, 0) + 1
        categories[category] = categories.get(category, 0) + 1
    return {"count": len(traces), "codes": codes, "categories": categories}


def _validate_no_model_artifacts(run_dir: Path) -> None:
    for directory in (run_dir / "checkpoints", run_dir / "weights"):
        files = [path for path in directory.rglob("*") if path.is_file()] if directory.exists() else []
        if files:
            raise AuditFailure(f"zero-LR audit wrote forbidden model artifacts under {directory}")


def validate(run_dir: Path, expected_revision: str = DEFAULT_REVISION) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise AuditFailure(f"run directory does not exist: {run_dir}")
    revisions = _validate_configs(run_dir, expected_revision)
    metrics = _validate_metrics(run_dir)
    traces = _validate_traces(run_dir)
    _validate_no_model_artifacts(run_dir)
    return {
        "verdict": "pass",
        "mechanism": "feedback-conditioned-sdpo-zero-lr",
        "expected_revision": expected_revision,
        "resolved_revisions": revisions,
        "metrics": metrics,
        "traces": traces,
        "model_artifacts_written": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate(args.run_dir, args.expected_revision)
    except (AuditFailure, json.JSONDecodeError) as error:
        raise SystemExit(f"zero-LR SDPO audit failed: {error}") from error
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
