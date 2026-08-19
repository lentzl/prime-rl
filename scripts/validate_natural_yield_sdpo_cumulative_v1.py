"""Validate four cumulative natural passive-yield SDPO updates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts import validate_natural_yield_sdpo_update_v1 as update
from scripts import validate_natural_yield_sdpo_zero_lr_v1 as zero
from scripts.validate_prime_agent_sdpo_zero_lr_audit_v1 import (
    _exported_component_token_counts,
    _read_json,
    _read_jsonl,
    _require_all,
    _require_finite,
)

EXPECTED_STEPS = (1, 2, 3, 4)
EXPECTED_START_INDEX = 3_600_000


class CumulativeUpdateFailure(ValueError):
    """The run does not prove four correctly routed cumulative updates."""


def _validate_configs(run_dir: Path, expected_model_path: str) -> None:
    trainer = _read_json(run_dir / "configs" / "trainer.json")
    orchestrator = _read_json(run_dir / "configs" / "orchestrator.json")
    inference = _read_json(run_dir / "configs" / "inference.json")
    expected_steps = len(EXPECTED_STEPS)
    if (
        trainer.get("max_steps") != expected_steps
        or orchestrator.get("max_steps") != expected_steps
    ):
        raise CumulativeUpdateFailure("resolved services must run exactly four steps")
    if not math.isclose(
        float(trainer.get("optim", {}).get("lr", -1)),
        update.EXPECTED_LR,
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise CumulativeUpdateFailure("cumulative learning rate changed")
    trainer_ckpt = trainer.get("ckpt", {})
    orchestrator_ckpt = orchestrator.get("ckpt", {})
    if (
        trainer_ckpt.get("interval") != 1
        or trainer_ckpt.get("keep_last") != expected_steps
        or trainer_ckpt.get("weights_only") is not True
        or orchestrator_ckpt.get("interval") != 1
        or orchestrator_ckpt.get("keep_last") != expected_steps
    ):
        raise CumulativeUpdateFailure("every cumulative checkpoint must be retained")
    if trainer.get("model", {}).get("lora") is not None:
        raise CumulativeUpdateFailure("cumulative update must not use LoRA")
    if trainer.get("model", {}).get("optimization_dtype") != "bfloat16":
        raise CumulativeUpdateFailure("cumulative update must optimize full BF16 weights")
    if trainer.get("sdpo_loss", {}).get("teacher_regularization") != "ema":
        raise CumulativeUpdateFailure("cumulative update must preserve the EMA teacher")

    sources = orchestrator.get("train", {}).get("source")
    if not isinstance(sources, list) or len(sources) != 1:
        raise CumulativeUpdateFailure("cumulative update must have one source")
    source = sources[0]
    algo = source.get("algo", {})
    taskset = source.get("env", {}).get("taskset", {})
    if (
        source.get("name") != zero.ENV_NAME
        or source.get("group_size") != 1
        or algo.get("type") != "sdpo"
        or algo.get("required_feedback_contract_schema") != zero.FEEDBACK_SCHEMA
        or algo.get("environment_feedback_only_without_solution") is not True
        or algo.get("multi_turn_replay") is not False
        or algo.get("filter", {}).get("import_path")
        != "procedural_harness_master_v1.taskset.keep_natural_yield_feedback_response"
    ):
        raise CumulativeUpdateFailure("failure-local SDPO routing changed")
    if (
        taskset.get("curriculum_rung") != "natural_n1"
        or taskset.get("private_payload_mode") != "finding_card"
        or taskset.get("record_causal_feedback") is not True
        or taskset.get("start_index") != EXPECTED_START_INDEX
    ):
        raise CumulativeUpdateFailure("fresh cumulative task boundary changed")
    model_paths = {
        trainer.get("model", {}).get("name"),
        orchestrator.get("model", {}).get("name"),
        inference.get("vllm", {}).get("model"),
    }
    if model_paths != {expected_model_path}:
        raise CumulativeUpdateFailure(
            f"services did not all start from canonical R7: {model_paths!r}"
        )


def _step_metrics(
    run_dir: Path, records: list[dict[str, Any]], step: int
) -> dict[str, float]:
    step_records = [record for record in records if record.get("step") == step]
    if not step_records:
        raise CumulativeUpdateFailure(f"step {step} has no optimizer metrics")
    _require_all(step_records, "optim/lr", update.EXPECTED_LR)
    _require_all(step_records, "optim/update_succeeded", 1.0)
    counts = _exported_component_token_counts(run_dir, step)
    values = {
        "loss": _require_finite(step_records, "loss/mean"),
        "sdpo_loss": _require_finite(step_records, "sdpo/mean"),
        "gradient_norm": _require_finite(step_records, "optim/grad_norm"),
        "sdpo_tokens": _require_finite(step_records, "loss_tokens/sdpo"),
        "rl_tokens": _require_finite(step_records, "loss_tokens/rl"),
        "ce_tokens": _require_finite(step_records, "loss_tokens/ce"),
        "reference_kl_tokens": _require_finite(
            step_records, "loss_tokens/ref_kl"
        ),
    }
    if values["gradient_norm"] <= 0 or values["sdpo_tokens"] != counts["sdpo"]:
        raise CumulativeUpdateFailure(
            f"step {step} optimizer metrics disagree with exported SDPO signal"
        )
    for name in ("rl", "ce", "ref_kl"):
        if counts[name] != 0:
            raise CumulativeUpdateFailure(
                f"step {step} exported unexpected {name} token mass"
            )
    if any(
        values[name] != 0
        for name in ("rl_tokens", "ce_tokens", "reference_kl_tokens")
    ):
        raise CumulativeUpdateFailure(
            f"step {step} trained a non-SDPO loss component"
        )
    return values


def validate(
    run_dir: Path,
    prerequisite: Path,
    expected_model_path: str,
) -> dict[str, Any]:
    update._validate_prerequisite(prerequisite, expected_model_path)
    _validate_configs(run_dir, expected_model_path)
    metric_records = _read_jsonl(run_dir / "metrics.jsonl")
    metric_steps = {
        int(record["step"]) for record in metric_records if "step" in record
    }
    if metric_steps != set(EXPECTED_STEPS):
        raise CumulativeUpdateFailure(
            f"metrics cover steps {sorted(metric_steps)}, expected {EXPECTED_STEPS}"
        )

    steps = {}
    for step in EXPECTED_STEPS:
        traces = zero._validate_traces(run_dir, step)
        update._validate_pristine_prefixes(run_dir, traces, step)
        routing = zero._validate_token_routing(run_dir, traces, step)
        metrics = _step_metrics(run_dir, metric_records, step)
        weights = update._validate_weights(run_dir, step)
        steps[str(step)] = {
            "effective_traces": len(traces),
            "metrics": metrics,
            "token_routing": routing,
            "weights": str(weights),
        }
    return {
        "verdict": "pass",
        "mechanism": "natural-yield-feedback-conditioned-sdpo-cumulative-update",
        "source_model": expected_model_path,
        "learning_rate_per_step": update.EXPECTED_LR,
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--prerequisite", type=Path, required=True)
    parser.add_argument("--expected-model-path", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate(args.run_dir, args.prerequisite, args.expected_model_path)
    except (
        CumulativeUpdateFailure,
        update.UpdateFailure,
        zero.AuditFailure,
        json.JSONDecodeError,
        ValueError,
    ) as error:
        raise SystemExit(f"cumulative natural-yield SDPO validation failed: {error}") from error
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
