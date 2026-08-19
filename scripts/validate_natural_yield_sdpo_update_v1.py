"""Validate one low-dose natural passive-yield SDPO update."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from procedural_harness_master_v1.taskset import (
    ProceduralHarnessMasterData,
    _natural_yield_feedback_diagnostic,
)
from transformers import AutoTokenizer

from prime_rl.trainer.ckpt import (
    CHAT_EOS_TOKEN,
    _validate_chat_eos_metadata,
)
from scripts import validate_natural_yield_sdpo_zero_lr_v1 as zero
from scripts.validate_prime_agent_sdpo_zero_lr_audit_v1 import (
    _read_json,
    _read_jsonl,
    _require_all,
    _require_finite,
)

EXPECTED_LR = 5e-8
EXPECTED_EOS_TOKEN_ID = 248046


class UpdateFailure(ValueError):
    """The run does not prove a correctly routed natural-yield update."""


def _validate_prerequisite(path: Path, expected_model_path: str) -> None:
    report = _read_json(path)
    if (
        report.get("verdict") != "pass"
        or report.get("mechanism")
        != "natural-yield-feedback-conditioned-sdpo-zero-lr"
        or report.get("expected_model_path") != expected_model_path
        or report.get("model_artifacts_written") is not False
    ):
        raise UpdateFailure("zero-LR prerequisite audit is absent or incompatible")


def _validate_configs(run_dir: Path, expected_model_path: str) -> None:
    trainer = _read_json(run_dir / "configs" / "trainer.json")
    orchestrator = _read_json(run_dir / "configs" / "orchestrator.json")
    inference = _read_json(run_dir / "configs" / "inference.json")
    if trainer.get("max_steps") != 1 or orchestrator.get("max_steps") != 1:
        raise UpdateFailure("natural-yield update must run exactly one step")
    lr = float(trainer.get("optim", {}).get("lr", -1))
    if not math.isclose(lr, EXPECTED_LR, rel_tol=0, abs_tol=1e-15):
        raise UpdateFailure(f"natural-yield update learning rate is {lr:g}")
    if not trainer.get("enable_token_export"):
        raise UpdateFailure("natural-yield update must export token routing")
    trainer_ckpt = trainer.get("ckpt")
    if not isinstance(trainer_ckpt, dict) or trainer_ckpt.get("weights_only") is not True:
        raise UpdateFailure("natural-yield update must save weights only")
    if not isinstance(orchestrator.get("ckpt"), dict):
        raise UpdateFailure("natural-yield update must coordinate a stable export")
    if trainer.get("model", {}).get("lora") is not None:
        raise UpdateFailure("natural-yield update must not use LoRA")
    if trainer.get("model", {}).get("optimization_dtype") != "bfloat16":
        raise UpdateFailure("natural-yield update must optimize full BF16 weights")

    sources = orchestrator.get("train", {}).get("source")
    if not isinstance(sources, list) or len(sources) != 1:
        raise UpdateFailure("natural-yield update must have exactly one source")
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
        raise UpdateFailure("natural-yield feedback routing configuration changed")
    if (
        taskset.get("curriculum_rung") != "natural_n1"
        or taskset.get("private_payload_mode") != "finding_card"
        or taskset.get("record_causal_feedback") is not True
    ):
        raise UpdateFailure("natural-yield training boundary changed")
    model_paths = {
        trainer.get("model", {}).get("name"),
        orchestrator.get("model", {}).get("name"),
        inference.get("vllm", {}).get("model"),
    }
    if model_paths != {expected_model_path}:
        raise UpdateFailure(f"services did not all start from R7: {model_paths!r}")


def _validate_metrics(run_dir: Path, expected_tokens: int) -> dict[str, float]:
    records = _read_jsonl(run_dir / "metrics.jsonl")
    if {row.get("step") for row in records if "step" in row} != {1}:
        raise UpdateFailure("metrics must contain exactly optimizer step 1")
    _require_all(records, "optim/lr", EXPECTED_LR)
    _require_all(records, "optim/update_succeeded", 1.0)
    values = {
        "loss": _require_finite(records, "loss/mean"),
        "sdpo_loss": _require_finite(records, "sdpo/mean"),
        "gradient_norm": _require_finite(records, "optim/grad_norm"),
        "sdpo_tokens": _require_finite(records, "loss_tokens/sdpo"),
        "rl_tokens": _require_finite(records, "loss_tokens/rl"),
        "ce_tokens": _require_finite(records, "loss_tokens/ce"),
        "reference_kl_tokens": _require_finite(records, "loss_tokens/ref_kl"),
    }
    if values["gradient_norm"] <= 0 or values["sdpo_tokens"] != expected_tokens:
        raise UpdateFailure("optimizer metrics disagree with routed SDPO signal")
    if any(values[name] != 0 for name in ("rl_tokens", "ce_tokens", "reference_kl_tokens")):
        raise UpdateFailure("a non-SDPO loss component received token mass")
    return values


def _validate_weights(run_dir: Path, step: int = 1) -> Path:
    weights = run_dir / "weights" / f"step_{step}"
    if not (weights / "STABLE").is_file():
        raise UpdateFailure(f"weight export is not stable: {weights}")
    if not list(weights.glob("*.safetensors")):
        raise UpdateFailure("stable export contains no safetensors")
    tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
    if tokenizer.encode(CHAT_EOS_TOKEN, add_special_tokens=False) != [
        EXPECTED_EOS_TOKEN_ID
    ]:
        raise UpdateFailure("export tokenizer has the wrong ChatML end token")
    _validate_chat_eos_metadata(weights, tokenizer, EXPECTED_EOS_TOKEN_ID)
    checkpoint_files = (
        list((run_dir / "checkpoints").rglob("*"))
        if (run_dir / "checkpoints").exists()
        else []
    )
    if any(path.is_file() and "trainer" in path.parts for path in checkpoint_files):
        raise UpdateFailure("weights-only update wrote trainer optimizer state")
    return weights


def _validate_pristine_prefixes(
    run_dir: Path, traces: list[Any], step: int = 1
) -> None:
    records = _read_jsonl(
        run_dir
        / "rollouts"
        / f"step_{step}"
        / "train"
        / "effective"
        / "traces.jsonl"
    )
    if len(records) != len(traces):
        raise UpdateFailure("pristine-prefix audit trace count changed")
    for index, (record, trace) in enumerate(zip(records, traces, strict=True)):
        task_data = ProceduralHarnessMasterData.model_validate(
            record.get("task", {}).get("data", {})
        )
        diagnostic = _natural_yield_feedback_diagnostic(trace, task_data)
        contract = record.get("info", {}).get("feedback_contract", {})
        if (
            diagnostic is None
            or diagnostic.target_node_index != contract.get("target_node_index")
            or diagnostic.spawn_node_index
            != contract.get("evidence", {}).get("spawn_node_index")
        ):
            raise UpdateFailure(
                f"effective trace {index} does not have a pristine delegation prefix"
            )


def validate(
    run_dir: Path,
    prerequisite: Path,
    expected_model_path: str,
) -> dict[str, Any]:
    _validate_prerequisite(prerequisite, expected_model_path)
    _validate_configs(run_dir, expected_model_path)
    traces = zero._validate_traces(run_dir)
    _validate_pristine_prefixes(run_dir, traces)
    routing = zero._validate_token_routing(run_dir, traces)
    expected_tokens = sum(
        sum(
            value is not None and value != 0
            for value in record.get("sdpo_weights", [])
        )
        for path in sorted((run_dir / "token_exports" / "step_1").glob("rank_*.jsonl"))
        for record in _read_jsonl(path)
    )
    metrics = _validate_metrics(run_dir, expected_tokens)
    weights = _validate_weights(run_dir)
    return {
        "verdict": "pass",
        "mechanism": "natural-yield-feedback-conditioned-sdpo-low-dose-update",
        "source_model": expected_model_path,
        "learning_rate": EXPECTED_LR,
        "effective_traces": len(traces),
        "metrics": metrics,
        "token_routing": routing,
        "weights": str(weights),
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
    except (UpdateFailure, zero.AuditFailure, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"natural-yield SDPO update validation failed: {error}") from error
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
