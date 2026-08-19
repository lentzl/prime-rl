"""Validate the natural N1 passive-yield SDPO zero-LR mechanism audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from procedural_harness_master_v1.taskset import (
    keep_natural_yield_feedback_response,
)
from verifiers.v1.types import UserMessage, content_text

from prime_rl.orchestrator.trajectories import iter_trainable_branches
from scripts.validate_prime_agent_sdpo_zero_lr_audit_v1 import (
    AuditFailure,
    _active_component,
    _exported_component_token_counts,
    _read_json,
    _read_jsonl,
    _require_all,
    _require_finite,
    _validate_no_model_artifacts,
)

FEEDBACK_SCHEMA = "prime-agent/natural-yield-feedback/v1"
FEEDBACK_CODE = "tool_call_after_delegation"
ENV_NAME = "natural-yield-feedback-sdpo"
EXPECTED_BATCH_SIZE = 4
TRAINING_SEQ_LEN = 8192
MAX_COMPLETION_TOKENS = 1536


def _validate_configs(run_dir: Path, expected_model_path: str) -> None:
    trainer = _read_json(run_dir / "configs" / "trainer.json")
    orchestrator = _read_json(run_dir / "configs" / "orchestrator.json")
    inference = _read_json(run_dir / "configs" / "inference.json")

    if trainer.get("max_steps") != 1 or orchestrator.get("max_steps") != 1:
        raise AuditFailure("resolved services must run exactly one step")
    if trainer.get("optim", {}).get("lr") != 0:
        raise AuditFailure("resolved trainer learning rate is not zero")
    if not trainer.get("enable_token_export"):
        raise AuditFailure("resolved trainer must enable token export")
    if trainer.get("ckpt") is not None or orchestrator.get("ckpt") is not None:
        raise AuditFailure("zero-LR audit must not enable checkpointing")
    if trainer.get("model", {}).get("seq_len") != TRAINING_SEQ_LEN:
        raise AuditFailure("resolved trainer sequence length is not 8192")
    if orchestrator.get("seq_len") != TRAINING_SEQ_LEN:
        raise AuditFailure("resolved orchestrator sequence length is not 8192")
    if orchestrator.get("batch_size") != EXPECTED_BATCH_SIZE:
        raise AuditFailure("resolved audit batch size is not four")

    train = orchestrator.get("train", {})
    sampling = train.get("sampling", {})
    if sampling.get("reasoning_effort") != "high":
        raise AuditFailure("resolved audit does not use high reasoning effort")
    if sampling.get("max_completion_tokens") != MAX_COMPLETION_TOKENS:
        raise AuditFailure("resolved completion-token cap is not 1536")
    sources = train.get("source")
    if not isinstance(sources, list) or len(sources) != 1:
        raise AuditFailure("resolved audit must contain one training source")
    source = sources[0]
    algo = source.get("algo", {})
    taskset = source.get("env", {}).get("taskset", {})
    if source.get("name") != ENV_NAME or source.get("group_size") != 1:
        raise AuditFailure("resolved natural-yield source identity is invalid")
    if (
        algo.get("type") != "sdpo"
        or algo.get("require_explicit_feedback") is not True
        or algo.get("required_feedback_contract_schema") != FEEDBACK_SCHEMA
        or algo.get("multi_turn_replay") is not False
        or algo.get("filter", {}).get("import_path")
        != "procedural_harness_master_v1.taskset.keep_natural_yield_feedback_response"
    ):
        raise AuditFailure("resolved source does not pin failure-local yield SDPO")
    if (
        taskset.get("curriculum_rung") != "natural_n1"
        or taskset.get("private_payload_mode") != "finding_card"
        or taskset.get("record_causal_feedback") is not True
    ):
        raise AuditFailure("resolved taskset does not pin the natural N1 feedback boundary")

    pre_filters = orchestrator.get("pre_batch_filters")
    token_window = next(
        (
            item
            for item in pre_filters or []
            if item.get("type") == "trainable_token_window"
        ),
        None,
    )
    if (
        token_window is None
        or token_window.get("enforce") is not True
        or token_window.get("max_tokens") != TRAINING_SEQ_LEN
    ):
        raise AuditFailure("resolved audit does not enforce the trainer token window")

    model_paths = {
        trainer.get("model", {}).get("name"),
        orchestrator.get("model", {}).get("name"),
        inference.get("vllm", {}).get("model"),
    }
    if model_paths != {expected_model_path}:
        raise AuditFailure(
            f"resolved services do not all use canonical R7: {model_paths!r}"
        )


def _validate_metrics(run_dir: Path) -> dict[str, float]:
    records = _read_jsonl(run_dir / "metrics.jsonl")
    steps = {record.get("step") for record in records if "step" in record}
    if steps != {1}:
        raise AuditFailure(f"expected metrics only for step 1, found {sorted(steps)}")
    counts = _exported_component_token_counts(run_dir)
    if counts["sdpo"] <= 0:
        raise AuditFailure("token export has no SDPO signal")
    for name in ("rl", "ce", "ref_kl"):
        if counts[name] != 0:
            raise AuditFailure(f"unexpected {name} token mass: {counts[name]}")
    _require_all(records, "optim/lr", 0.0)
    _require_all(records, "optim/update_succeeded", 1.0)
    _require_all(records, "time/save_ckpt", 0.0)
    grad_norm = _require_finite(records, "optim/grad_norm")
    if grad_norm <= 0:
        raise AuditFailure(f"gradient norm must be positive, found {grad_norm:g}")
    return {
        "loss": _require_finite(records, "loss/mean"),
        "sdpo_loss": _require_finite(records, "sdpo/mean"),
        "gradient_norm": grad_norm,
        "sdpo_tokens": float(counts["sdpo"]),
    }


def _is_child_branch(branch: vf.Branch) -> bool:
    return any(
        isinstance(node.message, UserMessage)
        and content_text(node.message.content).lstrip().startswith("[task from parent]")
        for node in branch.nodes
    )


def _validate_traces(run_dir: Path, step: int = 1) -> list[vf.WireTrace]:
    path = (
        run_dir
        / "rollouts"
        / f"step_{step}"
        / "train"
        / "effective"
        / "traces.jsonl"
    )
    records = _read_jsonl(path)
    if len(records) != EXPECTED_BATCH_SIZE:
        raise AuditFailure(
            f"expected exactly {EXPECTED_BATCH_SIZE} effective traces, found {len(records)}"
        )
    traces = []
    for index, record in enumerate(records):
        if record.get("ok") is not True or record.get("errors"):
            raise AuditFailure(f"effective trace {index} is not a valid rollout")
        if record.get("info", {}).get("env_name") != ENV_NAME:
            raise AuditFailure(f"effective trace {index} came from another source")
        info = record.get("info", {})
        feedback = info.get("feedback")
        contract = info.get("feedback_contract")
        if (
            not isinstance(feedback, str)
            or not feedback.strip()
            or not isinstance(contract, dict)
            or contract.get("schema_version") != FEEDBACK_SCHEMA
            or contract.get("code") != FEEDBACK_CODE
            or contract.get("category") != "event_control"
            or contract.get("answer_free") is not True
            or contract.get("retryable") is not True
            or contract.get("message") != feedback
            or not isinstance(contract.get("target_node_index"), int)
            or not isinstance(contract.get("turn_index"), int)
        ):
            raise AuditFailure(f"effective trace {index} has invalid typed feedback")
        if record.get("rewards", {}).get("harness_score", {}).get("score") != 0:
            raise AuditFailure(f"effective trace {index} is not a diagnosed hard failure")
        task_data = record.get("task", {}).get("data", {})
        expected_results = [
            child.get("expected_result")
            for child in task_data.get("oracle", {}).get("children", [])
            if isinstance(child, dict)
        ]
        if any(str(value) in feedback for value in expected_results if value is not None):
            raise AuditFailure(f"effective trace {index} feedback leaks the task answer")

        trace = vf.WireTrace.model_validate(record)
        branches = list(iter_trainable_branches(trace))
        masks = keep_natural_yield_feedback_response(trace)
        if len(masks) != len(branches):
            raise AuditFailure(f"effective trace {index} has misaligned filter masks")
        if sum(any(mask) for mask in masks) != 1:
            raise AuditFailure(
                f"effective trace {index} does not select exactly one coordinator branch"
            )
        for branch_index, ((branch, _), mask) in enumerate(
            zip(branches, masks, strict=True)
        ):
            if len(mask) != len(branch.token_ids):
                raise AuditFailure(
                    f"effective trace {index} branch {branch_index} mask is misaligned"
                )
            if any(mask[TRAINING_SEQ_LEN:]):
                raise AuditFailure(
                    f"effective trace {index} selects tokens beyond the trainer window"
                )
            if _is_child_branch(branch) and any(mask):
                raise AuditFailure(f"effective trace {index} selects a child branch")
        traces.append(trace)
    return traces


def _validate_token_routing(
    run_dir: Path, traces: list[vf.WireTrace], step: int = 1
) -> dict[str, int]:
    export_dir = run_dir / "token_exports" / f"step_{step}"
    if not (export_dir / "STABLE").is_file():
        raise AuditFailure(f"token export is not stable: {export_dir}")
    exports = [
        record
        for path in sorted(export_dir.glob("rank_*.jsonl"))
        for record in _read_jsonl(path)
    ]
    by_sample: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = defaultdict(
        list
    )
    for index, record in enumerate(exports):
        if record.get("schema_version") != 1 or record.get("step") != step:
            raise AuditFailure(f"token export {index} has the wrong schema or step")
        token_ids = record.get("token_ids")
        if record.get("env_name") != ENV_NAME or not isinstance(token_ids, list):
            raise AuditFailure(f"token export {index} has invalid sample identity")
        by_sample[(ENV_NAME, tuple(token_ids))].append(record)

    consumed = 0
    coordinator_active = 0
    child_zero = 0
    for trace in traces:
        branches = list(iter_trainable_branches(trace))
        expected_masks = keep_natural_yield_feedback_response(trace)
        trace_active = 0
        for branch_index, ((branch, trainable_mask), expected) in enumerate(
            zip(branches, expected_masks, strict=True)
        ):
            candidates = by_sample.get((ENV_NAME, tuple(branch.token_ids)))
            if not candidates:
                raise AuditFailure(
                    f"no token export matches trace {trace.id} branch {branch_index}"
                )
            record = candidates.pop()
            consumed += 1
            if record.get("loss_mask") != trainable_mask:
                raise AuditFailure(f"trainer changed trace {trace.id}'s sampled mask")
            sdpo = _active_component(record, "sdpo_weights")
            if sdpo != expected:
                raise AuditFailure(f"SDPO mask differs from trace target in {trace.id}")
            if any(
                any(_active_component(record, f"{name}_weights"))
                for name in ("rl", "ce", "ref_kl")
            ):
                raise AuditFailure(f"another loss component leaked into trace {trace.id}")
            if _is_child_branch(branch):
                child_zero += 1
                if any(sdpo):
                    raise AuditFailure(f"child branch received SDPO in trace {trace.id}")
            elif any(sdpo):
                coordinator_active += 1
                trace_active += 1
        if trace_active != 1:
            raise AuditFailure(
                f"trace {trace.id} routes SDPO to {trace_active} coordinator branches"
            )

    leftovers = sum(len(records) for records in by_sample.values())
    if leftovers or consumed != len(exports):
        raise AuditFailure(
            "effective trace branches and token exports are not one-to-one"
        )
    if coordinator_active != EXPECTED_BATCH_SIZE or child_zero == 0:
        raise AuditFailure("routing audit did not cover every coordinator and child branch")
    return {
        "export_records": len(exports),
        "coordinator_active_samples": coordinator_active,
        "child_zero_sdpo_samples": child_zero,
    }


def validate(run_dir: Path, expected_model_path: str) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise AuditFailure(f"run directory does not exist: {run_dir}")
    _validate_configs(run_dir, expected_model_path)
    metrics = _validate_metrics(run_dir)
    traces = _validate_traces(run_dir)
    routing = _validate_token_routing(run_dir, traces)
    _validate_no_model_artifacts(run_dir)
    return {
        "verdict": "pass",
        "mechanism": "natural-yield-feedback-conditioned-sdpo-zero-lr",
        "expected_model_path": expected_model_path,
        "effective_traces": len(traces),
        "metrics": metrics,
        "token_routing": routing,
        "model_artifacts_written": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-model-path", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate(args.run_dir, args.expected_model_path)
    except (AuditFailure, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"natural-yield zero-LR audit failed: {error}") from error
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
