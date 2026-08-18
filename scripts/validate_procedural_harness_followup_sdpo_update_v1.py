"""Validate one failure-local procedural follow-up SDPO update."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from procedural_harness_master_v1.followup_feedback import FEEDBACK_SCHEMA_VERSION
from procedural_harness_master_v1.taskset import keep_followup_feedback_response

from prime_rl.orchestrator.trajectories import iter_trainable_branches

ENV_NAME = "procedural-followup-feedback-sdpo"
EXPECTED_CODE = "reply_to_child_request"


class UpdateValidationFailure(ValueError):
    """The run does not prove a valid failure-local SDPO update."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise UpdateValidationFailure(f"missing required file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise UpdateValidationFailure(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise UpdateValidationFailure(f"missing required file: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records or not all(isinstance(record, dict) for record in records):
        raise UpdateValidationFailure(f"no valid records found in {path}")
    return records


def _metric(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if key in record]
    if not values or not all(math.isfinite(value) for value in values):
        raise UpdateValidationFailure(f"missing finite metric: {key}")
    return values[-1]


def _effective_traces(run_dir: Path) -> list[vf.Trace]:
    records = _read_jsonl(run_dir / "rollouts" / "step_1" / "train" / "effective" / "traces.jsonl")
    rows: list[dict[str, Any]] = []
    for record in records:
        nested = record.get("traces")
        rows.extend(nested if isinstance(nested, list) else [record])
    return [vf.Trace.model_validate(row) for row in rows]


def _selected_tokens(trace: vf.Trace) -> int:
    masks = keep_followup_feedback_response(trace)
    branches = list(iter_trainable_branches(trace))
    if len(masks) != len(branches):
        raise UpdateValidationFailure(f"trace {trace.id} has misaligned branch masks")

    selected_nodes: list[int] = []
    selected_tokens = 0
    for (branch, _), mask in zip(branches, masks, strict=True):
        if len(mask) != len(branch.token_ids):
            raise UpdateValidationFailure(f"trace {trace.id} has a malformed branch mask")
        selected_tokens += sum(mask)
        offset = 0
        for node in branch.nodes:
            node_mask = mask[offset : offset + len(node.token_ids)]
            offset += len(node.token_ids)
            if any(node_mask):
                selected_nodes.append(next(index for index, item in enumerate(trace.nodes) if item is node))

    contract = trace.info.get("feedback_contract")
    if not isinstance(contract, dict):
        raise UpdateValidationFailure(f"trace {trace.id} has no typed feedback contract")
    target = contract.get("target_node_index")
    if selected_nodes != [target]:
        raise UpdateValidationFailure(
            f"trace {trace.id} selected nodes {selected_nodes}, expected {[target]}"
        )
    if selected_tokens <= 0:
        raise UpdateValidationFailure(f"trace {trace.id} has no selected reply tokens")
    return selected_tokens


def _validate_traces(run_dir: Path, expected_count: int) -> dict[str, Any]:
    traces = _effective_traces(run_dir)
    if len(traces) != expected_count:
        raise UpdateValidationFailure(
            f"expected {expected_count} effective traces, found {len(traces)}"
        )

    selected_tokens = 0
    for trace in traces:
        if trace.errors:
            raise UpdateValidationFailure(f"trace {trace.id} contains runtime errors")
        contract = trace.info.get("feedback_contract")
        if not isinstance(contract, dict):
            raise UpdateValidationFailure(f"trace {trace.id} has no feedback contract")
        if (
            contract.get("schema_version") != FEEDBACK_SCHEMA_VERSION
            or contract.get("code") != EXPECTED_CODE
            or contract.get("answer_free") is not True
            or contract.get("retryable") is not True
            or trace.info.get("feedback") != contract.get("message")
        ):
            raise UpdateValidationFailure(f"trace {trace.id} has an untrusted feedback contract")
        selected_tokens += _selected_tokens(trace)
    return {"count": len(traces), "selected_tokens": selected_tokens}


def _active_count(record: dict[str, Any], name: str) -> int:
    values = record.get(name)
    if not isinstance(values, list):
        raise UpdateValidationFailure(f"token export has no {name} stream")
    return sum(value is not None and value != 0 for value in values)


def _validate_token_exports(run_dir: Path, expected_tokens: int) -> dict[str, Any]:
    export_dir = run_dir / "token_exports" / "step_1"
    if not (export_dir / "STABLE").is_file():
        raise UpdateValidationFailure(f"token exports are not stable: {export_dir}")
    records = [record for path in sorted(export_dir.glob("rank_*.jsonl")) for record in _read_jsonl(path)]
    sdpo_tokens = 0
    for record in records:
        token_ids = record.get("token_ids")
        loss_mask = record.get("loss_mask")
        if not isinstance(token_ids, list) or not isinstance(loss_mask, list) or len(token_ids) != len(loss_mask):
            raise UpdateValidationFailure("token export has malformed token or loss-mask columns")
        if record.get("env_name") != ENV_NAME:
            raise UpdateValidationFailure(f"unexpected token-export environment: {record.get('env_name')}")
        sdpo_weights = record.get("sdpo_weights")
        if not isinstance(sdpo_weights, list) or len(sdpo_weights) != len(loss_mask):
            raise UpdateValidationFailure("token export has a malformed SDPO stream")
        if any(
            _active_count(record, name)
            for name in ("rl_weights", "ce_weights", "ref_kl_weights")
        ):
            raise UpdateValidationFailure("non-SDPO component received token mass")
        if any(weight not in (None, 0, 0.0) and not keep for keep, weight in zip(loss_mask, sdpo_weights, strict=True)):
            raise UpdateValidationFailure("SDPO weights escaped the loss mask")
        sdpo_tokens += _active_count(record, "sdpo_weights")
    if sdpo_tokens != expected_tokens:
        raise UpdateValidationFailure(
            f"trainer received {sdpo_tokens} SDPO tokens, expected {expected_tokens} from trace routing"
        )
    return {"records": len(records), "sdpo_tokens": sdpo_tokens}


def _validate_configs(run_dir: Path) -> dict[str, Any]:
    trainer = _read_json(run_dir / "configs" / "trainer.json")
    orchestrator = _read_json(run_dir / "configs" / "orchestrator.json")
    sources = orchestrator.get("train", {}).get("source", [])
    if trainer.get("max_steps") != 1 or orchestrator.get("max_steps") != 1:
        raise UpdateValidationFailure("failure-local update must run exactly one optimizer step")
    if len(sources) != 1 or sources[0].get("name") != ENV_NAME:
        raise UpdateValidationFailure("failure-local update must have exactly one dedicated source")
    source = sources[0]
    algo = source.get("algo", {})
    if (
        algo.get("type") != "sdpo"
        or algo.get("required_feedback_contract_schema") != FEEDBACK_SCHEMA_VERSION
        or algo.get("filter", {}).get("import_path")
        != "procedural_harness_master_v1.taskset.keep_followup_feedback_response"
        or algo.get("multi_turn_replay") is not True
        or algo.get("environment_feedback_only_without_solution") is not True
    ):
        raise UpdateValidationFailure("failure-local SDPO routing configuration changed")
    model = trainer.get("model", {})
    if model.get("lora") is not None or model.get("optimization_dtype") != "bfloat16":
        raise UpdateValidationFailure("failure-local update must train full BF16 weights")
    lr = float(trainer.get("optim", {}).get("lr", 0))
    if not math.isfinite(lr) or lr <= 0:
        raise UpdateValidationFailure("failure-local update has no positive finite learning rate")
    if not trainer.get("enable_token_export"):
        raise UpdateValidationFailure("failure-local update must export token routing")
    return {"batch_size": int(orchestrator.get("batch_size", 0)), "learning_rate": lr}


def _validate_metrics(run_dir: Path, expected_tokens: int) -> dict[str, float]:
    records = _read_jsonl(run_dir / "metrics.jsonl")
    values = {
        "rl_tokens": _metric(records, "loss_tokens/rl"),
        "ce_tokens": _metric(records, "loss_tokens/ce"),
        "ref_kl_tokens": _metric(records, "loss_tokens/ref_kl"),
        "sdpo_tokens": _metric(records, "loss_tokens/sdpo"),
        "grad_norm": _metric(records, "optim/grad_norm"),
        "update_succeeded": _metric(records, "optim/update_succeeded"),
    }
    other_token_counts = (values[name] for name in ("rl_tokens", "ce_tokens", "ref_kl_tokens"))
    if values["sdpo_tokens"] != expected_tokens or any(value != 0 for value in other_token_counts):
        raise UpdateValidationFailure("optimizer component token mass does not match routing")
    if values["update_succeeded"] != 1 or values["grad_norm"] <= 0:
        raise UpdateValidationFailure("optimizer update did not succeed with a positive gradient")
    return values


def _validate_weights(run_dir: Path) -> Path:
    weights = run_dir / "weights" / "step_1"
    if not (weights / "STABLE").is_file():
        raise UpdateValidationFailure(f"weight checkpoint is not stable: {weights}")
    if not list(weights.glob("*.safetensors")):
        raise UpdateValidationFailure(f"weight checkpoint has no safetensors: {weights}")
    return weights


def validate(run_dir: Path) -> dict[str, Any]:
    config = _validate_configs(run_dir)
    traces = _validate_traces(run_dir, config["batch_size"])
    exports = _validate_token_exports(run_dir, traces["selected_tokens"])
    metrics = _validate_metrics(run_dir, traces["selected_tokens"])
    weights = _validate_weights(run_dir)
    return {
        "verdict": "pass",
        "mechanism": "failure-local-procedural-followup-sdpo",
        "config": config,
        "traces": traces,
        "token_routing": exports,
        "metrics": metrics,
        "weights": str(weights),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate(args.run_dir)
    except (UpdateValidationFailure, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"follow-up SDPO update validation failed: {error}") from error
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
