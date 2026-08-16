"""Validate failure-local routing in the 27B resilience SDPO audit."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from prime_agent_resilience_v1.taskset import (
    _recoverable_failure,
    keep_failed_ipython_tool_calls,
)
from verifiers.v1.types import AssistantMessage, ToolMessage, content_text

from prime_rl.orchestrator.trajectories import iter_trainable_branches

DEFAULT_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
ENV_NAME = "resilience-repair-sdpo"
FILTER_PATH = "prime_agent_resilience_v1.taskset.keep_failed_ipython_tool_calls"
EXPECTED_FAMILIES = {"malformed_result_repair", "message_type_repair"}
EXPECTED_BATCH_SIZE = 4
TRAINING_SEQ_LEN = 8192


class AuditFailure(ValueError):
    """The completed run does not prove failure-local SDPO routing."""


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
    values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not values or not all(isinstance(value, dict) for value in values):
        raise AuditFailure(f"expected JSON objects in {path}")
    return values


def _validate_configs(run_dir: Path, expected_revision: str) -> dict[str, str]:
    trainer = _read_json(run_dir / "configs" / "trainer.json")
    orchestrator = _read_json(run_dir / "configs" / "orchestrator.json")
    inference = _read_json(run_dir / "configs" / "inference.json")
    if trainer.get("max_steps") != 1 or orchestrator.get("max_steps") != 1:
        raise AuditFailure("audit must run exactly one step")
    if trainer.get("optim", {}).get("lr") != 0:
        raise AuditFailure("audit learning rate is not zero")
    if trainer.get("ckpt") is not None or orchestrator.get("ckpt") is not None:
        raise AuditFailure("zero-LR audit unexpectedly enables checkpointing")
    if not trainer.get("enable_token_export"):
        raise AuditFailure("audit must enable token export")
    if orchestrator.get("batch_size") != EXPECTED_BATCH_SIZE:
        raise AuditFailure(f"audit batch size must be {EXPECTED_BATCH_SIZE}")
    if trainer.get("model", {}).get("seq_len") != TRAINING_SEQ_LEN:
        raise AuditFailure(f"trainer sequence length must be {TRAINING_SEQ_LEN}")
    sources = orchestrator.get("train", {}).get("source")
    if not isinstance(sources, list) or len(sources) != 1:
        raise AuditFailure("audit must contain exactly one training source")
    source = sources[0]
    algo = source.get("algo", {})
    taskset = source.get("env", {}).get("taskset", {})
    if source.get("name") != ENV_NAME or source.get("group_size") != 1:
        raise AuditFailure("audit source identity or group size changed")
    if (
        algo.get("type") != "sdpo"
        or algo.get("filter", {}).get("import_path") != FILTER_PATH
        or not algo.get("multi_turn_replay")
        or not algo.get("include_environment_feedback")
    ):
        raise AuditFailure("audit does not pin failure-local multi-turn SDPO")
    if set(taskset.get("families", [])) != EXPECTED_FAMILIES:
        raise AuditFailure(f"unexpected resilience families: {taskset.get('families')}")
    paths = {
        "trainer": trainer.get("model", {}).get("name"),
        "orchestrator": orchestrator.get("model", {}).get("name"),
        "inference": inference.get("vllm", {}).get("model"),
    }
    if not all(isinstance(path, str) for path in paths.values()):
        raise AuditFailure(f"resolved model paths are incomplete: {paths}")
    revisions = {name: Path(path).name for name, path in paths.items()}
    if set(revisions.values()) != {expected_revision}:
        raise AuditFailure(f"resolved model revisions do not match: {revisions}")
    return revisions


def _active_component(record: dict[str, Any], name: str) -> list[bool]:
    loss_mask = record.get("loss_mask")
    weights = record.get(name)
    if not isinstance(loss_mask, list) or not all(isinstance(value, bool) for value in loss_mask):
        raise AuditFailure("token export has an invalid loss mask")
    if not isinstance(weights, list) or len(weights) != len(loss_mask):
        raise AuditFailure(f"token export has an invalid {name} stream")
    default = 1.0 if name == "rl_weights" else 0.0
    return [
        keep and float(default if weight is None else weight) != 0.0
        for keep, weight in zip(loss_mask, weights, strict=True)
    ]


def _weighted_reward(trace: vf.WireTrace) -> float:
    return sum(reward.score * reward.weight for reward in trace.rewards.values())


def _failure_outputs(trace: vf.WireTrace, node: object) -> list[str]:
    node_index = next(
        (index for index, candidate in enumerate(trace.nodes) if id(candidate) == id(node)),
        None,
    )
    if node_index is None or not isinstance(node.message, AssistantMessage):
        return []
    call_ids = {
        call.id for call in node.message.tool_calls or [] if call.name == "ipython"
    }
    return [
        content_text(candidate.message.content)
        for candidate in trace.nodes
        if candidate.parent == node_index
        and isinstance(candidate.message, ToolMessage)
        and candidate.message.tool_call_id in call_ids
        and _recoverable_failure(content_text(candidate.message.content))
    ]


def _failure_kind(output: str) -> str:
    for line in reversed(output.splitlines()):
        text = line.strip()
        if text.startswith("<coroutine object"):
            return "unawaited_coroutine"
        if text.startswith("I/O Error:"):
            return "io_error"
        if ":" in text and text.split(":", 1)[0].endswith("Error"):
            return text.split(":", 1)[0]
    return "recoverable_failure"


def _validate_traces(run_dir: Path) -> tuple[list[vf.WireTrace], dict[str, Any]]:
    path = run_dir / "rollouts" / "step_1" / "train" / "effective" / "traces.jsonl"
    records = _read_jsonl(path)
    if len(records) != EXPECTED_BATCH_SIZE:
        raise AuditFailure(f"expected {EXPECTED_BATCH_SIZE} effective traces, found {len(records)}")
    traces = [vf.WireTrace.model_validate(record) for record in records]
    families: Counter[str] = Counter()
    failure_kinds: Counter[str] = Counter()
    active_traces = 0
    active_nodes = 0
    for trace in traces:
        family = getattr(trace.task.data, "family", None)
        if family not in EXPECTED_FAMILIES:
            raise AuditFailure(f"unexpected effective family: {family}")
        families[family] += 1
        branches = list(iter_trainable_branches(trace))
        masks = keep_failed_ipython_tool_calls(trace)
        if len(masks) != len(branches):
            raise AuditFailure(f"trace {trace.id} filter masks do not align with branches")
        trace_active = 0
        for branch_index, ((branch, _), mask) in enumerate(zip(branches, masks, strict=True)):
            if len(mask) != len(branch.token_ids):
                raise AuditFailure(f"trace {trace.id} branch {branch_index} has an invalid mask")
            offset = 0
            for node in branch.nodes:
                node_mask = mask[offset : offset + len(node.token_ids)]
                offset += len(node.token_ids)
                if not any(node_mask):
                    continue
                outputs = _failure_outputs(trace, node)
                if not outputs:
                    raise AuditFailure(f"trace {trace.id} selected a call without failed tool feedback")
                trace_active += 1
                active_nodes += 1
                for output in outputs:
                    failure_kinds[_failure_kind(output)] += 1
        if trace_active:
            active_traces += 1
    if set(families) != EXPECTED_FAMILIES:
        raise AuditFailure(f"effective batch lacks family coverage: {dict(families)}")
    if active_traces < 2 or active_nodes < 2:
        raise AuditFailure(
            f"audit needs at least two failure-bearing traces and nodes, found {active_traces}/{active_nodes}"
        )
    return traces, {
        "count": len(traces),
        "families": dict(families),
        "active_traces": active_traces,
        "active_failed_nodes": active_nodes,
        "failure_kinds": dict(failure_kinds),
    }


def _validate_exports(run_dir: Path, traces: list[vf.WireTrace]) -> dict[str, int]:
    export_dir = run_dir / "token_exports" / "step_1"
    if not (export_dir / "STABLE").is_file():
        raise AuditFailure("token export is not stable")
    records = []
    for path in sorted(export_dir.glob("rank_*.jsonl")):
        records.extend(_read_jsonl(path))
    by_tokens: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("env_name") != ENV_NAME or not isinstance(record.get("token_ids"), list):
            raise AuditFailure("token export has an unexpected sample identity")
        by_tokens[tuple(record["token_ids"])].append(record)
    consumed = 0
    sdpo_tokens = 0
    zero_target_samples = 0
    for trace in traces:
        branches = list(iter_trainable_branches(trace))
        expected_masks = keep_failed_ipython_tool_calls(trace)
        succeeded = _weighted_reward(trace) >= 1.0
        for (branch, trainable_mask), expected in zip(branches, expected_masks, strict=True):
            candidates = by_tokens.get(tuple(branch.token_ids))
            if not candidates:
                raise AuditFailure(f"no export matches trace {trace.id}")
            record = candidates.pop()
            consumed += 1
            if record.get("loss_mask") != trainable_mask:
                raise AuditFailure(f"export changed the trainable mask for trace {trace.id}")
            actual = _active_component(record, "sdpo_weights")
            expected = [False] * len(expected) if succeeded else expected
            if actual != expected:
                raise AuditFailure(f"SDPO routing differs from the failed-call filter in trace {trace.id}")
            if any(_active_component(record, name) for name in ("rl_weights", "ce_weights", "ref_kl_weights")):
                raise AuditFailure(f"another loss component leaked into trace {trace.id}")
            count = sum(actual)
            sdpo_tokens += count
            zero_target_samples += int(count == 0)
    leftovers = sum(len(values) for values in by_tokens.values())
    if consumed != len(records) or leftovers:
        raise AuditFailure(
            f"exports are not one-to-one with branches: {consumed}/{len(records)}, leftovers={leftovers}"
        )
    if sdpo_tokens <= 0:
        raise AuditFailure("audit produced no SDPO token mass")
    return {
        "records": len(records),
        "sdpo_tokens": sdpo_tokens,
        "zero_target_samples": zero_target_samples,
    }


def _metric(records: list[dict[str, Any]], key: str) -> float:
    values = [record[key] for record in records if isinstance(record.get(key), (int, float))]
    if not values or not math.isfinite(float(values[-1])):
        raise AuditFailure(f"missing finite metric: {key}")
    return float(values[-1])


def _validate_metrics(run_dir: Path, exported_sdpo_tokens: int) -> dict[str, float]:
    records = _read_jsonl(run_dir / "metrics.jsonl")
    values = {
        "loss": _metric(records, "loss/mean"),
        "sdpo_loss": _metric(records, "sdpo/mean"),
        "grad_norm": _metric(records, "optim/grad_norm"),
        "lr": _metric(records, "optim/lr"),
        "update_succeeded": _metric(records, "optim/update_succeeded"),
    }
    if values["lr"] != 0 or values["update_succeeded"] != 1 or values["grad_norm"] <= 0:
        raise AuditFailure(f"invalid zero-LR update metrics: {values}")
    for component in ("rl", "ce", "ref_kl"):
        if _metric(records, f"loss_tokens/{component}") != 0:
            raise AuditFailure(f"unexpected {component} token mass")
    if _metric(records, "loss_tokens/sdpo") != exported_sdpo_tokens:
        raise AuditFailure("metric and exported SDPO token counts differ")
    return values


def validate(run_dir: Path, expected_revision: str = DEFAULT_REVISION) -> dict[str, Any]:
    revisions = _validate_configs(run_dir, expected_revision)
    traces, trace_report = _validate_traces(run_dir)
    exports = _validate_exports(run_dir, traces)
    metrics = _validate_metrics(run_dir, exports["sdpo_tokens"])
    for directory in (run_dir / "weights", run_dir / "checkpoints"):
        if directory.exists() and any(path.is_file() for path in directory.rglob("*")):
            raise AuditFailure(f"zero-LR audit wrote model artifacts under {directory}")
    return {
        "verdict": "pass",
        "mechanism": "failure-local-feedback-conditioned-sdpo-zero-lr",
        "expected_revision": expected_revision,
        "resolved_revisions": revisions,
        "traces": trace_report,
        "token_routing": exports,
        "metrics": metrics,
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
    except (AuditFailure, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"resilience SDPO audit failed: {error}") from error
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
