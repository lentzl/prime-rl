#!/usr/bin/env python3
"""Fail-closed validation for the S6 first-call GRPO audit and update configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

S5_PATH = Path(
    "/home/ubuntu/rlm/outputs/q35-2b-source-worker-remedial-s5-v1/h-source-s5-remedial-step8-v1/weights/step_8"
)
E33_PATH = Path(
    "/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/"
    "c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2"
)
EXPECTED_FAMILIES = {"specialist_source_ast", "specialist_source_config"}
EXPECTED_SOURCE_FAMILIES = {
    "source-worker-ast-s6": "specialist_source_ast",
    "source-worker-config-s6": "specialist_source_config",
}
EXPECTED_SOURCE_MINIMUMS = {name: 8 for name in EXPECTED_SOURCE_FAMILIES}
EXPECTED_REWARD = "source_worker_first_call"
EXPECTED_BATCH_SIZE = 16
EXPECTED_GROUP_SIZE = 8
EXPECTED_TRAIN_SEED = 20270909
EXPECTED_TRAIN_OFFSET = 70000
EXPECTED_ROUTING_AUDIT = {
    "audit": Path("/home/ubuntu/rlm/results/q35-2b-source-first-call-s6-v1/zero-lr-routing-audit.jsonl"),
    "update": Path("/home/ubuntu/rlm/results/q35-2b-source-first-call-s6-v1/step1-routing-audit.jsonl"),
}
FORCED_ROUTE_MODE = "forced_specialist_assignment_generate_action"
ROUTE_SCHEMA = "qwen35-2b-dual-policy-route/v1"
SPECIALIST_CHILD_PREFIX = (
    "[task from parent]\n\n"
    "[selected terminal capability]\n"
    "expert_id=source_inspector\n"
    "session_role=terminal_worker"
)
MAX_MISMATCH_TO_ENTROPY_RATIO = 0.25
MAX_MISMATCH_OUTLIER_TO_ENTROPY_MAX = 100.0
MAX_DPPO_MASKED_FRACTION = 0.05


class AuditFailure(ValueError):
    """S6 did not prove that rewards and gradients reach only source-worker tokens."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AuditFailure(f"missing required JSONL: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records or not all(isinstance(record, dict) for record in records):
        raise AuditFailure(f"invalid or empty JSONL: {path}")
    return records


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditFailure(f"missing required JSON: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AuditFailure(f"expected a JSON object: {path}")
    return value


def _table(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or not isinstance(value.get(key), dict):
            raise AuditFailure(f"missing config table: {'.'.join(keys)}")
        value = value[key]
    return value


def _zero_advantage_filter(payload: dict[str, Any], slot: str) -> dict[str, Any]:
    filters = _table(payload, "orchestrator").get(slot)
    if not isinstance(filters, list):
        raise AuditFailure(f"missing {slot}")
    match = next((item for item in filters if item.get("type") == "zero_advantage"), None)
    if not isinstance(match, dict):
        raise AuditFailure(f"missing zero-advantage filter in {slot}")
    return match


def validate_config(path: Path, stage: Literal["audit", "update"]) -> dict[str, Any]:
    payload = tomllib.loads(path.read_text())
    if payload.get("max_steps") != 1 or payload.get("clean") is not False:
        raise AuditFailure("S6 requires one write-once step")
    deployment = _table(payload, "deployment")
    if deployment.get("num_train_gpus") != 1 or deployment.get("num_infer_gpus") != 1:
        raise AuditFailure("S6 must use exactly one trainer and one inference GPU")
    if _table(payload, "model").get("name") != str(S5_PATH):
        raise AuditFailure("S6 must initialize from the exact rejected S5 checkpoint")
    trainer = _table(payload, "trainer")
    if trainer.get("enable_token_export") is not True:
        raise AuditFailure("S6 requires token export")
    if "lora" in trainer.get("model", {}) or "lora" in _table(payload, "model"):
        raise AuditFailure("S6 must be full-dense")
    lr = _table(payload, "trainer", "optim").get("lr")
    expected_lr = 0.0 if stage == "audit" else 1e-6
    if lr != expected_lr:
        raise AuditFailure(f"{stage} LR must equal {expected_lr:g}")
    loss = _table(payload, "trainer", "loss")
    if loss.get("type") != "default" or loss.get("kl_tau") != 0.001:
        raise AuditFailure("S6 must retain its predeclared rollout-policy KL control")
    has_trainer_ckpt = isinstance(trainer.get("ckpt"), dict)
    has_orchestrator_ckpt = isinstance(_table(payload, "orchestrator").get("ckpt"), dict)
    if (has_trainer_ckpt or has_orchestrator_ckpt) != (stage == "update"):
        raise AuditFailure("only the conditional update may write a checkpoint")

    orchestrator = _table(payload, "orchestrator")
    if (
        orchestrator.get("batch_size") != EXPECTED_BATCH_SIZE
        or orchestrator.get("group_size") != EXPECTED_GROUP_SIZE
        or orchestrator.get("max_inflight_episodes") != EXPECTED_BATCH_SIZE
        or orchestrator.get("max_off_policy_steps") != 0
        or orchestrator.get("max_train_batch_lead") != 0
    ):
        raise AuditFailure("S6 requires two complete on-policy groups")
    if orchestrator.get("batch_source_minimums") != EXPECTED_SOURCE_MINIMUMS:
        raise AuditFailure("S6 requires one complete group from each family source")
    if _table(payload, "orchestrator", "algo") != {
        "type": "grpo",
        "sampled_session_scope": "non_root",
    }:
        raise AuditFailure("S6 top-level policy scope must be non-root GRPO")
    expected_enforce = stage == "update"
    for slot in ("pre_batch_filters", "post_batch_filters"):
        if _zero_advantage_filter(payload, slot).get("enforce") is not expected_enforce:
            raise AuditFailure(f"{stage} zero-variance policy is wrong in {slot}")

    sampling = _table(payload, "orchestrator", "train", "sampling")
    if (
        sampling.get("temperature") != 1.0
        or sampling.get("reasoning_effort") != "high"
        or sampling.get("max_completion_tokens") != 2048
        or sampling.get("extra_body", {}).get("return_token_ids") is not True
    ):
        raise AuditFailure("S6 sampling is not the exploratory token-visible contract")
    sources = _table(payload, "orchestrator", "train").get("source")
    if not isinstance(sources, list) or len(sources) != 2:
        raise AuditFailure("S6 requires exactly two family-isolated learning sources")
    sources_by_name = {source.get("name"): source for source in sources}
    if set(sources_by_name) != set(EXPECTED_SOURCE_FAMILIES):
        raise AuditFailure("S6 family source identities changed")
    for name, family in EXPECTED_SOURCE_FAMILIES.items():
        source = sources_by_name[name]
        if source.get("group_size") != EXPECTED_GROUP_SIZE or source.get("algo") != {
            "type": "grpo",
            "sampled_session_scope": "non_root",
        }:
            raise AuditFailure("S6 source must preserve eight-way non-root GRPO groups")
        taskset = source.get("env", {}).get("taskset", {})
        if (
            taskset.get("id") != "source-worker-first-call-v1"
            or taskset.get("split") != "train"
            or taskset.get("families") != [family]
            or taskset.get("instances_per_template") != 8
            or taskset.get("instance_offset") != EXPECTED_TRAIN_OFFSET
            or taskset.get("seed") != EXPECTED_TRAIN_SEED
            or taskset.get("teacher_conditioned", False)
            or taskset.get("ownership_guided", False)
            or taskset.get("task", {}).get("reward_mode") != EXPECTED_REWARD
        ):
            raise AuditFailure(f"S6 {name} is not its fresh train-only isolated reward taskset")

    router = _table(payload, "inference", "router")
    if (
        router.get("type") != "role-router"
        or router.get("policy_role") != "child"
        or router.get("anchor_model") != str(E33_PATH)
        or router.get("specialist_fixed_expert") != "source_inspector"
        or router.get("specialist_force_fixed_action") is not True
        or router.get("strip_child_tool_choice") is not True
        or router.get("leak_child_exact_action") is not False
        or router.get("leak_coordinator_exact_action") is not False
    ):
        raise AuditFailure("S6 role router does not isolate the source worker from frozen e33")
    return {
        "stage": stage,
        "learning_rate": lr,
        "batch_size": EXPECTED_BATCH_SIZE,
        "group_size": EXPECTED_GROUP_SIZE,
        "policy_scope": "non_root",
        "checkpoint_enabled": stage == "update",
        "family_sources": EXPECTED_SOURCE_FAMILIES,
        "batch_source_minimums": EXPECTED_SOURCE_MINIMUMS,
    }


def _metric(records: list[dict[str, Any]], key: str) -> float:
    values = [record[key] for record in records if key in record]
    if not values or not all(isinstance(value, (int, float)) for value in values):
        raise AuditFailure(f"missing numeric metric: {key}")
    value = float(values[-1])
    if not math.isfinite(value):
        raise AuditFailure(f"non-finite metric: {key}")
    return value


def _validate_prospective_lr0_health(
    records: list[dict[str, Any]], trace_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Reject censored or trainer/inference-misaligned LR=0 mechanisms.

    These are update-authorization checks, not specialist admission gates.
    They deliberately require at least one non-max-turn completion in each
    family and near-on-policy token likelihoods before a gradient may be used.
    """

    stops: dict[str, Counter[str]] = {
        family: Counter() for family in EXPECTED_FAMILIES
    }
    for index, record in enumerate(trace_records):
        family = record.get("task", {}).get("data", {}).get("family")
        stop = record.get("stop_condition")
        if family not in stops or not isinstance(stop, str) or not stop:
            raise AuditFailure(
                f"effective trace {index} lacks prospective stop/family evidence"
            )
        stops[family][stop] += 1
    for family, counts in stops.items():
        if sum(counts.values()) <= 0 or counts["max_turns"] == sum(counts.values()):
            raise AuditFailure(
                f"prospective LR0 audit is 100% max-turn censored for {family}"
            )

    entropy_mean = _metric(records, "entropy/all/mean")
    entropy_max = _metric(records, "entropy/all/max")
    mismatch_mean = _metric(records, "unmasked_mismatch_kl/mean")
    mismatch_max = _metric(records, "mismatch_kl/all/max")
    masked_fraction = _metric(records, "is_masked/mean")
    if entropy_mean <= 0 or entropy_max <= 0:
        raise AuditFailure("prospective LR0 audit has no finite positive entropy")
    mismatch_ratio = mismatch_mean / entropy_mean
    mismatch_outlier_ratio = mismatch_max / entropy_max
    if mismatch_ratio > MAX_MISMATCH_TO_ENTROPY_RATIO:
        raise AuditFailure(
            "prospective LR0 audit has pathological unmasked trainer/inference mismatch"
        )
    if mismatch_outlier_ratio > MAX_MISMATCH_OUTLIER_TO_ENTROPY_MAX:
        raise AuditFailure(
            "prospective LR0 audit has pathological mismatch-KL outliers"
        )
    if masked_fraction > MAX_DPPO_MASKED_FRACTION:
        raise AuditFailure(
            "prospective LR0 audit masks too much trainer/inference mismatch"
        )
    return {
        "stop_conditions": {
            family: dict(sorted(counts.items())) for family, counts in stops.items()
        },
        "unmasked_mismatch_to_entropy_ratio": mismatch_ratio,
        "mismatch_outlier_to_entropy_max_ratio": mismatch_outlier_ratio,
        "dppo_masked_fraction": masked_fraction,
        "limits": {
            "unmasked_mismatch_to_entropy_ratio": MAX_MISMATCH_TO_ENTROPY_RATIO,
            "mismatch_outlier_to_entropy_max_ratio": MAX_MISMATCH_OUTLIER_TO_ENTROPY_MAX,
            "dppo_masked_fraction": MAX_DPPO_MASKED_FRACTION,
        },
    }


def _observe_lr0_health(
    records: list[dict[str, Any]], trace_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Record calibration values without interpreting them as pass/fail gates."""

    stops: dict[str, Counter[str]] = {
        family: Counter() for family in EXPECTED_FAMILIES
    }
    for index, record in enumerate(trace_records):
        family = record.get("task", {}).get("data", {}).get("family")
        stop = record.get("stop_condition")
        if family not in stops or not isinstance(stop, str) or not stop:
            raise AuditFailure(
                f"effective trace {index} lacks calibration stop/family evidence"
            )
        stops[family][stop] += 1
    aggregate_metric_keys = (
        "entropy/all/mean",
        "entropy/all/std",
        "entropy/all/max",
        "unmasked_mismatch_kl/mean",
        "unmasked_mismatch_kl/max",
        "mismatch_kl/all/mean",
        "mismatch_kl/all/std",
        "mismatch_kl/all/max",
        "masked_mismatch_kl/mean",
        "masked_mismatch_kl/max",
        "is_masked/mean",
        "is_masked/max",
        "is_masked_low/mean",
        "is_masked_low/max",
        "is_masked_high/mean",
        "is_masked_high/max",
        "kl_ent_ratio/mean",
    )
    per_source_metric_keys = tuple(
        f"{metric}/{source}/{statistic}"
        for metric in ("entropy", "mismatch_kl")
        for source in EXPECTED_SOURCE_FAMILIES
        for statistic in ("mean", "std", "max")
    )
    metrics = {
        key: _metric(records, key)
        for key in (*aggregate_metric_keys, *per_source_metric_keys)
    }
    return {
        "interpretation": "observational_only_no_thresholds_applied",
        "stop_conditions": {
            family: dict(sorted(counts.items())) for family, counts in stops.items()
        },
        "max_turn_fraction": {
            family: counts["max_turns"] / sum(counts.values())
            for family, counts in stops.items()
        },
        "metrics": metrics,
        "thresholds_evaluated": False,
        "thresholds_frozen": False,
    }


def _is_child_branch(branch: Any, user_message_type: type, content_text_fn: Any) -> bool:
    return any(
        isinstance(node.message, user_message_type)
        and content_text_fn(node.message.content).lstrip().startswith("[task from parent]")
        for node in branch.nodes
    )


def _active_rl_tokens(record: dict[str, Any]) -> int:
    mask = record.get("loss_mask")
    weights = record.get("rl_weights")
    if not isinstance(mask, list) or not all(isinstance(value, bool) for value in mask):
        raise AuditFailure("invalid token-export loss mask")
    if weights is None:
        return sum(mask)
    if not isinstance(weights, list) or len(mask) != len(weights):
        raise AuditFailure("invalid token-export RL stream")
    return sum(
        keep and float(1.0 if weight is None else weight) != 0.0 for keep, weight in zip(mask, weights, strict=True)
    )


def _validate_runtime_configs(
    run_dir: Path,
    stage: Literal["audit", "update"],
    *,
    routing_audit_path: Path | None = None,
    expected_train_seed: int = EXPECTED_TRAIN_SEED,
    expected_train_offset: int = EXPECTED_TRAIN_OFFSET,
) -> dict[str, Any]:
    trainer = _read_json(run_dir / "configs" / "trainer.json")
    orchestrator = _read_json(run_dir / "configs" / "orchestrator.json")
    inference = _read_json(run_dir / "configs" / "inference.json")
    if trainer.get("max_steps") != 1 or orchestrator.get("max_steps") != 1:
        raise AuditFailure("resolved S6 audit must run exactly one step")
    expected_lr = 0.0 if stage == "audit" else 1e-6
    if trainer.get("optim", {}).get("lr") != expected_lr:
        raise AuditFailure(f"resolved S6 {stage} learning rate is wrong")
    if trainer.get("enable_token_export") is not True:
        raise AuditFailure("resolved S6 audit disabled token export")
    checkpoint_enabled = isinstance(trainer.get("ckpt"), dict) and isinstance(orchestrator.get("ckpt"), dict)
    if checkpoint_enabled != (stage == "update"):
        raise AuditFailure(f"resolved S6 {stage} checkpoint policy is wrong")
    if trainer.get("loss", {}).get("kl_tau") != 0.001:
        raise AuditFailure("resolved S6 audit changed the rollout-policy KL control")
    if (
        orchestrator.get("batch_size") != EXPECTED_BATCH_SIZE
        or orchestrator.get("group_size") != EXPECTED_GROUP_SIZE
        or orchestrator.get("max_inflight_episodes") != EXPECTED_BATCH_SIZE
        or orchestrator.get("max_off_policy_steps") != 0
        or orchestrator.get("max_train_batch_lead") != 0
        or orchestrator.get("algo", {}).get("type") != "grpo"
        or orchestrator.get("algo", {}).get("sampled_session_scope") != "non_root"
    ):
        raise AuditFailure("resolved S6 audit changed its GRPO batch or role scope")
    if orchestrator.get("batch_source_minimums") != EXPECTED_SOURCE_MINIMUMS:
        raise AuditFailure("resolved S6 audit lost deterministic family balance")
    for slot in ("pre_batch_filters", "post_batch_filters"):
        filters = orchestrator.get(slot)
        zero = next(
            (item for item in filters or [] if item.get("type") == "zero_advantage"),
            None,
        )
        if not isinstance(zero, dict) or zero.get("enforce") is not (stage == "update"):
            raise AuditFailure(f"resolved {stage} zero-variance policy is wrong in {slot}")
    sources = orchestrator.get("train", {}).get("source")
    if not isinstance(sources, list) or len(sources) != 2:
        raise AuditFailure("resolved S6 audit does not have two isolated sources")
    sources_by_name = {source.get("name"): source for source in sources}
    if set(sources_by_name) != set(EXPECTED_SOURCE_FAMILIES):
        raise AuditFailure("resolved S6 family source identities changed")
    for name, family in EXPECTED_SOURCE_FAMILIES.items():
        source = sources_by_name[name]
        taskset = source.get("env", {}).get("taskset", {})
        if (
            source.get("group_size") != EXPECTED_GROUP_SIZE
            or source.get("algo", {}).get("sampled_session_scope") != "non_root"
            or taskset.get("id") != "source-worker-first-call-v1"
            or taskset.get("split") != "train"
            or taskset.get("families") != [family]
            or taskset.get("instance_offset") != expected_train_offset
            or taskset.get("seed") != expected_train_seed
            or taskset.get("task", {}).get("reward_mode") != EXPECTED_REWARD
        ):
            raise AuditFailure(f"resolved S6 {name} taskset or scope changed")
    router = inference.get("router")
    model_paths = {
        "trainer": trainer.get("model", {}).get("name"),
        "orchestrator": orchestrator.get("model", {}).get("name"),
        "inference": inference.get("vllm", {}).get("model"),
    }
    if set(model_paths.values()) != {str(S5_PATH)}:
        raise AuditFailure(f"resolved trainable model identity mismatch: {model_paths}")
    if (
        not isinstance(router, dict)
        or router.get("type") != "role-router"
        or router.get("policy_role") != "child"
        or router.get("anchor_model") != str(E33_PATH)
        or router.get("specialist_fixed_expert") != "source_inspector"
        or router.get("specialist_force_fixed_action") is not True
    ):
        raise AuditFailure("resolved S6 audit changed role isolation or fixed routing")
    audit_log = router.get("audit_log")
    expected_routing_audit = routing_audit_path or EXPECTED_ROUTING_AUDIT[stage]
    if audit_log != str(expected_routing_audit):
        raise AuditFailure("resolved S6 audit changed its write-once routing audit")
    return {
        "trainer_model": model_paths["trainer"],
        "anchor_model": router["anchor_model"],
        "policy_scope": "non_root",
        "checkpoint_enabled": checkpoint_enabled,
        "routing_audit": audit_log,
    }


def _ipython_code(tool_call: Any) -> str | None:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = tool_call.get("name")
        arguments = tool_call.get("arguments")
    if name != "ipython" or not isinstance(arguments, str):
        return None
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    code = payload.get("code") if isinstance(payload, dict) else None
    return code if isinstance(code, str) else None


def _forced_assignment_identity(record: dict[str, Any], index: int) -> str:
    matches: list[str] = []
    calls = record.get("calls")
    if not isinstance(calls, list):
        raise AuditFailure(f"effective trace {index} lacks call/session evidence")
    for node_index, node in enumerate(record.get("nodes", [])):
        # The harness may retain a normalized, non-sampled copy of the accepted
        # assignment later in the coordinator branch.  Only the sampled node
        # corresponds to an inference call and therefore carries session evidence.
        if isinstance(node, dict) and node.get("sampled") is False:
            continue
        message = node.get("message") if isinstance(node, dict) else None
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            code = _ipython_code(tool_call)
            if (
                code is not None
                and "task_worker = await rlm(" in code
                and "expert_id=source_inspector" in code
                and 'name="task-worker"' in code
            ):
                session_ids = {
                    call.get("client_session_id")
                    for call in calls
                    if isinstance(call, dict) and call.get("node") == node_index
                }
                if len(session_ids) != 1 or not all(
                    isinstance(session_id, str) and session_id for session_id in session_ids
                ):
                    raise AuditFailure(f"effective trace {index} forced action lacks one session id")
                matches.append(hashlib.sha256(code.encode()).hexdigest())
    if len(matches) != 1:
        raise AuditFailure(f"effective trace {index} has {len(matches)} canonical forced assignments")
    return matches[0]


def _validate_forced_assignment_routes(
    trace_records: list[dict[str, Any]], audit_records: list[dict[str, Any]]
) -> dict[str, Any]:
    trace_actions: Counter[str] = Counter()
    group_actions: dict[str, set[str]] = defaultdict(set)
    for index, record in enumerate(trace_records):
        task = record.get("task", {})
        task_key = task.get("key") or task.get("hash")
        if not isinstance(task_key, str):
            raise AuditFailure(f"effective trace {index} lacks a task key for route audit")
        action_sha = _forced_assignment_identity(record, index)
        trace_actions[action_sha] += 1
        group_actions[task_key].add(action_sha)
    if len(group_actions) != 2 or any(len(actions) != 1 for actions in group_actions.values()):
        raise AuditFailure("forced assignment is not stable within both GRPO groups")

    forced = [record for record in audit_records if record.get("mode") == FORCED_ROUTE_MODE]
    forced_by_session: dict[str, str] = {}
    for index, record in enumerate(forced):
        session_sha = record.get("session_sha256")
        action_sha = record.get("action_sha256")
        if (
            record.get("schema_version") != "qwen35-2b-dual-policy-route/v1"
            or record.get("endpoint") != "/inference/v1/generate"
            or record.get("role") != "coordinator"
            or record.get("expert_id") != "source_inspector"
            or record.get("status") != 200
            or not isinstance(session_sha, str)
            or len(session_sha) != 64
            or not isinstance(action_sha, str)
            or len(action_sha) != 64
        ):
            raise AuditFailure(f"invalid forced-assignment route event {index}")
        if session_sha in forced_by_session:
            raise AuditFailure("forced assignment was emitted more than once in a session")
        forced_by_session[session_sha] = action_sha
    audit_actions = Counter(forced_by_session.values())
    if trace_actions - audit_actions:
        raise AuditFailure("effective traces lack forced-assignment route action multiplicity")
    return {
        "mode": FORCED_ROUTE_MODE,
        "events": len(forced),
        "matched_effective_events": sum(trace_actions.values()),
        "extra_filtered_events": len(forced) - sum(trace_actions.values()),
        "sessions": len(forced_by_session),
        "groups": len(group_actions),
        "effective_action_sha256_counts": dict(trace_actions),
    }


def _wire_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def _call_branch_texts(
    nodes: list[Any], node_index: Any, *, trace_index: int, call_index: int
) -> list[str]:
    if not isinstance(node_index, int):
        raise AuditFailure(
            f"effective trace {trace_index} call {call_index} is unattached"
        )
    texts: list[str] = []
    visited: set[int] = set()
    current: int | None = node_index
    while current is not None:
        if current in visited or current < 0 or current >= len(nodes):
            raise AuditFailure(
                f"effective trace {trace_index} call {call_index} has invalid ancestry"
            )
        visited.add(current)
        node = nodes[current]
        if not isinstance(node, dict):
            raise AuditFailure(
                f"effective trace {trace_index} call {call_index} has invalid node"
            )
        message = node.get("message")
        if isinstance(message, dict):
            text = _wire_content_text(message.get("content"))
            if text:
                texts.append(text)
        parent = node.get("parent")
        if parent is not None and not isinstance(parent, int):
            raise AuditFailure(
                f"effective trace {trace_index} call {call_index} has invalid parent"
            )
        current = parent
    return texts


def _attached_call_role(
    nodes: list[Any], call: dict[str, Any], *, trace_index: int, call_index: int
) -> str:
    texts = _call_branch_texts(
        nodes,
        call.get("node"),
        trace_index=trace_index,
        call_index=call_index,
    )
    parent_tasks = [
        text for text in texts if text.lstrip().startswith("[task from parent]")
    ]
    if parent_tasks:
        if len(parent_tasks) != 1 or not parent_tasks[0].startswith(
            SPECIALIST_CHILD_PREFIX
        ):
            raise AuditFailure(
                f"effective trace {trace_index} call {call_index} is not the exact source_inspector child"
            )
        return "child"
    if any("[specialist worker routing contract]" in text for text in texts):
        return "coordinator"
    raise AuditFailure(f"effective trace {trace_index} call {call_index} is unattached")


def _validated_route_event_role(record: dict[str, Any], index: int) -> str:
    role = record.get("role")
    session_sha = record.get("session_sha256")
    if (
        record.get("schema_version") != ROUTE_SCHEMA
        or record.get("endpoint") != "/inference/v1/generate"
        or record.get("status") != 200
        or role not in {"coordinator", "child"}
        or not isinstance(session_sha, str)
        or len(session_sha) != 64
    ):
        raise AuditFailure(f"invalid effective route audit event {index}")
    if role == "child":
        if (
            record.get("upstream_model") != str(S5_PATH)
            or record.get("expert_id") != "source_inspector"
            or record.get("route_evidence") != "exact_specialist_child_prefix"
            or not str(record.get("mode", "")).startswith("forwarded")
        ):
            raise AuditFailure(
                f"effective child route {index} is not exact source_inspector/S5"
            )
    elif (
        record.get("upstream_model") != str(E33_PATH)
        or record.get("route_evidence")
        != "coordinator_without_specialist_child_prefix"
    ):
        raise AuditFailure(f"effective coordinator route {index} is not frozen e33")
    if record.get("mode") == FORCED_ROUTE_MODE and role != "coordinator":
        raise AuditFailure("forced specialist assignment was not coordinator-owned")
    return role


def _validate_effective_call_routes(
    trace_records: list[dict[str, Any]], audit_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Reconcile node ancestry with every successful generate route event.

    The proxy request id is rollout-scoped, not call-scoped, so there is no
    honest one-to-one call join. The caller first proves the raw and effective
    trace sets are identical; this check then requires exact aggregate
    call/event multiplicity, rejects unattached calls, and requires each event
    to carry token-native role evidence plus the exact upstream identity.
    """

    expected: Counter[str] = Counter()
    for trace_index, record in enumerate(trace_records):
        nodes = record.get("nodes")
        calls = record.get("calls")
        if not isinstance(nodes, list) or not isinstance(calls, list):
            raise AuditFailure(
                f"effective trace {trace_index} lacks route reconciliation evidence"
            )
        for call_index, call in enumerate(calls):
            if not isinstance(call, dict):
                raise AuditFailure(
                    f"effective trace {trace_index} call {call_index} is invalid"
                )
            texts = _call_branch_texts(
                nodes,
                call.get("node"),
                trace_index=trace_index,
                call_index=call_index,
            )
            parent_tasks = [
                text for text in texts if text.lstrip().startswith("[task from parent]")
            ]
            if parent_tasks:
                if len(parent_tasks) != 1 or not parent_tasks[0].startswith(
                    SPECIALIST_CHILD_PREFIX
                ):
                    raise AuditFailure(
                        f"effective trace {trace_index} call {call_index} is not the exact source_inspector child"
                    )
                expected["child"] += 1
            elif any("[specialist worker routing contract]" in text for text in texts):
                expected["coordinator"] += 1
            else:
                raise AuditFailure(
                    f"effective trace {trace_index} call {call_index} is unattached"
                )

    observed: Counter[str] = Counter()
    sessions: set[str] = set()
    forced = 0
    for index, record in enumerate(audit_records):
        role = record.get("role")
        session_sha = record.get("session_sha256")
        if (
            record.get("schema_version") != ROUTE_SCHEMA
            or record.get("endpoint") != "/inference/v1/generate"
            or record.get("status") != 200
            or role not in {"coordinator", "child"}
            or not isinstance(session_sha, str)
            or len(session_sha) != 64
        ):
            raise AuditFailure(f"invalid effective route audit event {index}")
        sessions.add(session_sha)
        if role == "child":
            if (
                record.get("upstream_model") != str(S5_PATH)
                or record.get("expert_id") != "source_inspector"
                or record.get("route_evidence") != "exact_specialist_child_prefix"
                or not str(record.get("mode", "")).startswith("forwarded")
            ):
                raise AuditFailure(
                    f"effective child route {index} is not exact source_inspector/S5"
                )
        elif (
            record.get("upstream_model") != str(E33_PATH)
            or record.get("route_evidence")
            != "coordinator_without_specialist_child_prefix"
        ):
            raise AuditFailure(f"effective coordinator route {index} is not frozen e33")
        if record.get("mode") == FORCED_ROUTE_MODE:
            if role != "coordinator":
                raise AuditFailure("forced specialist assignment was not coordinator-owned")
            forced += 1
        observed[role] += 1

    if expected != observed:
        raise AuditFailure(
            f"effective call/route multiplicity differs: expected={dict(expected)} observed={dict(observed)}"
        )
    if forced != EXPECTED_BATCH_SIZE or len(sessions) != EXPECTED_BATCH_SIZE:
        raise AuditFailure(
            "effective route audit must have one forced assignment and one session per rollout"
        )
    return {
        "expected_calls": dict(expected),
        "observed_routes": dict(observed),
        "forced_assignments": forced,
        "rollout_sessions": len(sessions),
        "child_upstream": str(S5_PATH),
        "coordinator_upstream": str(E33_PATH),
    }


def _validate_p0_terminal_route_artifacts(
    trace_records: list[dict[str, Any]], audit_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Forensically reconcile P0's already-completed terminal call artifacts.

    This is intentionally exact to the retained P0 evidence. It is not an
    alternative admission rule: prospective audits continue to use
    ``_validate_effective_call_routes`` and reject every unattached call.
    """

    if len(trace_records) != EXPECTED_BATCH_SIZE:
        raise AuditFailure("P0 recovery requires the exact 16 retained traces")
    audit_by_session: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    sequences: set[int] = set()
    for index, record in enumerate(audit_records):
        _validated_route_event_role(record, index)
        session_sha = record["session_sha256"]
        sequence = record.get("sequence")
        if not isinstance(sequence, int) or sequence in sequences:
            raise AuditFailure("P0 route audit lacks unique integer sequence evidence")
        sequences.add(sequence)
        audit_by_session[session_sha].append((index, record))
    if len(audit_by_session) != EXPECTED_BATCH_SIZE:
        raise AuditFailure("P0 recovery requires exactly 16 rollout audit sessions")

    attached_counts: Counter[str] = Counter()
    terminal_artifacts: list[dict[str, Any]] = []
    audit_only_artifacts: list[dict[str, Any]] = []
    session_inventory: list[dict[str, Any]] = []
    matched_sessions: set[str] = set()
    for trace_index, record in enumerate(trace_records):
        trace_id = record.get("id")
        nodes = record.get("nodes")
        calls = record.get("calls")
        if not isinstance(trace_id, str) or not isinstance(nodes, list) or not isinstance(calls, list):
            raise AuditFailure(f"P0 retained trace {trace_index} is malformed")
        text = "\n".join(
            _wire_content_text(node.get("message", {}).get("content"))
            for node in nodes
            if isinstance(node, dict) and isinstance(node.get("message"), dict)
        )
        prefixes = set(
            re.findall(r"/vf-prime-agent-runs/([0-9a-f]{16})/", text)
        )
        matching_sessions = [
            session for session in audit_by_session if session[:16] in prefixes
        ]
        if len(prefixes) != 1 or len(matching_sessions) != 1:
            raise AuditFailure(
                f"P0 retained trace {trace_index} lacks one rollout/audit session bijection"
            )
        session_sha = matching_sessions[0]
        if session_sha in matched_sessions:
            raise AuditFailure("P0 rollout audit session matched more than one trace")
        matched_sessions.add(session_sha)

        per_attached: Counter[str] = Counter()
        null_calls: list[tuple[int, dict[str, Any]]] = []
        for call_index, call in enumerate(calls):
            if not isinstance(call, dict):
                raise AuditFailure(
                    f"P0 retained trace {trace_index} call {call_index} is malformed"
                )
            if call.get("node") is None:
                null_calls.append((call_index, call))
                continue
            if call.get("error") is not None:
                raise AuditFailure("P0 attached call contains an inference error")
            role = _attached_call_role(
                nodes, call, trace_index=trace_index, call_index=call_index
            )
            per_attached[role] += 1
            attached_counts[role] += 1

        session_events = sorted(
            audit_by_session[session_sha], key=lambda item: item[1]["sequence"]
        )
        audit_roles = Counter(record["role"] for _, record in session_events)
        residual_roles = audit_roles - per_attached
        if per_attached - audit_roles:
            raise AuditFailure("P0 attached calls exceed audited routes in a session")
        extra_audits = len(session_events) - len(calls)
        if extra_audits < 0 or extra_audits > 1:
            raise AuditFailure("P0 session has an unaccountable route/call cardinality")
        if sum(residual_roles.values()) != len(null_calls) + extra_audits:
            raise AuditFailure("P0 terminal route residual does not close exactly")
        if residual_roles and len(residual_roles) != 1:
            raise AuditFailure("P0 terminal artifact role is ambiguous")
        if null_calls and extra_audits:
            raise AuditFailure("P0 mixes null-call and audit-only residue in one session")

        residual_role = next(iter(residual_roles), None)
        residual_events = (
            [
                (index, event)
                for index, event in session_events
                if event["role"] == residual_role
            ][-sum(residual_roles.values()) :]
            if residual_role is not None
            else []
        )
        if len(residual_events) != len(null_calls) + extra_audits:
            raise AuditFailure("P0 cannot identify the terminal route event tail")

        stop = record.get("stop_condition")
        for (call_index, call), (audit_index, event) in zip(
            null_calls, residual_events[: len(null_calls)], strict=True
        ):
            timing = call.get("time")
            if (
                not isinstance(timing, dict)
                or not all(isinstance(timing.get(key), (int, float)) for key in ("start", "end"))
                or timing["end"] < timing["start"]
            ):
                raise AuditFailure("P0 terminal trace-call artifact lacks valid timing")
            error = call.get("error")
            if error is None and stop == "max_turns":
                category = "successful_call_unattached_at_max_turns"
            elif (
                isinstance(error, dict)
                and error.get("type") == "ClientConnectionResetError"
                and stop == "agent_completed"
            ):
                category = "delivery_reset_after_agent_completed"
            else:
                raise AuditFailure("P0 null-call artifact is not an exact terminal/cancelled case")
            terminal_artifacts.append(
                {
                    "trace_index": trace_index,
                    "trace_id": trace_id,
                    "call_index": call_index,
                    "session_sha256": session_sha,
                    "inferred_role": residual_role,
                    "category": category,
                    "stop_condition": stop,
                    "finish_reason": call.get("finish_reason"),
                    "error_type": error.get("type") if isinstance(error, dict) else None,
                    "start": float(timing["start"]),
                    "end": float(timing["end"]),
                    "matched_audit_sequence": event["sequence"],
                    "matched_audit_request_sha256": event.get("request_sha256"),
                    "matched_audit_index": audit_index,
                }
            )

        if extra_audits:
            audit_index, event = residual_events[-1]
            if stop != "max_turns" or residual_role != "child":
                raise AuditFailure("P0 audit-only route is not the exact terminal child case")
            audit_only_artifacts.append(
                {
                    "trace_index": trace_index,
                    "trace_id": trace_id,
                    "session_sha256": session_sha,
                    "inferred_role": residual_role,
                    "category": "successful_child_route_after_trace_finalization",
                    "stop_condition": stop,
                    "audit_index": audit_index,
                    "audit_sequence": event["sequence"],
                    "request_sha256": event.get("request_sha256"),
                    "latency_ms": event.get("latency_ms"),
                }
            )

        forced = sum(
            event.get("mode") == FORCED_ROUTE_MODE for _, event in session_events
        )
        if forced != 1:
            raise AuditFailure("P0 session lacks exactly one forced coordinator assignment")
        session_inventory.append(
            {
                "trace_index": trace_index,
                "trace_id": trace_id,
                "session_sha256": session_sha,
                "attached_calls": dict(per_attached),
                "null_calls": len(null_calls),
                "audit_routes": dict(audit_roles),
                "audit_only_routes": extra_audits,
            }
        )

    if matched_sessions != set(audit_by_session):
        raise AuditFailure("P0 route audit contains a session with no retained trace")
    terminal_categories = Counter(item["category"] for item in terminal_artifacts)
    audit_role_counts = Counter(record["role"] for record in audit_records)
    forced_role_counts = Counter(
        record["role"]
        for record in audit_records
        if record.get("mode") == FORCED_ROUTE_MODE
    )
    natural_role_counts = audit_role_counts - forced_role_counts
    if (
        attached_counts != Counter({"child": 73, "coordinator": 55})
        or audit_role_counts != Counter({"child": 76, "coordinator": 57})
        or forced_role_counts != Counter({"coordinator": 16})
        or natural_role_counts != Counter({"child": 76, "coordinator": 41})
        or terminal_categories
        != Counter(
            {
                "successful_call_unattached_at_max_turns": 3,
                "delivery_reset_after_agent_completed": 1,
            }
        )
        or len(audit_only_artifacts) != 1
        or len(audit_records) != 133
    ):
        raise AuditFailure("P0 terminal-artifact inventory differs from retained evidence")
    return {
        "scope": "retained_P0_postflight_only_not_prospective_admission",
        "attached_successful_calls": dict(attached_counts),
        "attached_successful_call_total": sum(attached_counts.values()),
        "terminal_trace_call_artifacts": terminal_artifacts,
        "terminal_trace_call_categories": dict(terminal_categories),
        "audit_only_successful_events": audit_only_artifacts,
        "route_events": len(audit_records),
        "route_events_by_role": dict(audit_role_counts),
        "forced_route_events_by_role": dict(forced_role_counts),
        "natural_route_events_by_role": dict(natural_role_counts),
        "rollout_sessions": len(audit_by_session),
        "session_inventory": session_inventory,
        "residual_matching_basis": "exact rollout-path/session-prefix bijection, role cardinality residual, and terminal sequence tail",
        "future_proof_unchanged": "prospective audits reject every node=null call and any audit surplus",
    }


def _validate_p1_timed_call_routes(
    trace_records: list[dict[str, Any]], audit_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Prospectively join every P1 call to one route by session, role, and time."""

    audits_by_session: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    sequences: set[int] = set()
    request_hashes: set[str] = set()
    for index, event in enumerate(audit_records):
        _validated_route_event_role(event, index)
        sequence = event.get("sequence")
        request_sha = event.get("request_sha256")
        start = event.get("request_start_unix_s")
        end = event.get("request_end_unix_s")
        if (
            not isinstance(sequence, int)
            or sequence in sequences
            or not isinstance(request_sha, str)
            or len(request_sha) != 64
            or request_sha in request_hashes
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or end < start
        ):
            raise AuditFailure("P1 route event lacks unique sequence/request/timing evidence")
        sequences.add(sequence)
        request_hashes.add(request_sha)
        audits_by_session[event["session_sha256"]].append((index, event))
    if sequences != set(range(len(audit_records))):
        raise AuditFailure("P1 route sequences are not contiguous from zero")

    attached: Counter[str] = Counter()
    terminal_residues: list[dict[str, Any]] = []
    used_audits: set[int] = set()
    matched_sessions: set[str] = set()
    total_calls = 0
    for trace_index, trace in enumerate(trace_records):
        nodes = trace.get("nodes")
        calls = trace.get("calls")
        if not isinstance(nodes, list) or not isinstance(calls, list):
            raise AuditFailure(f"P1 trace {trace_index} lacks call ancestry")
        wire_text = "\n".join(
            _wire_content_text(node.get("message", {}).get("content"))
            for node in nodes
            if isinstance(node, dict) and isinstance(node.get("message"), dict)
        )
        prefixes = set(re.findall(r"/vf-prime-agent-runs/([0-9a-f]{16})/", wire_text))
        sessions = [session for session in audits_by_session if session[:16] in prefixes]
        if len(prefixes) != 1 or len(sessions) != 1 or sessions[0] in matched_sessions:
            raise AuditFailure("P1 trace lacks an exact unique rollout/audit session")
        session = sessions[0]
        matched_sessions.add(session)
        events = audits_by_session[session]
        if sum(event.get("mode") == FORCED_ROUTE_MODE for _, event in events) != 1:
            raise AuditFailure("P1 rollout lacks exactly one forced coordinator route")
        attached_ends = [
            call.get("time", {}).get("end")
            for call in calls
            if isinstance(call, dict) and call.get("node") is not None
        ]
        if not attached_ends or not all(isinstance(value, (int, float)) for value in attached_ends):
            raise AuditFailure("P1 trace lacks attached-call terminal timing")
        max_attached_end = max(attached_ends)
        for call_index, call in enumerate(calls):
            total_calls += 1
            if not isinstance(call, dict) or not isinstance(call.get("time"), dict):
                raise AuditFailure("P1 call lacks wall-clock timing")
            call_start = call["time"].get("start")
            call_end = call["time"].get("end")
            if (
                not isinstance(call_start, (int, float))
                or not isinstance(call_end, (int, float))
                or call_end < call_start
            ):
                raise AuditFailure("P1 call has invalid wall-clock timing")
            role = (
                _attached_call_role(
                    nodes, call, trace_index=trace_index, call_index=call_index
                )
                if call.get("node") is not None
                else None
            )
            candidates = [
                (audit_index, event)
                for audit_index, event in events
                if audit_index not in used_audits
                and (role is None or event["role"] == role)
                and event["request_start_unix_s"] <= call_end
                and event["request_end_unix_s"] >= call_start
            ]
            if len(candidates) != 1:
                raise AuditFailure(
                    f"P1 trace {trace_index} call {call_index} lacks one timed route join"
                )
            audit_index, event = candidates[0]
            used_audits.add(audit_index)
            if role is not None:
                if call.get("error") is not None:
                    raise AuditFailure("P1 attached call contains an inference error")
                attached[role] += 1
                continue
            error = call.get("error")
            stop = trace.get("stop_condition")
            if (
                call_index != len(calls) - 1
                or call_end < max_attached_end
                or stop not in {"max_turns", "agent_completed"}
                or (
                error is not None
                and (
                    not isinstance(error, dict)
                    or error.get("type") != "ClientConnectionResetError"
                )
                )
            ):
                raise AuditFailure("P1 node=null call is not a terminal cancelled residue")
            usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
            completion_tokens = usage.get("completion_tokens")
            if (
                not isinstance(completion_tokens, int)
                or completion_tokens < 0
                or event["request_end_unix_s"] < max_attached_end
            ):
                raise AuditFailure("P1 residue lacks terminal wasted-token evidence")
            terminal_residues.append(
                {
                    "trace_index": trace_index,
                    "call_index": call_index,
                    "role": event["role"],
                    "audit_sequence": event["sequence"],
                    "request_sha256": event["request_sha256"],
                    "training_effect_evidence": "node=null excludes the call from every trainable branch/export linkage",
                    "wasted_completion_tokens": completion_tokens,
                }
            )

    if matched_sessions != set(audits_by_session) or len(used_audits) != len(audit_records):
        raise AuditFailure("P1 has an audit-only event, missing trace, or missing join")
    if len(terminal_residues) > 1 or (
        terminal_residues and (len(terminal_residues) / total_calls) > 0.01
    ):
        raise AuditFailure("P1 terminal residue exceeds one call or one percent")
    if len({item["trace_index"] for item in terminal_residues}) > 1:
        raise AuditFailure("P1 terminal residue spans more than one rollout")
    return {
        "attached_successful_calls": dict(attached),
        "total_trace_calls": total_calls,
        "route_events": len(audit_records),
        "terminal_residues": terminal_residues,
        "audit_only_events": 0,
        "timing_session_bijection": True,
        "rollout_sessions": len(audits_by_session),
    }


def _validate_raw_equals_effective(
    raw_records: list[dict[str, Any]], effective_records: list[dict[str, Any]]
) -> dict[str, Any]:
    def identities(records: list[dict[str, Any]], label: str) -> set[str]:
        values = [record.get("id") for record in records]
        if not all(isinstance(value, str) and value for value in values):
            raise AuditFailure(f"{label} trace set lacks stable identities")
        if len(set(values)) != len(values):
            raise AuditFailure(f"{label} trace set contains duplicate identities")
        return set(values)

    raw_ids = identities(raw_records, "raw")
    effective_ids = identities(effective_records, "effective")
    if raw_ids != effective_ids:
        raise AuditFailure(
            "calibration requires identical raw and effective trace sets before route reconciliation"
        )
    return {"raw": len(raw_ids), "effective": len(effective_ids), "identical": True}


def validate_runtime(
    run_dir: Path,
    stage: Literal["audit", "update"],
    *,
    calibration_only: bool = False,
    routing_audit_path: Path | None = None,
    routing_audit_read_path: Path | None = None,
    p0_terminal_artifact_recovery: bool = False,
    p1_timed_route_admission: bool = False,
    expected_train_seed: int = EXPECTED_TRAIN_SEED,
    expected_train_offset: int = EXPECTED_TRAIN_OFFSET,
) -> dict[str, Any]:
    import verifiers.v1 as vf
    from verifiers.v1.types import UserMessage, content_text

    from prime_rl.orchestrator.trajectories import iter_trainable_branches

    if calibration_only and stage != "audit":
        raise AuditFailure("calibration-only validation requires LR=0 audit stage")
    if p0_terminal_artifact_recovery and not calibration_only:
        raise AuditFailure("P0 terminal-artifact recovery is calibration-only")
    if p0_terminal_artifact_recovery and p1_timed_route_admission:
        raise AuditFailure("P0 recovery and P1 admission routes are mutually exclusive")
    config_report = _validate_runtime_configs(
        run_dir,
        stage,
        routing_audit_path=routing_audit_path,
        expected_train_seed=expected_train_seed,
        expected_train_offset=expected_train_offset,
    )
    records = _read_jsonl(run_dir / "metrics.jsonl")
    expected_lr = 0.0 if stage == "audit" else 1e-6
    if _metric(records, "optim/lr") != expected_lr:
        raise AuditFailure(f"runtime {stage} learning rate changed")
    if _metric(records, "optim/update_succeeded") != 1:
        raise AuditFailure("zero-LR forward/backward did not complete")
    grad_norm = _metric(records, "optim/grad_norm")
    if grad_norm <= 0 and not calibration_only:
        raise AuditFailure("zero-LR audit did not produce a positive finite gradient")

    trace_path = run_dir / "rollouts" / "step_1" / "train" / "effective" / "traces.jsonl"
    trace_records = _read_jsonl(trace_path)
    if len(trace_records) != EXPECTED_BATCH_SIZE:
        raise AuditFailure(f"S6 audit requires exactly {EXPECTED_BATCH_SIZE} effective traces")
    health_report = None
    if stage == "audit":
        health_report = (
            _observe_lr0_health(records, trace_records)
            if calibration_only
            else _validate_prospective_lr0_health(records, trace_records)
        )
    raw_trace_records = _read_jsonl(
        run_dir / "rollouts" / "step_1" / "train" / "all" / "traces.jsonl"
    )
    trace_set_report = _validate_raw_equals_effective(
        raw_trace_records, trace_records
    )
    audit_records = _read_jsonl(
        routing_audit_read_path or Path(config_report["routing_audit"])
    )
    routing_report = _validate_forced_assignment_routes(trace_records, audit_records)
    effective_call_routes = (
        _validate_p0_terminal_route_artifacts(trace_records, audit_records)
        if p0_terminal_artifact_recovery
        else (
            _validate_p1_timed_call_routes(trace_records, audit_records)
            if p1_timed_route_admission
            else _validate_effective_call_routes(trace_records, audit_records)
        )
    )
    task_rewards: dict[str, list[float]] = defaultdict(list)
    families: Counter[str] = Counter()
    source_rollouts: Counter[str] = Counter()
    source_groups: dict[str, set[str]] = defaultdict(set)
    typed: list[vf.WireTrace] = []
    child_branches = 0
    child_trainable_tokens = 0
    raw_coordinator_sampled_tokens = 0
    expected_exports: list[tuple[tuple[int, ...], list[bool]]] = []
    for index, record in enumerate(trace_records):
        if record.get("ok") is not True or record.get("errors"):
            raise AuditFailure(f"effective trace {index} is invalid")
        task = record.get("task", {}).get("data", {})
        family = task.get("family")
        task_key = record.get("task", {}).get("key") or record.get("task", {}).get("hash")
        reward = record.get("rewards", {}).get(EXPECTED_REWARD, {}).get("score")
        if family not in EXPECTED_FAMILIES or not isinstance(task_key, str):
            raise AuditFailure(f"effective trace {index} has invalid source task identity")
        if not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise AuditFailure(f"effective trace {index} has no finite S6 reward")
        expected_source = next(
            name for name, source_family in EXPECTED_SOURCE_FAMILIES.items() if source_family == family
        )
        info = record.get("info", {})
        env_name = info.get("env_name") if isinstance(info, dict) else None
        group_id = info.get("group_id") if isinstance(info, dict) else None
        if env_name != expected_source or not isinstance(group_id, str) or not group_id:
            raise AuditFailure(f"effective trace {index} is not assigned to its family source/group")
        leaked_rewards = {
            name: payload.get("score")
            for name, payload in record.get("rewards", {}).items()
            if name != EXPECTED_REWARD and isinstance(payload, dict) and payload.get("score") not in (0, 0.0)
        }
        if leaked_rewards:
            raise AuditFailure(f"effective trace {index} has non-S6 reward leakage: {leaked_rewards}")
        families[family] += 1
        source_rollouts[env_name] += 1
        source_groups[env_name].add(group_id)
        task_rewards[task_key].append(float(reward))
        trace = vf.WireTrace.model_validate(record)
        for branch, mask in iter_trainable_branches(trace):
            active = sum(mask)
            if _is_child_branch(branch, UserMessage, content_text):
                child_branches += 1
                child_trainable_tokens += active
                expected_exports.append((tuple(branch.token_ids), mask))
            else:
                raw_coordinator_sampled_tokens += active
        typed.append(trace)
    if families != Counter({family: 8 for family in EXPECTED_FAMILIES}):
        raise AuditFailure(f"S6 effective batch is not balanced: {dict(families)}")
    if source_rollouts != Counter(EXPECTED_SOURCE_MINIMUMS) or any(
        len(source_groups[name]) != 1 for name in EXPECTED_SOURCE_FAMILIES
    ):
        raise AuditFailure("S6 effective batch is not exactly one group per family source")
    if len(task_rewards) != 2 or any(len(values) != 8 for values in task_rewards.values()):
        raise AuditFailure("S6 audit did not contain two complete eight-way groups")
    if not calibration_only and any(
        len(set(values)) < 2 for values in task_rewards.values()
    ):
        raise AuditFailure("S6 audit contains a zero-variance reward group")
    if child_trainable_tokens <= 0 or child_branches <= 0:
        raise AuditFailure("S6 has no positive trainable child token mass")
    export_dir = run_dir / "token_exports" / "step_1"
    if not (export_dir / "STABLE").is_file():
        raise AuditFailure("S6 token export is not stable")
    exports = [record for path in sorted(export_dir.glob("rank_*.jsonl")) for record in _read_jsonl(path)]
    exports_by_tokens: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(exports):
        if record.get("schema_version") != 1 or record.get("step") != 1:
            raise AuditFailure(f"invalid S6 token-export identity at record {index}")
        token_ids = record.get("token_ids")
        if not isinstance(token_ids, list) or not all(isinstance(token_id, int) for token_id in token_ids):
            raise AuditFailure(f"invalid S6 token ids at export record {index}")
        exports_by_tokens[tuple(token_ids)].append(record)
    exported_rl_tokens = 0
    for token_ids, mask in expected_exports:
        candidates = exports_by_tokens.get(token_ids)
        if not candidates:
            raise AuditFailure("a trainable child branch has no matching token export")
        record = candidates.pop()
        if record.get("loss_mask") != mask:
            raise AuditFailure("token export changed a child branch training mask")
        active = _active_rl_tokens(record)
        if active <= 0:
            raise AuditFailure("a trainable child branch has no positive RL token mass")
        exported_rl_tokens += active
    leftovers = sum(len(records) for records in exports_by_tokens.values())
    if leftovers or len(exports) != len(expected_exports):
        raise AuditFailure("S6 token exports are not one-to-one with trainable child branches")
    if exported_rl_tokens <= 0:
        raise AuditFailure("S6 exported no positive RL token mass")
    if exported_rl_tokens != child_trainable_tokens:
        raise AuditFailure("S6 child branch and exported RL token mass differ")
    if _metric(records, "loss_tokens/rl") != exported_rl_tokens:
        raise AuditFailure("S6 trainer consumed reward mass outside child exports")
    if any(
        any(value not in (None, 0, 0.0) for value in (record.get(stream) or []))
        for record in exports
        for stream in ("ce_weights", "ref_kl_weights", "sdpo_weights")
    ):
        raise AuditFailure("non-RL loss component leaked into S6")
    model_files = [
        path
        for directory in (run_dir / "checkpoints", run_dir / "weights")
        if directory.exists()
        for path in directory.rglob("*")
        if path.is_file()
    ]
    if stage == "audit" and model_files:
        raise AuditFailure("zero-LR audit wrote a forbidden model artifact")
    if stage == "update" and not (
        (run_dir / "weights" / "step_1" / "STABLE").is_file()
        and (run_dir / "weights" / "step_1" / "model.safetensors").is_file()
    ):
        raise AuditFailure("S6 update did not write one stable step-1 checkpoint")
    return {
        "schema_version": f"q35-2b-source-first-call-grpo-{stage}/v1",
        "verdict": "pass",
        "stage": stage,
        "config": config_report,
        "routing": routing_report,
        "effective_call_routes": effective_call_routes,
        "trace_sets": trace_set_report,
        "prospective_lr0_health": health_report,
        "gradient_norm": grad_norm,
        "families": dict(families),
        "source_rollouts": dict(source_rollouts),
        "source_groups": {name: sorted(group_ids) for name, group_ids in source_groups.items()},
        "groups": {key: values for key, values in task_rewards.items()},
        "child_branches": child_branches,
        "child_trainable_tokens": child_trainable_tokens,
        "raw_coordinator_sampled_tokens": raw_coordinator_sampled_tokens,
        "coordinator_exported_trainable_tokens": 0,
        "exported_rl_tokens": exported_rl_tokens,
        "checkpoint_written": stage == "update",
        "calibration_only": calibration_only,
        "p0_terminal_artifact_recovery": p0_terminal_artifact_recovery,
        "p1_timed_route_admission": p1_timed_route_admission,
        "optimizer_update_authorized": False if calibration_only else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("--stage", choices=("audit", "update"), default="audit")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate_runtime(args.target, args.stage) if args.runtime else validate_config(args.target, args.stage)
    except (AuditFailure, json.JSONDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise SystemExit(f"S6 validation failed: {error}") from error
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite S6 audit result: {args.output}")
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
