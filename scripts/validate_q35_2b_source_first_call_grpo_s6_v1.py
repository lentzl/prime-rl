#!/usr/bin/env python3
"""Fail-closed validation for the S6 first-call GRPO audit and update configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def _validate_runtime_configs(run_dir: Path, stage: Literal["audit", "update"]) -> dict[str, Any]:
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
            or taskset.get("instance_offset") != EXPECTED_TRAIN_OFFSET
            or taskset.get("seed") != EXPECTED_TRAIN_SEED
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
    if audit_log != str(EXPECTED_ROUTING_AUDIT[stage]):
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


def validate_runtime(run_dir: Path, stage: Literal["audit", "update"]) -> dict[str, Any]:
    import verifiers.v1 as vf
    from verifiers.v1.types import UserMessage, content_text

    from prime_rl.orchestrator.trajectories import iter_trainable_branches

    config_report = _validate_runtime_configs(run_dir, stage)
    records = _read_jsonl(run_dir / "metrics.jsonl")
    expected_lr = 0.0 if stage == "audit" else 1e-6
    if _metric(records, "optim/lr") != expected_lr:
        raise AuditFailure(f"runtime {stage} learning rate changed")
    if _metric(records, "optim/update_succeeded") != 1:
        raise AuditFailure("zero-LR forward/backward did not complete")
    grad_norm = _metric(records, "optim/grad_norm")
    if grad_norm <= 0:
        raise AuditFailure("zero-LR audit did not produce a positive finite gradient")

    trace_path = run_dir / "rollouts" / "step_1" / "train" / "effective" / "traces.jsonl"
    trace_records = _read_jsonl(trace_path)
    if len(trace_records) != EXPECTED_BATCH_SIZE:
        raise AuditFailure(f"S6 audit requires exactly {EXPECTED_BATCH_SIZE} effective traces")
    routing_report = _validate_forced_assignment_routes(
        trace_records, _read_jsonl(Path(config_report["routing_audit"]))
    )
    task_rewards: dict[str, list[float]] = defaultdict(list)
    families: Counter[str] = Counter()
    source_rollouts: Counter[str] = Counter()
    source_groups: dict[str, set[str]] = defaultdict(set)
    typed: list[vf.WireTrace] = []
    child_branches = 0
    child_trainable_tokens = 0
    coordinator_trainable_tokens = 0
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
                coordinator_trainable_tokens += active
        typed.append(trace)
    if families != Counter({family: 8 for family in EXPECTED_FAMILIES}):
        raise AuditFailure(f"S6 effective batch is not balanced: {dict(families)}")
    if source_rollouts != Counter(EXPECTED_SOURCE_MINIMUMS) or any(
        len(source_groups[name]) != 1 for name in EXPECTED_SOURCE_FAMILIES
    ):
        raise AuditFailure("S6 effective batch is not exactly one group per family source")
    if len(task_rewards) != 2 or any(len(values) != 8 for values in task_rewards.values()):
        raise AuditFailure("S6 audit did not contain two complete eight-way groups")
    if any(len(set(values)) < 2 for values in task_rewards.values()):
        raise AuditFailure("S6 audit contains a zero-variance reward group")
    if child_trainable_tokens <= 0 or child_branches <= 0:
        raise AuditFailure("S6 has no positive trainable child token mass")
    if coordinator_trainable_tokens != 0:
        raise AuditFailure("S6 leaked trainable reward mass to the coordinator")

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
        "gradient_norm": grad_norm,
        "families": dict(families),
        "source_rollouts": dict(source_rollouts),
        "source_groups": {name: sorted(group_ids) for name, group_ids in source_groups.items()},
        "groups": {key: values for key, values in task_rewards.items()},
        "child_branches": child_branches,
        "child_trainable_tokens": child_trainable_tokens,
        "coordinator_trainable_tokens": coordinator_trainable_tokens,
        "exported_rl_tokens": exported_rl_tokens,
        "checkpoint_written": stage == "update",
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
