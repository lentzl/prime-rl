#!/usr/bin/env python3
"""Fail-closed audit for one resolved Qwen 3.5 2B role-GRPO step."""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path
from typing import Any


class AuditFailure(ValueError):
    pass


def _table(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or not isinstance(current.get(key), dict):
            raise AuditFailure(f"missing table: {'.'.join(keys)}")
        current = current[key]
    return current


def audit(
    config_path: Path,
    *,
    role: str,
    model_path: Path,
    anchor_model_path: Path,
    run_name: str,
    bootstrap_path: Path,
    phase: str,
    start_index: int,
    task_count: int,
) -> dict[str, Any]:
    payload = tomllib.loads(config_path.read_text())
    expected_scope = {"coordinator": "root", "child": "non_root"}.get(role)
    if expected_scope is None:
        raise AuditFailure(f"unsupported role: {role}")
    if payload.get("max_steps") != 1 or payload.get("clean") is not False:
        raise AuditFailure("role GRPO must make exactly one non-destructive optimizer step")
    if _table(payload, "run").get("name") != run_name or _table(payload, "run").get("dir") != run_name:
        raise AuditFailure("run name and directory must equal the unique requested label")
    if _table(payload, "model").get("name") != str(model_path):
        raise AuditFailure("resolved model path mismatch")
    trainer_model = _table(payload, "trainer", "model")
    if "lora" in trainer_model or "lora" in _table(payload, "model"):
        raise AuditFailure("role GRPO must be a full-dense update")
    lr = _table(payload, "trainer", "optim").get("lr")
    if not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise AuditFailure("training learning rate must be positive and finite")
    orchestrator = _table(payload, "orchestrator")
    if orchestrator.get("batch_size") != 8 or orchestrator.get("group_size") != 8:
        raise AuditFailure("role GRPO requires one complete eight-rollout comparison group")
    if orchestrator.get("max_inflight_episodes") != 8:
        raise AuditFailure(
            "role GRPO must keep the complete eight-rollout group logically in flight"
        )
    if orchestrator.get("max_off_policy_steps") != 0:
        raise AuditFailure("the first role GRPO step must be strictly on-policy")
    algo = _table(payload, "orchestrator", "algo")
    if algo.get("type") != "grpo" or algo.get("sampled_session_scope") != expected_scope:
        raise AuditFailure("top-level GRPO role scope mismatch")
    renderer = _table(payload, "orchestrator", "renderer")
    expected_thinking = role == "child"
    if renderer.get("name") != "qwen3.5" or renderer.get("enable_thinking") is not expected_thinking:
        raise AuditFailure(
            "child GRPO must preserve reasoning, while the early coordinator scaffold disables it"
        )
    sources = _table(payload, "orchestrator", "train").get("source")
    if not isinstance(sources, list) or len(sources) != 1 or not isinstance(sources[0], dict):
        raise AuditFailure("role GRPO requires exactly one training source")
    source = sources[0]
    if source.get("group_size") != 8:
        raise AuditFailure("source group size must remain eight")
    if source.get("env", {}).get("max_concurrent_agents") != 2:
        raise AuditFailure(
            "Prime Agent environment concurrency must remain capped at two"
        )
    expected_env_server_max_concurrent = 1 if role == "coordinator" else 2
    if source.get("serve", {}).get("max_concurrent") != expected_env_server_max_concurrent:
        raise AuditFailure(
            "Prime Agent EnvServer episode concurrency does not match the role memory budget"
        )
    source_algo = source.get("algo")
    if not isinstance(source_algo, dict) or source_algo.get("type") != "grpo" or source_algo.get(
        "sampled_session_scope"
    ) != expected_scope:
        raise AuditFailure("source GRPO role scope mismatch")
    taskset = source.get("env", {}).get("taskset")
    if not isinstance(taskset, dict):
        raise AuditFailure("missing procedural taskset")
    if taskset.get("id") != "procedural-harness-master-v1" or taskset.get("split") != "train_gen":
        raise AuditFailure("role GRPO must use the procedural train generator")
    if taskset.get("curriculum_rung") != "natural_n1a" or "families" in taskset:
        raise AuditFailure(
            "first role GRPO step must select the reachable natural_n1a curriculum rung"
        )
    if taskset.get("private_payload_mode") != "finding_card":
        raise AuditFailure("role GRPO task evidence must match the finding-card bootstrap")
    agent = source.get("env", {}).get("agent")
    harness = agent.get("harness") if isinstance(agent, dict) else None
    sampling = _table(payload, "orchestrator", "train", "sampling")
    early_rung = phase == "e0_full_actions"
    expected_budgets = {
        "max_completion_tokens": 1024 if early_rung else 2048,
        "max_turns": 8 if early_rung else 16,
        "max_output_tokens": 8192 if early_rung else 16384,
        "max_total_tokens": 32768 if early_rung else 65536,
        "autonomous_max_tokens": 32768 if early_rung else 65536,
    }
    if (
        not isinstance(agent, dict)
        or not isinstance(harness, dict)
        or sampling.get("max_completion_tokens") != expected_budgets["max_completion_tokens"]
        or agent.get("max_turns") != expected_budgets["max_turns"]
        or agent.get("max_output_tokens") != expected_budgets["max_output_tokens"]
        or agent.get("max_total_tokens") != expected_budgets["max_total_tokens"]
        or harness.get("autonomous_max_turns") != expected_budgets["max_turns"]
        or harness.get("autonomous_max_tokens") != expected_budgets["autonomous_max_tokens"]
    ):
        raise AuditFailure("role GRPO episode budgets do not match the requested curriculum phase")
    if taskset.get("start_index") != start_index or taskset.get("count") != task_count:
        raise AuditFailure("resolved task bank mismatch")
    if taskset.get("privileged_bootstrap_path") != str(bootstrap_path):
        raise AuditFailure("resolved bootstrap artifact mismatch")
    bootstrap = json.loads(bootstrap_path.read_text()) if bootstrap_path.is_file() else None
    if not isinstance(bootstrap, dict):
        raise AuditFailure(f"bootstrap artifact is missing or invalid: {bootstrap_path}")
    expected_leak_level = "solution_replay" if phase == "e0_full_actions" else "action_scaffold"
    expected_axis = [{"name": "natural_n1a", "start_index": start_index}]
    if (
        bootstrap.get("schema_version") != "qwen35-2b-environment-bootstrap-context/v1"
        or bootstrap.get("status") != "complete"
        or bootstrap.get("split") != "train_gen"
        or bootstrap.get("master_seed") != taskset.get("master_seed")
        or bootstrap.get("private_payload_mode") != taskset.get("private_payload_mode")
        or bootstrap.get("leak_level") != expected_leak_level
        or bootstrap.get("tasks_per_axis") != task_count
        or bootstrap.get("axes") != expected_axis
    ):
        raise AuditFailure("bootstrap generator coordinates do not match the resolved task bank")
    contexts = bootstrap.get("contexts")
    if not isinstance(contexts, dict) or len(contexts) != task_count:
        raise AuditFailure("bootstrap context cardinality does not match the resolved task bank")
    task = taskset.get("task")
    # A frozen child that fails to report makes every coordinator trajectory
    # zero-advantage. Keep its private-context send action scaffolded for both
    # role updates; the coordinator never sees the hidden value directly.
    expected_child_action_leak = True
    if (
        not isinstance(task, dict)
        or task.get("reward_mode") != "event_control"
        or task.get("leak_child_exact_action") is not expected_child_action_leak
    ):
        raise AuditFailure("role GRPO must reward partial causal protocol progress")
    zero_advantage = next(
        (item for item in orchestrator.get("pre_batch_filters", []) if item.get("type") == "zero_advantage"),
        None,
    )
    if zero_advantage is None or zero_advantage.get("enforce") is not True:
        raise AuditFailure("zero-advantage groups must never reach the optimizer")
    deployment = _table(payload, "deployment")
    if deployment.get("num_train_gpus") != 1 or deployment.get("num_infer_gpus") != 1:
        raise AuditFailure("role GRPO must fit the available one-trainer/one-inference topology")
    inference = _table(payload, "inference")
    role_router = _table(payload, "inference", "router")
    if (
        role_router.get("type") != "role-router"
        or role_router.get("policy_role") != role
        or role_router.get("anchor_model") != str(anchor_model_path)
    ):
        raise AuditFailure("role router does not pair the requested policy and frozen counterpart")
    # Once exact child delivery makes complete coordinator trajectories reliable,
    # tighten the root side: the disclosed spawn remains in the task prompt, but
    # the trainable coordinator must now emit its native IPython call on-policy.
    # Child updates still freeze and synthesize the coordinator anchor so their
    # non-root learning signal remains reachable.
    expected_coordinator_leak = role == "child"
    if role_router.get("leak_coordinator_exact_action") is not expected_coordinator_leak:
        raise AuditFailure(
            "role GRPO coordinator-spawn sampling does not match the tapered curriculum"
        )
    if role_router.get("leak_child_exact_action") is not True:
        raise AuditFailure(
            "early role GRPO must synthesize the private child send action"
        )
    expected_child_strip = role == "child"
    if role_router.get("strip_child_tool_choice") is not expected_child_strip:
        raise AuditFailure(
            "child GRPO must sample without the broken named tool-choice grammar, "
            "while coordinator GRPO must preserve the frozen child request"
        )
    expected_coordinator_strip = False
    if (
        role_router.get("strip_coordinator_tool_choice")
        is not expected_coordinator_strip
    ):
        raise AuditFailure(
            "the early coordinator curriculum must preserve its first named IPython action, "
            "and child GRPO must preserve the leaked frozen coordinator request"
        )
    policy_vllm = _table(payload, "inference", "vllm")
    policy_memory = policy_vllm.get("gpu_memory_utilization")
    anchor_memory = role_router.get("anchor_gpu_memory_utilization")
    if (
        not isinstance(policy_memory, (int, float))
        or not isinstance(anchor_memory, (int, float))
        or policy_memory <= 0
        or anchor_memory <= 0
        or policy_memory + anchor_memory > 0.8
    ):
        raise AuditFailure("same-GPU role engines must stay within the audited memory budget")
    expected_kv_cache_bytes = 4 * 1024**3
    if (
        policy_vllm.get("kv_cache_memory_bytes") != expected_kv_cache_bytes
        or role_router.get("anchor_kv_cache_memory_bytes") != expected_kv_cache_bytes
    ):
        raise AuditFailure(
            "same-GPU role engines require explicit four-GiB KV caches; percentage-only sizing is unsafe"
        )
    if (
        inference.get("backend_port") != 8100
        or role_router.get("anchor_backend_port") != 8200
        or role_router.get("anchor_data_parallel_rpc_port") == policy_vllm.get("data_parallel_rpc_port")
    ):
        raise AuditFailure("role-router engine ports must be distinct and stable")
    return {
        "role": role,
        "scope": expected_scope,
        "group_size": 8,
        "max_inflight_episodes": 8,
        "max_concurrent_agents": 2,
        "env_server_max_concurrent": expected_env_server_max_concurrent,
        "reward_mode": "event_control",
        "full_dense": True,
        "coordinator_action_leak": expected_coordinator_leak,
        "child_action_leak": expected_child_action_leak,
        "child_action_sampling": "synthetic_exact_send",
        "first_action_sampling": (
            "prompted_native_spawn" if role == "coordinator" else "masked_frozen_anchor"
        ),
        "child_tool_choice_stripped": expected_child_strip,
        "coordinator_tool_choice_stripped": expected_coordinator_strip,
        "enable_thinking": expected_thinking,
        "bootstrap_leak_level": expected_leak_level,
        "early_rung_bounded": early_rung,
        "promotion_minimum": 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--role", required=True, choices=("coordinator", "child"))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--anchor-model-path", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--bootstrap-path", required=True, type=Path)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--start-index", required=True, type=int)
    parser.add_argument("--task-count", required=True, type=int)
    args = parser.parse_args()
    print(
        audit(
            args.config,
            role=args.role,
            model_path=args.model_path,
            anchor_model_path=args.anchor_model_path,
            run_name=args.run_name,
            bootstrap_path=args.bootstrap_path,
            phase=args.phase,
            start_index=args.start_index,
            task_count=args.task_count,
        )
    )


if __name__ == "__main__":
    main()
