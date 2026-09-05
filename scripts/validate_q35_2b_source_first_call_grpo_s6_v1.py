#!/usr/bin/env python3
"""Fail-closed validation for the S6 first-call GRPO audit and update configs."""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

S5_PATH = Path(
    "/home/ubuntu/rlm/outputs/q35-2b-source-worker-remedial-s5-v1/"
    "h-source-s5-remedial-step8-v1/weights/step_8"
)
E33_PATH = Path(
    "/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/"
    "c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2"
)
EXPECTED_FAMILIES = {"specialist_source_ast", "specialist_source_config"}
EXPECTED_REWARD = "source_worker_first_call"
EXPECTED_BATCH_SIZE = 16
EXPECTED_GROUP_SIZE = 8
EXPECTED_TRAIN_SEED = 20270909
EXPECTED_TRAIN_OFFSET = 70000


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
    ):
        raise AuditFailure("S6 requires two complete on-policy groups")
    if _table(payload, "orchestrator", "algo") != {
        "type": "grpo",
        "sampled_session_scope": "non_root",
    }:
        raise AuditFailure("S6 top-level policy scope must be non-root GRPO")
    expected_enforce = stage == "update"
    for slot in ("pre_batch_filters", "post_batch_filters"):
        if _zero_advantage_filter(payload, slot).get("enforce") is not expected_enforce:
            raise AuditFailure(
                f"{stage} zero-variance policy is wrong in {slot}"
            )

    sampling = _table(payload, "orchestrator", "train", "sampling")
    if (
        sampling.get("temperature") != 1.0
        or sampling.get("reasoning_effort") != "high"
        or sampling.get("max_completion_tokens") != 2048
        or sampling.get("extra_body", {}).get("return_token_ids") is not True
    ):
        raise AuditFailure("S6 sampling is not the exploratory token-visible contract")
    sources = _table(payload, "orchestrator", "train").get("source")
    if not isinstance(sources, list) or len(sources) != 1:
        raise AuditFailure("S6 requires exactly one learning source")
    source = sources[0]
    if source.get("group_size") != EXPECTED_GROUP_SIZE or source.get("algo") != {
        "type": "grpo",
        "sampled_session_scope": "non_root",
    }:
        raise AuditFailure("S6 source must preserve eight-way non-root GRPO groups")
    taskset = source.get("env", {}).get("taskset", {})
    if (
        taskset.get("id") != "source-worker-first-call-v1"
        or taskset.get("split") != "train"
        or set(taskset.get("families", [])) != EXPECTED_FAMILIES
        or taskset.get("instances_per_template") != 8
        or taskset.get("instance_offset") != EXPECTED_TRAIN_OFFSET
        or taskset.get("seed") != EXPECTED_TRAIN_SEED
        or taskset.get("teacher_conditioned", False)
        or taskset.get("ownership_guided", False)
        or taskset.get("task", {}).get("reward_mode") != EXPECTED_REWARD
    ):
        raise AuditFailure("S6 taskset is not the fresh train-only balanced reward taskset")

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
        keep and float(1.0 if weight is None else weight) != 0.0
        for keep, weight in zip(mask, weights, strict=True)
    )


def _validate_runtime_configs(
    run_dir: Path, stage: Literal["audit", "update"]
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
    checkpoint_enabled = isinstance(trainer.get("ckpt"), dict) and isinstance(
        orchestrator.get("ckpt"), dict
    )
    if checkpoint_enabled != (stage == "update"):
        raise AuditFailure(f"resolved S6 {stage} checkpoint policy is wrong")
    if trainer.get("loss", {}).get("kl_tau") != 0.001:
        raise AuditFailure("resolved S6 audit changed the rollout-policy KL control")
    if (
        orchestrator.get("batch_size") != EXPECTED_BATCH_SIZE
        or orchestrator.get("group_size") != EXPECTED_GROUP_SIZE
        or orchestrator.get("max_inflight_episodes") != EXPECTED_BATCH_SIZE
        or orchestrator.get("max_off_policy_steps") != 0
        or orchestrator.get("algo", {}).get("type") != "grpo"
        or orchestrator.get("algo", {}).get("sampled_session_scope") != "non_root"
    ):
        raise AuditFailure("resolved S6 audit changed its GRPO batch or role scope")
    for slot in ("pre_batch_filters", "post_batch_filters"):
        filters = orchestrator.get(slot)
        zero = next(
            (item for item in filters or [] if item.get("type") == "zero_advantage"),
            None,
        )
        if not isinstance(zero, dict) or zero.get("enforce") is not (
            stage == "update"
        ):
            raise AuditFailure(f"resolved {stage} zero-variance policy is wrong in {slot}")
    sources = orchestrator.get("train", {}).get("source")
    if not isinstance(sources, list) or len(sources) != 1:
        raise AuditFailure("resolved S6 audit does not have exactly one source")
    source = sources[0]
    taskset = source.get("env", {}).get("taskset", {})
    if (
        source.get("group_size") != EXPECTED_GROUP_SIZE
        or source.get("algo", {}).get("sampled_session_scope") != "non_root"
        or taskset.get("id") != "source-worker-first-call-v1"
        or taskset.get("split") != "train"
        or set(taskset.get("families", [])) != EXPECTED_FAMILIES
        or taskset.get("instance_offset") != EXPECTED_TRAIN_OFFSET
        or taskset.get("seed") != EXPECTED_TRAIN_SEED
        or taskset.get("task", {}).get("reward_mode") != EXPECTED_REWARD
    ):
        raise AuditFailure("resolved S6 taskset or source scope changed")
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
    return {
        "trainer_model": model_paths["trainer"],
        "anchor_model": router["anchor_model"],
        "policy_scope": "non_root",
        "checkpoint_enabled": checkpoint_enabled,
    }


def validate_runtime(
    run_dir: Path, stage: Literal["audit", "update"]
) -> dict[str, Any]:
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
        raise AuditFailure(
            f"S6 audit requires exactly {EXPECTED_BATCH_SIZE} effective traces"
        )
    task_rewards: dict[str, list[float]] = defaultdict(list)
    families: Counter[str] = Counter()
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
        families[family] += 1
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
    exports = [
        record
        for path in sorted(export_dir.glob("rank_*.jsonl"))
        for record in _read_jsonl(path)
    ]
    exports_by_tokens: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(exports):
        if record.get("schema_version") != 1 or record.get("step") != 1:
            raise AuditFailure(f"invalid S6 token-export identity at record {index}")
        token_ids = record.get("token_ids")
        if not isinstance(token_ids, list) or not all(
            isinstance(token_id, int) for token_id in token_ids
        ):
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
        raise AuditFailure(
            "S6 token exports are not one-to-one with trainable child branches"
        )
    if exported_rl_tokens <= 0:
        raise AuditFailure("S6 exported no positive RL token mass")
    if any(
        any(
            value not in (None, 0, 0.0)
            for value in (record.get(stream) or [])
        )
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
        "gradient_norm": grad_norm,
        "families": dict(families),
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
        report = (
            validate_runtime(args.target, args.stage)
            if args.runtime
            else validate_config(args.target, args.stage)
        )
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
