"""Validate the one-step Qwen3.5 27B mixed SDPO/GRPO mechanism audit."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from subagent_communication_v1.taskset import keep_first_coordinator_tool_call
from verifiers.v1.types import UserMessage, content_text

from prime_rl.orchestrator.trajectories import iter_trainable_branches

FEEDBACK_SCHEMA = "prime-agent/ownership-decision-feedback/v1"
DEFAULT_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
DIAGNOSTIC_ENV = "ownership-child-diagnostic-sdpo"
RETENTION_ENVS = {
    "ownership-coordinator-retention",
    "communication-direct-retention",
    "communication-single-retention",
    "communication-parallel-retention",
    "communication-causal-retention",
}
EXPECTED_ENVS = {DIAGNOSTIC_ENV, *RETENTION_ENVS}
EXPECTED_RATIOS = {
    DIAGNOSTIC_ENV: 4.0,
    "ownership-coordinator-retention": 2.0,
    "communication-direct-retention": 2.0,
    "communication-single-retention": 2.0,
    "communication-parallel-retention": 1.0,
    "communication-causal-retention": 4.0,
}
EXPECTED_BATCH_SIZE = 16
TRAINING_SEQ_LEN = 8192
MAX_COMPLETION_TOKENS = 1024


class AuditFailure(ValueError):
    """The completed run does not prove the intended mixed mechanism."""


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
    if not trainer.get("enable_token_export"):
        raise AuditFailure("resolved trainer must enable token export")
    if orchestrator.get("batch_size") != EXPECTED_BATCH_SIZE:
        raise AuditFailure(f"resolved batch size must be {EXPECTED_BATCH_SIZE}")
    if trainer.get("model", {}).get("seq_len") != TRAINING_SEQ_LEN:
        raise AuditFailure(f"resolved trainer sequence length must be {TRAINING_SEQ_LEN}")
    if orchestrator.get("seq_len") != TRAINING_SEQ_LEN:
        raise AuditFailure(f"resolved orchestrator sequence length must be {TRAINING_SEQ_LEN}")
    if trainer.get("ckpt") is not None:
        raise AuditFailure("resolved trainer unexpectedly enables checkpointing")
    if orchestrator.get("ckpt") is not None:
        raise AuditFailure("resolved orchestrator unexpectedly enables checkpointing")
    if orchestrator.get("train", {}).get("sampling", {}).get("reasoning_effort") != "high":
        raise AuditFailure("resolved train sampling does not use high reasoning effort")
    if (
        orchestrator.get("train", {}).get("sampling", {}).get("max_completion_tokens")
        != MAX_COMPLETION_TOKENS
    ):
        raise AuditFailure(
            f"resolved train sampling must cap each completion at {MAX_COMPLETION_TOKENS} tokens"
        )
    pre_filters = orchestrator.get("pre_batch_filters")
    token_window = next(
        (item for item in pre_filters or [] if item.get("type") == "trainable_token_window"),
        None,
    )
    if (
        token_window is None
        or token_window.get("enforce") is not True
        or token_window.get("max_tokens") != TRAINING_SEQ_LEN
    ):
        raise AuditFailure(
            "resolved audit must enforce a trainable-token window equal to the trainer sequence length"
        )
    post_filters = orchestrator.get("post_batch_filters")
    zero_advantage = next(
        (item for item in post_filters or [] if item.get("type") == "zero_advantage"),
        None,
    )
    if zero_advantage is None or zero_advantage.get("enforce") is not False:
        raise AuditFailure("resolved audit must retain zero-advantage groups")

    sources = orchestrator.get("train", {}).get("source")
    if not isinstance(sources, list):
        raise AuditFailure("resolved orchestrator has no training sources")
    by_name = {source.get("name"): source for source in sources}
    if set(by_name) != EXPECTED_ENVS:
        raise AuditFailure(f"resolved training sources do not match the mixed audit: {sorted(by_name)}")
    for name, source in by_name.items():
        expected_algo = "sdpo" if name == DIAGNOSTIC_ENV else "grpo"
        expected_group = 1 if name == DIAGNOSTIC_ENV else 2
        if source.get("algo", {}).get("type") != expected_algo:
            raise AuditFailure(f"{name} must use {expected_algo}")
        if source.get("group_size") != expected_group:
            raise AuditFailure(f"{name} must use group_size={expected_group}")
        if source.get("ratio") != EXPECTED_RATIOS[name]:
            raise AuditFailure(f"{name} must use ratio={EXPECTED_RATIOS[name]:g}")
    diagnostic = by_name[DIAGNOSTIC_ENV]
    algo = diagnostic["algo"]
    taskset = diagnostic.get("env", {}).get("taskset", {})
    if (
        not algo.get("require_explicit_feedback")
        or algo.get("required_feedback_contract_schema") != FEEDBACK_SCHEMA
        or algo.get("filter", {}).get("import_path")
        != "subagent_communication_v1.taskset.keep_first_coordinator_tool_call"
        or taskset.get("ownership") != "child"
        or not taskset.get("record_causal_feedback")
    ):
        raise AuditFailure("diagnostic SDPO source does not pin the typed causal contract")

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

    exported_counts = _exported_component_token_counts(run_dir)

    def component_tokens(name: str) -> float:
        key = f"loss_tokens/{name}"
        if any(key in record for record in records):
            return _require_finite(records, key)
        return float(exported_counts[name])

    rl_tokens = component_tokens("rl")
    sdpo_tokens = component_tokens("sdpo")
    if rl_tokens <= 0 or sdpo_tokens <= 0:
        raise AuditFailure(
            f"RL and SDPO token mass must both be positive, found {rl_tokens:g}/{sdpo_tokens:g}"
        )
    for name in ("ce", "ref_kl"):
        value = component_tokens(name)
        if value != 0:
            raise AuditFailure(f"expected loss_tokens/{name}=0, found {value:g}")

    _require_all(records, "optim/lr", 0.0)
    _require_all(records, "optim/update_succeeded", 1.0)
    grad_norm = _require_finite(records, "optim/grad_norm")
    if grad_norm <= 0:
        raise AuditFailure(f"gradient norm must be positive, found {grad_norm:g}")
    loss = _require_finite(records, "loss/mean")
    sdpo_loss = _require_finite(records, "sdpo/mean")
    _require_all(records, "time/save_ckpt", 0.0)
    aggregate_trainable_fraction = _require_finite(
        records, "train/agg/effective/agent/is_trainable/mean"
    )
    if not 0 < aggregate_trainable_fraction <= 1:
        raise AuditFailure(
            "aggregate trainable fraction must be in (0, 1], "
            f"found {aggregate_trainable_fraction:g}"
        )
    _require_all(records, "train/agg/effective/agent/is_filtered/mean", 0.0)

    rollouts = _require_finite(records, "progress/rollouts")
    tasks = _require_finite(records, "progress/tasks")
    if rollouts < EXPECTED_BATCH_SIZE or tasks < len(EXPECTED_ENVS):
        raise AuditFailure(
            f"expected at least {EXPECTED_BATCH_SIZE} rollouts and {len(EXPECTED_ENVS)} tasks, "
            f"found {rollouts:g}/{tasks:g}"
        )
    return {
        "rl_tokens": rl_tokens,
        "sdpo_tokens": sdpo_tokens,
        "grad_norm": grad_norm,
        "loss": loss,
        "sdpo_loss": sdpo_loss,
        "aggregate_trainable_fraction": aggregate_trainable_fraction,
        "rollouts": rollouts,
        "tasks": tasks,
    }


def _trace_env(trace: dict[str, Any], index: int) -> str:
    info = trace.get("info")
    if not isinstance(info, dict) or not isinstance(info.get("env_name"), str):
        raise AuditFailure(f"effective trace {index} has no env_name")
    return info["env_name"]


def _validate_traces(run_dir: Path) -> tuple[dict[str, Any], list[vf.WireTrace]]:
    path = run_dir / "rollouts" / "step_1" / "train" / "effective" / "traces.jsonl"
    records = _read_jsonl(path)
    if len(records) < EXPECTED_BATCH_SIZE:
        raise AuditFailure(f"expected at least {EXPECTED_BATCH_SIZE} effective traces, found {len(records)}")

    counts: Counter[str] = Counter()
    codes: Counter[str] = Counter()
    resource_families: Counter[str] = Counter()
    phrasing_variants: Counter[int] = Counter()
    causal_families: Counter[str] = Counter()
    typed: list[vf.WireTrace] = []
    for index, record in enumerate(records):
        if record.get("run", {}).get("type") != "train" or record.get("run", {}).get("step") != 1:
            raise AuditFailure(f"effective trace {index} is not from train step 1")
        if record.get("ok") is not True or record.get("errors"):
            raise AuditFailure(f"effective trace {index} is not a valid rollout")
        if record.get("agent", {}).get("trainable") is not True:
            raise AuditFailure(f"effective trace {index} is not trainable")
        env_name = _trace_env(record, index)
        if env_name not in EXPECTED_ENVS:
            raise AuditFailure(f"effective trace {index} came from unexpected environment {env_name!r}")
        counts[env_name] += 1
        info = record["info"]
        contract = info.get("feedback_contract")
        task_data = record.get("task", {}).get("data", {})
        if env_name == DIAGNOSTIC_ENV:
            feedback = info.get("feedback")
            if not isinstance(feedback, str) or not feedback.strip() or not isinstance(contract, dict):
                raise AuditFailure(f"diagnostic trace {index} has no explicit typed feedback")
            required = {
                "schema_version": FEEDBACK_SCHEMA,
                "answer_free": True,
                "retryable": True,
                "turn_index": 0,
                "ownership": "child",
            }
            if any(contract.get(key) != value for key, value in required.items()):
                raise AuditFailure(f"diagnostic trace {index} has an invalid feedback contract")
            if contract.get("message") not in (None, feedback):
                raise AuditFailure(f"diagnostic trace {index} feedback text disagrees with its contract")
            if record.get("metrics", {}).get("strict_success") != 0:
                raise AuditFailure(f"diagnostic trace {index} is not a diagnosed failure")
            code = contract.get("code")
            family = contract.get("family")
            phrasing = task_data.get("phrasing_variant")
            if not isinstance(code, str) or not code:
                raise AuditFailure(f"diagnostic trace {index} lacks a stable feedback code")
            if not isinstance(family, str) or family != task_data.get("resource_family"):
                raise AuditFailure(f"diagnostic trace {index} has inconsistent resource-family evidence")
            if not isinstance(phrasing, int):
                raise AuditFailure(f"diagnostic trace {index} lacks a phrasing variant")
            codes[code] += 1
            resource_families[family] += 1
            phrasing_variants[phrasing] += 1
        elif isinstance(contract, dict) and contract.get("schema_version") == FEEDBACK_SCHEMA:
            raise AuditFailure(f"typed ownership feedback leaked into retention trace {index}")

        family = task_data.get("family")
        if env_name == "communication-causal-retention" and isinstance(family, str):
            causal_families[family] += 1
        trace = vf.WireTrace.model_validate(record)
        branches = list(iter_trainable_branches(trace))
        active_masks = (
            keep_first_coordinator_tool_call(trace)
            if env_name == DIAGNOSTIC_ENV
            else [mask for _, mask in branches]
        )
        if len(active_masks) != len(branches):
            raise AuditFailure(f"effective trace {index} has misaligned training masks")
        for branch_index, ((branch, _), active_mask) in enumerate(
            zip(branches, active_masks, strict=True)
        ):
            if len(active_mask) != len(branch.token_ids):
                raise AuditFailure(
                    f"effective trace {index} branch {branch_index} has an invalid training mask"
                )
            if any(active_mask[TRAINING_SEQ_LEN:]):
                raise AuditFailure(
                    f"effective trace {index} branch {branch_index} has trainable tokens "
                    f"beyond the {TRAINING_SEQ_LEN}-token trainer window"
                )
        typed.append(trace)

    missing = EXPECTED_ENVS - counts.keys()
    if missing:
        raise AuditFailure(f"effective batch omitted training sources: {sorted(missing)}")
    if len(codes) < 2:
        raise AuditFailure(f"diagnostic source needs at least two feedback codes, found {dict(codes)}")
    if len(resource_families) < 2 or len(phrasing_variants) < 2:
        raise AuditFailure(
            "diagnostic source needs at least two resource families and two phrasing variants"
        )
    if set(causal_families) != {"followup", "handshake"}:
        raise AuditFailure(
            f"causal retention must include followup and handshake, found {dict(causal_families)}"
        )
    return (
        {
            "count": len(records),
            "sources": dict(counts),
            "feedback_codes": dict(codes),
            "resource_families": dict(resource_families),
            "phrasing_variants": dict(phrasing_variants),
            "causal_families": dict(causal_families),
        },
        typed,
    )


def _active_component(record: dict[str, Any], name: str) -> list[bool]:
    mask = record.get("loss_mask")
    weights = record.get(name)
    if not isinstance(mask, list) or not all(isinstance(value, bool) for value in mask):
        raise AuditFailure("token export has an invalid loss_mask")
    if not isinstance(weights, list) or len(weights) != len(mask):
        raise AuditFailure(f"token export has an invalid {name} stream")
    default = 1.0 if name == "rl_weights" else 0.0
    active = []
    for keep, weight in zip(mask, weights, strict=True):
        if weight is not None and not isinstance(weight, (int, float)):
            raise AuditFailure(f"token export has a nonnumeric {name} value")
        active.append(keep and float(default if weight is None else weight) != 0.0)
    return active


def _exported_component_token_counts(run_dir: Path) -> dict[str, int]:
    export_dir = run_dir / "token_exports" / "step_1"
    if not (export_dir / "STABLE").is_file():
        raise AuditFailure(f"token export is not stable: {export_dir}")
    counts = {name: 0 for name in ("rl", "ce", "ref_kl", "sdpo")}
    records = []
    for path in sorted(export_dir.glob("rank_*.jsonl")):
        records.extend(_read_jsonl(path))
    if not records:
        raise AuditFailure("token export contains no records")
    for record in records:
        for name in counts:
            counts[name] += sum(_active_component(record, f"{name}_weights"))
    return counts


def _is_child_branch(branch: vf.Branch) -> bool:
    return any(
        isinstance(node.message, UserMessage)
        and content_text(node.message.content).lstrip().startswith("[task from parent]")
        for node in branch.nodes
    )


def _validate_token_routing(run_dir: Path, traces: list[vf.WireTrace]) -> dict[str, int]:
    export_dir = run_dir / "token_exports" / "step_1"
    if not (export_dir / "STABLE").is_file():
        raise AuditFailure(f"token export is not stable: {export_dir}")
    export_records = []
    for path in sorted(export_dir.glob("rank_*.jsonl")):
        export_records.extend(_read_jsonl(path))
    if not export_records:
        raise AuditFailure("token export contains no records")

    by_sample: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(export_records):
        if record.get("schema_version") != 1 or record.get("step") != 1:
            raise AuditFailure(f"token export record {index} has the wrong schema or step")
        env_name = record.get("env_name")
        token_ids = record.get("token_ids")
        if env_name not in EXPECTED_ENVS or not isinstance(token_ids, list):
            raise AuditFailure(f"token export record {index} has invalid sample identity")
        by_sample[(env_name, tuple(token_ids))].append(record)

    consumed = 0
    coordinator_sdpo_samples = 0
    child_sdpo_samples = 0
    retention_rl_samples = 0
    for trace in traces:
        env_name = trace.info["env_name"]
        branches = list(iter_trainable_branches(trace))
        expected_masks = keep_first_coordinator_tool_call(trace) if env_name == DIAGNOSTIC_ENV else None
        if expected_masks is not None and len(expected_masks) != len(branches):
            raise AuditFailure("diagnostic filter and trainable branches are misaligned")
        for branch_index, (branch, trainable_mask) in enumerate(branches):
            key = (env_name, tuple(branch.token_ids))
            candidates = by_sample.get(key)
            if not candidates:
                raise AuditFailure(
                    f"no token export matches {env_name} trace {trace.id} branch {branch_index}"
                )
            record = candidates.pop()
            consumed += 1
            if record.get("loss_mask") != trainable_mask:
                raise AuditFailure(f"token export changed the trainable mask for trace {trace.id}")
            rl_active = _active_component(record, "rl_weights")
            ce_active = _active_component(record, "ce_weights")
            ref_kl_active = _active_component(record, "ref_kl_weights")
            sdpo_active = _active_component(record, "sdpo_weights")
            if any(ce_active) or any(ref_kl_active):
                raise AuditFailure(f"CE/reference-KL leaked into {env_name}")
            if env_name == DIAGNOSTIC_ENV:
                expected = expected_masks[branch_index]
                if sdpo_active != expected:
                    raise AuditFailure(
                        f"SDPO mask is not the first serialized coordinator tool call in trace {trace.id}"
                    )
                if any(rl_active):
                    raise AuditFailure(f"RL leaked into diagnostic SDPO trace {trace.id}")
                if _is_child_branch(branch):
                    child_sdpo_samples += 1
                    if any(sdpo_active):
                        raise AuditFailure(f"child branch received SDPO weight in trace {trace.id}")
                else:
                    coordinator_sdpo_samples += 1
                    if not any(sdpo_active):
                        raise AuditFailure(f"diagnostic coordinator has no causal SDPO tokens in trace {trace.id}")
            else:
                if any(sdpo_active):
                    raise AuditFailure(f"SDPO leaked into GRPO retention source {env_name}")
                if not any(rl_active):
                    raise AuditFailure(f"GRPO retention sample has no RL token mass in {env_name}")
                retention_rl_samples += 1

    leftovers = sum(len(records) for records in by_sample.values())
    if leftovers or consumed != len(export_records):
        raise AuditFailure(
            f"token exports and effective trace branches are not one-to-one: "
            f"consumed={consumed}, exported={len(export_records)}, leftover={leftovers}"
        )
    if coordinator_sdpo_samples == 0 or child_sdpo_samples == 0 or retention_rl_samples == 0:
        raise AuditFailure("token routing audit did not exercise coordinator, child, and retention branches")
    return {
        "export_records": len(export_records),
        "coordinator_sdpo_samples": coordinator_sdpo_samples,
        "child_zero_sdpo_samples": child_sdpo_samples,
        "retention_rl_samples": retention_rl_samples,
    }


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
    trace_report, traces = _validate_traces(run_dir)
    token_routing = _validate_token_routing(run_dir, traces)
    _validate_no_model_artifacts(run_dir)
    return {
        "verdict": "pass",
        "mechanism": "mixed-feedback-conditioned-sdpo-grpo-zero-lr",
        "expected_revision": expected_revision,
        "resolved_revisions": revisions,
        "metrics": metrics,
        "traces": trace_report,
        "token_routing": token_routing,
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
        raise SystemExit(f"zero-LR SDPO audit failed: {error}") from error
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
