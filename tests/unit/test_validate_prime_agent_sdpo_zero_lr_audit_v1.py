import importlib.util
import json
from itertools import cycle
from pathlib import Path

import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolCall, UserMessage

SCRIPT = Path(__file__).parents[2] / "scripts" / "validate_prime_agent_sdpo_zero_lr_audit_v1.py"
SPEC = importlib.util.spec_from_file_location("validate_prime_agent_sdpo_zero_lr_audit_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _source(name: str, algo: str, group_size: int) -> dict:
    source = {
        "name": name,
        "group_size": group_size,
        "ratio": MODULE.EXPECTED_RATIOS[name],
        "algo": {"type": algo},
        "env": {"taskset": {}},
    }
    if name == MODULE.DIAGNOSTIC_ENV:
        source["algo"].update(
            require_explicit_feedback=True,
            required_feedback_contract_schema=MODULE.FEEDBACK_SCHEMA,
            filter={"import_path": "subagent_communication_v1.taskset.keep_first_coordinator_tool_call"},
        )
        source["env"]["taskset"] = {
            "ownership": "child",
            "record_causal_feedback": True,
        }
    return source


def _trace(
    index: int,
    env_name: str,
    *,
    code: str = "required_delegation_missing",
    resource_family: str = "json_sum",
    phrasing_variant: int = 0,
    family: str | None = None,
) -> vf.Trace:
    base = 1000 + index * 20
    task_data = vf.WireTaskData(
        idx=index,
        prompt=f"task {index}",
        resource_family=resource_family,
        phrasing_variant=phrasing_variant,
        family=family,
    )
    nodes = [
        MessageNode(
            parent=None,
            message=UserMessage(content=f"task {index}"),
            token_ids=[base],
            mask=[False],
        ),
        MessageNode(
            parent=0,
            sampled=True,
            message=AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"call-{index}",
                        name="ipython",
                        arguments='{"code":"decision"}',
                    )
                ],
            ),
            token_ids=[base + 1, 248058, base + 2, 248059, base + 3],
            mask=[True] * 5,
            logprobs=[-0.1] * 5,
        ),
    ]
    if env_name == MODULE.DIAGNOSTIC_ENV:
        nodes.extend(
            [
                MessageNode(
                    parent=None,
                    message=UserMessage(content="[task from parent] child work"),
                    token_ids=[base + 4],
                    mask=[False],
                ),
                MessageNode(
                    parent=2,
                    sampled=True,
                    message=AssistantMessage(content="child result"),
                    token_ids=[base + 5, base + 6],
                    mask=[True, True],
                    logprobs=[-0.2, -0.2],
                ),
                MessageNode(
                    parent=None,
                    message=UserMessage(content=f"task {index} continuation"),
                    token_ids=[base + 7],
                    mask=[False],
                ),
                MessageNode(
                    parent=4,
                    sampled=True,
                    message=AssistantMessage(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id=f"continuation-{index}",
                                name="ipython",
                                arguments='{"code":"consume child reply"}',
                            )
                        ],
                    ),
                    token_ids=[base + 8, 248058, base + 9, 248059, base + 10],
                    mask=[True] * 5,
                    logprobs=[-0.3] * 5,
                ),
            ]
        )
    feedback = "Repair the observed ownership decision."
    info = {"env_name": env_name}
    metrics = {}
    if env_name == MODULE.DIAGNOSTIC_ENV:
        info.update(
            feedback=feedback,
            feedback_contract={
                "schema_version": MODULE.FEEDBACK_SCHEMA,
                "code": code,
                "category": "routing",
                "family": resource_family,
                "ownership": "child",
                "turn_index": 0,
                "answer_free": True,
                "retryable": True,
                "message": feedback,
            },
        )
        metrics["strict_success"] = 0
    return vf.Trace(
        run=vf.TrainRunInfo(id="run", step=1),
        task=vf.TraceTask(type="Task", data=task_data),
        agent=vf.AgentInfo(config=vf.AgentConfig(), trainable=True),
        nodes=nodes,
        info=info,
        metrics=metrics,
        is_completed=True,
        ok=True,
        stop_condition="agent_completed",
    )


def _export_records(trace: vf.Trace) -> list[dict]:
    env_name = trace.info["env_name"]
    branches = list(MODULE.iter_trainable_branches(trace))
    expected = MODULE.keep_first_coordinator_tool_call(trace) if env_name == MODULE.DIAGNOSTIC_ENV else None
    records = []
    for branch_index, (branch, mask) in enumerate(branches):
        length = len(branch.token_ids)
        records.append(
            {
                "schema_version": 1,
                "step": 1,
                "env_name": env_name,
                "token_ids": branch.token_ids,
                "loss_mask": mask,
                "rl_weights": ([0.0] * length if env_name == MODULE.DIAGNOSTIC_ENV else [1.0] * length),
                "ce_weights": [0.0] * length,
                "ref_kl_weights": [0.0] * length,
                "sdpo_weights": (
                    [float(value) for value in expected[branch_index]] if expected is not None else [0.0] * length
                ),
            }
        )
    return records


def _make_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "zero-lr-audit"
    revision = MODULE.DEFAULT_REVISION
    snapshot = f"/cache/Qwen3.5-27B/snapshots/{revision}"
    sources = [
        _source(MODULE.DIAGNOSTIC_ENV, "sdpo", 1),
        *[_source(name, "grpo", 2) for name in sorted(MODULE.RETENTION_ENVS)],
    ]
    _write_json(
        run_dir / "configs" / "trainer.json",
        {
            "max_steps": 1,
            "model": {"name": snapshot, "seq_len": MODULE.TRAINING_SEQ_LEN},
            "optim": {"lr": 0},
            "ckpt": None,
            "enable_token_export": True,
        },
    )
    _write_json(
        run_dir / "configs" / "orchestrator.json",
        {
            "max_steps": 1,
            "batch_size": MODULE.EXPECTED_BATCH_SIZE,
            "seq_len": MODULE.TRAINING_SEQ_LEN,
            "model": {"name": snapshot},
            "ckpt": None,
            "pre_batch_filters": [
                {
                    "type": "trainable_token_window",
                    "enforce": True,
                    "max_tokens": MODULE.TRAINING_SEQ_LEN,
                }
            ],
            "post_batch_filters": [{"type": "zero_advantage", "enforce": False}],
            "train": {
                "sampling": {
                    "reasoning_effort": "high",
                    "max_completion_tokens": MODULE.MAX_COMPLETION_TOKENS,
                },
                "source": sources,
            },
        },
    )
    _write_json(run_dir / "configs" / "inference.json", {"vllm": {"model": snapshot}})
    metrics = [
        {
            "step": 1,
            "progress/rollouts": MODULE.EXPECTED_BATCH_SIZE,
            "progress/tasks": 16,
            "time/save_ckpt": 0,
            "train/agg/effective/agent/is_trainable/mean": 1,
            "train/agg/effective/agent/is_filtered/mean": 0,
        },
        {
            "step": 1,
            "loss_tokens/rl": 256,
            "loss_tokens/ce": 0,
            "loss_tokens/ref_kl": 0,
            "loss_tokens/sdpo": 128,
        },
        {
            "step": 1,
            "optim/lr": 0,
            "optim/update_succeeded": 1,
            "optim/grad_norm": 0.25,
        },
        {"step": 1, "loss/mean": 0.1, "sdpo/mean": 0.2},
        {"step": 1, "time/save_ckpt": 0},
    ]
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("".join(json.dumps(record) + "\n" for record in metrics))

    trace_specs = [
        (MODULE.DIAGNOSTIC_ENV, "required_delegation_missing", "json_sum", 0, None),
        (MODULE.DIAGNOSTIC_ENV, "child_handle_not_retained", "csv_amount_total", 1, None),
        (MODULE.DIAGNOSTIC_ENV, "required_delegation_missing", "text_keyword_count", 0, None),
        (MODULE.DIAGNOSTIC_ENV, "child_handle_not_retained", "markdown_heading_count", 1, None),
    ]
    retention = cycle(
        [
            ("ownership-coordinator-retention", None),
            ("communication-direct-retention", "direct"),
            ("communication-single-retention", "single"),
            ("communication-parallel-retention", "parallel"),
            ("communication-causal-retention", "followup"),
            ("communication-causal-retention", "handshake"),
        ]
    )
    while len(trace_specs) < MODULE.EXPECTED_BATCH_SIZE:
        env_name, family = next(retention)
        trace_specs.append((env_name, "", "json_sum", 0, family))

    traces = [
        _trace(
            index,
            env_name,
            code=code,
            resource_family=resource_family,
            phrasing_variant=phrasing,
            family=family,
        )
        for index, (env_name, code, resource_family, phrasing, family) in enumerate(trace_specs)
    ]
    trace_path = run_dir / "rollouts" / "step_1" / "train" / "effective" / "traces.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("".join(json.dumps(trace.to_record()) + "\n" for trace in traces))

    export_dir = run_dir / "token_exports" / "step_1"
    export_dir.mkdir(parents=True)
    records = [record for trace in traces for record in _export_records(trace)]
    (export_dir / "rank_0.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    (export_dir / "STABLE").touch()
    return run_dir


def test_validator_accepts_complete_mixed_zero_lr_mechanism_audit(tmp_path: Path) -> None:
    report = MODULE.validate(_make_run(tmp_path))

    assert report["verdict"] == "pass"
    assert report["metrics"]["rl_tokens"] == 256
    assert report["metrics"]["sdpo_tokens"] == 128
    assert report["metrics"]["aggregate_trainable_fraction"] == 1
    assert report["traces"]["count"] == MODULE.EXPECTED_BATCH_SIZE
    assert report["token_routing"]["coordinator_sdpo_samples"] == 4
    assert report["token_routing"]["coordinator_zero_sdpo_continuations"] == 4
    assert report["token_routing"]["child_zero_sdpo_samples"] == 4
    assert report["model_artifacts_written"] is False


def test_validator_reconstructs_component_counts_from_stable_exports(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    metrics_path = run_dir / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    for record in records:
        for key in tuple(record):
            if key.startswith("loss_tokens/"):
                del record[key]
    metrics_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    report = MODULE.validate(run_dir)

    assert report["metrics"]["rl_tokens"] > 0
    assert report["metrics"]["sdpo_tokens"] > 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nonzero_lr", "learning rate is not zero"),
        ("drops_zero_advantage", "must retain zero-advantage groups"),
        ("no_token_window", "has no pre-batch filters"),
        ("late_token_window", "must first enforce a trainable-token window"),
        ("overlong_trainable", "has trainable tokens beyond"),
        ("no_sdpo_tokens", "RL and SDPO token mass must both be positive"),
        ("no_rl_tokens", "RL and SDPO token mass must both be positive"),
        ("competing_loss", "expected loss_tokens/ce=0"),
        ("no_aggregate_trainable", "aggregate trainable fraction must be in"),
        ("zero_gradient", "gradient norm must be positive"),
        ("bad_contract", "invalid feedback contract"),
        ("sdpo_leak", "SDPO leaked into GRPO retention source"),
        ("child_sdpo", "SDPO mask is not the first serialized coordinator tool call"),
        ("checkpoint", "forbidden model artifacts"),
    ],
)
def test_validator_fails_closed(tmp_path: Path, mutation: str, message: str) -> None:
    run_dir = _make_run(tmp_path)
    if mutation == "nonzero_lr":
        path = run_dir / "configs" / "trainer.json"
        config = json.loads(path.read_text())
        config["optim"]["lr"] = 1e-7
        _write_json(path, config)
    elif mutation == "drops_zero_advantage":
        path = run_dir / "configs" / "orchestrator.json"
        config = json.loads(path.read_text())
        config["post_batch_filters"][0]["enforce"] = True
        _write_json(path, config)
    elif mutation == "no_token_window":
        path = run_dir / "configs" / "orchestrator.json"
        config = json.loads(path.read_text())
        config["pre_batch_filters"] = []
        _write_json(path, config)
    elif mutation == "late_token_window":
        path = run_dir / "configs" / "orchestrator.json"
        config = json.loads(path.read_text())
        config["pre_batch_filters"].insert(0, {"type": "gibberish", "enforce": False})
        _write_json(path, config)
    elif mutation == "overlong_trainable":
        trace_path = run_dir / "rollouts" / "step_1" / "train" / "effective" / "traces.jsonl"
        traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
        trace = next(item for item in traces if item["info"]["env_name"] in MODULE.RETENTION_ENVS)
        node = trace["nodes"][-1]
        extra = MODULE.TRAINING_SEQ_LEN + 1 - sum(len(item["token_ids"]) for item in trace["nodes"])
        node["token_ids"].extend([123] * extra)
        node["mask"].extend([True] * extra)
        node["logprobs"].extend([-0.1] * extra)
        trace_path.write_text("".join(json.dumps(item) + "\n" for item in traces))
    elif mutation in {
        "no_sdpo_tokens",
        "no_rl_tokens",
        "competing_loss",
        "zero_gradient",
        "no_aggregate_trainable",
    }:
        path = run_dir / "metrics.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        key, value = {
            "no_sdpo_tokens": ("loss_tokens/sdpo", 0),
            "no_rl_tokens": ("loss_tokens/rl", 0),
            "competing_loss": ("loss_tokens/ce", 1),
            "zero_gradient": ("optim/grad_norm", 0),
            "no_aggregate_trainable": ("train/agg/effective/agent/is_trainable/mean", 0),
        }[mutation]
        next(record for record in records if key in record)[key] = value
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
    elif mutation == "bad_contract":
        path = run_dir / "rollouts" / "step_1" / "train" / "effective" / "traces.jsonl"
        traces = [json.loads(line) for line in path.read_text().splitlines()]
        traces[0]["info"]["feedback_contract"]["answer_free"] = False
        path.write_text("".join(json.dumps(trace) + "\n" for trace in traces))
    elif mutation in {"sdpo_leak", "child_sdpo"}:
        path = run_dir / "token_exports" / "step_1" / "rank_0.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if mutation == "sdpo_leak":
            record = next(record for record in records if record["env_name"] in MODULE.RETENTION_ENVS)
        else:
            diagnostic = [record for record in records if record["env_name"] == MODULE.DIAGNOSTIC_ENV]
            record = diagnostic[1]
        position = next(index for index, keep in enumerate(record["loss_mask"]) if keep)
        record["sdpo_weights"][position] = 1.0
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
    else:
        artifact = run_dir / "weights" / "step_1" / "model.safetensors"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"weights")

    with pytest.raises(MODULE.AuditFailure, match=message):
        MODULE.validate(run_dir)
