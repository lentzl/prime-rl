import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file, save_file


def _module(name: str):
    scripts = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_dual_policy_mastery_launcher_can_force_recursive_coordinator_return() -> None:
    launcher = (
        Path(__file__).parents[2] / "scripts" / "run_q35_2b_dual_policy_mastery_v1.sh"
    ).read_text()

    assert (
        "leak_coordinator_return_action=${DUAL_LEAK_COORDINATOR_RETURN_ACTION:-0}"
        in launcher
    )
    assert 'proxy_args+=(--leak-coordinator-return-action)' in launcher
    assert 'if [[ "$leak_coordinator_return_action" == 1 ]]' in launcher
    assert "typed_coordinator_return=${DUAL_TYPED_COORDINATOR_RETURN:-0}" in launcher
    assert 'proxy_args+=(--typed-coordinator-return)' in launcher
    assert (
        "leak_coordinator_exact_action=${DUAL_LEAK_COORDINATOR_EXACT_ACTION:-0}"
        in launcher
    )
    assert 'proxy_args+=(--leak-coordinator-exact-action)' in launcher
    assert "typed_child_report=${DUAL_TYPED_CHILD_REPORT:-0}" in launcher
    assert 'proxy_args+=(--typed-child-report)' in launcher
    assert (
        "leaf inline-evidence and typed child-report scaffolds are mutually exclusive"
        not in launcher
    )
    assert "leaf_inline_evidence=${DUAL_LEAF_INLINE_EVIDENCE:-0}" in launcher
    assert 'proxy_args+=(--leaf-inline-evidence)' in launcher
    assert (
        "leaf_compute_report_scaffold=${DUAL_LEAF_COMPUTE_REPORT_SCAFFOLD:-0}"
        in launcher
    )
    assert 'proxy_args+=(--leaf-compute-report-scaffold)' in launcher


def test_natural_child_replay_runner_keeps_role_and_promotion_gates_separate() -> None:
    root = Path(__file__).parents[2]
    runner = (root / "scripts" / "run_c160_natural_child_replay_v2.sh").read_text()
    config = (
        root
        / "experiments"
        / "qwen35-2b-recursive-coordinator-return-v1"
        / "c160-child-natural-compute-replay-v2.toml"
    ).read_text()

    assert "--natural-child-actions" in runner
    assert "--replay-anchor-corpus" in runner
    assert "accepted < 1" in runner
    assert ".traces[0].metrics.child_action_completed == 1" in runner
    assert ".traces[0].rewards.harness_score.score == 1" in runner
    assert "admission_floor: 4" in runner
    assert "DUAL_LEAK_COORDINATOR_EXACT_ACTION=1" not in runner
    assert "DUAL_LEAF_INLINE_EVIDENCE=1" in runner
    assert "PROCEDURAL_NATURAL_YIELD_SCAFFOLD=1" in runner
    assert "DUAL_TYPED_CHILD_REPORT" not in runner
    assert "lora" not in config.lower()
    assert "max_steps = 8" in config
    assert "lr = 3e-06" in config


def test_compute_report_curriculum_is_scaffolded_harvest_not_admission() -> None:
    runner = (
        Path(__file__).parents[2]
        / "scripts"
        / "run_c160_child_compute_report_curriculum_v1.sh"
    ).read_text()

    assert "DUAL_LEAF_COMPUTE_REPORT_SCAFFOLD=1" in runner
    assert "DUAL_LEAF_INLINE_EVIDENCE=1" in runner
    assert "PROCEDURAL_NATURAL_YIELD_SCAFFOLD=1" in runner
    assert "never admission evidence" in runner
    assert "child_action_completed == 1" in runner
    assert "harness_score.score == 1" in runner
    assert "stop_condition == \"user_closed\"" in runner
    assert "sft @" not in runner


def test_runtime_compute_update_trains_full_weights_then_uses_natural_gate() -> None:
    root = Path(__file__).parents[2]
    runner = (root / "scripts" / "run_c160_child_runtime_compute_v4.sh").read_text()
    config = (
        root
        / "experiments"
        / "qwen35-2b-recursive-coordinator-return-v1"
        / "c160-child-runtime-compute-v4.toml"
    ).read_text()

    assert "--scaffolded-compute-actions" in runner
    assert runner.count("--forced-return-traces") == 3
    assert "--minimum-return-traces 16" in runner
    assert '"log_error"' in runner
    assert ".resource_family_counts | keys | sort" in runner
    assert "expected_corpus_sha=" in runner
    assert "runtime compute corpus checksum mismatch" in runner
    assert "DUAL_LEAF_INLINE_EVIDENCE=1" in runner
    assert "DUAL_LEAF_COMPUTE_REPORT_SCAFFOLD" not in runner
    assert "admission_floor: 4" in runner
    assert "max_steps = 8" in config
    assert "lr = 0.000001" in config
    assert "lora" not in config.lower()
    assert "step_10" in config


def test_live_context_compute_update_excludes_answer_replay_and_keeps_gate() -> None:
    root = Path(__file__).parents[2]
    runner = (root / "scripts" / "run_c160_child_live_context_compute_v5.sh").read_text()
    config = (
        root
        / "experiments"
        / "qwen35-2b-recursive-coordinator-return-v1"
        / "c160-child-live-context-compute-v5.toml"
    ).read_text()

    assert "--leaf-reporter-contract" in runner
    assert "--replay-anchor-corpus" not in runner
    assert ".replay_anchor_rows == 0" in runner
    assert "LEAF_REPORTER_CONTRACT" in runner
    assert "audited {len(rows)} live-context, answer-free compute targets" in runner
    assert "DUAL_LEAF_REPORTER_CONTRACT=1" in runner
    assert "DUAL_LEAF_INLINE_EVIDENCE=1" in runner
    assert "DUAL_LEAF_COMPUTE_REPORT_SCAFFOLD" not in runner
    assert "admission_floor: 4" in runner
    assert "max_steps = 8" in config
    assert "lr = 0.000001" in config
    assert "lora" not in config.lower()
    assert "step_10" in config


def test_balanced_live_compute_update_uses_unique_data_and_protected_start() -> None:
    root = Path(__file__).parents[2]
    runner = (root / "scripts" / "run_c160_child_balanced_live_compute_v6.sh").read_text()
    config = (
        root
        / "experiments"
        / "qwen35-2b-recursive-coordinator-return-v1"
        / "c160-child-balanced-live-compute-v6.toml"
    ).read_text()

    assert "--examples-per-family 32" in runner
    assert ".row_count == 224" in runner
    assert ".unique_task_count == .row_count" in runner
    assert ".context_contract.replay_rows == 0" in runner
    assert "green" in runner and "retry" in runner and "stable" in runner
    assert "DUAL_LEAF_REPORTER_CONTRACT=1" in runner
    assert "DUAL_LEAF_INLINE_EVIDENCE=1" in runner
    assert "DUAL_LEAF_COMPUTE_REPORT_SCAFFOLD" not in runner
    assert "admission_floor: 4" in runner
    assert "max_steps = 8" in config
    assert "lr = 0.000001" in config
    assert "lora" not in config.lower()
    assert "grpo-auto-000160-child" in config
    assert "weights/step_4" in config


def _dense_candidate(tmp_path: Path, name: str, content: bytes) -> tuple[Path, str]:
    path = tmp_path / name
    path.mkdir()
    (path / "STABLE").touch()
    (path / "model.safetensors").write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def _initialize_controller(module, tmp_path: Path, *, trainable_roles: list[str] | None = None):
    coordinator, coordinator_sha = _dense_candidate(tmp_path, "coordinator-c0", b"c0")
    child, child_sha = _dense_candidate(tmp_path, "child-k0", b"k0")
    events = tmp_path / "events.jsonl"
    initialized = {
        "kind": "initialized",
        "recorded_at_utc": "2026-08-23T00:00:00Z",
        "model_revision": "revision",
        "candidates": {
            "coordinator": module._verified_dense_candidate(
                coordinator,
                coordinator_sha,
                label="C0",
                model="q35-coordinator-c0",
            ),
            "child": module._verified_dense_candidate(
                child,
                child_sha,
                label="K0",
                model="q35-child-k0",
            ),
        },
        "initial_targets": {
            "child": "e0c25_inline_evidence",
            "yield": "e0d2_capped_yield_exact_child",
        },
        "bank_policy": {
            "next_start_index": 100,
            "index_stride": 100,
            "tasks_per_bank": 4,
        },
        "invariants": {
            "minimum_complete_qualifying_trajectories_per_source": 4,
            "minimum_distinct_qualifying_task_keys_per_source": 4,
            "acceptance_floor_relaxed": False,
            "lora_updates_authorized": 0,
        },
    }
    if trainable_roles is not None:
        initialized["trainable_roles"] = trainable_roles
    module._append_event(
        events,
        initialized,
        create=True,
    )
    return events


def _record_passing_arm(
    module,
    events: Path,
    tmp_path: Path,
    track: str,
    *,
    qualifying: int = 4,
) -> None:
    status = module.project(module._load_events(events))
    arm = next(item for item in status["next_action"]["arms"] if item["track"] == track)
    artifact_dir = tmp_path / f"{track}-{arm['start_index']}"
    artifact_dir.mkdir()
    summary = artifact_dir / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": module.SUMMARY_SCHEMA_VERSION,
                "phase": arm["phase"],
                "episodes": 4,
                "qualifying_trajectories": qualifying,
                "distinct_qualifying_task_keys": qualifying,
                "gate": {
                    "required_qualifying_trajectories": 4,
                    "required_distinct_task_keys": 4,
                    "acceptance_floor_relaxed": False,
                    "gradient_gate_open": qualifying >= 4,
                },
            }
        )
    )
    versions = artifact_dir / "VERSIONS.txt"
    versions.write_text(
        "\n".join(
            [
                f"interaction_curriculum={arm['phase']}",
                *[
                    f"{role}_model_sha256={candidate['model_sha256']}"
                    for role, candidate in status["candidates"].items()
                ],
            ]
        )
        + "\n"
    )
    traces = artifact_dir / "traces.jsonl"
    traces.write_text("{}\n" * 4)
    bootstrap = artifact_dir / "bootstrap.json"
    bootstrap.write_text("{}\n")
    routing = artifact_dir / "ROUTING_AUDIT.jsonl"
    routing.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "qwen35-2b-dual-policy-route/v1",
                    "role": role,
                    "upstream_model": status["candidates"][role]["model_path"],
                    "status": 200,
                }
            )
            + "\n"
            for role in ("coordinator", "child")
        )
    )
    module._record_evaluation(
        argparse.Namespace(
            events=events,
            track=track,
            phase=arm["phase"],
            start_index=arm["start_index"],
            bank_id=f"bank-{track}-{arm['start_index']}",
            summary=summary,
            versions=versions,
            traces=traces,
            bootstrap=bootstrap,
            routing_audit=routing,
            recorded_at="2026-08-23T00:01:00Z",
        )
    )


def test_open_flow_training_admission_is_separate_from_four_row_promotion(tmp_path) -> None:
    module = _module("q35_2b_spade_dual_dense_loop_v1")
    events = _initialize_controller(module, tmp_path, trainable_roles=["coordinator"])
    module._set_admission_policy(
        argparse.Namespace(
            events=events,
            training_minimum=1,
            promotion_minimum=4,
            failed_trajectories_trainable=False,
            aggressive_frontier=False,
            reason="owner_authorized_open_flow",
            recorded_at="2026-08-24T00:00:00Z",
        )
    )

    _record_passing_arm(module, events, tmp_path, "yield", qualifying=1)
    status = module.project(module._load_events(events))

    assert status["admission_policy"] == {
        "minimum_training_qualifiers": 1,
        "minimum_promotion_qualifiers": 4,
        "failed_trajectory_rows_trainable": False,
        "aggressive_frontier": False,
    }
    assert status["status"] == "training_authorized"
    assert status["accepted_sources"]["yield"]["qualifying_trajectories"] == 1
    event = module._load_events(events)[-1]
    assert event["admission"]["gate_open"] is True
    assert event["admission"]["promotion_gate_open"] is False


def test_lora_merge_changes_only_the_targeted_dense_tensor(tmp_path) -> None:
    module = _module("merge_q35_2b_lora_into_dense_v1")
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    output = tmp_path / "merged"
    base.mkdir()
    adapter.mkdir()
    (base / "STABLE").touch()
    (base / "config.json").write_text('{"model_type":"test"}\n')
    base_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    untouched = torch.tensor([5.0], dtype=torch.bfloat16)
    save_file(
        {"model.layer.weight": base_weight, "model.untouched.weight": untouched},
        base / "model.safetensors",
    )
    factor_a = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
    factor_b = torch.tensor([[1.0], [3.0]], dtype=torch.float32)
    save_file(
        {
            "base_model.model.model.layer.lora_A.weight": factor_a,
            "base_model.model.model.layer.lora_B.weight": factor_b,
        },
        adapter / "adapter_model.safetensors",
    )
    (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 1, "lora_alpha": 2}))

    manifest = module.merge(base_model=base, adapter_dir=adapter, output_dir=output)

    merged = load_file(output / "model.safetensors")
    expected = base_weight + (factor_b @ factor_a * 2).to(torch.bfloat16)
    assert torch.equal(merged["model.layer.weight"], expected)
    assert torch.equal(merged["model.untouched.weight"], untouched)
    assert manifest["merged_target_count"] == 1
    assert manifest["output_model_sha256"] != manifest["base_model_sha256"]
    assert (output / "STABLE").is_file()
    assert (output / "config.json").is_file()


def test_proxy_routes_private_evidence_only_to_the_child_model() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    coordinator = {"messages": [{"role": "user", "content": "solve this"}], "model": "external"}
    child = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "[private evidence supplied to this reviewer] x"}],
            }
        ],
        "model": "external",
    }

    assert module.routed_payload(coordinator, coordinator_model="coordinator", child_model="child") == (
        "coordinator",
        {**coordinator, "model": "coordinator"},
    )
    assert module.routed_payload(child, coordinator_model="coordinator", child_model="child") == (
        "child",
        {**child, "model": "child"},
    )

    coordinator_tokens = {"token_ids": [10, 11, 12], "model": "external"}
    child_tokens = {"token_ids": [10, 91, 92, 12], "model": "external"}
    route_kwargs = {
        "coordinator_model": "coordinator",
        "child_model": "child",
        "private_evidence_token_ids": [91, 92],
    }
    assert module.routed_payload(coordinator_tokens, **route_kwargs) == (
        "coordinator",
        {**coordinator_tokens, "model": "coordinator"},
    )
    assert module.routed_payload(child_tokens, **route_kwargs) == (
        "child",
        {**child_tokens, "model": "child"},
    )


def test_proxy_routes_recursive_coordinator_context_to_coordinator_model() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    recursive = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "[recursive coordinator session contract]\n"
                    "session_role=coordinator\nis_root=false"
                ),
            }
        ],
        "model": "external",
    }

    assert module.routed_payload(
        recursive, coordinator_model="coordinator", child_model="child"
    ) == ("coordinator", {**recursive, "model": "coordinator"})

    conflicting = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "[recursive coordinator session contract] "
                    "[private evidence supplied to this reviewer]"
                ),
            }
        ]
    }
    with pytest.raises(ValueError, match="conflicting delegated-session"):
        module.request_role(conflicting)

    assert module.request_role(
        {"token_ids": [1, 71, 72, 2]},
        private_evidence_token_ids=[81, 82],
        recursive_coordinator_token_ids=[71, 72],
    ) == "coordinator"

    bounded_leaf = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "[recursive coordinator session contract]\n"
                    "is_root=false\n"
                    "can_delegate=false\n"
                    "can_finalize_user=false\n"
                    "return_contract=exactly_one_parent_report"
                ),
            }
        ],
        "model": "external",
    }
    assert module.routed_payload(
        bounded_leaf, coordinator_model="coordinator", child_model="child"
    ) == ("child", {**bounded_leaf, "model": "child"})

    bounded_private_leaf = {
        "messages": [
            {
                "role": "user",
                "content": (
                    bounded_leaf["messages"][0]["content"]
                    + "\n[private evidence supplied to this reviewer]"
                ),
            }
        ]
    }
    assert module.request_role(bounded_private_leaf) == "child"


def test_proxy_injects_root_coordinator_contract_only_at_depth_zero() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    root = {
        "messages": [
            {
                "role": "user",
                "content": "Prime Agent session\nRecursive agent depth: 0\nSolve the task.",
            }
        ],
        "model": "external",
    }

    assert module.is_root_coordinator_request(root) is True
    injected = module.with_root_coordinator_contract(root)
    assert injected["messages"][0] == {
        "role": "system",
        "content": module.ROOT_COORDINATOR_CONTRACT,
    }
    assert injected["messages"][1:] == root["messages"]
    assert "This identity persists" in injected["messages"][0]["content"]
    assert "delegation and resume" in injected["messages"][0]["content"]
    assert "never adopt the child's worker identity" in injected["messages"][0]["content"]
    assert "has_parent=false" in injected["messages"][0]["content"]
    assert "first report as the final child evidence" in injected["messages"][0]["content"]
    assert "Do not message the child, poll, or call another tool" in injected["messages"][0][
        "content"
    ]

    already_injected = module.with_root_coordinator_contract(injected)
    assert already_injected is injected

    child = {
        "messages": [
            {
                "role": "user",
                "content": "Prime Agent session\nRecursive agent depth: 1\n[task from parent]",
            }
        ]
    }
    private_child = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Recursive agent depth: 0\n"
                    "[private evidence supplied to this reviewer]"
                ),
            }
        ]
    }
    assert module.is_root_coordinator_request(child) is False
    assert module.is_root_coordinator_request(private_child) is False


def test_proxy_injects_idempotent_one_shot_leaf_reporter_contract() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    child = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Recursive agent depth: 1\n"
                    "[private evidence supplied to this reviewer]"
                ),
            }
        ],
        "model": "external",
    }

    injected = module.with_leaf_reporter_contract(child)
    assert injected["messages"][0] == {
        "role": "system",
        "content": module.LEAF_REPORTER_CONTRACT,
    }
    assert injected["messages"][1:] == child["messages"]
    contract = injected["messages"][0]["content"]
    assert "return_contract=exactly_one_parent_report" in contract
    assert "await agent_message.send(str(result), receiver_role='parent')" in contract
    assert "INLINE_EVIDENCE is already bound" in contract
    assert "A successful send completes your task" in contract
    assert "do not call another tool" in contract

    assert module.with_leaf_reporter_contract(injected) is injected


def test_inline_evidence_scaffold_removes_only_the_fake_leaf_path() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    path = "/workspace/harness-v1/train_gen/batch-1/review.md"
    payload = {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Evidence label: {path}\n"
                    "Required review: count level-2 Markdown headings\n"
                    "Evidence contents:\n## A\nbody"
                ),
            },
            {
                "role": "user",
                "content": f"Only this session owns {path}. Do not use open or Path.",
            },
        ]
    }

    rewritten = module.without_leaf_evidence_path(payload)

    prompt = json.dumps(rewritten["messages"])
    assert path not in prompt
    assert prompt.count(module.PATHLESS_INLINE_EVIDENCE_LABEL) == 2
    assert "Evidence contents:\\n## A\\nbody" in prompt
    assert payload["messages"][0]["content"].startswith(f"Evidence label: {path}")


def test_pathless_inline_evidence_rejects_conflicting_labels() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    payload = {
        "messages": [
            {"role": "user", "content": "Evidence label: /workspace/a"},
            {"role": "user", "content": "Evidence label: /workspace/b"},
        ]
    }

    with pytest.raises(ValueError, match="conflicting labels"):
        module.without_leaf_evidence_path(payload)


@pytest.mark.parametrize(
    ("operation", "required_fragment"),
    [
        ("sum the top-level JSON integer list", "sum(json.loads(INLINE_EVIDENCE))"),
        ("sum the CSV amount column", "csv.DictReader(io.StringIO(INLINE_EVIDENCE))"),
        ("count exact 'stable' tokens", "keyword = 'stable'"),
        ("count level-2 Markdown headings", "line.startswith('## ')"),
        ("count ERROR-level log lines", "line.startswith('ERROR ')"),
        (
            "count top-level sync and async function definitions",
            "ast.parse(INLINE_EVIDENCE)",
        ),
        ("return the largest JSON integer value", "json.loads(INLINE_EVIDENCE).values()"),
    ],
)
def test_leaf_compute_report_scaffold_is_answer_free_and_single_action(
    operation: str, required_fragment: str
) -> None:
    module = _module("dual_policy_openai_proxy_v1")
    messages = [
        {
            "role": "user",
            "content": (
                "[recursive coordinator session contract]\n"
                f"Required review: {operation}\n"
                "Evidence contents:\n[91, 4, 7]"
            ),
        }
    ]

    assert module.required_review_from_messages(messages) == operation
    code = module.leaf_compute_report_code(operation)
    assert required_fragment in code
    assert "91" not in code
    assert code.count("agent_message.send") == 1
    assert code.endswith("receiver_role='parent')")
    ast.parse(code)


def test_leaf_compute_report_scaffold_rejects_unknown_or_conflicting_operations() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    with pytest.raises(ValueError, match="unsupported"):
        module.leaf_compute_report_code("guess the answer")
    with pytest.raises(ValueError, match="conflicting"):
        module.required_review_from_messages(
            [
                {"content": "Required review: sum the CSV amount column"},
                {"content": "Required review: count ERROR-level log lines"},
            ]
        )


def test_proxy_normalizes_vllm_abort_finish_reason_for_json_and_split_sse() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    payload = {
        "id": "completion",
        "choices": [{"index": 0, "finish_reason": "abort"}],
    }
    body = json.dumps(payload).encode()

    normalized, rewrites = module.normalize_openai_finish_reason(body)
    assert rewrites == 1
    assert json.loads(normalized)["choices"][0]["finish_reason"] == "stop"

    event = b"event: message\n" + b"data: " + body
    normalized_event, rewrites = module.normalize_sse_event(event)
    assert rewrites == 1
    data_line = normalized_event.splitlines()[1]
    assert json.loads(data_line.removeprefix(b"data: "))["choices"][0][
        "finish_reason"
    ] == "stop"

    done, rewrites = module.normalize_sse_event(b"data: [DONE]")
    assert done == b"data: [DONE]"
    assert rewrites == 0


def test_dual_policy_launch_uses_checkpoint_tokenizer_for_logical_external_model() -> None:
    launcher = (
        Path(__file__).parents[2] / "scripts/dual_policy_openai_proxy_v1.py"
    ).read_text()
    assert "args.tokenizer_model or args.coordinator_model" in launcher


def test_child_grpo_proxy_leaks_only_disclosed_exact_coordinator_action() -> None:
    module = _module("dual_policy_openai_proxy_v1")

    disclosed = (
        "prefix [interaction-curriculum exact action] suffix\n"
        "In the root coordinator's first IPython call, execute this code exactly:\n\n"
        "```python\nreviewer = await rlm('review', name='relay-worker')\n```"
    )
    assert module.disclosed_root_action(disclosed) == (
        "reviewer = await rlm('review', name='relay-worker')"
    )
    assert module.disclosed_root_action(disclosed.replace(module.EXACT_ACTION_MARKER, "")) is None
    messages = [{"role": "user", "content": [{"type": "text", "text": disclosed}]}]
    assert module.disclosed_root_action_from_messages(messages) == (
        "reviewer = await rlm('review', name='relay-worker')"
    )
    markerless_messages = [
        {
            "role": "user",
            "content": disclosed.replace(f"{module.EXACT_ACTION_MARKER} suffix\n", ""),
        }
    ]
    assert module.disclosed_root_action_from_messages(markerless_messages) == (
        "reviewer = await rlm('review', name='relay-worker')"
    )


def test_child_grpo_proxy_encodes_executable_ipython_code_not_a_string_literal() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    code = "reviewer = await rlm('review', name='relay-worker')"
    prompt = (
        "prefix [interaction-curriculum exact action] suffix\n"
        "In the root coordinator's first IPython call, execute this code exactly:\n\n"
        f"```python\n{code}\n```"
    )

    class RecordingTokenizer:
        encoded: str | None = None

        def decode(self, token_ids, *, skip_special_tokens):
            assert token_ids == [1, 2, 3]
            assert skip_special_tokens is False
            return prompt

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            self.encoded = text
            return [10, 11]

        def convert_tokens_to_ids(self, token):
            assert token == "<|im_end|>"
            return 12

    tokenizer = RecordingTokenizer()
    result = module.exact_ipython_completion_ids(tokenizer, [1, 2, 3])

    assert result == ([10, 11, 12], hashlib.sha256(code.encode()).hexdigest())
    assert tokenizer.encoded == (
        "<tool_call><function=ipython><parameter=code>\n"
        f"{code}\n"
        "</parameter></function></tool_call>"
    )
    assert json.dumps(code) not in tokenizer.encoded


def test_coordinator_grpo_proxy_encodes_private_child_send_action() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    code = "await agent_message.send('17', receiver_role='parent')"
    prompt = (
        "prefix [private evidence supplied to this reviewer] suffix\n"
        "[training-only child action scaffold]\n"
        "In your first IPython call execute exactly:\n\n"
        f"```python\n{code}\n```"
    )

    class RecordingTokenizer:
        encoded: str | None = None

        def decode(self, token_ids, *, skip_special_tokens):
            assert token_ids == [1, 2, 3]
            assert skip_special_tokens is False
            return prompt

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            self.encoded = text
            return [20, 21]

        def convert_tokens_to_ids(self, token):
            assert token == "<|im_end|>"
            return 22

    tokenizer = RecordingTokenizer()
    result = module.exact_child_ipython_completion_ids(tokenizer, [1, 2, 3])

    assert result == ([20, 21, 22], hashlib.sha256(code.encode()).hexdigest())
    assert tokenizer.encoded == (
        "<tool_call><function=ipython><parameter=code>\n"
        f"{code}\n"
        "</parameter></function></tool_call>"
    )
    assert module.disclosed_child_action(prompt.replace(module.CHILD_ACTION_SCAFFOLD_HEADER, "")) is None


def test_proxy_forces_disclosed_recursive_return_in_chat_tool_schema() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    code = "await agent_message.send('17', receiver_role='parent')"
    payload = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "[training-only child action scaffold]\n"
                    "In your first IPython call execute exactly:\n\n"
                    f"```python\n{code}\n```"
                ),
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ipython",
                    "description": "Run Python",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }

    disclosed = module.disclosed_child_action_from_messages(payload["messages"])
    rewritten = module.force_ipython_code_schema(payload, disclosed)

    assert disclosed == code
    assert rewritten["tool_choice"] == {
        "type": "function",
        "function": {"name": "ipython"},
    }
    assert rewritten["parallel_tool_calls"] is False
    assert rewritten["tools"][0]["function"]["parameters"] == {
        "type": "object",
        "properties": {"code": {"type": "string", "enum": [code]}},
        "required": ["code"],
        "additionalProperties": False,
    }
    assert payload["tools"][0]["function"]["parameters"] == {"type": "object"}


def test_proxy_exposes_typed_parent_return_without_answer_or_routing_fields() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    payload = {
        "messages": [{"role": "user", "content": "compute from private evidence"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ipython",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }

    rewritten = module.force_typed_parent_return_schema(payload)

    assert rewritten["stream"] is False
    assert "stream_options" not in rewritten
    assert rewritten["tool_choice"]["function"]["name"] == "return_to_parent"
    assert rewritten["parallel_tool_calls"] is False
    assert len(rewritten["tools"]) == 1
    function = rewritten["tools"][0]["function"]
    assert function["name"] == "return_to_parent"
    assert function["parameters"] == {
        "type": "object",
        "properties": {
            "payload": {
                "type": "string",
                "description": "The result computed from this session's evidence.",
            }
        },
        "required": ["payload"],
        "additionalProperties": False,
    }
    serialized = json.dumps(rewritten)
    assert "receiver_role" not in serialized
    assert "agent_message.send" not in serialized
    assert "17" not in serialized


def test_proxy_forces_one_model_authored_ipython_compute_turn() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    payload = {
        "messages": [{"role": "user", "content": "Evidence contents: 2, 3, 5"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ipython",
                    "description": "Run Python",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {"name": "another_tool", "parameters": {}},
            },
        ],
    }

    rewritten = module.force_parent_return_compute_schema(payload)

    assert rewritten["stream"] is False
    assert "stream_options" not in rewritten
    assert rewritten["tool_choice"]["function"]["name"] == "ipython"
    assert rewritten["parallel_tool_calls"] is False
    assert len(rewritten["tools"]) == 1
    function = rewritten["tools"][0]["function"]
    assert function["name"] == "ipython"
    assert function["parameters"] == payload["tools"][0]["function"]["parameters"]
    assert "Do not message the parent" in function["description"]
    serialized = json.dumps(rewritten)
    assert "return_to_parent" not in serialized
    assert "receiver_role" not in serialized
    assert "agent_message.send" not in serialized


def test_proxy_forces_one_step_child_compute_report_without_answer() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    payload = {
        "messages": [{"role": "user", "content": "compute from inline evidence"}],
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ipython",
                    "description": "Run Python",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }

    rewritten = module.force_child_compute_report_schema(payload)

    assert rewritten["tool_choice"]["function"]["name"] == "ipython"
    description = rewritten["tools"][0]["function"]["description"]
    assert "End the cell with the computed result" in description
    assert "harness routes that value" in description
    assert "next turn" not in description
    assert "return_to_parent" not in json.dumps(rewritten)


def test_proxy_extracts_visible_recursive_inline_evidence() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    messages = [
        {
            "role": "developer",
            "content": (
                "[recursive coordinator session contract]\n"
                "Required review: sum values\n"
                "Evidence contents:\n{\"a\": 2, \"b\": 5}\n"
            ),
        },
        {"role": "user", "content": "[task from parent]\nCompute it."},
    ]

    assert module.inline_evidence_from_messages(messages) == '{"a": 2, "b": 5}'


@pytest.mark.parametrize(
    ("content", "failed"),
    [
        ("Out[1]: 7", False),
        ("Traceback (most recent call last)\nFileNotFoundError: missing", True),
        ("Cell In[1], line 1\nSyntaxError: invalid syntax", True),
    ],
)
def test_proxy_detects_failed_latest_ipython_result(content: str, failed: bool) -> None:
    module = _module("dual_policy_openai_proxy_v1")
    messages = [
        {"role": "assistant", "tool_calls": []},
        {"role": "tool", "name": "ipython", "content": content},
    ]

    assert module.latest_ipython_tool_failed(messages) is failed


def test_proxy_repairs_only_parse_blocking_literal_newlines_in_compute_code() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    broken_code = "values = [2, 3, 5]\\nsum(values)"
    upstream = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "ipython",
                                "arguments": json.dumps({"code": broken_code}),
                            }
                        }
                    ]
                }
            }
        ]
    }

    body, rewrites, action_sha256 = module.rewrite_ipython_literal_newlines_response(
        json.dumps(upstream).encode()
    )

    assert rewrites == 1
    repaired = json.loads(
        json.loads(body)["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ]
    )["code"]
    assert repaired == "values = [2, 3, 5]\nsum(values)"
    assert action_sha256 == hashlib.sha256(repaired.encode()).hexdigest()


def test_proxy_prebinds_inline_evidence_and_redirects_path_read() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    code = (
        "import json\nfrom pathlib import Path\n"
        "data = json.loads(Path('/workspace/missing.json').read_text())\n"
        "max(data.values())"
    )
    upstream = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "ipython",
                                "arguments": json.dumps({"code": code}),
                            }
                        }
                    ]
                }
            }
        ]
    }

    body, rewrites, _ = module.rewrite_ipython_literal_newlines_response(
        json.dumps(upstream).encode(), inline_evidence='{"a": 2, "b": 5}'
    )

    assert rewrites == 1
    grounded = json.loads(
        json.loads(body)["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ]
    )["code"]
    assert grounded.startswith('INLINE_EVIDENCE = "{\\"a\\": 2, \\"b\\": 5}"\n')
    assert "Path('/workspace/missing.json').read_text()" not in grounded
    assert "json.loads(INLINE_EVIDENCE)" in grounded
    compile(grounded, "<grounded-compute>", "exec")


def test_proxy_prebinds_leaf_evidence_without_removing_parent_send() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    code = (
        "result = sum(line.startswith('ERROR') for line in INLINE_EVIDENCE.splitlines())\n"
        "await agent_message.send(str(result), receiver_role='parent')"
    )
    upstream = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "ipython",
                                "arguments": json.dumps({"code": code}),
                            }
                        }
                    ]
                }
            }
        ]
    }

    body, rewrites, _ = module.rewrite_ipython_literal_newlines_response(
        json.dumps(upstream).encode(),
        inline_evidence="INFO one\nERROR two\nERROR three\n",
        preserve_parent_send=True,
    )

    assert rewrites == 1
    grounded = json.loads(
        json.loads(body)["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ]
    )["code"]
    assert grounded.startswith('INLINE_EVIDENCE = "INFO one\\nERROR two\\nERROR three\\n"\n')
    assert "await agent_message.send(str(result), receiver_role='parent')" in grounded
    compile(grounded, "<grounded-leaf>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def test_proxy_awaits_model_authored_bare_leaf_parent_send() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    code = (
        "result = sum(line.startswith('## ') for line in INLINE_EVIDENCE.splitlines())\n"
        "agent_message.send(str(result), receiver_role='parent')"
    )
    upstream = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "ipython",
                                "arguments": json.dumps({"code": code}),
                            }
                        }
                    ]
                }
            }
        ]
    }

    body, rewrites, _ = module.rewrite_ipython_literal_newlines_response(
        json.dumps(upstream).encode(),
        inline_evidence="# Report\n## A\n## B\n",
        preserve_parent_send=True,
    )

    assert rewrites == 1
    grounded = json.loads(
        json.loads(body)["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ]
    )["code"]
    assert "await agent_message.send(str(result), receiver_role='parent')" in grounded
    assert "result = sum(" in grounded
    compile(grounded, "<awaited-leaf>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def test_proxy_converts_compute_stage_parent_send_to_value_expression() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    code = "await agent_message.send(str(2 + 2), receiver_role='parent')"
    upstream = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "ipython",
                                "arguments": json.dumps({"code": code}),
                            }
                        }
                    ]
                }
            }
        ]
    }

    body, rewrites, _ = module.rewrite_ipython_literal_newlines_response(
        json.dumps(upstream).encode(), inline_evidence="unused evidence"
    )

    assert rewrites == 1
    grounded = json.loads(
        json.loads(body)["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ]
    )["code"]
    assert "agent_message.send" not in grounded
    assert "str(2 + 2)" in grounded
    compile(grounded, "<grounded-compute>", "exec")


def test_proxy_wraps_model_computed_final_value_in_first_parent_report() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    code = (
        "result = sum(line.startswith('## ') for line in "
        "INLINE_EVIDENCE.splitlines())\nprint(result)"
    )
    upstream = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "ipython",
                                "arguments": json.dumps({"code": code}),
                            }
                        }
                    ]
                }
            }
        ]
    }

    body, rewrites, action_sha256 = module.rewrite_ipython_literal_newlines_response(
        json.dumps(upstream).encode(),
        inline_evidence="# Report\n## A\n## B\n",
        report_final_value_to_parent=True,
    )

    assert rewrites == 1
    grounded = json.loads(
        json.loads(body)["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ]
    )["code"]
    assert grounded.count("agent_message.send") == 1
    assert "await agent_message.send(str(result), receiver_role='parent')" in grounded
    assert "print(" not in grounded
    assert action_sha256 == hashlib.sha256(grounded.encode()).hexdigest()
    compile(grounded, "<computed-report>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def test_proxy_truncates_tool_markup_after_complete_computed_value() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    code = (
        "result = INLINE_EVIDENCE.splitlines().count('ERROR')\n"
        "print(result)\n"
        "</parameter></function><|endoftext|><|im_start|>user"
    )
    upstream = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "ipython",
                                "arguments": json.dumps({"code": code}),
                            }
                        }
                    ]
                }
            }
        ]
    }

    body, rewrites, _ = module.rewrite_ipython_literal_newlines_response(
        json.dumps(upstream).encode(),
        inline_evidence="ERROR\nINFO\n",
        report_final_value_to_parent=True,
    )

    assert rewrites == 1
    grounded = json.loads(
        json.loads(body)["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ]
    )["code"]
    assert "</parameter>" not in grounded
    assert "<|endoftext|>" not in grounded
    assert "await agent_message.send(str(result), receiver_role='parent')" in grounded
    compile(grounded, "<truncated-report>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def test_proxy_preserves_valid_python_containing_literal_newline_escape() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    code = 'separator = "\\n"'
    upstream = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "ipython",
                                "arguments": json.dumps({"code": code}),
                            }
                        }
                    ]
                }
            }
        ]
    }
    original = json.dumps(upstream).encode()

    assert module.rewrite_ipython_literal_newlines_response(original) == (
        original,
        0,
        None,
    )


def test_proxy_translates_model_computed_typed_return_to_native_send() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    upstream = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "return_to_parent",
                                "arguments": json.dumps({"payload": "17"}),
                            },
                        }
                    ],
                },
            }
        ]
    }

    body, rewrites, action_sha256 = module.rewrite_typed_parent_return_response(
        json.dumps(upstream).encode()
    )

    assert rewrites == 1
    rewritten = json.loads(body)
    function = rewritten["choices"][0]["message"]["tool_calls"][0]["function"]
    assert function["name"] == "ipython"
    code = json.loads(function["arguments"])["code"]
    assert code == "await agent_message.send(\"17\", receiver_role='parent')"
    assert action_sha256 == hashlib.sha256(code.encode()).hexdigest()

    sse = module.chat_completion_to_sse(body)
    assert sse.endswith(b"\n\ndata: [DONE]\n\n")
    event = json.loads(sse.splitlines()[0].removeprefix(b"data: "))
    assert event["object"] == "chat.completion.chunk"
    streamed_function = event["choices"][0]["delta"]["tool_calls"][0]["function"]
    assert streamed_function["name"] == "ipython"
    assert json.loads(streamed_function["arguments"])["code"] == code


def test_proxy_synthesizes_terminal_stream_after_typed_return() -> None:
    module = _module("dual_policy_openai_proxy_v1")

    body = module.synthetic_chat_stop_response(model="external", sequence=9)
    response = json.loads(body)

    assert response["id"] == "typed-parent-return-stop-9"
    assert response["model"] == "external"
    assert response["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Return delivered."},
            "logprobs": None,
            "finish_reason": "stop",
        }
    ]
    sse = module.chat_completion_to_sse(body)
    assert sse.endswith(b"\n\ndata: [DONE]\n\n")

    waiting = json.loads(
        module.synthetic_chat_stop_response(
            model="external", sequence=10, content="Waiting for the child report."
        )
    )
    assert waiting["choices"][0]["message"]["content"] == (
        "Waiting for the child report."
    )


def test_proxy_recognizes_only_incomplete_root_gate_continuations() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    gate = {
        "messages": [
            {"role": "user", "content": "original task"},
            {
                "role": "user",
                "content": (
                    "Autonomous quality gate failed (attempt 1/3): gate exited 1.\n"
                    "completion gate: the end-to-end coordinator task is not complete."
                ),
            },
        ]
    }

    assert module.is_incomplete_root_wait_request(gate)
    assert not module.is_incomplete_root_wait_request(
        {
            "messages": gate["messages"]
            + [{"role": "user", "content": "[from child:worker]\n5"}]
        }
    )


def test_proxy_tracks_completed_typed_returns_for_terminal_guard() -> None:
    source = (
        Path(__file__).parents[2] / "scripts/dual_policy_openai_proxy_v1.py"
    ).read_text()

    assert "self.completed_typed_return_hashes" in source
    assert "self.typed_return_compute_hashes" in source
    assert "self.typed_return_compute_attempts" in source
    assert "latest_ipython_tool_failed" in source
    assert '"forwarded_typed_return_compute"' in source
    assert 'mode="typed_return_session_terminated"' in source
    assert "session_sha256 in self.completed_typed_return_hashes" in source
    assert "self.completed_typed_child_report_hashes" in source
    assert "self.typed_child_report_compute_hashes" in source
    assert "self.typed_child_report_compute_attempts" in source
    assert 'mode="typed_child_report_session_terminated"' in source
    assert '"forwarded_typed_child_report"' in source
    assert '"forwarded_typed_child_report_compute"' in source
    assert '"root_wait_child_report_pending"' in source
    assert '"root_wait_child_report_completed"' in source
    assert "session_sha256 in self.completed_typed_child_report_hashes" in source
    assert "self.completed_leaf_compute_report_hashes" in source
    assert 'mode="leaf_compute_report_session_terminated"' in source
    assert '"forwarded_leaf_compute_report"' in source


def test_typed_compute_phase_starts_once_retries_failures_and_then_returns() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    session = "session-sha"
    failed_messages = [
        {
            "role": "tool",
            "name": "ipython",
            "content": "Traceback (most recent call last)\nNameError: missing",
        }
    ]
    successful_messages = [
        {"role": "tool", "name": "ipython", "content": "93"}
    ]

    assert module.should_run_typed_compute(session, set(), {}, successful_messages)
    assert module.should_run_typed_compute(
        session, {session}, {session: 1}, failed_messages
    )
    assert module.should_run_typed_compute(
        session, {session}, {session: 2}, failed_messages
    )
    assert module.should_run_typed_compute(
        session,
        {session},
        {session: 1},
        [{"role": "tool", "tool_call_id": "call-1", "content": "SyntaxError: bad"}],
    )
    assert not module.should_run_typed_compute(
        session, {session}, {session: 3}, failed_messages
    )
    assert not module.should_run_typed_compute(
        session, {session}, {session: 1}, successful_messages
    )


@pytest.mark.parametrize(
    "arguments",
    [json.dumps({"payload": 17}), json.dumps({"answer": "17"}), "not-json"],
)
def test_proxy_rejects_malformed_typed_return_payloads(arguments: str) -> None:
    module = _module("dual_policy_openai_proxy_v1")
    upstream = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "return_to_parent",
                                "arguments": arguments,
                            }
                        }
                    ]
                }
            }
        ]
    }

    original = json.dumps(upstream).encode()
    assert module.rewrite_typed_parent_return_response(original) == (original, 0, None)


def test_synthetic_generate_response_has_cardinality_matched_finite_logprobs() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    payload = json.loads(module.synthetic_generate_response([10, 11, 12], sequence=7))

    assert payload["request_id"] == "role-router-leak-7"
    choice = payload["choices"][0]
    assert choice["token_ids"] == [10, 11, 12]
    assert [item["token"] for item in choice["logprobs"]["content"]] == [
        "token_id:10",
        "token_id:11",
        "token_id:12",
    ]


def test_child_direct_generation_strips_broken_named_tool_choice_constraint() -> None:
    module = _module("dual_policy_openai_proxy_v1")
    payload = {
        "model": "external",
        "token_ids": [1, 2, 3],
        "sampling_params": {
            "temperature": 1.0,
            "tool_choice": {"type": "function", "function": {"name": "ipython"}},
            "parallel_tool_calls": False,
        },
    }

    cleaned, removed = module.without_tool_choice_constraints(payload)

    assert removed == ("tool_choice", "parallel_tool_calls")
    assert cleaned["sampling_params"] == {"temperature": 1.0}
    assert payload["sampling_params"]["tool_choice"]["function"]["name"] == "ipython"
    assert module.should_strip_tool_choice(
        "child", strip_child=True, strip_coordinator=False
    )
    assert module.should_strip_tool_choice(
        "coordinator", strip_child=False, strip_coordinator=True
    )
    assert not module.should_strip_tool_choice(
        "child", strip_child=False, strip_coordinator=True
    )
    assert not module.should_strip_tool_choice(
        "coordinator", strip_child=True, strip_coordinator=False
    )


def test_proxy_source_makes_exact_coordinator_leak_one_shot_per_session() -> None:
    source = (
        Path(__file__).parents[2] / "scripts/dual_policy_openai_proxy_v1.py"
    ).read_text()

    assert "self.leaked_session_hashes" in source
    assert "session_sha256 in self.leaked_session_hashes[leak_scope]" in source
    assert '"coordinator_return": set()' in source
    assert 'request.headers.get("x-session-id")' in source
    assert "leak_rejected_missing_{leak_scope}_session" in source
    assert '"coordinator": set()' in source
    assert '"child": set()' in source


def test_qualification_driver_resolves_and_records_one_master_seed() -> None:
    launcher = (
        Path(__file__).parents[2]
        / "scripts/run_qwen38_27b_prime_harness_qualification_v1.sh"
    ).read_text()
    assert "master_seed=${QWEN38_QUALIFICATION_MASTER_SEED:-20260819}" in launcher
    assert "f\"master_seed = {sys.argv[15]}\"" in launcher
    assert "printf 'master_seed=%s\\n' \"$master_seed\"" in launcher


def test_two_role_gates_authorize_two_full_updates_and_no_lora(tmp_path) -> None:
    module = _module("q35_2b_spade_dual_dense_loop_v1")
    events = _initialize_controller(module, tmp_path)

    initial = module.project(module._load_events(events))
    assert initial["model_revision"] == "revision"
    assert {arm["role_model"] for arm in initial["next_action"]["arms"]} == {
        "coordinator",
        "child",
    }
    _record_passing_arm(module, events, tmp_path, "child")
    _record_passing_arm(module, events, tmp_path, "yield")

    authorized = module.project(module._load_events(events))
    assert authorized["status"] == "training_authorized"
    assert authorized["next_action"]["full_optimizer_steps_authorized"] == {
        "child": 1,
        "coordinator": 1,
    }
    assert authorized["next_action"]["lora_updates_authorized"] == 0
    assert authorized["next_action"]["failed_trajectory_rows_trainable"] is False

    yield_event = next(
        event
        for event in module._load_events(events)
        if event["kind"] == "evaluation_recorded" and event["track"] == "yield"
    )
    module._append_event(
        events,
        {
            "kind": "evaluation_invalidated",
            "recorded_at_utc": "2026-08-23T00:02:00Z",
            "bank_id": yield_event["bank"]["id"],
            "target_event_sha256": yield_event["event_sha256"],
            "reason": "provider_error",
            "provider_error_count": 1,
        },
    )
    reopened = module.project(module._load_events(events))
    assert reopened["status"] == "collecting"
    assert reopened["accepted_sources"].keys() == {"child"}
    assert reopened["next_action"]["arms"][0]["track"] == "yield"


def test_failed_floor_candidate_selects_previous_role_checkpoint_without_update(
    tmp_path,
) -> None:
    module = _module("q35_2b_spade_dual_dense_loop_v1")
    events = _initialize_controller(module, tmp_path)
    c1, c1_sha = _dense_candidate(tmp_path, "coordinator-c1", b"c1")
    k1, k1_sha = _dense_candidate(tmp_path, "child-k1", b"k1")
    module._append_event(
        events,
        {
            "kind": "update_pair_recorded",
            "recorded_at_utc": "2026-08-23T00:01:00Z",
            "training_sources": {
                "child": {"phase": "e0_full_actions"},
                "yield": {"phase": "e0d2_capped_yield_exact_child"},
            },
            "output_candidates": {
                "coordinator": module._verified_dense_candidate(c1, c1_sha, label="C1", model="q35-coordinator-c1"),
                "child": module._verified_dense_candidate(k1, k1_sha, label="K1", model="q35-child-k1"),
            },
        },
    )
    for start_index, phase in (
        (100, "e0c_natural_child"),
        (200, "e0_full_actions"),
    ):
        module._append_event(
            events,
            {
                "kind": "evaluation_recorded",
                "recorded_at_utc": "2026-08-23T00:02:00Z",
                "track": "child",
                "phase": phase,
                "bank": {"id": f"failed-{start_index}", "start_index": start_index},
                "admission": {
                    "qualifying_trajectories": 0,
                    "distinct_task_keys": 0,
                    "gate_open": False,
                },
                "artifacts": {"summary_sha256": f"summary-{start_index}"},
            },
        )

    rejected = module.project(module._load_events(events))
    assert rejected["status"] == "candidate_rejected"
    assert rejected["next_action"]["kind"] == "select_roles"
    assert rejected["next_action"]["full_optimizer_steps_authorized"] == {
        "child": 0,
        "coordinator": 0,
    }
    assert rejected["next_action"]["selected_candidates"]["child"]["label"] == "K0"
    assert rejected["next_action"]["selected_candidates"]["coordinator"]["label"] == "C1"

    module._record_selection(argparse.Namespace(events=events, recorded_at=None))
    selected = module.project(module._load_events(events))
    assert selected["candidates"]["child"]["label"] == "K0"
    assert selected["candidates"]["coordinator"]["label"] == "C1"
    assert selected["next_action"]["kind"] == "collect"
    assert module._previous_viable_candidate_pair(module._load_events(events), current=selected["candidates"]) is None


def test_candidate_is_rejected_when_it_fails_its_parent_retention_rung(tmp_path) -> None:
    module = _module("q35_2b_spade_dual_dense_loop_v1")
    events = _initialize_controller(module, tmp_path)
    c1, c1_sha = _dense_candidate(tmp_path, "coordinator-c1", b"c1")
    k1, k1_sha = _dense_candidate(tmp_path, "child-k1", b"k1")
    module._append_event(
        events,
        {
            "kind": "update_pair_recorded",
            "recorded_at_utc": "2026-08-23T00:01:00Z",
            "training_sources": {
                "child": {"phase": "e0c2_natural_child_no_template"},
                "yield": {"phase": "e0d2_capped_yield_exact_child"},
            },
            "output_candidates": {
                "coordinator": module._verified_dense_candidate(c1, c1_sha, label="C1", model="q35-coordinator-c1"),
                "child": module._verified_dense_candidate(k1, k1_sha, label="K1", model="q35-child-k1"),
            },
        },
    )
    for start_index, phase in (
        (100, "e0c25_inline_evidence"),
        (200, "e0c2_natural_child_no_template"),
    ):
        module._append_event(
            events,
            {
                "kind": "evaluation_recorded",
                "recorded_at_utc": "2026-08-23T00:02:00Z",
                "track": "child",
                "phase": phase,
                "bank": {"id": f"failed-{start_index}", "start_index": start_index},
                "admission": {
                    "qualifying_trajectories": 3,
                    "distinct_task_keys": 3,
                    "gate_open": False,
                },
                "artifacts": {"summary_sha256": f"summary-{start_index}"},
            },
        )

    rejected = module.project(module._load_events(events))
    assert rejected["status"] == "candidate_rejected"
    assert rejected["cycle_retention_floors"]["child"] == "e0c2_natural_child_no_template"
    evidence = rejected["next_action"]["rejection_evidence"]["child"]
    assert evidence["phase"] == "e0c2_natural_child_no_template"
    assert evidence["required_retention_phase"] == "e0c2_natural_child_no_template"
    assert rejected["next_action"]["selected_candidates"]["child"]["label"] == "K0"
    assert rejected["next_action"]["selected_candidates"]["coordinator"]["label"] == "C1"


def test_coordinator_only_cycle_freezes_child_and_rolls_back_only_coordinator(tmp_path) -> None:
    module = _module("q35_2b_spade_dual_dense_loop_v1")
    events = _initialize_controller(module, tmp_path, trainable_roles=["coordinator"])

    initial = module.project(module._load_events(events))
    assert initial["trainable_roles"] == ["coordinator"]
    assert [(arm["track"], arm["role_model"]) for arm in initial["next_action"]["arms"]] == [("yield", "coordinator")]
    _record_passing_arm(module, events, tmp_path, "yield")
    authorized = module.project(module._load_events(events))
    assert authorized["next_action"]["kind"] == "train_roles"
    assert authorized["next_action"]["full_optimizer_steps_authorized"] == {
        "child": 0,
        "coordinator": 1,
    }
    assert authorized["next_action"]["sources"].keys() == {"yield"}

    c1, c1_sha = _dense_candidate(tmp_path, "coordinator-c1", b"c1")
    updated = dict(authorized["candidates"])
    updated["coordinator"] = module._verified_dense_candidate(c1, c1_sha, label="C1", model="q35-coordinator-c1")
    module._append_event(
        events,
        {
            "kind": "update_roles_recorded",
            "recorded_at_utc": "2026-08-23T00:02:00Z",
            "training_sources": {"yield": {"phase": "e0d2_capped_yield_exact_child"}},
            "output_candidates": updated,
            "updated_roles": ["coordinator"],
        },
    )
    for start_index, phase in (
        (200, "e0d3_uncapped_yield_exact_child"),
        (300, "e0d2_capped_yield_exact_child"),
    ):
        module._append_event(
            events,
            {
                "kind": "evaluation_recorded",
                "recorded_at_utc": "2026-08-23T00:03:00Z",
                "track": "yield",
                "phase": phase,
                "bank": {"id": f"failed-{start_index}", "start_index": start_index},
                "admission": {
                    "qualifying_trajectories": 3,
                    "distinct_task_keys": 3,
                    "gate_open": False,
                },
                "artifacts": {"summary_sha256": f"summary-{start_index}"},
            },
        )

    rejected = module.project(module._load_events(events))
    assert rejected["status"] == "candidate_rejected"
    assert rejected["next_action"]["kind"] == "select_roles"
    assert rejected["next_action"]["rejected_roles"] == ["coordinator"]
    assert rejected["next_action"]["selected_candidates"]["coordinator"]["label"] == "C0"
    assert rejected["next_action"]["selected_candidates"]["child"]["label"] == "K0"


def test_stricter_pass_still_requires_exact_parent_retention_bank(tmp_path) -> None:
    module = _module("q35_2b_spade_dual_dense_loop_v1")
    events = _initialize_controller(module, tmp_path, trainable_roles=["coordinator"])
    _record_passing_arm(module, events, tmp_path, "yield")
    authorized = module.project(module._load_events(events))
    c1, c1_sha = _dense_candidate(tmp_path, "coordinator-c1", b"c1")
    updated = dict(authorized["candidates"])
    updated["coordinator"] = module._verified_dense_candidate(c1, c1_sha, label="C1", model="q35-coordinator-c1")
    module._append_event(
        events,
        {
            "kind": "update_roles_recorded",
            "recorded_at_utc": "2026-08-23T00:02:00Z",
            "training_sources": {"yield": {"phase": "e0d2_capped_yield_exact_child"}},
            "output_candidates": updated,
            "updated_roles": ["coordinator"],
        },
    )

    _record_passing_arm(module, events, tmp_path, "yield")
    retention = module.project(module._load_events(events))
    assert retention["accepted_sources"]["yield"]["phase"] == "e0d3_uncapped_yield_exact_child"
    assert retention["next_action"]["kind"] == "collect"
    assert len(retention["next_action"]["arms"]) == 1
    arm = retention["next_action"]["arms"][0]
    assert arm["track"] == "yield"
    assert arm["role_model"] == "coordinator"
    assert arm["phase"] == "e0d2_capped_yield_exact_child"
    assert arm["reason"] == "verify_exact_parent_retention_rung"
    assert arm["start_index"] not in {100, 200}
    assert arm["tasks"] == 4
    assert arm["optimizer_updates_during_collection"] == 0


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (
            {
                "mode": "paired_hint_regret",
                "trained_batch_ids": ["batch-1"],
                "selected_environment_ids": ["env-1"],
                "training_stage": "delayed_reward_filtered_coevolution",
            },
            "action_scaffold",
        ),
        (
            {
                "mode": "scaffolded_repair",
                "trained_batch_ids": ["batch-2"],
                "selected_environment_ids": [],
                "training_stage": "scaffolded_schema_and_safety_repair",
            },
            "action_scaffold",
        ),
    ],
)
def test_dense_runner_uses_base_leak_for_coevolution_designer_metadata(
    metadata: dict, expected: str
) -> None:
    module = _module("run_q35_2b_spade_dual_dense_autonomous_v1")
    candidates = {"coordinator": {"replay": {"environment_designer": metadata}}}
    arm = {"track": "yield", "reason": "evaluate_current_target"}

    assert module.AutonomousDualDenseRunner._designer_leak_level(candidates, arm) == expected


def test_dense_runner_rejects_malformed_coevolution_designer_metadata() -> None:
    module = _module("run_q35_2b_spade_dual_dense_autonomous_v1")
    candidates = {
        "coordinator": {
            "replay": {
                "environment_designer": {
                    "mode": "paired_hint_regret",
                    "trained_batch_ids": ["batch-1"],
                    "selected_environment_ids": ["env-1"],
                    "training_stage": "wrong-stage",
                }
            }
        }
    }

    with pytest.raises(ValueError, match="invalid coevolution Designer metadata"):
        module.AutonomousDualDenseRunner._designer_leak_level(
            candidates,
            {"track": "yield", "reason": "evaluate_current_target"},
        )


def test_dense_runner_config_has_no_adapter_and_skips_partial_attempts(tmp_path) -> None:
    module = _module("run_q35_2b_spade_dual_dense_autonomous_v1")
    config = module.RunnerConfig(
        repo_root=tmp_path,
        events=tmp_path / "events.jsonl",
        artifacts_root=tmp_path / "artifacts",
        results_root=tmp_path / "results",
        output_root=tmp_path / "outputs",
        experiment_dir=tmp_path / "experiment",
        journal=tmp_path / "journal.jsonl",
        stop_file=tmp_path / "STOP",
        uv_bin="uv",
        learning_rate=1e-6,
        max_evaluations=1,
        max_update_pairs=1,
        max_actions=1,
        open_ended=False,
        dry_run=False,
    )
    runner = module.AutonomousDualDenseRunner(config)
    rendered = runner._training_config(
        run_name="coordinator-c1",
        model_path=tmp_path / "coordinator-c0",
        replay=tmp_path / "replay",
        rows=4,
    )

    assert "max_steps = 1" in rendered
    assert 'optimization_dtype = "bfloat16"' in rendered
    assert 'reduce_dtype = "bfloat16"' in rendered
    assert "lr = 1e-06" in rendered
    assert "[model.lora]" not in rendered
    assert "batch_size = 12" in runner._training_config(
        run_name="coordinator-c1-designer",
        model_path=tmp_path / "coordinator-c0",
        replay=tmp_path / "replay",
        rows=12,
    )
    assert "batch_size = 16" in runner._training_config(
        run_name="coordinator-c1-odd-replay",
        model_path=tmp_path / "coordinator-c0",
        replay=tmp_path / "replay",
        rows=15,
    )
    partial = config.results_root / "bank"
    partial.mkdir(parents=True)
    (partial / module.TASK_AXIS).mkdir()
    (partial / module.TASK_AXIS / "traces.jsonl").write_text("{}\n")
    label, run, summarized = runner._evaluation_attempt("bank", 4)
    assert (label, run, summarized) == ("bank-attempt2", config.results_root / "bank-attempt2", False)

    replay = tmp_path / "replay-v2"
    replay.mkdir()
    (replay / "MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": "qwen35-2b-dual-role-replay/v2",
                "role": "coordinator",
                "new_rows": 4,
                "rows": 8,
            }
        )
    )
    assert (
        runner._build_replay(
            candidate={},
            role="coordinator",
            source_dir=tmp_path / "unused-source",
            output_dir=replay,
        )["schema_version"]
        == "qwen35-2b-dual-role-replay/v2"
    )

    mutable = tmp_path / "mutable-traces.jsonl"
    snapshot = tmp_path / "PARTIAL_TRACES.jsonl"
    mutable.write_text('{"episode":1}\n')
    assert runner._snapshot_abort_evidence(mutable, snapshot) == snapshot
    mutable.write_text('{"episode":1}\n{"episode":2}\n')
    assert snapshot.read_text() == '{"episode":1}\n'
    with pytest.raises(ValueError, match="refusing to replace"):
        runner._snapshot_abort_evidence(mutable, snapshot)


def test_summary_qualifier_counts_must_be_typed_integers(tmp_path) -> None:
    module = _module("q35_2b_spade_dual_dense_loop_v1")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": module.SUMMARY_SCHEMA_VERSION,
                "phase": "e0c25_inline_evidence",
                "qualifying_trajectories": None,
                "distinct_qualifying_task_keys": 4,
                "gate": {
                    "required_qualifying_trajectories": 4,
                    "required_distinct_task_keys": 4,
                    "acceptance_floor_relaxed": False,
                    "gradient_gate_open": False,
                },
            }
        )
    )

    with pytest.raises(ValueError, match="counts must be integers"):
        module._validated_summary(summary, phase="e0c25_inline_evidence")

    traces = tmp_path / "provider-errors.jsonl"
    traces.write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "calls": [
                            {"error": {"type": "ProviderError"}},
                            {"error": None},
                        ]
                    }
                ]
            }
        )
        + "\n"
    )
    assert module._provider_error_count(traces) == 1


def test_partial_failed_bank_can_only_descend_after_gate_is_mathematically_closed(
    tmp_path,
) -> None:
    module = _module("q35_2b_spade_dual_dense_loop_v1")
    events = _initialize_controller(module, tmp_path)
    _record_passing_arm(module, events, tmp_path, "child")
    status = module.project(module._load_events(events))
    arm = status["next_action"]["arms"][0]
    assert arm["track"] == "yield"

    artifact_dir = tmp_path / "partial-yield"
    artifact_dir.mkdir()
    traces = artifact_dir / "traces.jsonl"
    episodes = []
    for task_offset, score in enumerate((1, 1, 0)):
        episodes.append(
            {
                "ok": True,
                "errors": [],
                "traces": [
                    {
                        "ok": True,
                        "is_completed": True,
                        "errors": [],
                        "calls": [],
                        "task": {
                            "data": {
                                "idx": task_offset,
                                "name": f"train_gen-{arm['start_index'] + task_offset:08d}-task",
                            }
                        },
                        "rewards": {"harness_score": {"score": score, "weight": 1}},
                    }
                ],
            }
        )
    traces.write_text("".join(json.dumps(episode) + "\n" for episode in episodes))
    versions = artifact_dir / "VERSIONS.txt"
    versions.write_text(f"interaction_curriculum={arm['phase']}\n")
    bootstrap = artifact_dir / "bootstrap.json"
    bootstrap.write_text("{}\n")
    routing = artifact_dir / "ROUTING_AUDIT.jsonl"
    routing.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "qwen35-2b-dual-policy-route/v1",
                    "role": role,
                    "upstream_model": status["candidates"][role]["model_path"],
                    "status": 200,
                }
            )
            + "\n"
            for role in ("coordinator", "child")
        )
    )
    namespace = argparse.Namespace(
        events=events,
        track=arm["track"],
        phase=arm["phase"],
        start_index=arm["start_index"],
        bank_id="partial-yield-bank",
        versions=versions,
        traces=traces,
        bootstrap=bootstrap,
        routing_audit=routing,
        reason="evaluation_process_exceeded_memory_limit",
        recorded_at="2026-08-23T00:03:00Z",
    )
    module._abort_evaluation(namespace)

    aborted = module._load_events(events)[-1]
    assert aborted["kind"] == "evaluation_aborted"
    assert aborted["bank"]["completed_tasks"] == 3
    assert aborted["admission"]["qualifying_trajectories"] == 2
    assert aborted["admission"]["maximum_possible_qualifying_trajectories"] == 3
    assert aborted["admission"]["gate_open"] is False
    projected = module.project(module._load_events(events))
    assert projected["accepted_sources"].keys() == {"child"}
    assert projected["next_action"]["arms"][0]["track"] == "yield"
    assert projected["next_action"]["arms"][0]["phase"] == "e0d2_capped_yield_exact_child"

    namespace.bank_id = "not-closed"
    namespace.start_index = projected["next_action"]["arms"][0]["start_index"]
    traces.write_text(
        "".join(
            json.dumps(
                {
                    "ok": True,
                    "errors": [],
                    "traces": [
                        {
                            "ok": True,
                            "is_completed": True,
                            "errors": [],
                            "calls": [],
                            "task": {
                                "data": {
                                    "idx": task_offset,
                                    "name": (f"train_gen-{namespace.start_index + task_offset:08d}-task"),
                                }
                            },
                            "rewards": {"harness_score": {"score": 1, "weight": 1}},
                        }
                    ],
                }
            )
            + "\n"
            for task_offset in range(2)
        )
    )
    with pytest.raises(ValueError, match="not mathematically closed"):
        module._abort_evaluation(namespace)


def test_role_replay_keeps_hardest_prior_anchors_and_all_new_rows(tmp_path) -> None:
    module = _module("combine_q35_2b_role_replay_sft_v1")

    def write_corpus(path: Path, rows: list[dict], *, replay: bool) -> None:
        path.mkdir()
        parquet = path / "train.parquet"
        Dataset.from_list(rows).to_parquet(str(parquet))
        manifest = {
            "schema_version": ("qwen35-2b-dual-role-replay/v1" if replay else module.SOURCE_SCHEMA_VERSION),
            "dataset": {"sha256": module.sha256_file(parquet)},
        }
        if replay:
            manifest["role"] = "child"
        else:
            manifest.update(
                {
                    "selected_roles": ["child"],
                    "rows_by_role": {"child": 4},
                    "student": {"dense_weight_mutated": True},
                }
            )
        (path / "MANIFEST.json").write_text(json.dumps(manifest))

    prior_rows = [
        {
            "messages": [],
            "tools": [],
            "axis": "natural_n1a",
            "phase": phase,
            "task_key": f"prior-{index}",
            "trace_id": f"prior-{index}",
            "role": "child",
        }
        for index, phase in enumerate(["e0c_natural_child"] * 4 + ["e0_full_actions"] * 4)
    ]
    new_rows = [
        {
            "messages": [],
            "tools": [],
            "axis": "natural_n1a",
            "phase": "e0_full_actions",
            "task_key": f"new-{index}",
            "trace_id": f"new-{index}",
            "role": "child",
        }
        for index in range(4)
    ]
    prior = tmp_path / "prior"
    new = tmp_path / "new"
    output = tmp_path / "output"
    write_corpus(prior, prior_rows, replay=True)
    write_corpus(new, new_rows, replay=False)

    manifest = module.combine(
        new_source=new,
        prior_replay=prior,
        output_dir=output,
        role="child",
        max_rows=8,
    )

    assert manifest["schema_version"] == "qwen35-2b-dual-role-replay/v2"
    assert manifest["trace_ids"] == [
        "prior-0",
        "prior-1",
        "prior-2",
        "prior-3",
        "new-0",
        "new-1",
        "new-2",
        "new-3",
    ]
    assert manifest["phase_counts"] == {
        "e0c_natural_child": 4,
        "e0_full_actions": 4,
    }


def test_coordinator_replay_keeps_new_interaction_and_designer_rows(tmp_path) -> None:
    module = _module("combine_q35_2b_role_replay_sft_v1")

    def write(path: Path, rows: list[dict], manifest: dict) -> None:
        path.mkdir()
        parquet = path / "train.parquet"
        Dataset.from_list(rows).to_parquet(str(parquet))
        manifest["dataset"] = {"sha256": module.sha256_file(parquet)}
        (path / "MANIFEST.json").write_text(json.dumps(manifest))

    def rows(prefix: str, count: int, *, objective: str | None = None) -> list[dict]:
        return [
            {
                "messages": [],
                "tools": "[]",
                "axis": "natural_n1a",
                "phase": "e0d2_capped_yield_exact_child",
                "task_key": f"{prefix}-{index}",
                "trace_id": f"{prefix}-{index}",
                "role": "coordinator",
                **({"objective": objective} if objective else {}),
            }
            for index in range(count)
        ]

    prior = tmp_path / "prior"
    source = tmp_path / "source"
    designer = tmp_path / "designer"
    output = tmp_path / "output"
    prior_rows = rows("prior", 8)
    prior_rows[-1]["phase"] = "spade:yield:e0d3_uncapped_yield"
    write(
        prior,
        prior_rows,
        {"schema_version": module.SCHEMA_VERSION, "role": "coordinator"},
    )
    write(
        source,
        rows("interaction", 4),
        {
            "schema_version": module.SOURCE_SCHEMA_VERSION,
            "selected_roles": ["coordinator"],
            "rows_by_role": {"coordinator": 4},
            "student": {"dense_weight_mutated": True},
        },
    )
    write(
        designer,
        rows("designer", 4, objective="environment_designer"),
        {
            "schema_version": module.DESIGNER_SOURCE_SCHEMA_VERSION,
            "role": "coordinator",
            "objective": "environment_designer",
            "rows": 4,
            "selection_count": 4,
            "acceptance_floor_relaxed": False,
            "exact_answer_rows": 0,
            "leak_level": "action_scaffold",
            "leak_stage_index": 0,
            "leak_ladder": [
                "action_scaffold",
                "child_contract_scaffold",
                "spawn_contract_scaffold",
                "ownership_scaffold",
                "strategy_hint",
            ],
        },
    )

    manifest = module.combine(
        new_source=source,
        prior_replay=prior,
        designer_source=designer,
        output_dir=output,
        role="coordinator",
        max_rows=12,
    )

    assert manifest["rows"] == 12
    assert manifest["new_rows"] == 4
    assert manifest["new_designer_rows"] == 4
    assert manifest["added_rows"] == 8
    assert manifest["environment_designer"]["next_stage_index"] == 1
    assert set(manifest["trace_ids"][-8:]) == {
        *(f"interaction-{index}" for index in range(4)),
        *(f"designer-{index}" for index in range(4)),
    }


def test_role_replay_ranks_cross_track_spade_designer_phase() -> None:
    module = _module("combine_q35_2b_role_replay_sft_v1")

    assert module._replay_phase_rank("yield", "spade:child:e0c29_evidence_available") == (
        6 / 7,
        0,
    )
    assert module._replay_phase_rank("yield", "spade-repair:child:e0c29_evidence_available") == (
        6 / 7,
        0,
    )


def test_role_replay_rejects_unknown_spade_track() -> None:
    module = _module("combine_q35_2b_role_replay_sft_v1")

    with pytest.raises(ValueError, match="invalid embedded track"):
        module._replay_phase_rank("yield", "spade:unknown:e0d3_uncapped_yield")
    with pytest.raises(ValueError, match="invalid namespace"):
        module._replay_phase_rank("yield", "unknown:yield:e0d3_uncapped_yield")


def test_spade_reward_and_hint_context_are_causal_and_answer_free() -> None:
    module = _module("q35_2b_spade_coevolution_v1")

    assert module.plateau_reward(0.0) == 0.0
    assert module.plateau_reward(0.4) == 1.0
    assert module.plateau_reward(0.5) == 1.0
    assert module.plateau_reward(0.6) == 1.0
    assert module.plateau_reward(1.0) == 0.0
    reward = module.blended_designer_reward(no_hint_win_rate=0.5, hint_win_rate=0.75)
    assert reward["regret"] == 0.25
    assert reward["reward"] == 1.0

    spec = {
        "title": "A deliberately varied coordination environment",
        "root_environment": "Inspect the contract and delegate once before producing the terminal JSON.",
        "child_environment": (
            "Use the supplied inline evidence, calculate through IPython, and report once with agent_message.send."
        ),
        "root_hint": "Preserve the child handle and resume only after the passive report arrives.",
        "child_hint": "Treat the evidence card as inline text, calculate in Python, and send the integer.",
        "novelty": "This environment emphasizes passive resumption and explicit child-side computation.",
    }
    validated = module._validate_spec(spec, forbidden_answers={'{"answer":37}'})
    no_hint = module._context(environment_id="env-1", spec=validated, include_hint=False)
    hint = module._context(environment_id="env-1", spec=validated, include_hint=True)

    assert no_hint != hint
    assert spec["child_hint"] not in no_hint
    assert spec["child_hint"] in hint
    assert '{"answer":37}' not in hint
    marker = hint.split(f"{module.COEVOLUTION_CONTEXT_HEADER}\n", 1)[1].splitlines()[0]
    assert spec["child_hint"] in json.loads(marker)["child_context"]
    unsafe = dict(spec)
    unsafe["root_environment"] = (
        "Read the child-owned private evidence file and multiply its integer; that result is the reward."
    )
    with pytest.raises(ValueError, match="reward"):
        module._validate_spec(unsafe, forbidden_answers=set())


def test_spade_designer_document_grounding_is_validated_and_deterministic(tmp_path) -> None:
    module = _module("q35_2b_spade_coevolution_v1")
    documents = []
    for index in range(4):
        content = f"Protocol document {index}: " + "retain state and use explicit messages. " * 4
        documents.append(
            {
                "document_id": f"doc-{index}",
                "source_path": f"docs/{index}.md",
                "heading": f"Protocol {index}",
                "tags": ["delegation"],
                "content": content,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
        )
    path = tmp_path / "documents.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": module.DOCUMENT_CORPUS_SCHEMA_VERSION,
                "upstream": {
                    "repository": "https://github.com/PrimeIntellect-ai/prime-agent",
                    "revision": "a" * 40,
                },
                "documents": documents,
            }
        )
    )

    corpus = module._document_corpus(path)
    first = module._sample_documents(corpus, selector="batch:0", count=3)
    repeated = module._sample_documents(corpus, selector="batch:0", count=3)
    second = module._sample_documents(corpus, selector="batch:1", count=3)

    assert first == repeated
    assert len(first) == 3
    assert first != second
    prompt = json.loads(
        module._generation_prompt(
            track="yield",
            phase="e0d3_uncapped_yield",
            corpus=[],
            memory={},
            grounding_documents=first,
            candidate_index=0,
        )
    )
    assert prompt["prime_agent_protocol_grounding"] == first
    assert "document_id" in prompt["requirements"]["source_grounding"]

    documents[0]["content_sha256"] = "0" * 64
    path.write_text(
        json.dumps(
            {
                "schema_version": module.DOCUMENT_CORPUS_SCHEMA_VERSION,
                "upstream": {"repository": "https://example.test", "revision": "a" * 40},
                "documents": documents,
            }
        )
    )
    with pytest.raises(ValueError, match="invalid Environment Designer document"):
        module._document_corpus(path)


def test_failed_trace_exposes_only_hash_matched_positive_role_prefixes() -> None:
    module = _module("summarize_q35_2b_interaction_curriculum_v1")
    spawn = "worker = await rlm('extract the inline integer', name='worker')"
    child_send = "await agent_message.send('7', receiver_role='parent')"
    def digest(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()
    trace = {
        "is_completed": True,
        "ok": True,
        "errors": [],
        "stop_condition": "user_closed",
        "metrics": {
            "no_forbidden_atoms": 1.0,
            "cardinality_exact": 1.0,
            "local_work_before_yield": 1.0,
            "forbidden_post_spawn_tool_before_child": 0.0,
        },
        "info": {
            "interaction_curriculum": {
                "schema_version": module.CURRICULUM_SCHEMA_VERSION,
                "phase": "e0d3_uncapped_yield",
                "events": [
                    {"kind": "root_retained_spawn", "code_sha256": digest(spawn)},
                    {"kind": "child_value_send", "sampled_code_sha256": digest(child_send)},
                ],
            },
            "natural_yield_scaffold": {"spawn_node_index": 1},
        },
        "nodes": [
            {"parent": None, "sampled": False, "message": {"role": "user", "content": "task"}},
            {
                "parent": 0,
                "sampled": True,
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"name": "ipython", "arguments": json.dumps({"code": spawn})}],
                },
            },
            {"parent": 1, "sampled": True, "message": {"role": "assistant", "content": "Waiting."}},
            {"parent": None, "sampled": False, "message": {"role": "user", "content": "child system"}},
            {"parent": 3, "sampled": False, "message": {"role": "user", "content": "child task"}},
            {
                "parent": 4,
                "sampled": True,
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"name": "ipython", "arguments": json.dumps({"code": child_send})}],
                },
            },
        ],
    }

    assert module.positive_prefix_audit(trace, "e0d3_uncapped_yield") == {
        "coordinator": {"target_node_index": 2, "atoms": ["root_retained_spawn", "passive_yield"]},
        "child": {"target_node_index": 5, "atoms": ["child_value_send"]},
    }
    trace["nodes"][5]["message"]["tool_calls"][0]["arguments"] = json.dumps(
        {"code": "await agent_message.send('8', receiver_role='parent')"}
    )
    assert module.positive_prefix_audit(trace, "e0d3_uncapped_yield") == {}


def test_scaffolded_designer_repair_is_safe_and_complete() -> None:
    module = _module("q35_2b_spade_coevolution_v1")

    repaired = module._repair_spec(track="yield", index=0)

    assert module._validate_spec(repaired, forbidden_answers=set()) == repaired
    assert set(repaired) == module.SPEC_KEYS
    assert all("reward" not in value.lower() for value in repaired.values())


def test_rewarded_designer_and_auxiliary_rows_enter_bounded_replay(tmp_path) -> None:
    module = _module("combine_q35_2b_role_replay_sft_v1")

    def write(path: Path, rows: list[dict], manifest: dict) -> None:
        path.mkdir()
        parquet = path / "train.parquet"
        Dataset.from_list(rows).to_parquet(str(parquet))
        manifest["dataset"] = {"sha256": module.sha256_file(parquet)}
        (path / "MANIFEST.json").write_text(json.dumps(manifest))

    def rows(prefix: str, count: int, *, objective: str | None = None) -> list[dict]:
        return [
            {
                "messages": [],
                "tools": "[]",
                "axis": "natural_n1a",
                "phase": "e0d3_uncapped_yield",
                "task_key": f"{prefix}-{index}",
                "trace_id": f"{prefix}-{index}",
                "role": "coordinator",
                **({"objective": objective} if objective else {}),
            }
            for index in range(count)
        ]

    source = tmp_path / "source"
    auxiliary = tmp_path / "auxiliary"
    designer = tmp_path / "designer"
    output = tmp_path / "output"
    interaction_manifest = {
        "schema_version": module.SOURCE_SCHEMA_VERSION,
        "selected_roles": ["coordinator"],
        "rows_by_role": {"coordinator": 2},
        "student": {"dense_weight_mutated": True},
    }
    write(source, rows("main", 2), dict(interaction_manifest))
    write(auxiliary, rows("paired", 2), dict(interaction_manifest))
    write(
        designer,
        rows("designer", 1, objective="environment_designer"),
        {
            "schema_version": module.REWARDED_DESIGNER_SOURCE_SCHEMA_VERSION,
            "role": "coordinator",
            "objective": "environment_designer",
            "training_stage": "delayed_reward_filtered_coevolution",
            "rows": 1,
            "selection_count": 1,
            "batch_id": "batch-previous",
            "selected_environment_ids": ["env-previous"],
            "exact_answer_rows": 0,
        },
    )

    manifest = module.combine(
        new_source=source,
        auxiliary_sources=[auxiliary],
        designer_source=designer,
        output_dir=output,
        role="coordinator",
        max_rows=8,
    )

    assert manifest["rows"] == 5
    assert manifest["new_interaction_rows"] == 2
    assert manifest["new_auxiliary_rows"] == 2
    assert manifest["new_designer_rows"] == 1
    assert manifest["environment_designer"] == {
        "mode": "paired_hint_regret",
        "trained_batch_ids": ["batch-previous"],
        "selected_environment_ids": ["env-previous"],
        "training_stage": "delayed_reward_filtered_coevolution",
    }
