import importlib.util
import json
import sys
from pathlib import Path

from datasets import Dataset


def _module(name: str):
    scripts = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _trace(family: str, variant: int) -> dict:
    root = f"/workspace/document-recursion/v{variant}-i20000"
    return {
        "id": f"{family}-{variant}",
        "ok": False,
        "task": {
            "data": {
                "name": f"{family}-v{variant}-i20000",
                "family": family,
                "answer": {
                    "alpha_words": 20,
                    "alpha_h2": 2,
                    "beta_words": 30,
                    "beta_h2": 3,
                    "gamma_words": 40,
                    "gamma_h2": 4,
                    "total_words": 90,
                    "total_h2": 9,
                },
                "files": {f"{root}/{stem}.md": f"# {stem}\n" for stem in ("alpha", "beta", "gamma")},
            }
        },
        "nodes": [
            {
                "sampled": False,
                "parent": None,
                "message": {"role": "user", "content": "runtime contract"},
            },
            {
                "sampled": False,
                "parent": 0,
                "message": {"role": "user", "content": [{"type": "text", "text": "task"}]},
            },
            {
                "sampled": True,
                "parent": 1,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "I should act, but fail to emit a tool call.",
                },
            },
        ],
        "tools": [
            {
                "name": "ipython",
                "description": "execute code",
                "parameters": {
                    "type": "object",
                    "required": ["code"],
                    "properties": {"code": {"type": "string"}},
                },
                "strict": False,
            }
        ],
    }


def _cleanup_trace(variant: int) -> dict:
    trace = _trace("document_flat", variant)
    root = f"/workspace/document-recursion/v{variant}-i20000"
    trace["metrics"] = {
        "answer_accuracy": 1,
        "protocol_aligned": 1,
        "clean_protocol_aligned": 0,
        "spawn_calls": 3,
        "failed_spawn_calls": 0,
        "retained_handles": 3,
        "named_children": 3,
        "delegated_payloads": 3,
        "coordinator_delegated_path_accesses": 0,
        "messages_to_parent": 3,
        "fan_in_complete": 1,
        "failed_cells": 3,
    }
    spawn_id = f"spawn-{variant}"
    nodes = trace["nodes"][:2]
    nodes.extend(
        [
            {
                "sampled": True,
                "parent": 1,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Spawn three workers and retain their handles.",
                    "tool_calls": [
                        {
                            "id": spawn_id,
                            "name": "ipython",
                            "arguments": json.dumps({"code": "\n".join("await rlm('task')" for _ in range(3))}),
                        }
                    ],
                },
            },
            {
                "sampled": False,
                "parent": 2,
                "message": {
                    "role": "tool",
                    "name": "ipython",
                    "tool_call_id": spawn_id,
                    "content": "",
                },
            },
            {
                "sampled": True,
                "parent": 3,
                "message": {
                    "role": "assistant",
                    "content": "Waiting for explicit child reports.",
                    "reasoning_content": "I will yield without polling.",
                    "tool_calls": [],
                },
            },
        ]
    )
    report_parent = 4
    for stem in ("alpha", "beta", "gamma"):
        child_root = len(nodes)
        task_index = child_root + 1
        failed_action = child_root + 2
        failed_receipt = child_root + 3
        successful_action = child_root + 4
        successful_receipt = child_root + 5
        action_id = f"child-{variant}-{stem}"
        nodes.extend(
            [
                {
                    "sampled": False,
                    "parent": None,
                    "message": {
                        "role": "user",
                        "content": "Recursive agent depth: 1\nYou are a child agent",
                    },
                },
                {
                    "sampled": False,
                    "parent": child_root,
                    "message": {
                        "role": "user",
                        "content": f"Read {root}/{stem}.md and report to parent.",
                    },
                },
                {
                    "sampled": True,
                    "parent": task_index,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"failed-{action_id}",
                                "name": "ipython",
                                "arguments": json.dumps({"code": "json.dumps(result)"}),
                            }
                        ],
                    },
                },
                {
                    "sampled": False,
                    "parent": failed_action,
                    "message": {
                        "role": "tool",
                        "name": "ipython",
                        "tool_call_id": f"failed-{action_id}",
                        "content": "Traceback: NameError: json is not defined",
                    },
                },
                {
                    "sampled": True,
                    "parent": failed_receipt,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Import json, compute, report, and stop.",
                        "tool_calls": [
                            {
                                "id": action_id,
                                "name": "ipython",
                                "arguments": json.dumps(
                                    {
                                        "code": (
                                            "from pathlib import Path\nimport json\n"
                                            f"text = Path('{root}/{stem}.md').read_text()\n"
                                            "result = {'words': len(text.split()), "
                                            "'h2': sum(line.startswith('## ') for line in text.splitlines())}\n"
                                            "await agent_message.send(json.dumps(result), "
                                            "receiver_role='parent')"
                                        )
                                    }
                                ),
                            }
                        ],
                    },
                },
                {
                    "sampled": False,
                    "parent": successful_action,
                    "message": {
                        "role": "tool",
                        "name": "ipython",
                        "tool_call_id": action_id,
                        "content": "{'deliveryStatus': 'queued'}",
                    },
                },
                {
                    "sampled": True,
                    "parent": successful_receipt,
                    "message": {
                        "role": "assistant",
                        "content": "Done.",
                        "reasoning_content": "The report was delivered, so I stop.",
                        "tool_calls": [],
                    },
                },
            ]
        )
        words = trace["task"]["data"]["answer"][f"{stem}_words"]
        h2 = trace["task"]["data"]["answer"][f"{stem}_h2"]
        report_index = len(nodes)
        nodes.append(
            {
                "sampled": False,
                "parent": report_parent,
                "message": {
                    "role": "user",
                    "content": (
                        f"[from child:{stem}-document-worker]\n"
                        "Agent-to-agent message received.\n\n" + json.dumps({"words": words, "h2": h2})
                    ),
                },
            }
        )
        report_parent = report_index
    nodes.append(
        {
            "sampled": True,
            "parent": report_parent,
            "message": {
                "role": "assistant",
                "content": json.dumps(trace["task"]["data"]["answer"]),
                "reasoning_content": "All reports are present; return the exact aggregate.",
                "tool_calls": [],
            },
        }
    )
    trace["nodes"] = nodes
    return trace


def test_document_decision_actions_are_topology_specific_and_answer_free() -> None:
    module = _module("export_q35_2b_document_decision_sft_v1")
    root = "/workspace/document-recursion/v0-i20000"

    _, direct = module.canonical_first_action("document_direct", root)
    _, flat = module.canonical_first_action("document_flat", root)
    _, hierarchical = module.canonical_first_action("document_hierarchical", root)

    assert "Path('/workspace/document-recursion/v0-i20000')" in direct
    assert "glob('*.md')" in direct
    assert "len(text.split())" in direct
    assert "await rlm(" not in direct
    assert flat.count("await rlm(") == 3
    assert "alpha-document-worker" in flat
    assert "gamma-document-worker" in flat
    assert "agent_message.send" in flat
    assert hierarchical.count("await rlm(") == 1
    assert "[recursive document coordinator session contract]" in hierarchical
    assert "maximum_descendant_depth=1" in hierarchical
    assert "name='document-manager'" in hierarchical
    for action in (direct, flat, hierarchical):
        assert '"alpha_words":20' not in action
        assert '"total_words":90' not in action


def test_document_decision_export_is_balanced_and_preserves_failed_prefixes(tmp_path: Path) -> None:
    module = _module("export_q35_2b_document_decision_sft_v1")
    sources = []
    for family in sorted(module.FAMILIES):
        path = tmp_path / f"{family}.jsonl"
        path.write_text("\n".join(json.dumps({"traces": [_trace(family, variant)]}) for variant in range(4)) + "\n")
        sources.append(path)

    output = tmp_path / "dataset"
    manifest = module.export(traces=sources, output_dir=output)
    rows = Dataset.from_parquet(str(output / "train.parquet"))

    assert manifest["rows"] == 12
    assert manifest["family_counts"] == {
        "document_direct": 4,
        "document_flat": 4,
        "document_hierarchical": 4,
    }
    assert manifest["answer_free"] is True
    assert len(rows) == 12
    assert all(row["role"] == "coordinator" for row in rows)
    assert all(len(row["messages"]) == 3 for row in rows)
    assert all(row["messages"][-1]["tool_calls"][0]["type"] == "function" for row in rows)
    assert all(row["messages"][-1]["tool_calls"][0]["function"]["name"] == "ipython" for row in rows)
    assert all(json.loads(row["messages"][-1]["tool_calls"][0]["function"]["arguments"])["code"] for row in rows)
    assert manifest["tool_call_format"] == "openai_function_v1"


def test_document_decision_training_is_one_full_dense_update() -> None:
    module = _module("run_q35_2b_document_decision_sft_v1")
    config = module.training_config(
        run_name="test",
        model_path=Path("/models/c177"),
        dataset_dir=Path("/data/decision"),
        output_root=Path("/outputs"),
        learning_rate=2e-6,
        optimizer_updates=8,
    )

    assert "max_steps = 8" in config
    assert "gpus_per_node = 2" in config
    assert "batch_size = 12" in config
    assert "micro_batch_size = 1" in config
    assert "assistant = true" in config
    assert "tool = false" in config
    assert "lr = 2e-06" in config
    assert "lora" not in config.lower()
    assert 'optimization_dtype = "bfloat16"' in config
    assert 'reduce_dtype = "bfloat16"' in config
    assert module.DATASET_CONTRACTS["qwen35-2b-document-child-sft/v1"] == (
        "child",
        "canonical_answer_free_document_leaf_compute_report_stop",
    )
    assert module.DATASET_CONTRACTS["qwen35-2b-document-coordinator-fanin-sft/v1"] == (
        "coordinator",
        "grounded_document_coordinator_spawn_partial_yield_fanin",
    )
    assert module.DATASET_ANSWER_FREE["qwen35-2b-document-coordinator-fanin-sft/v1"] is False


def test_dual_policy_launcher_exposes_inference_sibling_binaries() -> None:
    launcher = (Path(__file__).parents[2] / "scripts" / "run_q35_2b_dual_policy_mastery_v1.sh").read_text()

    assert 'inference_dir=$(cd "$(dirname "$inference_bin")" && pwd)' in launcher
    assert 'export PATH="$inference_dir:$PATH"' in launcher


def test_document_child_export_uses_real_depth_one_context_and_answer_free_code(
    tmp_path: Path,
) -> None:
    module = _module("export_q35_2b_document_child_sft_v1")
    nodes = [
        {
            "sampled": False,
            "parent": None,
            "message": {"role": "user", "content": "root runtime"},
        }
    ]
    for stem in ("alpha", "beta", "gamma"):
        root_index = len(nodes)
        nodes.extend(
            [
                {
                    "sampled": False,
                    "parent": None,
                    "message": {
                        "role": "user",
                        "content": (f"Recursive agent depth: 1\nYou are a child agent\ntemplate={stem}"),
                    },
                },
                {
                    "sampled": False,
                    "parent": root_index,
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (f"[task from parent]\nRead /workspace/document-recursion/v3-i20000/{stem}.md"),
                            }
                        ],
                    },
                },
            ]
        )
    trace = {
        "id": "flat-three-child-template",
        "task": {"data": {"family": "document_flat"}},
        "metrics": {
            "retained_handles": 3,
            "named_children": 3,
            "delegated_payloads": 3,
        },
        "nodes": nodes,
        "tools": [
            {
                "name": "ipython",
                "description": "execute code",
                "parameters": {
                    "type": "object",
                    "required": ["code"],
                    "properties": {"code": {"type": "string"}},
                },
                "strict": False,
            }
        ],
    }
    source = tmp_path / "flat.jsonl"
    source.write_text(json.dumps({"traces": [trace]}) + "\n")
    output = tmp_path / "child-dataset"

    manifest = module.export(traces=[source], output_dir=output)
    rows = Dataset.from_parquet(str(output / "train.parquet"))

    assert manifest["rows"] == 12
    assert manifest["role"] == "child"
    assert manifest["answer_free"] is True
    assert manifest["tool_call_format"] == "openai_function_v1"
    assert set(manifest["family_counts"].values()) == {4}
    assert len(rows) == 12
    for row in rows:
        assert "Recursive agent depth: 1" in row["messages"][0]["content"]
        call = row["messages"][2]["tool_calls"][0]
        assert call["function"]["name"] == "ipython"
        code = json.loads(call["function"]["arguments"])["code"]
        assert "path.read_text()" in code
        assert "len(text.split())" in code
        assert "receiver_role='parent'" in code
        assert '"words":' not in code
        assert row["messages"][-1]["content"] == "Done."


def test_document_coordinator_fanin_export_is_grounded_and_phase_balanced(
    tmp_path: Path,
) -> None:
    module = _module("export_q35_2b_document_coordinator_fanin_sft_v1")
    traces = []
    for variant in range(4):
        trace = _trace("document_flat", variant)
        root = f"/workspace/document-recursion/v{variant}-i20000"
        trace["task"]["data"]["files"] = {
            f"{root}/alpha.md": "# Alpha\n\n## One\n\na b c\n",
            f"{root}/beta.md": "# Beta\n\n## One\n\n## Two\n\na b c d\n",
            f"{root}/gamma.md": "# Gamma\n\n## One\n\na b c d e\n",
        }
        traces.append(trace)
    source = tmp_path / "flat.jsonl"
    source.write_text("\n".join(json.dumps({"traces": [trace]}) for trace in traces) + "\n")
    output = tmp_path / "fanin-dataset"

    manifest = module.export(traces=[source], output_dir=output)
    rows = Dataset.from_parquet(str(output / "train.parquet"))

    assert manifest["rows"] == 12
    assert manifest["answer_free"] is False
    assert manifest["grounded_on_fixture_contents"] is True
    assert manifest["leakage_mode"] == "environment_designer_grounded_v1"
    assert manifest["family_counts"] == {
        "document_coordinator_complete_fanin": 4,
        "document_coordinator_partial_fanin": 4,
        "document_coordinator_spawn": 4,
    }
    spawn_row = next(row for row in rows if row["family"].endswith("_spawn"))
    spawn_code = json.loads(spawn_row["messages"][-1]["tool_calls"][0]["function"]["arguments"])["code"]
    assert spawn_code.count('task = """') == 3
    assert spawn_code.count("await rlm(") == 3
    assert "receiver_role='parent'" in spawn_code
    assert "await rlm('Read" not in spawn_code

    partial_row = next(row for row in rows if row["family"].endswith("partial_fanin"))
    assert "waiting for beta-document-worker" in partial_row["messages"][-1]["content"]
    assert partial_row["messages"][-1]["tool_calls"] == []

    complete_row = next(row for row in rows if row["family"].endswith("complete_fanin"))
    assert (
        sum(
            message["role"] == "user" and "[from child:" in message.get("content", "")
            for message in complete_row["messages"]
        )
        == 3
    )
    final = json.loads(complete_row["messages"][-1]["content"])
    assert final == {
        "alpha_words": 7,
        "alpha_h2": 1,
        "beta_words": 10,
        "beta_h2": 2,
        "gamma_words": 9,
        "gamma_h2": 1,
        "total_words": 26,
        "total_h2": 4,
    }


def test_document_recursive_execution_export_targets_only_missing_coordinator_phases(
    tmp_path: Path,
) -> None:
    module = _module("export_q35_2b_document_recursive_execution_sft_v1")
    source = tmp_path / "hierarchical.jsonl"
    source.write_text(
        "\n".join(json.dumps({"traces": [_trace("document_hierarchical", variant)]}) for variant in range(4)) + "\n"
    )
    runtime_trace = _trace("document_hierarchical", 9)
    runtime_trace["nodes"].append(
        {
            "sampled": False,
            "parent": None,
            "message": {
                "role": "user",
                "content": ("Recursive agent depth: 1\nYou are a child agent spawned by root-coordinator."),
            },
        }
    )
    runtime = tmp_path / "runtime.jsonl"
    runtime.write_text(json.dumps({"traces": [runtime_trace]}) + "\n")
    output = tmp_path / "recursive-execution"

    manifest = module.export(traces=[source], runtime_traces=[runtime], output_dir=output)
    rows = Dataset.from_parquet(str(output / "train.parquet"))

    assert manifest["rows"] == 12
    assert manifest["answer_free"] is True
    assert manifest["topology_choice_targeted"] is False
    assert manifest["child_policy_targeted"] is False
    assert manifest["family_counts"] == {
        "document_recursive_manager_leaf_admission": 4,
        "document_recursive_manager_passive_yield": 4,
        "document_recursive_root_manager_admission": 4,
    }

    root_row = next(row for row in rows if row["phase"] == "root_manager_admission")
    root_code = json.loads(root_row["messages"][-1]["tool_calls"][0]["function"]["arguments"])["code"]
    assert root_code.startswith("document_manager = await rlm(")
    assert "[recursive document coordinator session contract]" in root_code
    assert "maximum_descendant_depth=1" in root_code
    assert "name='document-manager'" in root_code

    manager_row = next(row for row in rows if row["phase"] == "manager_leaf_admission")
    assert "Recursive agent depth: 1" in manager_row["messages"][0]["content"]
    assert "[task from parent]" in manager_row["messages"][1]["content"]
    manager_code = json.loads(manager_row["messages"][-1]["tool_calls"][0]["function"]["arguments"])["code"]
    assert manager_code.count("await rlm(") == 3
    assert "alpha-document-worker" in manager_code
    assert "gamma-document-worker" in manager_code

    passive_row = next(row for row in rows if row["phase"] == "manager_passive_yield")
    assert passive_row["messages"][-2]["role"] == "tool"
    assert passive_row["messages"][-1]["tool_calls"] == []
    assert "no polling or messaging call" in passive_row["messages"][-1]["reasoning_content"]

    trainer = _module("run_q35_2b_document_decision_sft_v1")
    assert trainer.DATASET_CONTRACTS[manifest["schema_version"]] == (
        "coordinator",
        manifest["objective"],
    )
    assert trainer.DATASET_ANSWER_FREE[manifest["schema_version"]] is True


def test_document_manager_admission_export_uses_each_exact_live_depth_two_context(
    tmp_path: Path,
) -> None:
    module = _module("export_q35_2b_document_manager_admission_sft_v1")
    traces = []
    for variant in range(4):
        trace = _trace("document_hierarchical", variant)
        manager_root = len(trace["nodes"])
        trace["nodes"].extend(
            [
                {
                    "sampled": False,
                    "parent": None,
                    "message": {
                        "role": "user",
                        "content": ("Recursive agent depth: 1\nYou are a child agent spawned by root-coordinator."),
                    },
                },
                {
                    "sampled": False,
                    "parent": manager_root,
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "[task from parent]\n\n"
                                    "[recursive document coordinator session contract]\n"
                                    "session_role=document_coordinator\n"
                                    f"variant={variant}"
                                ),
                            }
                        ],
                    },
                },
            ]
        )
        traces.append(trace)
    source = tmp_path / "depth-two-runtime.jsonl"
    source.write_text(json.dumps({"traces": traces}) + "\n")
    output = tmp_path / "manager-admission"

    manifest = module.export(traces=[source], output_dir=output)
    rows = Dataset.from_parquet(str(output / "train.parquet"))

    assert manifest["rows"] == 4
    assert manifest["family_counts"] == {"document_recursive_manager_leaf_admission": 4}
    assert manifest["answer_free"] is True
    assert manifest["exact_live_depth_two_context"] is True
    assert manifest["root_policy_targeted"] is False
    assert manifest["child_policy_targeted"] is False
    assert {row["messages"][1]["content"].split("variant=")[-1] for row in rows} == {
        "0",
        "1",
        "2",
        "3",
    }
    for row in rows:
        assert "Recursive agent depth: 1" in row["messages"][0]["content"]
        assert "[recursive document coordinator session contract]" in row["messages"][1]["content"]
        action = row["messages"][2]["tool_calls"][0]
        code = json.loads(action["function"]["arguments"])["code"]
        assert code.count("await rlm(") == 3
        assert code.count("_worker = await rlm(") == 3
        assert code.count('name="') == 3

    trainer = _module("run_q35_2b_document_decision_sft_v1")
    validated = trainer._validated_dataset(output)
    assert validated["rows"] == 4
    config = trainer.training_config(
        run_name="manager-admission",
        model_path=Path("/models/c177"),
        dataset_dir=output,
        output_root=Path("/outputs"),
        learning_rate=5e-7,
        batch_size=validated["rows"],
    )
    assert "batch_size = 4" in config
    assert trainer.PROMOTION_MINIMUM == 4


def test_document_manager_fanin_export_teaches_passive_yield_and_one_parent_report(
    tmp_path: Path,
) -> None:
    module = _module("export_q35_2b_document_manager_fanin_sft_v1")
    traces = []
    for variant in range(4):
        trace = _trace("document_hierarchical", variant)
        root = f"/workspace/document-recursion/v{variant}-i20000"
        trace["task"]["data"]["files"] = {
            f"{root}/alpha.md": "one two\n## A\n",
            f"{root}/beta.md": "one two three\n## B\n## C\n",
            f"{root}/gamma.md": "one two three four\n## D\n",
        }
        manager_root = len(trace["nodes"])
        trace["nodes"].extend(
            [
                {
                    "sampled": False,
                    "parent": None,
                    "message": {
                        "role": "user",
                        "content": ("Recursive agent depth: 1\nYou are a child agent spawned by root-coordinator."),
                    },
                },
                {
                    "sampled": False,
                    "parent": manager_root,
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "[task from parent]\n\n"
                                    "[recursive document coordinator session contract]\n"
                                    "session_role=document_coordinator"
                                ),
                            }
                        ],
                    },
                },
            ]
        )
        traces.append(trace)
    source = tmp_path / "depth-two-runtime.jsonl"
    source.write_text(json.dumps({"traces": traces}) + "\n")
    output = tmp_path / "manager-fanin"

    manifest = module.export(traces=[source], output_dir=output)
    rows = Dataset.from_parquet(str(output / "train.parquet"))

    assert manifest["rows"] == 12
    assert manifest["answer_free"] is False
    assert manifest["exact_live_depth_two_context"] is True
    assert manifest["manager_admission_targeted"] is False
    assert manifest["family_counts"] == {
        "document_manager_complete_fanin_report": 4,
        "document_manager_partial_fanin_yield": 4,
        "document_manager_post_admission_yield": 4,
    }

    for row in rows:
        assert "Recursive agent depth: 1" in row["messages"][0]["content"]
        spawn_code = json.loads(row["messages"][2]["tool_calls"][0]["function"]["arguments"])["code"]
        assert spawn_code.count("await rlm(") == 3
        if row["phase"].endswith("yield"):
            assert row["messages"][-1]["tool_calls"] == []
            assert "polling" in row["messages"][-1]["reasoning_content"]
        else:
            assert (
                sum(
                    message["role"] == "user" and "[from child:" in message.get("content", "")
                    for message in row["messages"]
                )
                == 3
            )
            report_code = json.loads(row["messages"][-1]["tool_calls"][0]["function"]["arguments"])["code"]
            assert "'alpha_words': 4" in report_code
            assert "'beta_h2': 2" in report_code
            assert "'total_words': 17" in report_code
            assert "json.dumps(parent_report" in report_code
            assert "receiver_role='parent'" in report_code

    trainer = _module("run_q35_2b_document_decision_sft_v1")
    validated = trainer._validated_dataset(output)
    assert validated["rows"] == 12
    assert trainer.DATASET_ANSWER_FREE[manifest["schema_version"]] is False


def test_document_manager_aggregation_export_keeps_only_complete_fanin(
    tmp_path: Path,
) -> None:
    module = _module("export_q35_2b_document_manager_aggregation_sft_v1")
    traces = []
    for variant in range(4):
        trace = _trace("document_hierarchical", variant)
        root = f"/workspace/document-recursion/v{variant}-i20000"
        trace["task"]["data"]["files"] = {
            f"{root}/alpha.md": "one two\n## A\n",
            f"{root}/beta.md": "one two three\n## B\n## C\n",
            f"{root}/gamma.md": "one two three four\n## D\n",
        }
        manager_root = len(trace["nodes"])
        trace["nodes"].extend(
            [
                {
                    "sampled": False,
                    "parent": None,
                    "message": {
                        "role": "user",
                        "content": "Recursive agent depth: 1\nYou are a child agent spawned by root-coordinator.",
                    },
                },
                {
                    "sampled": False,
                    "parent": manager_root,
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "[task from parent]\n\n"
                                    "[recursive document coordinator session contract]\n"
                                    "session_role=document_coordinator"
                                ),
                            }
                        ],
                    },
                },
            ]
        )
        traces.append(trace)
    source = tmp_path / "depth-two-runtime.jsonl"
    source.write_text(json.dumps({"traces": traces}) + "\n")
    output = tmp_path / "manager-aggregation"

    manifest = module.export(traces=[source], output_dir=output)
    rows = Dataset.from_parquet(str(output / "train.parquet"))

    assert manifest["rows"] == 4
    assert manifest["manager_aggregation_targeted"] is True
    assert manifest["family_counts"] == {
        "document_manager_complete_fanin_report": 4
    }
    assert {row["phase"] for row in rows} == {"complete_fanin_report"}
    for row in rows:
        assert sum(
            message["role"] == "user" and "[from child:" in message.get("content", "")
            for message in row["messages"]
        ) == 3
        report_code = json.loads(
            row["messages"][-1]["tool_calls"][0]["function"]["arguments"]
        )["code"]
        assert "parent_report =" in report_code
        assert "json.dumps(parent_report" in report_code
        assert "receiver_role='parent'" in report_code

    trainer = _module("run_q35_2b_document_decision_sft_v1")
    validated = trainer._validated_dataset(output)
    assert validated["rows"] == 4
    assert trainer.DATASET_ANSWER_FREE[manifest["schema_version"]] is False


def test_document_manager_permuted_aggregation_export_balances_all_orders(
    tmp_path: Path,
) -> None:
    module = _module("export_q35_2b_document_manager_aggregation_permuted_sft_v1")
    traces = []
    for variant in range(4):
        trace = _trace("document_hierarchical", variant)
        root = f"/workspace/document-recursion/v{variant}-i20000"
        trace["task"]["data"]["files"] = {
            f"{root}/alpha.md": "one two\n## A\n",
            f"{root}/beta.md": "one two three\n## B\n## C\n",
            f"{root}/gamma.md": "one two three four\n## D\n",
        }
        manager_root = len(trace["nodes"])
        trace["nodes"].extend(
            [
                {
                    "sampled": False,
                    "parent": None,
                    "message": {
                        "role": "user",
                        "content": "Recursive agent depth: 1\nYou are a child agent spawned by root-coordinator.",
                    },
                },
                {
                    "sampled": False,
                    "parent": manager_root,
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "[task from parent]\n\n"
                                    "[recursive document coordinator session contract]\n"
                                    "session_role=document_coordinator"
                                ),
                            }
                        ],
                    },
                },
            ]
        )
        traces.append(trace)
    source = tmp_path / "depth-two-runtime.jsonl"
    source.write_text(json.dumps({"traces": traces}) + "\n")
    output = tmp_path / "manager-aggregation-permuted"

    manifest = module.export(traces=[source], output_dir=output)
    rows = Dataset.from_parquet(str(output / "train.parquet"))

    assert manifest["rows"] == 24
    assert manifest["training_batch_size"] == 12
    assert manifest["arrival_order_balanced"] is True
    assert len(manifest["arrival_orders"]) == 6
    assert set(manifest["family_counts"].values()) == {4}
    for row in rows:
        observed = [
            message["content"].split("[from child:", 1)[1].split("-document-worker]", 1)[0]
            for message in row["messages"]
            if message["role"] == "user" and "[from child:" in message.get("content", "")
        ]
        assert "-".join(observed) == row["arrival_order"]
        report_code = json.loads(
            row["messages"][-1]["tool_calls"][0]["function"]["arguments"]
        )["code"]
        assert "'total_words': 17" in report_code
        assert "'total_h2': 4" in report_code
        assert "receiver_role='parent'" in report_code

    trainer = _module("run_q35_2b_document_decision_sft_v1")
    validated = trainer._validated_dataset(output)
    assert validated["rows"] == 24
    assert trainer.DATASET_BATCH_SIZES[manifest["schema_version"]] == 12


def test_document_cleanup_export_projects_only_admitted_role_lineage(
    tmp_path: Path,
) -> None:
    module = _module("export_q35_2b_document_cleanup_sft_v1")
    source = tmp_path / "successful-traces.jsonl"
    source.write_text(json.dumps({"traces": [_cleanup_trace(variant) for variant in range(4)]}) + "\n")

    child_dir = tmp_path / "child-cleanup"
    child_manifest = module.export(traces_path=source, output_dir=child_dir, role="child")
    child_rows = Dataset.from_parquet(str(child_dir / "train.parquet"))

    assert child_manifest["rows"] == 12
    assert child_manifest["successful_on_policy_sources_only"] is True
    assert child_manifest["failed_prefixes_removed"] is True
    assert child_manifest["answer_free"] is True
    assert child_manifest["family_counts"] == {
        "document_cleanup_child_alpha": 4,
        "document_cleanup_child_beta": 4,
        "document_cleanup_child_gamma": 4,
    }
    for row in child_rows:
        assert len(row["messages"]) == 5
        assert all("Traceback" not in message["content"] for message in row["messages"])
        action = row["messages"][2]["tool_calls"][0]
        assert action["function"]["name"] == "ipython"
        code = json.loads(action["function"]["arguments"])["code"]
        assert "import json" in code
        assert "receiver_role='parent'" in code
        assert row["messages"][-1]["content"] == "Done."

    coordinator_dir = tmp_path / "coordinator-cleanup"
    coordinator_manifest = module.export(traces_path=source, output_dir=coordinator_dir, role="coordinator")
    coordinator_rows = Dataset.from_parquet(str(coordinator_dir / "train.parquet"))

    assert coordinator_manifest["rows"] == 12
    assert coordinator_manifest["answer_free"] is False
    assert coordinator_manifest["family_counts"] == {
        "document_cleanup_coordinator_complete_fanin": 4,
        "document_cleanup_coordinator_partial_fanin": 4,
        "document_cleanup_coordinator_passive_yield": 4,
    }
    for row in coordinator_rows:
        assert all("Traceback" not in message["content"] for message in row["messages"])
        assert row["messages"][-1]["tool_calls"] == []
        if row["family"].endswith("complete_fanin"):
            assert (
                sum(message["role"] == "user" and "[from child:" in message["content"] for message in row["messages"])
                == 3
            )
