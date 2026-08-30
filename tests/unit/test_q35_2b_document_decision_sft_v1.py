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
        path.write_text(
            "\n".join(json.dumps({"traces": [_trace(family, variant)]}) for variant in range(4))
            + "\n"
        )
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
    assert all(
        row["messages"][-1]["tool_calls"][0]["type"] == "function" for row in rows
    )
    assert all(
        row["messages"][-1]["tool_calls"][0]["function"]["name"] == "ipython"
        for row in rows
    )
    assert all(
        json.loads(row["messages"][-1]["tool_calls"][0]["function"]["arguments"])[
            "code"
        ]
        for row in rows
    )
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
    assert module.DATASET_CONTRACTS[
        "qwen35-2b-document-coordinator-fanin-sft/v1"
    ] == (
        "coordinator",
        "grounded_document_coordinator_spawn_partial_yield_fanin",
    )
    assert module.DATASET_ANSWER_FREE[
        "qwen35-2b-document-coordinator-fanin-sft/v1"
    ] is False


def test_dual_policy_launcher_exposes_inference_sibling_binaries() -> None:
    launcher = (
        Path(__file__).parents[2] / "scripts" / "run_q35_2b_dual_policy_mastery_v1.sh"
    ).read_text()

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
                        "content": (
                            "Recursive agent depth: 1\nYou are a child agent\n"
                            f"template={stem}"
                        ),
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
                                "text": (
                                    "[task from parent]\nRead "
                                    f"/workspace/document-recursion/v3-i20000/{stem}.md"
                                ),
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
    spawn_code = json.loads(
        spawn_row["messages"][-1]["tool_calls"][0]["function"]["arguments"]
    )["code"]
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
            message["role"] == "user"
            and "[from child:" in message.get("content", "")
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
