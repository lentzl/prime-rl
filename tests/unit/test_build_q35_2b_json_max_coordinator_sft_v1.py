from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_q35_2b_json_max_coordinator_sft_v1.py"
SPEC = importlib.util.spec_from_file_location("json_max_coordinator_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def episode() -> dict:
    return {
        "episode_id": "train_gen-json_max_two_shard-00000001-test",
        "split": "train_gen",
        "index": 1,
        "seed": 7,
        "public": {
            "system_prompt": builder.REQUIRED_SYSTEM_PROMPT,
            "user_prompt": (
                "review-worker owns /workspace/review.json. The coordinator owns "
                "/workspace/coordinator.json. Return remote_max, local_max, and global_max."
            ),
            "workspace_files": {"/workspace/coordinator.json": '{"a":-4,"b":9}'},
        },
        "oracle": {
            "children": [
                {
                    "name": "review-worker",
                    "resource_path": "/workspace/review.json",
                    "operation": "return the largest JSON integer value",
                    "expected_result": 6,
                }
            ],
            "resource_ownership": {
                "/workspace/review.json": {"owner": "child:review-worker", "family": "json_max"},
                "/workspace/coordinator.json": {"owner": "coordinator", "family": "json_max"},
            },
            "private_resources": {"/workspace/review.json": '{"x":6,"y":-2}'},
            "final_answer": {"remote_max": 6, "local_max": 9, "global_max": 9},
        },
    }


def runtime() -> dict:
    return {"role": "user", "content": "exact root runtime"}


def test_direct_row_parses_both_owned_shards_and_finalizes() -> None:
    direct = episode()
    direct["public"]["workspace_files"]["/workspace/review.json"] = direct["oracle"][
        "private_resources"
    ]["/workspace/review.json"]
    row = builder.direct_row(direct, runtime=runtime(), tools="[]")
    builder.validate_target_row(row)
    call = row["messages"][2]["tool_calls"][0]
    code = json.loads(call["function"]["arguments"])["code"]
    assert code.count("Path(") == 2
    assert "await rlm(" not in code
    assert row["messages"][-1]["content"] == '{"remote_max":6,"local_max":9,"global_max":9}'


def test_composed_rows_cover_spawn_wait_and_integration() -> None:
    rows = builder.composed_rows(episode(), runtime=runtime(), tools="[]")
    assert [row["phase"] for row in rows] == [
        "composed_spawn",
        "composed_wait",
        "composed_complete",
    ]
    for row in rows:
        builder.validate_target_row(row)
        call = row["messages"][2]["tool_calls"][0]
        code = json.loads(call["function"]["arguments"])["code"]
        assert code.count("await rlm(") == 1
        assert code.index("await rlm(") < code.index("Path(")
        assert "Path('/workspace/review.json')" not in code
    complete = rows[-1]
    assert complete["messages"][-2]["content"].endswith("\n\n6")
    assert complete["messages"][-1]["content"] == '{"remote_max":6,"local_max":9,"global_max":9}'


def test_runtime_contract_names_native_interfaces_without_conflict(tmp_path: Path) -> None:
    content = "\n".join(
        (
            "Recursive agent depth: 0",
            "await rlm('sub-task') returns immediately after task admission",
            "Children reply explicitly with `await agent_message.send",
            "Spawn independent children in separate calls and end your turn instead of awaiting completion.",
            builder.REQUIRED_SYSTEM_PROMPT,
        )
    )
    trace = {
        "traces": [
            {
                "nodes": [
                    {
                        "parent": None,
                        "sampled": False,
                        "message": {"role": "user", "content": content},
                    }
                ],
                "tools": [{"name": "ipython", "description": "run code", "parameters": {}}],
                "task": {"data": {"system_prompt": builder.REQUIRED_SYSTEM_PROMPT}},
            }
        ]
    }
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(trace) + "\n")
    _runtime, tools, audit = builder.load_runtime_surface(path)
    assert [tool["name"] for tool in json.loads(tools)] == ["ipython"]
    assert audit["conflict_found"] is False


def test_target_validator_rejects_agent_message_as_spawn() -> None:
    row = builder.composed_rows(episode(), runtime=runtime(), tools="[]")[0]
    call = row["messages"][2]["tool_calls"][0]
    arguments = json.loads(call["function"]["arguments"])
    arguments["code"] = arguments["code"].replace("await rlm(", "await agent_message.send(")
    call["function"]["arguments"] = json.dumps(arguments)
    try:
        builder.validate_target_row(row)
    except ValueError:
        pass
    else:
        raise AssertionError("non-native spawn target was accepted")


def test_training_configs_are_bounded_fresh_e33_descendants() -> None:
    root = Path(__file__).resolve().parents[2]
    config_dir = root / "experiments" / "qwen35-2b-json-max-coordinator-v1"
    tiny = tomllib.loads((config_dir / "tiny-fit-e33-step8.toml").read_text())
    curve = tomllib.loads((config_dir / "fresh-curve-e33-4pass.toml").read_text())

    e33 = (
        "/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/"
        "c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2"
    )
    for config in (tiny, curve):
        assert config["model"]["name"] == e33
        assert config["tokenizer"]["name"] == e33
        assert config["deployment"] == {
            "type": "single_node",
            "gpus_per_node": 2,
            "num_gpus": 2,
        }
        assert config["model"]["optimization_dtype"] == "bfloat16"
        assert config["model"]["reduce_dtype"] == "bfloat16"
        assert config["data"]["batch_size"] == 12
        assert config["data"]["micro_batch_size"] == 1
        assert config["data"]["seq_len"] == 16384
        assert config["data"]["shuffle"] is False
        assert config["data"]["loss_mask"] == {
            "system": False,
            "user": False,
            "assistant": True,
            "tool": False,
        }
        assert config["optim"] == {
            "type": "adamw",
            "lr": 1e-6,
            "weight_decay": 0.01,
            "max_norm": 1.0,
            "betas1": 0.9,
            "betas2": 0.999,
        }
        assert config["scheduler"] == {"type": "constant"}

    assert tiny["max_steps"] == 8
    assert tiny["data"]["name"].endswith("/tiny_fit")
    assert tiny["ckpt"] == {
        "interval": 1,
        "keep_last": 1,
        "weights_only": True,
        "weights": {"save_sharded": True, "save_format": "safetensors"},
    }

    assert curve["max_steps"] == 192
    assert curve["data"]["name"].endswith("/train")
    assert curve["ckpt"]["interval"] == 48
    assert curve["ckpt"]["keep_interval"] == 48


def test_qualification_driver_supports_an_explicit_train_fit_split() -> None:
    launcher = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_qwen38_27b_prime_harness_qualification_v1.sh"
    ).read_text()
    assert "split_override=${QWEN38_QUALIFICATION_SPLIT:-}" in launcher
    assert '""|train_gen|valid_gen' in launcher
    assert 'split=$split_override' in launcher
    assert "printf 'split_override=%s\\n'" in launcher
