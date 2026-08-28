import importlib.util
import json
import sys
from pathlib import Path

from datasets import Dataset

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/export_q35_2b_child_action_booster_sft_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("child_action_booster", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

BOOSTER_SCRIPT = ROOT / "scripts/run_q35_2b_child_action_sft_booster_v1.py"
BOOSTER_SPEC = importlib.util.spec_from_file_location("child_action_sft_booster", BOOSTER_SCRIPT)
assert BOOSTER_SPEC is not None and BOOSTER_SPEC.loader is not None
BOOSTER_MODULE = importlib.util.module_from_spec(BOOSTER_SPEC)
BOOSTER_SPEC.loader.exec_module(BOOSTER_MODULE)


def _trace(trace_id: str, value: int) -> dict:
    tools = [{"name": "ipython", "parameters": {"type": "object"}}]
    return {
        "id": trace_id,
        "ok": True,
        "errors": [],
        "stop_condition": "user_closed",
        "rewards": {"harness_score": {"score": 1}},
        "task": {
            "data": {
                "episode_id": f"task-{trace_id}",
                "oracle": {"children": [{"expected_result": value}]},
            }
        },
        "tools": tools,
        "nodes": [
            {"parent": None, "sampled": False, "message": {"role": "user", "content": "root"}},
            {"parent": 0, "sampled": True, "message": {"role": "assistant", "content": "root action"}},
            {"parent": None, "sampled": False, "message": {"role": "user", "content": "child private evidence"}},
            {"parent": 2, "sampled": False, "message": {"role": "user", "content": "child task"}},
            {"parent": 3, "sampled": True, "message": {"role": "assistant", "content": "bad attempt"}},
        ],
    }


def test_export_builds_canonical_one_action_child_targets(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    traces.write_text(
        "".join(json.dumps({"traces": [_trace(f"trace-{index}", 10 + index)]}) + "\n" for index in range(3))
    )
    output = tmp_path / "sft"

    manifest = MODULE.export(traces=[traces], output_dir=output, max_rows=2)

    assert manifest["rows"] == 2
    rows = Dataset.from_parquet(str(output / "train.parquet"))
    assert list(rows["expected_result"]) == [11, 12]
    for row in rows:
        target = row["messages"][-1]
        assert target["role"] == "assistant"
        assert target["content"] == ""
        assert len(target["tool_calls"]) == 1
        arguments = json.loads(target["tool_calls"][0]["arguments"])
        assert arguments["code"] == (f"await agent_message.send('{row['expected_result']}', receiver_role='parent')")
        assert len(row["messages"]) == 3


def test_export_rejects_non_success_only_bank(tmp_path: Path) -> None:
    trace = _trace("failed", 5)
    trace["rewards"]["harness_score"]["score"] = 0
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps({"traces": [trace]}) + "\n")

    try:
        MODULE.export(traces=[traces], output_dir=tmp_path / "sft")
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("non-success child bank was exported")


def test_contract_recovery_canonicalizes_near_miss_and_trains_stop(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    rows = []
    for index in range(3):
        trace = _trace(f"near-miss-{index}", 20 + index)
        trace["stop_condition"] = "max_turns"
        trace["rewards"]["harness_score"]["score"] = 0
        rows.append({"traces": [trace]})
    traces.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "sft"

    manifest = MODULE.export(
        traces=[traces],
        output_dir=output,
        max_rows=3,
        contract_recovery=True,
    )

    assert manifest["schema_version"] == MODULE.CONTRACT_RECOVERY_SCHEMA_VERSION
    assert manifest["objective"] == "canonical_exact_parent_send_ack_then_stop"
    exported = Dataset.from_parquet(str(output / "train.parquet"))
    assert len(exported) == 3
    for row in exported:
        send, acknowledgement, stop = row["messages"][-3:]
        assert send["role"] == "assistant"
        assert acknowledgement["role"] == "tool"
        assert acknowledgement["tool_call_id"] == send["tool_calls"][0]["id"]
        assert stop["role"] == "assistant"
        assert stop["content"] == "Done."
        assert stop["tool_calls"] == []
        assert "stop" not in send["reasoning_content"].lower()


def test_contract_recovery_skips_traces_without_a_child_branch(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    root_only = _trace("root-only", 19)
    root_only["nodes"] = root_only["nodes"][:2]
    rows = [root_only, _trace("child-1", 20), _trace("child-2", 21)]
    traces.write_text(json.dumps({"traces": rows}) + "\n")
    output = tmp_path / "sft"

    manifest = MODULE.export(
        traces=[traces],
        output_dir=output,
        contract_recovery=True,
    )

    assert manifest["rows"] == 2
    assert manifest["task_keys"] == ["task-child-1", "task-child-2"]


def test_booster_config_is_bounded_full_dense_sft(tmp_path: Path) -> None:
    config = BOOSTER_MODULE.training_config(
        run_name="child-booster",
        model_path=tmp_path / "model",
        dataset_dir=tmp_path / "dataset",
        output_root=tmp_path / "outputs",
        rows=8,
        lr=5e-6,
        optimizer_updates=4,
    )

    assert "max_steps = 4" in config
    assert 'type = "sft"' in config
    assert "batch_size = 8" in config
    assert "lr = 5e-06" in config
    assert "lora" not in config.lower()


def test_booster_config_accepts_bounded_contract_recovery_bank(tmp_path: Path) -> None:
    config = BOOSTER_MODULE.training_config(
        run_name="child-contract-recovery",
        model_path=tmp_path / "model",
        dataset_dir=tmp_path / "dataset",
        output_root=tmp_path / "outputs",
        rows=24,
        lr=5e-6,
        optimizer_updates=2,
    )

    assert "max_steps = 2" in config
    assert "batch_size = 24" in config
