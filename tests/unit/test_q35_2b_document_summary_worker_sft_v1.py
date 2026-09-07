import importlib.util
import json
import sys
from pathlib import Path

from datasets import Dataset


def _module():
    scripts = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "export_q35_2b_document_summary_worker_sft_v1",
            scripts / "export_q35_2b_document_summary_worker_sft_v1.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _runner_module():
    scripts = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "run_q35_2b_document_decision_sft_v1",
            scripts / "run_q35_2b_document_decision_sft_v1.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _repair_module():
    scripts = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "export_q35_2b_document_summary_worker_repair_sft_v1",
            scripts / "export_q35_2b_document_summary_worker_repair_sft_v1.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _mixed_module():
    scripts = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "export_q35_2b_document_summary_worker_mixed_sft_v1",
            scripts / "export_q35_2b_document_summary_worker_mixed_sft_v1.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _source_trace(tmp_path: Path) -> Path:
    trace = {
        "id": "failed-but-authentic-summary-probe",
        "task": {"type": "DocumentSummaryWorkerTask", "data": {}},
        "nodes": [
            {
                "parent": None,
                "sampled": False,
                "message": {"role": "user", "content": "Prime Agent runtime contract"},
            },
            {
                "parent": 0,
                "sampled": False,
                "message": {"role": "user", "content": "probe task"},
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
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps({"traces": [trace]}) + "\n")
    return path


def test_summary_worker_export_is_balanced_native_and_held_out(tmp_path: Path) -> None:
    module = _module()
    source = _source_trace(tmp_path)
    output = tmp_path / "dataset"

    manifest = module.export(traces=[source], output_dir=output)
    rows = Dataset.from_parquet(str(output / "train.parquet"))

    assert manifest["rows"] == 12
    assert manifest["family_counts"] == {
        "summary_operations": 4,
        "summary_planning": 4,
        "summary_safety": 4,
    }
    assert manifest["answer_free"] is False
    assert manifest["authored_reference"] is True
    assert manifest["native_prime_agent_context"] is True
    assert manifest["evaluation_document_excluded"] is True
    assert len(rows) == 12
    assert len({row["task_key"] for row in rows}) == 12
    assert all(len(row["messages"]) == 7 for row in rows)
    rendered = json.dumps([row["messages"] for row in rows])
    assert module.EVALUATION_DOCUMENT_ID not in rendered
    assert "Project Northstar" not in rendered
    assert _runner_module()._validated_dataset(output) == manifest

    for chapter in module.TRAINING_CHAPTERS:
        report = chapter["report"]
        paragraph_ids = {row["id"] for row in chapter["paragraphs"]}
        assert report["worker"] == f"{chapter['slug']}-summarizer"
        assert report["chapter_id"] == chapter["slug"]
        assert report["issues"] == []
        assert [row["id"] for row in report["bullets"]] == [f"{chapter['slug']}-b{index:02d}" for index in range(1, 4)]
        assert {source_id for bullet in report["bullets"] for source_id in bullet["source_ids"]} == paragraph_ids
        assert all(5 <= len(row["text"].split()) <= 45 for row in report["bullets"])
        assert all(
            bullet["text"] not in {paragraph["text"] for paragraph in chapter["paragraphs"]}
            for bullet in report["bullets"]
        )


def test_summary_targets_use_correct_paths_schema_and_grounding(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "dataset"
    module.export(traces=[_source_trace(tmp_path)], output_dir=output)
    rows = Dataset.from_parquet(str(output / "train.parquet"))

    for row in rows:
        messages = row["messages"]
        read_call = messages[2]["tool_calls"][0]["function"]
        write_call = messages[4]["tool_calls"][0]["function"]
        read_code = json.loads(read_call["arguments"])["code"]
        write_code = json.loads(write_call["arguments"])["code"]
        assert "job_path.read_text" in read_code
        assert "output_path.write_text" in write_code
        assert module.OUTPUT_PATH in write_code
        assert "json.dump(" not in write_code
        assert "output_path.parent.mkdir" in write_code
        assert "set(ordered_source_ids) == expected_source_ids" in write_code
        assert "len(ordered_source_ids) == len(expected_source_ids)" in write_code
        assert "each paragraph ID must appear exactly once" in messages[1]["content"]
        assert "keep every cited source's facts in that same bullet" in messages[0]["content"]
        assert "keep each fact with its literal source ID" in messages[4]["reasoning_content"]
        assert messages[6]["tool_calls"] == []


def test_training_runner_accepts_summary_worker_contract() -> None:
    module = _runner_module()

    assert module.DATASET_CONTRACTS["qwen35-2b-document-summary-worker-sft/v1"] == (
        "child",
        "grounded_english_chapter_summary_report",
    )
    assert module.DATASET_ANSWER_FREE["qwen35-2b-document-summary-worker-sft/v1"] is False
    assert module.DATASET_ROWS["qwen35-2b-document-summary-worker-sft/v1"] == 12


def test_summary_repair_export_teaches_atomic_correction_without_bad_turns(
    tmp_path: Path,
) -> None:
    module = _repair_module()
    output = tmp_path / "repair-dataset"
    manifest = module.export(traces=[_source_trace(tmp_path)], output_dir=output)
    rows = Dataset.from_parquet(str(output / "train.parquet"))

    assert manifest["rows"] == 12
    assert manifest["family_counts"] == {
        "summary_repair_operations": 4,
        "summary_repair_planning": 4,
        "summary_repair_safety": 4,
    }
    assert manifest["repair_only"] is True
    assert manifest["initial_bad_assistant_turns"] == 0
    assert manifest["on_policy_failure_context"] is True
    assert _runner_module()._validated_dataset(output) == manifest
    rendered = json.dumps([row["messages"] for row in rows])
    assert module.EVALUATION_DOCUMENT_ID not in rendered
    assert "Project Northstar" not in rendered
    for row in rows:
        messages = row["messages"]
        assert len(messages) == 7
        assert messages[1]["role"] == "user"
        assert "missing paragraph coverage" in messages[1]["content"]
        assert "verbatim source copying" in messages[1]["content"]
        assert "There is no parent receiver" in messages[1]["content"]
        assert "do not call agent_message" in messages[1]["content"]
        assert "Do not edit the job" in messages[1]["content"]
        assert "update only the required worker-report.json" in messages[1]["content"]
        assert "I am the terminal worker" in messages[2]["reasoning_content"]
        assert "will not message a parent" in messages[2]["reasoning_content"]
        write_code = json.loads(
            messages[4]["tool_calls"][0]["function"]["arguments"]
        )["code"]
        assert "report['bullets'] =" in write_code
        assert "covered_ids == expected_ids" in write_code
        assert messages[6]["tool_calls"] == []

        tool_payload = messages[3]["content"]
        missing = messages[1]["content"].split("missing paragraph coverage: ", 1)[1]
        assert "report" in tool_payload
        assert missing.startswith("['")


def test_training_runner_accepts_summary_repair_contract() -> None:
    module = _runner_module()

    assert module.DATASET_CONTRACTS[
        "qwen35-2b-document-summary-worker-repair-sft/v1"
    ] == ("child", "grounded_english_chapter_summary_gate_repair")
    assert (
        module.DATASET_ANSWER_FREE[
            "qwen35-2b-document-summary-worker-repair-sft/v1"
        ]
        is False
    )
    assert module.DATASET_ROWS["qwen35-2b-document-summary-worker-repair-sft/v1"] == 12


def test_summary_repair_training_wrapper_is_bounded() -> None:
    wrapper = (
        Path(__file__).parents[2]
        / "scripts/run_q35_2b_document_summary_worker_repair_sft_v1.sh"
    ).read_text()

    assert "optimizer_updates=${4:-2}" in wrapper
    assert '--optimizer-updates "$optimizer_updates"' in wrapper


def test_mixed_summary_export_interleaves_base_and_repair_rows(
    tmp_path: Path,
) -> None:
    module = _mixed_module()
    output = tmp_path / "mixed-dataset"
    manifest = module.export(traces=[_source_trace(tmp_path)], output_dir=output)
    rows = Dataset.from_parquet(str(output / "train.parquet"))

    assert manifest["rows"] == 24
    assert manifest["base_rows"] == manifest["repair_rows"] == 12
    assert manifest["batch_size"] == 12
    assert manifest["initial_bad_assistant_turns"] == 0
    assert manifest["family_counts"] == {
        "summary_base_operations": 4,
        "summary_base_planning": 4,
        "summary_base_safety": 4,
        "summary_repair_operations": 4,
        "summary_repair_planning": 4,
        "summary_repair_safety": 4,
    }
    assert len(rows) == 24
    kinds = [row["task_key"].split("-")[2] for row in rows]
    assert kinds == ["base", "repair"] * 12
    assert all(kinds[offset : offset + 12].count("base") == 6 for offset in (0, 12))
    assert _runner_module()._validated_dataset(output) == manifest
    rendered = json.dumps([row["messages"] for row in rows])
    assert module.EVALUATION_DOCUMENT_ID not in rendered
    assert "Project Northstar" not in rendered


def test_training_runner_accepts_mixed_summary_contract() -> None:
    module = _runner_module()
    schema = "qwen35-2b-document-summary-worker-mixed-sft/v1"

    assert module.DATASET_CONTRACTS[schema] == (
        "child",
        "grounded_english_chapter_summary_and_gate_repair",
    )
    assert module.DATASET_ANSWER_FREE[schema] is False
    assert module.DATASET_ROWS[schema] == 24
    assert module.DATASET_BATCH_SIZES[schema] == 12


def test_mixed_summary_training_wrapper_is_bounded() -> None:
    wrapper = (
        Path(__file__).parents[2]
        / "scripts/run_q35_2b_document_summary_worker_mixed_sft_v1.sh"
    ).read_text()

    assert "optimizer_updates=${4:-2}" in wrapper
    assert '--optimizer-updates "$optimizer_updates"' in wrapper


def test_summary_training_wrapper_accepts_a_bounded_update_count() -> None:
    wrapper = (
        Path(__file__).parents[2]
        / "scripts/run_q35_2b_document_summary_worker_sft_v1.sh"
    ).read_text()

    assert "optimizer_updates=${4:-2}" in wrapper
    assert '--optimizer-updates "$optimizer_updates"' in wrapper


def test_summary_worker_smoke_routes_depth_zero_to_worker_checkpoint() -> None:
    wrapper = (
        Path(__file__).parents[2]
        / "scripts/run_q35_2b_document_summary_smoke_v1.sh"
    ).read_text()

    assert '"$worker_model" "$worker_model" "$label" "$revision"' in wrapper
    assert '"$owner_model" "$worker_model" "$label" "$revision"' not in wrapper
    assert "depth_zero_routed_model=%s" in wrapper
