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
        assert messages[6]["tool_calls"] == []


def test_training_runner_accepts_summary_worker_contract() -> None:
    module = _runner_module()

    assert module.DATASET_CONTRACTS["qwen35-2b-document-summary-worker-sft/v1"] == (
        "child",
        "grounded_english_chapter_summary_report",
    )
    assert module.DATASET_ANSWER_FREE["qwen35-2b-document-summary-worker-sft/v1"] is False
    assert module.DATASET_ROWS["qwen35-2b-document-summary-worker-sft/v1"] == 12
