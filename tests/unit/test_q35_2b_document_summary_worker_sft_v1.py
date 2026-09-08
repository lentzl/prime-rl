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


def _text_revision_module():
    scripts = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "export_q35_2b_document_summary_text_revision_sft_v1",
            scripts / "export_q35_2b_document_summary_text_revision_sft_v1.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _live_revision_module():
    scripts = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "export_q35_2b_document_summary_live_revision_sft_v2",
            scripts / "export_q35_2b_document_summary_live_revision_sft_v2.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _commit_revision_module():
    scripts = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "export_q35_2b_document_summary_commit_revision_sft_v3",
            scripts
            / "export_q35_2b_document_summary_commit_revision_sft_v3.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _margin_revision_module():
    scripts = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "export_q35_2b_document_summary_margin_revision_sft_v4",
            scripts
            / "export_q35_2b_document_summary_margin_revision_sft_v4.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _source_trace(
    tmp_path: Path, task_type: str = "DocumentSummaryWorkerTask"
) -> Path:
    trace = {
        "id": "failed-but-authentic-summary-probe",
        "task": {"type": task_type, "data": {}},
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


def _commit_revision_traces(tmp_path: Path) -> list[Path]:
    module = _commit_revision_module()
    fixture = module._load_fixture_module()
    document, _ = fixture.build_fixture()
    drafts = _text_revision_module().DRAFTS
    tools = [
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
    ]
    paths = []
    for chapter in document["chapters"]:
        chapter_id = chapter["id"]
        draft = drafts[chapter_id]
        budget = int(sum(len(row["text"].split()) for row in chapter["paragraphs"]) * 0.8)
        trace = {
            "id": f"observed-{chapter_id}-revision",
            "task": {"type": "DocumentSummaryTextTask", "data": {}},
            "nodes": [
                {
                    "parent": None,
                    "message": {
                        "role": "user",
                        "content": "Prime Agent runtime contract",
                    },
                },
                {
                    "parent": 0,
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": fixture.render_text_summary_prompt(chapter),
                            }
                        ],
                    },
                },
                {
                    "parent": 1,
                    "message": {
                        "role": "assistant",
                        "content": draft,
                        "reasoning_content": "first-turn reasoning",
                    },
                },
                {
                    "parent": 2,
                    "message": {
                        "role": "user",
                        "content": module._expected_feedback(
                            fixture=fixture,
                            draft=draft,
                            word_budget=budget,
                        ),
                    },
                },
                {
                    "parent": 3,
                    "message": {"role": "assistant", "content": draft},
                },
            ],
            "tools": tools,
        }
        path = tmp_path / f"{chapter_id}.jsonl"
        path.write_text(json.dumps({"traces": [trace]}) + "\n")
        paths.append(path)
    return paths


def _margin_revision_traces(tmp_path: Path) -> list[Path]:
    module = _margin_revision_module()
    fixture = module._load_fixture_module()
    document, _ = fixture.build_fixture()
    drafts = _text_revision_module().DRAFTS
    tools = [
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
    ]
    paths = []
    for chapter in document["chapters"]:
        chapter_id = chapter["id"]
        draft = drafts[chapter_id]
        budget = int(
            sum(len(row["text"].split()) for row in chapter["paragraphs"])
            * 0.8
        )
        trace = {
            "id": f"observed-{chapter_id}-margin-revision",
            "task": {"type": "DocumentSummaryTextTask", "data": {}},
            "nodes": [
                {
                    "parent": None,
                    "message": {
                        "role": "user",
                        "content": "Prime Agent runtime contract",
                    },
                },
                {
                    "parent": 0,
                    "message": {
                        "role": "user",
                        "content": fixture.render_text_summary_prompt(chapter),
                    },
                },
                {
                    "parent": 1,
                    "sampled": True,
                    "message": {
                        "role": "assistant",
                        "content": draft,
                        "reasoning_content": "untrained first-turn reasoning",
                    },
                },
                {
                    "parent": 1,
                    "sampled": False,
                    "message": {
                        "role": "assistant",
                        "content": draft,
                    },
                },
                {
                    "parent": 3,
                    "message": {
                        "role": "user",
                        "content": module._expected_feedback(
                            fixture=fixture,
                            draft=draft,
                            word_budget=budget,
                        ),
                    },
                },
                {
                    "parent": 4,
                    "message": {"role": "assistant", "content": draft},
                },
            ],
            "tools": tools,
        }
        path = tmp_path / f"{chapter_id}-margin.jsonl"
        path.write_text(json.dumps({"traces": [trace]}) + "\n")
        paths.append(path)
    return paths


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
    assert manifest["surgical_single_bullet_repair"] is True
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
        assert "verbatim source copying" not in messages[1]["content"]
        assert "There is no parent receiver" in messages[1]["content"]
        assert "do not call agent_message" in messages[1]["content"]
        assert "Do not edit the job" in messages[1]["content"]
        assert "update only the required worker-report.json" in messages[1]["content"]
        assert "I am the terminal worker" in messages[2]["reasoning_content"]
        assert "will not message a parent" in messages[2]["reasoning_content"]
        write_code = json.loads(
            messages[4]["tool_calls"][0]["function"]["arguments"]
        )["code"]
        assert "report['bullets'][repair_index] =" in write_code
        assert "if i != repair_index" in write_code
        assert "set(ordered_ids) == expected_ids" in write_code
        assert "len(ordered_ids) == len(expected_ids)" in write_code
        assert messages[6]["tool_calls"] == []

        tool_payload = messages[3]["content"]
        missing = messages[1]["content"].split("missing paragraph coverage: ", 1)[1]
        assert "report" in tool_payload
        assert missing.startswith("['")


def test_training_runner_accepts_summary_repair_contract() -> None:
    module = _runner_module()

    assert module.DATASET_CONTRACTS[
        "qwen35-2b-document-summary-worker-repair-sft/v2"
    ] == ("child", "grounded_english_chapter_summary_gate_surgical_repair")
    assert (
        module.DATASET_ANSWER_FREE[
            "qwen35-2b-document-summary-worker-repair-sft/v2"
        ]
        is False
    )
    assert module.DATASET_ROWS["qwen35-2b-document-summary-worker-repair-sft/v2"] == 12
    assert module.DATASET_CONTRACTS[
        "qwen35-2b-document-summary-worker-repair-sft/v1"
    ] == ("child", "grounded_english_chapter_summary_gate_repair")


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


def test_text_revision_export_masks_failure_context_and_balances_cases(
    tmp_path: Path,
) -> None:
    module = _text_revision_module()
    output = tmp_path / "text-revision-dataset"
    manifest = module.export(
        traces=[_source_trace(tmp_path, "DocumentSummaryTextTask")],
        output_dir=output,
    )
    rows = Dataset.from_parquet(str(output / "train.parquet"))

    assert manifest["rows"] == 12
    assert manifest["family_counts"] == {
        "summary_text_revision_exceptions": 4,
        "summary_text_revision_operations": 4,
        "summary_text_revision_scope": 4,
    }
    assert manifest["case_kind_counts"] == {
        "already_compliant": 3,
        "looser_budget": 3,
        "observed_budget": 3,
        "tighter_qualification": 3,
    }
    assert manifest["assistant_target_messages_per_row"] == 1
    assert manifest["context_assistant_messages_per_row"] == 0
    assert manifest["failed_reasoning_tokens_in_targets"] is False
    assert manifest["native_prime_agent_context"] is True
    assert manifest["prime_agent_tools_available"] is True
    assert manifest["fresh_confirmation_documents_reserved"] is True
    assert _runner_module()._validated_dataset(output) == manifest

    for row in rows:
        messages = row["messages"]
        assistants = [message for message in messages if message["role"] == "assistant"]
        assert len(messages) == 3
        assert len(assistants) == 1
        assert assistants[0]["content"].startswith("* ")
        assert assistants[0].get("reasoning_content") in (None, "")
        assert assistants[0].get("tool_calls") == []
        assert "Previous draft:" in messages[1]["content"]
        assert "Answer immediately" in messages[1]["content"]
        assert "Do not count words out loud" in messages[1]["content"]

    for record in manifest["case_records"]:
        assert record["target_words"] <= record["budget"]
        assert record["target_words"] <= record["source_budget"]
        assert record["expects_change"] is (
            record["kind"] != "already_compliant"
        )
        assert (record["draft_words"] > record["budget"]) is record[
            "expects_change"
        ]


def test_training_runner_accepts_text_revision_contract() -> None:
    module = _runner_module()
    schema = "qwen35-2b-document-summary-text-revision-sft/v1"

    assert module.DATASET_CONTRACTS[schema] == (
        "child",
        "grounded_english_chapter_summary_constrained_revision",
    )
    assert module.DATASET_ANSWER_FREE[schema] is False
    assert module.DATASET_ROWS[schema] == 12
    assert module.DATASET_BATCH_SIZES[schema] == 12


def test_live_revision_export_matches_the_real_role_sequence_and_masks_context(
    tmp_path: Path,
) -> None:
    module = _live_revision_module()
    output = tmp_path / "live-revision-dataset"
    manifest = module.export(
        traces=[_source_trace(tmp_path, "DocumentSummaryTextTask")],
        output_dir=output,
    )
    rows = Dataset.from_parquet(str(output / "train.parquet"))

    assert manifest["rows"] == 12
    assert manifest["distinct_conversation_payloads"] == 3
    assert manifest["repetitions_per_chapter"] == 4
    assert manifest["broad_skill_claim"] is False
    assert manifest["completion_boundary_alignment"] == "renderer_common_prefix_v1"
    assert manifest["live_transfer_check_required"] is True
    assert manifest["live_revision_role_sequence"] == [
        "runtime_user",
        "task_user",
        "assistant_draft_context",
        "revision_feedback_user",
        "assistant_corrected_target",
    ]
    assert _runner_module()._validated_dataset(output) == manifest

    for row in rows:
        messages = row["messages"]
        assert [message["role"] for message in messages] == [
            "user",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert messages[2]["trainable"] is False
        assert messages[4]["trainable"] is True
        assert messages[4]["mask_generation_prompt"] is True
        assert "Summarize the chapter below" in messages[1]["content"]
        assert "Your draft has" in messages[3]["content"]
        assert "Do not call tools" in messages[3]["content"]
        assert messages[4]["content"].startswith("* ")

    operations = next(
        row for row in rows if row["family"] == "summary_live_revision_operations"
    )
    target = operations["messages"][-1]["content"]
    assert "P0, P1 or P2" in target
    assert "P0 immediately pages" in target
    assert "receiving owners confirm in the ticket system" in target


def test_training_runner_accepts_live_revision_contract() -> None:
    module = _runner_module()
    schema = "qwen35-2b-document-summary-live-revision-sft/v2"

    assert module.DATASET_CONTRACTS[schema] == (
        "child",
        "grounded_english_chapter_summary_live_prefix_revision",
    )
    assert module.DATASET_ANSWER_FREE[schema] is False
    assert module.DATASET_ROWS[schema] == 12
    assert module.DATASET_BATCH_SIZES[schema] == 12
    config = module.training_config(
        run_name="live-revision",
        model_path=Path("/models/summary"),
        dataset_dir=Path("/data/live-revision"),
        output_root=Path("/outputs"),
        learning_rate=2e-7,
        enable_thinking=True,
    )
    assert "enable_thinking = true" in config


def test_commit_revision_export_uses_observed_scaffold_prefix_and_masks_context(
    tmp_path: Path,
) -> None:
    module = _commit_revision_module()
    output = tmp_path / "commit-revision-dataset"
    manifest = module.export(
        traces=_commit_revision_traces(tmp_path), output_dir=output
    )
    rows = Dataset.from_parquet(str(output / "train.parquet"))

    assert manifest["schema_version"] == module.SCHEMA_VERSION
    assert manifest["rows"] == 12
    assert manifest["observed_live_draft_context"] is True
    assert manifest["renderer_enable_thinking"] is False
    assert manifest["live_prior_assistant_reasoning_stripped_by_scaffold"] is True
    assert len(manifest["source_traces"]) == 3
    assert _runner_module()._validated_dataset(output) == manifest
    for row in rows:
        messages = row["messages"]
        assert [message["role"] for message in messages] == [
            "user",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert messages[2]["trainable"] is False
        assert messages[2].get("reasoning_content") is None
        assert messages[4]["trainable"] is True
        assert messages[4]["mask_generation_prompt"] is True
        assert "Return exactly 4 bullets" in messages[3]["content"]


def test_training_runner_accepts_non_thinking_commit_revision_contract() -> None:
    module = _runner_module()
    schema = "qwen35-2b-document-summary-commit-revision-sft/v3"

    assert module.DATASET_CONTRACTS[schema] == (
        "child",
        "grounded_english_chapter_summary_scaffold_aligned_commit_revision",
    )
    assert module.DATASET_ANSWER_FREE[schema] is False
    assert module.DATASET_ROWS[schema] == 12
    assert module.DATASET_BATCH_SIZES[schema] == 12
    config = module.training_config(
        run_name="commit-revision",
        model_path=Path("/models/summary"),
        dataset_dir=Path("/data/commit-revision"),
        output_root=Path("/outputs"),
        learning_rate=2e-7,
        enable_thinking=False,
    )
    assert "enable_thinking = false" in config


def test_margin_revision_export_uses_current_scaffold_and_source_checkpoint(
    tmp_path: Path,
) -> None:
    module = _margin_revision_module()
    source_model = tmp_path / "source-model"
    source_model.mkdir()
    (source_model / "model.safetensors").write_bytes(b"current-development-model")
    (source_model / "STABLE").write_text("stable\n")
    output = tmp_path / "margin-revision-dataset"
    manifest = module.export(
        traces=_margin_revision_traces(tmp_path),
        source_model=source_model,
        output_dir=output,
    )
    rows = Dataset.from_parquet(str(output / "train.parquet"))

    assert manifest["schema_version"] == module.SCHEMA_VERSION
    assert manifest["rows"] == 12
    assert manifest["feedback_safety_margin_words"] == 3
    assert manifest["feedback_contract"] == (
        "commit_once_with_three_word_safety_margin_v1"
    )
    assert manifest["observed_current_revision_outputs_trainable"] is False
    assert manifest["on_policy_development_failures_only"] is True
    assert len(manifest["on_policy_source_model_sha256"]) == 64
    assert _runner_module()._validated_dataset(output) == manifest
    for row in rows:
        messages = row["messages"]
        assert messages[2]["trainable"] is False
        assert messages[4]["trainable"] is True
        assert "Leave a three-word safety margin" in messages[3]["content"]
        chapter_id = row["family"].removeprefix("summary_margin_revision_")
        target = manifest["feedback_target_words_by_chapter"][chapter_id]
        assert f"return no more than {target} words" in messages[3]["content"]


def test_training_runner_accepts_margin_revision_contract() -> None:
    module = _runner_module()
    schema = "qwen35-2b-document-summary-margin-revision-sft/v4"

    assert module.DATASET_CONTRACTS[schema] == (
        "child",
        "grounded_english_chapter_summary_on_policy_margin_commit_revision",
    )
    assert module.DATASET_ANSWER_FREE[schema] is False
    assert module.DATASET_ROWS[schema] == 12
    assert module.DATASET_BATCH_SIZES[schema] == 12


def test_live_revision_training_requires_a_matching_renderer_audit(
    tmp_path: Path,
) -> None:
    module = _runner_module()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "MANIFEST.json").write_text("{}")
    (dataset / "train.parquet").write_bytes(b"parquet")

    try:
        module._validated_renderer_audit(
            dataset,
            {"schema_version": "qwen35-2b-document-summary-live-revision-sft/v2"},
        )
    except ValueError as error:
        assert "missing live revision renderer audit" in str(error)
    else:
        raise AssertionError("missing renderer audit was accepted")


def test_text_revision_training_wrapper_defaults_to_one_update() -> None:
    wrapper = (
        Path(__file__).parents[2]
        / "scripts/run_q35_2b_document_summary_text_revision_sft_v1.sh"
    ).read_text()

    assert "optimizer_updates=${4:-1}" in wrapper
    assert "--learning-rate 2e-7" in wrapper
    assert '--optimizer-updates "$optimizer_updates"' in wrapper


def test_live_revision_training_wrapper_is_bounded_and_thinking_explicit() -> None:
    wrapper = (
        Path(__file__).parents[2]
        / "scripts/run_q35_2b_document_summary_live_revision_sft_v2.sh"
    ).read_text()

    assert "optimizer_updates=${4:-1}" in wrapper
    assert "--learning-rate 2e-7" in wrapper
    assert '--optimizer-updates "$optimizer_updates"' in wrapper
    assert "--enable-thinking" in wrapper
    assert "RENDERER-AUDIT.json" in wrapper
    assert 'export PYTHONPATH="$root/src:$root/scripts' in wrapper


def test_live_revision_renderer_audit_checks_exact_completion_suffix() -> None:
    audit = (
        Path(__file__).parents[2]
        / "scripts/audit_q35_2b_document_summary_live_revision_renderer_v2.py"
    ).read_text()

    assert "live_prompt.token_ids == generation_prompt.token_ids" in audit
    assert "trainable_ids != expected_completion" in audit
    assert "prior assistant draft contributes to SFT loss" in audit
    assert "for enable_thinking in (False, True)" in audit
    assert '"selected_enable_thinking": True' in audit


def test_commit_revision_renderer_audit_requires_exact_stripped_live_prefix() -> None:
    audit = (
        Path(__file__).parents[2]
        / "scripts/audit_q35_2b_document_summary_commit_revision_renderer_v3.py"
    ).read_text()

    assert "stripped_prompt.token_ids != generation_prompt.token_ids" in audit
    assert "raw_prompt.token_ids == prompt_ids" in audit
    assert "trainable_ids != expected_completion" in audit
    assert "prior assistant draft contributes to SFT loss" in audit
    assert '"selected_enable_thinking": False' in audit


def test_margin_revision_renderer_audit_uses_shared_exact_prefix_checks() -> None:
    audit = (
        Path(__file__).parents[2]
        / "scripts/audit_q35_2b_document_summary_margin_revision_renderer_v4.py"
    ).read_text()

    assert "audit_commit_revision" in audit
    assert "observed_cases_fn=_observed_cases" in audit
    assert 'family_prefix="summary_margin_revision"' in audit
    assert 'result["on_policy_margin_prefix_verified"] = True' in audit


def test_commit_revision_training_wrapper_is_bounded_and_non_thinking() -> None:
    wrapper = (
        Path(__file__).parents[2]
        / "scripts/run_q35_2b_document_summary_commit_revision_sft_v3.sh"
    ).read_text()

    assert "optimizer_updates=${6:-1}" in wrapper
    assert '--optimizer-updates "$optimizer_updates"' in wrapper
    assert "--enable-thinking" not in wrapper
    assert wrapper.count('--traces "$') == 6
    assert "RENDERER-AUDIT.json" in wrapper
    assert 'export PYTHONPATH="$root/src:$root/scripts' in wrapper


def test_margin_revision_training_wrapper_uses_four_bounded_updates() -> None:
    wrapper = (
        Path(__file__).parents[2]
        / "scripts/run_q35_2b_document_summary_margin_revision_sft_v4.sh"
    ).read_text()

    assert "optimizer_updates=${6:-4}" in wrapper
    assert "--learning-rate 2e-7" in wrapper
    assert '--optimizer-updates "$optimizer_updates"' in wrapper
    assert "--enable-thinking" not in wrapper
    assert wrapper.count('--traces "$') == 6
    assert '--source-model "$source_model"' in wrapper
    assert "RENDERER-AUDIT.json" in wrapper


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
