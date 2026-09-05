import importlib.util
import json
import sys
import tomllib
from pathlib import Path

from datasets import Dataset


def _module():
    scripts = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "export_q35_2b_specialist_worker_sft_v1.py"
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _runner():
    scripts = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "run_q35_2b_specialist_worker_sft_v1.py"
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _remedial_module():
    scripts = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "export_q35_2b_source_worker_remedial_sft_v1.py"
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _remedial_runner():
    scripts = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "run_q35_2b_source_worker_remedial_sft_v1.py"
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _source(path: Path) -> None:
    trace = {
        "id": "latent-specialist-screen",
        "nodes": [
            {
                "sampled": False,
                "parent": None,
                "message": {
                    "role": "user",
                    "content": (
                        "Prime runtime\nRecursive agent depth: 1\nYou are a child agent spawned by a coordinator."
                    ),
                },
            }
        ],
        "tools": [
            {
                "name": "ipython",
                "description": "execute Python",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            }
        ],
    }
    path.write_text(json.dumps({"traces": [trace]}) + "\n", encoding="utf-8")


def _tool_code(row: dict) -> str:
    message = row["messages"][2]
    arguments = message["tool_calls"][0]["function"]["arguments"]
    return json.loads(arguments)["code"]


def test_table_specialist_export_is_balanced_answer_free_and_excludes_heldout(
    tmp_path: Path,
) -> None:
    module = _module()
    source = tmp_path / "traces.jsonl"
    output = tmp_path / "table"
    _source(source)

    manifest = module.export(
        traces=[source],
        output_dir=output,
        expert_id="table_analyst",
        instances_per_variant=2,
        instance_offset=36000,
    )
    rows = list(Dataset.from_parquet(str(output / "train.parquet")))

    assert manifest["rows"] == 16
    assert manifest["family_counts"] == {"table_join": 8, "table_reconcile": 8}
    assert manifest["answer_free"] is True
    assert manifest["heldout_template_variants_excluded"] == [4, 5]
    assert all("-v4-" not in row["task_key"] and "-v5-" not in row["task_key"] for row in rows)
    assert all("expert_id=table_analyst" in row["messages"][1]["content"] for row in rows)
    join_code = _tool_code(next(row for row in rows if row["family"] == "specialist_table_join"))
    assert ".map(rates)" in join_code
    assert ".eq('posted')" in join_code
    assert "json.dumps({'value': value}" in join_code
    assert "833" not in join_code


def test_source_specialist_targets_ast_and_config_operations(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "traces.jsonl"
    output = tmp_path / "source"
    _source(source)

    manifest = module.export(
        traces=[source],
        output_dir=output,
        expert_id="source_inspector",
        instances_per_variant=1,
        instance_offset=36100,
    )
    rows = list(Dataset.from_parquet(str(output / "train.parquet")))

    assert manifest["rows"] == 8
    assert manifest["family_counts"] == {"source_ast": 4, "source_config": 4}
    ast_code = _tool_code(next(row for row in rows if row["family"] == "specialist_source_ast"))
    config_code = _tool_code(next(row for row in rows if row["family"] == "specialist_source_config"))
    assert "ast.walk" in ast_code
    assert "ast.AsyncFunctionDef" in ast_code
    assert "tomllib.loads" in config_code
    assert "value == 'true'" in config_code
    assert all(row["role"] == "child" for row in rows)
    assert all(row["objective"] == module.OBJECTIVE for row in rows)


def test_specialist_runner_validates_isolated_lineage_contract(tmp_path: Path) -> None:
    exporter = _module()
    runner = _runner()
    source = tmp_path / "traces.jsonl"
    output = tmp_path / "table"
    _source(source)
    exporter.export(
        traces=[source],
        output_dir=output,
        expert_id="table_analyst",
        instances_per_variant=1,
        instance_offset=36200,
    )

    manifest = runner._validated_dataset(output, "table_analyst")
    config = runner.training_config(
        run_name="table-specialist-v1",
        model_path=Path("/models/H176"),
        dataset_dir=output,
        output_root=Path("/outputs/specialists"),
        learning_rate=2e-6,
        optimizer_updates=4,
        batch_size=16,
    )

    assert manifest["expert_id"] == "table_analyst"
    assert "max_steps = 4" in config
    assert 'name = "table-specialist-v1"' in config
    assert "lora" not in config.lower()

    bad = json.loads((output / "MANIFEST.json").read_text())
    bad["heldout_template_variants_excluded"] = [5]
    (output / "MANIFEST.json").write_text(json.dumps(bad), encoding="utf-8")
    try:
        runner._validated_dataset(output, "table_analyst")
    except ValueError as error:
        assert "invalid table_analyst" in str(error)
    else:
        raise AssertionError("runner accepted a held-out-contaminated specialist corpus")


def test_source_remedial_curriculum_leaks_then_restores_live_shape(
    tmp_path: Path,
) -> None:
    exporter = _module()
    remedial = _remedial_module()
    source = tmp_path / "traces.jsonl"
    base = tmp_path / "source-base"
    output = tmp_path / "source-remedial"
    _source(source)
    exporter.export(
        traces=[source],
        output_dir=base,
        expert_id="source_inspector",
        instances_per_variant=1,
        instance_offset=36100,
    )

    manifest = remedial.export(
        base_dataset_dir=base,
        output_dir=output,
        instances_per_variant=1,
        instance_offset=60000,
    )
    rows = list(Dataset.from_parquet(str(output / "train.parquet")))

    assert manifest["rows"] == 16
    assert manifest["family_counts"] == {"source_ast": 8, "source_config": 8}
    assert manifest["phase_counts"] == {"procedure_leak": 8, "live_shape": 8}
    assert manifest["curriculum_phase_order"] == ["procedure_leak", "live_shape"]
    assert [row["training_phase"] for row in rows[:8]] == ["procedure_leak"] * 8
    assert [row["training_phase"] for row in rows[8:]] == ["live_shape"] * 8
    assert all("-v4-" not in row["task_key"] and "-v5-" not in row["task_key"] for row in rows)
    assert all("answer" not in row for row in rows)
    assert all("agent_message.send" in _tool_code(row) for row in rows)
    assert all("json.dumps({'value': value}" in _tool_code(row) for row in rows)

    leaked = next(row for row in rows if row["training_phase"] == "procedure_leak")
    live = next(row for row in rows if row["training_phase"] == "live_shape")
    assert "[training-only first-call procedure leak]" in leaked["messages"][1]["content"]
    assert "[training-only first-call procedure leak]" not in live["messages"][1]["content"]
    ast_code = _tool_code(next(row for row in rows if row["family"] == "specialist_source_ast"))
    config_code = _tool_code(next(row for row in rows if row["family"] == "specialist_source_config"))
    assert "ast.walk(ast.parse(path.read_text()))" in ast_code
    assert "bool(node.decorator_list)" in ast_code
    assert "tomllib.loads" in config_code
    assert "config['runtime']['workers']" in config_code
    assert "line.split('=', 1)" in config_code


def test_source_remedial_runner_freezes_curriculum_and_one_epoch(
    tmp_path: Path,
) -> None:
    exporter = _module()
    remedial = _remedial_module()
    runner = _remedial_runner()
    source = tmp_path / "traces.jsonl"
    base = tmp_path / "source-base"
    output = tmp_path / "source-remedial"
    _source(source)
    exporter.export(
        traces=[source],
        output_dir=base,
        expert_id="source_inspector",
        instances_per_variant=1,
        instance_offset=36100,
    )
    remedial.export(base_dataset_dir=base, output_dir=output)
    manifest_sha = runner.sha256_file(output / "MANIFEST.json")
    parquet_sha = runner.sha256_file(output / "train.parquet")

    manifest = runner._validated_dataset(
        output,
        expected_manifest_sha256=manifest_sha,
        expected_parquet_sha256=parquet_sha,
    )
    config = runner.training_config(
        run_name="source-remedial-v1",
        model_path=Path("/models/H-source-s2"),
        dataset_dir=output,
        output_root=Path("/outputs/source-remedial"),
        learning_rate=2e-6,
        optimizer_updates=8,
        batch_size=16,
    )

    assert manifest["rows"] == 128
    assert manifest["phase_counts"] == {"procedure_leak": 64, "live_shape": 64}
    assert "max_steps = 8" in config
    assert "batch_size = 16" in config
    assert "shuffle = false" in config
    assert "lora" not in config.lower()

    bad = json.loads((output / "MANIFEST.json").read_text())
    bad["heldout_tasks_or_values_used"] = True
    (output / "MANIFEST.json").write_text(json.dumps(bad), encoding="utf-8")
    try:
        runner._validated_dataset(
            output,
            expected_manifest_sha256=runner.sha256_file(output / "MANIFEST.json"),
            expected_parquet_sha256=parquet_sha,
        )
    except ValueError as error:
        assert "invalid source-worker remedial dataset" in str(error)
    else:
        raise AssertionError("runner accepted heldout-contaminated remedial metadata")


def test_source_remedial_s5_assets_freeze_three_arm_rung() -> None:
    root = Path(__file__).parents[2]
    experiment = root / "experiments/qwen35-2b-document-recursion-zero-update-v1"
    config = tomllib.loads(
        (experiment / "specialist-source-competence-s5-remedial-heldout-v1.toml").read_text()
    )
    sampling = json.loads(
        (
            experiment
            / "specialist-source-competence-s5-remedial-sampling-contract-v1.json"
        ).read_text()
    )
    plan = json.loads(
        (
            experiment
            / "specialist-source-competence-s5-remedial-curriculum-v1-plan.json"
        ).read_text()
    )
    materializer = (
        root / "scripts/materialize_q35_2b_source_worker_remedial_s5_v1.sh"
    ).read_text()
    launcher = (
        root / "scripts/run_q35_2b_source_worker_competence_s5_remedial_v1.sh"
    ).read_text()

    assert config["num_tasks"] == 16
    assert config["num_rollouts"] == 1
    assert config["shuffle"] is False
    assert config["env"]["taskset"]["families"] == [
        "specialist_source_ast",
        "specialist_source_config",
    ]
    assert config["env"]["taskset"]["instances_per_template"] == 4
    assert config["env"]["taskset"]["instance_offset"] == 61000
    assert config["env"]["taskset"]["seed"] == 20270908
    assert config["sampling"] == {
        "seed": 20270908,
        **sampling["environment_sampling"],
    }
    assert sampling["route_policy"]["worker_computation_scaffolded"] is False
    assert sampling["route_policy"]["worker_parent_send_scaffolded"] is False
    assert plan["admission"]["final_thresholds"] == {
        "forced_worker_activations_each_arm": 16,
        "minimum_treatment_hard_successes": 4,
        "minimum_hard_successes_per_family": 2,
        "minimum_paired_h176_fail_to_candidate_success_recoveries": 4,
        "maximum_paired_regressions": 0,
    }
    assert plan["admission"]["acceptance_gates_relaxed"] is False
    assert plan["status"] == "root_frozen_launch_ready"
    assert (
        plan["training_corpus"]["manifest_sha256"]
        == "d406639dcc0d172e13c134328f5926bf30caabd17b16ad3c4d76887585f7ba82"
    )
    assert (
        plan["training_corpus"]["train_parquet_sha256"]
        == "9b30b54bd1dc25feece22f0c86f6493cb4f2ab4adec3db39ff33ed99c813306c"
    )
    assert (
        plan["paired_evaluation"]["heldout_task_bank_sha256"]
        == "9be882279cee4e558094c41f7a7c160f425a78aaf28ea93fc98ee6ec345ecf66"
    )
    assert plan["training_corpus"]["manifest_sha256"] in launcher
    assert plan["training_corpus"]["train_parquet_sha256"] in launcher
    assert plan["paired_evaluation"]["heldout_task_bank_sha256"] in launcher
    assert 'treatment.get("hard_successes") != 0' in launcher
    assert 'task.get("clean_protocol_aligned") == 1' in launcher
    assert "--minimum-treatment-hard-successes 4" in launcher
    assert "--minimum-hard-successes-per-family 2" in launcher
    assert "--minimum-paired-recoveries 4" in launcher
    assert "--maximum-paired-regressions 0" in launcher
    assert launcher.index("$control_label") < launcher.index("$bridge_label")
    assert launcher.index("$bridge_label") < launcher.index("scripts/run_q35_2b_source_worker_remedial_sft_v1.py")
    assert launcher.index("scripts/run_q35_2b_source_worker_remedial_sft_v1.py") < launcher.index("$treatment_label")
    assert "refusing duplicate or partial S5 result/output root" in launcher
    assert "refusing to overwrite S5 remedial dataset" in materializer
