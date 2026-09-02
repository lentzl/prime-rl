import importlib.util
import json
import sys
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
