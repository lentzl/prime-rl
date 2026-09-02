import importlib.util
import json
import sys
from pathlib import Path


def _module(name: str):
    path = Path(__file__).parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _runtime() -> dict[int, dict[str, str]]:
    return {
        0: {"role": "user", "content": "You are the root. Recursive agent depth: 0"},
        1: {
            "role": "user",
            "content": "You are a child agent. Recursive agent depth: 1",
        },
    }


def test_specialist_routing_pools_cover_all_routes_and_both_roles() -> None:
    module = _module("export_q35_2b_specialist_routing_sft_v1")
    pools = module._candidate_rows(_runtime())

    assert set(pools) == set(module.ROUTE_CLASSES)
    assert {key: len(rows) for key, rows in pools.items()} == {
        key: module.ROWS_PER_ROUTE for key in module.ROUTE_CLASSES
    }
    assert all(
        row["expert_id"] == "source_inspector"
        for row in pools["delegate_terminal:source_inspector"]
    )
    assert all(
        row["expert_id"] == "table_analyst"
        for row in pools["delegate_terminal:table_analyst"]
    )
    for route_class in (
        "delegate_terminal:source_inspector",
        "delegate_terminal:table_analyst",
    ):
        assert {row["role_scope"] for row in pools[route_class]} == {
            "root",
            "nonroot_specialist_manager",
        }
    assert all(row["action"] == "solve_owned" for row in pools["solve_owned:none"])
    assert all(
        row["action"] == "delegate_coordinator"
        for row in pools["delegate_coordinator:none"]
    )

    for rows in pools.values():
        for row in rows:
            target = row["messages"][-1]["tool_calls"][0]["function"]
            assert target["name"] == "select_cognitive_action"
            assert json.loads(target["arguments"]) == {
                "action": row["action"],
                "expert_id": row["expert_id"],
            }
            assert "answer" not in row


def test_specialist_routing_runner_validates_frozen_manifest(tmp_path: Path) -> None:
    exporter = _module("export_q35_2b_specialist_routing_sft_v1")
    runner = _module("run_q35_2b_specialist_routing_sft_v1")
    parquet = tmp_path / "train.parquet"
    parquet.write_bytes(b"immutable routing corpus")
    first_batch = {key: 3 for key in exporter.ROUTE_CLASSES}
    first_batch[exporter.ROUTE_CLASSES[0]] = 4
    manifest = {
        "schema_version": exporter.SCHEMA_VERSION,
        "status": "complete",
        "role": "coordinator",
        "objective": exporter.OBJECTIVE,
        "rows": exporter.ROWS,
        "training_batch_size": 16,
        "route_counts": {
            key: exporter.ROWS_PER_ROUTE for key in exporter.ROUTE_CLASSES
        },
        "first_batch_route_counts": first_batch,
        "training_template_variants": [0, 1, 2, 3],
        "heldout_template_variants_excluded": [4, 5],
        "observed_instance_offset_excluded": 35100,
        "answer_free": True,
        "public_capability_registry_only": True,
        "action_and_expert_only_tool_arguments": True,
        "root_and_nonroot_coordinator_rows": True,
        "tool_call_format": "openai_function_v1",
        "dataset": {
            "path": parquet.name,
            "sha256": runner.sha256_file(parquet),
        },
    }
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    validated = runner._validated_dataset(tmp_path)
    assert validated["rows"] == 80

    manifest["heldout_template_variants_excluded"] = [4]
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    try:
        runner._validated_dataset(tmp_path)
    except ValueError as error:
        assert "invalid specialist routing dataset" in str(error)
    else:
        raise AssertionError("routing runner accepted a contaminated manifest")
