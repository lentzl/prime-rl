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


def test_expert_only_pools_balance_identity_and_natural_roles() -> None:
    module = _module("export_q35_2b_specialist_expert_sft_v1")
    pools = module._candidate_rows(_runtime())

    assert set(pools) == set(module.EXPERT_IDS)
    assert {key: len(rows) for key, rows in pools.items()} == {
        key: module.ROWS_PER_EXPERT for key in module.EXPERT_IDS
    }
    assert {row["role_scope"] for row in pools["generic_worker"]} == {"root"}
    for expert_id in ("table_analyst", "source_inspector"):
        assert sum(row["role_scope"] == "root" for row in pools[expert_id]) == 8
        assert (
            sum(
                row["role_scope"] == "nonroot_specialist_manager"
                for row in pools[expert_id]
            )
            == 8
        )
    for expert_id, rows in pools.items():
        for row in rows:
            target = row["messages"][-1]["tool_calls"][0]["function"]
            assert target["name"] == "select_expert"
            assert json.loads(target["arguments"]) == {"expert_id": expert_id}
            assert "action" not in row
            assert "action" not in target["arguments"]


def test_expert_only_runner_validates_frozen_manifest(tmp_path: Path) -> None:
    exporter = _module("export_q35_2b_specialist_expert_sft_v1")
    runner = _module("run_q35_2b_specialist_expert_sft_v1")
    parquet = tmp_path / "train.parquet"
    parquet.write_bytes(b"immutable expert-only corpus")
    role_counts = {
        "generic_worker": {"root": 16, "nonroot_specialist_manager": 0},
        "table_analyst": {"root": 8, "nonroot_specialist_manager": 8},
        "source_inspector": {"root": 8, "nonroot_specialist_manager": 8},
    }
    manifest = {
        "schema_version": exporter.SCHEMA_VERSION,
        "status": "complete",
        "role": "coordinator",
        "objective": exporter.OBJECTIVE,
        "rows": exporter.ROWS,
        "training_batch_size": 12,
        "expert_counts": {
            key: exporter.ROWS_PER_EXPERT for key in exporter.EXPERT_IDS
        },
        "role_counts": role_counts,
        "first_batch_expert_counts": {key: 4 for key in exporter.EXPERT_IDS},
        "training_instance_offset": 37600,
        "training_template_variants": [0, 1, 2, 3],
        "heldout_template_variants_excluded": [4, 5],
        "observed_instance_offsets_excluded": [35100, 37100, 37200, 37300],
        "answer_free": True,
        "public_capability_registry_only": True,
        "expert_only_tool_arguments": True,
        "cognitive_action_labels_present": False,
        "root_and_nonroot_coordinator_rows": True,
        "tool_call_format": "openai_function_v1",
        "dataset": {
            "path": parquet.name,
            "sha256": runner.sha256_file(parquet),
        },
    }
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    validated = runner._validated_dataset(tmp_path)
    assert validated["rows"] == 48

    manifest["cognitive_action_labels_present"] = True
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    try:
        runner._validated_dataset(tmp_path)
    except ValueError as error:
        assert "invalid specialist expert dataset" in str(error)
    else:
        raise AssertionError("expert-only runner accepted action-label contamination")
