import importlib.util
import json
import sys
from collections import Counter
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


def test_causal_contrast_pools_follow_registry_not_fixed_identity() -> None:
    module = _module("export_q35_2b_specialist_router_contrast_sft_v1")
    pools = module._candidate_rows(_runtime())
    rows = [row for values in pools.values() for row in values]

    assert {key: len(values) for key, values in pools.items()} == {
        key: module.ROWS_PER_EXPERT for key in module.EXPERT_IDS
    }
    assert Counter(row["contrast_group"] for row in rows) == Counter(
        {index: 6 for index in range(module.BASE_GROUPS)}
    )
    for row in rows:
        profile_map = row["profile_by_expert_id"]
        assert profile_map[row["expert_id"]] == row["required_capability_profile"]
        target = json.loads(
            row["messages"][-1]["tool_calls"][0]["function"]["arguments"]
        )
        assert target == {"expert_id": row["expert_id"]}
        assert "action" not in row

    for group in range(module.BASE_GROUPS):
        group_rows = [row for row in rows if row["contrast_group"] == group]
        assert Counter(row["expert_id"] for row in group_rows) == Counter(
            {expert_id: 2 for expert_id in module.EXPERT_IDS}
        )
        assert len(
            {
                tuple(sorted(row["profile_by_expert_id"].items()))
                for row in group_rows
            }
        ) == 6


def test_contrast_curriculum_first_half_covers_each_group_per_target() -> None:
    module = _module("export_q35_2b_specialist_router_contrast_sft_v1")
    pools = module._candidate_rows(_runtime())
    rows = [
        pools[expert_id][index]
        for index in range(module.ROWS_PER_EXPERT)
        for expert_id in module.EXPERT_IDS
    ]
    first_half = rows[:48]
    assert Counter(row["expert_id"] for row in first_half) == Counter(
        {expert_id: 16 for expert_id in module.EXPERT_IDS}
    )
    assert Counter(row["contrast_group"] for row in first_half) == Counter(
        {index: 3 for index in range(module.BASE_GROUPS)}
    )
    for start in range(0, 48, 12):
        assert Counter(
            row["expert_id"] for row in first_half[start : start + 12]
        ) == Counter({expert_id: 4 for expert_id in module.EXPERT_IDS})


def test_expert_runner_accepts_only_complete_contrast_manifest(tmp_path: Path) -> None:
    exporter = _module("export_q35_2b_specialist_router_contrast_sft_v1")
    runner = _module("run_q35_2b_specialist_expert_sft_v1")
    parquet = tmp_path / "train.parquet"
    parquet.write_bytes(b"immutable causal contrast corpus")
    manifest = {
        "schema_version": exporter.SCHEMA_VERSION,
        "status": "complete",
        "role": "specialist_router",
        "objective": exporter.OBJECTIVE,
        "rows": 96,
        "training_batch_size": 12,
        "expert_counts": {expert_id: 32 for expert_id in exporter.EXPERT_IDS},
        "role_counts": {
            expert_id: {"root": 20, "nonroot_specialist_manager": 12}
            for expert_id in exporter.EXPERT_IDS
        },
        "first_batch_expert_counts": {
            expert_id: 4 for expert_id in exporter.EXPERT_IDS
        },
        "first_half_expert_counts": {
            expert_id: 16 for expert_id in exporter.EXPERT_IDS
        },
        "base_assignment_groups": 16,
        "registry_permutations_per_group": 6,
        "causal_registry_permutations": True,
        "training_instance_offset": 37900,
        "training_template_variants": [0, 1, 2, 3],
        "heldout_template_variants_excluded": [4, 5],
        "observed_instance_offsets_excluded": [
            35100,
            37100,
            37200,
            37300,
            37400,
            37500,
            37700,
        ],
        "answer_free": True,
        "public_capability_registry_only": True,
        "expert_only_tool_arguments": True,
        "cognitive_action_labels_present": False,
        "tool_call_format": "openai_function_v1",
        "dataset": {
            "path": parquet.name,
            "sha256": runner.sha256_file(parquet),
        },
    }
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert runner._validated_dataset(tmp_path)["rows"] == 96

    manifest["causal_registry_permutations"] = False
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    try:
        runner._validated_dataset(tmp_path)
    except ValueError as error:
        assert "invalid specialist expert dataset" in str(error)
    else:
        raise AssertionError("runner accepted a non-causal contrast manifest")
