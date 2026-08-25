import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[2] / "scripts" / "q35_2b_spade_interaction_loop_v1.py"
    spec = importlib.util.spec_from_file_location("q35_2b_spade_interaction_loop_v1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _initialized(module):
    return {
        "kind": "initialized",
        "candidate": {
            "label": "R6Y3",
            "model": "q35-2b-r6y3",
            "model_revision": "revision",
            "base_sha256": "b" * 64,
            "adapter_sha256": "a" * 64,
        },
        "initial_targets": {
            "child": "e0c28_inline_only",
            "yield": "e0d3_uncapped_yield_exact_child",
        },
        "bank_policy": {
            "next_start_index": 4_008_100,
            "index_stride": 100,
            "tasks_per_bank": 6,
        },
        "invariants": {
            "minimum_complete_qualifying_trajectories_per_source": 4,
            "minimum_distinct_qualifying_task_keys_per_source": 4,
            "acceptance_floor_relaxed": False,
        },
        "event_sha256": "0" * 64,
    }


def _evaluation(*, track, phase, start_index, qualifiers, event_sha):
    return {
        "kind": "evaluation_recorded",
        "track": track,
        "phase": phase,
        "bank": {"id": f"{track}-{start_index}", "start_index": start_index},
        "admission": {
            "qualifying_trajectories": qualifiers,
            "distinct_task_keys": qualifiers,
            "gate_open": qualifiers >= 4,
        },
        "artifacts": {"summary_sha256": str(start_index).zfill(64)},
        "event_sha256": event_sha,
    }


def test_failed_strict_probes_increase_help_without_authorizing_training() -> None:
    module = _module()
    events = [
        _initialized(module),
        _evaluation(
            track="child",
            phase="e0c28_inline_only",
            start_index=4_008_100,
            qualifiers=0,
            event_sha="1" * 64,
        ),
        _evaluation(
            track="yield",
            phase="e0d3_uncapped_yield_exact_child",
            start_index=4_008_200,
            qualifiers=1,
            event_sha="2" * 64,
        ),
    ]

    status = module.project(events)

    assert status["status"] == "collecting"
    assert status["next_action"]["optimizer_steps_authorized"] == 0
    assert status["next_action"]["arms"] == [
        {
            "track": "child",
            "phase": "e0c275_inline_location",
            "reason": "increase_environment_help_after_failed_admission",
            "start_index": 4_008_300,
            "tasks": 6,
            "split": "train_gen",
            "optimizer_updates_during_collection": 0,
        },
        {
            "track": "yield",
            "phase": "e0d2_capped_yield_exact_child",
            "reason": "increase_environment_help_after_failed_admission",
            "start_index": 4_008_400,
            "tasks": 6,
            "split": "train_gen",
            "optimizer_updates_during_collection": 0,
        },
    ]


def test_two_admitted_sources_authorize_exactly_one_update_then_tighten() -> None:
    module = _module()
    events = [
        _initialized(module),
        _evaluation(
            track="child",
            phase="e0c275_inline_location",
            start_index=4_008_300,
            qualifiers=5,
            event_sha="1" * 64,
        ),
        _evaluation(
            track="yield",
            phase="e0d2_capped_yield_exact_child",
            start_index=4_008_400,
            qualifiers=4,
            event_sha="2" * 64,
        ),
    ]

    authorized = module.project(events)

    assert authorized["status"] == "training_authorized"
    assert authorized["next_action"]["optimizer_steps_authorized"] == 1
    assert authorized["next_action"]["failed_trajectory_rows_trainable"] is False

    events.append(
        {
            "kind": "update_recorded",
            "training_sources": authorized["next_action"]["sources"],
            "output_candidate": {
                **authorized["candidate"],
                "label": "R6Y4",
                "adapter_sha256": "c" * 64,
            },
            "event_sha256": "3" * 64,
        }
    )
    tightened = module.project(events)

    assert tightened["status"] == "collecting"
    assert tightened["cycle_targets"] == {
        "child": "e0c28_inline_only",
        "yield": "e0d3_uncapped_yield_exact_child",
    }
    assert [arm["phase"] for arm in tightened["next_action"]["arms"]] == [
        "e0c28_inline_only",
        "e0d3_uncapped_yield_exact_child",
    ]


def test_summary_validation_cannot_relax_four_trajectory_floor(tmp_path) -> None:
    module = _module()
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": module.SUMMARY_SCHEMA_VERSION,
                "phase": "e0c275_inline_location",
                "qualifying_trajectories": 3,
                "distinct_qualifying_task_keys": 3,
                "gate": {
                    "required_qualifying_trajectories": 3,
                    "required_distinct_task_keys": 3,
                    "acceptance_floor_relaxed": True,
                    "gradient_gate_open": True,
                },
            }
        )
    )

    with pytest.raises(ValueError, match="four-trajectory floor"):
        module._validated_summary(summary_path, phase="e0c275_inline_location")


def test_recorded_update_persists_verified_adapter_path(tmp_path) -> None:
    module = _module()
    initial = tmp_path / "initial"
    initial.mkdir()
    initial_weight = initial / "adapter_model.safetensors"
    initial_weight.write_bytes(b"initial")
    initial_sha = module._sha256_file(initial_weight)
    output = tmp_path / "output"
    output.mkdir()
    output_weight = output / "adapter_model.safetensors"
    output_weight.write_bytes(b"output")
    output_sha = module._sha256_file(output_weight)
    events_path = tmp_path / "events.jsonl"
    module._append_event(
        events_path,
        {
            "kind": "initialized",
            "candidate": {
                "label": "R6Y3",
                "model": "q35-2b-r6y3",
                "model_revision": "revision",
                "base_sha256": "b" * 64,
                "adapter_sha256": initial_sha,
                "adapter_path": str(initial),
            },
            "initial_targets": {
                "child": "e0c25_inline_evidence",
                "yield": "e0d2_capped_yield_exact_child",
            },
            "bank_policy": {"next_start_index": 100, "index_stride": 100, "tasks_per_bank": 6},
            "invariants": {},
        },
        create=True,
    )
    for track, phase, index in (
        ("child", "e0c25_inline_evidence", 100),
        ("yield", "e0d2_capped_yield_exact_child", 200),
    ):
        module._append_event(
            events_path,
            {
                "kind": "evaluation_recorded",
                "track": track,
                "phase": phase,
                "bank": {"id": track, "start_index": index},
                "admission": {
                    "qualifying_trajectories": 4,
                    "distinct_task_keys": 4,
                    "gate_open": True,
                },
                "artifacts": {"summary_sha256": str(index).zfill(64)},
            },
        )
    status = module.project(module._load_events(events_path))
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": module.UPDATE_SCHEMA_VERSION,
                "initial_adapter_sha256": initial_sha,
                "base_sha256_before": "b" * 64,
                "base_sha256_after": "b" * 64,
                "optimizer_steps": 1,
                "dense_base_updates": 0,
                "failed_trajectory_rows": 0,
                "output_adapter_sha256": output_sha,
                "output_adapter_path": str(output),
                "output_candidate_label": "R6Y4",
                "output_model": "q35-2b-r6y4",
                "source_summary_sha256": {
                    track: source["summary_sha256"]
                    for track, source in status["next_action"]["sources"].items()
                },
            }
        )
    )
    args = type("Args", (), {"events": events_path, "receipt": receipt_path, "recorded_at": None})

    module._record_update(args)
    candidate = module.project(module._load_events(events_path))["candidate"]

    assert candidate["adapter_path"] == str(output.resolve())
    assert candidate["adapter_sha256"] == output_sha
