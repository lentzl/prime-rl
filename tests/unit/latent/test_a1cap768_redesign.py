import copy
from pathlib import Path

import pytest

from prime_rl.latent.a0 import canonical_json_hash
from prime_rl.latent.a1cap768_redesign import (
    AUTHORIZED_RUN_ID,
    CASE_SCHEDULE,
    CASE_SCHEDULE_SHA256,
    COMPARISON_SCHEDULE_SHA256,
    DESCRIPTIVE_FLAG_NAMES,
    DESCRIPTIVE_FLAG_NAMES_SHA256,
    FLAG0_INCOMPLETE_EVIDENCE,
    FLAG_NAMES,
    GATING_FLAG_NAMES,
    GATING_FLAG_NAMES_SHA256,
    MEMORY_LABELS_SHA256,
    OPERATION_SCHEDULE_SHA256,
    RESOURCE_BOUNDS,
    DiagnosticIncomplete,
    NoCacheRejected,
    ResourceFitRejected,
    build_comparison_schedule,
    build_operation_schedule,
    classification,
    classify_failure,
    memory_labels,
    validate_comparison,
    validate_constants,
    validate_flag0_incomplete,
)
from prime_rl.latent.cap768_redesign_invariants import (
    inspect_no_training_runner,
    require_pre_model_static_guard,
)


def test_exact_frozen_schedules_and_partitions():
    validate_constants()
    assert canonical_json_hash(CASE_SCHEDULE) == CASE_SCHEDULE_SHA256
    assert canonical_json_hash(build_operation_schedule()) == OPERATION_SCHEDULE_SHA256
    assert canonical_json_hash(build_comparison_schedule()) == COMPARISON_SCHEDULE_SHA256
    assert canonical_json_hash(GATING_FLAG_NAMES) == GATING_FLAG_NAMES_SHA256
    assert canonical_json_hash(DESCRIPTIVE_FLAG_NAMES) == DESCRIPTIVE_FLAG_NAMES_SHA256
    assert canonical_json_hash(memory_labels()) == MEMORY_LABELS_SHA256
    assert len(CASE_SCHEDULE) == 8
    assert len(build_operation_schedule()) == 56
    assert len(build_comparison_schedule()) == 104
    assert len(memory_labels()) == len(set(memory_labels())) == 123
    assert [row["modality"] for row in CASE_SCHEDULE] == ["PARENT", "MSELF"] * 4


def test_only_two_full_matrix_flags_are_descriptive():
    assert DESCRIPTIVE_FLAG_NAMES == [
        "keep0_last_logits_keep1_bitwise",
        "proj_id0_matches_id0_last_logits_bitwise",
    ]
    assert len(GATING_FLAG_NAMES) == 23
    assert [name for name in FLAG_NAMES if name not in GATING_FLAG_NAMES] == DESCRIPTIVE_FLAG_NAMES


def test_false_descriptive_flags_do_not_reject_but_any_gating_flag_does():
    flags = dict.fromkeys(FLAG_NAMES, True)
    for name in DESCRIPTIVE_FLAG_NAMES:
        flags[name] = False
    assert all(flags[name] for name in GATING_FLAG_NAMES)
    flags[GATING_FLAG_NAMES[-1]] = False
    assert not all(flags[name] for name in GATING_FLAG_NAMES)


def test_terminal_classification_is_exact():
    assert classification(True) == "capture768_redesign_validated"
    assert classification(False) == "capture768_redesign_rejected"
    assert classify_failure(NoCacheRejected("x")) == (
        "capture768_redesign_nocache_rejected",
        "cache_allocation_pkv_or_rope",
    )
    assert classify_failure(ResourceFitRejected("x")) == (
        "capture768_redesign_resource_fit_rejected",
        "allocator_cap_oom_or_compute_timeout",
    )
    assert classify_failure(DiagnosticIncomplete("x")) == (
        "capture768_redesign_incomplete",
        "diagnostic_operation_or_evidence_incomplete",
    )
    assert classify_failure(ValueError("x"))[0] == "infrastructure_invalid"


def test_comparison_validator_rejects_tamper():
    expected = build_comparison_schedule()[0]
    row = {
        **expected,
        "lhs_dtype": "torch.bfloat16",
        "rhs_dtype": "torch.bfloat16",
        "lhs_shape": [1, 1, 248320],
        "rhs_shape": [1, 1, 248320],
        "lhs_sha256": "a" * 64,
        "rhs_sha256": "a" * 64,
        "torch_equal": True,
        "element_count": 248320,
        "mismatch_count": 0,
        "count_nonzero": 0,
        "first_flat_mismatch": None,
        "metrics_defined": True,
        "max_abs": 0.0,
        "rms_diff": 0.0,
        "rhs_rms": 1.0,
        "normalized_rms": 0.0,
    }
    validate_comparison(row, expected)
    changed = copy.deepcopy(row)
    changed["comparison_index"] = 2
    with pytest.raises(DiagnosticIncomplete):
        validate_comparison(changed, expected)
    changed = copy.deepcopy(row)
    changed["torch_equal"] = False
    with pytest.raises(DiagnosticIncomplete):
        validate_comparison(changed, expected)


def test_bound_flag0_incomplete_artifacts_are_exact():
    evidence = validate_flag0_incomplete(Path("."))
    assert evidence == FLAG0_INCOMPLETE_EVIDENCE
    assert evidence["false_flags"] == DESCRIPTIVE_FLAG_NAMES
    assert evidence["failure_audit_errors"] == []


def test_real_runner_ast_guard_and_pre_model_order():
    runner = Path("scripts/latent/run_a1_nc0_cap768_redesign_v1.py")
    evidence = inspect_no_training_runner(runner)
    assert not evidence.forbidden_calls
    assert not evidence.forbidden_identifiers
    assert not evidence.forbidden_imports
    require_pre_model_static_guard(
        runner,
        run_function="run",
        guard_function="inspect_no_training_runner",
    )


def test_launcher_freezes_namespace_resources_and_no_update_boundary():
    source = Path("scripts/latent/run_a1_nc0_cap768_redesign_v1.sh").read_text()
    assert AUTHORIZED_RUN_ID in source
    assert RESOURCE_BOUNDS["output_root"] in source
    assert "62914560" in source and "67108864" in source
    assert "CUDA_VISIBLE_DEVICES=0" in source
    assert "timeout --signal=TERM --kill-after=60s 3600s" in source
    assert "--owner-approved" in source
    assert "--execution-commit" in source
