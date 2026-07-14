import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from prime_rl.trainer.rl.sdpo_export_verify import (
    find_broadcast_step_dirs,
    find_token_export_files,
    verify_sdpo_ema_broadcasts,
    verify_sdpo_smoke_artifacts,
    verify_sdpo_token_exports,
)

TEST_GIT_DIFF_SHA256 = hashlib.sha256(b"diff").hexdigest()
TEST_GIT_CACHED_DIFF_SHA256 = hashlib.sha256(b"cached").hexdigest()


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_raw_jsonl(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line.rstrip("\n") + "\n")


def _manifest_hash(lines):
    payload = "" if not lines else "\n".join(lines) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_smoke_provenance(
    path, *, mode="live", config="configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml"
):
    manifest_lines = ["untrackedhash  scripts/sdpo-example.py"]
    teacher_regularization = "ema" if mode == "ema" else "live-policy"
    path.write_text(
        "\n".join(
            [
                "sdpo_smoke_provenance_version=1",
                f"mode={mode}",
                f"config={config}",
                "expected_topk=2",
                "inference.vllm_extra.max_logprobs=2",
                "orchestrator.algo.distillation_topk=2",
                "orchestrator.algo.distillation_topk_support=student",
                "orchestrator.train.sampling.temperature=1.0",
                f"orchestrator.algo.teacher_regularization={teacher_regularization}",
                "orchestrator.algo.teacher_update_rate=0.05",
                "orchestrator.algo.success_reward_threshold=0.5",
                "orchestrator.algo.successful_demonstration_selection=batch_order",
                "orchestrator.algo.dont_reprompt_on_self_success=True",
                "orchestrator.algo.remove_thinking_from_demonstration=True",
                "orchestrator.algo.include_environment_feedback=True",
                "orchestrator.algo.environment_feedback_only_without_solution=True",
                "orchestrator.algo.max_reprompt_len=10240",
                "orchestrator.algo.reprompt_truncation=right",
                "orchestrator.algo.assistant_prefix=",
                "orchestrator.algo.multi_turn=False",
                "orchestrator.algo.template_target=first_user",
                "trainer.sdpo_loss.full_logit_distillation=True",
                "trainer.sdpo_loss.distillation_topk=2",
                "trainer.sdpo_loss.distillation_add_tail=True",
                "trainer.sdpo_loss.alpha=0.5",
                "trainer.sdpo_loss.is_clip=2.0",
                "trainer.sdpo_loss.rollout_is=token",
                "trainer.sdpo_loss.rollout_is_threshold=2.0",
                "trainer.sdpo_loss.rollout_is_batch_normalize=False",
                f"trainer.sdpo_runtime.teacher_regularization={teacher_regularization}",
                "trainer.sdpo_runtime.teacher_update_rate=0.05",
                "git_commit=abc123",
                "git_branch=codex/sdpo",
                f"git_diff_sha256={TEST_GIT_DIFF_SHA256}",
                f"git_cached_diff_sha256={TEST_GIT_CACHED_DIFF_SHA256}",
                f"git_untracked_manifest_sha256={_manifest_hash(manifest_lines)}",
                "python_runner=uv run python",
                "rl_runner=uv run rl",
                "git_untracked_manifest_begin",
                *manifest_lines,
                "git_untracked_manifest_end",
                "git_status_short_begin",
                " M some-local-file.py",
                "git_status_short_end",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _record(**overrides):
    record = {
        "schema_version": 2,
        "sample_id": "sample-a",
        "env_name": "sdpo_env",
        "token_ids": [10, 11, 12],
        "position_ids": [0, 1, 2],
        "loss_mask": [False, True, True],
        "temperatures": [1.0, 0.75, 0.5],
        "trainer_logprobs": [None, -1.0, -1.5],
        "inference_logprobs": [None, -1.25, -1.0],
        "log_importance_ratio": [None, 0.25, -0.5],
        "importance_ratio": [None, 1.2840254, 0.6065307],
        "prob_delta": [None, 0.08137433, -0.14474928],
        "preflight_only": False,
        "sdpo_weights": [0.0, 1.0, 1.0],
        "sdpo_rollout_is_weights": [0.0, 1.2840254, 0.6065307],
        "sdpo_topk_token_ids": [[0, 0], [101, 102], [201, 202]],
        "sdpo_topk_logprobs": [[0.0, 0.0], [-0.5, -2.0], [-0.3, -1.5]],
        "sdpo_student_topk_token_ids": [[0, 0], [111, 112], [211, 212]],
        "sdpo_student_topk_logprobs": [[0.0, 0.0], [-0.75, -1.25], [-0.625, -1.5]],
    }
    record.update(overrides)
    if "sdpo_weights" in overrides and "sdpo_rollout_is_weights" not in overrides:
        record["sdpo_rollout_is_weights"] = [
            0.0 if weight in (None, 0, 0.0) else 1.0 for weight in record["sdpo_weights"]
        ]
    return record


def _student_preflight_record(**overrides):
    return _record(
        preflight_only=True,
        sdpo_topk_token_ids=[None, None, None],
        sdpo_topk_logprobs=[None, None, None],
        **overrides,
    )


def _student_final_record(**overrides):
    return _record(
        sdpo_topk_token_ids=[[0, 0], [111, 112], [211, 212]],
        sdpo_topk_logprobs=[[0.0, 0.0], [-3.0, -13.0], [-4.0, -14.0]],
        **overrides,
    )


def test_verify_sdpo_token_exports_accepts_output_dir(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])
    _write_stable(export_file.parent)

    stats = verify_sdpo_token_exports(tmp_path, expected_topk=2)

    assert stats.files == 1
    assert stats.records == 1
    assert stats.sdpo_records == 1
    assert stats.transported_rows == 2
    assert stats.student_rows == 2
    assert stats.paired_rows == 2
    assert stats.matching_support_rows == 0
    assert stats.student_preflight_rows == 0
    assert stats.importance_ratio_rows == 2
    assert stats.rollout_is_weight_rows == 2
    assert stats.temperature_rows == 2
    assert stats.sample_id_records == 1
    assert stats.stable_steps == 1
    assert stats.step_names == ("step_3",)
    assert stats.paired_step_names == ("step_3",)
    assert stats.matching_support_step_names == ()
    assert stats.student_preflight_step_names == ()
    assert stats.matching_support_sample_keys == ()
    assert stats.student_preflight_sample_keys == ()
    assert stats.matched_support_sample_keys == ()
    assert stats.importance_ratio_row_keys == ()
    assert stats.rollout_is_weight_row_keys == ()


def test_verify_sdpo_token_exports_ignores_inactive_record_without_support_rows(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    inactive = _record(
        sample_id=None,
        sdpo_weights=[0.0, 0.0, 0.0],
        sdpo_topk_token_ids=[None, None, None],
        sdpo_topk_logprobs=[None, None, None],
        sdpo_student_topk_token_ids=[None, None, None],
        sdpo_student_topk_logprobs=[None, None, None],
    )
    _write_jsonl(export_file, [inactive, _record()])

    stats = verify_sdpo_token_exports(tmp_path, expected_topk=2)

    assert stats.records == 2
    assert stats.sdpo_records == 1


@pytest.mark.parametrize("record", [[], None, "not-object"])
def test_verify_sdpo_token_exports_rejects_non_object_jsonl_records(tmp_path, record):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [record])

    with pytest.raises(ValueError, match="expected JSON object record"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_duplicate_json_object_keys(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    raw_record = json.dumps(_record())
    duplicate_key_record = raw_record.replace('"sample_id": "sample-a"', '"sample_id": "sample-a", "sample_id": "b"', 1)
    _write_raw_jsonl(export_file, [duplicate_key_record])

    with pytest.raises(ValueError, match="duplicate JSON object key: sample_id"):
        verify_sdpo_token_exports(tmp_path, expected_topk=2)


@pytest.mark.parametrize(
    ("field_prefix", "expected_error"),
    [
        ("sdpo", "inactive SDPO record must not carry sdpo top-k support rows"),
        ("sdpo_student", "inactive SDPO record must not carry sdpo_student top-k support rows"),
    ],
)
def test_verify_sdpo_token_exports_rejects_inactive_record_with_support_rows(tmp_path, field_prefix, expected_error):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    inactive = _record(
        sample_id=None,
        sdpo_weights=[0.0, 0.0, 0.0],
        sdpo_topk_token_ids=[None, None, None],
        sdpo_topk_logprobs=[None, None, None],
        sdpo_student_topk_token_ids=[None, None, None],
        sdpo_student_topk_logprobs=[None, None, None],
    )
    inactive[f"{field_prefix}_topk_token_ids"] = [[0, 0], [0, 0], [0, 0]]
    inactive[f"{field_prefix}_topk_logprobs"] = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    _write_jsonl(export_file, [inactive, _record()])

    with pytest.raises(ValueError, match=expected_error):
        verify_sdpo_token_exports(tmp_path, expected_topk=2)


def test_verify_sdpo_token_exports_rejects_boolean_inactive_sdpo_weights(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    inactive = _record(
        sample_id=None,
        sdpo_weights=[False, False, False],
        sdpo_topk_token_ids=[None, None, None],
        sdpo_topk_logprobs=[None, None, None],
        sdpo_student_topk_token_ids=[None, None, None],
        sdpo_student_topk_logprobs=[None, None, None],
    )
    _write_jsonl(export_file, [inactive, _record()])

    with pytest.raises(ValueError, match="sdpo_weights must contain finite numeric values at token 0"):
        verify_sdpo_token_exports(tmp_path, expected_topk=2)


@pytest.mark.parametrize("env_name", [None, "", "   ", 123])
def test_verify_sdpo_token_exports_requires_env_name_on_active_sdpo_records(tmp_path, env_name):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(env_name=env_name)])

    with pytest.raises(ValueError, match="env_name must be a non-empty string"):
        verify_sdpo_token_exports(tmp_path, expected_topk=2)


def test_verify_sdpo_token_exports_rejects_unexpected_topk_width(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])

    with pytest.raises(ValueError, match="sdpo row width 2 at 0 != expected_topk 3"):
        verify_sdpo_token_exports(tmp_path, expected_topk=3)


@pytest.mark.parametrize("expected_topk", [True, 2.5, "2"])
def test_verify_sdpo_token_exports_rejects_non_integer_expected_topk(tmp_path, expected_topk):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])

    with pytest.raises(ValueError, match="expected_topk must be an integer"):
        verify_sdpo_token_exports(tmp_path, expected_topk=expected_topk)


@pytest.mark.parametrize("expected_topk", [0, -1])
def test_verify_sdpo_token_exports_rejects_non_positive_expected_topk(tmp_path, expected_topk):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])

    with pytest.raises(ValueError, match="expected_topk must be positive"):
        verify_sdpo_token_exports(tmp_path, expected_topk=expected_topk)


def test_verify_sdpo_token_exports_rejects_non_integer_token_ids(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(token_ids=[10, True, 12])])

    with pytest.raises(ValueError, match="token_ids must contain integer token ids"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_missing_position_ids(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    record = _record()
    del record["position_ids"]
    _write_jsonl(export_file, [record])

    with pytest.raises(ValueError, match="position_ids must be a list"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_misaligned_position_ids(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(position_ids=[0, 1])])

    with pytest.raises(ValueError, match="position_ids length 2 != token_ids length 3"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_duplicate_transported_support_ids(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_topk_token_ids=[[0, 0], [101, 101], [201, 202]])])

    with pytest.raises(ValueError, match=r"sdpo_topk_token_ids\[1\] must contain distinct token ids"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_duplicate_student_support_ids(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_student_topk_token_ids=[[0, 0], [111, 111], [211, 212]])])

    with pytest.raises(ValueError, match=r"sdpo_student_topk_token_ids\[1\] must contain distinct token ids"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_accepts_top1_token_id_zero_with_real_logprob(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_topk_token_ids=[[0], [0], [201]],
                sdpo_topk_logprobs=[[0.0], [-0.5], [-0.3]],
                sdpo_student_topk_token_ids=[[0], [111], [0]],
                sdpo_student_topk_logprobs=[[0.0], [-0.75], [-0.625]],
            )
        ],
    )

    stats = verify_sdpo_token_exports(tmp_path, expected_topk=1)

    assert stats.transported_rows == 2
    assert stats.student_rows == 2


def test_verify_sdpo_token_exports_rejects_malformed_unweighted_transport_support_row(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_topk_token_ids=[[99, 99], [101, 102], [201, 202]],
                sdpo_topk_logprobs=[[-0.5, -2.0], [-0.5, -2.0], [-0.3, -1.5]],
            )
        ],
    )

    with pytest.raises(ValueError, match=r"sdpo_topk_token_ids\[0\] must contain distinct token ids"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_dense_unweighted_transport_support_row(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_topk_token_ids=[[99, 100], [101, 102], [201, 202]],
                sdpo_topk_logprobs=[[-1.0, -2.0], [-0.5, -2.0], [-0.3, -1.5]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo support at unweighted token 0 must be null or an all-zero placeholder"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_dense_unweighted_student_support_row(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_token_ids=[[99, 100], [111, 112], [211, 212]],
                sdpo_student_topk_logprobs=[[-1.0, -2.0], [-0.75, -1.25], [-0.625, -1.5]],
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="sdpo_student support at unweighted token 0 must be null or an all-zero placeholder",
    ):
        verify_sdpo_token_exports(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sdpo_topk_token_ids", [[False, 0], [101, 102], [201, 202]], "must contain integer token ids"),
        ("sdpo_topk_logprobs", [[False, 0.0], [-0.5, -2.0], [-0.3, -1.5]], "finite numeric values"),
        (
            "sdpo_student_topk_token_ids",
            [[False, 0], [111, 112], [211, 212]],
            "must contain integer token ids",
        ),
        (
            "sdpo_student_topk_logprobs",
            [[False, 0.0], [-0.75, -1.25], [-0.625, -1.5]],
            "finite numeric values",
        ),
    ],
)
def test_verify_sdpo_token_exports_rejects_boolean_unweighted_placeholder_values(tmp_path, field, value, message):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(**{field: value})])

    with pytest.raises(ValueError, match=message):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_nonzero_ids_in_unweighted_placeholder_row(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_token_ids=[[99, 100], [111, 112], [211, 212]],
                sdpo_student_topk_logprobs=[[0.0, 0.0], [-0.75, -1.25], [-0.625, -1.5]],
            )
        ],
    )

    with pytest.raises(ValueError, match=r"sdpo_student_topk_token_ids\[0\] must be a zero placeholder row"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_non_boolean_loss_mask(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(loss_mask=[False, 1, True])])

    with pytest.raises(ValueError, match="loss_mask must contain booleans"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_non_numeric_sdpo_weights(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_weights=[0.0, True, 1.0])])

    with pytest.raises(ValueError, match="sdpo_weights must contain finite numeric values"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_non_list_sdpo_weights(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_weights=1.0, sdpo_rollout_is_weights=[0.0, 0.75, 1.25])])

    with pytest.raises(ValueError, match="sdpo_weights must be a list"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_negative_sdpo_weights(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_weights=[0.0, -0.5, 1.0])])

    with pytest.raises(ValueError, match=r"sdpo_weights\[1\] must be non-negative"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_nonzero_rollout_is_weight_outside_sdpo_component(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_rollout_is_weights=[0.5, 0.75, 1.25])])

    with pytest.raises(ValueError, match=r"sdpo_rollout_is_weights\[0\] is nonzero outside SDPO component"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_negative_rollout_is_weight(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_rollout_is_weights=[0.0, -0.75, 1.25])])

    with pytest.raises(ValueError, match=r"sdpo_rollout_is_weights\[1\] must be non-negative"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_integer_rollout_is_weight(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_rollout_is_weights=[0.0, 1, 1.25])])

    with pytest.raises(ValueError, match=r"sdpo_rollout_is_weights\[1\] must be a floating-point number"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_rollout_is_weight_above_threshold_when_requested(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_rollout_is_weights=[0.0, 2.01, 1.25])])

    with pytest.raises(ValueError, match=r"sdpo_rollout_is_weights\[1\] exceeds rollout_is_threshold=2.0"):
        verify_sdpo_token_exports(tmp_path, rollout_is_threshold=2.0)


def test_verify_sdpo_token_exports_requires_rollout_is_weights_when_threshold_requested(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_rollout_is_weights=None)])

    with pytest.raises(ValueError, match="final weighted SDPO rows are missing sdpo_rollout_is_weights"):
        verify_sdpo_token_exports(tmp_path, rollout_is_threshold=2.0)


def test_verify_sdpo_token_exports_requires_each_final_rollout_is_weight_when_threshold_requested(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_rollout_is_weights=[0.0, None, 1.25])])

    with pytest.raises(ValueError, match=r"sdpo_rollout_is_weights\[1\] is missing"):
        verify_sdpo_token_exports(tmp_path, rollout_is_threshold=2.0)


@pytest.mark.parametrize("rollout_is_threshold", [True, "2.0"])
def test_verify_sdpo_token_exports_rejects_non_numeric_rollout_is_threshold(tmp_path, rollout_is_threshold):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])

    with pytest.raises(ValueError, match="rollout_is_threshold must be numeric"):
        verify_sdpo_token_exports(tmp_path, rollout_is_threshold=rollout_is_threshold)


@pytest.mark.parametrize("rollout_is_threshold", [float("nan"), float("inf")])
def test_verify_sdpo_token_exports_rejects_non_finite_rollout_is_threshold(tmp_path, rollout_is_threshold):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])

    with pytest.raises(ValueError, match="rollout_is_threshold must be finite"):
        verify_sdpo_token_exports(tmp_path, rollout_is_threshold=rollout_is_threshold)


@pytest.mark.parametrize("rollout_is_threshold", [0.0, -1.0])
def test_verify_sdpo_token_exports_rejects_non_positive_rollout_is_threshold(tmp_path, rollout_is_threshold):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])

    with pytest.raises(ValueError, match="rollout_is_threshold must be positive"):
        verify_sdpo_token_exports(tmp_path, rollout_is_threshold=rollout_is_threshold)


def test_verify_sdpo_token_exports_rejects_unknown_rollout_is_mode(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])

    with pytest.raises(ValueError, match="rollout_is must be 'token' or 'sequence'"):
        verify_sdpo_token_exports(tmp_path, rollout_is="batch")


def test_verify_sdpo_token_exports_requires_threshold_for_rollout_is_matching(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])

    with pytest.raises(ValueError, match="rollout_is matching requires rollout_is_threshold"):
        verify_sdpo_token_exports(tmp_path, rollout_is="token")


def test_verify_sdpo_token_exports_accepts_split_preflight_and_final_records(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _student_preflight_record(),
            _record(),
        ],
    )

    stats = verify_sdpo_token_exports(tmp_path)

    assert stats.files == 1
    assert stats.records == 2
    assert stats.sdpo_records == 2
    assert stats.transported_rows == 2
    assert stats.student_rows == 4
    assert stats.paired_rows == 2
    assert stats.matching_support_rows == 0
    assert stats.student_preflight_rows == 2
    assert stats.importance_ratio_rows == 2
    assert stats.rollout_is_weight_rows == 2
    assert stats.temperature_rows == 4
    assert stats.sample_id_records == 2
    assert stats.paired_step_names == ("step_3",)
    assert stats.matching_support_step_names == ()
    assert stats.student_preflight_step_names == ("step_3",)
    assert stats.matching_support_sample_keys == ()
    assert stats.student_preflight_sample_keys == ("run_default:step_3:sample-a",)
    assert stats.matched_support_sample_keys == ()


def test_verify_sdpo_token_exports_accepts_exact_student_support_smoke_pattern(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])

    stats = verify_sdpo_token_exports(tmp_path, require_student_preflight=True, expected_topk=2)

    assert stats.student_preflight_rows == 2
    assert stats.matching_support_rows == 2
    assert stats.distinct_teacher_logprob_rows == 2
    assert stats.importance_ratio_rows == 2
    assert stats.rollout_is_weight_rows == 2
    assert stats.paired_rows == 2
    assert stats.student_preflight_step_names == ("step_3",)
    assert stats.paired_step_names == ("step_3",)
    assert stats.matching_support_step_names == ("step_3",)
    assert stats.student_preflight_sample_keys == ("run_default:step_3:sample-a",)
    assert stats.matching_support_sample_keys == ("run_default:step_3:sample-a",)
    assert stats.matched_support_sample_keys == ("run_default:step_3:sample-a",)
    assert stats.student_preflight_row_keys == (
        "run_default:step_3:sample-a:token-1",
        "run_default:step_3:sample-a:token-2",
    )
    assert stats.matching_support_row_keys == (
        "run_default:step_3:sample-a:token-1",
        "run_default:step_3:sample-a:token-2",
    )
    assert stats.matched_support_row_keys == (
        "run_default:step_3:sample-a:token-1",
        "run_default:step_3:sample-a:token-2",
    )
    assert stats.importance_ratio_row_keys == (
        "run_default:step_3:sample-a:token-1",
        "run_default:step_3:sample-a:token-2",
    )
    assert stats.rollout_is_weight_row_keys == (
        "run_default:step_3:sample-a:token-1",
        "run_default:step_3:sample-a:token-2",
    )
    assert stats.distinct_teacher_logprob_row_keys == (
        "run_default:step_3:sample-a:token-1",
        "run_default:step_3:sample-a:token-2",
    )


def test_verify_sdpo_token_exports_requires_importance_ratio_evidence_when_requested(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    missing_ratio_evidence = _student_final_record(
        log_importance_ratio=[None, None, None],
        importance_ratio=[None, None, None],
        prob_delta=[None, None, None],
    )
    _write_jsonl(export_file, [_student_preflight_record(), missing_ratio_evidence])

    with pytest.raises(ValueError, match="missing rollout-IS ratio evidence"):
        verify_sdpo_token_exports(
            tmp_path,
            require_student_preflight=True,
            require_importance_ratio_evidence=True,
        )


def test_verify_sdpo_token_exports_requires_ratio_evidence_on_every_final_weighted_row(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    partial_ratio_evidence = _record(
        log_importance_ratio=[None, 0.25, None],
        importance_ratio=[None, 1.2840254, None],
        prob_delta=[None, 0.08137433, None],
    )
    _write_jsonl(export_file, [partial_ratio_evidence])

    with pytest.raises(ValueError, match=r"missing rollout-IS ratio evidence.*token position.*\[2\]"):
        verify_sdpo_token_exports(
            tmp_path,
            require_importance_ratio_evidence=True,
        )


def test_verify_sdpo_token_exports_rejects_ratio_evidence_without_source_logprobs(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    missing_source_logprobs = _student_final_record()
    missing_source_logprobs.pop("trainer_logprobs")
    _write_jsonl(export_file, [_student_preflight_record(), missing_source_logprobs])

    with pytest.raises(ValueError, match="trainer_logprobs must be a list"):
        verify_sdpo_token_exports(
            tmp_path,
            require_student_preflight=True,
            require_importance_ratio_evidence=True,
        )


def test_verify_sdpo_token_exports_rejects_inconsistent_importance_ratio_evidence(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    inconsistent_ratio = _student_final_record(importance_ratio=[None, 9.0, 0.6065307])
    _write_jsonl(export_file, [_student_preflight_record(), inconsistent_ratio])

    with pytest.raises(ValueError, match=r"importance_ratio\[1\] does not match exp"):
        verify_sdpo_token_exports(
            tmp_path,
            require_student_preflight=True,
            require_importance_ratio_evidence=True,
        )


def test_verify_sdpo_token_exports_rejects_inconsistent_probability_delta_evidence(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    inconsistent_delta = _student_final_record(prob_delta=[None, 0.08137433, 9.0])
    _write_jsonl(export_file, [_student_preflight_record(), inconsistent_delta])

    with pytest.raises(ValueError, match=r"prob_delta\[2\] does not match"):
        verify_sdpo_token_exports(
            tmp_path,
            require_student_preflight=True,
            require_importance_ratio_evidence=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("trainer_logprobs", [None, -1, -1.5], r"trainer_logprobs\[1\] must be a floating-point number"),
        ("inference_logprobs", [None, -1.25, -1], r"inference_logprobs\[2\] must be a floating-point number"),
        ("log_importance_ratio", [None, 0, -0.5], r"log_importance_ratio\[1\] must be a floating-point number"),
        ("importance_ratio", [None, 1, 0.6065307], r"importance_ratio\[1\] must be a floating-point number"),
        ("prob_delta", [None, 0.08137433, 0], r"prob_delta\[2\] must be a floating-point number"),
    ],
)
def test_verify_sdpo_token_exports_rejects_integer_ratio_evidence_values(tmp_path, field, value, expected_error):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    integer_evidence = _student_final_record(**{field: value})
    _write_jsonl(export_file, [_student_preflight_record(), integer_evidence])

    with pytest.raises(ValueError, match=expected_error):
        verify_sdpo_token_exports(
            tmp_path,
            require_student_preflight=True,
            require_importance_ratio_evidence=True,
        )


def test_verify_sdpo_smoke_artifacts_requires_importance_ratio_evidence(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    missing_ratio_evidence = _student_final_record(
        log_importance_ratio=[None, None, None],
        importance_ratio=[None, None, None],
        prob_delta=[None, None, None],
    )
    _write_jsonl(export_file, [_student_preflight_record(), missing_ratio_evidence])
    _write_stable(export_file.parent)

    with pytest.raises(ValueError, match="missing rollout-IS ratio evidence"):
        verify_sdpo_smoke_artifacts(tmp_path, expected_topk=2)


def test_verify_sdpo_smoke_artifacts_requires_reference_rollout_is_threshold(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    above_reference_threshold = _student_final_record(sdpo_rollout_is_weights=[0.0, 2.01, 1.25])
    _write_jsonl(export_file, [_student_preflight_record(), above_reference_threshold])
    _write_stable(export_file.parent)

    with pytest.raises(ValueError, match=r"sdpo_rollout_is_weights\[1\] exceeds rollout_is_threshold=2.0"):
        verify_sdpo_smoke_artifacts(tmp_path, expected_topk=2)


def test_verify_sdpo_smoke_artifacts_requires_rollout_is_weight_evidence(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    missing_rollout_is_weights = _student_final_record(sdpo_rollout_is_weights=None)
    _write_jsonl(export_file, [_student_preflight_record(), missing_rollout_is_weights])
    _write_stable(export_file.parent)

    with pytest.raises(ValueError, match="final weighted SDPO rows are missing sdpo_rollout_is_weights"):
        verify_sdpo_smoke_artifacts(tmp_path, expected_topk=2)


def test_verify_sdpo_smoke_artifacts_requires_token_rollout_is_weights_to_match_ratio(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    mismatched_rollout_is_weights = _student_final_record(sdpo_rollout_is_weights=[0.0, 0.75, 1.25])
    _write_jsonl(export_file, [_student_preflight_record(), mismatched_rollout_is_weights])
    _write_stable(export_file.parent)

    with pytest.raises(ValueError, match="does not match token rollout-IS"):
        verify_sdpo_smoke_artifacts(tmp_path, expected_topk=2)


def test_verify_sdpo_token_exports_accepts_sequence_rollout_is_weight_matching(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    sequence_rollout_is = _student_final_record(sdpo_rollout_is_weights=[0.0, 0.7788008, 0.7788008])
    _write_jsonl(export_file, [_student_preflight_record(), sequence_rollout_is])

    stats = verify_sdpo_token_exports(
        tmp_path,
        require_student_preflight=True,
        require_importance_ratio_evidence=True,
        expected_topk=2,
        rollout_is_threshold=2.0,
        rollout_is="sequence",
    )

    assert stats.rollout_is_weight_rows == 2


def test_verify_sdpo_token_exports_rejects_sequence_rollout_is_weight_mismatch(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    token_rollout_is = _student_final_record()
    _write_jsonl(export_file, [_student_preflight_record(), token_rollout_is])

    with pytest.raises(ValueError, match="does not match sequence rollout-IS"):
        verify_sdpo_token_exports(
            tmp_path,
            require_student_preflight=True,
            require_importance_ratio_evidence=True,
            expected_topk=2,
            rollout_is_threshold=2.0,
            rollout_is="sequence",
        )


def test_verify_sdpo_token_exports_rejects_final_teacher_logprobs_copied_from_student(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    copied_student_scores = _student_final_record()
    copied_student_scores["sdpo_topk_logprobs"] = [[0.0, 0.0], [-0.75, -1.25], [-0.625, -1.5]]
    _write_jsonl(export_file, [_student_preflight_record(), copied_student_scores])

    with pytest.raises(ValueError, match="teacher logprobs differ"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True, expected_topk=2)


def test_verify_sdpo_token_exports_rejects_preflight_final_sdpo_weight_drift(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _student_preflight_record(sdpo_weights=[0.0, 1.0, 1.0]),
            _student_final_record(sdpo_weights=[0.0, 0.5, 1.0]),
        ],
    )

    with pytest.raises(ValueError, match="different sample signatures"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True, expected_topk=2)


def test_verify_sdpo_token_exports_rejects_preflight_final_support_id_drift(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _student_preflight_record(),
            _record(
                sdpo_topk_token_ids=[[0, 0], [901, 902], [211, 212]],
                sdpo_topk_logprobs=[[0.0, 0.0], [-3.0, -13.0], [-4.0, -14.0]],
                sdpo_student_topk_token_ids=[[0, 0], [901, 902], [211, 212]],
                sdpo_student_topk_logprobs=[[0.0, 0.0], [-0.75, -1.25], [-0.625, -1.5]],
            ),
        ],
    )

    with pytest.raises(ValueError, match="different support ids for same-step token row"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True, expected_topk=2)


def test_verify_sdpo_token_exports_rejects_transported_support_on_preflight_record(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_final_record(preflight_only=True)])

    with pytest.raises(ValueError, match="transported teacher SDPO support rows require preflight_only=false"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_duplicate_preflight_sample_key(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_preflight_record(), _student_final_record()])

    with pytest.raises(
        ValueError, match="duplicate preflight-only student support record for run_default:step_3:sample-a"
    ):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True)


def test_verify_sdpo_token_exports_rejects_duplicate_matching_final_sample_key(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record(), _student_final_record()])

    with pytest.raises(ValueError, match="duplicate final SDPO record for run_default:step_3:sample-a"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True)


def test_verify_sdpo_token_exports_rejects_duplicate_final_sample_key_even_when_extra_record_does_not_match(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _student_preflight_record(),
            _student_final_record(),
            _record(
                sample_id="sample-a",
                sdpo_topk_token_ids=[[0, 0], [901, 902], [211, 212]],
                sdpo_topk_logprobs=[[0.0, 0.0], [-3.0, -13.0], [-4.0, -14.0]],
                sdpo_student_topk_token_ids=[[0, 0], [901, 902], [211, 212]],
                sdpo_student_topk_logprobs=[[0.0, 0.0], [-0.75, -1.25], [-0.625, -1.5]],
            ),
        ],
    )

    with pytest.raises(ValueError, match="duplicate final SDPO record for run_default:step_3:sample-a"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True)


def test_verify_sdpo_token_exports_keys_same_step_samples_by_run_dir(tmp_path):
    first = tmp_path / "run_a" / "token_exports" / "step_3" / "rank_0.jsonl"
    second = tmp_path / "run_b" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(first, [_student_preflight_record(), _student_final_record()])
    _write_jsonl(second, [_student_preflight_record(), _student_final_record()])

    stats = verify_sdpo_token_exports(tmp_path, require_student_preflight=True)

    assert stats.student_preflight_sample_keys == (
        "run_a:step_3:sample-a",
        "run_b:step_3:sample-a",
    )
    assert stats.matching_support_sample_keys == (
        "run_a:step_3:sample-a",
        "run_b:step_3:sample-a",
    )
    assert stats.matched_support_sample_keys == (
        "run_a:step_3:sample-a",
        "run_b:step_3:sample-a",
    )


def test_verify_sdpo_smoke_artifacts_forwards_expected_topk(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)

    with pytest.raises(ValueError, match="expected_topk 3"):
        verify_sdpo_smoke_artifacts(tmp_path, expected_topk=3)


def test_verify_sdpo_smoke_artifacts_requires_expected_topk(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)

    with pytest.raises(ValueError, match="requires expected_topk"):
        verify_sdpo_smoke_artifacts(tmp_path)


@pytest.mark.parametrize("expected_topk", [True, 2.5, "2"])
def test_verify_sdpo_smoke_artifacts_rejects_non_integer_expected_topk(tmp_path, expected_topk):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)

    with pytest.raises(ValueError, match="expected_topk must be an integer"):
        verify_sdpo_smoke_artifacts(tmp_path, expected_topk=expected_topk)


def test_verify_sdpo_token_exports_rejects_missing_student_preflight_when_required(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_final_record()])

    with pytest.raises(ValueError, match="preflight-only trainer-forward student support"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True)


def test_verify_sdpo_token_exports_rejects_mismatched_student_support_when_required(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _record()])

    with pytest.raises(ValueError, match="transported support ids match"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True)


def test_verify_sdpo_token_exports_requires_same_step_preflight_and_matching_final_support(tmp_path):
    preflight_step = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    matching_step = tmp_path / "run_default" / "token_exports" / "step_9" / "rank_0.jsonl"
    _write_jsonl(preflight_step, [_student_preflight_record(), _record()])
    _write_jsonl(matching_step, [_student_final_record()])

    with pytest.raises(ValueError, match="matching final SDPO support rows do not overlap"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True)


def test_verify_sdpo_token_exports_requires_same_sample_preflight_and_matching_final_support(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _student_preflight_record(sample_id="sample-a"),
            _student_final_record(sample_id="sample-b"),
        ],
    )

    with pytest.raises(ValueError, match="same-step sample_id"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True)


def test_verify_sdpo_token_exports_requires_same_sample_signature_for_preflight_and_final(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _student_preflight_record(sample_id="sample-a"),
            _student_final_record(sample_id="sample-a", token_ids=[10, 99, 12]),
        ],
    )

    with pytest.raises(ValueError, match="different sample signatures"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True)


def test_verify_sdpo_token_exports_requires_same_env_name_for_preflight_and_final(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _student_preflight_record(sample_id="sample-a", env_name="env-a"),
            _student_final_record(sample_id="sample-a", env_name="env-b"),
        ],
    )

    with pytest.raises(ValueError, match="different sample signatures"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True)


def test_verify_sdpo_token_exports_requires_same_position_ids_for_preflight_and_final(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _student_preflight_record(sample_id="sample-a"),
            _student_final_record(sample_id="sample-a", position_ids=[0, 4, 5]),
        ],
    )

    with pytest.raises(ValueError, match="different sample signatures"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True)


def test_verify_sdpo_token_exports_rejects_partial_preflight_final_sample_coverage(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _student_preflight_record(sample_id="sample-a"),
            _student_preflight_record(sample_id="sample-b"),
            _student_final_record(sample_id="sample-a"),
        ],
    )

    with pytest.raises(ValueError, match="missing matching final SDPO support"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True)


def test_verify_sdpo_token_exports_rejects_extra_final_sample_without_preflight(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _student_preflight_record(sample_id="sample-a"),
            _student_final_record(sample_id="sample-a"),
            _student_final_record(sample_id="sample-b"),
        ],
    )

    with pytest.raises(ValueError, match="no preflight-only student support for same-step sample_id"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True)


def test_verify_sdpo_token_exports_rejects_partial_preflight_final_row_coverage(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _student_preflight_record(sample_id="sample-a"),
            _record(
                sample_id="sample-a",
                sdpo_weights=[0.0, 1.0, 0.0],
                sdpo_topk_token_ids=[[0, 0], [111, 112], [0, 0]],
                sdpo_topk_logprobs=[[0.0, 0.0], [-3.0, -13.0], [0.0, 0.0]],
                sdpo_student_topk_token_ids=[[0, 0], [111, 112], [0, 0]],
                sdpo_student_topk_logprobs=[[0.0, 0.0], [-0.75, -1.25], [0.0, 0.0]],
            ),
        ],
    )

    with pytest.raises(ValueError, match="missing matching final SDPO support for same-step token row"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True)


def test_verify_sdpo_token_exports_rejects_extra_final_row_without_preflight(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _student_preflight_record(
                sample_id="sample-a",
                sdpo_weights=[0.0, 1.0, 0.0],
                sdpo_student_topk_token_ids=[[0, 0], [111, 112], [0, 0]],
                sdpo_student_topk_logprobs=[[0.0, 0.0], [-0.75, -1.25], [0.0, 0.0]],
            ),
            _student_final_record(sample_id="sample-a"),
        ],
    )

    with pytest.raises(ValueError, match="no preflight-only student support for same-step token row"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True)


def test_find_token_export_files_accepts_step_dir(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])

    assert find_token_export_files(export_file.parent) == [export_file]


def test_verify_sdpo_token_exports_can_require_stable_marker(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])

    with pytest.raises(ValueError, match="missing token export STABLE"):
        verify_sdpo_token_exports(tmp_path, require_stable=True)

    _write_stable(export_file.parent)

    stats = verify_sdpo_token_exports(tmp_path, require_stable=True)
    assert stats.stable_steps == 1


def test_find_token_export_files_accepts_sibling_broadcast_step_dir(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    broadcast_step_dir = tmp_path / "run_default" / "broadcasts" / "step_3"
    _write_jsonl(export_file, [_record()])
    _write_stable(broadcast_step_dir)

    assert find_token_export_files(broadcast_step_dir) == [export_file]


def test_find_token_export_files_requires_matching_sibling_step(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_9" / "rank_0.jsonl"
    broadcast_step_dir = tmp_path / "run_default" / "broadcasts" / "step_3"
    _write_jsonl(export_file, [_record()])
    _write_stable(broadcast_step_dir)

    with pytest.raises(FileNotFoundError, match="No token export"):
        find_token_export_files(broadcast_step_dir)


def test_verify_sdpo_token_exports_rejects_missing_student_support(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_token_ids=[None, None, None],
                sdpo_student_topk_logprobs=[None, None, None],
            )
        ],
    )

    with pytest.raises(ValueError, match="found no supported trainer-forward sdpo_student_topk rows"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_absent_student_support_columns(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    record = _record()
    del record["sdpo_student_topk_token_ids"]
    del record["sdpo_student_topk_logprobs"]
    _write_jsonl(export_file, [record])

    with pytest.raises(ValueError, match="found no supported trainer-forward sdpo_student_topk rows"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_missing_temperatures(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    record = _record()
    del record["temperatures"]
    _write_jsonl(export_file, [record])

    with pytest.raises(ValueError, match="temperatures must be a list"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_missing_sample_id(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    record = _record()
    del record["sample_id"]
    _write_jsonl(export_file, [record])

    with pytest.raises(ValueError, match="sample_id must be a non-empty string"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_blank_sample_id(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sample_id="   ")])

    with pytest.raises(ValueError, match="sample_id must be a non-empty string"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_non_boolean_preflight_flag(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(preflight_only="yes")])

    with pytest.raises(ValueError, match="preflight_only must be a boolean"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_requires_explicit_preflight_flag_for_preflight_rows(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    record = _student_preflight_record()
    del record["preflight_only"]
    _write_jsonl(export_file, [record, _student_final_record()])

    with pytest.raises(ValueError, match="preflight_only must be a boolean"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True)


def test_verify_sdpo_token_exports_requires_explicit_preflight_flag_for_final_rows(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    record = _student_final_record()
    del record["preflight_only"]
    _write_jsonl(export_file, [_student_preflight_record(), record])

    with pytest.raises(ValueError, match="preflight_only must be a boolean"):
        verify_sdpo_token_exports(tmp_path, require_student_preflight=True)


def test_verify_sdpo_token_exports_rejects_nonpositive_weighted_temperature(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(temperatures=[1.0, 0.0, 0.5])])

    with pytest.raises(ValueError, match="temperatures\\[1\\] must be a positive number"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_sdpo_weight_outside_loss_mask(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_weights=[1.0, 1.0, 1.0])])

    with pytest.raises(ValueError, match="sdpo weight at token 0 is nonzero outside loss_mask"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_unpaired_student_and_transport_support(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_weights=[0.0, 1.0, 0.0],
                sdpo_topk_token_ids=[[0, 0], [101, 102], [0, 0]],
                sdpo_topk_logprobs=[[0.0, 0.0], [-0.5, -2.0], [0.0, 0.0]],
                sdpo_student_topk_token_ids=[None, None, None],
                sdpo_student_topk_logprobs=[None, None, None],
            ),
            _record(
                preflight_only=True,
                sdpo_weights=[0.0, 0.0, 1.0],
                sdpo_topk_token_ids=[None, None, None],
                sdpo_topk_logprobs=[None, None, None],
                sdpo_student_topk_token_ids=[[0, 0], [0, 0], [211, 212]],
                sdpo_student_topk_logprobs=[[0.0, 0.0], [0.0, 0.0], [-0.625, -1.5]],
            ),
        ],
    )

    with pytest.raises(ValueError, match="same weighted token positions"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_paired_support_width_mismatch(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_token_ids=[[0, 0], [111, 112, 113], [211, 212]],
                sdpo_student_topk_logprobs=[[0.0, 0.0], [-0.75, -1.25, -1.75], [-0.625, -1.5]],
            )
        ],
    )

    with pytest.raises(ValueError, match="support widths differ at weighted token 1"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_partial_student_row(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_token_ids=[[0, 0], [111, 112], [211, 212]],
                sdpo_student_topk_logprobs=[None, None, None],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_student_topk_logprobs\\[0\\] must be non-empty"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_partial_transport_support(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_topk_token_ids=[[0, 0], [101, 102], None],
                sdpo_topk_logprobs=[[0.0, 0.0], [-0.5, -2.0], None],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo support is missing at weighted token positions \\[2\\]"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_final_record_with_no_transport_support(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(),
            _record(
                sample_id="sample-b",
                sdpo_topk_token_ids=[None, None, None],
                sdpo_topk_logprobs=[None, None, None],
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="final SDPO records require transported teacher support at weighted token positions \\[1, 2\\]",
    ):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_partial_student_support(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_token_ids=[[0, 0], None, [211, 212]],
                sdpo_student_topk_logprobs=[[0.0, 0.0], None, [-0.625, -1.5]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_student support is missing at weighted token positions \\[1\\]"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_weighted_transport_placeholder_row(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_topk_token_ids=[[0, 0], [0, 0], [201, 202]],
                sdpo_topk_logprobs=[[0.0, 0.0], [0.0, 0.0], [-0.3, -1.5]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_topk_logprobs\\[1\\] looks like an unfilled placeholder row"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_non_integer_transport_topk_ids(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_topk_token_ids=[[0, 0], [101, "102"], [201, 202]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_topk_token_ids\\[1\\] must contain integer token ids"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_negative_transport_topk_ids(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_topk_token_ids=[[0, 0], [101, -102], [201, 202]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_topk_token_ids\\[1\\] must contain non-negative token ids"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_nonfinite_transport_logprob_row(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_topk_logprobs=[[0.0, 0.0], [float("nan"), -2.0], [-0.3, -1.5]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_topk_logprobs\\[1\\] must contain finite numeric values"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_transport_logprob_mass_above_one(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_topk_logprobs=[[0.0, 0.0], [-0.1, -2.0], [-0.3, -1.5]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_topk_logprobs\\[1\\] probability mass exceeds 1"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_integer_transport_logprob_row(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_topk_logprobs=[[0.0, 0.0], [-2, -3], [-0.3, -1.5]],
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="sdpo_topk_logprobs\\[1\\] must contain floating-point logprob values",
    ):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_weighted_student_placeholder_row(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_token_ids=[[0, 0], [0, 0], [211, 212]],
                sdpo_student_topk_logprobs=[[0.0, 0.0], [0.0, 0.0], [-0.625, -1.5]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_student_topk_logprobs\\[1\\] looks like an unfilled placeholder row"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_non_integer_student_topk_ids(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_token_ids=[[0, 0], [111, True], [211, 212]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_student_topk_token_ids\\[1\\] must contain integer token ids"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_nonfinite_student_logprob_row(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_logprobs=[[0.0, 0.0], [-0.75, float("inf")], [-0.625, -1.5]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_student_topk_logprobs\\[1\\] must contain finite numeric values"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_student_logprob_mass_above_one(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_logprobs=[[0.0, 0.0], [-0.1, -2.0], [-0.625, -1.5]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_student_topk_logprobs\\[1\\] probability mass exceeds 1"):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_integer_student_logprob_row(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_logprobs=[[0.0, 0.0], [-2, -3], [-0.625, -1.5]],
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="sdpo_student_topk_logprobs\\[1\\] must contain floating-point logprob values",
    ):
        verify_sdpo_token_exports(tmp_path)


def test_verify_sdpo_token_exports_rejects_legacy_schema(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(schema_version=1)])

    with pytest.raises(ValueError, match="expected schema_version == 2"):
        verify_sdpo_token_exports(tmp_path)


def _write_stable(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "STABLE").write_text("", encoding="utf-8")


def test_find_broadcast_step_dirs_accepts_output_dir(tmp_path):
    step_dir = tmp_path / "run_default" / "broadcasts" / "step_7"
    _write_stable(step_dir)

    assert find_broadcast_step_dirs(tmp_path) == [step_dir]


def test_find_broadcast_step_dirs_accepts_sibling_token_exports_step_dir(tmp_path):
    broadcast_step_dir = tmp_path / "run_default" / "broadcasts" / "step_7"
    export_step_dir = tmp_path / "run_default" / "token_exports" / "step_7"
    _write_stable(broadcast_step_dir)
    export_step_dir.mkdir(parents=True)

    assert find_broadcast_step_dirs(export_step_dir) == [broadcast_step_dir]


def test_find_broadcast_step_dirs_requires_matching_sibling_step(tmp_path):
    broadcast_step_dir = tmp_path / "run_default" / "broadcasts" / "step_9"
    export_step_dir = tmp_path / "run_default" / "token_exports" / "step_7"
    _write_stable(broadcast_step_dir)
    export_step_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No stable filesystem broadcast"):
        find_broadcast_step_dirs(export_step_dir)


def test_verify_sdpo_ema_broadcasts_accepts_teacher_role_artifacts(tmp_path):
    step_dir = tmp_path / "run_default" / "broadcasts" / "step_7"
    _write_stable(step_dir)
    _write_stable(step_dir / "sdpo_teacher")
    (step_dir / "sdpo_teacher" / "model.safetensors").write_text("weights", encoding="utf-8")

    stats = verify_sdpo_ema_broadcasts(tmp_path)

    assert stats.steps == 1
    assert stats.teacher_steps == 1
    assert stats.role == "sdpo_teacher"
    assert stats.step_names == ("step_7",)
    assert stats.step_keys == ("run_default:step_7",)


def _write_teacher_broadcast_step(path):
    _write_stable(path)
    _write_stable(path / "sdpo_teacher")
    (path / "sdpo_teacher" / "model.safetensors").write_text("weights", encoding="utf-8")


def test_verify_sdpo_smoke_artifacts_requires_ema_step_overlap(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)
    _write_teacher_broadcast_step(tmp_path / "run_default" / "broadcasts" / "step_7")

    stats = verify_sdpo_smoke_artifacts(tmp_path, require_ema_teacher=True, expected_topk=2)

    assert stats.token_exports.step_names == ("step_7",)
    assert stats.token_exports.paired_step_names == ("step_7",)
    assert stats.token_exports.matching_support_step_names == ("step_7",)
    assert stats.token_exports.student_preflight_step_names == ("step_7",)
    assert stats.token_exports.matched_support_sample_keys == ("run_default:step_7:sample-a",)
    assert stats.ema_broadcasts is not None
    assert stats.ema_broadcasts.step_names == ("step_7",)
    assert stats.ema_broadcasts.step_keys == ("run_default:step_7",)
    assert stats.matched_steps == ("step_7",)
    assert stats.matched_step_keys == ("run_default:step_7",)


def test_verify_sdpo_smoke_artifacts_allows_initial_export_but_requires_later_ema_overlap(tmp_path):
    initial_export = tmp_path / "run_default" / "token_exports" / "step_0" / "rank_0.jsonl"
    later_export = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(initial_export, [_student_preflight_record(), _student_final_record()])
    _write_jsonl(later_export, [_student_preflight_record(), _student_final_record()])
    _write_stable(initial_export.parent)
    _write_stable(later_export.parent)
    _write_teacher_broadcast_step(tmp_path / "run_default" / "broadcasts" / "step_7")

    stats = verify_sdpo_smoke_artifacts(tmp_path, require_ema_teacher=True, expected_topk=2)

    assert stats.token_exports.matching_support_step_names == ("step_0", "step_7")
    assert stats.matched_steps == ("step_7",)
    assert stats.matched_step_keys == ("run_default:step_7",)


def test_verify_sdpo_smoke_artifacts_accepts_multi_run_ema_broadcast_overlap(tmp_path):
    for run_id, step_name in (("run_a", "step_7"), ("run_b", "step_9")):
        export_file = tmp_path / run_id / "token_exports" / step_name / "rank_0.jsonl"
        _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
        _write_stable(export_file.parent)
        _write_teacher_broadcast_step(tmp_path / run_id / "broadcasts" / step_name)

    stats = verify_sdpo_smoke_artifacts(tmp_path, require_ema_teacher=True, expected_topk=2)

    assert stats.token_exports.matched_support_sample_keys == (
        "run_a:step_7:sample-a",
        "run_b:step_9:sample-a",
    )
    assert stats.ema_broadcasts is not None
    assert stats.ema_broadcasts.step_keys == ("run_a:step_7", "run_b:step_9")
    assert stats.matched_step_keys == ("run_a:step_7", "run_b:step_9")


def test_verify_sdpo_smoke_artifacts_rejects_ema_smoke_with_only_initial_matching_export(tmp_path):
    initial_export = tmp_path / "run_default" / "token_exports" / "step_0" / "rank_0.jsonl"
    _write_jsonl(initial_export, [_student_preflight_record(), _student_final_record()])
    _write_stable(initial_export.parent)
    _write_teacher_broadcast_step(tmp_path / "run_default" / "broadcasts" / "step_1")

    with pytest.raises(ValueError, match="no post-initial matching SDPO token-export steps"):
        verify_sdpo_smoke_artifacts(tmp_path, require_ema_teacher=True, expected_topk=2)


def test_verify_sdpo_smoke_artifacts_cli_accepts_valid_student_support_smoke(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)
    script = "scripts/verify_sdpo_smoke_artifacts.py"

    result = subprocess.run(
        [sys.executable, script, str(tmp_path), "--expected-topk", "2"],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Verified SDPO token exports" in result.stdout
    assert "student_preflight_rows=2" in result.stdout
    assert "distinct_teacher_logprob_rows=2" in result.stdout
    assert "rollout_is_weight_rows=2" in result.stdout
    assert "matched_support_samples=1" in result.stdout
    assert "matched_support_token_rows=2" in result.stdout
    assert "distinct_teacher_logprob_token_rows=2" in result.stdout
    assert "rollout_is_weight_token_rows=2" in result.stdout


def test_verify_sdpo_smoke_artifacts_cli_accepts_required_provenance(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)
    _write_smoke_provenance(tmp_path / "sdpo_smoke_provenance.txt")
    script = "scripts/verify_sdpo_smoke_artifacts.py"

    result = subprocess.run(
        [
            sys.executable,
            script,
            str(tmp_path),
            "--expected-topk",
            "2",
            "--require-provenance",
            "--expected-provenance-mode",
            "live",
            "--expected-provenance-config",
            "configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml",
        ],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"Verified SDPO smoke provenance: file={tmp_path / 'sdpo_smoke_provenance.txt'}" in result.stdout
    assert "mode=live" in result.stdout
    assert "config=configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml" in result.stdout
    assert "expected_topk=2" in result.stdout
    assert "Verified SDPO token exports" in result.stdout


@pytest.mark.parametrize(
    "path_selector",
    [
        lambda output_dir, export_file: output_dir / "run_default",
        lambda output_dir, export_file: output_dir / "run_default" / "token_exports",
        lambda output_dir, export_file: output_dir / "run_default" / "token_exports" / "step_7",
        lambda output_dir, export_file: output_dir / "run_default" / "broadcasts",
        lambda output_dir, export_file: output_dir / "run_default" / "broadcasts" / "step_7",
        lambda output_dir, export_file: export_file,
    ],
)
def test_verify_sdpo_smoke_artifacts_cli_finds_required_provenance_from_nested_artifact_paths(tmp_path, path_selector):
    export_file = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)
    (tmp_path / "run_default" / "broadcasts" / "step_7").mkdir(parents=True)
    _write_smoke_provenance(tmp_path / "sdpo_smoke_provenance.txt")
    script = "scripts/verify_sdpo_smoke_artifacts.py"
    artifact_path = path_selector(tmp_path, export_file)

    result = subprocess.run(
        [
            sys.executable,
            script,
            str(artifact_path),
            "--expected-topk",
            "2",
            "--require-provenance",
            "--expected-provenance-mode",
            "live",
            "--expected-provenance-config",
            "configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml",
        ],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Verified SDPO token exports" in result.stdout


def test_verify_sdpo_smoke_artifacts_cli_rejects_provenance_topk_mismatch(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)
    (tmp_path / "sdpo_smoke_provenance.txt").write_text(
        "sdpo_smoke_provenance_version=1\nmode=live\nexpected_topk=3\n",
        encoding="utf-8",
    )
    script = "scripts/verify_sdpo_smoke_artifacts.py"

    result = subprocess.run(
        [sys.executable, script, str(tmp_path), "--expected-topk", "2", "--require-provenance"],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "SDPO smoke provenance mismatch for expected_topk" in result.stderr


def test_verify_sdpo_smoke_artifacts_cli_rejects_provenance_mode_mismatch(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)
    (tmp_path / "sdpo_smoke_provenance.txt").write_text(
        "sdpo_smoke_provenance_version=1\nmode=ema\nexpected_topk=2\n",
        encoding="utf-8",
    )
    script = "scripts/verify_sdpo_smoke_artifacts.py"

    result = subprocess.run(
        [
            sys.executable,
            script,
            str(tmp_path),
            "--expected-topk",
            "2",
            "--require-provenance",
            "--expected-provenance-mode",
            "live",
        ],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "SDPO smoke provenance mismatch for mode" in result.stderr


def test_verify_sdpo_smoke_artifacts_cli_rejects_provenance_config_mismatch(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)
    (tmp_path / "sdpo_smoke_provenance.txt").write_text(
        "\n".join(
            [
                "sdpo_smoke_provenance_version=1",
                "mode=live",
                "config=configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml",
                "expected_topk=2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    script = "scripts/verify_sdpo_smoke_artifacts.py"

    result = subprocess.run(
        [
            sys.executable,
            script,
            str(tmp_path),
            "--expected-topk",
            "2",
            "--require-provenance",
            "--expected-provenance-config",
            "configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml",
        ],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "SDPO smoke provenance mismatch for config" in result.stderr


def test_verify_sdpo_smoke_artifacts_cli_rejects_reference_knob_mismatch(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)
    provenance_file = tmp_path / "sdpo_smoke_provenance.txt"
    _write_smoke_provenance(provenance_file)
    provenance_file.write_text(
        provenance_file.read_text(encoding="utf-8").replace(
            "orchestrator.algo.successful_demonstration_selection=batch_order",
            "orchestrator.algo.successful_demonstration_selection=highest_reward",
        ),
        encoding="utf-8",
    )
    script = "scripts/verify_sdpo_smoke_artifacts.py"

    result = subprocess.run(
        [
            sys.executable,
            script,
            str(tmp_path),
            "--expected-topk",
            "2",
            "--require-provenance",
            "--expected-provenance-mode",
            "live",
            "--expected-provenance-config",
            "configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml",
        ],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "SDPO smoke provenance mismatch for orchestrator.algo.successful_demonstration_selection" in result.stderr


def test_verify_sdpo_smoke_artifacts_cli_rejects_required_provenance_without_source_fingerprints(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)
    (tmp_path / "sdpo_smoke_provenance.txt").write_text(
        "\n".join(
            [
                "sdpo_smoke_provenance_version=1",
                "mode=live",
                "config=configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml",
                "expected_topk=2",
                "git_commit=abc123",
                "git_branch=codex/sdpo",
                f"git_diff_sha256={TEST_GIT_DIFF_SHA256}",
                f"git_cached_diff_sha256={TEST_GIT_CACHED_DIFF_SHA256}",
                "python_runner=uv run python",
                "rl_runner=uv run rl",
                "",
            ]
        ),
        encoding="utf-8",
    )
    script = "scripts/verify_sdpo_smoke_artifacts.py"

    result = subprocess.run(
        [sys.executable, script, str(tmp_path), "--expected-topk", "2", "--require-provenance"],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "SDPO smoke provenance is missing required field: git_untracked_manifest_sha256" in result.stderr


def test_verify_sdpo_smoke_artifacts_cli_rejects_required_provenance_without_untracked_manifest_section(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)
    manifest_lines = ["untrackedhash  scripts/sdpo-example.py"]
    (tmp_path / "sdpo_smoke_provenance.txt").write_text(
        "\n".join(
            [
                "sdpo_smoke_provenance_version=1",
                "mode=live",
                "config=configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml",
                "expected_topk=2",
                "git_commit=abc123",
                "git_branch=codex/sdpo",
                f"git_diff_sha256={TEST_GIT_DIFF_SHA256}",
                f"git_cached_diff_sha256={TEST_GIT_CACHED_DIFF_SHA256}",
                f"git_untracked_manifest_sha256={_manifest_hash(manifest_lines)}",
                "python_runner=uv run python",
                "rl_runner=uv run rl",
                "",
            ]
        ),
        encoding="utf-8",
    )
    script = "scripts/verify_sdpo_smoke_artifacts.py"

    result = subprocess.run(
        [sys.executable, script, str(tmp_path), "--expected-topk", "2", "--require-provenance"],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "SDPO smoke provenance is missing required field: git_untracked_manifest_begin" in result.stderr


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("git_commit", "unknown"),
        ("git_branch", "unknown"),
        ("git_diff_sha256", "unavailable"),
        ("git_cached_diff_sha256", "unavailable"),
        ("git_untracked_manifest_sha256", "unavailable"),
    ],
)
def test_verify_sdpo_smoke_artifacts_cli_rejects_placeholder_required_provenance_values(tmp_path, field, value):
    export_file = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)
    manifest_lines = ["untrackedhash  scripts/sdpo-example.py"]
    provenance_values = {
        "sdpo_smoke_provenance_version": "1",
        "mode": "live",
        "config": "configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml",
        "expected_topk": "2",
        "git_commit": "abc123",
        "git_branch": "codex/sdpo",
        "git_diff_sha256": TEST_GIT_DIFF_SHA256,
        "git_cached_diff_sha256": TEST_GIT_CACHED_DIFF_SHA256,
        "git_untracked_manifest_sha256": _manifest_hash(manifest_lines),
        "python_runner": "uv run python",
        "rl_runner": "uv run rl",
    }
    provenance_values[field] = value
    (tmp_path / "sdpo_smoke_provenance.txt").write_text(
        "\n".join(
            [
                *[f"{key}={item}" for key, item in provenance_values.items()],
                "git_untracked_manifest_begin",
                *manifest_lines,
                "git_untracked_manifest_end",
                "git_status_short_begin",
                "git_status_short_end",
                "",
            ]
        ),
        encoding="utf-8",
    )
    script = "scripts/verify_sdpo_smoke_artifacts.py"

    result = subprocess.run(
        [sys.executable, script, str(tmp_path), "--expected-topk", "2", "--require-provenance"],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert f"SDPO smoke provenance field {field} must not be '{value}'" in result.stderr


def test_verify_sdpo_smoke_artifacts_cli_rejects_required_provenance_without_git_status_end(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)
    manifest_lines = ["untrackedhash  scripts/sdpo-example.py"]
    (tmp_path / "sdpo_smoke_provenance.txt").write_text(
        "\n".join(
            [
                "sdpo_smoke_provenance_version=1",
                "mode=live",
                "config=configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml",
                "expected_topk=2",
                "git_commit=abc123",
                "git_branch=codex/sdpo",
                f"git_diff_sha256={TEST_GIT_DIFF_SHA256}",
                f"git_cached_diff_sha256={TEST_GIT_CACHED_DIFF_SHA256}",
                f"git_untracked_manifest_sha256={_manifest_hash(manifest_lines)}",
                "python_runner=uv run python",
                "rl_runner=uv run rl",
                "git_untracked_manifest_begin",
                *manifest_lines,
                "git_untracked_manifest_end",
                "git_status_short_begin",
                " M some-local-file.py",
                "",
            ]
        ),
        encoding="utf-8",
    )
    script = "scripts/verify_sdpo_smoke_artifacts.py"

    result = subprocess.run(
        [sys.executable, script, str(tmp_path), "--expected-topk", "2", "--require-provenance"],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Invalid SDPO smoke provenance:" in result.stderr
    assert "is missing git_status_short_end" in result.stderr


def test_verify_sdpo_smoke_artifacts_cli_rejects_untracked_manifest_hash_mismatch(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)
    provenance_file = tmp_path / "sdpo_smoke_provenance.txt"
    _write_smoke_provenance(provenance_file)
    provenance_file.write_text(
        provenance_file.read_text(encoding="utf-8").replace(
            "git_untracked_manifest_sha256=0065e292b57dc76ce29c07662d2d4f7266fdd68f644baaa0c5c337f30a44d52f",
            "git_untracked_manifest_sha256=0000000000000000000000000000000000000000000000000000000000000000",
        ),
        encoding="utf-8",
    )
    script = "scripts/verify_sdpo_smoke_artifacts.py"

    result = subprocess.run(
        [sys.executable, script, str(tmp_path), "--expected-topk", "2", "--require-provenance"],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "SDPO smoke provenance mismatch for git_untracked_manifest_sha256" in result.stderr


def test_verify_sdpo_smoke_artifacts_cli_rejects_missing_required_provenance(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)
    script = "scripts/verify_sdpo_smoke_artifacts.py"

    result = subprocess.run(
        [sys.executable, script, str(tmp_path), "--expected-topk", "2", "--require-provenance"],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Missing SDPO smoke provenance file" in result.stderr


def test_verify_sdpo_smoke_artifacts_cli_requires_expected_topk(tmp_path):
    script = "scripts/verify_sdpo_smoke_artifacts.py"

    result = subprocess.run(
        [sys.executable, script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--expected-topk" in result.stderr


def test_verify_sdpo_smoke_artifacts_cli_help_documents_strict_preflight_contract():
    script = "scripts/verify_sdpo_smoke_artifacts.py"

    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    help_text = " ".join(result.stdout.split())
    assert "stable token exports" in help_text
    assert "preflight-only student support" in help_text
    assert "final transported SDPO support matched on run, step, sample_id" in help_text
    assert "--expected-topk" in help_text
    assert "Required. Require every supported SDPO top-k row" in help_text
    assert "--require-ema-teacher" in help_text
    assert "non-empty teacher artifacts" in help_text
    assert "--require-provenance" in help_text
    assert "sdpo_smoke_provenance.txt" in help_text
    assert "--expected-provenance-mode" in help_text
    assert "--expected-provenance-config" in help_text
    assert "rollout-IS ratio evidence" in help_text
    assert "sdpo_rollout_is_weights matching token-level truncated rollout-IS" in help_text


def test_verify_sdpo_token_exports_cli_reports_matched_support_token_rows(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)
    script = "scripts/verify_sdpo_token_exports.py"

    result = subprocess.run(
        [
            sys.executable,
            script,
            str(tmp_path),
            "--require-stable",
            "--require-student-preflight",
            "--require-importance-ratio-evidence",
            "--expected-topk",
            "2",
            "--rollout-is-threshold",
            "2.0",
            "--rollout-is",
            "token",
        ],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Verified SDPO token exports" in result.stdout
    assert "distinct_teacher_logprob_rows=2" in result.stdout
    assert "importance_ratio_rows=2" in result.stdout
    assert "rollout_is_weight_rows=2" in result.stdout
    assert "matched_support_samples=1" in result.stdout
    assert "matched_support_token_rows=2" in result.stdout
    assert "distinct_teacher_logprob_token_rows=2" in result.stdout
    assert "importance_ratio_token_rows=2" in result.stdout
    assert "rollout_is_weight_token_rows=2" in result.stdout


def test_verify_sdpo_token_exports_cli_rejects_rollout_is_weight_above_threshold(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_rollout_is_weights=[0.0, 2.01, 1.25])])
    script = "scripts/verify_sdpo_token_exports.py"

    result = subprocess.run(
        [
            sys.executable,
            script,
            str(tmp_path),
            "--expected-topk",
            "2",
            "--rollout-is-threshold",
            "2.0",
        ],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "sdpo_rollout_is_weights[1] exceeds rollout_is_threshold=2.0" in result.stderr


def test_verify_sdpo_token_exports_cli_help_documents_token_row_matching():
    script = "scripts/verify_sdpo_token_exports.py"

    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    help_text = " ".join(result.stdout.split())
    assert "same training step, sample_id, env-aware sample signature, and weighted token rows" in help_text
    assert "final teacher logprobs distinct from trainer" in help_text
    assert "student logprobs" in help_text
    assert "--require-importance-ratio-evidence" in help_text
    assert "log_importance_ratio, importance_ratio, prob_delta" in help_text
    assert "Final SDPO records must also carry transported teacher support" in help_text
    assert "--rollout-is-threshold" in help_text
    assert "every final weighted SDPO row" in help_text
    assert "rollout-IS truncation threshold" in help_text
    assert "--rollout-is" in help_text
    assert "selected truncated rollout-IS mode" in help_text


def test_verify_sdpo_smoke_artifacts_requires_ema_overlap_with_paired_export_step(tmp_path):
    paired_export = tmp_path / "run_default" / "token_exports" / "step_9" / "rank_0.jsonl"
    _write_jsonl(paired_export, [_student_preflight_record(), _student_final_record()])
    _write_stable(paired_export.parent)
    _write_teacher_broadcast_step(tmp_path / "run_default" / "broadcasts" / "step_7")

    with pytest.raises(ValueError, match="matching SDPO token exports and EMA teacher broadcasts do not overlap"):
        verify_sdpo_smoke_artifacts(tmp_path, require_ema_teacher=True, expected_topk=2)


def test_verify_sdpo_smoke_artifacts_requires_ema_overlap_with_same_run_step(tmp_path):
    export_file = tmp_path / "run_a" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)
    _write_teacher_broadcast_step(tmp_path / "run_b" / "broadcasts" / "step_7")

    with pytest.raises(ValueError, match="do not overlap on any run/step"):
        verify_sdpo_smoke_artifacts(tmp_path, require_ema_teacher=True, expected_topk=2)


def test_verify_sdpo_smoke_artifacts_rejects_non_overlapping_ema_steps(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    _write_jsonl(export_file, [_student_preflight_record(), _student_final_record()])
    _write_stable(export_file.parent)
    _write_teacher_broadcast_step(tmp_path / "run_default" / "broadcasts" / "step_9")

    with pytest.raises(ValueError, match="do not overlap"):
        verify_sdpo_smoke_artifacts(tmp_path, require_ema_teacher=True, expected_topk=2)


def test_verify_sdpo_smoke_artifacts_rejects_missing_later_ema_broadcast_step(tmp_path):
    step_7_export = tmp_path / "run_default" / "token_exports" / "step_7" / "rank_0.jsonl"
    step_9_export = tmp_path / "run_default" / "token_exports" / "step_9" / "rank_0.jsonl"
    _write_jsonl(step_7_export, [_student_preflight_record(), _student_final_record()])
    _write_jsonl(step_9_export, [_student_preflight_record(), _student_final_record()])
    _write_stable(step_7_export.parent)
    _write_stable(step_9_export.parent)
    _write_teacher_broadcast_step(tmp_path / "run_default" / "broadcasts" / "step_7")

    with pytest.raises(ValueError, match="missing EMA teacher broadcasts: \\['run_default:step_9'\\]"):
        verify_sdpo_smoke_artifacts(tmp_path, require_ema_teacher=True, expected_topk=2)


def test_verify_sdpo_ema_broadcasts_rejects_missing_teacher_stable(tmp_path):
    step_dir = tmp_path / "run_default" / "broadcasts" / "step_7"
    _write_stable(step_dir)

    with pytest.raises(ValueError, match="missing sdpo_teacher/STABLE"):
        verify_sdpo_ema_broadcasts(tmp_path)


def test_verify_sdpo_ema_broadcasts_rejects_empty_teacher_marker(tmp_path):
    step_dir = tmp_path / "run_default" / "broadcasts" / "step_7"
    _write_stable(step_dir)
    _write_stable(step_dir / "sdpo_teacher")

    with pytest.raises(ValueError, match="no non-empty teacher model artifacts"):
        verify_sdpo_ema_broadcasts(tmp_path)


def test_verify_sdpo_ema_broadcasts_rejects_zero_byte_teacher_artifacts(tmp_path):
    step_dir = tmp_path / "run_default" / "broadcasts" / "step_7"
    _write_stable(step_dir)
    _write_stable(step_dir / "sdpo_teacher")
    (step_dir / "sdpo_teacher" / "model.safetensors").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no non-empty teacher model artifacts"):
        verify_sdpo_ema_broadcasts(tmp_path)


def test_verify_sdpo_ema_broadcasts_rejects_directory_only_teacher_artifacts(tmp_path):
    step_dir = tmp_path / "run_default" / "broadcasts" / "step_7"
    _write_stable(step_dir)
    _write_stable(step_dir / "sdpo_teacher")
    (step_dir / "sdpo_teacher" / "empty-subdir").mkdir()

    with pytest.raises(ValueError, match="no non-empty teacher model artifacts"):
        verify_sdpo_ema_broadcasts(tmp_path)
