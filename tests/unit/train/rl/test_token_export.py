import json
import shutil

import pytest
import torch

from prime_rl.configs.trainer import DefaultLossConfig, SDPOComponentConfig
from prime_rl.trainer.rl.sdpo_export_verify import verify_sdpo_smoke_artifacts, verify_sdpo_token_exports
from prime_rl.trainer.rl.sdpo_train_support import select_sdpo_student_topk_support
from prime_rl.trainer.rl.token_export import TokenExporter, token_export_ready_run_ids


def _sdpo_micro_batch():
    return {
        "input_ids": torch.tensor([10, 11, 12, 0]),
        "position_ids": torch.tensor([0, 1, 2, 0]),
        "loss_mask": torch.tensor([False, True, True, False]),
        "temperatures": torch.tensor([1.0, 0.75, 0.5, 1.0]),
        "advantages": torch.tensor([0.0, 1.0, 1.0, 0.0]),
        "rewards": None,
        "inference_logprobs": torch.tensor([0.0, -0.2, -0.3, 0.0]),
        "rl_weights": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        "ref_kl_weights": None,
        "sdpo_weights": torch.tensor([0.0, 1.0, 1.0, 0.0]),
        "sdpo_rollout_is_weights": torch.tensor([0.0, 0.75, 1.25, 0.0]),
        "sdpo_topk_token_ids": torch.tensor([[0, 0], [101, 102], [201, 202], [0, 0]]),
        "sdpo_topk_logprobs": torch.tensor([[0.0, 0.0], [-0.5, -2.0], [-0.3, -1.5], [0.0, 0.0]]),
        "env_names": ["sdpo-env", "sdpo-env", "sdpo-env", ""],
        "sample_ids": ["sample-a"],
    }


def _sdpo_model_output():
    return {
        "logprobs": torch.tensor([0.0, -0.25, -0.35, 0.0]),
        "entropy": torch.tensor([0.0, 0.1, 0.2, 0.0]),
        "sdpo_student_topk_token_ids": torch.tensor([[[0, 0], [111, 112], [211, 212], [0, 0]]]),
        "sdpo_student_topk_logprobs": torch.tensor([[[0.0, 0.0], [-0.75, -1.25], [-0.625, -1.5], [0.0, 0.0]]]),
    }


def _export_sdpo_batch(exporter: TokenExporter, *, step: int = 7, micro_step: int = 0) -> None:
    exporter.export(
        step=step,
        micro_step=micro_step,
        micro_batch=_sdpo_micro_batch(),
        model_output=_sdpo_model_output(),
        sequence_lengths=[4],
        loss_config=DefaultLossConfig(),
    )


def _export_sdpo_student_preflight_batch(exporter: TokenExporter, *, step: int = 7, micro_step: int = 0) -> None:
    micro_batch = _sdpo_micro_batch()
    micro_batch["sdpo_topk_token_ids"] = None
    micro_batch["sdpo_topk_logprobs"] = None
    micro_batch["preflight_only"] = True
    exporter.export(
        step=step,
        micro_step=micro_step,
        micro_batch=micro_batch,
        model_output=_sdpo_model_output(),
        sequence_lengths=[4],
        loss_config=DefaultLossConfig(),
    )


def _export_sdpo_student_final_batch(exporter: TokenExporter, *, step: int = 7, micro_step: int = 1) -> None:
    micro_batch = _sdpo_micro_batch()
    micro_batch["sdpo_rollout_is_weights"] = None
    micro_batch["sdpo_topk_token_ids"] = torch.tensor([[0, 0], [111, 112], [211, 212], [0, 0]])
    micro_batch["sdpo_topk_logprobs"] = torch.tensor([[0.0, 0.0], [-3.0, -13.0], [-4.0, -14.0], [0.0, 0.0]])
    exporter.export(
        step=step,
        micro_step=micro_step,
        micro_batch=micro_batch,
        model_output=_sdpo_model_output(),
        sequence_lengths=[4],
        loss_config=DefaultLossConfig(),
        sdpo_loss_config=SDPOComponentConfig(rollout_is="token", rollout_is_threshold=2.0),
    )


def test_token_export_ready_run_ids_respects_preflight_completion() -> None:
    class FakeManager:
        idx_2_id = {0: "run-a", 1: "run-b"}
        ready_to_update_idxs = [1, 2]

    micro_batches = [
        {"run_id": "run-a", "preflight_step_complete": True},
        {"run_id": "run-b", "preflight_step_complete": False},
        {"run_id": None, "preflight_step_complete": True},
    ]

    assert token_export_ready_run_ids(micro_batches, preflight_only=True, multi_run_manager=FakeManager()) == {"run-a"}
    assert token_export_ready_run_ids(micro_batches, preflight_only=False, multi_run_manager=FakeManager()) == {"run-b"}


def test_token_export_ready_run_ids_rejects_mixed_preflight_completion_for_same_run() -> None:
    micro_batches = [
        {"run_id": "run-a", "preflight_step_complete": False},
        {"run_id": "run-a", "preflight_step_complete": True},
    ]

    with pytest.raises(ValueError, match="disagrees across micro batches"):
        token_export_ready_run_ids(micro_batches, preflight_only=True, multi_run_manager=object())


@pytest.mark.parametrize("preflight_step_complete", ["true", 1, None])
def test_token_export_ready_run_ids_rejects_non_boolean_preflight_completion(preflight_step_complete) -> None:
    with pytest.raises(ValueError, match="preflight_step_complete must be a boolean"):
        token_export_ready_run_ids(
            [{"run_id": "run-a", "preflight_step_complete": preflight_step_complete}],
            preflight_only=True,
            multi_run_manager=object(),
        )


@pytest.mark.parametrize("run_id", ["", 123])
def test_token_export_ready_run_ids_rejects_malformed_preflight_run_id(run_id) -> None:
    with pytest.raises(ValueError, match="run_id to be a non-empty string"):
        token_export_ready_run_ids(
            [{"run_id": run_id, "preflight_step_complete": True}],
            preflight_only=True,
            multi_run_manager=object(),
        )


def test_token_export_preserves_sdpo_weight_streams(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)

    try:
        _export_sdpo_batch(exporter)
    finally:
        exporter.close()

    export_path = tmp_path / "token_exports" / "step_7" / "rank_0.jsonl"
    record = json.loads(export_path.read_text().strip())

    assert record["schema_version"] == 2
    assert record["sample_id"] == "sample-a"
    assert record["preflight_only"] is False
    assert record["token_ids"] == [10, 11, 12]
    assert record["loss_mask"] == [False, True, True]
    assert record["temperatures"] == [1.0, 0.75, 0.5]
    assert record["sdpo_weights"] == [0.0, 1.0, 1.0]
    assert record["sdpo_rollout_is_weights"] == [0.0, 0.75, 1.25]
    assert record["sdpo_topk_token_ids"] == [[0, 0], [101, 102], [201, 202]]
    assert record["sdpo_topk_logprobs"] == [[0.0, 0.0], [-0.5, -2.0], [-0.30000001192092896, -1.5]]
    assert record["sdpo_student_topk_token_ids"] == [[0, 0], [111, 112], [211, 212]]
    assert record["sdpo_student_topk_logprobs"] == [[0.0, 0.0], [-0.75, -1.25], [-0.625, -1.5]]


def test_token_export_derives_non_normalized_rollout_is_weights_for_sdpo_loss_config(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    micro_batch["sdpo_rollout_is_weights"] = None
    model_output = _sdpo_model_output()
    model_output["logprobs"] = torch.tensor(
        [0.0, -0.2 + torch.log(torch.tensor(3.0)), -0.3 + torch.log(torch.tensor(0.25)), 0.0]
    )

    try:
        exporter.export(
            step=7,
            micro_step=0,
            micro_batch=micro_batch,
            model_output=model_output,
            sequence_lengths=[4],
            loss_config=DefaultLossConfig(),
            sdpo_loss_config=SDPOComponentConfig(rollout_is="token", rollout_is_threshold=2.0),
        )
        exporter.mark_stable()
    finally:
        exporter.close()

    record = json.loads((tmp_path / "token_exports" / "step_7" / "rank_0.jsonl").read_text().strip())

    assert record["sdpo_rollout_is_weights"] == [0.0, 2.0, 0.25]
    verify_sdpo_token_exports(
        tmp_path,
        require_stable=True,
        require_importance_ratio_evidence=True,
        expected_topk=2,
        rollout_is_threshold=2.0,
    )


def test_token_export_derives_sequence_rollout_is_weights_per_packed_sequence(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)
    log_two = torch.log(torch.tensor(2.0))
    log_half = torch.log(torch.tensor(0.5))
    micro_batch = {
        "input_ids": torch.tensor([10, 11, 12, 20, 21, 22]),
        "position_ids": torch.tensor([0, 1, 2, 0, 1, 2]),
        "loss_mask": torch.tensor([False, True, True, False, True, True]),
        "temperatures": torch.ones(6),
        "advantages": torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
        "rewards": None,
        "inference_logprobs": torch.tensor([0.0, -0.2, -0.3, 0.0, -0.4, -0.5]),
        "rl_weights": torch.zeros(6),
        "ref_kl_weights": None,
        "sdpo_weights": torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
        "sdpo_rollout_is_weights": None,
        "sdpo_topk_token_ids": torch.tensor([[0, 0], [101, 102], [201, 202], [0, 0], [301, 302], [401, 402]]),
        "sdpo_topk_logprobs": torch.tensor(
            [[0.0, 0.0], [-0.5, -2.0], [-0.3, -1.5], [0.0, 0.0], [-0.7, -1.7], [-0.8, -1.8]]
        ),
        "env_names": ["sdpo-env"] * 6,
        "sample_ids": ["sample-a", "sample-b"],
    }
    model_output = {
        "logprobs": micro_batch["inference_logprobs"] + torch.tensor([0.0, log_two, log_two, 0.0, log_half, log_half]),
        "entropy": torch.zeros(6),
        "sdpo_student_topk_token_ids": torch.tensor([[[0, 0], [111, 112], [211, 212], [0, 0], [311, 312], [411, 412]]]),
        "sdpo_student_topk_logprobs": torch.tensor(
            [[[0.0, 0.0], [-0.75, -1.25], [-0.625, -1.5], [0.0, 0.0], [-0.9, -1.9], [-1.0, -2.0]]]
        ),
    }

    try:
        exporter.export(
            step=7,
            micro_step=0,
            micro_batch=micro_batch,
            model_output=model_output,
            sequence_lengths=[3, 3],
            loss_config=DefaultLossConfig(),
            sdpo_loss_config=SDPOComponentConfig(rollout_is="sequence", rollout_is_threshold=2.0),
        )
        exporter.mark_stable()
    finally:
        exporter.close()

    records = [
        json.loads(line) for line in (tmp_path / "token_exports" / "step_7" / "rank_0.jsonl").read_text().splitlines()
    ]

    assert records[0]["sample_id"] == "sample-a"
    assert records[0]["sdpo_rollout_is_weights"] == [0.0, 2.0, 2.0]
    assert records[1]["sample_id"] == "sample-b"
    assert records[1]["sdpo_rollout_is_weights"] == [0.0, 0.25, 0.25]
    verify_sdpo_token_exports(
        tmp_path,
        require_stable=True,
        require_importance_ratio_evidence=True,
        expected_topk=2,
        rollout_is_threshold=2.0,
    )


def test_token_export_does_not_derive_batch_normalized_rollout_is_weights_without_sequence_boundaries(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    micro_batch["sdpo_rollout_is_weights"] = None

    try:
        exporter.export(
            step=7,
            micro_step=0,
            micro_batch=micro_batch,
            model_output=_sdpo_model_output(),
            sequence_lengths=[4],
            loss_config=DefaultLossConfig(),
            sdpo_loss_config=SDPOComponentConfig(rollout_is="token", rollout_is_batch_normalize=True),
        )
    finally:
        exporter.close()

    record = json.loads((tmp_path / "token_exports" / "step_7" / "rank_0.jsonl").read_text().strip())

    assert record["sdpo_rollout_is_weights"] == [None, None, None]


def test_token_exported_sdpo_rows_pass_artifact_verifier(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)

    try:
        _export_sdpo_batch(exporter)
        exporter.mark_stable()
    finally:
        exporter.close()

    stats = verify_sdpo_token_exports(
        tmp_path,
        require_stable=True,
        require_importance_ratio_evidence=True,
        expected_topk=2,
    )

    assert stats.files == 1
    assert stats.records == 1
    assert stats.sdpo_records == 1
    assert stats.transported_rows == 2
    assert stats.student_rows == 2
    assert stats.paired_rows == 2
    assert stats.importance_ratio_rows == 2
    assert stats.temperature_rows == 2
    assert stats.sample_id_records == 1
    assert stats.stable_steps == 1


@pytest.mark.parametrize(
    ("sequence_lengths", "expected_error"),
    [
        ([], "sequence_lengths must contain at least one sequence"),
        ([True], r"sequence_lengths\[0\] must be an integer"),
        ([0, 4], r"sequence_lengths\[0\] must be positive"),
        ([3], "sequence_lengths must sum to flattened token length"),
        ([5], "sequence_lengths must sum to flattened token length"),
    ],
)
def test_token_export_rejects_malformed_sequence_lengths(tmp_path, sequence_lengths, expected_error):
    exporter = TokenExporter(tmp_path, rank=0)

    try:
        with pytest.raises(ValueError, match=expected_error):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=_sdpo_micro_batch(),
                model_output=_sdpo_model_output(),
                sequence_lengths=sequence_lengths,
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


@pytest.mark.parametrize(
    ("field", "bad_tensor", "expected_error"),
    [
        ("input_ids", torch.tensor([10.0, 11.0, 12.0, 0.0]), "input_ids must contain integer token ids"),
        ("input_ids", torch.tensor([10, -11, 12, 0]), "input_ids must contain non-negative token ids"),
        ("position_ids", torch.tensor([0.0, 1.0, 2.0, 0.0]), "position_ids must contain integer token ids"),
        ("position_ids", torch.tensor([0, -1, 2, 0]), "position_ids must contain non-negative token ids"),
        ("loss_mask", torch.tensor([0, 1, 1, 0]), "loss_mask must be a boolean tensor"),
    ],
)
def test_token_export_rejects_malformed_identity_and_mask_tensors(tmp_path, field, bad_tensor, expected_error):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    micro_batch[field] = bad_tensor

    try:
        with pytest.raises(ValueError, match=expected_error):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=_sdpo_model_output(),
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


@pytest.mark.parametrize(
    ("source", "field", "bad_tensor", "expected_error"),
    [
        (
            "micro_batch",
            "temperatures",
            torch.tensor([1.0, float("nan"), 0.5, 1.0]),
            "temperatures must contain finite values",
        ),
        (
            "micro_batch",
            "advantages",
            torch.tensor([0.0, 1.0, float("inf"), 0.0]),
            "advantages must contain finite values",
        ),
        ("micro_batch", "rewards", torch.tensor([0.0, float("nan"), 1.0, 0.0]), "rewards must contain finite values"),
        (
            "micro_batch",
            "ce_weights",
            torch.tensor([False, True, False, False]),
            "ce_weights must contain numeric values",
        ),
        ("model_output", "entropy", torch.tensor([0.0, 0.1, float("inf"), 0.0]), "entropy must contain finite values"),
    ],
)
def test_token_export_rejects_malformed_float_evidence_tensors(tmp_path, source, field, bad_tensor, expected_error):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    model_output = _sdpo_model_output()
    if source == "micro_batch":
        micro_batch[field] = bad_tensor
    else:
        model_output[field] = bad_tensor

    try:
        with pytest.raises(ValueError, match=expected_error):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=model_output,
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


@pytest.mark.parametrize(
    ("source", "field", "bad_tensor", "expected_error"),
    [
        (
            "micro_batch",
            "inference_logprobs",
            torch.tensor([0, -1, -1, 0]),
            "inference_logprobs must use a floating-point dtype",
        ),
        (
            "model_output",
            "logprobs",
            torch.tensor([0, -1, -1, 0]),
            "trainer_logprobs must use a floating-point dtype",
        ),
        (
            "model_output",
            "entropy",
            torch.tensor([0, 1, 1, 0]),
            "entropy must use a floating-point dtype",
        ),
    ],
)
def test_token_export_rejects_non_floating_probability_evidence_tensors(
    tmp_path, source, field, bad_tensor, expected_error
):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    model_output = _sdpo_model_output()
    if source == "micro_batch":
        micro_batch[field] = bad_tensor
    else:
        model_output[field] = bad_tensor

    try:
        with pytest.raises(ValueError, match=expected_error):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=model_output,
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


def test_token_export_rejects_nonfinite_computed_importance_ratio_evidence(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)
    model_output = _sdpo_model_output()
    model_output["logprobs"] = torch.tensor([0.0, 1000.0, -0.35, 0.0])

    try:
        with pytest.raises(ValueError, match="mismatch_kl must contain finite values"):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=_sdpo_micro_batch(),
                model_output=model_output,
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


def test_token_export_preserves_strict_sdpo_preflight_then_final_smoke_pattern(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)
    step_dir = tmp_path / "token_exports" / "step_7"
    export_path = step_dir / "rank_0.jsonl"

    try:
        _export_sdpo_student_preflight_batch(exporter, step=7, micro_step=0)
        exporter.mark_stable()

        assert (step_dir / "STABLE").exists()

        _export_sdpo_student_final_batch(exporter, step=7, micro_step=1)

        assert not (step_dir / "STABLE").exists()

        exporter.mark_stable()
    finally:
        exporter.close()

    records = [json.loads(line) for line in export_path.read_text().splitlines()]
    assert [record["micro_step"] for record in records] == [0, 1]
    assert [record["preflight_only"] for record in records] == [True, False]
    assert records[0]["sdpo_topk_token_ids"] == [None, None, None]
    assert records[1]["sdpo_topk_token_ids"] == [[0, 0], [111, 112], [211, 212]]

    stats = verify_sdpo_smoke_artifacts(tmp_path, expected_topk=2)

    assert stats.token_exports.student_preflight_rows == 2
    assert stats.token_exports.matching_support_rows == 2
    assert stats.token_exports.importance_ratio_rows == 2
    assert stats.token_exports.importance_ratio_row_keys == (
        f"{tmp_path.name}:step_7:sample-a:token-1",
        f"{tmp_path.name}:step_7:sample-a:token-2",
    )
    assert stats.token_exports.matched_support_row_keys == (
        f"{tmp_path.name}:step_7:sample-a:token-1",
        f"{tmp_path.name}:step_7:sample-a:token-2",
    )


def test_token_export_rejects_transported_teacher_support_on_preflight_record(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    micro_batch["preflight_only"] = True

    try:
        with pytest.raises(ValueError, match="preflight-only SDPO token exports must not carry transported teacher"):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=_sdpo_model_output(),
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


def test_token_export_rejects_final_sdpo_record_without_transported_teacher_support(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    micro_batch["sdpo_topk_token_ids"] = None
    micro_batch["sdpo_topk_logprobs"] = None

    try:
        with pytest.raises(ValueError, match="final SDPO token exports require transported teacher top-k support"):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=_sdpo_model_output(),
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


def test_token_export_rejects_final_sdpo_record_with_weighted_placeholder_teacher_support(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    micro_batch["sdpo_topk_token_ids"] = torch.tensor([[0, 0], [0, 0], [201, 202], [0, 0]])
    micro_batch["sdpo_topk_logprobs"] = torch.tensor([[0.0, 0.0], [0.0, 0.0], [-0.3, -1.5], [0.0, 0.0]])

    try:
        with pytest.raises(ValueError, match="final SDPO token exports require transported teacher top-k support"):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=_sdpo_model_output(),
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


def test_token_export_masks_dense_student_support_to_sdpo_weighted_rows(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)
    logits = torch.tensor(
        [
            [
                [0.0, 4.0, 1.0, -1.0, 2.0],
                [0.5, -0.5, 3.0, 2.0, 0.0],
                [1.0, 2.5, -0.5, 0.25, 3.5],
                [2.0, -1.0, 1.5, 0.0, 0.5],
            ]
        ]
    )
    dense_topk_token_ids, dense_topk_logprobs = select_sdpo_student_topk_support(
        logits,
        torch.tensor([[1.0, 0.75, 0.5, 1.0]]),
        topk=2,
    )
    assert dense_topk_logprobs[0, 0].tolist() != [0.0, 0.0]
    assert dense_topk_logprobs[0, 3].tolist() != [0.0, 0.0]

    model_output = _sdpo_model_output()
    model_output["sdpo_student_topk_token_ids"] = dense_topk_token_ids
    model_output["sdpo_student_topk_logprobs"] = dense_topk_logprobs
    micro_batch = _sdpo_micro_batch()
    micro_batch["sdpo_topk_token_ids"] = None
    micro_batch["sdpo_topk_logprobs"] = None
    micro_batch["preflight_only"] = True

    try:
        exporter.export(
            step=7,
            micro_step=0,
            micro_batch=micro_batch,
            model_output=model_output,
            sequence_lengths=[4],
            loss_config=DefaultLossConfig(),
        )
        exporter.mark_stable()
    finally:
        exporter.close()

    record = json.loads((tmp_path / "token_exports" / "step_7" / "rank_0.jsonl").read_text().strip())

    assert record["sdpo_topk_token_ids"] == [None, None, None]
    assert record["sdpo_student_topk_token_ids"][0] == [0, 0]
    assert record["sdpo_student_topk_logprobs"][0] == [0.0, 0.0]
    assert record["sdpo_student_topk_token_ids"][1] == dense_topk_token_ids[0, 1].tolist()
    assert record["sdpo_student_topk_token_ids"][2] == dense_topk_token_ids[0, 2].tolist()
    assert len(record["sdpo_student_topk_token_ids"]) == 3


@pytest.mark.parametrize("bad_sample_id", ["", "   ", 123])
def test_token_export_rejects_malformed_sample_ids(tmp_path, bad_sample_id):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    micro_batch["sample_ids"] = [bad_sample_id]

    try:
        with pytest.raises(ValueError, match=r"sample_ids\[0\] must be null or a non-empty string"):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=_sdpo_model_output(),
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


@pytest.mark.parametrize("preflight_only", ["False", 0, None])
def test_token_export_rejects_non_boolean_preflight_flag(tmp_path, preflight_only):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    micro_batch["preflight_only"] = preflight_only

    try:
        with pytest.raises(ValueError, match="preflight_only must be a boolean"):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=_sdpo_model_output(),
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


def test_token_export_allows_null_sample_id_for_non_sdpo_record(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    micro_batch["sample_ids"] = [None]
    micro_batch["sdpo_weights"] = torch.zeros_like(micro_batch["sdpo_weights"])
    micro_batch["sdpo_topk_token_ids"] = None
    micro_batch["sdpo_topk_logprobs"] = None
    micro_batch["sdpo_rollout_is_weights"] = None
    model_output = _sdpo_model_output()
    model_output["sdpo_student_topk_token_ids"] = None
    model_output["sdpo_student_topk_logprobs"] = None

    try:
        exporter.export(
            step=7,
            micro_step=0,
            micro_batch=micro_batch,
            model_output=model_output,
            sequence_lengths=[4],
            loss_config=DefaultLossConfig(),
        )
    finally:
        exporter.close()

    record = json.loads((tmp_path / "token_exports" / "step_7" / "rank_0.jsonl").read_text().strip())

    assert record["sample_id"] is None
    assert record["sdpo_weights"] == [0.0, 0.0, 0.0]
    assert record["sdpo_topk_token_ids"] == [None, None, None]
    assert record["sdpo_student_topk_token_ids"] == [None, None, None]


@pytest.mark.parametrize(
    ("field", "expected_error"),
    [
        ("sdpo", "sdpo top-k support requires nonzero sdpo_weights"),
        ("sdpo_student", "sdpo_student top-k support requires nonzero sdpo_weights"),
    ],
)
def test_token_export_rejects_sdpo_support_without_active_component(tmp_path, field, expected_error):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    micro_batch["sdpo_weights"] = torch.zeros_like(micro_batch["sdpo_weights"])
    micro_batch["sdpo_rollout_is_weights"] = None
    model_output = _sdpo_model_output()
    if field == "sdpo":
        model_output["sdpo_student_topk_token_ids"] = None
        model_output["sdpo_student_topk_logprobs"] = None
    else:
        micro_batch["sdpo_topk_token_ids"] = None
        micro_batch["sdpo_topk_logprobs"] = None

    try:
        with pytest.raises(ValueError, match=expected_error):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=model_output,
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


def test_token_export_rejects_active_sdpo_record_without_sample_id(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    micro_batch["sample_ids"] = [None]

    try:
        with pytest.raises(ValueError, match="SDPO token export records require a non-empty sample_id"):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=_sdpo_model_output(),
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


@pytest.mark.parametrize("env_names", [["", "", "", ""], ["   ", "   ", "   ", ""]])
def test_token_export_rejects_active_sdpo_record_without_env_name(tmp_path, env_names):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    micro_batch["env_names"] = env_names

    try:
        with pytest.raises(ValueError, match="SDPO token export records require a non-empty env_name"):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=_sdpo_model_output(),
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


def test_token_export_rejects_nonzero_sdpo_rollout_is_weight_outside_sdpo_component(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    micro_batch["sdpo_rollout_is_weights"] = torch.tensor([0.5, 0.75, 1.25, 0.0])

    try:
        with pytest.raises(ValueError, match=r"sdpo_rollout_is_weights\[0\] is nonzero outside SDPO component"):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=_sdpo_model_output(),
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


@pytest.mark.parametrize("preflight_only", [False, True])
def test_token_export_rejects_nonzero_sdpo_weight_outside_loss_mask(tmp_path, preflight_only):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    micro_batch["sdpo_weights"] = torch.tensor([1.0, 1.0, 1.0, 0.0])
    if preflight_only:
        micro_batch["preflight_only"] = True
        micro_batch["sdpo_topk_token_ids"] = None
        micro_batch["sdpo_topk_logprobs"] = None

    try:
        with pytest.raises(ValueError, match=r"sdpo_weights\[0\] is nonzero outside loss_mask"):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=_sdpo_model_output(),
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


@pytest.mark.parametrize(
    ("field", "bad_weights", "expected_error"),
    [
        ("sdpo_weights", torch.tensor([0.0, float("nan"), 1.0, 0.0]), "sdpo_weights must contain finite weights"),
        ("sdpo_weights", torch.tensor([0.0, -0.5, 1.0, 0.0]), "sdpo_weights must contain non-negative weights"),
        ("sdpo_weights", torch.tensor([0.0, 1.0, 1.0]), "sdpo_weights length 3 != sequence length 4"),
        (
            "sdpo_rollout_is_weights",
            torch.tensor([0.0, float("inf"), 1.25, 0.0]),
            "sdpo_rollout_is_weights must contain finite weights",
        ),
        (
            "sdpo_rollout_is_weights",
            torch.tensor([0, 1, 1, 0]),
            "sdpo_rollout_is_weights must use a floating-point dtype",
        ),
        (
            "sdpo_rollout_is_weights",
            torch.tensor([0.0, -0.25, 1.25, 0.0]),
            "sdpo_rollout_is_weights must contain non-negative weights",
        ),
        (
            "sdpo_rollout_is_weights",
            torch.tensor([0.0, 0.75, 1.25]),
            "sdpo_rollout_is_weights length 3 != sequence length 4",
        ),
    ],
)
def test_token_export_rejects_invalid_sdpo_weight_streams(tmp_path, field, bad_weights, expected_error):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    micro_batch[field] = bad_weights

    try:
        with pytest.raises(ValueError, match=expected_error):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=_sdpo_model_output(),
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


def test_token_export_allows_absent_sample_ids_for_legacy_exports(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    del micro_batch["sample_ids"]
    micro_batch["sdpo_weights"] = torch.zeros_like(micro_batch["sdpo_weights"])
    micro_batch["sdpo_topk_token_ids"] = None
    micro_batch["sdpo_topk_logprobs"] = None
    micro_batch["sdpo_rollout_is_weights"] = None
    model_output = _sdpo_model_output()
    model_output["sdpo_student_topk_token_ids"] = None
    model_output["sdpo_student_topk_logprobs"] = None

    try:
        exporter.export(
            step=7,
            micro_step=0,
            micro_batch=micro_batch,
            model_output=model_output,
            sequence_lengths=[4],
            loss_config=DefaultLossConfig(),
        )
    finally:
        exporter.close()

    record = json.loads((tmp_path / "token_exports" / "step_7" / "rank_0.jsonl").read_text().strip())
    assert record["sample_id"] is None


@pytest.mark.parametrize(
    ("field", "bad_rows", "expected_error"),
    [
        (
            "sdpo_topk_token_ids",
            torch.tensor([[0.0, 0.0], [111.9, 112.0], [211.0, 212.0], [0.0, 0.0]]),
            r"sdpo_topk_token_ids must contain integer token ids",
        ),
        (
            "sdpo_student_topk_token_ids",
            torch.tensor([[[0, 0], [111, -112], [211, 212], [0, 0]]]),
            r"sdpo_student_topk_token_ids must contain non-negative token ids",
        ),
        (
            "sdpo_topk_token_ids",
            torch.empty((4, 0), dtype=torch.long),
            r"sdpo_topk_token_ids must contain non-empty top-k rows",
        ),
    ],
)
def test_token_export_rejects_malformed_sdpo_support_token_ids(tmp_path, field, bad_rows, expected_error):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    model_output = _sdpo_model_output()
    if field == "sdpo_topk_token_ids":
        micro_batch[field] = bad_rows
    else:
        model_output[field] = bad_rows

    try:
        with pytest.raises(ValueError, match=expected_error):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=model_output,
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


@pytest.mark.parametrize(
    ("field", "expected_error"),
    [
        ("sdpo_topk_logprobs", r"sdpo_topk_logprobs must contain finite values"),
        ("sdpo_student_topk_logprobs", r"sdpo_student_topk_logprobs must contain finite values"),
    ],
)
def test_token_export_rejects_nonfinite_sdpo_support_logprobs(tmp_path, field, expected_error):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    model_output = _sdpo_model_output()
    if field == "sdpo_topk_logprobs":
        micro_batch[field] = torch.tensor([[0.0, 0.0], [float("nan"), -2.0], [-0.3, -1.5], [0.0, 0.0]])
    else:
        model_output[field] = torch.tensor([[[0.0, 0.0], [-0.75, float("inf")], [-0.625, -1.5], [0.0, 0.0]]])

    try:
        with pytest.raises(ValueError, match=expected_error):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=model_output,
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


@pytest.mark.parametrize(
    ("field", "bad_ids", "bad_logprobs", "expected_error"),
    [
        (
            "sdpo",
            torch.tensor([[0, 0], [101, 101], [201, 202], [0, 0]]),
            torch.tensor([[0.0, 0.0], [-0.5, -2.0], [-0.3, -1.5], [0.0, 0.0]]),
            r"sdpo top-k token ids row 1 must contain distinct token ids",
        ),
        (
            "sdpo_student",
            torch.tensor([[[0, 0], [111, 111], [211, 212], [0, 0]]]),
            torch.tensor([[[0.0, 0.0], [-0.75, -1.25], [-0.625, -1.5], [0.0, 0.0]]]),
            r"sdpo_student top-k token ids row 1 must contain distinct token ids",
        ),
        (
            "sdpo",
            torch.tensor([[0, 0], [101, 102], [201, 202], [0, 0]]),
            torch.tensor([[0.0, 0.0], [-0.1, -0.2], [-0.3, -1.5], [0.0, 0.0]]),
            r"sdpo top-k logprobs row 1 probability mass exceeds 1",
        ),
        (
            "sdpo_student",
            torch.tensor([[[0, 0], [111, 112], [211, 212], [0, 0]]]),
            torch.tensor([[[0.0, 0.0], [-0.75, -1.25], [-0.1, -0.2], [0.0, 0.0]]]),
            r"sdpo_student top-k logprobs row 2 probability mass exceeds 1",
        ),
        (
            "sdpo",
            torch.tensor([[0, 0], [101, 102], [201, 202], [0, 0]]),
            torch.tensor([[0.0, 0.0], [0.0, 0.0], [-0.3, -1.5], [0.0, 0.0]]),
            r"sdpo top-k token ids row 1 must be zero when logprobs are placeholders",
        ),
        (
            "sdpo_student",
            torch.tensor([[[0, 0], [111, 112], [211, 212], [0, 0]]]),
            torch.tensor([[[0.0, 0.0], [-0.75, -1.25], [0.0, 0.0], [0.0, 0.0]]]),
            r"sdpo_student top-k token ids row 2 must be zero when logprobs are placeholders",
        ),
        (
            "sdpo",
            torch.tensor([[0, 0], [101, 102], [201, 202], [0, 0]]),
            torch.tensor([[False, False], [False, False], [False, False], [False, False]]),
            r"sdpo_topk_logprobs must use a floating-point dtype",
        ),
        (
            "sdpo_student",
            torch.tensor([[[0, 0], [111, 112], [211, 212], [0, 0]]]),
            torch.tensor([[[False, False], [False, False], [False, False], [False, False]]]),
            r"sdpo_student_topk_logprobs must use a floating-point dtype",
        ),
    ],
)
def test_token_export_rejects_malformed_sdpo_support_rows(tmp_path, field, bad_ids, bad_logprobs, expected_error):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    model_output = _sdpo_model_output()
    if field == "sdpo":
        micro_batch["sdpo_topk_token_ids"] = bad_ids
        micro_batch["sdpo_topk_logprobs"] = bad_logprobs
    else:
        model_output["sdpo_student_topk_token_ids"] = bad_ids
        model_output["sdpo_student_topk_logprobs"] = bad_logprobs

    try:
        with pytest.raises(ValueError, match=expected_error):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=model_output,
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


def test_token_export_accepts_top1_token_id_zero_with_real_logprob(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    model_output = _sdpo_model_output()
    micro_batch["sdpo_topk_token_ids"] = torch.tensor([[0], [0], [201], [0]])
    micro_batch["sdpo_topk_logprobs"] = torch.tensor([[0.0], [-0.5], [-0.3], [0.0]])
    model_output["sdpo_student_topk_token_ids"] = torch.tensor([[[0], [111], [0], [0]]])
    model_output["sdpo_student_topk_logprobs"] = torch.tensor([[[0.0], [-0.75], [-0.625], [0.0]]])

    try:
        exporter.export(
            step=7,
            micro_step=0,
            micro_batch=micro_batch,
            model_output=model_output,
            sequence_lengths=[4],
            loss_config=DefaultLossConfig(),
        )
    finally:
        exporter.close()

    record = json.loads((tmp_path / "token_exports" / "step_7" / "rank_0.jsonl").read_text().strip())
    assert record["sdpo_topk_token_ids"] == [[0], [0], [201]]
    assert [row[0] for row in record["sdpo_topk_logprobs"]] == pytest.approx([0.0, -0.5, -0.3])
    assert record["sdpo_student_topk_token_ids"] == [[0], [111], [0]]
    assert [row[0] for row in record["sdpo_student_topk_logprobs"]] == pytest.approx([0.0, -0.75, -0.625])


@pytest.mark.parametrize(
    ("field", "expected_error"),
    [
        ("sdpo_topk_token_ids", r"sdpo top-k token ids and logprobs must be exported as a pair"),
        ("sdpo_topk_logprobs", r"sdpo top-k token ids and logprobs must be exported as a pair"),
        ("sdpo_student_topk_token_ids", r"sdpo_student top-k token ids and logprobs must be exported as a pair"),
        ("sdpo_student_topk_logprobs", r"sdpo_student top-k token ids and logprobs must be exported as a pair"),
    ],
)
def test_token_export_rejects_unpaired_sdpo_support_columns(tmp_path, field, expected_error):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    model_output = _sdpo_model_output()
    if field.startswith("sdpo_student"):
        model_output[field] = None
    else:
        micro_batch[field] = None

    try:
        with pytest.raises(ValueError, match=expected_error):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=model_output,
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


@pytest.mark.parametrize(
    ("field", "expected_error"),
    [
        ("sdpo_topk_logprobs", r"sdpo top-k token ids/logprobs width mismatch: 2 != 3"),
        ("sdpo_student_topk_logprobs", r"sdpo_student top-k token ids/logprobs width mismatch: 2 != 3"),
    ],
)
def test_token_export_rejects_sdpo_support_width_mismatch(tmp_path, field, expected_error):
    exporter = TokenExporter(tmp_path, rank=0)
    micro_batch = _sdpo_micro_batch()
    model_output = _sdpo_model_output()
    bad_rows = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [-0.75, -1.25, -1.75],
            [-0.625, -1.5, -2.0],
            [0.0, 0.0, 0.0],
        ]
    )
    if field.startswith("sdpo_student"):
        model_output[field] = bad_rows.unsqueeze(0)
    else:
        micro_batch[field] = bad_rows

    try:
        with pytest.raises(ValueError, match=expected_error):
            exporter.export(
                step=7,
                micro_step=0,
                micro_batch=micro_batch,
                model_output=model_output,
                sequence_lengths=[4],
                loss_config=DefaultLossConfig(),
            )
    finally:
        exporter.close()


def test_token_export_invalidates_stable_marker_before_appending_same_step(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)

    try:
        _export_sdpo_batch(exporter, step=7, micro_step=0)
        exporter.mark_stable()
        stable_marker = tmp_path / "token_exports" / "step_7" / "STABLE"
        export_path = tmp_path / "token_exports" / "step_7" / "rank_0.jsonl"

        assert stable_marker.exists()

        _export_sdpo_batch(exporter, step=7, micro_step=1)

        assert not stable_marker.exists()
        assert len(export_path.read_text().splitlines()) == 2

        exporter.mark_stable()

        assert stable_marker.exists()
    finally:
        exporter.close()


def test_token_export_resets_in_memory_state_when_step_dir_is_deleted(tmp_path):
    exporter = TokenExporter(tmp_path, rank=0)
    step_dir = tmp_path / "token_exports" / "step_7"
    export_path = step_dir / "rank_0.jsonl"

    try:
        _export_sdpo_batch(exporter, step=7, micro_step=0)
        exporter.mark_stable()
        shutil.rmtree(step_dir)

        _export_sdpo_batch(exporter, step=7, micro_step=1)

        records = [json.loads(line) for line in export_path.read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["micro_step"] == 1
        assert records[0]["export_sequence_idx"] == 0
        assert not (step_dir / "STABLE").exists()
    finally:
        exporter.close()
