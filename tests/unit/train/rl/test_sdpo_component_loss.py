import pytest
import torch

from prime_rl.configs.trainer import SDPOComponentConfig
from prime_rl.trainer.rl.loss import LossOutputs, compute_loss
from prime_rl.trainer.rl.sdpo_loss import SDPOLossConfig, compute_sdpo_loss
from tests.unit.train.rl.sdpo_reference_cases import REFERENCE_CASES, tensor_from_case_value


def test_compute_loss_forwards_sdpo_rollout_is_weights_separately():
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}
    component_weights = torch.tensor([2.0, 1.0])
    rollout_is_weights = torch.tensor([0.5, 1.75])
    loss_mask = tensors["response_mask"].squeeze(0)

    direct = compute_sdpo_loss(
        config=SDPOLossConfig(**case["config"]),
        component_weights=component_weights.unsqueeze(0),
        rollout_is_weights=rollout_is_weights.unsqueeze(0),
        **tensors,
    )
    loss, metrics = compute_loss(
        trainer_logprobs=[tensors["student_log_probs"].squeeze(0)],
        inference_logprobs=[tensors["student_log_probs"].squeeze(0)],
        ref_logprobs=None,
        advantages=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
        loss_mask=[loss_mask],
        rl_weights=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
        ce_weights=None,
        ref_kl_weights=None,
        sdpo_weights=[component_weights],
        sdpo_rollout_is_weights=[rollout_is_weights],
        student_topk_log_probs=[tensors["student_topk_log_probs"].squeeze(0)],
        teacher_topk_log_probs=[tensors["teacher_topk_log_probs"].squeeze(0)],
        sdpo_loss_config=SDPOComponentConfig(**case["config"]),
        rl_loss_fn=lambda inputs: LossOutputs(loss=torch.tensor(0.0), metrics={}),
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
        sdpo_scale=int(loss_mask.sum()),
    )

    assert torch.isclose(loss, direct, atol=1e-6)
    assert torch.isclose(metrics["sdpo"][0], direct.detach(), atol=1e-6)


def test_compute_loss_keeps_sdpo_denominator_separate_from_other_components():
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}
    sequence = tensors["student_log_probs"].squeeze(0)
    loss_mask = tensors["response_mask"].squeeze(0)
    sdpo_weights = torch.tensor([1.0, 1.0])
    ce_weights = torch.tensor([0.0, 2.0])

    direct_sdpo = compute_sdpo_loss(
        config=SDPOLossConfig(**case["config"]),
        component_weights=sdpo_weights.unsqueeze(0),
        **tensors,
    )
    ce_sum = (-sequence * ce_weights)[ce_weights != 0].sum()

    loss, metrics = compute_loss(
        trainer_logprobs=[sequence],
        inference_logprobs=[sequence],
        ref_logprobs=None,
        advantages=[torch.zeros_like(sequence)],
        loss_mask=[loss_mask],
        rl_weights=[torch.zeros_like(sequence)],
        ce_weights=[ce_weights],
        ref_kl_weights=None,
        sdpo_weights=[sdpo_weights],
        student_topk_log_probs=[tensors["student_topk_log_probs"].squeeze(0)],
        teacher_topk_log_probs=[tensors["teacher_topk_log_probs"].squeeze(0)],
        sdpo_loss_config=SDPOComponentConfig(**case["config"]),
        rl_loss_fn=lambda inputs: LossOutputs(loss=torch.tensor(0.0), metrics={}),
        rl_scale=1,
        ce_scale=4,
        ref_kl_scale=1,
        sdpo_scale=2,
    )

    expected = direct_sdpo + ce_sum / 4
    assert torch.isclose(loss, expected, atol=1e-6)
    assert torch.isclose(metrics["sdpo"][0], direct_sdpo.detach(), atol=1e-6)


def test_compute_loss_rejects_sdpo_weight_on_non_sampled_token():
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}
    loss_mask = torch.tensor([False, True])

    with pytest.raises(ValueError, match="SDPO weights may only select sampled/loss-mask tokens"):
        compute_loss(
            trainer_logprobs=[tensors["student_log_probs"].squeeze(0)],
            inference_logprobs=[tensors["student_log_probs"].squeeze(0)],
            ref_logprobs=None,
            advantages=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
            loss_mask=[loss_mask],
            rl_weights=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
            ce_weights=None,
            ref_kl_weights=None,
            sdpo_weights=[torch.tensor([1.0, 1.0])],
            student_topk_log_probs=[tensors["student_topk_log_probs"].squeeze(0)],
            teacher_topk_log_probs=[tensors["teacher_topk_log_probs"].squeeze(0)],
            sdpo_loss_config=SDPOComponentConfig(**case["config"]),
            rl_loss_fn=lambda inputs: LossOutputs(loss=torch.tensor(0.0), metrics={}),
            rl_scale=1,
            ce_scale=1,
            ref_kl_scale=1,
            sdpo_scale=1,
        )


@pytest.mark.parametrize(
    ("sdpo_weights", "message"),
    [
        (torch.tensor([1.0, 1.0], dtype=torch.bool), "SDPO weights must contain real numeric values"),
        (torch.tensor([1.0, 1.0], dtype=torch.complex64), "SDPO weights must contain real numeric values"),
        (torch.tensor([1.0, float("nan")]), "SDPO weights must contain finite values"),
        (torch.tensor([1.0, -0.25]), "SDPO weights must be non-negative"),
    ],
)
def test_compute_loss_rejects_malformed_sdpo_weights(sdpo_weights, message):
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}

    with pytest.raises(ValueError, match=message):
        compute_loss(
            trainer_logprobs=[tensors["student_log_probs"].squeeze(0)],
            inference_logprobs=[tensors["student_log_probs"].squeeze(0)],
            ref_logprobs=None,
            advantages=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
            loss_mask=[torch.ones_like(tensors["student_log_probs"].squeeze(0), dtype=torch.bool)],
            rl_weights=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
            ce_weights=None,
            ref_kl_weights=None,
            sdpo_weights=[sdpo_weights],
            student_topk_log_probs=[tensors["student_topk_log_probs"].squeeze(0)],
            teacher_topk_log_probs=[tensors["teacher_topk_log_probs"].squeeze(0)],
            sdpo_loss_config=SDPOComponentConfig(**case["config"]),
            rl_loss_fn=lambda inputs: LossOutputs(loss=torch.tensor(0.0), metrics={}),
            rl_scale=1,
            ce_scale=1,
            ref_kl_scale=1,
            sdpo_scale=1,
        )


@pytest.mark.parametrize("sdpo_weights", [None, torch.tensor([0.0, 0.0])])
def test_compute_loss_rejects_sdpo_topk_logprobs_without_active_sdpo_weights(sdpo_weights):
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}

    with pytest.raises(ValueError, match="SDPO top-k logprobs require active SDPO weights"):
        compute_loss(
            trainer_logprobs=[tensors["student_log_probs"].squeeze(0)],
            inference_logprobs=[tensors["student_log_probs"].squeeze(0)],
            ref_logprobs=None,
            advantages=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
            loss_mask=[torch.ones_like(tensors["student_log_probs"].squeeze(0), dtype=torch.bool)],
            rl_weights=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
            ce_weights=None,
            ref_kl_weights=None,
            sdpo_weights=None if sdpo_weights is None else [sdpo_weights],
            student_topk_log_probs=[tensors["student_topk_log_probs"].squeeze(0)],
            teacher_topk_log_probs=[tensors["teacher_topk_log_probs"].squeeze(0)],
            sdpo_loss_config=SDPOComponentConfig(**case["config"]),
            rl_loss_fn=lambda inputs: LossOutputs(loss=torch.tensor(0.0), metrics={}),
            rl_scale=1,
            ce_scale=1,
            ref_kl_scale=1,
            sdpo_scale=1,
        )


@pytest.mark.parametrize(
    ("sdpo_weights", "rollout_is_weights", "message"),
    [
        (
            torch.tensor([1.0, 1.0]),
            torch.tensor([1.0, 1.0], dtype=torch.bool),
            "SDPO rollout-IS weights must contain real numeric values",
        ),
        (
            torch.tensor([1.0, 1.0]),
            torch.tensor([1.0, 1.0], dtype=torch.complex64),
            "SDPO rollout-IS weights must contain real numeric values",
        ),
        (
            torch.tensor([1.0, 1.0]),
            torch.tensor([1.0, 1.0, 1.0]),
            "SDPO rollout-IS weights shape",
        ),
        (
            torch.tensor([1.0, 1.0]),
            torch.tensor([1.0, float("inf")]),
            "SDPO rollout-IS weights must contain finite values",
        ),
        (
            torch.tensor([1.0, 1.0]),
            torch.tensor([1.0, -0.25]),
            "SDPO rollout-IS weights must be non-negative",
        ),
        (
            torch.tensor([1.0, 1.0]),
            torch.tensor([1, 1], dtype=torch.long),
            "SDPO rollout-IS weights must use a floating-point dtype",
        ),
        (
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.5, 1.0]),
            "SDPO rollout-IS weights may only be nonzero on SDPO component tokens",
        ),
        (
            None,
            torch.tensor([0.0, 1.0]),
            "SDPO rollout-IS weights may only be nonzero on SDPO component tokens",
        ),
    ],
)
def test_compute_loss_rejects_malformed_sdpo_rollout_is_weights(sdpo_weights, rollout_is_weights, message):
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}

    with pytest.raises(ValueError, match=message):
        compute_loss(
            trainer_logprobs=[tensors["student_log_probs"].squeeze(0)],
            inference_logprobs=[tensors["student_log_probs"].squeeze(0)],
            ref_logprobs=None,
            advantages=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
            loss_mask=[torch.ones_like(tensors["student_log_probs"].squeeze(0), dtype=torch.bool)],
            rl_weights=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
            ce_weights=None,
            ref_kl_weights=None,
            sdpo_weights=None if sdpo_weights is None else [sdpo_weights],
            sdpo_rollout_is_weights=[rollout_is_weights],
            student_topk_log_probs=[tensors["student_topk_log_probs"].squeeze(0)],
            teacher_topk_log_probs=[tensors["teacher_topk_log_probs"].squeeze(0)],
            sdpo_loss_config=SDPOComponentConfig(**case["config"]),
            rl_loss_fn=lambda inputs: LossOutputs(loss=torch.tensor(0.0), metrics={}),
            rl_scale=1,
            ce_scale=1,
            ref_kl_scale=1,
            sdpo_scale=1,
        )


def test_compute_loss_rejects_supplied_rollout_is_weights_above_non_normalized_threshold():
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}

    with pytest.raises(ValueError, match="rollout_is_weights must not exceed rollout_is_threshold=2.0"):
        compute_loss(
            trainer_logprobs=[tensors["student_log_probs"].squeeze(0)],
            inference_logprobs=[tensors["student_log_probs"].squeeze(0)],
            ref_logprobs=None,
            advantages=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
            loss_mask=[torch.ones_like(tensors["student_log_probs"].squeeze(0), dtype=torch.bool)],
            rl_weights=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
            ce_weights=None,
            ref_kl_weights=None,
            sdpo_weights=[torch.tensor([1.0, 1.0])],
            sdpo_rollout_is_weights=[torch.tensor([2.01, 1.0])],
            student_topk_log_probs=[tensors["student_topk_log_probs"].squeeze(0)],
            teacher_topk_log_probs=[tensors["teacher_topk_log_probs"].squeeze(0)],
            sdpo_loss_config=SDPOComponentConfig(**case["config"], rollout_is="token", rollout_is_threshold=2.0),
            rl_loss_fn=lambda inputs: LossOutputs(loss=torch.tensor(0.0), metrics={}),
            rl_scale=1,
            ce_scale=1,
            ref_kl_scale=1,
            sdpo_scale=1,
        )


def test_compute_loss_allows_precomputed_batch_normalized_rollout_is_weights_above_threshold():
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}
    sequence = tensors["student_log_probs"].squeeze(0)

    loss, metrics = compute_loss(
        trainer_logprobs=[sequence],
        inference_logprobs=[sequence],
        ref_logprobs=None,
        advantages=[torch.zeros_like(sequence)],
        loss_mask=[torch.ones_like(sequence, dtype=torch.bool)],
        rl_weights=[torch.zeros_like(sequence)],
        ce_weights=None,
        ref_kl_weights=None,
        sdpo_weights=[torch.ones_like(sequence)],
        sdpo_rollout_is_weights=[torch.tensor([2.01, 1.0])],
        student_topk_log_probs=[tensors["student_topk_log_probs"].squeeze(0)],
        teacher_topk_log_probs=[tensors["teacher_topk_log_probs"].squeeze(0)],
        sdpo_loss_config=SDPOComponentConfig(
            **case["config"],
            rollout_is="token",
            rollout_is_batch_normalize=True,
        ),
        rl_loss_fn=lambda inputs: LossOutputs(loss=torch.tensor(0.0), metrics={}),
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
        sdpo_scale=2,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["sdpo"]).all()


def test_compute_loss_rejects_deriving_batch_normalized_rollout_is_weights_per_sequence():
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}
    sequence = tensors["student_log_probs"].squeeze(0)

    with pytest.raises(ValueError, match="cannot derive batch-normalized SDPO rollout-IS weights"):
        compute_loss(
            trainer_logprobs=[sequence],
            inference_logprobs=[sequence - 0.1],
            ref_logprobs=None,
            advantages=[torch.zeros_like(sequence)],
            loss_mask=[torch.ones_like(sequence, dtype=torch.bool)],
            rl_weights=[torch.zeros_like(sequence)],
            ce_weights=None,
            ref_kl_weights=None,
            sdpo_weights=[torch.ones_like(sequence)],
            sdpo_rollout_is_weights=None,
            student_topk_log_probs=[tensors["student_topk_log_probs"].squeeze(0)],
            teacher_topk_log_probs=[tensors["teacher_topk_log_probs"].squeeze(0)],
            sdpo_loss_config=SDPOComponentConfig(
                **case["config"],
                rollout_is="token",
                rollout_is_batch_normalize=True,
            ),
            rl_loss_fn=lambda inputs: LossOutputs(loss=torch.tensor(0.0), metrics={}),
            rl_scale=1,
            ce_scale=1,
            ref_kl_scale=1,
            sdpo_scale=2,
        )


def test_compute_loss_allows_zero_sdpo_rollout_is_weights_without_sdpo_component():
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}
    sequence = tensors["student_log_probs"].squeeze(0)

    loss, metrics = compute_loss(
        trainer_logprobs=[sequence],
        inference_logprobs=[sequence],
        ref_logprobs=None,
        advantages=[torch.zeros_like(sequence)],
        loss_mask=[torch.ones_like(sequence, dtype=torch.bool)],
        rl_weights=[torch.zeros_like(sequence)],
        ce_weights=None,
        ref_kl_weights=None,
        sdpo_weights=None,
        sdpo_rollout_is_weights=[torch.zeros_like(sequence)],
        sdpo_loss_config=SDPOComponentConfig(**case["config"]),
        rl_loss_fn=lambda inputs: LossOutputs(loss=torch.tensor(0.0), metrics={}),
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
        sdpo_scale=1,
    )

    assert torch.isclose(loss, torch.tensor(0.0))
    assert "sdpo" not in metrics


def test_compute_loss_rejects_required_sequence_list_length_mismatch():
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}

    with pytest.raises(ValueError, match="advantages has 1 sequence\\(s\\), expected 2"):
        compute_loss(
            trainer_logprobs=[
                tensors["student_log_probs"].squeeze(0),
                tensors["student_log_probs"].squeeze(0),
            ],
            inference_logprobs=[
                tensors["student_log_probs"].squeeze(0),
                tensors["student_log_probs"].squeeze(0),
            ],
            ref_logprobs=None,
            advantages=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
            loss_mask=[
                tensors["response_mask"].squeeze(0),
                tensors["response_mask"].squeeze(0),
            ],
            rl_weights=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))] * 2,
            ce_weights=None,
            ref_kl_weights=None,
            sdpo_loss_config=SDPOComponentConfig(**case["config"]),
            rl_loss_fn=lambda inputs: LossOutputs(loss=torch.tensor(0.0), metrics={}),
            rl_scale=1,
            ce_scale=1,
            ref_kl_scale=1,
        )


def test_compute_loss_rejects_sdpo_topk_sequence_list_length_mismatch():
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}
    sequence = tensors["student_log_probs"].squeeze(0)
    mask = tensors["response_mask"].squeeze(0)
    weights = torch.tensor([1.0, 1.0])

    with pytest.raises(ValueError, match="student_topk_log_probs has 1 sequence\\(s\\), expected 2"):
        compute_loss(
            trainer_logprobs=[sequence, sequence],
            inference_logprobs=[sequence, sequence],
            ref_logprobs=None,
            advantages=[torch.zeros_like(sequence), torch.zeros_like(sequence)],
            loss_mask=[mask, mask],
            rl_weights=[torch.zeros_like(sequence), torch.zeros_like(sequence)],
            ce_weights=None,
            ref_kl_weights=None,
            sdpo_weights=[weights, weights],
            student_topk_log_probs=[tensors["student_topk_log_probs"].squeeze(0)],
            teacher_topk_log_probs=[
                tensors["teacher_topk_log_probs"].squeeze(0),
                tensors["teacher_topk_log_probs"].squeeze(0),
            ],
            sdpo_loss_config=SDPOComponentConfig(**case["config"]),
            rl_loss_fn=lambda inputs: LossOutputs(loss=torch.tensor(0.0), metrics={}),
            rl_scale=1,
            ce_scale=1,
            ref_kl_scale=1,
            sdpo_scale=1,
        )


@pytest.mark.parametrize(
    ("student_topk", "teacher_topk", "message"),
    [
        (None, torch.zeros(2, 3), "both student and teacher streams"),
        (torch.zeros(2, 3), torch.zeros(2, 4), "student and teacher logprob shapes must match"),
        (torch.zeros(2), torch.zeros(2), "rank 2 per sequence"),
        (torch.zeros(3, 2), torch.zeros(3, 2), "sequence length 3 must match loss_mask length 2"),
        (torch.zeros(2, 0), torch.zeros(2, 0), "non-empty top-k dimension"),
    ],
)
def test_compute_loss_rejects_malformed_sdpo_topk_geometry_before_active_slicing(student_topk, teacher_topk, message):
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}
    sequence = tensors["student_log_probs"].squeeze(0)
    mask = tensors["response_mask"].squeeze(0)
    weights = torch.tensor([1.0, 1.0])

    with pytest.raises(ValueError, match=message):
        compute_loss(
            trainer_logprobs=[sequence],
            inference_logprobs=[sequence],
            ref_logprobs=None,
            advantages=[torch.zeros_like(sequence)],
            loss_mask=[mask],
            rl_weights=[torch.zeros_like(sequence)],
            ce_weights=None,
            ref_kl_weights=None,
            sdpo_weights=[weights],
            student_topk_log_probs=[student_topk],
            teacher_topk_log_probs=[teacher_topk],
            sdpo_loss_config=SDPOComponentConfig(**case["config"]),
            rl_loss_fn=lambda inputs: LossOutputs(loss=torch.tensor(0.0), metrics={}),
            rl_scale=1,
            ce_scale=1,
            ref_kl_scale=1,
            sdpo_scale=1,
        )


def test_compute_loss_normalizes_sdpo_over_sampled_weighted_tokens_only():
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}
    loss_mask = torch.tensor([True, True])
    sdpo_weights = torch.tensor([0.0, 1.5])
    student_topk = tensors["student_topk_log_probs"].squeeze(0).clone()
    teacher_topk = tensors["teacher_topk_log_probs"].squeeze(0).clone()
    student_topk[0] = 0.0
    teacher_topk[0] = 0.0
    direct = compute_sdpo_loss(
        config=SDPOLossConfig(**case["config"]),
        response_mask=torch.tensor([[False, True]]),
        component_weights=sdpo_weights.unsqueeze(0),
        student_log_probs=tensors["student_log_probs"],
        teacher_log_probs=tensors["teacher_log_probs"],
        student_topk_log_probs=tensors["student_topk_log_probs"],
        teacher_topk_log_probs=tensors["teacher_topk_log_probs"],
    )

    loss, metrics = compute_loss(
        trainer_logprobs=[tensors["student_log_probs"].squeeze(0)],
        inference_logprobs=[tensors["student_log_probs"].squeeze(0)],
        ref_logprobs=None,
        advantages=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
        loss_mask=[loss_mask],
        rl_weights=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
        ce_weights=None,
        ref_kl_weights=None,
        sdpo_weights=[sdpo_weights],
        student_topk_log_probs=[student_topk],
        teacher_topk_log_probs=[teacher_topk],
        sdpo_loss_config=SDPOComponentConfig(**case["config"]),
        rl_loss_fn=lambda inputs: LossOutputs(loss=torch.tensor(0.0), metrics={}),
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
        sdpo_scale=1,
    )

    assert torch.isclose(loss, direct, atol=1e-6)
    assert torch.isclose(metrics["sdpo"][0], direct.detach(), atol=1e-6)


def test_compute_loss_rejects_non_placeholder_topk_rows_outside_sdpo_component():
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}
    loss_mask = torch.tensor([True, True])
    sdpo_weights = torch.tensor([0.0, 1.5])

    with pytest.raises(ValueError, match="placeholders outside active SDPO component"):
        compute_loss(
            trainer_logprobs=[tensors["student_log_probs"].squeeze(0)],
            inference_logprobs=[tensors["student_log_probs"].squeeze(0)],
            ref_logprobs=None,
            advantages=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
            loss_mask=[loss_mask],
            rl_weights=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
            ce_weights=None,
            ref_kl_weights=None,
            sdpo_weights=[sdpo_weights],
            student_topk_log_probs=[tensors["student_topk_log_probs"].squeeze(0)],
            teacher_topk_log_probs=[tensors["teacher_topk_log_probs"].squeeze(0)],
            sdpo_loss_config=SDPOComponentConfig(**case["config"]),
            rl_loss_fn=lambda inputs: LossOutputs(loss=torch.tensor(0.0), metrics={}),
            rl_scale=1,
            ce_scale=1,
            ref_kl_scale=1,
            sdpo_scale=1,
        )


def test_compute_loss_ignores_placeholder_topk_rows_outside_sdpo_component():
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}
    loss_mask = torch.tensor([True, True])
    sdpo_weights = torch.tensor([0.0, 1.5])
    student_topk = tensors["student_topk_log_probs"].squeeze(0).clone()
    teacher_topk = tensors["teacher_topk_log_probs"].squeeze(0).clone()
    student_topk[0] = 0.0
    teacher_topk[0] = 0.0
    direct = compute_sdpo_loss(
        config=SDPOLossConfig(**case["config"]),
        response_mask=torch.tensor([[True]]),
        component_weights=sdpo_weights[1:].unsqueeze(0),
        student_log_probs=tensors["student_log_probs"][:, 1:],
        teacher_log_probs=tensors["teacher_log_probs"][:, 1:],
        student_topk_log_probs=tensors["student_topk_log_probs"][:, 1:],
        teacher_topk_log_probs=tensors["teacher_topk_log_probs"][:, 1:],
    )

    loss, metrics = compute_loss(
        trainer_logprobs=[tensors["student_log_probs"].squeeze(0)],
        inference_logprobs=[tensors["student_log_probs"].squeeze(0)],
        ref_logprobs=None,
        advantages=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
        loss_mask=[loss_mask],
        rl_weights=[torch.zeros_like(tensors["student_log_probs"].squeeze(0))],
        ce_weights=None,
        ref_kl_weights=None,
        sdpo_weights=[sdpo_weights],
        student_topk_log_probs=[student_topk],
        teacher_topk_log_probs=[teacher_topk],
        sdpo_loss_config=SDPOComponentConfig(**case["config"]),
        rl_loss_fn=lambda inputs: LossOutputs(loss=torch.tensor(0.0), metrics={}),
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
        sdpo_scale=1,
    )

    assert torch.isclose(loss, direct, atol=1e-6)
    assert torch.isclose(metrics["sdpo"][0], direct.detach(), atol=1e-6)
