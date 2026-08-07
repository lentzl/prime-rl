import pytest
import torch

from prime_rl.configs.trainer import DefaultLossConfig, SDPOComponentConfig
from prime_rl.trainer.rl.loss import compute_loss, setup_rl_loss_fn
from prime_rl.trainer.rl.sdpo_loss import SDPOLossConfig, compute_rollout_is_weights, compute_sdpo_loss
from prime_rl.trainer.rl.sdpo_support import (
    gather_sdpo_student_topk_logprobs,
    gather_sdpo_teacher_topk_logprobs,
    pack_sdpo_teacher_span_batches,
    pack_sdpo_teacher_spans,
    select_sdpo_student_topk_support,
)
from prime_rl.transport import SDPOTeacherSpan
from tests.unit.train.rl.sdpo_reference_cases import (
    REFERENCE_CASES,
    ROLLOUT_IS_WEIGHT_REFERENCE_CASES,
    tensor_from_case_value,
)


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=[case["name"] for case in REFERENCE_CASES])
def test_sdpo_loss_matches_huebotter_reference(case):
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}

    loss = compute_sdpo_loss(config=SDPOLossConfig(**case["config"]), **tensors)

    torch.testing.assert_close(loss, torch.tensor(case["expected_loss"]), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "case",
    ROLLOUT_IS_WEIGHT_REFERENCE_CASES,
    ids=[case["name"] for case in ROLLOUT_IS_WEIGHT_REFERENCE_CASES],
)
def test_rollout_importance_weights_match_huebotter_reference(case):
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}

    weights = compute_rollout_is_weights(**tensors, **case["config"])

    torch.testing.assert_close(
        weights,
        tensor_from_case_value(case["expected_weights"]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_topk_sdpo_loss_backpropagates_through_student_distribution():
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: tensor_from_case_value(value) for name, value in case["tensors"].items()}
    student_topk = tensors["student_topk_log_probs"].requires_grad_()
    tensors["student_topk_log_probs"] = student_topk

    loss = compute_sdpo_loss(config=SDPOLossConfig(**case["config"]), **tensors)
    loss.backward()

    assert student_topk.grad is not None
    assert torch.isfinite(student_topk.grad).all()
    assert torch.count_nonzero(student_topk.grad) > 0


def test_student_support_logprobs_use_the_distribution_that_predicted_each_token():
    logits = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [0.0, 2.0, 1.0]]])
    token_ids = torch.tensor([[[0, 1], [2, 1], [0, 2]]])

    actual = gather_sdpo_student_topk_logprobs(logits, torch.ones(1, 3), token_ids)
    shifted_logits = torch.cat([torch.zeros(1, 1, 3), logits[:, :-1]], dim=1)
    expected = torch.gather(shifted_logits.log_softmax(dim=-1), dim=-1, index=token_ids)

    torch.testing.assert_close(actual, expected)


def test_student_selected_support_is_scored_at_matching_teacher_positions():
    student_logits = torch.tensor([[[0.0, 3.0, 1.0], [2.0, 0.0, 1.0], [1.0, 2.0, 0.0]]])
    support_ids = select_sdpo_student_topk_support(student_logits, torch.ones(1, 3), topk=2)
    teacher_logits = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 3.0, 0.0], [2.0, 0.0, 1.0]]])

    actual = gather_sdpo_teacher_topk_logprobs(
        teacher_logits,
        positions=torch.tensor([1, 2]),
        token_ids=support_ids[0, 1:],
    )
    shifted_teacher = torch.cat([torch.zeros(1, 1, 3), teacher_logits[:, :-1]], dim=1).log_softmax(dim=-1)
    expected = torch.gather(shifted_teacher[0, 1:], dim=-1, index=support_ids[0, 1:])

    torch.testing.assert_close(actual, expected)


def test_teacher_spans_pack_with_independent_positions():
    packed = pack_sdpo_teacher_spans(
        [
            SDPOTeacherSpan(prefix_ids=[1, 2], completion_ids=[3, 4], student_positions=[5], target_offsets=[1]),
            SDPOTeacherSpan(prefix_ids=[6], completion_ids=[7], student_positions=[9], target_offsets=[0]),
        ]
    )

    assert packed == ([1, 2, 3, 4, 6, 7], [0, 1, 2, 3, 0, 1], [3, 5], [5, 9], [4, 2])


def test_teacher_spans_split_into_bounded_batches_without_reordering_targets():
    spans = [
        SDPOTeacherSpan(prefix_ids=[1, 2], completion_ids=[3, 4], student_positions=[5], target_offsets=[1]),
        SDPOTeacherSpan(prefix_ids=[6], completion_ids=[7], student_positions=[9], target_offsets=[0]),
        SDPOTeacherSpan(prefix_ids=[8], completion_ids=[9, 10], student_positions=[12], target_offsets=[1]),
    ]

    batches = pack_sdpo_teacher_span_batches(spans, max_seq_len=5)

    assert batches == [
        ([1, 2, 3, 4], [0, 1, 2, 3], [3], [5], [4]),
        ([6, 7, 8, 9, 10], [0, 1, 0, 1, 2], [1, 4], [9, 12], [2, 3]),
    ]


def test_teacher_span_batch_rejects_one_replay_larger_than_model_context():
    span = SDPOTeacherSpan(prefix_ids=[1, 2, 3], completion_ids=[4, 5], student_positions=[9], target_offsets=[0])

    with pytest.raises(ValueError, match="teacher span has 5 tokens"):
        pack_sdpo_teacher_span_batches([span], max_seq_len=4)


def test_sdpo_component_composes_with_prime_loss_and_backpropagates():
    student_topk = torch.tensor(
        [[-1.2, -1.5], [-0.8, -2.0], [-1.0, -1.4]],
        requires_grad=True,
    )
    teacher_topk = torch.tensor([[-1.0, -1.7], [-1.2, -1.4], [-0.7, -2.2]])
    trainer_logprobs = torch.tensor([-0.4, -0.3, -0.5], requires_grad=True)

    loss, metrics = compute_loss(
        trainer_logprobs=[trainer_logprobs],
        inference_logprobs=[torch.tensor([-0.5, -0.4, -0.6])],
        ref_logprobs=None,
        advantages=[torch.zeros(3)],
        loss_mask=[torch.tensor([False, True, True])],
        rl_weights=[torch.zeros(3)],
        ce_weights=None,
        ref_kl_weights=None,
        sdpo_weights=[torch.tensor([0.0, 1.0, 1.0])],
        student_topk_logprobs=[student_topk],
        teacher_topk_logprobs=[teacher_topk],
        rl_loss_fn=setup_rl_loss_fn(DefaultLossConfig()),
        sdpo_loss_config=SDPOComponentConfig(distillation_topk=2, is_clip=None, rollout_is=None),
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
        sdpo_scale=2,
    )
    loss.backward()

    assert "sdpo" in metrics
    assert student_topk.grad is not None
    assert torch.count_nonzero(student_topk.grad[1:]) > 0
