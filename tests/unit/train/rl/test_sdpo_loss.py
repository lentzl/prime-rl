"""SDPO loss reference tests."""

import pytest
import torch

from prime_rl.configs.trainer import SDPOComponentConfig
from prime_rl.trainer.rl.loss import (
    LossInputs,
    sdpo_loss_fn,
)
from prime_rl.trainer.rl.sdpo_loss import SDPOLossConfig, compute_rollout_is_weights, compute_sdpo_loss
from tests.unit.train.rl.sdpo_reference_cases import REFERENCE_CASES, ROLLOUT_IS_WEIGHT_REFERENCE_CASES
from tests.unit.train.rl.sdpo_reference_cases import tensor_from_case_value as _tensor


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=[case["name"] for case in REFERENCE_CASES])
def test_sdpo_loss_matches_huebotter_reference_constants(case):
    tensors = {name: _tensor(value) for name, value in case["tensors"].items()}
    loss = compute_sdpo_loss(config=SDPOLossConfig(**case["config"]), **tensors)
    assert torch.isclose(loss, torch.tensor(case["expected_loss"], dtype=loss.dtype), atol=1e-6)


def test_sampled_token_path_rejects_non_reverse_kl():
    with pytest.raises(ValueError, match="sampled-token SDPO only supports alpha=1.0"):
        compute_sdpo_loss(
            student_log_probs=torch.zeros(1, 1),
            teacher_log_probs=torch.zeros(1, 1),
            response_mask=torch.ones(1, 1, dtype=torch.bool),
            config=SDPOLossConfig(full_logit_distillation=False, distillation_topk=None, alpha=0.5),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"full_logit_distillation": "true"}, "full_logit_distillation"),
        ({"distillation_add_tail": "true"}, "distillation_add_tail"),
        ({"rollout_is_batch_normalize": 1}, "rollout_is_batch_normalize"),
        ({"alpha": True}, "alpha"),
        ({"alpha": float("nan")}, "alpha"),
        ({"alpha": -0.1}, "alpha"),
        ({"alpha": 1.1}, "alpha"),
        ({"distillation_topk": True}, "distillation_topk"),
        ({"distillation_topk": 2.5}, "distillation_topk"),
        ({"distillation_topk": 0}, "distillation_topk"),
        ({"is_clip": True}, "is_clip"),
        ({"is_clip": float("nan")}, "is_clip"),
        ({"is_clip": 0.0}, "is_clip"),
        ({"rollout_is": "batch"}, "rollout_is"),
        ({"rollout_is_threshold": True}, "rollout_is_threshold"),
        ({"rollout_is_threshold": float("inf")}, "rollout_is_threshold"),
        ({"rollout_is_threshold": 0.0}, "rollout_is_threshold"),
    ],
)
def test_sdpo_loss_config_rejects_invalid_knobs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SDPOLossConfig(**kwargs)


def _minimal_topk_loss_kwargs():
    return {
        "student_log_probs": torch.zeros(1, 2),
        "teacher_log_probs": torch.zeros(1, 2),
        "response_mask": torch.ones(1, 2, dtype=torch.bool),
        "student_topk_log_probs": torch.log_softmax(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), dim=-1),
        "teacher_topk_log_probs": torch.log_softmax(torch.tensor([[[0.5, 0.0], [0.0, 0.5]]]), dim=-1),
        "config": SDPOLossConfig(full_logit_distillation=True, distillation_topk=2),
    }


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"teacher_log_probs": torch.zeros(1, 3)}, "teacher_log_probs shape"),
        ({"response_mask": torch.ones(1, 3, dtype=torch.bool)}, "response_mask shape"),
        ({"teacher_topk_log_probs": torch.zeros(1, 2, 3)}, "teacher_topk_log_probs shape"),
        ({"student_topk_log_probs": torch.zeros(1, 3, 2)}, "student_topk_log_probs leading shape"),
        (
            {
                "student_topk_log_probs": torch.zeros(1, 2),
                "teacher_topk_log_probs": torch.zeros(1, 2),
            },
            "final top-k dimension",
        ),
        (
            {
                "student_topk_log_probs": torch.zeros(1, 2, 3),
                "teacher_topk_log_probs": torch.zeros(1, 2, 3),
            },
            "distillation_topk=2",
        ),
        (
            {"old_log_probs": torch.zeros(1, 3), "config": SDPOLossConfig(is_clip=2.0, distillation_topk=2)},
            "old_log_probs shape",
        ),
        ({"rollout_is_weights": torch.ones(1, 3)}, "rollout_is_weights shape"),
        ({"component_weights": torch.ones(1, 3)}, "component_weights shape"),
    ],
)
def test_sdpo_loss_rejects_misaligned_tensor_shapes(override, message):
    kwargs = _minimal_topk_loss_kwargs()
    kwargs.update(override)

    with pytest.raises(ValueError, match=message):
        compute_sdpo_loss(**kwargs)


def test_sdpo_full_vocab_loss_rejects_misaligned_distribution_shapes():
    with pytest.raises(ValueError, match="student_all_log_probs leading shape"):
        compute_sdpo_loss(
            student_log_probs=torch.zeros(1, 2),
            teacher_log_probs=torch.zeros(1, 2),
            response_mask=torch.ones(1, 2, dtype=torch.bool),
            student_all_log_probs=torch.zeros(1, 3, 4).log_softmax(dim=-1),
            teacher_all_log_probs=torch.zeros(1, 3, 4).log_softmax(dim=-1),
            config=SDPOLossConfig(full_logit_distillation=True, distillation_topk=None),
        )


def test_sdpo_loss_rejects_misaligned_self_distillation_mask():
    kwargs = _minimal_topk_loss_kwargs()
    kwargs["self_distillation_mask"] = torch.ones(2, dtype=torch.bool)

    with pytest.raises(ValueError, match="self_distillation_mask shape"):
        compute_sdpo_loss(**kwargs)


def test_sdpo_loss_rejects_non_boolean_response_mask():
    kwargs = _minimal_topk_loss_kwargs()
    kwargs["response_mask"] = torch.ones(1, 2)

    with pytest.raises(ValueError, match="response_mask must be a boolean mask"):
        compute_sdpo_loss(**kwargs)


def test_sdpo_loss_rejects_non_boolean_self_distillation_mask():
    kwargs = _minimal_topk_loss_kwargs()
    kwargs["self_distillation_mask"] = torch.ones(1)

    with pytest.raises(ValueError, match="self_distillation_mask must be a boolean mask"):
        compute_sdpo_loss(**kwargs)


def test_sdpo_topk_loss_rejects_invalid_log_probability_mass():
    kwargs = _minimal_topk_loss_kwargs()
    kwargs["teacher_topk_log_probs"] = torch.tensor([[[0.0, 0.0], [-0.5, -1.0]]])

    with pytest.raises(ValueError, match="teacher_topk_log_probs rows must have probability mass <= 1"):
        compute_sdpo_loss(**kwargs)


def test_sdpo_topk_loss_ignores_placeholder_probability_mass_outside_response_mask():
    kwargs = _minimal_topk_loss_kwargs()
    kwargs["response_mask"] = torch.tensor([[False, True]])
    kwargs["student_topk_log_probs"] = kwargs["student_topk_log_probs"].clone()
    kwargs["teacher_topk_log_probs"] = kwargs["teacher_topk_log_probs"].clone()
    kwargs["student_topk_log_probs"][0, 0] = torch.tensor([0.0, 0.0])
    kwargs["teacher_topk_log_probs"][0, 0] = torch.tensor([0.0, 0.0])

    sliced = dict(kwargs)
    sliced["student_log_probs"] = kwargs["student_log_probs"][:, 1:]
    sliced["teacher_log_probs"] = kwargs["teacher_log_probs"][:, 1:]
    sliced["response_mask"] = torch.ones(1, 1, dtype=torch.bool)
    sliced["student_topk_log_probs"] = kwargs["student_topk_log_probs"][:, 1:]
    sliced["teacher_topk_log_probs"] = kwargs["teacher_topk_log_probs"][:, 1:]

    loss = compute_sdpo_loss(**kwargs)
    expected = compute_sdpo_loss(**sliced)

    torch.testing.assert_close(loss, expected)


def test_sdpo_topk_loss_ignores_placeholder_probability_mass_outside_component_weights():
    kwargs = _minimal_topk_loss_kwargs()
    kwargs["component_weights"] = torch.tensor([[0.0, 1.5]])
    kwargs["student_topk_log_probs"] = kwargs["student_topk_log_probs"].clone()
    kwargs["teacher_topk_log_probs"] = kwargs["teacher_topk_log_probs"].clone()
    kwargs["student_topk_log_probs"][0, 0] = torch.tensor([0.0, 0.0])
    kwargs["teacher_topk_log_probs"][0, 0] = torch.tensor([0.0, 0.0])

    sliced = dict(kwargs)
    sliced["student_log_probs"] = kwargs["student_log_probs"][:, 1:]
    sliced["teacher_log_probs"] = kwargs["teacher_log_probs"][:, 1:]
    sliced["response_mask"] = torch.ones(1, 1, dtype=torch.bool)
    sliced["student_topk_log_probs"] = kwargs["student_topk_log_probs"][:, 1:]
    sliced["teacher_topk_log_probs"] = kwargs["teacher_topk_log_probs"][:, 1:]
    sliced["component_weights"] = kwargs["component_weights"][:, 1:]

    loss = compute_sdpo_loss(**kwargs)
    expected = compute_sdpo_loss(**sliced)

    torch.testing.assert_close(loss, expected)


def test_sdpo_full_vocab_loss_rejects_invalid_log_probability_mass():
    with pytest.raises(ValueError, match="student_all_log_probs rows must have probability mass <= 1"):
        compute_sdpo_loss(
            student_log_probs=torch.zeros(1, 1),
            teacher_log_probs=torch.zeros(1, 1),
            response_mask=torch.ones(1, 1, dtype=torch.bool),
            student_all_log_probs=torch.zeros(1, 1, 2),
            teacher_all_log_probs=torch.zeros(1, 1, 2).log_softmax(dim=-1),
            config=SDPOLossConfig(full_logit_distillation=True, distillation_topk=None),
        )


def test_sdpo_full_vocab_loss_ignores_placeholder_probability_mass_outside_response_mask():
    loss = compute_sdpo_loss(
        student_log_probs=torch.zeros(1, 2),
        teacher_log_probs=torch.zeros(1, 2),
        response_mask=torch.tensor([[False, True]]),
        student_all_log_probs=torch.tensor([[[0.0, 0.0], [-0.31326166, -1.31326163]]]),
        teacher_all_log_probs=torch.tensor([[[0.0, 0.0], [-0.474077, -0.974077]]]),
        config=SDPOLossConfig(full_logit_distillation=True, distillation_topk=None),
    )

    assert torch.isfinite(loss)


def test_sdpo_loss_rejects_nonfinite_distribution_rows():
    kwargs = _minimal_topk_loss_kwargs()
    kwargs["student_topk_log_probs"] = torch.tensor([[[float("nan"), -1.0], [-0.5, -1.0]]])

    with pytest.raises(ValueError, match="student_topk_log_probs must contain finite"):
        compute_sdpo_loss(**kwargs)


@pytest.mark.parametrize(
    ("field", "bad_tensor", "message"),
    [
        ("student_log_probs", torch.zeros(1, 2, dtype=torch.long), "student_log_probs must use a floating-point dtype"),
        ("teacher_log_probs", torch.zeros(1, 2, dtype=torch.bool), "teacher_log_probs must use a floating-point dtype"),
        (
            "student_topk_log_probs",
            torch.zeros(1, 2, 2, dtype=torch.long),
            "student_topk_log_probs must use a floating-point dtype",
        ),
        (
            "teacher_topk_log_probs",
            torch.zeros(1, 2, 2, dtype=torch.long),
            "teacher_topk_log_probs must use a floating-point dtype",
        ),
    ],
)
def test_sdpo_loss_rejects_non_floating_logprob_tensors(field, bad_tensor, message):
    kwargs = _minimal_topk_loss_kwargs()
    kwargs[field] = bad_tensor

    with pytest.raises(ValueError, match=message):
        compute_sdpo_loss(**kwargs)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"student_log_probs": torch.tensor([[0.0, float("nan")]])}, "student_log_probs must contain finite values"),
        ({"teacher_log_probs": torch.tensor([[0.0, float("inf")]])}, "teacher_log_probs must contain finite values"),
        (
            {
                "old_log_probs": torch.tensor([[0.0, float("nan")]]),
                "config": SDPOLossConfig(full_logit_distillation=True, distillation_topk=2, is_clip=2.0),
            },
            "old_log_probs must contain finite values",
        ),
        (
            {
                "old_log_probs": torch.tensor([[0.0, float("inf")]]),
                "config": SDPOLossConfig(
                    full_logit_distillation=True,
                    distillation_topk=2,
                    is_clip=None,
                    rollout_is="token",
                ),
            },
            "old_log_probs must contain finite values",
        ),
    ],
)
def test_sdpo_loss_rejects_nonfinite_sampled_logprob_streams(override, message):
    kwargs = _minimal_topk_loss_kwargs()
    kwargs.update(override)

    with pytest.raises(ValueError, match=message):
        compute_sdpo_loss(**kwargs)


@pytest.mark.parametrize(
    ("field", "bad_weights", "message"),
    [
        (
            "component_weights",
            torch.tensor([[True, False]]),
            "component_weights must contain finite non-negative numeric weights",
        ),
        (
            "component_weights",
            torch.tensor([[1.0 + 0.0j, 0.0 + 0.0j]]),
            "component_weights must contain finite non-negative numeric weights",
        ),
        ("component_weights", torch.tensor([[1.0, float("nan")]]), "component_weights must contain finite weights"),
        ("component_weights", torch.tensor([[1.0, -0.5]]), "component_weights must contain non-negative weights"),
        (
            "rollout_is_weights",
            torch.tensor([[True, False]]),
            "rollout_is_weights must contain finite non-negative numeric weights",
        ),
        (
            "rollout_is_weights",
            torch.tensor([[1.0 + 0.0j, 0.0 + 0.0j]]),
            "rollout_is_weights must contain finite non-negative numeric weights",
        ),
        ("rollout_is_weights", torch.tensor([[float("inf"), 1.0]]), "rollout_is_weights must contain finite weights"),
        ("rollout_is_weights", torch.tensor([[1.0, -0.25]]), "rollout_is_weights must contain non-negative weights"),
        (
            "rollout_is_weights",
            torch.tensor([[1, 1]], dtype=torch.long),
            "rollout_is_weights must use a floating-point dtype",
        ),
    ],
)
def test_sdpo_loss_rejects_invalid_weight_streams(field, bad_weights, message):
    kwargs = _minimal_topk_loss_kwargs()
    kwargs[field] = bad_weights

    with pytest.raises(ValueError, match=message):
        compute_sdpo_loss(**kwargs)


def test_sdpo_loss_rejects_supplied_rollout_is_weights_above_non_normalized_threshold():
    kwargs = _minimal_topk_loss_kwargs()
    kwargs["config"] = SDPOLossConfig(full_logit_distillation=True, distillation_topk=2, rollout_is="token")
    kwargs["rollout_is_weights"] = torch.tensor([[2.01, 1.0]])

    with pytest.raises(ValueError, match="rollout_is_weights must not exceed rollout_is_threshold=2.0"):
        compute_sdpo_loss(**kwargs)


def test_sdpo_loss_allows_batch_normalized_supplied_rollout_is_weights_above_threshold():
    kwargs = _minimal_topk_loss_kwargs()
    kwargs["config"] = SDPOLossConfig(
        full_logit_distillation=True,
        distillation_topk=2,
        rollout_is="token",
        rollout_is_batch_normalize=True,
    )
    kwargs["rollout_is_weights"] = torch.tensor([[2.01, 1.0]])

    loss = compute_sdpo_loss(**kwargs)

    assert torch.isfinite(loss)


def test_prime_sdpo_component_matches_topk_reference_constant():
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: _tensor(value) for name, value in case["tensors"].items()}
    loss_mask = tensors["response_mask"].squeeze(0)
    result = sdpo_loss_fn(
        LossInputs(
            trainer_logprobs=tensors["student_log_probs"].squeeze(0),
            inference_logprobs=tensors["student_log_probs"].squeeze(0),
            ref_logprobs=None,
            advantages=torch.zeros_like(tensors["student_log_probs"].squeeze(0)),
            loss_mask=loss_mask,
            student_topk_log_probs=tensors["student_topk_log_probs"].squeeze(0),
            teacher_topk_log_probs=tensors["teacher_topk_log_probs"].squeeze(0),
        ),
        SDPOComponentConfig(**case["config"]),
    )

    expected_sum = torch.tensor(case["expected_loss"], dtype=result.loss.dtype) * loss_mask.sum()
    assert torch.isclose(result.loss, expected_sum, atol=1e-6)
    assert torch.isclose(
        result.metrics["sdpo"], torch.tensor(case["expected_loss"], dtype=result.loss.dtype), atol=1e-6
    )


def test_sdpo_component_rejects_unpaired_topk_geometry():
    kwargs = _minimal_topk_loss_kwargs()
    with pytest.raises(ValueError, match="student and teacher logprob shapes must match"):
        sdpo_loss_fn(
            LossInputs(
                trainer_logprobs=kwargs["student_log_probs"].squeeze(0),
                inference_logprobs=kwargs["student_log_probs"].squeeze(0),
                ref_logprobs=None,
                advantages=torch.zeros_like(kwargs["student_log_probs"].squeeze(0)),
                loss_mask=kwargs["response_mask"].squeeze(0),
                student_topk_log_probs=kwargs["student_topk_log_probs"].squeeze(0),
                teacher_topk_log_probs=torch.zeros(2, 3),
            ),
            SDPOComponentConfig(full_logit_distillation=True, distillation_topk=2),
        )


def test_sdpo_component_weights_are_separate_from_rollout_is_weights():
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: _tensor(value) for name, value in case["tensors"].items()}
    component_weights = torch.tensor([[2.0, 1.0]])
    rollout_is_weights = torch.tensor([[0.5, 1.75]])

    direct = compute_sdpo_loss(
        config=SDPOLossConfig(**case["config"]),
        component_weights=component_weights,
        rollout_is_weights=rollout_is_weights,
        **tensors,
    )
    result = sdpo_loss_fn(
        LossInputs(
            trainer_logprobs=tensors["student_log_probs"].squeeze(0),
            inference_logprobs=tensors["student_log_probs"].squeeze(0),
            ref_logprobs=None,
            advantages=torch.zeros_like(tensors["student_log_probs"].squeeze(0)),
            loss_mask=tensors["response_mask"].squeeze(0),
            loss_weights=component_weights.squeeze(0),
            student_topk_log_probs=tensors["student_topk_log_probs"].squeeze(0),
            teacher_topk_log_probs=tensors["teacher_topk_log_probs"].squeeze(0),
            rollout_is_weights=rollout_is_weights.squeeze(0),
        ),
        SDPOComponentConfig(**case["config"]),
    )

    expected_sum = direct * tensors["response_mask"].sum()
    assert torch.isclose(result.loss, expected_sum, atol=1e-6)


def test_sdpo_component_weights_apply_as_membership_mask_before_topk_validation():
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: _tensor(value) for name, value in case["tensors"].items()}
    component_weights = torch.tensor([0.0, 1.5])
    student_topk = tensors["student_topk_log_probs"].squeeze(0).clone()
    teacher_topk = tensors["teacher_topk_log_probs"].squeeze(0).clone()
    student_topk[0] = 0.0
    teacher_topk[0] = 0.0

    direct = compute_sdpo_loss(
        config=SDPOLossConfig(**case["config"]),
        response_mask=torch.tensor([[True]]),
        component_weights=component_weights[1:].unsqueeze(0),
        student_log_probs=tensors["student_log_probs"][:, 1:],
        teacher_log_probs=tensors["teacher_log_probs"][:, 1:],
        student_topk_log_probs=tensors["student_topk_log_probs"][:, 1:],
        teacher_topk_log_probs=tensors["teacher_topk_log_probs"][:, 1:],
    )
    result = sdpo_loss_fn(
        LossInputs(
            trainer_logprobs=tensors["student_log_probs"].squeeze(0),
            inference_logprobs=tensors["student_log_probs"].squeeze(0),
            ref_logprobs=None,
            advantages=torch.zeros_like(tensors["student_log_probs"].squeeze(0)),
            loss_mask=tensors["response_mask"].squeeze(0),
            loss_weights=component_weights,
            student_topk_log_probs=student_topk,
            teacher_topk_log_probs=teacher_topk,
        ),
        SDPOComponentConfig(**case["config"]),
    )

    assert torch.isclose(result.loss, direct, atol=1e-6)
    assert torch.isclose(result.metrics["sdpo"], direct.detach(), atol=1e-6)


def test_rollout_is_weights_match_verl_token_level_truncation():
    log_ratio = torch.tensor([[torch.log(torch.tensor(3.0)), torch.log(torch.tensor(0.25)), 0.0]])
    mask = torch.tensor([[True, True, False]])

    weights = compute_rollout_is_weights(
        log_ratio=log_ratio,
        response_mask=mask,
        rollout_is="token",
        rollout_is_threshold=2.0,
    )

    assert torch.allclose(weights, torch.tensor([[2.0, 0.25, 0.0]]), atol=1e-6)


def test_rollout_is_weights_match_verl_sequence_level_truncation():
    log_ratio = torch.log(torch.tensor([[2.0, 0.5, 2.0], [2.0, 2.0, 1.0]]))
    mask = torch.tensor([[True, True, False], [True, True, False]])

    weights = compute_rollout_is_weights(
        log_ratio=log_ratio,
        response_mask=mask,
        rollout_is="sequence",
        rollout_is_threshold=2.0,
    )

    assert torch.allclose(weights, torch.tensor([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]]), atol=1e-6)


@pytest.mark.parametrize(
    "case",
    ROLLOUT_IS_WEIGHT_REFERENCE_CASES,
    ids=[case["name"] for case in ROLLOUT_IS_WEIGHT_REFERENCE_CASES],
)
def test_rollout_is_weights_batch_normalize_matches_huebotter_reference_constants(case):
    tensors = {name: _tensor(value) for name, value in case["tensors"].items()}

    weights = compute_rollout_is_weights(
        **tensors,
        **case["config"],
    )

    torch.testing.assert_close(weights, _tensor(case["expected_weights"]), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"log_ratio": torch.tensor([[0.0, float("inf")]])}, "log_ratio must contain finite values"),
        ({"response_mask": torch.ones(1, 3, dtype=torch.bool)}, "response_mask shape"),
        ({"response_mask": torch.tensor([[1.0, 0.5]])}, "response_mask must contain boolean/binary mask values"),
        ({"rollout_is_threshold": True}, "rollout_is_threshold must be a finite number"),
        ({"rollout_is_threshold": float("nan")}, "rollout_is_threshold must be a finite number"),
        ({"rollout_is_batch_normalize": "yes"}, "rollout_is_batch_normalize must be a boolean"),
    ],
)
def test_rollout_is_weights_rejects_malformed_inputs(override, message):
    kwargs = {
        "log_ratio": torch.zeros(1, 2),
        "response_mask": torch.ones(1, 2, dtype=torch.bool),
        "rollout_is": "token",
        "rollout_is_threshold": 2.0,
        "rollout_is_batch_normalize": False,
    }
    kwargs.update(override)

    with pytest.raises(ValueError, match=message):
        compute_rollout_is_weights(**kwargs)


def test_rollout_is_weights_rejects_non_floating_log_ratio():
    with pytest.raises(ValueError, match="log_ratio must use a floating-point dtype"):
        compute_rollout_is_weights(
            log_ratio=torch.zeros(1, 2, dtype=torch.long),
            response_mask=torch.ones(1, 2, dtype=torch.bool),
            rollout_is="token",
        )


def test_sdpo_component_computes_rollout_is_from_trainer_and_inference_logprobs():
    case = next(case for case in REFERENCE_CASES if case["name"] == "topk_distillation_with_tail")
    tensors = {name: _tensor(value) for name, value in case["tensors"].items()}
    loss_mask = tensors["response_mask"].squeeze(0)
    inference_logprobs = tensors["student_log_probs"].squeeze(0) - torch.log(torch.tensor([2.0, 0.25]))

    direct = compute_sdpo_loss(
        config=SDPOLossConfig(**case["config"], rollout_is="token", rollout_is_threshold=2.0),
        old_log_probs=inference_logprobs.unsqueeze(0),
        **tensors,
    )
    result = sdpo_loss_fn(
        LossInputs(
            trainer_logprobs=tensors["student_log_probs"].squeeze(0),
            inference_logprobs=inference_logprobs,
            ref_logprobs=None,
            advantages=torch.zeros_like(tensors["student_log_probs"].squeeze(0)),
            loss_mask=loss_mask,
            student_topk_log_probs=tensors["student_topk_log_probs"].squeeze(0),
            teacher_topk_log_probs=tensors["teacher_topk_log_probs"].squeeze(0),
        ),
        SDPOComponentConfig(**case["config"], rollout_is="token", rollout_is_threshold=2.0),
    )

    assert torch.isclose(result.loss, direct * loss_mask.sum(), atol=1e-6)
