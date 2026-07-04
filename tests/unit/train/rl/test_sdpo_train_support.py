from pathlib import Path

import pytest
import torch

from prime_rl.trainer.rl.sdpo_train_support import (
    active_sdpo_weight_mask,
    has_weighted_sdpo_tokens,
    require_sdpo_student_support_export_supported,
    require_sdpo_student_support_logits,
    resolve_preflight_batch_mode,
    should_export_sdpo_student_support,
)


def test_resolve_preflight_batch_mode_accepts_uniform_train_batches():
    assert not resolve_preflight_batch_mode(
        [{"preflight_only": False}, {"preflight_only": False}],
        enable_token_export=False,
    )


def test_resolve_preflight_batch_mode_accepts_uniform_preflight_batches_with_token_export():
    assert resolve_preflight_batch_mode(
        [{"preflight_only": True}, {"preflight_only": True}],
        enable_token_export=True,
    )


def test_resolve_preflight_batch_mode_rejects_empty_batches():
    with pytest.raises(ValueError, match="empty trainer batch"):
        resolve_preflight_batch_mode([], enable_token_export=True)


def test_resolve_preflight_batch_mode_rejects_missing_preflight_flags():
    with pytest.raises(ValueError, match="missing at index 1"):
        resolve_preflight_batch_mode(
            [{"preflight_only": True}, {}],
            enable_token_export=True,
        )


def test_resolve_preflight_batch_mode_rejects_mixed_preflight_and_train_batches():
    with pytest.raises(ValueError, match="Cannot mix preflight-only and train micro batches"):
        resolve_preflight_batch_mode(
            [{"preflight_only": True}, {"preflight_only": False}],
            enable_token_export=True,
        )


@pytest.mark.parametrize("preflight_only", ["False", 0, None])
def test_resolve_preflight_batch_mode_rejects_non_boolean_preflight_flags(preflight_only):
    with pytest.raises(ValueError, match="preflight_only must be a boolean"):
        resolve_preflight_batch_mode(
            [{"preflight_only": preflight_only}],
            enable_token_export=True,
        )


def test_resolve_preflight_batch_mode_requires_token_export_for_preflight_batches():
    with pytest.raises(ValueError, match="preflight-only batches require trainer.enable_token_export=true"):
        resolve_preflight_batch_mode(
            [{"preflight_only": True}],
            enable_token_export=False,
        )


def test_train_loop_preflight_continue_precedes_training_side_effects():
    source = (Path(__file__).resolve().parents[4] / "src" / "prime_rl" / "trainer" / "rl" / "train.py").read_text()

    mark_stable_idx = source.index("token_exporter.mark_stable(ready_run_ids)")
    preflight_guard_idx = source.index("if preflight_only:", mark_stable_idx)
    continue_idx = source.index("continue", preflight_guard_idx)

    side_effects = [
        "param.grad.mul_(parallel_dims.fsdp_gradient_divide_factor)",
        "optimizer.step()",
        "optimizer.zero_grad()",
        "scheduler.step()",
        "step_sdpo_teacher_regularization_if_updated(",
        "progress.total_tokens +=",
        "progress.total_samples +=",
        "monitor.log(",
        "progress.step +=",
    ]

    for side_effect in side_effects:
        assert continue_idx < source.index(side_effect), side_effect


def test_train_loop_step_maintenance_is_cached_across_preflight_reentry():
    source = (Path(__file__).resolve().parents[4] / "src" / "prime_rl" / "trainer" / "rl" / "train.py").read_text()

    loop_idx = source.index("while True:")
    cache_guard_idx = source.index("if step_maintenance_step == progress.step:", loop_idx)
    cached_broadcast_idx = source.index(
        "broadcast_weights_time = step_maintenance_broadcast_weights_time", cache_guard_idx
    )
    cached_ckpt_idx = source.index("save_ckpt_time = step_maintenance_save_ckpt_time", cache_guard_idx)
    broadcast_idx = source.index("weight_broadcast.broadcast_weights(", cache_guard_idx)
    ckpt_idx = source.index("ckpt_manager.save(", cache_guard_idx)
    cache_update_idx = source.index("step_maintenance_step = progress.step", ckpt_idx)
    max_steps_guard_idx = source.index(
        "if config.max_steps is not None and progress.step >= config.max_steps:", loop_idx
    )

    assert cache_guard_idx < cached_broadcast_idx < broadcast_idx
    assert cache_guard_idx < cached_ckpt_idx < ckpt_idx
    assert broadcast_idx < cache_update_idx < max_steps_guard_idx


def test_should_export_sdpo_student_support_requires_weighted_sdpo_tokens():
    assert not should_export_sdpo_student_support(
        enable_token_export=True,
        sdpo_weights=None,
        loss_mask=torch.tensor([[True, True]]),
        full_logit_distillation=True,
        distillation_topk=2,
    )
    assert not should_export_sdpo_student_support(
        enable_token_export=True,
        sdpo_weights=torch.tensor([[0.0, 0.0]]),
        loss_mask=torch.tensor([[True, True]]),
        full_logit_distillation=True,
        distillation_topk=2,
    )
    assert not should_export_sdpo_student_support(
        enable_token_export=True,
        sdpo_weights=torch.tensor([[1.0, 0.0]]),
        loss_mask=torch.tensor([[False, True]]),
        full_logit_distillation=True,
        distillation_topk=2,
    )
    assert should_export_sdpo_student_support(
        enable_token_export=True,
        sdpo_weights=torch.tensor([[0.0, 1.0]]),
        loss_mask=torch.tensor([[False, True]]),
        full_logit_distillation=True,
        distillation_topk=2,
    )


def test_active_sdpo_weight_mask_defines_trainer_tensor_membership():
    weights = torch.tensor([[0.0, -0.0, 0.25, 1.0]])

    mask = active_sdpo_weight_mask(weights)

    assert mask.tolist() == [[False, False, True, True]]
    assert has_weighted_sdpo_tokens(weights)
    assert not has_weighted_sdpo_tokens(weights, loss_mask=torch.tensor([[True, True, False, False]]))


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (torch.tensor([[False, True]]), "finite non-negative numeric values"),
        (torch.tensor([[0.0, float("nan")]]), "finite weights"),
        (torch.tensor([[0.0, float("inf")]]), "finite weights"),
        (torch.tensor([[0.0, -0.25]]), "non-negative weights"),
        (torch.tensor([[0.0, 0.25 + 0j]]), "finite non-negative numeric values"),
    ],
)
def test_active_sdpo_weight_mask_rejects_malformed_trainer_tensors(weights, message):
    with pytest.raises(ValueError, match=message):
        active_sdpo_weight_mask(weights)


def test_should_export_sdpo_student_support_rejects_malformed_sdpo_weights():
    with pytest.raises(ValueError, match="finite non-negative numeric values"):
        should_export_sdpo_student_support(
            enable_token_export=True,
            sdpo_weights=torch.tensor([[False, True]]),
            loss_mask=torch.tensor([[True, True]]),
            full_logit_distillation=True,
            distillation_topk=2,
        )


def test_should_export_sdpo_student_support_is_not_preflight_only():
    # Final SDPO batches also export student support for smoke verification.
    assert should_export_sdpo_student_support(
        enable_token_export=True,
        sdpo_weights=torch.tensor([[1.0]]),
        loss_mask=torch.tensor([[True]]),
        full_logit_distillation=True,
        distillation_topk=2,
    )


@pytest.mark.parametrize(
    ("enable_token_export", "full_logit_distillation", "distillation_topk"),
    [
        (False, True, 2),
        (True, False, 2),
        (True, True, None),
    ],
)
def test_should_export_sdpo_student_support_requires_topk_export_mode(
    enable_token_export, full_logit_distillation, distillation_topk
):
    assert not should_export_sdpo_student_support(
        enable_token_export=enable_token_export,
        sdpo_weights=torch.tensor([[1.0]]),
        loss_mask=torch.tensor([[True]]),
        full_logit_distillation=full_logit_distillation,
        distillation_topk=distillation_topk,
    )


def test_require_sdpo_student_support_export_supported_rejects_context_parallelism():
    with pytest.raises(NotImplementedError, match="context parallelism"):
        require_sdpo_student_support_export_supported(cp_enabled=True)


def test_require_sdpo_student_support_export_supported_accepts_non_context_parallelism():
    require_sdpo_student_support_export_supported(cp_enabled=False)


def test_require_sdpo_student_support_logits_returns_float_logits():
    logits = torch.zeros(1, 2, 3)

    assert require_sdpo_student_support_logits({"logits": logits}) is logits


@pytest.mark.parametrize(
    ("model_output", "message"),
    [
        ({}, "requires logits"),
        ({"logits": [[0.0]]}, "logits to be a tensor"),
        ({"logits": torch.zeros(1, 2, 3, dtype=torch.long)}, "floating-point logits"),
        ({"logits": torch.zeros(2, 3)}, "shape"),
        ({"logits": torch.tensor([[[0.0, float("inf")]]])}, "finite logits"),
    ],
)
def test_require_sdpo_student_support_logits_rejects_missing_or_malformed_logits(model_output, message):
    with pytest.raises(ValueError, match=message):
        require_sdpo_student_support_logits(model_output)
