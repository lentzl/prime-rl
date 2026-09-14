"""CPU objective checks, not policy learning or distributed-framework tests."""

from decimal import Decimal, localcontext
import math

import pytest
import torch

from flash_reinforce import (
    loss_fragment,
    merge_statistics,
    prepare_update,
    sampled_action_kl,
    trajectory_statistics,
)


def make_plan(current, behavior, index, mask, rewards, threshold=1.0):
    stats = trajectory_statistics(current, behavior, index, mask, batch_size=len(rewards))
    return prepare_update(rewards, stats, threshold=threshold)


def tensor(values, *, grad=False, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype, requires_grad=grad)


def test_gradient_matches_independent_ratio_objective():
    logits = tensor([[1.0, -0.5], [0.2, 0.1], [-0.1, 1.5], [1.0, 1.1], [0.2, 1.2]], grad=True)
    current = logits.log_softmax(-1)[:, 0]
    behavior = tensor([-0.4, -0.5, -1.2, -0.6, -0.8])
    index = torch.tensor([0, 1, 2, 0, 1])
    mask = torch.tensor([True, True, True, True, True])
    rewards = tensor([1.0, 0.0, 0.4])
    plan = make_plan(current, behavior, index, mask, rewards)
    actual = loss_fragment(current, behavior, index, mask, plan)
    # Independent Eq. 9 construction: differentiate ratios, not detached ratios*logp.
    reference = torch.zeros((), dtype=torch.float64)
    for i in range(3):
        selected = index == i
        reference -= (rewards[i] - rewards.mean()) * (current[selected] - behavior[selected]).exp().mean() / 3
    expected_grad = torch.autograd.grad(reference, logits, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)


def test_advantage_sign_and_detachment():
    current = tensor([-0.5, -0.5], grad=True)
    behavior = tensor([-0.5, -0.5], grad=True)
    rewards = tensor([1.0, 0.0], grad=True)
    index, mask = torch.tensor([0, 1]), torch.ones(2, dtype=torch.bool)
    plan = make_plan(current, behavior, index, mask, rewards)
    loss_fragment(current, behavior, index, mask, plan).backward()
    torch.testing.assert_close(current.grad, tensor([-0.25, 0.25]))
    assert behavior.grad is None and rewards.grad is None
    assert not plan.advantages.requires_grad and not plan.mean_divergence.requires_grad
    assert plan.has_reward_signal


def test_rejected_trajectories_remain_in_original_baseline_and_denominator():
    current = tensor([-0.5, -0.001, -0.001], grad=True)
    behavior = tensor([-0.5, -5.0, -5.0])
    rewards = tensor([1.0, 0.0, 0.0])
    index, mask = torch.arange(3), torch.ones(3, dtype=torch.bool)
    plan = make_plan(current, behavior, index, mask, rewards, threshold=0.01)
    assert plan.accepted.tolist() == [True, False, False]
    loss = loss_fragment(current, behavior, index, mask, plan)
    torch.testing.assert_close(loss, tensor(1 / 9))
    loss.backward()
    torch.testing.assert_close(current.grad, tensor([-2 / 9, 0.0, 0.0]))


def test_length_replication_does_not_increase_trajectory_weight():
    def run(lengths):
        parameters = tensor([-0.4, -0.7], grad=True)
        index = torch.repeat_interleave(torch.arange(2), torch.tensor(lengths))
        current = parameters[index]
        behavior = tensor([-0.5, -0.6])[index]
        mask, rewards = torch.ones(len(index), dtype=torch.bool), tensor([1.0, 0.0])
        plan = make_plan(current, behavior, index, mask, rewards)
        loss = loss_fragment(current, behavior, index, mask, plan)
        return loss.detach(), torch.autograd.grad(loss, parameters)[0]
    a, b = run([1, 1]), run([7, 19])
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-12)


def test_fragmented_accumulation_matches_whole_batch_and_whole_trajectory_gate():
    current = tensor([math.log(0.5), math.log(0.9), math.log(0.55), math.log(0.52), math.log(0.53)], grad=True)
    behavior = tensor([math.log(0.5)] * 5)
    index, mask, rewards = torch.tensor([0, 0, 1, 1, 1]), torch.ones(5, dtype=torch.bool), tensor([1.0, 0.0])
    whole = trajectory_statistics(current, behavior, index, mask, batch_size=2)
    whole_plan = prepare_update(rewards, whole, threshold=0.1)
    # The first fragment of trajectory 0 looks harmless, but the WHOLE episode fails the gate.
    assert whole_plan.accepted.tolist() == [False, True]
    whole_loss = loss_fragment(current, behavior, index, mask, whole_plan)
    whole_grad = torch.autograd.grad(whole_loss, current)[0]
    fragments = [torch.tensor([0, 2]), torch.tensor([1, 4]), torch.tensor([3]), torch.tensor([], dtype=torch.int64)]
    parts = [trajectory_statistics(current[f], behavior[f], index[f], mask[f], batch_size=2) for f in fragments]
    merged = merge_statistics(parts)
    split_plan = prepare_update(rewards, merged, threshold=0.1, expected_token_counts=torch.tensor([2, 3]))
    torch.testing.assert_close(split_plan.mean_divergence, whole_plan.mean_divergence)
    total = torch.zeros((), dtype=torch.float64)
    for f in fragments:
        fragment_loss = loss_fragment(current[f], behavior[f], index[f], mask[f], split_plan)
        total += fragment_loss.detach()
        fragment_loss.backward()
    torch.testing.assert_close(total, whole_loss.detach())
    torch.testing.assert_close(current.grad, whole_grad)
    torch.testing.assert_close(current.grad[:2], torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize('mode', ['equal_rewards', 'all_rejected', 'single_trajectory'])
def test_zero_signal_is_differentiable_and_finite(mode):
    current = tensor([-0.5, -0.7], grad=True)
    index = torch.tensor([0, 0] if mode == 'single_trajectory' else [0, 1])
    rewards = tensor([1.0] if mode == 'single_trajectory' else ([0.0, 0.0] if mode == 'equal_rewards' else [1.0, 0.0]))
    behavior = tensor([-5.0, -5.0] if mode == 'all_rejected' else [-0.5, -0.7])
    mask = torch.ones(2, dtype=torch.bool)
    plan = make_plan(current, behavior, index, mask, rewards, threshold=0.001)
    loss = loss_fragment(current, behavior, index, mask, plan)
    assert loss.item() == 0 and not plan.has_reward_signal
    loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros(2, dtype=torch.float64))


def test_masked_context_is_never_exponentiated_or_used_in_counts():
    current = tensor([float('nan'), -0.4, float('inf'), -0.8], grad=True)
    behavior = tensor([float('nan'), -0.5, float('inf'), -0.7])
    index, mask, rewards = torch.tensor([-999, 0, -999, 1]), torch.tensor([False, True, False, True]), tensor([1.0, 0.0])
    plan = make_plan(current, behavior, index, mask, rewards)
    assert plan.token_counts.tolist() == [1, 1]
    loss = loss_fragment(current, behavior, index, mask, plan)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(current.grad).all()
    assert current.grad[0] == 0 and current.grad[2] == 0


def decimal_kl(old, new):
    with localcontext() as ctx:
        ctx.prec = 70
        old, new = Decimal(str(old)), Decimal(str(new))
        p, q = old.exp(), new.exp()
        return float(p * (old - new) + (1 - p) * ((1 - p).ln() - (1 - q).ln()))


@pytest.mark.parametrize('dtype,rtol,atol', [(torch.float64, 2e-9, 1e-15), (torch.float32, 3e-5, 2e-8)])
def test_kl_near_one_and_small_probabilities_against_decimal(dtype, rtol, atol):
    old = tensor([-3.0135979652404785, -0.0000003576278118089249, -0.00003075552376685664, -1e-12, -90.0, -1000.0], dtype=dtype)
    new = old * 1.7
    expected = tensor([decimal_kl(a, b) for a, b in zip(old.tolist(), new.tolist())], dtype=dtype)
    actual = sampled_action_kl(old, new)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(sampled_action_kl(old, old), torch.zeros_like(old), rtol=0, atol=0)


def test_probability_one_endpoints_and_rejection():
    old, new = tensor([0.0, -0.1, 0.0]), tensor([-0.1, 0.0, 0.0], grad=True)
    values = sampled_action_kl(old, new)
    assert values[0] == pytest.approx(0.1) and torch.isposinf(values[1]) and values[2] == 0
    index, mask, rewards = torch.arange(3), torch.ones(3, dtype=torch.bool), tensor([1.0, 0.0, 0.0])
    plan = make_plan(new, old, index, mask, rewards, threshold=1.0)
    assert plan.accepted.tolist() == [True, False, True]
    loss = loss_fragment(new, old, index, mask, plan)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(new.grad).all() and new.grad[1] == 0


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_scalars_are_promoted(dtype):
    current, behavior = tensor([-0.4, -0.8], grad=True, dtype=dtype), tensor([-0.5, -0.7], dtype=dtype)
    index, mask, rewards = torch.arange(2), torch.ones(2, dtype=torch.bool), tensor([1.0, 0.0], dtype=dtype)
    plan = make_plan(current, behavior, index, mask, rewards)
    loss = loss_fragment(current, behavior, index, mask, plan)
    assert loss.dtype == torch.float32 and plan.mean_divergence.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(current.grad).all()


def test_historical_behavior_example_is_only_a_numerical_fixture():
    # Trainer issue comment 5669869979; real old behavior values, no old learner values or current rollout claim.
    values = [-3.0135979652404785, -0.00003075552376685664, -0.0000003576278118089249,
              -0.3231099843978882, -1.106938123703003, -0.0002383901592111215,
              -0.019170014187693596, -1.7358872890472412, -0.00013553177996072918,
              -0.00003099393507000059, -0.000004529942543740617,
              -0.000013828182090946939, -0.00005709961988031864]
    behavior = tensor(values)
    current = (behavior * 1.1).requires_grad_()  # Explicit SYNTHETIC learner perturbation.
    kl = sampled_action_kl(behavior, current)
    expected = tensor([decimal_kl(a, b) for a, b in zip(behavior.tolist(), current.tolist())])
    torch.testing.assert_close(kl, expected, rtol=1e-9, atol=1e-15)
    assert torch.isfinite(kl).all() and (kl >= 0).all()


def test_overflow_is_not_silently_ratio_clipped_and_rejected_tokens_are_safe():
    current = tensor([-0.1, -0.2], grad=True, dtype=torch.float32)
    behavior = tensor([-1000.0, -0.2], dtype=torch.float32)
    index, mask, rewards = torch.arange(2), torch.ones(2, dtype=torch.bool), tensor([1.0, 0.0], dtype=torch.float32)
    admitted = make_plan(current, behavior, index, mask, rewards, threshold=3.0)
    with pytest.raises(FloatingPointError, match='ratio overflow'):
        loss_fragment(current, behavior, index, mask, admitted)
    rejected = make_plan(current, behavior, index, mask, rewards, threshold=0.001)
    loss_fragment(current, behavior, index, mask, rejected).backward()
    assert current.grad[0] == 0 and torch.isfinite(current.grad).all()


def test_missing_duplicate_and_zero_action_trajectories_are_not_silently_renormalized():
    current, old, index, mask = tensor([-0.5, -0.6]), tensor([-0.5, -0.6]), torch.arange(2), torch.ones(2, dtype=torch.bool)
    stats = trajectory_statistics(current, old, index, mask, batch_size=2)
    rewards = tensor([1.0, 0.0])
    with pytest.raises(ValueError, match='missing, duplicated or truncated'):
        prepare_update(rewards, stats, threshold=0.01, expected_token_counts=torch.tensor([2, 1]))
    with pytest.raises(ValueError, match='missing, duplicated or truncated'):
        prepare_update(rewards, merge_statistics([stats, stats]), threshold=0.01, expected_token_counts=torch.tensor([1, 1]))
    incomplete = trajectory_statistics(current[:1], old[:1], index[:1], mask[:1], batch_size=2)
    with pytest.raises(ValueError, match='omit a trajectory'):
        prepare_update(rewards, incomplete, threshold=0.01)
    with pytest.raises(ValueError, match='one reward per original'):
        prepare_update(rewards[:1], stats, threshold=0.01)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf'), 0.1])
def test_invalid_selected_logprobs_raise(bad):
    with pytest.raises(ValueError, match='finite log-probabilities'):
        trajectory_statistics(tensor([bad]), tensor([-0.5]), torch.tensor([0]), torch.tensor([True]), batch_size=1)


def test_invalid_masks_ids_and_thresholds_raise():
    current, old, index, mask = tensor([-0.5]), tensor([-0.5]), torch.tensor([0]), torch.tensor([True])
    with pytest.raises(TypeError, match='must be bool'):
        trajectory_statistics(current, old, index, mask.long(), batch_size=1)
    with pytest.raises(ValueError, match='indices must lie'):
        trajectory_statistics(current, old, index + 1, mask, batch_size=1)
    with pytest.raises(ValueError, match='aligned 1-D'):
        trajectory_statistics(current[:, None], old, index, mask, batch_size=1)
    stats = trajectory_statistics(current, old, index, mask, batch_size=1)
    for threshold in (-1.0, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='threshold'):
            prepare_update(tensor([0.0]), stats, threshold=threshold)
