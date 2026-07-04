import torch

from prime_rl.trainer.optim import optimizer_update_succeeded


def test_optimizer_update_succeeded_treats_absent_grad_norm_as_success():
    assert optimizer_update_succeeded(None)


def test_optimizer_update_succeeded_checks_grad_norm_finiteness():
    assert optimizer_update_succeeded(torch.tensor(1.0))
    assert not optimizer_update_succeeded(torch.tensor(float("nan")))
    assert not optimizer_update_succeeded(torch.tensor(float("inf")))


def test_optimizer_update_succeeded_accepts_all_finite_tensor_values():
    assert optimizer_update_succeeded(torch.tensor([1.0, 2.0]))
    assert not optimizer_update_succeeded(torch.tensor([1.0, float("nan")]))
