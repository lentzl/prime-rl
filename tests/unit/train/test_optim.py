import pytest
import torch

from prime_rl.trainer.optim import optimizer_update_succeeded


@pytest.mark.parametrize("value", [None, torch.tensor(0.0), torch.tensor(1.0), torch.tensor([-1.0, 2.0])])
def test_optimizer_update_succeeds_for_absent_or_finite_grad_norm(value):
    assert optimizer_update_succeeded(value)


@pytest.mark.parametrize(
    "value", [torch.tensor(float("nan")), torch.tensor(float("inf")), torch.tensor([1.0, float("-inf")])]
)
def test_optimizer_update_rejects_nonfinite_grad_norm(value):
    assert not optimizer_update_succeeded(value)
