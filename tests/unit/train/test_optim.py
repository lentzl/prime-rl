import pytest
import torch

from prime_rl.configs.trainer import AdamWConfig
from prime_rl.trainer.optim import optimizer_update_succeeded, setup_optimizer
from prime_rl.trainer.parallel_dims import ParallelDims


@pytest.mark.parametrize("foreach", [None, False, True])
def test_adamw_foreach_is_configurable(foreach):
    parameter = torch.nn.Parameter(torch.ones(1))
    parallel_dims = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, pp=1, ep=1, world_size=1)

    optimizer = setup_optimizer(
        AdamWConfig(foreach=foreach),
        [("parameter", parameter)],
        parallel_dims,
    )

    assert optimizer.defaults["foreach"] is foreach


@pytest.mark.parametrize("value", [None, torch.tensor(0.0), torch.tensor(1.0), torch.tensor([-1.0, 2.0])])
def test_optimizer_update_succeeds_for_absent_or_finite_grad_norm(value):
    assert optimizer_update_succeeded(value)


@pytest.mark.parametrize(
    "value", [torch.tensor(float("nan")), torch.tensor(float("inf")), torch.tensor([1.0, float("-inf")])]
)
def test_optimizer_update_rejects_nonfinite_grad_norm(value):
    assert not optimizer_update_succeeded(value)
