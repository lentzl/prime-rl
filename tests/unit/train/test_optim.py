import pytest
import torch
from torch.optim import AdamW

from prime_rl.trainer.optim import CPUOffloadOptimizer, optimizer_update_succeeded


@pytest.mark.parametrize("value", [None, torch.tensor(0.0), torch.tensor(1.0), torch.tensor([-1.0, 2.0])])
def test_optimizer_update_succeeds_for_absent_or_finite_grad_norm(value):
    assert optimizer_update_succeeded(value)


@pytest.mark.parametrize(
    "value", [torch.tensor(float("nan")), torch.tensor(float("inf")), torch.tensor([1.0, float("-inf")])]
)
def test_optimizer_update_rejects_nonfinite_grad_norm(value):
    assert not optimizer_update_succeeded(value)


@pytest.mark.gpu
def test_cpu_offload_adamw_chunking_matches_regular_adamw():
    baseline_params = [
        torch.nn.Parameter(torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3)),
        torch.nn.Parameter(torch.arange(4, dtype=torch.float32, device="cuda")),
    ]
    chunked_params = [torch.nn.Parameter(parameter.detach().clone()) for parameter in baseline_params]
    baseline = AdamW(baseline_params, lr=0.03, weight_decay=0.01, foreach=True)
    chunked = CPUOffloadOptimizer(
        AdamW(chunked_params, lr=0.03, weight_decay=0.01, foreach=True),
        max_state_chunk_numel=4,
    )

    for step in range(2):
        for index, (baseline_param, chunked_param) in enumerate(zip(baseline_params, chunked_params)):
            gradient = torch.full_like(baseline_param, 0.2 + step + index)
            baseline_param.grad = gradient
            chunked_param.grad = gradient.clone()

        baseline.step()
        chunked.step()

        for baseline_param, chunked_param in zip(baseline_params, chunked_params):
            torch.testing.assert_close(chunked_param, baseline_param)
        assert chunked.param_groups[0]["params"] == chunked_params
        for state in chunked.state.values():
            assert all(not isinstance(value, torch.Tensor) or value.device.type == "cpu" for value in state.values())


@pytest.mark.gpu
def test_cpu_offload_adamw_initializes_state_in_chunks_without_an_update():
    parameters = [
        torch.nn.Parameter(torch.arange(3, dtype=torch.float32, device="cuda")),
        torch.nn.Parameter(torch.arange(4, dtype=torch.float32, device="cuda")),
    ]
    optimizer = AdamW(parameters, lr=0.03, weight_decay=0.01, foreach=True)
    step_parameter_counts = []
    original_step = optimizer.step

    def tracked_step(*args, **kwargs):
        step_parameter_counts.append(sum(len(group["params"]) for group in optimizer.param_groups))
        return original_step(*args, **kwargs)

    optimizer.step = tracked_step
    wrapped = CPUOffloadOptimizer(optimizer, max_state_chunk_numel=4)
    original_parameters = [parameter.detach().clone() for parameter in parameters]

    wrapped.initialize_state()

    assert wrapped._initialized
    assert step_parameter_counts == [1, 1]
    assert wrapped.param_groups[0]["params"] == parameters
    assert wrapped.param_groups[0]["lr"] == 0.03
    for parameter, original_parameter in zip(parameters, original_parameters):
        torch.testing.assert_close(parameter, original_parameter)
        assert parameter.grad is None
        state = wrapped.state[parameter]
        assert state["step"].item() == 1
        assert all(not isinstance(value, torch.Tensor) or value.device.type == "cpu" for value in state.values())


def test_cpu_offload_optimizer_rejects_nonpositive_chunk_size():
    parameter = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="max_state_chunk_numel must be positive"):
        CPUOffloadOptimizer(AdamW([parameter]), max_state_chunk_numel=0)
