import torch

from prime_rl.latent.recurrent import (
    OneShotFeedForwardSidecar,
    RecurrentState,
    TimestepFreeRecurrentSidecar,
    diagnose_recurrent_states,
)


def test_default_recurrent_and_feed_forward_controls_match_active_parameter_count():
    recurrent = TimestepFreeRecurrentSidecar()
    feed_forward = OneShotFeedForwardSidecar()

    recurrent_parameters = sum(parameter.numel() for parameter in recurrent.parameters())
    feed_forward_parameters = sum(parameter.numel() for parameter in feed_forward.parameters())

    assert recurrent_parameters == 10_562_048
    assert feed_forward_parameters == 10_562_155
    assert abs(recurrent_parameters - feed_forward_parameters) / recurrent_parameters < 0.01


def test_recurrent_sidecar_is_timestep_free_and_exports_only_visible_workspace():
    torch.manual_seed(7)
    sidecar = TimestepFreeRecurrentSidecar(workspace_dim=4, transition_dim=12)
    anchor = torch.randn(2, 3, 4)
    state = sidecar.initial_state(anchor)

    first = sidecar.step(anchor, state)
    repeated = sidecar.step(anchor, state)

    assert torch.equal(first.visible_workspace, repeated.visible_workspace)
    assert torch.equal(first.private_memory, repeated.private_memory)
    assert first.export_workspace() is first.visible_workspace
    assert first.private_memory is not first.export_workspace()


def test_recurrent_sidecar_has_exact_noop_visible_initialization_and_evolving_private_memory():
    torch.manual_seed(11)
    sidecar = TimestepFreeRecurrentSidecar(workspace_dim=5, transition_dim=16)
    anchor = torch.randn(1, 2, 5)

    trajectory = sidecar.rollout(anchor, 3, return_trajectory=True)

    assert all(torch.equal(state.visible_workspace, anchor) for state in trajectory)
    assert not torch.equal(trajectory[1].private_memory, trajectory[2].private_memory)


def test_truncated_rollout_detaches_persistent_history():
    torch.manual_seed(13)
    sidecar = TimestepFreeRecurrentSidecar(workspace_dim=3, transition_dim=9)
    sidecar.output_scale.data.fill_(0.2)
    anchor = torch.randn(1, 2, 3)
    initial_memory = torch.zeros_like(anchor, requires_grad=True)
    initial = RecurrentState(visible_workspace=anchor, private_memory=initial_memory)

    final = sidecar.rollout(anchor, 4, state=initial, truncate_every=2)
    final.visible_workspace.sum().backward()

    assert initial_memory.grad is None
    assert sidecar.transition.weight.grad is not None
    assert torch.isfinite(sidecar.transition.weight.grad).all()


def test_fixed_residual_scale_diagnostic_opens_full_path_without_mutating_parameters():
    torch.manual_seed(19)
    recurrent = TimestepFreeRecurrentSidecar(workspace_dim=3, transition_dim=9)
    feed_forward = OneShotFeedForwardSidecar(workspace_dim=3, hidden_dim=12)
    anchor = torch.randn(1, 2, 3)
    recurrent_scale = recurrent.output_scale.detach().clone()
    feed_forward_scale = feed_forward.output_scale.detach().clone()

    recurrent_result = recurrent.rollout(anchor, 2, residual_scale_override=0.001)
    feed_forward_result = feed_forward(anchor, residual_scale_override=0.001)
    (recurrent_result.visible_workspace.sum() + feed_forward_result.sum()).backward()

    assert recurrent.transition.weight.grad is not None
    assert recurrent.workspace_delta.weight.grad is not None
    assert feed_forward.hidden.weight.grad is not None
    assert feed_forward.output.weight.grad is not None
    assert torch.equal(recurrent.output_scale.detach(), recurrent_scale)
    assert torch.equal(feed_forward.output_scale.detach(), feed_forward_scale)
    assert recurrent.output_scale.grad is None
    assert feed_forward.output_scale.grad is None


def test_recurrent_diagnostics_separate_contraction_from_oscillation():
    def state(value: float) -> RecurrentState:
        tensor = torch.tensor([[[value]]])
        return RecurrentState(visible_workspace=tensor, private_memory=tensor)

    contracting = diagnose_recurrent_states(tuple(state(value) for value in (0.0, 1.0, 1.5, 1.75)))
    oscillating = diagnose_recurrent_states(tuple(state(value) for value in (0.0, 1.0, 0.0, 1.0)))

    assert torch.allclose(contracting.visible_contraction_ratios, torch.full((2, 1), 0.5))
    assert contracting.visible_oscillation_rate == 0.0
    assert oscillating.visible_oscillation_rate == 1.0
    assert not contracting.nonfinite
