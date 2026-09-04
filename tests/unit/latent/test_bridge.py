import torch

from prime_rl.latent.bridge import WorkspaceBridge, WorkspaceBridgeConfig


def test_bridge_has_expected_shape_scale_and_parameter_budget() -> None:
    config = WorkspaceBridgeConfig(
        source_width=16,
        workspace_width=8,
        receiver_width=16,
        slots=3,
        initial_receiver_gate=0.0,
    )
    bridge = WorkspaceBridge(config)
    parent = torch.randn(2, 5, 16)
    mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]])

    projected = bridge(parent, parent_attention_mask=mask, embedding_shell_norm=torch.tensor(4.0))

    assert projected.shape == (2, 3, 16)
    assert torch.count_nonzero(projected).item() == 0
    assert bridge.trainable_parameter_count() > 0
    assert len(config.checksum()) == 64


def test_receiver_gate_is_bounded_and_receives_gradient() -> None:
    config = WorkspaceBridgeConfig(source_width=16, workspace_width=8, receiver_width=16, slots=2)
    bridge = WorkspaceBridge(config)
    parent = torch.randn(1, 4, 16)

    projected = bridge(parent, embedding_shell_norm=torch.tensor(3.0))
    projected.sum().backward()

    assert bridge.decoder.receiver_gate.grad is not None
    with torch.no_grad():
        bridge.decoder.receiver_gate.fill_(100.0)
    projected = bridge(parent, embedding_shell_norm=torch.tensor(3.0))
    norms = torch.linalg.vector_norm(projected, dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, 3.0), atol=1e-5)
