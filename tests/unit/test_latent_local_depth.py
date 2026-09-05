import torch

from prime_rl.latent.local_depth import LocalDepthCodec, compose_local_depth_inputs


def test_local_depth_codec_detaches_frozen_features_and_matches_embedding_shell():
    torch.manual_seed(17)
    codec = LocalDepthCodec(model_dim=8, workspace_dim=4, slots=2)
    hidden = torch.randn(1, 4, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1]])

    workspace = codec.encode(hidden, mask)
    latent = codec.decode(workspace, torch.tensor(3.0))
    latent.square().sum().backward()

    expected_norm = 3.0 * torch.tanh(codec.receiver_gate.detach()).abs()
    assert hidden.grad is None
    assert workspace.shape == (1, 2, 4)
    assert torch.allclose(torch.linalg.vector_norm(latent.detach(), dim=-1), expected_norm.expand(1, 2))
    assert codec.source_projection.weight.grad is not None


def test_compose_local_depth_inputs_inserts_masked_loss_positions_and_shifts_positions():
    tokens = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    mask = torch.ones((1, 3), dtype=torch.long)
    positions = torch.tensor([[0, 1, 2]])
    labels = torch.tensor([[10, 11, 12]])
    latent = torch.full((1, 2, 8), -1.0)

    result = compose_local_depth_inputs(tokens, mask, positions, labels, latent, injection_index=2)

    assert result.inputs_embeds.shape == (1, 5, 8)
    assert torch.equal(result.inputs_embeds[:, 2:4], latent)
    assert result.attention_mask.tolist() == [[1, 1, 1, 1, 1]]
    assert result.position_ids.tolist() == [[0, 1, 2, 3, 4]]
    assert result.labels.tolist() == [[10, 11, -100, -100, 12]]
    assert result.latent_span == (2, 4)
