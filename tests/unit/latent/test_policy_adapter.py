import torch

from prime_rl.latent.policy_adapter import (
    HiddenStateCaptureSpec,
    capture_parent_features,
    compose_receiver_inputs,
)


def test_gate_zero_is_exact_embedding_path_equivalence() -> None:
    token_embeddings = torch.randn(1, 6, 8)
    attention_mask = torch.ones(1, 6, dtype=torch.long)
    position_ids = torch.arange(6).unsqueeze(0)
    labels = torch.arange(6).unsqueeze(0)
    workspace = torch.randn(3, 8)

    result = compose_receiver_inputs(
        token_embeddings,
        attention_mask,
        workspace,
        injection_index=4,
        gate=0.0,
        position_ids=position_ids,
        labels=labels,
    )

    assert result.inputs_embeds is token_embeddings
    assert result.attention_mask is attention_mask
    assert result.position_ids is position_ids
    assert result.labels is labels
    assert result.workspace_span is None
    projection = torch.randn(8, 5)
    assert torch.equal(result.inputs_embeds @ projection, token_embeddings @ projection)


def test_workspace_insertion_updates_masks_positions_and_labels_only_at_boundary() -> None:
    token_embeddings = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    attention_mask = torch.ones(1, 6, dtype=torch.long)
    position_ids = torch.arange(6).unsqueeze(0)
    labels = torch.arange(6).unsqueeze(0)
    workspace = torch.full((2, 4), 10.0)

    result = compose_receiver_inputs(
        token_embeddings,
        attention_mask,
        workspace,
        injection_index=4,
        gate=0.5,
        position_ids=position_ids,
        labels=labels,
    )

    assert result.workspace_span == (4, 6)
    assert torch.equal(result.inputs_embeds[:, :4], token_embeddings[:, :4])
    assert torch.equal(result.inputs_embeds[:, 4:6], torch.full((1, 2, 4), 5.0))
    assert torch.equal(result.inputs_embeds[:, 6:], token_embeddings[:, 4:])
    assert result.attention_mask.tolist() == [[1] * 8]
    assert result.position_ids.tolist() == [list(range(8))]
    assert result.labels.tolist() == [[0, 1, 2, 3, -100, -100, 4, 5]]


def test_trainable_zero_gate_keeps_slots_and_receives_gradient() -> None:
    tokens = torch.ones(1, 3, 4)
    workspace = torch.ones(2, 4)
    gate = torch.tensor(0.0, requires_grad=True)

    result = compose_receiver_inputs(
        tokens,
        torch.ones(1, 3, dtype=torch.long),
        workspace,
        injection_index=2,
        gate=gate,
    )
    result.inputs_embeds.sum().backward()

    assert result.workspace_span == (2, 4)
    assert gate.grad is not None
    assert gate.grad.item() == 8.0


def test_capture_selects_last_nonpadding_states_and_detaches() -> None:
    hidden = torch.arange(40, dtype=torch.float32).reshape(1, 5, 8).requires_grad_()
    mask = torch.tensor([[0, 1, 1, 1, 0]])
    spec = HiddenStateCaptureSpec(max_non_padding_tokens=2)

    captured = capture_parent_features(hidden, mask, spec)

    assert captured.token_indices.tolist() == [2, 3]
    assert captured.attention_mask.tolist() == [[1, 1]]
    assert torch.equal(captured.hidden_states, hidden[:, 2:4].detach())
    assert captured.hidden_states.requires_grad is False
    assert len(captured.capture_spec_hash) == 64


def test_capture_pads_to_fixed_compute_shape() -> None:
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    captured = capture_parent_features(
        hidden,
        torch.tensor([[0, 0, 1]]),
        HiddenStateCaptureSpec(max_non_padding_tokens=3),
    )

    assert captured.hidden_states.shape == (1, 3, 8)
    assert captured.attention_mask.tolist() == [[0, 0, 1]]
    assert captured.token_indices.tolist() == [-1, -1, 2]
    assert torch.equal(captured.hidden_states[:, -1], hidden[:, -1])
