from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LocalDepthInputs:
    inputs_embeds: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    labels: Tensor
    latent_span: tuple[int, int]


class LocalDepthCodec(nn.Module):
    """Encode one node's prompt state and decode a local workspace to soft inputs."""

    def __init__(
        self,
        model_dim: int = 2048,
        workspace_dim: int = 256,
        slots: int = 8,
        initial_receiver_gate: float = 0.001,
    ) -> None:
        super().__init__()
        if min(model_dim, workspace_dim, slots) <= 0:
            raise ValueError("codec dimensions and slot count must be positive")
        if not 0.0 < initial_receiver_gate < 1.0:
            raise ValueError("initial_receiver_gate must be between zero and one")

        self.model_dim = model_dim
        self.workspace_dim = workspace_dim
        self.slots = slots
        self.source_norm = nn.LayerNorm(model_dim)
        self.source_projection = nn.Linear(model_dim, workspace_dim)
        self.workspace_norm = nn.LayerNorm(workspace_dim)
        self.receiver_projection = nn.Linear(workspace_dim, model_dim)
        self.receiver_gate = nn.Parameter(torch.tensor(math.atanh(initial_receiver_gate)))

    def encode(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.model_dim:
            raise ValueError("hidden states must have shape [batch, sequence, model_dim]")
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention mask shape does not match hidden states")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)).item():
            raise ValueError("attention mask must be binary")

        selected = []
        detached = hidden_states.detach()
        for batch_index in range(hidden_states.shape[0]):
            indices = torch.nonzero(attention_mask[batch_index].to(dtype=torch.bool), as_tuple=False).flatten()
            if indices.numel() < self.slots:
                raise ValueError(f"each prompt must have at least {self.slots} visible tokens")
            selected.append(detached[batch_index, indices[-self.slots :], :])
        source = torch.stack(selected)
        return self.source_projection(self.source_norm(source))

    def decode(self, workspace: Tensor, embedding_shell_norm: Tensor) -> Tensor:
        if workspace.ndim != 3 or workspace.shape[1:] != (self.slots, self.workspace_dim):
            raise ValueError("workspace shape does not match codec configuration")
        if embedding_shell_norm.numel() != 1:
            raise ValueError("embedding_shell_norm must be scalar")
        if not torch.isfinite(embedding_shell_norm.detach()).item() or embedding_shell_norm.detach().item() <= 0:
            raise ValueError("embedding_shell_norm must be positive and finite")

        projected = self.receiver_projection(self.workspace_norm(workspace))
        projected_norm = torch.linalg.vector_norm(projected, dim=-1, keepdim=True).clamp_min(1e-6)
        shell = projected * (embedding_shell_norm / projected_norm)
        return torch.tanh(self.receiver_gate) * shell


def compose_local_depth_inputs(
    token_embeddings: Tensor,
    attention_mask: Tensor,
    position_ids: Tensor,
    labels: Tensor,
    latent_embeddings: Tensor,
    injection_index: int,
) -> LocalDepthInputs:
    """Insert local latent vectors at an already verified chat-template boundary."""

    if token_embeddings.ndim != 3 or token_embeddings.shape[0] != 1:
        raise ValueError("the Phase B smoke requires one token sequence")
    if attention_mask.shape != token_embeddings.shape[:2]:
        raise ValueError("attention mask shape does not match token embeddings")
    if position_ids.shape != attention_mask.shape or labels.shape != attention_mask.shape:
        raise ValueError("position IDs and labels must match the attention mask")
    if latent_embeddings.ndim != 3 or latent_embeddings.shape[0] != 1:
        raise ValueError("latent embeddings must have shape [1, slots, model_dim]")
    if latent_embeddings.shape[-1] != token_embeddings.shape[-1]:
        raise ValueError("latent and token embedding widths differ")
    if latent_embeddings.device != token_embeddings.device or latent_embeddings.dtype != token_embeddings.dtype:
        raise ValueError("latent and token embeddings must share device and dtype")
    if not 0 <= injection_index <= token_embeddings.shape[1]:
        raise ValueError("injection index is outside the token sequence")

    slots = latent_embeddings.shape[1]
    prefix = slice(0, injection_index)
    suffix = slice(injection_index, None)
    inputs_embeds = torch.cat(
        (token_embeddings[:, prefix, :], latent_embeddings, token_embeddings[:, suffix, :]),
        dim=1,
    )
    latent_mask = torch.ones((1, slots), dtype=attention_mask.dtype, device=attention_mask.device)
    result_mask = torch.cat((attention_mask[:, prefix], latent_mask, attention_mask[:, suffix]), dim=1)

    next_position = (
        position_ids[:, injection_index - 1 : injection_index] + 1
        if injection_index
        else torch.zeros((1, 1), dtype=position_ids.dtype, device=position_ids.device)
    )
    latent_positions = next_position + torch.arange(slots, dtype=position_ids.dtype, device=position_ids.device)[None]
    result_positions = torch.cat(
        (
            position_ids[:, prefix],
            latent_positions,
            position_ids[:, suffix] + slots,
        ),
        dim=1,
    )
    latent_labels = torch.full((1, slots), -100, dtype=labels.dtype, device=labels.device)
    result_labels = torch.cat((labels[:, prefix], latent_labels, labels[:, suffix]), dim=1)
    return LocalDepthInputs(
        inputs_embeds=inputs_embeds,
        attention_mask=result_mask,
        position_ids=result_positions,
        labels=result_labels,
        latent_span=(injection_index, injection_index + slots),
    )
