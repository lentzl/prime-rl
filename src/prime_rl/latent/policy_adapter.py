from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True, slots=True)
class HiddenStateCaptureSpec:
    schema_version: str = "prime-rl/hidden-state-capture/v1"
    layer: int = -1
    max_non_padding_tokens: int = 128
    boundary: str = "accepted_delegation"
    detach: bool = True

    def validate(self) -> None:
        if self.schema_version != "prime-rl/hidden-state-capture/v1":
            raise ValueError("unknown hidden-state capture schema")
        if self.layer != -1:
            raise ValueError("A0/A1 permits final-layer capture only")
        if not 1 <= self.max_non_padding_tokens <= 128:
            raise ValueError("capture span must contain between 1 and 128 tokens")
        if self.boundary != "accepted_delegation":
            raise ValueError("capture must occur at an accepted delegation boundary")
        if not self.detach:
            raise ValueError("parent hidden states must be detached in A0/A1")

    def checksum(self) -> str:
        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CapturedFeatures:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    token_indices: torch.Tensor
    capture_spec_hash: str


def capture_parent_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    spec: HiddenStateCaptureSpec,
) -> CapturedFeatures:
    spec.validate()
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("A0/A1 feature capture requires one rank-3 sequence")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention mask shape does not match hidden states")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)).item():
        raise ValueError("attention mask must be binary")
    token_indices = torch.nonzero(attention_mask[0].to(dtype=torch.bool), as_tuple=False).flatten()
    if token_indices.numel() == 0:
        raise ValueError("cannot capture an empty parent sequence")
    token_indices = token_indices[-spec.max_non_padding_tokens :]
    selected = hidden_states[:, token_indices, :]
    captured = torch.zeros(
        (1, spec.max_non_padding_tokens, hidden_states.shape[-1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    captured_mask = torch.zeros(
        (1, spec.max_non_padding_tokens),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    padded_indices = torch.full((spec.max_non_padding_tokens,), -1, dtype=torch.long)
    captured[:, -selected.shape[1] :, :] = selected
    captured_mask[:, -selected.shape[1] :] = 1
    padded_indices[-token_indices.shape[0] :] = token_indices.detach().to(device="cpu")
    if spec.detach:
        captured = captured.detach()
    return CapturedFeatures(
        hidden_states=captured,
        attention_mask=captured_mask,
        token_indices=padded_indices,
        capture_spec_hash=spec.checksum(),
    )


@dataclass(frozen=True, slots=True)
class ReceiverEmbeddingBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor | None
    labels: torch.Tensor | None
    workspace_span: tuple[int, int] | None


def _is_hard_zero_gate(gate: float | torch.Tensor) -> bool:
    if isinstance(gate, torch.Tensor):
        return not gate.requires_grad and torch.count_nonzero(gate).item() == 0
    return gate == 0.0


def compose_receiver_inputs(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    workspace_embeddings: torch.Tensor,
    *,
    injection_index: int,
    gate: float | torch.Tensor,
    position_ids: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> ReceiverEmbeddingBatch:
    """Insert soft workspace tokens at the verified assistant-opening boundary.

    An exact, non-trainable zero gate is the standard/no-feedback bypass and does
    not materialize latent positions. A trainable gate, including one initialized
    to zero, keeps the positions so gradients can reach the gate.
    """

    if token_embeddings.ndim != 3 or token_embeddings.shape[0] != 1:
        raise ValueError("A0/A1 receiver integration requires one embedding sequence")
    if attention_mask.shape != token_embeddings.shape[:2]:
        raise ValueError("attention mask shape does not match token embeddings")
    if workspace_embeddings.ndim == 2:
        workspace_embeddings = workspace_embeddings.unsqueeze(0)
    if workspace_embeddings.ndim != 3 or workspace_embeddings.shape[0] != 1:
        raise ValueError("workspace embeddings must have shape [K, D] or [1, K, D]")
    if workspace_embeddings.shape[2] != token_embeddings.shape[2]:
        raise ValueError("workspace and token embedding dimensions differ")
    if workspace_embeddings.device != token_embeddings.device:
        raise ValueError("workspace and token embeddings must share a device")
    if workspace_embeddings.dtype != token_embeddings.dtype:
        raise ValueError("workspace and token embeddings must share a dtype")
    sequence_length = token_embeddings.shape[1]
    if not 0 <= injection_index <= sequence_length:
        raise ValueError("injection index is outside the receiver sequence")
    if position_ids is not None and position_ids.shape != attention_mask.shape:
        raise ValueError("position IDs shape does not match attention mask")
    if labels is not None and labels.shape != attention_mask.shape:
        raise ValueError("labels shape does not match attention mask")
    if isinstance(gate, torch.Tensor):
        if gate.numel() != 1:
            raise ValueError("receiver gate must be scalar")
        if gate.device != token_embeddings.device:
            raise ValueError("receiver gate and embeddings must share a device")
        if not torch.isfinite(gate.detach()).item() or not 0.0 <= gate.detach().item() <= 1.0:
            raise ValueError("receiver gate must be between zero and one")
    elif not 0.0 <= gate <= 1.0:
        raise ValueError("receiver gate must be between zero and one")

    if _is_hard_zero_gate(gate):
        return ReceiverEmbeddingBatch(
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            workspace_span=None,
        )

    gated_workspace = workspace_embeddings * gate
    prefix = token_embeddings[:, :injection_index, :]
    suffix = token_embeddings[:, injection_index:, :]
    inputs_embeds = torch.cat((prefix, gated_workspace, suffix), dim=1)
    slot_count = workspace_embeddings.shape[1]
    latent_mask = torch.ones((1, slot_count), dtype=attention_mask.dtype, device=attention_mask.device)
    result_mask = torch.cat(
        (attention_mask[:, :injection_index], latent_mask, attention_mask[:, injection_index:]),
        dim=1,
    )

    result_positions = None
    if position_ids is not None:
        prefix_positions = position_ids[:, :injection_index]
        if injection_index:
            next_position = prefix_positions[:, -1:] + 1
        else:
            next_position = torch.zeros((1, 1), dtype=position_ids.dtype, device=position_ids.device)
        latent_offsets = torch.arange(slot_count, dtype=position_ids.dtype, device=position_ids.device)[None]
        latent_positions = next_position + latent_offsets
        suffix_positions = position_ids[:, injection_index:] + slot_count
        result_positions = torch.cat((prefix_positions, latent_positions, suffix_positions), dim=1)

    result_labels = None
    if labels is not None:
        latent_labels = torch.full((1, slot_count), -100, dtype=labels.dtype, device=labels.device)
        result_labels = torch.cat((labels[:, :injection_index], latent_labels, labels[:, injection_index:]), dim=1)

    return ReceiverEmbeddingBatch(
        inputs_embeds=inputs_embeds,
        attention_mask=result_mask,
        position_ids=result_positions,
        labels=result_labels,
        workspace_span=(injection_index, injection_index + slot_count),
    )
