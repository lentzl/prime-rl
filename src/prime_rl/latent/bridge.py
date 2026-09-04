from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class WorkspaceBridgeConfig:
    schema_version: str = "prime-rl/workspace-bridge/v1"
    source_width: int = 2048
    workspace_width: int = 256
    receiver_width: int = 2048
    slots: int = 8
    attention_heads: int = 8
    initial_receiver_gate: float = 0.001

    def validate(self) -> None:
        if self.schema_version != "prime-rl/workspace-bridge/v1":
            raise ValueError("unknown workspace bridge schema")
        if not 1 <= self.slots <= 64:
            raise ValueError("workspace slot count is outside v1 bounds")
        if self.workspace_width % self.attention_heads:
            raise ValueError("workspace width must be divisible by attention heads")
        if min(self.source_width, self.workspace_width, self.receiver_width) <= 0:
            raise ValueError("bridge dimensions must be positive")
        if not 0.0 <= self.initial_receiver_gate < 1.0:
            raise ValueError("initial receiver gate must be in [0, 1)")

    def checksum(self) -> str:
        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


class LearnedQueryWorkspaceEncoder(nn.Module):
    def __init__(self, config: WorkspaceBridgeConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.source_norm = nn.LayerNorm(config.source_width)
        self.source_projection = nn.Linear(config.source_width, config.workspace_width)
        self.learned_queries = nn.Parameter(torch.empty(config.slots, config.workspace_width))
        self.resampler = nn.MultiheadAttention(
            config.workspace_width,
            config.attention_heads,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(config.workspace_width)
        nn.init.normal_(self.learned_queries, std=config.workspace_width**-0.5)

    def forward(
        self,
        parent_hidden_states: torch.Tensor,
        parent_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if parent_hidden_states.ndim != 3:
            raise ValueError("parent hidden states must have shape [B, T, D]")
        if parent_hidden_states.shape[-1] != self.config.source_width:
            raise ValueError("parent hidden width does not match bridge configuration")
        key_padding_mask = None
        if parent_attention_mask is not None:
            if parent_attention_mask.shape != parent_hidden_states.shape[:2]:
                raise ValueError("parent attention mask shape is invalid")
            if not torch.all((parent_attention_mask == 0) | (parent_attention_mask == 1)).item():
                raise ValueError("parent attention mask must be binary")
            if not parent_attention_mask.to(dtype=torch.bool).any(dim=1).all().item():
                raise ValueError("every parent sequence must contain at least one visible token")
            key_padding_mask = ~parent_attention_mask.to(dtype=torch.bool)
        source = self.source_projection(self.source_norm(parent_hidden_states))
        queries = self.learned_queries.unsqueeze(0).expand(parent_hidden_states.shape[0], -1, -1)
        workspace, _ = self.resampler(
            queries,
            source,
            source,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.output_norm(workspace)


class WorkspaceDecoder(nn.Module):
    def __init__(self, config: WorkspaceBridgeConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.workspace_norm = nn.LayerNorm(config.workspace_width)
        self.projection = nn.Linear(config.workspace_width, config.receiver_width)
        self.receiver_gate = nn.Parameter(torch.tensor(math.atanh(config.initial_receiver_gate)))

    def forward(self, workspace: torch.Tensor, *, embedding_shell_norm: torch.Tensor) -> torch.Tensor:
        if workspace.ndim != 3 or workspace.shape[1:] != (
            self.config.slots,
            self.config.workspace_width,
        ):
            raise ValueError("workspace shape does not match bridge configuration")
        if (
            embedding_shell_norm.numel() != 1
            or not torch.isfinite(embedding_shell_norm).item()
            or embedding_shell_norm.item() <= 0
        ):
            raise ValueError("embedding shell norm must be a positive finite scalar")
        projected = self.projection(self.workspace_norm(workspace))
        projected_norm = torch.linalg.vector_norm(projected, dim=-1, keepdim=True).clamp_min(1e-6)
        shell = projected * (embedding_shell_norm / projected_norm)
        return torch.tanh(self.receiver_gate) * shell


class WorkspaceBridge(nn.Module):
    def __init__(self, config: WorkspaceBridgeConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = LearnedQueryWorkspaceEncoder(config)
        self.decoder = WorkspaceDecoder(config)

    def forward(
        self,
        parent_hidden_states: torch.Tensor,
        *,
        embedding_shell_norm: torch.Tensor,
        parent_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        workspace = self.encoder(parent_hidden_states, parent_attention_mask)
        return self.decoder(workspace, embedding_shell_norm=embedding_shell_norm)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
