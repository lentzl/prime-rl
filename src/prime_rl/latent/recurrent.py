from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RecurrentState:
    """State of one node's local recurrent computation.

    ``visible_workspace`` is the only field eligible for a governed workspace
    export. ``private_memory`` is node-local and must never cross a node boundary.
    """

    visible_workspace: Tensor
    private_memory: Tensor

    def detached(self) -> RecurrentState:
        return RecurrentState(
            visible_workspace=self.visible_workspace.detach(),
            private_memory=self.private_memory.detach(),
        )

    def export_workspace(self) -> Tensor:
        return self.visible_workspace


@dataclass(frozen=True)
class RecurrentDiagnostics:
    visible_change_norms: Tensor
    memory_change_norms: Tensor
    visible_contraction_ratios: Tensor
    memory_contraction_ratios: Tensor
    visible_direction_cosines: Tensor
    memory_direction_cosines: Tensor
    nonfinite: bool

    @property
    def visible_oscillation_rate(self) -> float:
        if self.visible_direction_cosines.numel() == 0:
            return 0.0
        return float((self.visible_direction_cosines < 0.0).float().mean().item())

    @property
    def memory_oscillation_rate(self) -> float:
        if self.memory_direction_cosines.numel() == 0:
            return 0.0
        return float((self.memory_direction_cosines < 0.0).float().mean().item())


class TimestepFreeRecurrentSidecar(nn.Module):
    """A bounded recurrent update over a compact coordinator workspace.

    The transition receives no cycle index. Progress must be represented in the
    persistent private memory. The immutable task anchor is reinjected at every
    update, and the visible output remains a bounded residual around that anchor.
    """

    def __init__(self, workspace_dim: int = 256, transition_dim: int = 8192) -> None:
        super().__init__()
        if workspace_dim <= 0:
            raise ValueError("workspace_dim must be positive")
        if transition_dim <= 0:
            raise ValueError("transition_dim must be positive")

        self.workspace_dim = workspace_dim
        self.transition_dim = transition_dim
        self.anchor_norm = nn.LayerNorm(workspace_dim)
        self.visible_norm = nn.LayerNorm(workspace_dim)
        self.memory_norm = nn.LayerNorm(workspace_dim)
        self.transition = nn.Linear(3 * workspace_dim, transition_dim)
        self.memory_candidate = nn.Linear(transition_dim, workspace_dim)
        self.memory_gate = nn.Linear(transition_dim, workspace_dim)
        self.workspace_delta = nn.Linear(workspace_dim, workspace_dim)
        self.output_scale = nn.Parameter(torch.zeros(workspace_dim))

    def initial_state(self, task_anchor: Tensor) -> RecurrentState:
        self._validate_workspace(task_anchor, "task_anchor")
        return RecurrentState(
            visible_workspace=task_anchor,
            private_memory=torch.zeros_like(task_anchor),
        )

    def step(self, task_anchor: Tensor, state: RecurrentState) -> RecurrentState:
        self._validate_workspace(task_anchor, "task_anchor")
        self._validate_workspace(state.visible_workspace, "state.visible_workspace")
        self._validate_workspace(state.private_memory, "state.private_memory")
        if task_anchor.shape != state.visible_workspace.shape or task_anchor.shape != state.private_memory.shape:
            raise ValueError("task anchor, visible workspace, and private memory must have identical shapes")

        transition_input = torch.cat(
            (
                self.anchor_norm(task_anchor),
                self.visible_norm(state.visible_workspace),
                self.memory_norm(state.private_memory),
            ),
            dim=-1,
        )
        features = torch.nn.functional.silu(self.transition(transition_input))
        candidate = torch.tanh(self.memory_candidate(features))
        retain = torch.sigmoid(self.memory_gate(features))
        private_memory = retain * state.private_memory + (1.0 - retain) * candidate

        delta = torch.tanh(self.workspace_delta(self.memory_norm(private_memory)))
        visible_workspace = task_anchor + torch.tanh(self.output_scale) * delta
        return RecurrentState(visible_workspace=visible_workspace, private_memory=private_memory)

    def rollout(
        self,
        task_anchor: Tensor,
        steps: int,
        *,
        state: RecurrentState | None = None,
        truncate_every: int | None = None,
        return_trajectory: bool = False,
    ) -> RecurrentState | tuple[RecurrentState, ...]:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if truncate_every is not None and truncate_every <= 0:
            raise ValueError("truncate_every must be positive when set")

        current = self.initial_state(task_anchor) if state is None else state
        trajectory = [current]
        for step_index in range(steps):
            current = self.step(task_anchor, current)
            trajectory.append(current)
            if truncate_every is not None and (step_index + 1) % truncate_every == 0 and step_index + 1 < steps:
                current = current.detached()
                trajectory[-1] = current
        if return_trajectory:
            return tuple(trajectory)
        return current

    def _validate_workspace(self, value: Tensor, name: str) -> None:
        _validate_workspace(value, name, self.workspace_dim)


class OneShotFeedForwardSidecar(nn.Module):
    """Matched-parameter, non-recurrent control for the local-depth claim."""

    def __init__(self, workspace_dim: int = 256, hidden_dim: int = 20587) -> None:
        super().__init__()
        if workspace_dim <= 0:
            raise ValueError("workspace_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.workspace_dim = workspace_dim
        self.input_norm = nn.LayerNorm(workspace_dim)
        self.hidden = nn.Linear(workspace_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, workspace_dim)
        self.output_scale = nn.Parameter(torch.zeros(workspace_dim))

    def forward(self, task_anchor: Tensor) -> Tensor:
        _validate_workspace(task_anchor, "task_anchor", self.workspace_dim)
        features = torch.nn.functional.silu(self.hidden(self.input_norm(task_anchor)))
        delta = torch.tanh(self.output(features))
        return task_anchor + torch.tanh(self.output_scale) * delta


def diagnose_recurrent_states(states: tuple[RecurrentState, ...], epsilon: float = 1e-8) -> RecurrentDiagnostics:
    """Measure contraction and alternating-step oscillation without task labels."""

    if len(states) < 2:
        raise ValueError("at least two recurrent states are required")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    if any(state.visible_workspace.shape != states[0].visible_workspace.shape for state in states):
        raise ValueError("visible workspace shapes must remain constant")
    if any(state.private_memory.shape != states[0].private_memory.shape for state in states):
        raise ValueError("private memory shapes must remain constant")
    visible = torch.stack([state.visible_workspace.detach().float() for state in states])
    memory = torch.stack([state.private_memory.detach().float() for state in states])

    visible_changes = visible[1:] - visible[:-1]
    memory_changes = memory[1:] - memory[:-1]
    visible_norms = _batch_rms_norm(visible_changes)
    memory_norms = _batch_rms_norm(memory_changes)
    return RecurrentDiagnostics(
        visible_change_norms=visible_norms,
        memory_change_norms=memory_norms,
        visible_contraction_ratios=_contraction_ratios(visible_norms, epsilon),
        memory_contraction_ratios=_contraction_ratios(memory_norms, epsilon),
        visible_direction_cosines=_direction_cosines(visible_changes, epsilon),
        memory_direction_cosines=_direction_cosines(memory_changes, epsilon),
        nonfinite=not bool(torch.isfinite(visible).all() and torch.isfinite(memory).all()),
    )


def _batch_rms_norm(changes: Tensor) -> Tensor:
    return changes.flatten(start_dim=2).square().mean(dim=-1).sqrt()


def _contraction_ratios(norms: Tensor, epsilon: float) -> Tensor:
    if norms.shape[0] < 2:
        return norms.new_empty((0, *norms.shape[1:]))
    return norms[1:] / norms[:-1].clamp_min(epsilon)


def _direction_cosines(changes: Tensor, epsilon: float) -> Tensor:
    if changes.shape[0] < 2:
        return changes.new_empty((0, changes.shape[1]))
    previous = changes[:-1].flatten(start_dim=2)
    current = changes[1:].flatten(start_dim=2)
    numerator = (previous * current).sum(dim=-1)
    denominator = previous.norm(dim=-1) * current.norm(dim=-1)
    return numerator / denominator.clamp_min(epsilon)


def _validate_workspace(value: Tensor, name: str, workspace_dim: int) -> None:
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, slots, workspace_dim]")
    if value.shape[-1] != workspace_dim:
        raise ValueError(f"{name} has width {value.shape[-1]}, expected {workspace_dim}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
