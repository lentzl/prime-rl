from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from prime_rl.configs.trainer import ModelObserverConfig


def project_model_states(hidden_states: Tensor, feature_dim: int) -> Tensor:
    """Deterministically sketch final-layer states without learned parameters."""
    flat = hidden_states.detach().float().reshape(-1, hidden_states.shape[-1])
    chunk_count = math.ceil(flat.shape[-1] / feature_dim)
    padded_width = feature_dim * chunk_count
    if padded_width != flat.shape[-1]:
        flat = F.pad(flat, (0, padded_width - flat.shape[-1]))
    chunks = flat.reshape(flat.shape[0], feature_dim, chunk_count)
    signs = torch.ones(chunk_count, dtype=chunks.dtype, device=chunks.device)
    signs[1::2] = -1.0
    projected = (chunks * signs).sum(dim=-1) / math.sqrt(chunk_count)
    return F.normalize(projected, dim=-1)


class RidgeEpiplexityObserver:
    """Ridge readout with spectral program length for a correction vector."""

    def __init__(
        self,
        config: ModelObserverConfig,
        correction_dim: int,
        *,
        device: torch.device | str = "cpu",
    ):
        if correction_dim < 1:
            raise ValueError("model observer correction_dim must be positive")
        self.config = config
        self.correction_dim = correction_dim
        self.device = torch.device(device)
        self.gram = torch.eye(config.feature_dim, dtype=torch.float64, device=self.device) * config.ridge_lambda
        self.cross = torch.zeros(
            (config.feature_dim, correction_dim), dtype=torch.float64, device=self.device
        )
        self.observation_count = 0

    @property
    def readout(self) -> Tensor:
        return torch.linalg.solve(self.gram, self.cross)

    @property
    def score_bits(self) -> Tensor:
        return self._program_length(self.readout)

    def _program_length(self, readout: Tensor) -> Tensor:
        readout_gram = readout.transpose(-2, -1) @ readout
        identity = torch.eye(
            self.correction_dim,
            dtype=readout.dtype,
            device=readout.device,
        )
        _, logdet = torch.linalg.slogdet(identity + self.config.code_resolution * readout_gram)
        return 0.5 * logdet / math.log(2.0)

    @torch.no_grad()
    def hypothetical_novelty(self, features: Tensor, corrections: Tensor) -> Tensor:
        if features.numel() == 0:
            return torch.empty(0, dtype=torch.float32, device=features.device)
        features64 = features.to(self.device, torch.float64)
        corrections64 = corrections.to(self.device, torch.float64)
        self._validate_observations(features64, corrections64)
        inverse_gram = torch.linalg.inv(self.gram)
        readout = inverse_gram @ self.cross
        p_features = features64 @ inverse_gram
        gain = p_features / (1.0 + (p_features * features64).sum(dim=-1, keepdim=True))
        innovation = corrections64 - features64 @ readout
        candidate_readouts = readout.unsqueeze(0) + gain.unsqueeze(-1) * innovation.unsqueeze(-2)
        candidate_scores = self._program_length(candidate_readouts)
        return (candidate_scores - self.score_bits).to(features.device, torch.float32)

    @torch.no_grad()
    def update(self, features: Tensor, corrections: Tensor, *, group=None) -> None:
        features64 = features.to(self.device, torch.float64)
        corrections64 = corrections.to(self.device, torch.float64)
        self._validate_observations(features64, corrections64)
        gram_update = features64.T @ features64
        cross_update = features64.T @ corrections64
        count = torch.tensor(features64.shape[0], dtype=torch.int64, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(gram_update, group=group)
            dist.all_reduce(cross_update, group=group)
            dist.all_reduce(count, group=group)
        self.gram += gram_update
        self.cross += cross_update
        self.observation_count += int(count.item())

    def _validate_observations(self, features: Tensor, corrections: Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.config.feature_dim:
            raise ValueError("model observer features do not match configured feature_dim")
        if corrections.shape != (features.shape[0], self.correction_dim):
            raise ValueError("model observer corrections must align with features and correction_dim")

    def state_dict(self) -> dict:
        return {
            "gram": self.gram.cpu(),
            "cross": self.cross.cpu(),
            "observation_count": self.observation_count,
        }

    def load_state_dict(self, state: dict) -> None:
        gram = state["gram"].to(self.device, torch.float64)
        cross = state["cross"].to(self.device, torch.float64)
        expected = self.config.feature_dim
        if gram.shape != (expected, expected) or cross.shape != (expected, self.correction_dim):
            raise ValueError("model observer state does not match configured dimensions")
        self.gram = gram
        self.cross = cross
        self.observation_count = int(state["observation_count"])


class ModelObserverBank:
    """One synchronized persistent observer per environment."""

    def __init__(self, config: ModelObserverConfig, *, device: torch.device | str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.observers: dict[str, RidgeEpiplexityObserver] = {}
        self.pending_updates: dict[str, tuple[Tensor, Tensor, Tensor]] = {}

    @torch.no_grad()
    def score_and_accumulate(
        self,
        hidden_states: Tensor,
        corrections: Tensor,
        novelty_weights: Tensor,
        env_names: list[str],
        *,
        group=None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        features = project_model_states(hidden_states, self.config.feature_dim)
        corrections_flat = corrections.detach().reshape(-1, corrections.shape[-1])
        weights_flat = novelty_weights.detach().reshape(-1)
        if len(env_names) != features.shape[0]:
            raise ValueError("model observer env_names must align with token features")
        if corrections_flat.shape[0] != features.shape[0]:
            raise ValueError("model observer corrections must align with token features")

        local_envs = sorted({name for name, weight in zip(env_names, weights_flat.tolist()) if weight != 0.0})
        all_envs = local_envs
        if dist.is_initialized():
            gathered: list[list[str] | None] = [None] * dist.get_world_size(group)
            dist.all_gather_object(gathered, local_envs, group=group)
            all_envs = sorted({name for names in gathered if names is not None for name in names})

        advantages = torch.zeros_like(weights_flat)
        raw_values: list[Tensor] = []
        correction_norms: list[Tensor] = []
        positive_values: list[Tensor] = []
        clipped_values: list[Tensor] = []
        for env_name in all_envs:
            observer = self.observers.get(env_name)
            if observer is None:
                observer = RidgeEpiplexityObserver(self.config, corrections_flat.shape[-1], device=self.device)
                self.observers[env_name] = observer
            if observer.correction_dim != corrections_flat.shape[-1]:
                raise ValueError("model observer correction dimension changed within a run")
            mask = torch.tensor([name == env_name for name in env_names], dtype=torch.bool, device=features.device) & (
                weights_flat != 0
            )
            env_features = features[mask]
            env_corrections = corrections_flat[mask] / self.config.correction_scale
            if self.config.shuffle_corrections:
                env_corrections = self._shuffle_corrections(env_name, observer, env_corrections)
            raw = observer.hypothetical_novelty(env_features, env_corrections)
            normalized = self._normalize(raw, group=group)
            positive = normalized > 0
            if self.config.positive_only:
                normalized = normalized.clamp(min=0.0)
            clipped = normalized.abs() > self.config.advantage_clip
            normalized = normalized.clamp(-self.config.advantage_clip, self.config.advantage_clip)
            advantages[mask] = normalized
            raw_values.append(raw)
            correction_norms.append(env_corrections.norm(dim=-1).to(features.device, torch.float32))
            positive_values.append(positive.to(features.device, torch.float32))
            clipped_values.append(clipped.to(features.device, torch.float32))
            self._accumulate(env_name, observer, env_features, env_corrections)

        metrics = {
            "model_observer/raw_novelty": torch.cat(raw_values)
            if raw_values
            else torch.empty(0, device=features.device),
            "model_observer/advantage": advantages[weights_flat != 0],
            "model_observer/correction_norm": torch.cat(correction_norms)
            if correction_norms
            else torch.empty(0, device=features.device),
            "model_observer/positive": torch.cat(positive_values)
            if positive_values
            else torch.empty(0, device=features.device),
            "model_observer/clipped": torch.cat(clipped_values)
            if clipped_values
            else torch.empty(0, device=features.device),
        }
        return advantages.reshape_as(novelty_weights), metrics

    @torch.no_grad()
    def commit(self, *, group=None) -> None:
        """Synchronize and apply one batch of observer sufficient statistics."""
        local_envs = sorted(self.pending_updates)
        all_envs = local_envs
        if dist.is_initialized():
            gathered: list[list[str] | None] = [None] * dist.get_world_size(group)
            dist.all_gather_object(gathered, local_envs, group=group)
            all_envs = sorted({name for names in gathered if names is not None for name in names})

        for env_name in all_envs:
            observer = self.observers.get(env_name)
            if observer is None:
                raise RuntimeError("model observer state diverged across distributed ranks")
            update = self.pending_updates.get(env_name)
            if update is None:
                gram_update = torch.zeros_like(observer.gram)
                cross_update = torch.zeros_like(observer.cross)
                count = torch.zeros((), dtype=torch.int64, device=self.device)
            else:
                gram_update, cross_update, count = update
            if dist.is_initialized():
                dist.all_reduce(gram_update, group=group)
                dist.all_reduce(cross_update, group=group)
                dist.all_reduce(count, group=group)
            observer.gram += gram_update
            observer.cross += cross_update
            observer.observation_count += int(count.item())
        self.pending_updates.clear()

    @torch.no_grad()
    def state_metrics(self) -> dict[str, Tensor]:
        observers = list(self.observers.values())
        return {
            "model_observer/score_bits": torch.stack([observer.score_bits.float() for observer in observers]),
            "model_observer/observation_count": torch.tensor(
                [observer.observation_count for observer in observers], dtype=torch.float32, device=self.device
            ),
            "model_observer/readout_norm": torch.stack(
                [observer.readout.float().norm() for observer in observers]
            ),
        }

    def _shuffle_corrections(
        self,
        env_name: str,
        observer: RidgeEpiplexityObserver,
        corrections: Tensor,
    ) -> Tensor:
        if corrections.shape[0] < 2:
            return corrections
        pending = self.pending_updates.get(env_name)
        pending_count = int(pending[2].item()) if pending is not None else 0
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.shuffle_seed + observer.observation_count + pending_count)
        permutation = torch.randperm(corrections.shape[0], generator=generator, device="cpu").to(corrections.device)
        return corrections[permutation]

    def _accumulate(
        self,
        env_name: str,
        observer: RidgeEpiplexityObserver,
        features: Tensor,
        corrections: Tensor,
    ) -> None:
        features64 = features.to(self.device, torch.float64)
        corrections64 = corrections.to(self.device, torch.float64)
        observer._validate_observations(features64, corrections64)
        gram_update = features64.T @ features64
        cross_update = features64.T @ corrections64
        count = torch.tensor(features64.shape[0], dtype=torch.int64, device=self.device)
        if env_name in self.pending_updates:
            previous_gram, previous_cross, previous_count = self.pending_updates[env_name]
            gram_update += previous_gram
            cross_update += previous_cross
            count += previous_count
        self.pending_updates[env_name] = (gram_update, cross_update, count)

    def save(self, path: Path) -> None:
        if self.pending_updates:
            raise RuntimeError("cannot checkpoint model observer before committing its batch update")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({name: observer.state_dict() for name, observer in self.observers.items()}, temporary)
        temporary.replace(path)

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"model observer checkpoint not found at {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        for name, observer_state in state.items():
            cross = observer_state.get("cross")
            if not isinstance(cross, Tensor) or cross.ndim != 2:
                raise ValueError("model observer checkpoint has an invalid correction readout")
            observer = RidgeEpiplexityObserver(self.config, cross.shape[1], device=self.device)
            observer.load_state_dict(observer_state)
            self.observers[name] = observer

    @staticmethod
    def _normalize(values: Tensor, *, group=None) -> Tensor:
        sum_square = values.square().sum()
        count = torch.tensor(values.numel(), dtype=torch.int64, device=values.device)
        if dist.is_initialized():
            dist.all_reduce(sum_square, group=group)
            dist.all_reduce(count, group=group)
        rms = torch.sqrt(sum_square / count.clamp(min=1)).clamp(min=1e-8)
        return values / rms
