"""Orchestrator-side algorithm runtime.

The config side (``prime_rl.configs.algorithm``) defines *what* an algorithm
is — a bundle of sampling and the per-token training signal. This package
turns the signal half into runtime objects (the sampling half is the env's
:class:`~prime_rl.orchestrator.sampler.Sampler`):

- one module per algorithm (``grpo``, ``echo``, ``max_rl``, ``opd``,
  ``opsd``, ``sdpo``, ``sft``, ``reward``, ``custom``) — each named class owns its
  scoring hooks (``score_rollout`` / ``score_group`` / ``score_batch``), its
  optional batch preflight hook, and declares what it needs (loss component, a
  "teacher", ...). One instance per env, built by :func:`build_algorithm`.
  Custom credit assignment plugs in through the ``custom`` algorithm type
  (:class:`CustomAlgorithm` imports a user function by path).
- ``base`` — the :class:`Algorithm` base class and the pipeline phase
  functions (:func:`finalize_rollout` / :func:`finalize_group` /
  :func:`finalize_batch_preflight` / :func:`finalize_batch`).
- ``advantage`` — pure advantage math (default group-norm + the
  custom-function interface). Advantages are per-token everywhere they are
  stored or shipped — there is no scalar advantage in the pipeline. A
  function takes ``RolloutView`` objects and returns one value per rollout: a
  scalar that the view *broadcasts* over the rollout's completion tokens
  (uniform credit, the common case), or an explicit per-token list.
- ``routing`` — wire-field stamping: per-token component weight streams
  (rl / ce / ref_kl / sdpo) and the per-token advantage stream.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from prime_rl.orchestrator.algo.advantage import (
    AdvantageFn,
    apply_advantage_fn,
    default_advantage_fn,
    max_rl_advantage_fn,
)
from prime_rl.orchestrator.algo.base import (
    Algorithm,
    connect_frozen_pool,
    finalize_batch,
    finalize_batch_preflight,
    finalize_group,
    finalize_rollout,
)
from prime_rl.orchestrator.algo.custom import CustomAlgorithm
from prime_rl.orchestrator.algo.echo import EchoAlgorithm
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.algo.max_rl import MaxRLAlgorithm
from prime_rl.orchestrator.algo.opd import OPDAlgorithm
from prime_rl.orchestrator.algo.opsd import OPSDAlgorithm
from prime_rl.orchestrator.algo.reward import RewardAlgorithm
from prime_rl.orchestrator.algo.routing import stamp_advantages, stamp_loss_routing
from prime_rl.orchestrator.algo.sdpo import SDPOAlgorithm
from prime_rl.orchestrator.algo.sft import SFTDistillAlgorithm
from prime_rl.orchestrator.types import RolloutView

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.configs.algorithm import AlgorithmConfig
    from prime_rl.utils.client import InferencePool

# Runtime dispatch is keyed on ``algo.type`` — it names the algorithm, and
# each config class's defaults are its vetted parameterization.
ALGORITHM_CLASSES: dict[str, type[Algorithm]] = {
    "grpo": GRPOAlgorithm,
    "echo": EchoAlgorithm,
    "max_rl": MaxRLAlgorithm,
    "opd": OPDAlgorithm,
    "opsd": OPSDAlgorithm,
    "sdpo": SDPOAlgorithm,
    "sft": SFTDistillAlgorithm,
    "reward": RewardAlgorithm,
    "custom": CustomAlgorithm,
}


def build_algorithm(
    config: AlgorithmConfig,
    policy_pool: InferencePool,
    renderer: Renderer | None,
    live_pools: Mapping[str, InferencePool] | None = None,
) -> Algorithm:
    cls = ALGORITHM_CLASSES[config.type]
    if cls.action_loss_type != config.action_loss_type:
        raise ValueError(
            f"Algorithm runtime/config action_loss_type mismatch for {config.type!r}: "
            f"runtime={cls.action_loss_type!r}, config={config.action_loss_type!r}."
        )
    # The Algorithm is the runtime of the algorithm config's training signal
    # (its sibling Sampler interprets the sampling half).
    return cls(config, policy_pool, renderer, live_pools=live_pools)


__all__ = [
    "AdvantageFn",
    "Algorithm",
    "CustomAlgorithm",
    "EchoAlgorithm",
    "GRPOAlgorithm",
    "MaxRLAlgorithm",
    "OPDAlgorithm",
    "OPSDAlgorithm",
    "SDPOAlgorithm",
    "RewardAlgorithm",
    "RolloutView",
    "SFTDistillAlgorithm",
    "apply_advantage_fn",
    "build_algorithm",
    "connect_frozen_pool",
    "default_advantage_fn",
    "finalize_batch",
    "finalize_batch_preflight",
    "finalize_group",
    "finalize_rollout",
    "max_rl_advantage_fn",
    "stamp_advantages",
    "stamp_loss_routing",
]
