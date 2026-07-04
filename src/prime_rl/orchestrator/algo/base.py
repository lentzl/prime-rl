"""The per-env algorithm runtime: base class and pipeline phase functions.

Each named class in this package *is* one training algorithm, one module per
algorithm: it owns the algorithm's scoring hooks directly — ``score_rollout``
(per arrival), ``score_group`` (per group), optional ``run_batch_preflight``,
and ``score_batch`` (per batch) — and declares what it needs
(``action_loss_type``, a ``model_role`` like "teacher"). Reading a module top
to bottom reads the algorithm; writing your own is subclassing
:class:`Algorithm` and overriding the hooks its signal needs. Shared math
(group normalization, prefill alignment) lives as plain functions in
``advantage.py``; duplication of
orchestration between similar algorithms (e.g. OPD and OPSD) is accepted so
each module stays self-contained.

The hooks are one scope-and-timing ladder — each wider scope is
unlocked by a later barrier, so the two axes coincide. Scoring hooks are
``async`` (any stage may do I/O); a hook that only does advantage math never
awaits:

- ``score_rollout(rollout)`` — one rollout, on arrival: rollout-local signals
  (raw reward, process rewards, echo's observation weighting). No siblings.
- ``score_group(group)`` — the cohort, on group completion, *before* filtering
  (filters read the streams): group-relative credit (GRPO/MaxRL baselines).
- ``run_batch_preflight(batch)`` — optional, after filtering and before
  reference scoring: forward/export-only trainer side channels needed by later
  scoring hooks.
- ``score_batch(batch)`` — the batch's survivors, *after* filtering: the home
  for reference I/O (``self.teacher_pool``), where queries are batched for
  concurrency and — running after filtering — dropped rollouts cost nothing.

How rollouts are *produced* is not the algorithm's concern: that is the env's
:class:`~prime_rl.orchestrator.sampler.Sampler`, and sample construction
(interleaving, with observation-token provenance recorded as ``obs_spans``)
is pure pipeline.

The pipeline (dispatcher, train sink, orchestrator) calls the module-level
phase functions (:func:`finalize_rollout`, :func:`finalize_group`,
:func:`finalize_batch_preflight`, :func:`finalize_batch`) and reads the class
declarations; it never branches on algorithm config fields or model roles.
The policy pool is the primary live model view passed into algorithms; frozen
model references are external endpoints the algorithm *connects to* (never
launches) in :meth:`Algorithm.setup`. Additional live-derived views, if
configured by a future algorithm, should attach to the orchestrator's
weight-update lifecycle rather than masquerading as frozen endpoints.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

from prime_rl.configs.algorithm import ActionLossType, AlgorithmConfig, FrozenModelConfig, ModelReference
from prime_rl.orchestrator.algo.routing import stamp_advantages, stamp_loss_routing
from prime_rl.orchestrator.types import RolloutView
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from renderers import RendererConfig
    from renderers.base import Renderer

    from prime_rl.orchestrator.envs import TrainEnvs
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.transport import TrainingSample
    from prime_rl.transport.base import TrainingBatchSender
    from prime_rl.utils.client import InferencePool


async def connect_frozen_pool(
    config: FrozenModelConfig, *, renderer_config: RendererConfig | None = None
) -> InferencePool:
    """Connect a client pool to an inline frozen model and wait for it to be
    ready. The endpoint is externally hosted — prime-rl connects and waits,
    never launches.

    When ``renderer_config`` is set, the pool's train client is the renderer
    (token-in/out) client — required when the frozen model *generates* rollouts
    (sft), so the rollout carries tokens. Left as plain chat-completions
    otherwise (opd/opsd read teacher logprobs via prefill, where the train
    client type is moot)."""
    from prime_rl.utils.client import setup_inference_pool

    get_logger().info(f"Initializing frozen model pool (model={config.name}, base_url={', '.join(config.base_url)})")
    if renderer_config is not None:
        pool = await setup_inference_pool(
            config, model_name=config.name, train_client_type="renderer", renderer_config=renderer_config
        )
    else:
        pool = await setup_inference_pool(config, model_name=config.name)
    await pool.wait_for_ready(config.name)
    return pool


class Algorithm:
    """Base class for one env's training algorithm — the runtime of the
    algorithm config's per-token training signal (its sibling :class:`Sampler`
    interprets the ``sampling`` half).

    Everything on this class is yours to override; the pipeline drives the
    compilation through the module-level phase functions below
    (:func:`finalize_rollout` / :func:`finalize_group` /
    :func:`finalize_batch_preflight` / :func:`finalize_batch`)
    and never calls anything else. The surface is:

    - declarations — which loss component the action tokens feed
      (``action_loss_type``) and what the algorithm calls its reference
      model, if it has one (``model_role``, e.g. "teacher");
    - lifecycle — :meth:`setup` connects client pools to the frozen models
      the algorithm declares, resolving each reference via :meth:`connect`;
    - the scoring hooks, each ``async`` and given a :class:`RolloutView`
      (a writable handle exposing only what is valid at its stage), plus an
      optional batch preflight hook for trainer-side side channels. Scoring
      hooks are async so any stage may do I/O — e.g. a process-reward model at
      arrival, or a judge at group time whose signal a pre-batch filter then
      reads; a hook that only does advantage math simply never awaits.

      - :meth:`score_rollout` — one rollout, on arrival: rollout-local credit
        or observation ce weights. Default: nothing.
      - :meth:`score_group` — the cohort, *before* filtering (filters read the
        streams): group-relative credit. Default: nothing — rollouts keep
        ``advantages=None``, so advantage-based filters skip them.
      - :meth:`run_batch_preflight` — optional, *after* filtering and before
        reference scoring: forward/export-only trainer side channels needed by
        a later scoring hook. Default: nothing.
      - :meth:`score_batch` — the batch's survivors, *after* filtering:
        query the algorithm's reference pool (e.g. ``self.teacher_pool``) and
        attach per-token results, or modulate advantages. Default: nothing.

    ``score_batch`` is the home for reference I/O: it runs after filtering, so
    only survivors cost reference compute. I/O in ``score_rollout`` /
    ``score_group`` runs *before* the pre-batch filters — do it when a filter
    must read the result, accepting that it pays compute on rollouts that may
    then be filtered out.

    Constructed with the algorithm config it interprets plus the two
    host-owned resources: the policy pool and the policy's renderer (the
    canonical messages → token ids path; ``None`` under MITO)."""

    action_loss_type: ClassVar[ActionLossType] = "rl"
    model_role: ClassVar[str | None] = None
    batch_preflight_name: ClassVar[str | None] = None

    def __init__(
        self,
        config: AlgorithmConfig,
        policy_pool: InferencePool,
        renderer: Renderer | None,
        live_pools: Mapping[str, InferencePool] | None = None,
    ):
        self.config = config
        self.policy_pool = policy_pool
        self.renderer = renderer
        self.live_pools = live_pools or {}
        self.connected_pools: list[InferencePool] = []  # client pools connected in setup(); closed at shutdown

    async def setup(self) -> None:
        """Connect client pools to the algorithm's frozen models — override
        and resolve each reference via :meth:`connect`. The base has nothing
        to connect."""

    async def connect(self, reference: ModelReference) -> InferencePool:
        """Resolve a model reference to a client pool: the live policy's own
        pool, or a freshly connected pool to a frozen endpoint. Only the
        latter is tracked in ``connected_pools`` — the host closes what the
        algorithm opened, and nothing else, at shutdown."""
        if reference == "policy":
            return self.policy_pool
        pool = await connect_frozen_pool(reference)
        self.connected_pools.append(pool)
        return pool

    async def connect_live(self, role: str) -> InferencePool:
        """Resolve an internal live-derived model role to its host-owned pool."""
        try:
            return self.live_pools[role]
        except KeyError as exc:
            raise NotImplementedError(f"live inference pool {role!r} is not configured") from exc

    async def score_rollout(self, rollout: RolloutView) -> None:
        """Arrival phase, one rollout, before its group is complete: write
        rollout-local credit (``rollout.assign_advantages``) or observation ce
        weights (echo). No siblings, no group stats."""

    async def score_group(self, group: list[RolloutView]) -> None:
        """Group phase, the finalized cohort, before filtering: write
        group-relative credit."""

    async def score_batch(self, batch: list[RolloutView]) -> None:
        """Ship phase, survivors only, after filtering, async: query the
        algorithm's reference models and attach per-token results, or modulate
        advantages."""

    def needs_batch_preflight(self, batch: list[RolloutView]) -> bool:
        """Whether this algorithm needs a forward/export-only trainer pass
        before batch scoring. Default: no preflight."""
        return False

    def select_batch_preflight_samples(
        self,
        batch: list[RolloutView],
        *,
        samples: list[TrainingSample],
    ) -> list[TrainingSample]:
        """Select this algorithm/env's samples for a shared preflight pass.

        Called per env after ``needs_batch_preflight`` and before shared
        preflight hooks are merged. Algorithms with target-dependent side
        effects, such as SDPO hindsight pruning, should do that work here so
        each env's own algorithm config gates its own rollouts even when the
        eventual preflight transport is shared across envs.
        """
        return samples

    def batch_preflight_signature(self) -> object:
        """Compatibility key for algorithms sharing ``batch_preflight_name``.

        Algorithms with a shared preflight side channel should return the
        settings that affect the exported artifact contract. The pipeline uses
        this to reject envs that would otherwise be merged under one hook name
        but require incompatible trainer-side outputs.
        """
        return None

    async def run_batch_preflight(
        self,
        batch: list[RolloutView],
        *,
        samples: list[TrainingSample],
        output_dir: Path,
        sender: TrainingBatchSender,
        step: int,
    ) -> None:
        """Run the algorithm's optional pre-batch side channel.

        Preflight runs after filtering and before ``score_batch``. Algorithms
        use it when reference scoring needs data only the trainer can produce
        from a live forward pass.
        """


async def finalize_rollout(algorithm: Algorithm, rollout: Rollout) -> None:
    """Arrival phase: rollout-local scoring as each rollout is tokenized."""
    if rollout.samples:
        await algorithm.score_rollout(RolloutView(rollout))


async def finalize_group(algorithm: Algorithm, rollouts: list[Rollout]) -> None:
    """Group phase: group-relative scoring, then stamp each sample's wire
    fields (the advantage stream + loss routing). After this the records are
    frozen — groups die at stamping."""
    await algorithm.score_group([RolloutView(rollout) for rollout in rollouts])
    for rollout in rollouts:
        stamp_advantages(rollout)
        for sample in rollout.samples:
            sample.reward = rollout.reward
            sample.env_name = rollout.env_name
            stamp_loss_routing(sample, algorithm.action_loss_type)


async def finalize_batch(train_envs: TrainEnvs, rollouts: list[Rollout]) -> None:
    """Ship phase: run each env's ``score_batch`` over its unfiltered rollouts
    (survivors), concurrently across envs. Per-env concurrency is bounded by
    the algorithm's own config; envs without references return immediately."""
    by_env: dict[str, list[Rollout]] = defaultdict(list)
    for rollout in rollouts:
        if not rollout.is_filtered:
            by_env[rollout.env_name].append(rollout)
    await asyncio.gather(
        *(
            train_envs.get(env_name).algorithm.score_batch([RolloutView(r) for r in env_rollouts])
            for env_name, env_rollouts in by_env.items()
        )
    )


async def finalize_batch_preflight(
    train_envs: TrainEnvs,
    rollouts: list[Rollout],
    *,
    samples: list[TrainingSample],
    output_dir: Path,
    sender: TrainingBatchSender,
    step: int,
) -> None:
    """Run algorithm-owned preflight hooks before batch reference scoring.

    The pipeline stays algorithm-blind here: it groups survivor rollouts by
    env, asks each env's algorithm whether preflight is needed, and runs one
    hook per logical preflight name. Shared names let multiple env instances of
    the same algorithm cooperate on one step-scoped side channel.
    """
    by_env: dict[str, list[Rollout]] = defaultdict(list)
    for rollout in rollouts:
        if not rollout.is_filtered:
            by_env[rollout.env_name].append(rollout)

    final_sample_ids = {id(sample) for sample in samples}
    preflight_groups: dict[str, tuple[Algorithm, object, list[Rollout], list[TrainingSample]]] = {}
    for env_name, env_rollouts in by_env.items():
        algorithm = train_envs.get(env_name).algorithm
        env_batch = [RolloutView(rollout) for rollout in env_rollouts]
        if not algorithm.needs_batch_preflight(env_batch):
            continue
        env_samples = [
            sample for rollout in env_rollouts for sample in rollout.samples if id(sample) in final_sample_ids
        ]
        selected_samples = algorithm.select_batch_preflight_samples(env_batch, samples=env_samples)
        env_sample_ids = {id(sample) for sample in env_samples}
        if any(id(sample) not in env_sample_ids for sample in selected_samples):
            raise ValueError(
                f"batch preflight selector for env '{env_name}' returned a sample outside its survivor scope"
            )
        selected_sample_ids = [id(sample) for sample in selected_samples]
        if len(selected_sample_ids) != len(set(selected_sample_ids)):
            raise ValueError(f"batch preflight selector for env '{env_name}' returned duplicate samples")
        if not selected_samples:
            continue
        key = algorithm.batch_preflight_name or env_name
        signature = algorithm.batch_preflight_signature()
        if key in preflight_groups:
            existing_algorithm, existing_signature, group_rollouts, group_samples = preflight_groups[key]
            if signature != existing_signature:
                raise ValueError(
                    f"batch preflight hook {key!r} received incompatible algorithm configurations "
                    f"between envs (existing={existing_signature!r}, env '{env_name}'={signature!r})."
                )
        else:
            existing_algorithm = algorithm
            group_rollouts = []
            group_samples = []
            preflight_groups[key] = (existing_algorithm, signature, group_rollouts, group_samples)
        group_rollouts.extend(env_rollouts)
        group_samples.extend(selected_samples)

    if not preflight_groups:
        return

    for algorithm, _signature, group_rollouts, group_samples in preflight_groups.values():
        await algorithm.run_batch_preflight(
            [RolloutView(rollout) for rollout in group_rollouts],
            samples=group_samples,
            output_dir=output_dir,
            sender=sender,
            step=step,
        )
