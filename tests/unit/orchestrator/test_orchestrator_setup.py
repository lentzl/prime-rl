import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from renderers import Qwen3VLRendererConfig

from prime_rl.orchestrator.orchestrator import Orchestrator
from prime_rl.orchestrator.types import TrainBatch
from prime_rl.orchestrator.utils import setup_policy_inference_pool


def test_setup_policy_inference_pool_uses_renderer_when_enabled():
    async def run() -> None:
        tokenizer = object()
        renderer_settings = Qwen3VLRendererConfig()
        config = SimpleNamespace(
            model=SimpleNamespace(
                client=SimpleNamespace(base_url="http://localhost:8000/v1"),
                name="policy-model",
            ),
            renderer=renderer_settings,
            any_policy_sourced=True,
        )
        renderer = object()
        inference_pool = object()

        with (
            patch("renderers.base.create_renderer", return_value=renderer) as create_renderer_mock,
            patch(
                "prime_rl.orchestrator.utils.InferencePool",
                new=MagicMock(return_value=inference_pool),
            ) as setup_pool_mock,
        ):
            returned_renderer, returned_pool = await setup_policy_inference_pool(
                config=config,
                tokenizer=tokenizer,
            )

        assert returned_renderer is renderer
        assert returned_pool is inference_pool
        create_renderer_mock.assert_called_once_with(tokenizer, renderer_settings)
        setup_pool_mock.assert_called_once_with(
            config.model.client,
            model_name="policy-model",
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_config=renderer_settings,
        )

    asyncio.run(run())


def test_setup_policy_inference_pool_keeps_renderer_without_policy_sampling():
    """Frozen-sourced runs (e.g. sft) have no train env sampling from the live
    policy, but training is renderer-only: the renderer is still built and the
    pool is wired with the renderer train client. ``any_policy_sourced`` only
    flips the log line, not the pool setup."""

    async def run() -> None:
        tokenizer = object()
        renderer_settings = Qwen3VLRendererConfig()
        config = SimpleNamespace(
            model=SimpleNamespace(
                client=SimpleNamespace(base_url="http://localhost:8000/v1"),
                name="policy-model",
            ),
            renderer=renderer_settings,
            any_policy_sourced=False,
        )
        renderer = object()
        inference_pool = object()

        with (
            patch("renderers.base.create_renderer", return_value=renderer) as create_renderer_mock,
            patch(
                "prime_rl.orchestrator.utils.InferencePool",
                new=MagicMock(return_value=inference_pool),
            ) as setup_pool_mock,
        ):
            returned_renderer, returned_pool = await setup_policy_inference_pool(
                config=config,
                tokenizer=tokenizer,
            )

        assert returned_renderer is renderer
        assert returned_pool is inference_pool
        create_renderer_mock.assert_called_once_with(tokenizer, renderer_settings)
        setup_pool_mock.assert_called_once_with(
            config.model.client,
            model_name="policy-model",
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_config=renderer_settings,
        )

    asyncio.run(run())


def test_zero_train_batch_lead_waits_for_updated_final_batch_policy():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.config = SimpleNamespace(
        max_train_batch_lead=0,
        max_steps=2,
        weight_broadcast=SimpleNamespace(type="nccl"),
    )
    orchestrator.progress = SimpleNamespace(step=2)
    orchestrator.policy = SimpleNamespace(version=0)
    orchestrator.dispatcher = SimpleNamespace(dispatch_allowed=asyncio.Event())
    orchestrator.dispatcher.dispatch_allowed.set()
    orchestrator.gate_closed_at = None
    orchestrator.wait_for_policy_time = 0.0

    orchestrator.update_dispatch_gate()

    assert not orchestrator.dispatcher.dispatch_allowed.is_set()

    orchestrator.policy.version = 1
    orchestrator.update_dispatch_gate()

    assert orchestrator.dispatcher.dispatch_allowed.is_set()


def test_begin_draining_immediately_disables_train_scheduling():
    async def run() -> None:
        dispatcher = SimpleNamespace(
            disable_train_scheduling=Mock(),
            cancel_inflight_train_rollouts=AsyncMock(return_value=3),
        )
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.draining = False
        orchestrator.dispatcher = dispatcher

        await orchestrator.begin_draining()

        assert orchestrator.draining
        dispatcher.disable_train_scheduling.assert_called_once_with()
        dispatcher.cancel_inflight_train_rollouts.assert_awaited_once_with()

    asyncio.run(run())


def test_exhausted_synchronous_cohort_flushes_partial_batch():
    async def run() -> None:
        batch = TrainBatch(rollouts=MagicMock(), samples=[MagicMock()])
        train_sink = SimpleNamespace(
            has_pending_observations=True,
            batch_progress=Mock(return_value=(11, 12, "rollouts")),
            flush_partial_batch=Mock(return_value=batch),
        )
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.dispatcher = SimpleNamespace(
            train_rollouts_per_policy=12,
            train_rollouts_scheduled=12,
            inflight_train_count=0,
            groups={},
            out_q=asyncio.Queue(),
        )
        orchestrator.train_sink = train_sink
        orchestrator.finalize_train_batch = AsyncMock()
        orchestrator.draining = False
        orchestrator.stopped = asyncio.Event()

        assert await orchestrator._maybe_flush_exhausted_sync_cohort()
        train_sink.flush_partial_batch.assert_called_once_with()
        orchestrator.finalize_train_batch.assert_awaited_once_with(batch)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("scheduled", "inflight", "groups", "queued"),
    [
        (11, 0, {}, False),
        (12, 1, {}, False),
        (12, 0, {"active": SimpleNamespace(kind="train")}, False),
        (12, 0, {}, True),
    ],
)
def test_active_synchronous_cohort_is_not_flushed(scheduled, inflight, groups, queued):
    async def run() -> None:
        out_q = asyncio.Queue()
        if queued:
            out_q.put_nowait([MagicMock()])
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.dispatcher = SimpleNamespace(
            train_rollouts_per_policy=12,
            train_rollouts_scheduled=scheduled,
            inflight_train_count=inflight,
            groups=groups,
            out_q=out_q,
        )
        orchestrator.train_sink = SimpleNamespace(has_pending_observations=True)
        orchestrator.draining = False
        orchestrator.stopped = asyncio.Event()

        assert not await orchestrator._maybe_flush_exhausted_sync_cohort()

    asyncio.run(run())


def test_empty_exhausted_synchronous_cohort_fails_explicitly():
    async def run() -> None:
        batch = TrainBatch(rollouts=MagicMock(), samples=[])
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.dispatcher = SimpleNamespace(
            train_rollouts_per_policy=12,
            train_rollouts_scheduled=12,
            inflight_train_count=0,
            groups={},
            out_q=asyncio.Queue(),
        )
        orchestrator.train_sink = SimpleNamespace(
            has_pending_observations=True,
            batch_progress=Mock(return_value=(0, 12, "rollouts")),
            flush_partial_batch=Mock(return_value=batch),
        )
        orchestrator.draining = False
        orchestrator.stopped = asyncio.Event()

        with pytest.raises(RuntimeError, match="without a trainable sample"):
            await orchestrator._maybe_flush_exhausted_sync_cohort()

    asyncio.run(run())


def test_draining_orchestrator_does_not_flush_partial_batch():
    async def run() -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.dispatcher = SimpleNamespace(train_rollouts_per_policy=12)
        orchestrator.train_sink = SimpleNamespace(has_pending_observations=True)
        orchestrator.draining = True
        orchestrator.stopped = asyncio.Event()

        assert not await orchestrator._maybe_flush_exhausted_sync_cohort()

    asyncio.run(run())
