import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from renderers import Qwen3VLRendererConfig

from prime_rl.orchestrator.orchestrator import Orchestrator
from prime_rl.orchestrator.utils import setup_policy_inference_pool


def test_setup_policy_inference_pool_uses_renderer_when_enabled():
    async def run() -> None:
        tokenizer = object()
        renderer_settings = Qwen3VLRendererConfig()
        config = SimpleNamespace(
            model=SimpleNamespace(
                client=SimpleNamespace(base_url=["http://localhost:8000/v1"]),
                name="policy-model",
            ),
            renderer=renderer_settings,
            pool_size=None,
            any_policy_sourced=True,
        )
        renderer = object()
        inference_pool = object()

        with (
            patch("renderers.base.create_renderer", return_value=renderer) as create_renderer_mock,
            patch(
                "prime_rl.orchestrator.utils.setup_inference_pool",
                new=AsyncMock(return_value=inference_pool),
            ) as setup_pool_mock,
        ):
            returned_renderer, returned_pool = await setup_policy_inference_pool(
                config=config,
                tokenizer=tokenizer,
            )

        assert returned_renderer is renderer
        assert returned_pool is inference_pool
        create_renderer_mock.assert_called_once_with(tokenizer, renderer_settings)
        setup_pool_mock.assert_awaited_once_with(
            config.model.client,
            model_name="policy-model",
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_config=renderer_settings,
            pool_size=None,
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
                client=SimpleNamespace(base_url=["http://localhost:8000/v1"]),
                name="policy-model",
            ),
            renderer=renderer_settings,
            pool_size=None,
            any_policy_sourced=False,
        )
        renderer = object()
        inference_pool = object()

        with (
            patch("renderers.base.create_renderer", return_value=renderer) as create_renderer_mock,
            patch(
                "prime_rl.orchestrator.utils.setup_inference_pool",
                new=AsyncMock(return_value=inference_pool),
            ) as setup_pool_mock,
        ):
            returned_renderer, returned_pool = await setup_policy_inference_pool(
                config=config,
                tokenizer=tokenizer,
            )

        assert returned_renderer is renderer
        assert returned_pool is inference_pool
        create_renderer_mock.assert_called_once_with(tokenizer, renderer_settings)
        setup_pool_mock.assert_awaited_once_with(
            config.model.client,
            model_name="policy-model",
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_config=renderer_settings,
            pool_size=None,
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


def test_empty_train_batch_reopens_synchronous_dispatch_budget():
    async def run() -> None:
        dispatcher = SimpleNamespace(reset_train_rollout_budget=Mock())
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.config = SimpleNamespace(max_steps=None)
        orchestrator.progress = SimpleNamespace(step=99)
        orchestrator.dispatcher = dispatcher
        orchestrator.consecutive_empty_batches = 0
        orchestrator.last_batch_at = None
        batch = SimpleNamespace(samples=[], rollouts=[object()] * 8)

        await orchestrator.finalize_train_batch(batch)

        dispatcher.reset_train_rollout_budget.assert_called_once_with()
        assert orchestrator.consecutive_empty_batches == 1

    asyncio.run(run())
