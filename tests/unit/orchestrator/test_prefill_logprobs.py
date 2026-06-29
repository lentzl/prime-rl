import asyncio
import json
from types import SimpleNamespace

import httpx
import openai

from prime_rl.orchestrator import utils as orchestrator_utils


class _FakeOpenAIClient:
    """Stand-in for ``AsyncOpenAI`` that captures the sole ``.post()`` call and
    returns a synthesized ``httpx.Response`` so ``cast_to=httpx.Response`` is
    handed back verbatim, mirroring the real SDK's short-circuit at
    ``AsyncAPIClient._process_response``."""

    def __init__(self, payload: dict, status_code: int = 200):
        # Match what AsyncOpenAI exposes — utils.py reads ``str(client.base_url)``.
        self.base_url = "http://fake-host:8000/v1"
        self._payload = payload
        self._status_code = status_code
        self.calls: list[dict] = []

    async def post(self, url, *, cast_to, body):
        self.calls.append({"url": url, "cast_to": cast_to, "body": body})
        request = httpx.Request("POST", url, json=body)
        return httpx.Response(
            status_code=self._status_code,
            content=json.dumps(self._payload).encode(),
            request=request,
        )


def test_compute_prefill_logprobs_uses_inference_generate(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                # Upstream wire shape: list[dict[token_id, Logprob] | None]
                "prompt_logprobs": [
                    None,
                    {"11": {"logprob": -0.1}, "2": {"logprob": -0.7}},
                    {"12": {"logprob": -0.2}, "3": {"logprob": -0.3}},
                ],
                "kv_transfer_params": None,
            }
        )
        # compute_prefill_logprobs builds AsyncOpenAI directly from the v1
        # ClientConfig (base_url / api_key_var / headers) via a call-time
        # ``from openai import AsyncOpenAI``; patch that name so the fake is
        # handed back.
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        result = await orchestrator_utils.compute_prefill_logprobs(
            SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
            model_name="ref-model",
            token_ids=[1, 2, 3],
        )

        assert result == [0.0, -0.7, -0.3]
        assert fake_client.calls == [
            {
                "url": "http://fake-host:8000/inference/v1/generate",
                "cast_to": httpx.Response,
                "body": {
                    "model": "ref-model",
                    "token_ids": [1, 2, 3],
                    "sampling_params": {
                        "max_tokens": 1,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "prompt_logprobs": 1,
                    },
                },
            }
        ]

    asyncio.run(_run())


def test_compute_prefill_logprobs_rejects_wrong_length(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [None, {"2": {"logprob": -0.7}}],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2, 3],
            )
        except ValueError as exc:
            assert "prefill logprobs length != token length" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    asyncio.run(_run())


def test_compute_prefill_logprobs_rejects_missing_token_id(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [None, {"11": {"logprob": -0.1}}, {"3": {"logprob": -0.3}}],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2, 3],
            )
        except ValueError as exc:
            assert "prefill logprobs missing token id 2" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    asyncio.run(_run())


def test_compute_prefill_logprobs_rejects_missing_non_leading_entry(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [None, None, {"3": {"logprob": -0.3}}],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2, 3],
            )
        except ValueError as exc:
            assert "prefill logprobs missing entry at position 1 for token id 2" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    asyncio.run(_run())


def test_compute_prefill_logprobs_raises_for_http_error(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient({"error": {"message": "bad key"}}, status_code=401)
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2, 3],
            )
        except httpx.HTTPStatusError as exc:
            assert exc.response.status_code == 401
        else:
            raise AssertionError("Expected HTTPStatusError")

    asyncio.run(_run())
