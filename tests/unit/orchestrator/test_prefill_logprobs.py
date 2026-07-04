import asyncio
import json
from types import SimpleNamespace

import httpx
import openai

from prime_rl.configs.algorithm import SDPO_MAX_STUDENT_SUPPORT_TOPK
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


def test_prefill_candidate_limit_matches_sdpo_student_support_config_limit():
    assert orchestrator_utils.MAX_PREFILL_CANDIDATE_TOKEN_IDS == SDPO_MAX_STUDENT_SUPPORT_TOPK


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


def test_compute_prefill_logprobs_rejects_nonfinite_logprob(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [None, {"2": {"logprob": float("nan")}}],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
            )
        except ValueError as exc:
            assert "prefill logprobs returned non-finite or non-floating logprob for token id 2" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    asyncio.run(_run())


def test_compute_prefill_logprobs_rejects_integer_logprob(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [None, {"2": {"logprob": -1}}],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
            )
        except ValueError as exc:
            assert "prefill logprobs returned non-finite or non-floating logprob for token id 2" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    asyncio.run(_run())


def test_compute_prefill_logprobs_rejects_non_integer_token_ids_before_request(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient({"request_id": "unused", "choices": [], "prompt_logprobs": []})
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, True],
            )
        except ValueError as exc:
            assert "prefill token_ids must contain integers" in str(exc)
        else:
            raise AssertionError("Expected ValueError")
        assert fake_client.calls == []

    asyncio.run(_run())


def test_compute_prefill_logprobs_rejects_non_list_token_ids_before_request(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient({"request_id": "unused", "choices": [], "prompt_logprobs": []})
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=(1, 2),
            )
        except ValueError as exc:
            assert "prefill token_ids must be a list" in str(exc)
        else:
            raise AssertionError("Expected ValueError")
        assert fake_client.calls == []

    asyncio.run(_run())


def test_compute_prefill_topk_logprobs_rejects_negative_token_ids_before_request(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient({"request_id": "unused", "choices": [], "prompt_logprobs": []})
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_topk_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, -2],
                topk=2,
            )
        except ValueError as exc:
            assert "prefill token_ids must contain non-negative ids" in str(exc)
        else:
            raise AssertionError("Expected ValueError")
        assert fake_client.calls == []

    asyncio.run(_run())


def test_compute_prefill_topk_logprobs_rejects_non_list_token_ids_before_request(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient({"request_id": "unused", "choices": [], "prompt_logprobs": []})
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_topk_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=(1, 2),
                topk=2,
            )
        except ValueError as exc:
            assert "prefill token_ids must be a list" in str(exc)
        else:
            raise AssertionError("Expected ValueError")
        assert fake_client.calls == []

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


def test_compute_prefill_logprobs_rejects_null_target_logprob(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [None, {"2": {"logprob": None}}],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
            )
        except ValueError as exc:
            assert "prefill logprobs missing logprob for token id 2 at position 1" in str(exc)
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


def test_compute_prefill_logprobs_rejects_malformed_response_entry(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [None, ["not", "a", "mapping"]],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
            )
        except ValueError as exc:
            assert "prefill logprobs returned malformed entry at position 1" in str(exc)
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


def test_compute_prefill_topk_logprobs_sorts_and_requests_topk(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [
                    None,
                    {"11": {"logprob": -0.2}, "12": {"logprob": -0.1}, "13": {"logprob": -0.4}},
                    {"21": {"logprob": -0.5}, "22": {"logprob": -0.3}, "23": {"logprob": -0.7}},
                ],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        token_ids, logprobs = await orchestrator_utils.compute_prefill_topk_logprobs(
            SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
            model_name="ref-model",
            token_ids=[1, 2, 3],
            topk=2,
        )

        assert token_ids == [[0, 0], [12, 11], [22, 21]]
        assert logprobs == [[0.0, 0.0], [-0.1, -0.2], [-0.3, -0.5]]
        assert fake_client.calls[0]["body"]["sampling_params"]["prompt_logprobs"] == 2

    asyncio.run(_run())


def test_compute_prefill_topk_logprobs_rejects_short_rows(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [None, {"11": {"logprob": -0.2}}],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_topk_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                topk=2,
            )
        except ValueError as exc:
            assert "returned 1 entries" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    asyncio.run(_run())


def test_compute_prefill_topk_logprobs_rejects_non_integer_topk_before_request(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient({"request_id": "unused", "choices": [], "prompt_logprobs": []})
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_topk_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                topk=True,
            )
        except ValueError as exc:
            assert "requires integer topk" in str(exc)
        else:
            raise AssertionError("Expected ValueError")
        assert fake_client.calls == []

    asyncio.run(_run())


def test_compute_prefill_topk_logprobs_rejects_non_positive_topk_before_request(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient({"request_id": "unused", "choices": [], "prompt_logprobs": []})
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_topk_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                topk=0,
            )
        except ValueError as exc:
            assert "requires topk > 0" in str(exc)
        else:
            raise AssertionError("Expected ValueError")
        assert fake_client.calls == []

    asyncio.run(_run())


def test_compute_prefill_topk_logprobs_rejects_null_logprob(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [
                    None,
                    {"11": {"logprob": -0.2}, "12": {"logprob": None}, "13": {"logprob": -0.4}},
                ],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_topk_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                topk=2,
            )
        except ValueError as exc:
            assert "prefill top-k logprobs missing logprob for token id 12 at position 1" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    asyncio.run(_run())


def test_compute_prefill_topk_logprobs_rejects_nonfinite_logprob(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [
                    None,
                    {"11": {"logprob": -0.2}, "12": {"logprob": float("inf")}},
                ],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_topk_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                topk=2,
            )
        except ValueError as exc:
            assert "prefill top-k logprobs returned non-finite or non-floating logprob for token id 12" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    asyncio.run(_run())


def test_compute_prefill_topk_logprobs_rejects_integer_logprob(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [
                    None,
                    {"11": {"logprob": -1}, "12": {"logprob": -0.1}},
                ],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_topk_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                topk=2,
            )
        except ValueError as exc:
            assert "prefill top-k logprobs returned non-finite or non-floating logprob for token id 11" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    asyncio.run(_run())


def test_compute_prefill_topk_logprobs_rejects_malformed_token_id(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [
                    None,
                    {"bad-token": {"logprob": -0.2}, "12": {"logprob": -0.1}},
                ],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_topk_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                topk=2,
            )
        except ValueError as exc:
            assert "prefill top-k logprobs returned malformed token id 'bad-token' at position 1" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    asyncio.run(_run())


def test_compute_prefill_topk_logprobs_rejects_malformed_logprob_entry(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [
                    None,
                    {"11": ["not", "a", "logprob"], "12": {"logprob": -0.1}},
                ],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_topk_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                topk=2,
            )
        except ValueError as exc:
            assert "prefill top-k logprobs returned malformed logprob entry for token id 11" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    asyncio.run(_run())


def test_compute_prefill_candidate_logprobs_requests_specific_token_ids(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [
                    None,
                    {"12": {"logprob": -0.1}, "11": {"logprob": -0.2}, "13": {"logprob": -0.4}},
                    {"22": {"logprob": -0.3}, "21": {"logprob": -0.5}, "13": {"logprob": -0.9}},
                ],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        logprobs = await orchestrator_utils.compute_prefill_candidate_logprobs(
            SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
            model_name="ref-model",
            token_ids=[1, 2, 3],
            candidate_token_ids=[[], [12, 11], [21, 22]],
        )

        assert logprobs == [[], [-0.1, -0.2], [-0.5, -0.3]]
        assert fake_client.calls[0]["body"]["sampling_params"] == {
            "max_tokens": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "prompt_logprobs": 4,
            "logprob_token_ids": [11, 12, 21, 22],
        }

    asyncio.run(_run())


def test_compute_prefill_candidate_logprobs_rejects_nonfinite_logprob(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [None, {"12": {"logprob": False}, "11": {"logprob": -0.2}}],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_candidate_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                candidate_token_ids=[[], [12, 11]],
            )
        except ValueError as exc:
            assert "prefill candidate logprobs returned non-finite or non-floating logprob for token id 12" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    asyncio.run(_run())


def test_compute_prefill_candidate_logprobs_rejects_integer_logprob(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [None, {"12": {"logprob": -1}, "11": {"logprob": -0.2}}],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_candidate_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                candidate_token_ids=[[], [12, 11]],
            )
        except ValueError as exc:
            assert "prefill candidate logprobs returned non-finite or non-floating logprob for token id 12" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    asyncio.run(_run())


def test_compute_prefill_candidate_logprobs_rejects_malformed_response_entry(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [None, ["not", "a", "mapping"]],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_candidate_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                candidate_token_ids=[[], [12]],
            )
        except ValueError as exc:
            assert "prefill candidate logprobs returned malformed entry at position 1" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    asyncio.run(_run())


def test_compute_prefill_candidate_logprobs_rejects_contextless_first_position_candidates(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient({"request_id": "unused", "choices": [], "prompt_logprobs": []})
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_candidate_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                candidate_token_ids=[[11], [12]],
            )
        except ValueError as exc:
            assert "cannot score candidate ids at position 0" in str(exc)
        else:
            raise AssertionError("Expected ValueError")
        assert fake_client.calls == []

    asyncio.run(_run())


def test_compute_prefill_candidate_logprobs_rejects_non_integer_candidates_before_request(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient({"request_id": "unused", "choices": [], "prompt_logprobs": []})
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_candidate_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                candidate_token_ids=[[], [12, True]],
            )
        except ValueError as exc:
            assert "candidate token ids must be integers" in str(exc)
        else:
            raise AssertionError("Expected ValueError")
        assert fake_client.calls == []

    asyncio.run(_run())


def test_compute_prefill_candidate_logprobs_rejects_non_list_candidate_container_before_request(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient({"request_id": "unused", "choices": [], "prompt_logprobs": []})
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_candidate_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                candidate_token_ids=([], [12]),
            )
        except ValueError as exc:
            assert "prefill candidate token ids must be a list of rows" in str(exc)
        else:
            raise AssertionError("Expected ValueError")
        assert fake_client.calls == []

    asyncio.run(_run())


def test_compute_prefill_candidate_logprobs_rejects_non_list_candidate_rows_before_request(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient({"request_id": "unused", "choices": [], "prompt_logprobs": []})
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_candidate_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                candidate_token_ids=[[], (12, 13)],
            )
        except ValueError as exc:
            assert "prefill candidate token ids rows must be lists" in str(exc)
        else:
            raise AssertionError("Expected ValueError")
        assert fake_client.calls == []

    asyncio.run(_run())


def test_compute_prefill_candidate_logprobs_rejects_negative_candidates_before_request(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient({"request_id": "unused", "choices": [], "prompt_logprobs": []})
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_candidate_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                candidate_token_ids=[[], [12, -1]],
            )
        except ValueError as exc:
            assert "candidate token ids must be non-negative" in str(exc)
        else:
            raise AssertionError("Expected ValueError")
        assert fake_client.calls == []

    asyncio.run(_run())


def test_compute_prefill_candidate_logprobs_rejects_null_candidate_logprob(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [None, {"12": {"logprob": None}, "11": {"logprob": -0.2}}],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_candidate_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                candidate_token_ids=[[], [12, 11]],
            )
        except ValueError as exc:
            assert "prefill candidate logprobs missing logprob for token id 12 at position 1" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    asyncio.run(_run())


def test_compute_prefill_candidate_logprobs_preserves_duplicate_candidate_rows(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [
                    None,
                    {"11": {"logprob": -0.2}, "12": {"logprob": -0.4}},
                ],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        logprobs = await orchestrator_utils.compute_prefill_candidate_logprobs(
            SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
            model_name="ref-model",
            token_ids=[1, 2],
            candidate_token_ids=[[], [11, 11, 12]],
        )

        assert logprobs == [[], [-0.2, -0.2, -0.4]]
        assert fake_client.calls[0]["body"]["sampling_params"]["logprob_token_ids"] == [11, 12]

    asyncio.run(_run())


def test_compute_prefill_candidate_logprobs_rejects_missing_candidate(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                "prompt_logprobs": [None, {"11": {"logprob": -0.2}}],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_candidate_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                candidate_token_ids=[[], [11, 12]],
            )
        except ValueError as exc:
            assert "missing token id 12 at position 1" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    asyncio.run(_run())


def test_compute_prefill_candidate_logprobs_rejects_too_many_unique_candidates(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient({"request_id": "unused", "choices": [], "prompt_logprobs": []})
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)

        try:
            await orchestrator_utils.compute_prefill_candidate_logprobs(
                SimpleNamespace(base_url="http://fake-host:8000/v1", api_key_var="VLLM_API_KEY", headers={}),
                model_name="ref-model",
                token_ids=[1, 2],
                candidate_token_ids=[[], list(range(129))],
            )
        except ValueError as exc:
            assert "at most 128" in str(exc)
        else:
            raise AssertionError("Expected ValueError")
        assert fake_client.calls == []

    asyncio.run(_run())
