#!/usr/bin/env python3
"""Route Prime Agent coordinator and child requests to separate OpenAI endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout, web

PRIVATE_EVIDENCE_HEADER = "[private evidence supplied to this reviewer]"
CHILD_ACTION_SCAFFOLD_HEADER = "[training-only child action scaffold]"
EXACT_ACTION_MARKER = "[interaction-curriculum exact action]"
ROOT_ACTION_PATTERN = re.compile(
    r"In the root coordinator's first IPython call, execute this code exactly:\s*"
    r"```python\s*\n(?P<code>.*?)\n```",
    re.DOTALL,
)
CHILD_ACTION_PATTERN = re.compile(
    re.escape(CHILD_ACTION_SCAFFOLD_HEADER)
    + r".*?In your first IPython call execute exactly:\s*"
    + r"```python\s*\n(?P<code>.*?)\n```",
    re.DOTALL,
)
CHILD_SEND_PATTERN = re.compile(
    r"await agent_message\.send\('[0-9]+', receiver_role='parent'\)"
)
HOP_BY_HOP = {
    "connection",
    "content-encoding",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _contains_private_evidence(value: Any) -> bool:
    if isinstance(value, str):
        return PRIVATE_EVIDENCE_HEADER in value
    if isinstance(value, list):
        return any(_contains_private_evidence(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_private_evidence(item) for item in value.values())
    return False


def _contains_token_subsequence(token_ids: list[int], marker_ids: list[int]) -> bool:
    if not marker_ids or len(marker_ids) > len(token_ids):
        return False
    width = len(marker_ids)
    return any(token_ids[start : start + width] == marker_ids for start in range(len(token_ids) - width + 1))


def request_role(payload: dict[str, Any], *, private_evidence_token_ids: list[int] | None = None) -> str:
    messages = payload.get("messages")
    if isinstance(messages, list):
        return "child" if _contains_private_evidence(messages) else "coordinator"
    token_ids = payload.get("token_ids")
    if isinstance(token_ids, list) and all(isinstance(item, int) for item in token_ids):
        marker_ids = private_evidence_token_ids or []
        return "child" if _contains_token_subsequence(token_ids, marker_ids) else "coordinator"
    raise ValueError("role-routed request lacks messages or token_ids")


def routed_payload(
    payload: dict[str, Any],
    *,
    coordinator_model: str,
    child_model: str,
    private_evidence_token_ids: list[int] | None = None,
) -> tuple[str, dict[str, Any]]:
    role = request_role(payload, private_evidence_token_ids=private_evidence_token_ids)
    model = child_model if role == "child" else coordinator_model
    return role, {**payload, "model": model}


def without_tool_choice_constraints(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Remove chat-only tool constraints from a token-in generation request.

    The renderer already places the tool schema in the prompt. Passing named
    ``tool_choice`` through raw vLLM sampling currently makes Qwen 3.5 emit the
    deterministic malformed call ``<function=>``, which has no useful policy
    gradient. Role bootstrap samples the native prompted distribution instead.
    """

    sampling_params = payload.get("sampling_params")
    if not isinstance(sampling_params, dict):
        return payload, ()
    cleaned = dict(sampling_params)
    removed = tuple(
        key
        for key in ("tool_choice", "parallel_tool_calls")
        if cleaned.pop(key, None) is not None
    )
    if not removed:
        return payload, ()
    return {**payload, "sampling_params": cleaned}, removed


def should_strip_tool_choice(
    role: str,
    *,
    strip_child: bool,
    strip_coordinator: bool,
) -> bool:
    return (role == "child" and strip_child) or (
        role == "coordinator" and strip_coordinator
    )


def disclosed_root_action(prompt: str) -> str | None:
    """Extract the public training-only coordinator action from a rendered prompt."""

    if EXACT_ACTION_MARKER not in prompt:
        return None
    matches = [match.group("code").strip() for match in ROOT_ACTION_PATTERN.finditer(prompt)]
    if not matches:
        return None
    if len(set(matches)) != 1:
        raise ValueError("rendered coordinator prompt contains conflicting exact actions")
    code = matches[0]
    if not code.startswith("reviewer = await rlm(") or "name=" not in code:
        raise ValueError("disclosed coordinator action is not the expected retained spawn")
    return code


def disclosed_child_action(prompt: str) -> str | None:
    """Extract a hidden training-only child send action from its private context."""

    if CHILD_ACTION_SCAFFOLD_HEADER not in prompt:
        return None
    matches = [match.group("code").strip() for match in CHILD_ACTION_PATTERN.finditer(prompt)]
    if not matches:
        return None
    if len(set(matches)) != 1:
        raise ValueError("rendered child prompt contains conflicting exact actions")
    code = matches[0]
    if CHILD_SEND_PATTERN.fullmatch(code) is None:
        raise ValueError("disclosed child action is not the expected typed parent send")
    return code


def _encoded_ipython_completion(tokenizer: Any, code: str) -> tuple[list[int], str]:
    if "</parameter>" in code or "</function>" in code or "</tool_call>" in code:
        raise ValueError("disclosed action contains a tool-call delimiter")
    completion = (
        "<tool_call><function=ipython><parameter=code>\n"
        f"{code}\n"
        "</parameter></function></tool_call>"
    )
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    stop_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(stop_id, int) or stop_id < 0:
        raise RuntimeError("Qwen tokenizer lacks the <|im_end|> stop token")
    return [*completion_ids, stop_id], hashlib.sha256(code.encode()).hexdigest()


def exact_ipython_completion_ids(
    tokenizer: Any, token_ids: list[int]
) -> tuple[list[int], str] | None:
    """Build the native Qwen tool completion for a disclosed coordinator action.

    This is an explicit early-curriculum environment leak. Child updates mask it via
    ``sampled_session_scope=non_root``; coordinator updates deliberately use it as a
    biased first-action scaffold so later protocol states become reachable.
    """

    prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
    code = disclosed_root_action(prompt)
    if code is None:
        return None
    return _encoded_ipython_completion(tokenizer, code)


def exact_child_ipython_completion_ids(
    tokenizer: Any, token_ids: list[int]
) -> tuple[list[int], str] | None:
    """Build the native Qwen tool completion for a private child send scaffold."""

    prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
    code = disclosed_child_action(prompt)
    if code is None:
        return None
    return _encoded_ipython_completion(tokenizer, code)


def synthetic_generate_response(completion_ids: list[int], *, sequence: int) -> bytes:
    payload = {
        "request_id": f"role-router-leak-{sequence}",
        "choices": [
            {
                "index": 0,
                "token_ids": completion_ids,
                # The renderer client requires finite, cardinality-matched evidence. These
                # placeholder values also make scaffolded coordinator updates explicitly
                # bootstrap-biased rather than strict policy-distribution samples.
                "logprobs": {
                    "content": [
                        {"token": f"token_id:{token_id}", "logprob": -0.1}
                        for token_id in completion_ids
                    ]
                },
                "finish_reason": "stop",
            }
        ],
    }
    return json.dumps(payload, separators=(",", ":")).encode()


class DualPolicyProxy:
    def __init__(
        self,
        *,
        coordinator_url: str,
        coordinator_model: str,
        child_url: str,
        child_model: str,
        external_model: str,
        audit_log: Path,
        private_evidence_token_ids: list[int],
        tokenizer: Any,
        leak_coordinator_exact_action: bool = False,
        leak_child_exact_action: bool = False,
        strip_child_tool_choice: bool = False,
        strip_coordinator_tool_choice: bool = False,
    ) -> None:
        self.urls = {
            "coordinator": coordinator_url.rstrip("/"),
            "child": child_url.rstrip("/"),
        }
        self.models = {"coordinator": coordinator_model, "child": child_model}
        self.external_model = external_model
        self.audit_log = audit_log
        self.private_evidence_token_ids = private_evidence_token_ids
        self.tokenizer = tokenizer
        self.leak_coordinator_exact_action = leak_coordinator_exact_action
        self.leak_child_exact_action = leak_child_exact_action
        self.strip_child_tool_choice = strip_child_tool_choice
        self.strip_coordinator_tool_choice = strip_coordinator_tool_choice
        self.client: ClientSession | None = None
        self.sequence = 0
        self.leaked_session_hashes: dict[str, set[str]] = {
            "coordinator": set(),
            "child": set(),
        }

    async def startup(self, _: web.Application) -> None:
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)
        if self.audit_log.exists():
            with self.audit_log.open(encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            self.sequence = len(events)
            for event in events:
                role = event.get("role")
                session_sha256 = event.get("session_sha256")
                if (
                    role in self.leaked_session_hashes
                    and event.get("mode") == f"leaked_exact_{role}_action"
                    and isinstance(session_sha256, str)
                ):
                    self.leaked_session_hashes[role].add(session_sha256)
        self.client = ClientSession(timeout=ClientTimeout(total=None, connect=30))

    async def cleanup(self, _: web.Application) -> None:
        if self.client is not None:
            await self.client.close()

    async def health(self, _: web.Request) -> web.Response:
        if self.client is None:
            raise RuntimeError("proxy client is not initialized")
        statuses = {}
        for role, url in self.urls.items():
            try:
                async with self.client.get(f"{url.removesuffix('/v1')}/health") as response:
                    statuses[role] = response.status
            except ClientError:
                statuses[role] = 0
        healthy = all(status == 200 for status in statuses.values())
        return web.json_response(
            {"status": "ok" if healthy else "unhealthy", "upstreams": statuses},
            status=200 if healthy else 503,
        )

    async def models_endpoint(self, _: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": self.external_model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "qwen35-2b-dual-policy",
                    }
                ],
            }
        )

    def _audit(
        self,
        *,
        role: str,
        endpoint: str,
        body: bytes,
        status: int,
        mode: str = "forwarded",
        action_sha256: str | None = None,
        session_sha256: str | None = None,
    ) -> None:
        event = {
            "schema_version": "qwen35-2b-dual-policy-route/v1",
            "sequence": self.sequence,
            "role": role,
            "endpoint": endpoint,
            "request_sha256": hashlib.sha256(body).hexdigest(),
            "upstream_model": self.models[role],
            "status": status,
            "mode": mode,
        }
        if action_sha256 is not None:
            event["action_sha256"] = action_sha256
        if session_sha256 is not None:
            event["session_sha256"] = session_sha256
        with self.audit_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        self.sequence += 1

    async def _route_json(self, request: web.Request, *, endpoint: str) -> web.StreamResponse:
        if self.client is None:
            raise RuntimeError("proxy client is not initialized")
        received = await request.read()
        payload = json.loads(received)
        if not isinstance(payload, dict):
            raise ValueError("chat completion payload must be an object")
        role, routed = routed_payload(
            payload,
            coordinator_model=self.models["coordinator"],
            child_model=self.models["child"],
            private_evidence_token_ids=self.private_evidence_token_ids,
        )
        stripped_sampling_fields: tuple[str, ...] = ()
        strip_tool_choice = should_strip_tool_choice(
            role,
            strip_child=self.strip_child_tool_choice,
            strip_coordinator=self.strip_coordinator_tool_choice,
        )
        if endpoint == "/inference/v1/generate" and strip_tool_choice:
            routed, stripped_sampling_fields = without_tool_choice_constraints(routed)
        body = json.dumps(routed, separators=(",", ":"), ensure_ascii=False).encode()
        session_id = request.headers.get("x-session-id")
        session_sha256 = (
            hashlib.sha256(session_id.encode()).hexdigest() if session_id else None
        )
        leak_exact_action = (
            role == "coordinator" and self.leak_coordinator_exact_action
        ) or (role == "child" and self.leak_child_exact_action)
        if endpoint == "/inference/v1/generate" and leak_exact_action:
            token_ids = payload.get("token_ids")
            completion_builder = (
                exact_ipython_completion_ids
                if role == "coordinator"
                else exact_child_ipython_completion_ids
            )
            leaked = (
                completion_builder(self.tokenizer, token_ids)
                if isinstance(token_ids, list)
                and all(isinstance(token_id, int) for token_id in token_ids)
                else None
            )
            if leaked is not None:
                completion_ids, action_sha256 = leaked
                if session_sha256 is None:
                    self._audit(
                        role=role,
                        endpoint=endpoint,
                        body=body,
                        status=400,
                        mode=f"leak_rejected_missing_{role}_session",
                    )
                    return web.json_response(
                        {"error": f"exact {role} action leak requires x-session-id"},
                        status=400,
                    )
                if session_sha256 in self.leaked_session_hashes[role]:
                    leaked = None
                else:
                    self.leaked_session_hashes[role].add(session_sha256)
            if leaked is not None:
                completion_ids, action_sha256 = leaked
                response_body = synthetic_generate_response(
                    completion_ids, sequence=self.sequence
                )
                self._audit(
                    role=role,
                    endpoint=endpoint,
                    body=body,
                    status=200,
                    mode=f"leaked_exact_{role}_action",
                    action_sha256=action_sha256,
                    session_sha256=session_sha256,
                )
                return web.Response(body=response_body, content_type="application/json")
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in HOP_BY_HOP and key.lower() != "host"
        }
        async with self.client.post(
            f"{self.urls[role].removesuffix('/v1')}{endpoint}",
            data=body,
            headers=headers,
        ) as upstream:
            response_headers = {
                key: value
                for key, value in upstream.headers.items()
                if key.lower() not in HOP_BY_HOP
            }
            response = web.StreamResponse(status=upstream.status, headers=response_headers)
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            self._audit(
                role=role,
                endpoint=endpoint,
                body=body,
                status=upstream.status,
                mode=(
                    "forwarded_without_tool_choice"
                    if stripped_sampling_fields
                    else "forwarded"
                ),
                session_sha256=session_sha256,
            )
            return response

    async def chat_completions(self, request: web.Request) -> web.StreamResponse:
        return await self._route_json(request, endpoint="/v1/chat/completions")

    async def generate(self, request: web.Request) -> web.StreamResponse:
        return await self._route_json(request, endpoint="/inference/v1/generate")


def build_app(proxy: DualPolicyProxy) -> web.Application:
    app = web.Application(client_max_size=1024**3)
    app.on_startup.append(proxy.startup)
    app.on_cleanup.append(proxy.cleanup)
    app.router.add_get("/health", proxy.health)
    app.router.add_get("/v1/models", proxy.models_endpoint)
    app.router.add_post("/v1/chat/completions", proxy.chat_completions)
    app.router.add_post("/inference/v1/generate", proxy.generate)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--coordinator-model", required=True)
    parser.add_argument("--child-url", required=True)
    parser.add_argument("--child-model", required=True)
    parser.add_argument("--external-model", required=True)
    parser.add_argument("--tokenizer-model")
    parser.add_argument("--leak-coordinator-exact-action", action="store_true")
    parser.add_argument("--leak-child-exact-action", action="store_true")
    parser.add_argument("--strip-child-tool-choice", action="store_true")
    parser.add_argument("--strip-coordinator-tool-choice", action="store_true")
    parser.add_argument("--audit-log", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model or args.coordinator_model)
    private_evidence_token_ids = tokenizer.encode(PRIVATE_EVIDENCE_HEADER, add_special_tokens=False)
    if not private_evidence_token_ids:
        raise RuntimeError("private-evidence role marker encoded to an empty token sequence")
    proxy = DualPolicyProxy(
        coordinator_url=args.coordinator_url,
        coordinator_model=args.coordinator_model,
        child_url=args.child_url,
        child_model=args.child_model,
        external_model=args.external_model,
        audit_log=args.audit_log.resolve(),
        private_evidence_token_ids=private_evidence_token_ids,
        tokenizer=tokenizer,
        leak_coordinator_exact_action=args.leak_coordinator_exact_action,
        leak_child_exact_action=args.leak_child_exact_action,
        strip_child_tool_choice=args.strip_child_tool_choice,
        strip_coordinator_tool_choice=args.strip_coordinator_tool_choice,
    )
    web.run_app(build_app(proxy), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
