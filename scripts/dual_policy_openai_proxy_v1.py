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
RECURSIVE_COORDINATOR_HEADER = "[recursive coordinator session contract]"
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
TYPED_PARENT_RETURN_TOOL = "return_to_parent"
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


def normalize_openai_finish_reason(body: bytes) -> tuple[bytes, int]:
    """Map vLLM's internal ``abort`` extension to the OpenAI ``stop`` enum.

    The OpenAI client rejects ``abort`` before the harness can retain the
    partial completion, turning one cancelled stream into an evaluation-wide
    infrastructure failure. The proxy is the protocol boundary, so keep the
    backend extension from escaping it while recording every rewrite.
    """

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body, 0
    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
        return body, 0
    rewrites = 0
    for choice in payload["choices"]:
        if isinstance(choice, dict) and choice.get("finish_reason") == "abort":
            choice["finish_reason"] = "stop"
            rewrites += 1
    if not rewrites:
        return body, 0
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(), rewrites


def normalize_sse_event(event: bytes) -> tuple[bytes, int]:
    """Normalize every JSON ``data:`` line in one complete SSE event."""

    output = []
    rewrites = 0
    for line in event.splitlines(keepends=True):
        content = line.rstrip(b"\r\n")
        ending = line[len(content) :]
        if not content.startswith(b"data:"):
            output.append(line)
            continue
        prefix, separator, data = content.partition(b" ")
        if not separator or data == b"[DONE]":
            output.append(line)
            continue
        normalized, count = normalize_openai_finish_reason(data)
        output.append(prefix + separator + normalized + ending)
        rewrites += count
    return b"".join(output), rewrites


def _contains_private_evidence(value: Any) -> bool:
    if isinstance(value, str):
        return PRIVATE_EVIDENCE_HEADER in value
    if isinstance(value, list):
        return any(_contains_private_evidence(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_private_evidence(item) for item in value.values())
    return False


def _contains_marker(value: Any, marker: str) -> bool:
    if isinstance(value, str):
        return marker in value
    if isinstance(value, list):
        return any(_contains_marker(item, marker) for item in value)
    if isinstance(value, dict):
        return any(_contains_marker(item, marker) for item in value.values())
    return False


def _contains_token_subsequence(token_ids: list[int], marker_ids: list[int]) -> bool:
    if not marker_ids or len(marker_ids) > len(token_ids):
        return False
    width = len(marker_ids)
    return any(token_ids[start : start + width] == marker_ids for start in range(len(token_ids) - width + 1))


def request_role(
    payload: dict[str, Any],
    *,
    private_evidence_token_ids: list[int] | None = None,
    recursive_coordinator_token_ids: list[int] | None = None,
) -> str:
    messages = payload.get("messages")
    if isinstance(messages, list):
        has_private = _contains_private_evidence(messages)
        has_recursive_coordinator = _contains_marker(
            messages, RECURSIVE_COORDINATOR_HEADER
        )
        if has_private and has_recursive_coordinator:
            raise ValueError("request contains conflicting delegated-session role markers")
        return "child" if has_private else "coordinator"
    token_ids = payload.get("token_ids")
    if isinstance(token_ids, list) and all(isinstance(item, int) for item in token_ids):
        marker_ids = private_evidence_token_ids or []
        coordinator_ids = recursive_coordinator_token_ids or []
        has_private = _contains_token_subsequence(token_ids, marker_ids)
        has_recursive_coordinator = _contains_token_subsequence(token_ids, coordinator_ids)
        if has_private and has_recursive_coordinator:
            raise ValueError("request contains conflicting delegated-session role markers")
        return "child" if has_private else "coordinator"
    raise ValueError("role-routed request lacks messages or token_ids")


def routed_payload(
    payload: dict[str, Any],
    *,
    coordinator_model: str,
    child_model: str,
    private_evidence_token_ids: list[int] | None = None,
    recursive_coordinator_token_ids: list[int] | None = None,
) -> tuple[str, dict[str, Any]]:
    role = request_role(
        payload,
        private_evidence_token_ids=private_evidence_token_ids,
        recursive_coordinator_token_ids=recursive_coordinator_token_ids,
    )
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


def disclosed_child_action_from_messages(messages: Any) -> str | None:
    """Extract the return scaffold from structured Chat Completions messages."""

    fragments: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            fragments.append(value)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    collect(messages)
    return disclosed_child_action("\n".join(fragments))


def force_ipython_code_schema(payload: dict[str, Any], code: str) -> dict[str, Any]:
    """Constrain one Chat Completions turn to the disclosed IPython action."""

    tools = payload.get("tools")
    if not isinstance(tools, list):
        raise ValueError("exact return action requires Chat Completions tools")
    rewritten_tools = []
    found_ipython = False
    for tool in tools:
        if not isinstance(tool, dict):
            rewritten_tools.append(tool)
            continue
        function = tool.get("function")
        if not isinstance(function, dict) or function.get("name") != "ipython":
            rewritten_tools.append(tool)
            continue
        found_ipython = True
        rewritten_tools.append(
            {
                **tool,
                "function": {
                    **function,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "enum": [code]},
                        },
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    if not found_ipython:
        raise ValueError("exact return action requires the IPython tool")
    return {
        **payload,
        "tools": rewritten_tools,
        "tool_choice": {"type": "function", "function": {"name": "ipython"}},
        "parallel_tool_calls": False,
    }


def force_typed_parent_return_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """Expose one parent-return action without disclosing its payload or routing code."""

    tools = payload.get("tools")
    if not isinstance(tools, list) or not any(
        isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and tool["function"].get("name") == "ipython"
        for tool in tools
    ):
        raise ValueError("typed parent return requires the IPython tool")
    rewritten = {
        **payload,
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": TYPED_PARENT_RETURN_TOOL,
                    "description": (
                        "Return the computed result to this session's direct parent. "
                        "The harness fixes routing, permits exactly one return, and "
                        "ends the delegated session after delivery."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "payload": {
                                "type": "string",
                                "description": "The result computed from this session's evidence.",
                            }
                        },
                        "required": ["payload"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": TYPED_PARENT_RETURN_TOOL},
        },
        "parallel_tool_calls": False,
    }
    rewritten.pop("stream_options", None)
    return rewritten


def rewrite_typed_parent_return_response(body: bytes) -> tuple[bytes, int, str | None]:
    """Translate a model-computed typed return into the native IPython send call."""

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body, 0, None
    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
        return body, 0, None
    rewrites = 0
    action_sha256 = None
    for choice in payload["choices"]:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            continue
        tool_call = tool_calls[0]
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict) or function.get("name") != TYPED_PARENT_RETURN_TOOL:
            continue
        arguments = function.get("arguments")
        try:
            parsed_arguments = (
                json.loads(arguments) if isinstance(arguments, str) else arguments
            )
        except json.JSONDecodeError:
            continue
        if (
            not isinstance(parsed_arguments, dict)
            or set(parsed_arguments) != {"payload"}
            or not isinstance(parsed_arguments["payload"], str)
        ):
            continue
        code = (
            "await agent_message.send("
            f"{json.dumps(parsed_arguments['payload'], ensure_ascii=False)}, "
            "receiver_role='parent')"
        )
        tool_call["type"] = "function"
        function["name"] = "ipython"
        function["arguments"] = json.dumps(
            {"code": code}, separators=(",", ":"), ensure_ascii=False
        )
        rewrites += 1
        action_sha256 = hashlib.sha256(code.encode()).hexdigest()
    if not rewrites:
        return body, 0, None
    return (
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(),
        rewrites,
        action_sha256,
    )


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
        recursive_coordinator_token_ids: list[int] | None = None,
        leak_coordinator_exact_action: bool = False,
        leak_coordinator_return_action: bool = False,
        typed_coordinator_return: bool = False,
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
        self.recursive_coordinator_token_ids = recursive_coordinator_token_ids or []
        self.tokenizer = tokenizer
        self.leak_coordinator_exact_action = leak_coordinator_exact_action
        self.leak_coordinator_return_action = leak_coordinator_return_action
        self.typed_coordinator_return = typed_coordinator_return
        self.leak_child_exact_action = leak_child_exact_action
        self.strip_child_tool_choice = strip_child_tool_choice
        self.strip_coordinator_tool_choice = strip_coordinator_tool_choice
        self.client: ClientSession | None = None
        self.sequence = 0
        self.leaked_session_hashes: dict[str, set[str]] = {
            "coordinator": set(),
            "coordinator_return": set(),
            "child": set(),
        }

    async def startup(self, _: web.Application) -> None:
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)
        if self.audit_log.exists():
            with self.audit_log.open(encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            self.sequence = len(events)
            for event in events:
                session_sha256 = event.get("session_sha256")
                leak_scope = next(
                    (
                        scope
                        for scope in self.leaked_session_hashes
                        if event.get("mode") == f"leaked_exact_{scope}_action"
                    ),
                    None,
                )
                if leak_scope is not None and isinstance(session_sha256, str):
                    self.leaked_session_hashes[leak_scope].add(session_sha256)
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
        response_rewrites: int = 0,
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
        if response_rewrites:
            event["response_rewrites"] = {
                "abort_finish_reason_to_stop": response_rewrites
            }
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
            recursive_coordinator_token_ids=self.recursive_coordinator_token_ids,
        )
        stripped_sampling_fields: tuple[str, ...] = ()
        strip_tool_choice = should_strip_tool_choice(
            role,
            strip_child=self.strip_child_tool_choice,
            strip_coordinator=self.strip_coordinator_tool_choice,
        )
        if endpoint == "/inference/v1/generate" and strip_tool_choice:
            routed, stripped_sampling_fields = without_tool_choice_constraints(routed)
        session_id = request.headers.get("x-session-id")
        session_sha256 = (
            hashlib.sha256(session_id.encode()).hexdigest() if session_id else None
        )
        forced_chat_scope = None
        forced_chat_action_sha256 = None
        typed_return_scope = False
        typed_return_action_sha256 = None
        if (
            endpoint == "/v1/chat/completions"
            and role == "coordinator"
            and self.leak_coordinator_return_action
        ):
            code = disclosed_child_action_from_messages(payload.get("messages"))
            if code is not None:
                forced_chat_scope = "coordinator_return"
                if session_sha256 is None:
                    body = json.dumps(
                        routed, separators=(",", ":"), ensure_ascii=False
                    ).encode()
                    self._audit(
                        role=role,
                        endpoint=endpoint,
                        body=body,
                        status=400,
                        mode="leak_rejected_missing_coordinator_return_session",
                    )
                    return web.json_response(
                        {
                            "error": (
                                "exact coordinator_return action leak requires "
                                "x-session-id"
                            )
                        },
                        status=400,
                    )
                if session_sha256 in self.leaked_session_hashes[forced_chat_scope]:
                    forced_chat_scope = None
                else:
                    self.leaked_session_hashes[forced_chat_scope].add(session_sha256)
                    routed = force_ipython_code_schema(routed, code)
                    forced_chat_action_sha256 = hashlib.sha256(code.encode()).hexdigest()
        if (
            endpoint == "/v1/chat/completions"
            and role == "coordinator"
            and self.typed_coordinator_return
            and _contains_marker(payload.get("messages"), RECURSIVE_COORDINATOR_HEADER)
        ):
            typed_return_scope = True
            if session_sha256 is None:
                body = json.dumps(
                    routed, separators=(",", ":"), ensure_ascii=False
                ).encode()
                self._audit(
                    role=role,
                    endpoint=endpoint,
                    body=body,
                    status=400,
                    mode="typed_return_rejected_missing_session",
                )
                return web.json_response(
                    {"error": "typed coordinator return requires x-session-id"},
                    status=400,
                )
            if session_sha256 in self.leaked_session_hashes["coordinator_return"]:
                typed_return_scope = False
            else:
                self.leaked_session_hashes["coordinator_return"].add(session_sha256)
                routed = force_typed_parent_return_schema(routed)
        body = json.dumps(routed, separators=(",", ":"), ensure_ascii=False).encode()
        leak_specs = []
        if role == "coordinator" and self.leak_coordinator_exact_action:
            leak_specs.append(("coordinator", exact_ipython_completion_ids))
        if role == "coordinator" and self.leak_coordinator_return_action:
            leak_specs.append(("coordinator_return", exact_child_ipython_completion_ids))
        if role == "child" and self.leak_child_exact_action:
            leak_specs.append(("child", exact_child_ipython_completion_ids))
        if endpoint == "/inference/v1/generate" and leak_specs:
            token_ids = payload.get("token_ids")
            leaked = None
            leak_scope = None
            if isinstance(token_ids, list) and all(
                isinstance(token_id, int) for token_id in token_ids
            ):
                for candidate_scope, completion_builder in leak_specs:
                    leaked = completion_builder(self.tokenizer, token_ids)
                    if leaked is not None:
                        leak_scope = candidate_scope
                        break
            if leaked is not None:
                assert leak_scope is not None
                completion_ids, action_sha256 = leaked
                if session_sha256 is None:
                    self._audit(
                        role=role,
                        endpoint=endpoint,
                        body=body,
                        status=400,
                        mode=f"leak_rejected_missing_{leak_scope}_session",
                    )
                    return web.json_response(
                        {"error": f"exact {leak_scope} action leak requires x-session-id"},
                        status=400,
                    )
                if session_sha256 in self.leaked_session_hashes[leak_scope]:
                    leaked = None
                else:
                    self.leaked_session_hashes[leak_scope].add(session_sha256)
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
                    mode=f"leaked_exact_{leak_scope}_action",
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
            response_rewrites = 0
            if upstream.headers.get("content-type", "").startswith("text/event-stream"):
                response = web.StreamResponse(status=upstream.status, headers=response_headers)
                await response.prepare(request)
                pending = b""
                async for chunk in upstream.content.iter_any():
                    pending += chunk
                    while b"\n\n" in pending:
                        event, pending = pending.split(b"\n\n", 1)
                        normalized, count = normalize_sse_event(event)
                        response_rewrites += count
                        await response.write(normalized + b"\n\n")
                if pending:
                    normalized, count = normalize_sse_event(pending)
                    response_rewrites += count
                    await response.write(normalized)
                await response.write_eof()
            else:
                upstream_body = await upstream.read()
                normalized, response_rewrites = normalize_openai_finish_reason(
                    upstream_body
                )
                if typed_return_scope and upstream.status == 200:
                    normalized, _, typed_return_action_sha256 = (
                        rewrite_typed_parent_return_response(normalized)
                    )
                response = web.Response(
                    status=upstream.status,
                    headers=response_headers,
                    body=normalized,
                )
            self._audit(
                role=role,
                endpoint=endpoint,
                body=body,
                status=upstream.status,
                mode=(
                    (
                        "forwarded_typed_coordinator_return"
                        if typed_return_action_sha256 is not None
                        else "forwarded_typed_coordinator_return_untranslated"
                    )
                    if typed_return_scope
                    else "forwarded_forced_exact_coordinator_return_schema"
                    if forced_chat_scope is not None
                    else (
                        "forwarded_normalized_abort_finish_reason"
                        if response_rewrites
                        else (
                            "forwarded_without_tool_choice"
                            if stripped_sampling_fields
                            else "forwarded"
                        )
                    )
                ),
                action_sha256=(
                    typed_return_action_sha256 or forced_chat_action_sha256
                ),
                session_sha256=session_sha256,
                response_rewrites=response_rewrites,
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
    parser.add_argument("--leak-coordinator-return-action", action="store_true")
    parser.add_argument("--typed-coordinator-return", action="store_true")
    parser.add_argument("--leak-child-exact-action", action="store_true")
    parser.add_argument("--strip-child-tool-choice", action="store_true")
    parser.add_argument("--strip-coordinator-tool-choice", action="store_true")
    parser.add_argument("--audit-log", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model or args.coordinator_model)
    private_evidence_token_ids = tokenizer.encode(PRIVATE_EVIDENCE_HEADER, add_special_tokens=False)
    recursive_coordinator_token_ids = tokenizer.encode(
        RECURSIVE_COORDINATOR_HEADER, add_special_tokens=False
    )
    if not private_evidence_token_ids:
        raise RuntimeError("private-evidence role marker encoded to an empty token sequence")
    if not recursive_coordinator_token_ids:
        raise RuntimeError("recursive-coordinator role marker encoded to an empty token sequence")
    proxy = DualPolicyProxy(
        coordinator_url=args.coordinator_url,
        coordinator_model=args.coordinator_model,
        child_url=args.child_url,
        child_model=args.child_model,
        external_model=args.external_model,
        audit_log=args.audit_log.resolve(),
        private_evidence_token_ids=private_evidence_token_ids,
        recursive_coordinator_token_ids=recursive_coordinator_token_ids,
        tokenizer=tokenizer,
        leak_coordinator_exact_action=args.leak_coordinator_exact_action,
        leak_coordinator_return_action=args.leak_coordinator_return_action,
        typed_coordinator_return=args.typed_coordinator_return,
        leak_child_exact_action=args.leak_child_exact_action,
        strip_child_tool_choice=args.strip_child_tool_choice,
        strip_coordinator_tool_choice=args.strip_coordinator_tool_choice,
    )
    web.run_app(build_app(proxy), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
