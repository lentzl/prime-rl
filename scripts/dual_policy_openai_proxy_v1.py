#!/usr/bin/env python3
"""Route Prime Agent coordinator and child requests to separate OpenAI endpoints."""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout, web

PRIVATE_EVIDENCE_HEADER = "[private evidence supplied to this reviewer]"
RECURSIVE_COORDINATOR_HEADER = "[recursive coordinator session contract]"
DOCUMENT_COORDINATOR_HEADER = "[recursive document coordinator session contract]"
FREE_DOCUMENT_TOPOLOGY_HEADER = "[free document topology contract]"
ROOT_COORDINATOR_HEADER = "[root coordinator session contract]"
LEAF_REPORTER_HEADER = "[leaf reporter session contract]"
CHILD_ACTION_SCAFFOLD_HEADER = "[training-only child action scaffold]"
EXACT_ACTION_MARKER = "[interaction-curriculum exact action]"
ROOT_COORDINATOR_CONTRACT = """[root coordinator session contract]
session_role=root_coordinator
is_root=true
has_parent=false
can_delegate=true
can_finalize_user=true
You are the root coordinator for the user's task. This identity persists across every
delegation and resume. A child task or child report is delegated-session traffic, not
a reassignment of your role: never adopt the child's worker identity and never send
your result to receiver_role='parent'. After a child report arrives, remain the root
coordinator and treat that first report as the final child evidence.
Do not message the child, poll, or call another tool after delivery; apply the user's simple formula
directly and finalize the requested answer."""
LEAF_REPORTER_CONTRACT = """[leaf reporter session contract]
session_role=leaf_reporter
is_root=false
has_parent=true
can_delegate=false
can_finalize_user=false
return_contract=exactly_one_parent_report
You are a leaf child with exactly one parent. The private evidence is inline in your
session; an evidence label is not a runtime path. When the inline-evidence scaffold is
enabled, the string INLINE_EVIDENCE is already bound in your IPython cell. Compute the
requested value once with Python, store it as result, then make one IPython call:
await agent_message.send(str(result), receiver_role='parent'). A successful send completes your task.
After it succeeds, do not call another tool, send another message, answer later traffic,
or finalize the user's request; remain stopped."""
ROOT_ACTION_PATTERN = re.compile(
    r"In the root coordinator's first IPython call, execute this code exactly:\s*"
    r"```python\s*\n(?P<code>.*?)\n```",
    re.DOTALL,
)
DOCUMENT_MANAGER_CONTRACT_PATTERN = re.compile(
    r"(?P<contract>\[recursive document coordinator session contract\]\n"
    r"session_role=document_coordinator\n"
    r"is_root=false\n"
    r"has_parent=true\n"
    r"can_delegate=true\n"
    r"can_finalize_user=false\n"
    r"maximum_descendant_depth=1\n"
    r"return_contract=exactly_one_parent_report\n"
    r".*?Send that object exactly once to receiver_role='parent', then stop\.)",
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
PATH_READ_TEXT_PATTERN = re.compile(
    r"(?:pathlib\.)?Path\((?P<quote>['\"])[^'\"]+(?P=quote)\)\.read_text\(\)"
)
DOCUMENT_LEAF_PATH_PATTERN = re.compile(
    r"(?P<path>/workspace/document-recursion/[^\s'\"`]+/(?:alpha|beta|gamma)\.md)"
)
MODEL_TOOL_CONTAMINATION_PATTERN = re.compile(
    r"</?(?:parameter|function|tool_call)>|<\|(?:endoftext|im_start|im_end)\|>"
)
TYPED_PARENT_RETURN_TOOL = "return_to_parent"
REQUIRED_REVIEW_MARKER = "Required review: "
EVIDENCE_LABEL_MARKER = "Evidence label: "
PATHLESS_INLINE_EVIDENCE_LABEL = "INLINE_EVIDENCE"
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


class _ComputeSendToValue(ast.NodeTransformer):
    """Turn a forbidden compute-stage parent send into its payload expression."""

    def __init__(self) -> None:
        self.rewrites = 0

    def visit_Await(self, node: ast.Await) -> ast.AST:
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "send"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "agent_message"
        ):
            payload = call.args[0] if call.args else next(
                (
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg in {"message", "payload"}
                ),
                None,
            )
            if payload is not None:
                self.rewrites += 1
                return ast.copy_location(self.visit(payload), node)
        return self.generic_visit(node)


class _AwaitBareParentSend(ast.NodeTransformer):
    """Normalize a model-authored bare parent-send expression into an await."""

    def __init__(self) -> None:
        self.rewrites = 0

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "send"
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "agent_message"
        ):
            self.rewrites += 1
            return ast.copy_location(ast.Expr(value=ast.Await(value=value)), node)
        return self.generic_visit(node)


def _report_final_value_to_parent(tree: ast.Module) -> int:
    """Wrap a model-authored cell's final value in the native parent report."""

    if not tree.body:
        return 0
    final = tree.body[-1]
    value = final.value if isinstance(final, ast.Expr) else None
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "print"
        and len(value.args) == 1
        and not value.keywords
    ):
        value = value.args[0]
    if value is None:
        assigned_result = any(
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == "result"
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            )
            for node in ast.walk(tree)
        )
        if not assigned_result:
            return 0
        value = ast.Name(id="result", ctx=ast.Load())
        final = None
    serialized_value = (
        value
        if isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "str"
        and len(value.args) == 1
        and not value.keywords
        else ast.Call(
            func=ast.Name(id="str", ctx=ast.Load()),
            args=[value],
            keywords=[],
        )
    )
    report = ast.Expr(
        value=ast.Await(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="agent_message", ctx=ast.Load()),
                    attr="send",
                    ctx=ast.Load(),
                ),
                args=[serialized_value],
                keywords=[
                    ast.keyword(arg="receiver_role", value=ast.Constant(value="parent"))
                ],
            )
        )
    )
    if final is None:
        tree.body.append(report)
    else:
        tree.body[-1] = report
    ast.fix_missing_locations(tree)
    return 1


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


def _is_bounded_leaf_session(value: Any) -> bool:
    return all(
        _contains_marker(value, marker)
        for marker in (
            RECURSIVE_COORDINATOR_HEADER,
            "is_root=false",
            "can_delegate=false",
            "can_finalize_user=false",
            "return_contract=exactly_one_parent_report",
        )
    )


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
    document_coordinator_token_ids: list[int] | None = None,
    root_depth_token_ids: list[int] | None = None,
    depth_default_child: bool = False,
) -> str:
    messages = payload.get("messages")
    if isinstance(messages, list):
        has_private = _contains_private_evidence(messages)
        has_recursive_coordinator = _contains_marker(
            messages, RECURSIVE_COORDINATOR_HEADER
        )
        has_document_coordinator = _contains_marker(
            messages, DOCUMENT_COORDINATOR_HEADER
        )
        bounded_leaf = _is_bounded_leaf_session(messages)
        if has_private and (has_recursive_coordinator or has_document_coordinator) and not bounded_leaf:
            raise ValueError("request contains conflicting delegated-session role markers")
        if has_document_coordinator:
            return "coordinator"
        if depth_default_child and not _contains_marker(messages, "Recursive agent depth: 0"):
            return "child"
        return "child" if has_private or bounded_leaf else "coordinator"
    token_ids = payload.get("token_ids")
    if isinstance(token_ids, list) and all(isinstance(item, int) for item in token_ids):
        marker_ids = private_evidence_token_ids or []
        coordinator_ids = recursive_coordinator_token_ids or []
        document_ids = document_coordinator_token_ids or []
        root_ids = root_depth_token_ids or []
        has_private = _contains_token_subsequence(token_ids, marker_ids)
        has_recursive_coordinator = _contains_token_subsequence(token_ids, coordinator_ids)
        has_document_coordinator = _contains_token_subsequence(token_ids, document_ids)
        if has_private and (has_recursive_coordinator or has_document_coordinator):
            raise ValueError("request contains conflicting delegated-session role markers")
        if has_document_coordinator:
            return "coordinator"
        if depth_default_child and not _contains_token_subsequence(token_ids, root_ids):
            return "child"
        return "child" if has_private else "coordinator"
    raise ValueError("role-routed request lacks messages or token_ids")


def routed_payload(
    payload: dict[str, Any],
    *,
    coordinator_model: str,
    child_model: str,
    private_evidence_token_ids: list[int] | None = None,
    recursive_coordinator_token_ids: list[int] | None = None,
    document_coordinator_token_ids: list[int] | None = None,
    root_depth_token_ids: list[int] | None = None,
    depth_default_child: bool = False,
) -> tuple[str, dict[str, Any]]:
    role = request_role(
        payload,
        private_evidence_token_ids=private_evidence_token_ids,
        recursive_coordinator_token_ids=recursive_coordinator_token_ids,
        document_coordinator_token_ids=document_coordinator_token_ids,
        root_depth_token_ids=root_depth_token_ids,
        depth_default_child=depth_default_child,
    )
    model = child_model if role == "child" else coordinator_model
    return role, {**payload, "model": model}


def is_root_coordinator_request(payload: dict[str, Any]) -> bool:
    """Identify the depth-zero Prime Agent session rather than delegated traffic."""

    messages = payload.get("messages")
    return (
        isinstance(messages, list)
        and _contains_marker(messages, "Recursive agent depth: 0")
        and not _contains_private_evidence(messages)
    )


def is_incomplete_root_wait_request(payload: dict[str, Any]) -> bool:
    """Identify a gate continuation that must remain passive pending child delivery."""

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    latest_user = next(
        (
            message.get("content")
            for message in reversed(messages)
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        None,
    )
    return _contains_marker(latest_user, "Autonomous quality gate failed") and (
        _contains_marker(latest_user, "end-to-end coordinator task is not complete")
    )


def is_incomplete_document_manager_wait_request(payload: dict[str, Any]) -> bool:
    """Keep a document root passive after manager admission until its report arrives."""

    messages = payload.get("messages")
    if not isinstance(messages, list) or not is_root_coordinator_request(payload):
        return False
    manager_admitted = _contains_marker(messages, DOCUMENT_COORDINATOR_HEADER) and (
        _contains_marker(messages, "document_manager = await rlm(")
    )
    spawn_receipt_present = any(
        isinstance(message, dict) and message.get("role") == "tool"
        for message in messages
    )
    valid_manager_report_present = (
        document_root_manager_report_from_messages(messages) is not None
    )
    return manager_admitted and spawn_receipt_present and not valid_manager_report_present


def document_leaf_path_from_messages(messages: Any) -> str | None:
    """Extract the one document shard owned by an unambiguous terminal leaf."""

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
    matches = {
        match.group("path")
        for fragment in fragments
        for match in DOCUMENT_LEAF_PATH_PATTERN.finditer(fragment)
    }
    # A multi-shard prompt belongs to a manager or another non-terminal role.
    # Decline the leaf scaffold instead of turning imperfect routing into a 500.
    if len(matches) != 1:
        return None
    return matches.pop()


def document_leaf_compute_report_code(path: str) -> str:
    """Return one deterministic file compute-and-parent-report action."""

    return (
        "from pathlib import Path\n"
        "import json\n"
        f"document_leaf_text = Path({path!r}).read_text()\n"
        "document_leaf_result = {\n"
        "    'words': len(document_leaf_text.split()),\n"
        "    'h2': sum(line.startswith('## ') for line in document_leaf_text.splitlines()),\n"
        "}\n"
        "document_leaf_delivery = await agent_message.send("
        "json.dumps(document_leaf_result, separators=(',', ':')), "
        "receiver_role='parent')\n"
        "print('DOCUMENT_LEAF_REPORT:' + "
        "json.dumps(document_leaf_result, separators=(',', ':')))\n"
        "document_leaf_delivery"
    )


def document_leaf_report_from_messages(
    messages: Any, path: str
) -> dict[str, dict[str, int]]:
    """Recover one scaffolded leaf report from its structured tool receipt."""

    stem = Path(path).stem
    if stem not in {"alpha", "beta", "gamma"} or not isinstance(messages, list):
        return {}
    marker = "DOCUMENT_LEAF_REPORT:"
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
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

        collect(message.get("content"))
        for fragment in reversed(fragments):
            if marker not in fragment:
                continue
            candidate_text = fragment.split(marker, 1)[1].splitlines()[0].strip()
            try:
                candidate = json.loads(candidate_text)
            except (json.JSONDecodeError, ValueError):
                continue
            if (
                isinstance(candidate, dict)
                and set(candidate) == {"words", "h2"}
                and all(type(candidate[key]) is int for key in ("words", "h2"))
            ):
                return {stem: {"words": candidate["words"], "h2": candidate["h2"]}}
    return {}


def document_leaf_report_completed(messages: Any) -> bool:
    """Recognize a successful scaffolded leaf send from structured history."""

    if not isinstance(messages, list):
        return False
    action_index = next(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, dict)
            and message.get("role") == "assistant"
            and _contains_marker(message, "document_leaf_text = Path(")
        ),
        None,
    )
    if action_index is None:
        return False
    return any(
        isinstance(message, dict)
        and message.get("role") == "tool"
        and _contains_marker(message, "deliveryStatus")
        for message in messages[action_index + 1 :]
    )


def document_manager_reports_from_messages(
    messages: Any,
) -> dict[str, dict[str, int]]:
    """Collect unique valid terminal document reports delivered to one manager."""

    if not isinstance(messages, list):
        return {}
    reports: dict[str, dict[str, int]] = {}

    def record(stem: str, payload: Any) -> None:
        if (
            not isinstance(payload, dict)
            or set(payload) != {"words", "h2"}
            or not all(type(payload[key]) is int for key in ("words", "h2"))
        ):
            return
        normalized = {"words": payload["words"], "h2": payload["h2"]}
        if stem in reports and reports[stem] != normalized:
            raise ValueError(f"document manager received conflicting {stem} reports")
        reports[stem] = normalized

    for message in messages:
        if not isinstance(message, dict):
            continue
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

        # Prime Agent persists agent messages as a custom role and renderers may
        # expose them as user, developer, or custom input. The exact directional
        # marker plus strict two-integer payload is the stable protocol boundary.
        collect(message)
        text = "\n".join(fragments)
        relay_marker = "Observed child report map: "
        for fragment in fragments:
            if relay_marker not in fragment:
                continue
            try:
                relayed = json.loads(fragment.split(relay_marker, 1)[1].splitlines()[0])
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(relayed, dict):
                continue
            for child_name, payload in relayed.items():
                match = re.fullmatch(
                    r"(alpha|beta|gamma)-document-worker", str(child_name)
                )
                if match is not None:
                    record(match.group(1), payload)
        sender_match = re.search(
            r"\[from child:(alpha|beta|gamma)-document-worker\]", text
        )
        if sender_match is None:
            continue
        payload = None
        for fragment in reversed(fragments):
            try:
                candidate = json.loads(fragment.rsplit("\n\n", 1)[-1].strip())
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break
        if payload is None:
            continue
        stem = sender_match.group(1)
        record(stem, payload)
    return reports


def document_manager_parent_report(reports: dict[str, dict[str, int]]) -> dict[str, int]:
    """Assemble the exact root-facing document report from three leaf reports."""

    stems = ("alpha", "beta", "gamma")
    if set(reports) != set(stems):
        raise ValueError("document manager fan-in requires alpha, beta, and gamma")
    result: dict[str, int] = {}
    for stem in stems:
        result[f"{stem}_words"] = reports[stem]["words"]
        result[f"{stem}_h2"] = reports[stem]["h2"]
    result["total_words"] = sum(reports[stem]["words"] for stem in stems)
    result["total_h2"] = sum(reports[stem]["h2"] for stem in stems)
    return result


def document_manager_parent_report_code(reports: dict[str, dict[str, int]]) -> str:
    """Return one exact serialized manager-to-root send action."""

    result = document_manager_parent_report(reports)
    return (
        "import json\n"
        f"document_manager_report = {result!r}\n"
        "await agent_message.send(json.dumps(document_manager_report, "
        "separators=(',', ':')), receiver_role='parent')"
    )


def document_manager_parent_report_completed(messages: Any) -> bool:
    """Recognize the delivery receipt for a scaffolded manager parent report."""

    if not isinstance(messages, list):
        return False
    action_index = next(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, dict)
            and message.get("role") == "assistant"
            and _contains_marker(message, "document_manager_report =")
        ),
        None,
    )
    if action_index is None:
        return False
    return any(
        isinstance(message, dict)
        and message.get("role") == "tool"
        and _contains_marker(message, "deliveryStatus")
        for message in messages[action_index + 1 :]
    )


def document_root_manager_report_from_messages(messages: Any) -> dict[str, int] | None:
    """Extract one validated final manager report delivered to the document root."""

    if not isinstance(messages, list):
        return None
    expected = {
        "alpha_words",
        "alpha_h2",
        "beta_words",
        "beta_h2",
        "gamma_words",
        "gamma_h2",
        "total_words",
        "total_h2",
    }
    results = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
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

        collect(message.get("content"))
        text = "\n".join(fragments)
        if "[from child:document-manager]" not in text:
            continue
        try:
            payload = json.loads(text.rsplit("\n\n", 1)[-1].strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if (
            isinstance(payload, dict)
            and set(payload) == expected
            and all(type(payload[key]) is int for key in expected)
        ):
            results.append(payload)
    if not results:
        return None
    if any(result != results[0] for result in results[1:]):
        raise ValueError("document root received conflicting manager reports")
    return results[0]


def document_root_direct_result_from_messages(messages: Any) -> dict[str, int] | None:
    """Recover one canonical direct-compute result from root tool history."""

    if not isinstance(messages, list):
        return None
    action_index = next(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, dict)
            and message.get("role") == "assistant"
            and _contains_marker(message, "document_result =")
        ),
        None,
    )
    if action_index is None:
        return None
    expected = {
        "alpha_words",
        "alpha_h2",
        "beta_words",
        "beta_h2",
        "gamma_words",
        "gamma_h2",
        "total_words",
        "total_h2",
    }
    results: list[dict[str, int]] = []
    for message in messages[action_index + 1 :]:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
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

        collect(message.get("content"))
        for fragment in fragments:
            try:
                candidate = ast.literal_eval(fragment.strip())
            except (SyntaxError, ValueError):
                continue
            if (
                isinstance(candidate, dict)
                and set(candidate) == expected
                and all(type(candidate[key]) is int for key in expected)
            ):
                results.append(candidate)
    if not results:
        return None
    if any(result != results[0] for result in results[1:]):
        raise ValueError("document root produced conflicting direct results")
    return results[0]


def with_root_coordinator_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Prepend a persistent role contract to a depth-zero chat request."""

    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("root coordinator contract requires chat messages")
    if _contains_marker(messages, ROOT_COORDINATOR_HEADER):
        return payload
    return {
        **payload,
        "messages": [
            {"role": "system", "content": ROOT_COORDINATOR_CONTRACT},
            *messages,
        ],
    }


def with_leaf_reporter_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Prepend the one-shot reporting contract to a private child request."""

    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("leaf reporter contract requires chat messages")
    if _contains_marker(messages, LEAF_REPORTER_HEADER):
        return payload
    return {
        **payload,
        "messages": [
            {"role": "system", "content": LEAF_REPORTER_CONTRACT},
            *messages,
        ],
    }


def root_final_answer_fields(messages: Any) -> tuple[str, ...]:
    """Extract the answer-free integer field contract from the root task."""

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
    candidates = []
    for fragment in fragments:
        for body in re.findall(r"Return\s+\{([^}]+)\}", fragment):
            fields = tuple(re.findall(r'"([^"]+)"\s*:\s*<integer>', body))
            if fields:
                candidates.append(fields)
    if not candidates:
        raise ValueError("root finalization lacks an integer output contract")
    if len(set(candidates)) != 1:
        raise ValueError("root finalization contains conflicting output contracts")
    return candidates[0]


def force_root_json_finalization(payload: dict[str, Any]) -> dict[str, Any]:
    """Constrain the post-report root turn to its answer-free JSON field schema."""

    rewritten = {**payload, "temperature": 0.0}
    for field in ("tools", "tool_choice", "parallel_tool_calls"):
        rewritten.pop(field, None)
    fields = root_final_answer_fields(payload.get("messages"))
    rewritten["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": "root_final_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {field: {"type": "integer"} for field in fields},
                "required": list(fields),
                "additionalProperties": False,
            },
        },
    }
    return rewritten


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

    if FREE_DOCUMENT_TOPOLOGY_HEADER in prompt:
        return None
    direct_action = disclosed_document_direct_action(prompt)
    if direct_action is not None:
        return direct_action
    manager_action = disclosed_document_manager_action(prompt)
    if manager_action is not None:
        return manager_action
    document_action = disclosed_document_spawn_action(prompt)
    if document_action is not None:
        return document_action
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


def disclosed_document_direct_action(prompt: str) -> str | None:
    """Build one answer-free local document action from an explicit direct contract."""

    roots = set(
        re.findall(
            r"Inspect every Markdown file in (?P<root>/[^\s]+) yourself using "
            r"the CLI or IPython; do not create a subagent\.",
            prompt,
        )
    )
    if not roots:
        return None
    if len(roots) != 1:
        raise ValueError("document direct scaffold contains conflicting roots")
    root = roots.pop()
    if not re.fullmatch(r"/workspace/document-recursion/[^/\s]+", root):
        raise ValueError("document direct scaffold contains an invalid root")
    return f"""from pathlib import Path
document_root = Path({root!r})
document_paths = sorted(document_root.glob('*.md'))
document_texts = {{path.stem: path.read_text() for path in document_paths}}
document_stats = {{
    stem: {{
        'words': len(text.split()),
        'h2': sum(line.startswith('## ') for line in text.splitlines()),
    }}
    for stem, text in document_texts.items()
}}
document_result = {{
    **{{f'{{stem}}_words': document_stats[stem]['words'] for stem in ('alpha', 'beta', 'gamma')}},
    **{{f'{{stem}}_h2': document_stats[stem]['h2'] for stem in ('alpha', 'beta', 'gamma')}},
    'total_words': sum(document_stats[stem]['words'] for stem in ('alpha', 'beta', 'gamma')),
    'total_h2': sum(document_stats[stem]['h2'] for stem in ('alpha', 'beta', 'gamma')),
}}
document_result"""


def disclosed_document_free_actions(prompt: str) -> dict[str, str] | None:
    """Build every answer-free legal action for a free-topology document task."""

    if FREE_DOCUMENT_TOPOLOGY_HEADER not in prompt:
        return None
    actions = {
        "direct": disclosed_document_direct_action(prompt),
        "flat": disclosed_document_spawn_action(prompt),
        "hierarchical": disclosed_document_manager_action(prompt),
    }
    missing = [name for name, code in actions.items() if code is None]
    if missing:
        raise ValueError(f"free document topology scaffold lacks actions: {missing}")
    return {name: code for name, code in actions.items() if code is not None}


def disclosed_document_manager_action(prompt: str) -> str | None:
    """Preserve one complete answer-free recursive manager contract at admission."""

    matches = [
        match.group("contract").strip()
        for match in DOCUMENT_MANAGER_CONTRACT_PATTERN.finditer(prompt)
    ]
    if not matches:
        return None
    if len(set(matches)) != 1:
        raise ValueError("document manager scaffold contains conflicting recursive contracts")
    contract = matches[0]
    if '"""' in contract:
        raise ValueError("document manager scaffold contains an unsafe recursive contract")
    root_match = re.search(
        r"You own the document directory (?P<root>/[^\s]+)\.", contract
    )
    if root_match is None:
        raise ValueError("document manager scaffold lacks one owned directory")
    root = root_match.group("root")
    stems = ("alpha", "beta", "gamma")
    expected_assignments = {
        f"- {stem}-document-worker owns {root}/{stem}.md" for stem in stems
    }
    actual_assignments = {
        line.strip()
        for line in contract.splitlines()
        if line.strip().startswith("- ")
    }
    if actual_assignments != expected_assignments:
        raise ValueError("document manager scaffold lacks the exact three leaf assignments")
    return (
        f"document_manager = await rlm({json.dumps(contract, ensure_ascii=False)}, "
        'name="document-manager")'
    )


def disclosed_document_spawn_action(prompt: str) -> str | None:
    """Build one answer-free, quoting-safe flat-document admission action."""

    assignments: dict[str, str] = {}
    for line in prompt.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- ") or ": Read " not in stripped:
            continue
        name, instruction = stripped[2:].split(": ", 1)
        match = re.fullmatch(r"(alpha|beta|gamma)-document-worker", name)
        if match is None:
            continue
        stem = match.group(1)
        if (
            f"/{stem}.md" not in instruction
            or "receiver_role='parent'" not in instruction
            or not instruction.endswith("stop.")
            or '"""' in instruction
        ):
            raise ValueError("document spawn scaffold contains an invalid child contract")
        if stem in assignments and assignments[stem] != instruction:
            raise ValueError("document spawn scaffold contains conflicting child contracts")
        assignments[stem] = instruction
    stems = ("alpha", "beta", "gamma")
    if not assignments:
        return None
    if set(assignments) != set(stems):
        raise ValueError("document spawn scaffold requires exactly three named contracts")
    calls = "\n".join(
        f"{stem}_worker = await rlm("
        f"{json.dumps(assignments[stem], ensure_ascii=False)}, "
        f'name="{stem}-document-worker")'
        for stem in stems
    )
    return calls


def disclosed_document_leaf_action(prompt: str) -> str | None:
    """Build the exact three-leaf action for a non-root document manager."""

    matches = [
        match.group("contract").strip()
        for match in DOCUMENT_MANAGER_CONTRACT_PATTERN.finditer(prompt)
    ]
    if not matches:
        return None
    if len(set(matches)) != 1:
        raise ValueError("document leaf scaffold contains conflicting manager contracts")
    root_match = re.search(
        r"You own the document directory (?P<root>/[^\s]+)\.", matches[0]
    )
    if root_match is None:
        raise ValueError("document leaf scaffold lacks one owned directory")
    root = root_match.group("root")
    stems = ("alpha", "beta", "gamma")
    declarations = "\n".join(
        f'{stem}_task = """Read {root}/{stem}.md using the CLI or IPython. '
        "Count words with Python str.split() over the complete file contents and count "
        "lines beginning exactly `## `. Send one JSON object with integer keys `words` "
        "and `h2` to your parent using await agent_message.send(json.dumps(result), "
        "receiver_role='parent'). After the delivery receipt succeeds, stop.\"\"\""
        for stem in stems
    )
    calls = "\n".join(
        f'{stem}_worker = await rlm({stem}_task, name="{stem}-document-worker")'
        for stem in stems
    )
    return f"{declarations}\n{calls}"


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


def disclosed_root_action_from_messages(messages: Any) -> str | None:
    """Extract the root spawn scaffold from structured Chat Completions messages."""

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
    prompt = "\n".join(fragments)
    if EXACT_ACTION_MARKER not in prompt:
        prompt = f"{EXACT_ACTION_MARKER}\n{prompt}"
    return disclosed_root_action(prompt)


def disclosed_document_free_actions_from_messages(
    messages: Any,
) -> dict[str, str] | None:
    """Extract all free-topology document actions from structured messages."""

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
    return disclosed_document_free_actions("\n".join(fragments))


def document_root_topology(code: str) -> str | None:
    """Classify only the delegation topology selected by a root IPython action."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        rlm_call_count = len(re.findall(r"\brlm\s*\(", code))
        matches = re.findall(
            r"\bname\s*=\s*(['\"])(?P<name>[^'\"]+)\1", code
        )
        names = [name for _, name in matches]
        if (
            rlm_call_count == 1
            and re.search(r"\bdocument_manager\s*=\s*await\s+rlm\s*\(", code)
        ):
            names = ["document-manager"]
        elif rlm_call_count != len(names):
            return None
    else:
        names = []
        rlm_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "rlm"
        ]
        assigned_document_manager_calls: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            value = (
                node.value.value if isinstance(node.value, ast.Await) else node.value
            )
            target = node.targets[0]
            if (
                isinstance(value, ast.Call)
                and value in rlm_calls
                and isinstance(target, ast.Name)
                and target.id == "document_manager"
            ):
                assigned_document_manager_calls.add(id(value))
        for node in rlm_calls:
            name = next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "name"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ),
                None,
            )
            if name is None and id(node) in assigned_document_manager_calls:
                name = "document-manager"
            if name is None:
                return None
            names.append(name)
    if not names:
        has_document_root = bool(
            re.search(r"/workspace/document-recursion/[^/'\"\s]+", code)
        )
        has_local_file_action = any(
            marker in code
            for marker in ("Path(", "read_text(", ".glob(", "open(", "listdir(")
        )
        has_explicit_direct_intent = bool(
            re.search(r"\b(?:direct inspection|directly inspect)\b", code, re.IGNORECASE)
        )
        names_all_documents = all(f"{stem}.md" in code for stem in ("alpha", "beta", "gamma"))
        direct_intent_action = (
            has_explicit_direct_intent
            and names_all_documents
            and "document-manager" not in code
            and not re.search(r"\bdelegate\b", code, re.IGNORECASE)
        )
        return (
            "direct"
            if has_document_root and (has_local_file_action or direct_intent_action)
            else None
        )
    if names == ["document-manager"]:
        return "hierarchical"
    if len(names) == 3 and set(names) == {
        "alpha-document-worker",
        "beta-document-worker",
        "gamma-document-worker",
    }:
        return "flat"
    return None


def rewrite_document_root_topology_response(
    body: bytes, *, canonical_code: str
) -> tuple[bytes, int, str | None, str | None, str]:
    """Repair mechanics only when the model selected the canonical topology."""

    expected = document_root_topology(canonical_code)
    if expected not in {"direct", "flat", "hierarchical"}:
        raise ValueError("document root topology scaffold requires a document action")
    return _rewrite_document_root_topology_response(
        body,
        canonical_codes={expected: canonical_code},
        expected=expected,
    )


def rewrite_document_root_free_topology_response(
    body: bytes, *, canonical_codes: dict[str, str]
) -> tuple[bytes, int, str | None, str | None, str]:
    """Normalize any one legal graph selected by a free-topology document policy."""

    if set(canonical_codes) != {"direct", "flat", "hierarchical"}:
        raise ValueError("free document topology scaffold requires three legal actions")
    if any(document_root_topology(code) != name for name, code in canonical_codes.items()):
        raise ValueError("free document topology actions do not match their labels")
    return _rewrite_document_root_topology_response(
        body,
        canonical_codes=canonical_codes,
        expected="free",
        reject_unclassified=True,
    )


def _rewrite_document_root_topology_response(
    body: bytes,
    *,
    canonical_codes: dict[str, str],
    expected: str,
    reject_unclassified: bool = False,
) -> tuple[bytes, int, str | None, str | None, str]:
    """Rewrite mechanics after, and only after, a recognized legal graph choice."""

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body, 0, None, None, expected
    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
        return body, 0, None, None, expected
    selected = None
    rewrites = 0
    for choice in payload["choices"]:
        message = choice.get("message") if isinstance(choice, dict) else None
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            continue
        tool_call = tool_calls[0]
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            function = tool_call
        if function.get("name") != "ipython":
            continue
        arguments = function.get("arguments")
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            continue
        code = parsed.get("code") if isinstance(parsed, dict) else None
        if not isinstance(code, str):
            continue
        selected = document_root_topology(code)
        canonical_code = canonical_codes.get(selected or "")
        if canonical_code is None:
            continue
        parsed["code"] = canonical_code
        function["arguments"] = json.dumps(
            parsed, separators=(",", ":"), ensure_ascii=False
        )
        rewrites += 1
    action_sha256 = None
    if rewrites and selected is not None:
        action_sha256 = hashlib.sha256(canonical_codes[selected].encode()).hexdigest()
    if not rewrites and selected is None and reject_unclassified:
        rejected = 0
        for choice in payload["choices"]:
            if not isinstance(choice, dict) or not isinstance(
                choice.get("message"), dict
            ):
                continue
            choice["message"] = {
                "role": "assistant",
                "content": (
                    "Choose exactly one legal document topology: direct, flat, "
                    "or hierarchical."
                ),
            }
            choice["finish_reason"] = "stop"
            rejected += 1
        if rejected:
            return (
                json.dumps(
                    payload, separators=(",", ":"), ensure_ascii=False
                ).encode(),
                rejected,
                None,
                None,
                expected,
            )
    if not rewrites:
        return body, 0, action_sha256, selected, expected
    return (
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(),
        rewrites,
        action_sha256,
        selected,
        expected,
    )


def disclosed_document_leaf_action_from_messages(messages: Any) -> str | None:
    """Extract the non-root document manager scaffold from structured messages."""

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
    return disclosed_document_leaf_action("\n".join(fragments))


def inline_evidence_from_messages(messages: Any) -> str | None:
    """Extract the already visible recursive-session evidence card."""

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
    marker = "Evidence contents:\n"
    matches = [fragment.split(marker, 1)[1].strip() for fragment in fragments if marker in fragment]
    if not matches:
        return None
    if len(set(matches)) != 1:
        raise ValueError("recursive coordinator prompt contains conflicting inline evidence")
    return matches[0]


def without_leaf_evidence_path(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove a leaf's deliberately non-runtime evidence path from its live prompt.

    The private bytes remain visible in the evidence card. Replacing only the exact
    label prevents a small policy from treating the opaque ownership label as a real
    filesystem path, while still requiring it to choose and execute the computation.
    """

    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("pathless leaf evidence requires chat messages")
    paths: list[str] = []

    def find(value: Any) -> None:
        if isinstance(value, str):
            for line in value.splitlines():
                if line.startswith(EVIDENCE_LABEL_MARKER):
                    label = line[len(EVIDENCE_LABEL_MARKER) :].strip()
                    if label:
                        paths.append(label)
        elif isinstance(value, list):
            for item in value:
                find(item)
        elif isinstance(value, dict):
            for item in value.values():
                find(item)

    find(messages)
    if not paths:
        raise ValueError("pathless leaf evidence lacks an evidence label")
    if len(set(paths)) != 1:
        raise ValueError("pathless leaf evidence contains conflicting labels")
    label = paths[0]
    if label == PATHLESS_INLINE_EVIDENCE_LABEL:
        return payload

    def rewrite(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(label, PATHLESS_INLINE_EVIDENCE_LABEL)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    return {**payload, "messages": rewrite(messages)}


def required_review_from_messages(messages: Any) -> str | None:
    """Extract the public operation assigned to one bounded leaf session."""

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
    matches = []
    for fragment in fragments:
        if REQUIRED_REVIEW_MARKER not in fragment:
            continue
        review = fragment.split(REQUIRED_REVIEW_MARKER, 1)[1].splitlines()[0].strip()
        if review:
            matches.append(review)
    if not matches:
        return None
    if len(set(matches)) != 1:
        raise ValueError("recursive coordinator prompt contains conflicting required reviews")
    return matches[0]


def leaf_compute_report_code(operation: str) -> str:
    """Return an answer-free, operation-grounded Python leaf action.

    This deliberately leaks the generic algorithm during the early curriculum while
    leaving the private evidence and resulting value to runtime computation.
    """

    if operation == "sum the top-level JSON integer list":
        body = "import json\nresult = sum(json.loads(INLINE_EVIDENCE))"
    elif operation == "sum the CSV amount column":
        body = (
            "import csv, io\n"
            "result = sum(int(row['amount']) for row in "
            "csv.DictReader(io.StringIO(INLINE_EVIDENCE)))"
        )
    elif operation == "count level-2 Markdown headings":
        body = (
            "result = sum(line.startswith('## ') "
            "for line in INLINE_EVIDENCE.splitlines())"
        )
    elif operation == "count ERROR-level log lines":
        body = (
            "result = sum(line.startswith('ERROR ') "
            "for line in INLINE_EVIDENCE.splitlines())"
        )
    elif operation == "count top-level sync and async function definitions":
        body = (
            "import ast\n"
            "tree = ast.parse(INLINE_EVIDENCE)\n"
            "result = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) "
            "for node in tree.body)"
        )
    elif operation == "return the largest JSON integer value":
        body = "import json\nresult = max(json.loads(INLINE_EVIDENCE).values())"
    else:
        evidence_key = re.fullmatch(
            r"report the integer stored under ([A-Za-z_][A-Za-z0-9_]*) "
            r"in the private evidence card",
            operation,
        )
        word_count = re.fullmatch(r"count exact '([^']+)' tokens", operation)
        if evidence_key is not None:
            body = (
                "import json\n"
                f"key = {evidence_key.group(1)!r}\n"
                "result = json.loads(INLINE_EVIDENCE)[key]"
            )
        elif word_count is not None:
            body = (
                f"keyword = {word_count.group(1)!r}\n"
                "result = INLINE_EVIDENCE.split().count(keyword)"
            )
        else:
            raise ValueError(f"unsupported leaf compute-report operation: {operation}")
    return (
        f"{body}\n"
        "await agent_message.send(str(result), receiver_role='parent')"
    )


def child_compute_method_hint(operation: str) -> str:
    """Provide an answer-free Python method hint for an early child curriculum."""

    hints = {
        "sum the top-level JSON integer list": (
            "Parse INLINE_EVIDENCE with json.loads, then sum the resulting integer list."
        ),
        "sum the CSV amount column": (
            "Parse INLINE_EVIDENCE as CSV text with csv.DictReader and sum the integer "
            "amount field from every data row."
        ),
        "count level-2 Markdown headings": (
            "Split INLINE_EVIDENCE into lines and count only lines that start exactly "
            "with '## '."
        ),
        "count ERROR-level log lines": (
            "Split INLINE_EVIDENCE into lines and count only lines that start exactly "
            "with 'ERROR '."
        ),
        "count top-level sync and async function definitions": (
            "Treat INLINE_EVIDENCE as Python source text: parse it with ast.parse and "
            "count FunctionDef and AsyncFunctionDef nodes only in tree.body. Do not "
            "inspect globals or execute the supplied source."
        ),
        "return the largest JSON integer value": (
            "Parse INLINE_EVIDENCE with json.loads and take the maximum of the mapping's "
            "integer values, not its keys or invented variable names."
        ),
    }
    if operation in hints:
        return hints[operation]
    evidence_key = re.fullmatch(
        r"report the integer stored under ([A-Za-z_][A-Za-z0-9_]*) "
        r"in the private evidence card",
        operation,
    )
    if evidence_key is not None:
        return (
            "Parse INLINE_EVIDENCE with json.loads and return the integer stored under "
            f"the exact key {evidence_key.group(1)!r}."
        )
    word_count = re.fullmatch(r"count exact '([^']+)' tokens", operation)
    if word_count is not None:
        return (
            "Split INLINE_EVIDENCE on whitespace and count tokens exactly equal to "
            f"{word_count.group(1)!r}; do not use substring counting."
        )
    raise ValueError(f"unsupported child compute operation: {operation}")


def latest_ipython_tool_failed(messages: Any) -> bool:
    """Detect whether the most recent IPython result is an execution failure."""

    if not isinstance(messages, list):
        return False
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        # OpenAI-compatible clients may identify a tool result only by
        # tool_call_id. Typed-compute turns expose IPython as the sole tool, so
        # an omitted name is still unambiguous; an explicit different name is not.
        if message.get("name") not in (None, "ipython"):
            return False
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        if not isinstance(content, str):
            return False
        return any(
            marker in content
            for marker in (
                "Traceback (most recent call last)",
                "SyntaxError:",
                "FileNotFoundError:",
                "NameError:",
                "TypeError:",
                "ValueError:",
            )
        )
    return False


def should_run_typed_compute(
    session_sha256: str,
    compute_hashes: set[str],
    compute_attempts: dict[str, int],
    messages: Any,
) -> bool:
    """Start computation once and retry explicit IPython failures at most twice."""

    if session_sha256 not in compute_hashes:
        return True
    return latest_ipython_tool_failed(messages) and compute_attempts.get(
        session_sha256, 0
    ) < 3


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


def force_parent_return_compute_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """Constrain the first recursive turn to one model-authored IPython computation."""

    tools = payload.get("tools")
    if not isinstance(tools, list):
        raise ValueError("typed parent-return computation requires Chat Completions tools")
    ipython_tool = next(
        (
            tool
            for tool in tools
            if isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and tool["function"].get("name") == "ipython"
        ),
        None,
    )
    if ipython_tool is None:
        raise ValueError("typed parent-return computation requires the IPython tool")
    function = ipython_tool["function"]
    compute_tool = {
        **ipython_tool,
        "function": {
            **function,
            "description": (
                "Use Python once to compute the requested result from the inline evidence. "
                "The harness prebinds the visible evidence as the string INLINE_EVIDENCE; "
                "use that variable and never Path, open, or filesystem access. Do not "
                "message the parent in this call; the next turn provides the typed "
                "parent-return action. End the cell with the computed result."
            ),
        },
    }
    rewritten = {
        **payload,
        "stream": False,
        "tools": [compute_tool],
        "tool_choice": {"type": "function", "function": {"name": "ipython"}},
        "parallel_tool_calls": False,
    }
    rewritten.pop("stream_options", None)
    return rewritten


def force_child_compute_report_schema(
    payload: dict[str, Any], *, operation: str | None = None
) -> dict[str, Any]:
    """Constrain one answer-free compute cell whose runtime value is reported."""

    rewritten = (
        force_parent_return_compute_schema(payload)
        if operation is None
        else force_ipython_code_schema(payload, leaf_compute_report_code(operation))
    )
    rewritten = {**rewritten, "stream": False}
    rewritten.pop("stream_options", None)
    rewritten["temperature"] = 0.0
    function = next(
        tool["function"]
        for tool in rewritten["tools"]
        if tool["function"].get("name") == "ipython"
    )
    function["description"] = (
        "Execute one answer-free Python computation over INLINE_EVIDENCE. The disclosed "
        "program contains the operation and parent routing but no result; runtime "
        "evidence determines the reported value and the harness then ends the session."
        if operation is not None
        else (
            "Use Python once to compute the requested result from INLINE_EVIDENCE. End "
            "the cell with the computed result; the harness routes that value to the "
            "direct parent and ends the session."
        )
    )
    if operation is not None:
        function["description"] += f" Method hint: {child_compute_method_hint(operation)}"
    return rewritten


def rewrite_ipython_literal_newlines_response(
    body: bytes,
    *,
    inline_evidence: str | None = None,
    preserve_parent_send: bool = False,
    report_final_value_to_parent: bool = False,
) -> tuple[bytes, int, str | None]:
    """Ground the compute cell and repair doubly escaped parse-blocking newlines."""

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body, 0, None
    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
        return body, 0, None
    rewrites = 0
    action_sha256 = None
    for choice in payload["choices"]:
        message = choice.get("message") if isinstance(choice, dict) else None
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            if not isinstance(function, dict) or function.get("name") != "ipython":
                continue
            arguments = function.get("arguments")
            try:
                parsed_arguments = (
                    json.loads(arguments) if isinstance(arguments, str) else arguments
                )
            except json.JSONDecodeError:
                continue
            code = (
                parsed_arguments.get("code")
                if isinstance(parsed_arguments, dict)
                else None
            )
            if not isinstance(code, str):
                continue
            repaired = code
            if inline_evidence is not None:
                repaired = PATH_READ_TEXT_PATTERN.sub("INLINE_EVIDENCE", repaired)
                repaired = (
                    f"INLINE_EVIDENCE = {json.dumps(inline_evidence, ensure_ascii=False)}\n"
                    f"{repaired}"
                )
            try:
                tree = ast.parse(repaired)
            except SyntaxError:
                clean_prefix = MODEL_TOOL_CONTAMINATION_PATTERN.split(
                    repaired, maxsplit=1
                )[0].rstrip()
                try:
                    tree = ast.parse(clean_prefix)
                except SyntaxError:
                    repaired = repaired.replace("\\r\\n", "\n").replace(
                        "\\n", "\n"
                    )
                    try:
                        tree = ast.parse(repaired)
                    except SyntaxError:
                        clean_prefix = MODEL_TOOL_CONTAMINATION_PATTERN.split(
                            repaired, maxsplit=1
                        )[0].rstrip()
                        if not clean_prefix:
                            continue
                        try:
                            tree = ast.parse(clean_prefix)
                        except SyntaxError:
                            continue
                        repaired = clean_prefix
                else:
                    repaired = clean_prefix
            if preserve_parent_send:
                await_transformer = _AwaitBareParentSend()
                tree = await_transformer.visit(tree)
                if await_transformer.rewrites:
                    ast.fix_missing_locations(tree)
                    repaired = ast.unparse(tree)
            else:
                send_transformer = _ComputeSendToValue()
                tree = send_transformer.visit(tree)
                if send_transformer.rewrites:
                    ast.fix_missing_locations(tree)
                    repaired = ast.unparse(tree)
            if report_final_value_to_parent:
                if not _report_final_value_to_parent(tree):
                    continue
                repaired = ast.unparse(tree)
            if repaired == code:
                continue
            parsed_arguments["code"] = repaired
            function["arguments"] = json.dumps(
                parsed_arguments, separators=(",", ":"), ensure_ascii=False
            )
            rewrites += 1
            action_sha256 = hashlib.sha256(repaired.encode()).hexdigest()
    if not rewrites:
        return body, 0, None
    return (
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(),
        rewrites,
        action_sha256,
    )


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


def chat_completion_to_sse(body: bytes) -> bytes:
    """Bridge one non-streaming chat completion back to an OpenAI SSE stream."""

    payload = json.loads(body)
    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
        raise ValueError("chat completion bridge requires choices")
    choices = []
    for choice in payload["choices"]:
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ValueError("chat completion bridge requires complete messages")
        message = choice["message"]
        delta = {
            key: message[key]
            for key in ("role", "content", "reasoning_content", "tool_calls")
            if key in message
        }
        choices.append(
            {
                "index": choice.get("index", 0),
                "delta": delta,
                "logprobs": choice.get("logprobs"),
                "finish_reason": choice.get("finish_reason"),
            }
        )
    chunk = {
        key: payload[key]
        for key in ("id", "created", "model", "system_fingerprint", "usage")
        if key in payload
    }
    chunk.update({"object": "chat.completion.chunk", "choices": choices})
    encoded = json.dumps(chunk, separators=(",", ":"), ensure_ascii=False).encode()
    return b"data: " + encoded + b"\n\ndata: [DONE]\n\n"


def synthetic_chat_stop_response(
    *, model: str, sequence: int, content: str = "Return delivered."
) -> bytes:
    """Return one terminal non-tool turn after a typed parent return was delivered."""

    payload = {
        "id": f"typed-parent-return-stop-{sequence}",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return json.dumps(payload, separators=(",", ":")).encode()


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
        document_coordinator_token_ids: list[int] | None = None,
        root_depth_token_ids: list[int] | None = None,
        depth_default_child: bool = False,
        leak_coordinator_exact_action: bool = False,
        leak_document_manager_exact_action: bool = False,
        leak_coordinator_return_action: bool = False,
        typed_coordinator_return: bool = False,
        leak_child_exact_action: bool = False,
        strip_child_tool_choice: bool = False,
        strip_coordinator_tool_choice: bool = False,
        root_coordinator_contract: bool = False,
        leaf_reporter_contract: bool = False,
        leaf_inline_evidence: bool = False,
        leaf_compute_report_scaffold: bool = False,
        document_leaf_compute_report_scaffold: bool = False,
        document_manager_fanin_scaffold: bool = False,
        document_manager_wait_scaffold: bool = False,
        document_manager_termination_scaffold: bool = False,
        document_root_report_relay_scaffold: bool = False,
        document_root_topology_normalization_scaffold: bool = False,
        document_root_flat_fanin_scaffold: bool = False,
        typed_child_report: bool = False,
        child_authored_compute: bool = False,
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
        self.document_coordinator_token_ids = document_coordinator_token_ids or []
        self.root_depth_token_ids = root_depth_token_ids or []
        self.depth_default_child = depth_default_child
        self.tokenizer = tokenizer
        self.leak_coordinator_exact_action = leak_coordinator_exact_action
        self.leak_document_manager_exact_action = leak_document_manager_exact_action
        self.leak_coordinator_return_action = leak_coordinator_return_action
        self.typed_coordinator_return = typed_coordinator_return
        self.leak_child_exact_action = leak_child_exact_action
        self.strip_child_tool_choice = strip_child_tool_choice
        self.strip_coordinator_tool_choice = strip_coordinator_tool_choice
        self.root_coordinator_contract = root_coordinator_contract
        self.leaf_reporter_contract = leaf_reporter_contract
        self.leaf_inline_evidence = leaf_inline_evidence
        self.leaf_compute_report_scaffold = leaf_compute_report_scaffold
        self.document_leaf_compute_report_scaffold = document_leaf_compute_report_scaffold
        self.document_manager_fanin_scaffold = document_manager_fanin_scaffold
        self.document_manager_wait_scaffold = (
            document_manager_wait_scaffold or document_manager_fanin_scaffold
        )
        self.document_manager_termination_scaffold = (
            document_manager_termination_scaffold or document_manager_fanin_scaffold
        )
        self.document_root_report_relay_scaffold = (
            document_root_report_relay_scaffold or document_manager_fanin_scaffold
        )
        self.document_root_topology_normalization_scaffold = (
            document_root_topology_normalization_scaffold
        )
        self.document_root_flat_fanin_scaffold = document_root_flat_fanin_scaffold
        self.typed_child_report = typed_child_report
        self.child_authored_compute = child_authored_compute
        self.client: ClientSession | None = None
        self.sequence = 0
        self.leaked_session_hashes: dict[str, set[str]] = {
            "coordinator": set(),
            "document_manager": set(),
            "coordinator_return": set(),
            "child": set(),
        }
        self.completed_typed_return_hashes: set[str] = set()
        self.completed_typed_child_report_hashes: set[str] = set()
        self.completed_leaf_compute_report_hashes: set[str] = set()
        self.typed_return_compute_hashes: set[str] = set()
        self.typed_return_compute_attempts: dict[str, int] = {}
        self.typed_child_report_compute_hashes: set[str] = set()
        self.typed_child_report_compute_attempts: dict[str, int] = {}
        self.document_report_state: dict[
            tuple[str, str], dict[str, dict[str, int]]
        ] = {}
        self.root_flat_passive_hashes: set[str] = set()
        self.root_document_topologies: dict[str, str] = {}

    def accumulate_document_reports(
        self,
        scope: str,
        session_sha256: str,
        reports: dict[str, dict[str, int]],
    ) -> dict[str, dict[str, int]]:
        """Retain reports across asynchronous child-message resumptions."""

        accumulated = self.document_report_state.setdefault((scope, session_sha256), {})
        for stem, report in reports.items():
            if stem in accumulated and accumulated[stem] != report:
                raise ValueError(f"document {scope} received conflicting {stem} reports")
            accumulated[stem] = report
        return dict(accumulated)

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
                if (
                    event.get("mode") == "forwarded_typed_coordinator_return"
                    and isinstance(session_sha256, str)
                ):
                    self.completed_typed_return_hashes.add(session_sha256)
                if (
                    event.get("mode") == "forwarded_typed_child_report"
                    and isinstance(session_sha256, str)
                ):
                    self.completed_typed_child_report_hashes.add(session_sha256)
                if (
                    event.get("mode") == "forwarded_leaf_compute_report"
                    and isinstance(session_sha256, str)
                ):
                    self.completed_leaf_compute_report_hashes.add(session_sha256)
                if (
                    str(event.get("mode", "")).startswith("forwarded_typed_return_compute")
                    and isinstance(session_sha256, str)
                ):
                    self.typed_return_compute_hashes.add(session_sha256)
                    self.typed_return_compute_attempts[session_sha256] = (
                        self.typed_return_compute_attempts.get(session_sha256, 0) + 1
                    )
                if (
                    str(event.get("mode", "")).startswith(
                        "forwarded_typed_child_report_compute"
                    )
                    and isinstance(session_sha256, str)
                ):
                    self.typed_child_report_compute_hashes.add(session_sha256)
                    self.typed_child_report_compute_attempts[session_sha256] = (
                        self.typed_child_report_compute_attempts.get(session_sha256, 0)
                        + 1
                    )
                if (
                    event.get("mode") == "document_root_flat_fanin_passive"
                    and isinstance(session_sha256, str)
                ):
                    self.root_flat_passive_hashes.add(session_sha256)
                mode = str(event.get("mode", ""))
                topology_prefix = "forwarded_document_root_topology_normalized_"
                if (
                    mode.startswith(topology_prefix)
                    and mode.removeprefix(topology_prefix)
                    in {"direct", "flat", "hierarchical"}
                    and isinstance(session_sha256, str)
                ):
                    self.root_document_topologies[session_sha256] = mode.removeprefix(
                        topology_prefix
                    )
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
        document_report_stems: tuple[str, ...] | None = None,
        document_root_topology: str | None = None,
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
        if document_report_stems is not None:
            event["document_report_stems"] = list(document_report_stems)
        if document_root_topology is not None:
            event["document_root_topology"] = document_root_topology
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
            document_coordinator_token_ids=self.document_coordinator_token_ids,
            root_depth_token_ids=self.root_depth_token_ids,
            depth_default_child=self.depth_default_child,
        )
        if (
            self.root_coordinator_contract
            and endpoint == "/v1/chat/completions"
            and role == "coordinator"
            and is_root_coordinator_request(payload)
        ):
            routed = with_root_coordinator_contract(routed)
        if (
            self.leaf_reporter_contract
            and endpoint == "/v1/chat/completions"
            and role == "child"
        ):
            routed = with_leaf_reporter_contract(routed)
        if (
            self.leaf_inline_evidence
            and endpoint == "/v1/chat/completions"
            and role == "child"
        ):
            routed = without_leaf_evidence_path(routed)
        client_requested_stream = payload.get("stream") is True
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
        recursive_coordinator_chat = (
            endpoint == "/v1/chat/completions"
            and role == "coordinator"
            and _contains_marker(payload.get("messages"), RECURSIVE_COORDINATOR_HEADER)
        )
        typed_child_chat = endpoint == "/v1/chat/completions" and role == "child"
        document_leaf_path = (
            document_leaf_path_from_messages(payload.get("messages"))
            if self.document_leaf_compute_report_scaffold and typed_child_chat
            else None
        )
        document_manager_chat = (
            endpoint == "/v1/chat/completions"
            and role == "coordinator"
            and not is_root_coordinator_request(payload)
            and _contains_marker(payload.get("messages"), DOCUMENT_COORDINATOR_HEADER)
            and not _contains_marker(
                payload.get("messages"), FREE_DOCUMENT_TOPOLOGY_HEADER
            )
        )
        leaf_document_reports = (
            document_leaf_report_from_messages(
                payload.get("messages"), document_leaf_path
            )
            if document_leaf_path is not None
            else {}
        )
        session_document_topology = (
            self.root_document_topologies.get(session_sha256)
            if session_sha256 is not None
            else None
        )
        if (
            document_leaf_path is not None
            and session_sha256 is not None
            and session_document_topology == "flat"
        ):
            self.accumulate_document_reports(
                "root_flat",
                session_sha256,
                leaf_document_reports,
            )
        manager_reports = (
            document_manager_reports_from_messages(payload.get("messages"))
            if (
                self.document_manager_fanin_scaffold
                or self.document_manager_wait_scaffold
            )
            and document_manager_chat
            else {}
        )
        root_manager_report = (
            document_root_manager_report_from_messages(payload.get("messages"))
            if self.document_root_report_relay_scaffold
            and endpoint == "/v1/chat/completions"
            and role == "coordinator"
            and is_root_coordinator_request(payload)
            else None
        )
        root_direct_result = (
            document_root_direct_result_from_messages(payload.get("messages"))
            if (
                endpoint == "/v1/chat/completions"
                and role == "coordinator"
                and session_sha256 is not None
                and self.root_document_topologies.get(session_sha256) == "direct"
            )
            else None
        )
        root_flat_reports = (
            self.accumulate_document_reports(
                "root_flat",
                session_sha256,
                document_manager_reports_from_messages(payload.get("messages")),
            )
            if (
                self.document_root_flat_fanin_scaffold
                and endpoint == "/v1/chat/completions"
                and role == "coordinator"
                and is_root_coordinator_request(payload)
                and not _contains_marker(
                    payload.get("messages"), DOCUMENT_COORDINATOR_HEADER
                )
            )
            else {}
        )
        root_flat_admission_complete = bool(root_flat_reports) or (
            self.document_root_flat_fanin_scaffold
            and endpoint == "/v1/chat/completions"
            and role == "coordinator"
            and is_root_coordinator_request(payload)
            and all(
                _contains_marker(
                    payload.get("messages"), f"{stem}_worker = await rlm("
                )
                for stem in ("alpha", "beta", "gamma")
            )
            and any(
                isinstance(message, dict) and message.get("role") == "tool"
                for message in payload.get("messages") or []
            )
        )
        if root_direct_result is not None:
            body = json.dumps(
                routed, separators=(",", ":"), ensure_ascii=False
            ).encode()
            response_body = synthetic_chat_stop_response(
                model=self.external_model,
                sequence=self.sequence,
                content=json.dumps(root_direct_result, separators=(",", ":")),
            )
            content_type = "application/json"
            if client_requested_stream:
                response_body = chat_completion_to_sse(response_body)
                content_type = "text/event-stream"
            self._audit(
                role=role,
                endpoint=endpoint,
                body=body,
                status=200,
                mode="document_root_direct_result_finalized",
                session_sha256=session_sha256,
            )
            return web.Response(body=response_body, content_type=content_type)
        if root_flat_admission_complete and len(root_flat_reports) == 3:
            body = json.dumps(
                routed, separators=(",", ":"), ensure_ascii=False
            ).encode()
            response_body = synthetic_chat_stop_response(
                model=self.external_model,
                sequence=self.sequence,
                content=json.dumps(
                    document_manager_parent_report(root_flat_reports),
                    separators=(",", ":"),
                ),
            )
            content_type = "application/json"
            if client_requested_stream:
                response_body = chat_completion_to_sse(response_body)
                content_type = "text/event-stream"
            self._audit(
                role=role,
                endpoint=endpoint,
                body=body,
                status=200,
                mode="document_root_flat_fanin_finalized",
                session_sha256=session_sha256,
                document_report_stems=tuple(sorted(root_flat_reports)),
            )
            return web.Response(body=response_body, content_type=content_type)
        if (
            root_flat_admission_complete
            and len(root_flat_reports) < 3
            and session_sha256 not in self.root_flat_passive_hashes
        ):
            self.root_flat_passive_hashes.add(session_sha256)
            body = json.dumps(
                routed, separators=(",", ":"), ensure_ascii=False
            ).encode()
            response_body = synthetic_chat_stop_response(
                model=self.external_model,
                sequence=self.sequence,
                content="Waiting for the remaining document reports.",
            )
            content_type = "application/json"
            if client_requested_stream:
                response_body = chat_completion_to_sse(response_body)
                content_type = "text/event-stream"
            self._audit(
                role=role,
                endpoint=endpoint,
                body=body,
                status=200,
                mode="document_root_flat_fanin_passive",
                session_sha256=session_sha256,
                document_report_stems=tuple(sorted(root_flat_reports)),
            )
            return web.Response(body=response_body, content_type=content_type)
        if root_manager_report is not None:
            body = json.dumps(
                routed, separators=(",", ":"), ensure_ascii=False
            ).encode()
            response_body = synthetic_chat_stop_response(
                model=self.external_model,
                sequence=self.sequence,
                content=json.dumps(root_manager_report, separators=(",", ":")),
            )
            content_type = "application/json"
            if client_requested_stream:
                response_body = chat_completion_to_sse(response_body)
                content_type = "text/event-stream"
            self._audit(
                role=role,
                endpoint=endpoint,
                body=body,
                status=200,
                mode="document_root_manager_report_finalized",
                session_sha256=session_sha256,
            )
            return web.Response(body=response_body, content_type=content_type)
        if (
            self.document_manager_termination_scaffold
            and document_manager_chat
            and document_manager_parent_report_completed(payload.get("messages"))
        ):
            body = json.dumps(
                routed, separators=(",", ":"), ensure_ascii=False
            ).encode()
            response_body = synthetic_chat_stop_response(
                model=self.external_model,
                sequence=self.sequence,
                content="Document manager report delivered.",
            )
            content_type = "application/json"
            if client_requested_stream:
                response_body = chat_completion_to_sse(response_body)
                content_type = "text/event-stream"
            self._audit(
                role=role,
                endpoint=endpoint,
                body=body,
                status=200,
                mode="document_manager_report_session_terminated",
                session_sha256=session_sha256,
            )
            return web.Response(body=response_body, content_type=content_type)
        manager_admission_complete = document_manager_chat and (
            _contains_marker(payload.get("messages"), "alpha_worker = await rlm(")
            and any(
                isinstance(message, dict) and message.get("role") == "tool"
                for message in payload.get("messages") or []
            )
        )
        if (
            self.document_manager_wait_scaffold
            and manager_admission_complete
            and len(manager_reports) < 3
        ):
            body = json.dumps(
                routed, separators=(",", ":"), ensure_ascii=False
            ).encode()
            response_body = synthetic_chat_stop_response(
                model=self.external_model,
                sequence=self.sequence,
                content="Waiting for the remaining document reports.",
            )
            content_type = "application/json"
            if client_requested_stream:
                response_body = chat_completion_to_sse(response_body)
                content_type = "text/event-stream"
            self._audit(
                role=role,
                endpoint=endpoint,
                body=body,
                status=200,
                mode="document_manager_fanin_session_passive",
                session_sha256=session_sha256,
            )
            return web.Response(body=response_body, content_type=content_type)
        if (
            document_leaf_path is not None
            and document_leaf_report_completed(payload.get("messages"))
        ):
            body = json.dumps(
                routed, separators=(",", ":"), ensure_ascii=False
            ).encode()
            response_body = synthetic_chat_stop_response(
                model=self.external_model,
                sequence=self.sequence,
                content="Document report delivered.",
            )
            content_type = "application/json"
            if client_requested_stream:
                response_body = chat_completion_to_sse(response_body)
                content_type = "text/event-stream"
            self._audit(
                role=role,
                endpoint=endpoint,
                body=body,
                status=200,
                mode="document_leaf_report_session_terminated",
                session_sha256=session_sha256,
                document_report_stems=tuple(sorted(leaf_document_reports)),
                document_root_topology=session_document_topology,
            )
            return web.Response(body=response_body, content_type=content_type)
        if (
            self.root_coordinator_contract
            and endpoint == "/v1/chat/completions"
            and role == "coordinator"
            and is_incomplete_document_manager_wait_request(payload)
        ):
            body = json.dumps(
                routed, separators=(",", ":"), ensure_ascii=False
            ).encode()
            response_body = synthetic_chat_stop_response(
                model=self.external_model,
                sequence=self.sequence,
                content="Waiting for the document manager's report.",
            )
            content_type = "application/json"
            if client_requested_stream:
                response_body = chat_completion_to_sse(response_body)
                content_type = "text/event-stream"
            self._audit(
                role=role,
                endpoint=endpoint,
                body=body,
                status=200,
                mode="document_manager_wait_session_passive",
                session_sha256=session_sha256,
            )
            return web.Response(body=response_body, content_type=content_type)
        if (
            self.root_coordinator_contract
            and self.typed_child_report
            and endpoint == "/v1/chat/completions"
            and role == "coordinator"
            and is_root_coordinator_request(payload)
            and is_incomplete_root_wait_request(payload)
            and session_sha256 is not None
        ):
            for _ in range(600):
                if session_sha256 in self.completed_typed_child_report_hashes:
                    break
                await asyncio.sleep(0.05)
            child_completed = (
                session_sha256 in self.completed_typed_child_report_hashes
            )
            body = json.dumps(
                routed, separators=(",", ":"), ensure_ascii=False
            ).encode()
            response_body = synthetic_chat_stop_response(
                model=self.external_model,
                sequence=self.sequence,
                content="Waiting for the child report.",
            )
            content_type = "application/json"
            if client_requested_stream:
                response_body = chat_completion_to_sse(response_body)
                content_type = "text/event-stream"
            self._audit(
                role=role,
                endpoint=endpoint,
                body=body,
                status=200,
                mode=(
                    "root_wait_child_report_completed"
                    if child_completed
                    else "root_wait_child_report_pending"
                ),
                session_sha256=session_sha256,
            )
            return web.Response(body=response_body, content_type=content_type)
        if (
            self.typed_coordinator_return
            and recursive_coordinator_chat
            and session_sha256 in self.completed_typed_return_hashes
        ):
            body = json.dumps(
                routed, separators=(",", ":"), ensure_ascii=False
            ).encode()
            response_body = synthetic_chat_stop_response(
                model=self.external_model, sequence=self.sequence
            )
            content_type = "application/json"
            if client_requested_stream:
                response_body = chat_completion_to_sse(response_body)
                content_type = "text/event-stream"
            self._audit(
                role=role,
                endpoint=endpoint,
                body=body,
                status=200,
                mode="typed_return_session_terminated",
                session_sha256=session_sha256,
            )
            return web.Response(body=response_body, content_type=content_type)
        if (
            self.typed_child_report
            and typed_child_chat
            and session_sha256 in self.completed_typed_child_report_hashes
        ):
            body = json.dumps(
                routed, separators=(",", ":"), ensure_ascii=False
            ).encode()
            response_body = synthetic_chat_stop_response(
                model=self.external_model, sequence=self.sequence
            )
            content_type = "application/json"
            if client_requested_stream:
                response_body = chat_completion_to_sse(response_body)
                content_type = "text/event-stream"
            self._audit(
                role=role,
                endpoint=endpoint,
                body=body,
                status=200,
                mode="typed_child_report_session_terminated",
                session_sha256=session_sha256,
            )
            return web.Response(body=response_body, content_type=content_type)
        if (
            self.leaf_compute_report_scaffold
            and typed_child_chat
            and session_sha256 in self.completed_leaf_compute_report_hashes
        ):
            body = json.dumps(
                routed, separators=(",", ":"), ensure_ascii=False
            ).encode()
            response_body = synthetic_chat_stop_response(
                model=self.external_model, sequence=self.sequence
            )
            content_type = "application/json"
            if client_requested_stream:
                response_body = chat_completion_to_sse(response_body)
                content_type = "text/event-stream"
            self._audit(
                role=role,
                endpoint=endpoint,
                body=body,
                status=200,
                mode="leaf_compute_report_session_terminated",
                session_sha256=session_sha256,
            )
            return web.Response(body=response_body, content_type=content_type)
        forced_chat_scope = None
        forced_chat_action_sha256 = None
        typed_compute_scope = False
        typed_compute_action_sha256 = None
        compute_inline_evidence = None
        typed_return_scope = False
        typed_return_action_sha256 = None
        typed_child_report_scope = False
        typed_child_report_action_sha256 = None
        typed_child_report_compute_scope = False
        typed_child_report_compute_action_sha256 = None
        typed_child_report_inline_evidence = None
        leaf_inline_evidence_scope = False
        leaf_inline_evidence_value = None
        leaf_inline_evidence_action_sha256 = None
        leaf_compute_report_scope = False
        leaf_compute_report_action_sha256 = None
        root_finalization_scope = False
        root_topology_scope = False
        root_topology_free_scope = False
        root_topology_canonical_code = None
        root_topology_canonical_codes = None
        root_topology_action_sha256 = None
        root_topology_selected = None
        root_topology_expected = None
        if (
            self.document_root_topology_normalization_scaffold
            and endpoint == "/v1/chat/completions"
            and role == "coordinator"
            and is_root_coordinator_request(payload)
            and not any(
                isinstance(message, dict)
                and message.get("role") in {"assistant", "tool"}
                for message in payload.get("messages") or []
            )
        ):
            root_topology_canonical_codes = (
                disclosed_document_free_actions_from_messages(payload.get("messages"))
            )
            if root_topology_canonical_codes is not None:
                root_topology_scope = True
                root_topology_free_scope = True
                routed = {**routed, "stream": False}
                routed.pop("stream_options", None)
            else:
                root_topology_canonical_code = disclosed_root_action_from_messages(
                    payload.get("messages")
                )
            if root_topology_canonical_code is not None:
                root_topology_scope = True
                routed = {**routed, "stream": False}
                routed.pop("stream_options", None)
        if self.document_manager_fanin_scaffold and len(manager_reports) == 3:
            code = document_manager_parent_report_code(manager_reports)
            forced_chat_scope = "document_manager_fanin"
            routed = force_ipython_code_schema(routed, code)
            forced_chat_action_sha256 = hashlib.sha256(code.encode()).hexdigest()
        if document_leaf_path is not None:
            code = document_leaf_compute_report_code(document_leaf_path)
            forced_chat_scope = "document_leaf_report"
            routed = force_ipython_code_schema(routed, code)
            forced_chat_action_sha256 = hashlib.sha256(code.encode()).hexdigest()
        if (
            endpoint == "/v1/chat/completions"
            and role == "coordinator"
            and self.leak_document_manager_exact_action
            and not is_root_coordinator_request(payload)
            and _contains_marker(payload.get("messages"), DOCUMENT_COORDINATOR_HEADER)
        ):
            code = disclosed_document_leaf_action_from_messages(payload.get("messages"))
            if code is not None:
                forced_chat_scope = "document_manager"
                if session_sha256 is None:
                    body = json.dumps(
                        routed, separators=(",", ":"), ensure_ascii=False
                    ).encode()
                    self._audit(
                        role=role,
                        endpoint=endpoint,
                        body=body,
                        status=400,
                        mode="leak_rejected_missing_document_manager_session",
                    )
                    return web.json_response(
                        {"error": "exact document manager action leak requires x-session-id"},
                        status=400,
                    )
                if session_sha256 in self.leaked_session_hashes[forced_chat_scope]:
                    forced_chat_scope = None
                else:
                    self.leaked_session_hashes[forced_chat_scope].add(session_sha256)
                    routed = force_ipython_code_schema(routed, code)
                    forced_chat_action_sha256 = hashlib.sha256(code.encode()).hexdigest()
        if (
            self.root_coordinator_contract
            and self.typed_child_report
            and endpoint == "/v1/chat/completions"
            and role == "coordinator"
            and is_root_coordinator_request(payload)
            and session_sha256 in self.completed_typed_child_report_hashes
        ):
            root_finalization_scope = True
            routed = force_root_json_finalization(routed)
        if (
            endpoint == "/v1/chat/completions"
            and role == "coordinator"
            and self.leak_coordinator_exact_action
            and is_root_coordinator_request(payload)
        ):
            code = disclosed_root_action_from_messages(payload.get("messages"))
            if code is not None:
                forced_chat_scope = "coordinator"
                if session_sha256 is None:
                    body = json.dumps(
                        routed, separators=(",", ":"), ensure_ascii=False
                    ).encode()
                    self._audit(
                        role=role,
                        endpoint=endpoint,
                        body=body,
                        status=400,
                        mode="leak_rejected_missing_coordinator_session",
                    )
                    return web.json_response(
                        {"error": "exact coordinator action leak requires x-session-id"},
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
            and recursive_coordinator_chat
        ):
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
            if should_run_typed_compute(
                session_sha256,
                self.typed_return_compute_hashes,
                self.typed_return_compute_attempts,
                payload.get("messages"),
            ):
                typed_compute_scope = True
                compute_inline_evidence = inline_evidence_from_messages(
                    payload.get("messages")
                )
                if compute_inline_evidence is None:
                    raise ValueError("typed parent-return computation lacks inline evidence")
                self.typed_return_compute_hashes.add(session_sha256)
                self.typed_return_compute_attempts[session_sha256] = (
                    self.typed_return_compute_attempts.get(session_sha256, 0) + 1
                )
                routed = force_parent_return_compute_schema(routed)
            elif session_sha256 in self.leaked_session_hashes["coordinator_return"]:
                typed_return_scope = False
            else:
                typed_return_scope = True
                self.leaked_session_hashes["coordinator_return"].add(session_sha256)
                routed = force_typed_parent_return_schema(routed)
        if self.typed_child_report and typed_child_chat:
            if session_sha256 is None:
                body = json.dumps(
                    routed, separators=(",", ":"), ensure_ascii=False
                ).encode()
                self._audit(
                    role=role,
                    endpoint=endpoint,
                    body=body,
                    status=400,
                    mode="typed_child_report_rejected_missing_session",
                )
                return web.json_response(
                    {"error": "typed child report requires x-session-id"},
                    status=400,
                )
            if should_run_typed_compute(
                session_sha256,
                self.typed_child_report_compute_hashes,
                self.typed_child_report_compute_attempts,
                payload.get("messages"),
            ):
                typed_child_report_compute_scope = True
                typed_child_report_inline_evidence = inline_evidence_from_messages(
                    payload.get("messages")
                )
                if typed_child_report_inline_evidence is None:
                    raise ValueError("typed child-report computation lacks inline evidence")
                child_operation = required_review_from_messages(payload.get("messages"))
                if child_operation is None:
                    raise ValueError("typed child-report computation lacks required review")
                self.typed_child_report_compute_hashes.add(session_sha256)
                self.typed_child_report_compute_attempts[session_sha256] = (
                    self.typed_child_report_compute_attempts.get(session_sha256, 0) + 1
                )
                routed = force_child_compute_report_schema(
                    routed,
                    operation=None if self.child_authored_compute else child_operation,
                )
            else:
                typed_child_report_scope = True
                routed = force_typed_parent_return_schema(routed)
        elif self.leaf_compute_report_scaffold and typed_child_chat:
            if session_sha256 is None:
                body = json.dumps(
                    routed, separators=(",", ":"), ensure_ascii=False
                ).encode()
                self._audit(
                    role=role,
                    endpoint=endpoint,
                    body=body,
                    status=400,
                    mode="leaf_compute_report_rejected_missing_session",
                )
                return web.json_response(
                    {"error": "leaf compute-report scaffold requires x-session-id"},
                    status=400,
                )
            leaf_inline_evidence_value = inline_evidence_from_messages(
                payload.get("messages")
            )
            operation = required_review_from_messages(payload.get("messages"))
            if leaf_inline_evidence_value is None:
                raise ValueError("leaf compute-report scaffold lacks inline evidence")
            if operation is None:
                raise ValueError("leaf compute-report scaffold lacks required review")
            code = leaf_compute_report_code(operation)
            leaf_compute_report_scope = True
            routed = force_ipython_code_schema(routed, code)
            routed = {**routed, "stream": False}
            routed.pop("stream_options", None)
        elif self.leaf_inline_evidence and typed_child_chat:
            leaf_inline_evidence_value = inline_evidence_from_messages(
                payload.get("messages")
            )
            if leaf_inline_evidence_value is None:
                raise ValueError("leaf inline-evidence scaffold lacks inline evidence")
            leaf_inline_evidence_scope = True
            routed = {**routed, "stream": False}
            routed.pop("stream_options", None)
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
                if typed_child_report_scope and upstream.status == 200:
                    normalized, _, typed_child_report_action_sha256 = (
                        rewrite_typed_parent_return_response(normalized)
                    )
                    if (
                        typed_child_report_action_sha256 is not None
                        and client_requested_stream
                    ):
                        normalized = chat_completion_to_sse(normalized)
                        response_headers = {
                            key: value
                            for key, value in response_headers.items()
                            if key.lower() != "content-type"
                        }
                        response_headers["Content-Type"] = "text/event-stream"
                    if typed_child_report_action_sha256 is not None:
                        assert session_sha256 is not None
                        self.completed_typed_child_report_hashes.add(session_sha256)
                elif typed_child_report_compute_scope and upstream.status == 200:
                    normalized, _, typed_child_report_compute_action_sha256 = (
                        rewrite_ipython_literal_newlines_response(
                            normalized,
                            inline_evidence=typed_child_report_inline_evidence,
                            report_final_value_to_parent=True,
                        )
                    )
                    if typed_child_report_compute_action_sha256 is not None:
                        assert session_sha256 is not None
                        self.completed_typed_child_report_hashes.add(session_sha256)
                    if client_requested_stream:
                        normalized = chat_completion_to_sse(normalized)
                        response_headers = {
                            key: value
                            for key, value in response_headers.items()
                            if key.lower() != "content-type"
                        }
                        response_headers["Content-Type"] = "text/event-stream"
                elif leaf_compute_report_scope and upstream.status == 200:
                    normalized, _, leaf_compute_report_action_sha256 = (
                        rewrite_ipython_literal_newlines_response(
                            normalized,
                            inline_evidence=leaf_inline_evidence_value,
                            preserve_parent_send=True,
                        )
                    )
                    if leaf_compute_report_action_sha256 is not None:
                        assert session_sha256 is not None
                        self.completed_leaf_compute_report_hashes.add(session_sha256)
                    if client_requested_stream:
                        normalized = chat_completion_to_sse(normalized)
                        response_headers = {
                            key: value
                            for key, value in response_headers.items()
                            if key.lower() != "content-type"
                        }
                        response_headers["Content-Type"] = "text/event-stream"
                elif leaf_inline_evidence_scope and upstream.status == 200:
                    normalized, _, leaf_inline_evidence_action_sha256 = (
                        rewrite_ipython_literal_newlines_response(
                            normalized,
                            inline_evidence=leaf_inline_evidence_value,
                            preserve_parent_send=True,
                        )
                    )
                    if client_requested_stream:
                        normalized = chat_completion_to_sse(normalized)
                        response_headers = {
                            key: value
                            for key, value in response_headers.items()
                            if key.lower() != "content-type"
                        }
                        response_headers["Content-Type"] = "text/event-stream"
                elif typed_return_scope and upstream.status == 200:
                    normalized, _, typed_return_action_sha256 = (
                        rewrite_typed_parent_return_response(normalized)
                    )
                    if typed_return_action_sha256 is not None and client_requested_stream:
                        normalized = chat_completion_to_sse(normalized)
                        response_headers = {
                            key: value
                            for key, value in response_headers.items()
                            if key.lower() != "content-type"
                        }
                        response_headers["Content-Type"] = "text/event-stream"
                    if typed_return_action_sha256 is not None:
                        assert session_sha256 is not None
                        self.completed_typed_return_hashes.add(session_sha256)
                elif typed_compute_scope and upstream.status == 200:
                    normalized, _, typed_compute_action_sha256 = (
                        rewrite_ipython_literal_newlines_response(
                            normalized, inline_evidence=compute_inline_evidence
                        )
                    )
                    if client_requested_stream:
                        normalized = chat_completion_to_sse(normalized)
                        response_headers = {
                            key: value
                            for key, value in response_headers.items()
                            if key.lower() != "content-type"
                        }
                        response_headers["Content-Type"] = "text/event-stream"
                elif root_topology_scope and upstream.status == 200:
                    if root_topology_free_scope:
                        assert root_topology_canonical_codes is not None
                        (
                            normalized,
                            _,
                            root_topology_action_sha256,
                            root_topology_selected,
                            root_topology_expected,
                        ) = rewrite_document_root_free_topology_response(
                            normalized, canonical_codes=root_topology_canonical_codes
                        )
                    else:
                        assert root_topology_canonical_code is not None
                        (
                            normalized,
                            _,
                            root_topology_action_sha256,
                            root_topology_selected,
                            root_topology_expected,
                        ) = rewrite_document_root_topology_response(
                            normalized, canonical_code=root_topology_canonical_code
                        )
                    if (
                        root_topology_action_sha256 is not None
                        and root_topology_selected
                        in {"direct", "flat", "hierarchical"}
                        and session_sha256 is not None
                    ):
                        self.root_document_topologies[session_sha256] = (
                            root_topology_selected
                        )
                    if client_requested_stream:
                        normalized = chat_completion_to_sse(normalized)
                        response_headers = {
                            key: value
                            for key, value in response_headers.items()
                            if key.lower() != "content-type"
                        }
                        response_headers["Content-Type"] = "text/event-stream"
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
                        "forwarded_document_root_topology_normalized_"
                        f"{root_topology_selected}"
                        if root_topology_action_sha256 is not None
                        else (
                            "forwarded_document_root_topology_mismatch_"
                            f"{root_topology_selected}_expected_{root_topology_expected}"
                        )
                    )
                    if root_topology_scope
                    else "forwarded_root_json_finalization"
                    if root_finalization_scope
                    else (
                        "forwarded_typed_child_report"
                        if typed_child_report_action_sha256 is not None
                        else "forwarded_typed_child_report_untranslated"
                    )
                    if typed_child_report_scope
                    else (
                        "forwarded_typed_child_report_compute_report"
                        if typed_child_report_compute_action_sha256 is not None
                        else "forwarded_typed_child_report_compute"
                    )
                    if typed_child_report_compute_scope
                    else (
                        "forwarded_leaf_compute_report"
                        if leaf_compute_report_action_sha256 is not None
                        else "forwarded_leaf_compute_report_untranslated"
                    )
                    if leaf_compute_report_scope
                    else (
                        "forwarded_leaf_inline_evidence_repaired"
                        if leaf_inline_evidence_action_sha256 is not None
                        else "forwarded_leaf_inline_evidence_unmodified"
                    )
                    if leaf_inline_evidence_scope
                    else (
                        "forwarded_typed_return_compute_repaired"
                        if typed_compute_action_sha256 is not None
                        else "forwarded_typed_return_compute"
                    )
                    if typed_compute_scope
                    else (
                        "forwarded_typed_coordinator_return"
                        if typed_return_action_sha256 is not None
                        else "forwarded_typed_coordinator_return_untranslated"
                    )
                    if typed_return_scope
                    else f"forwarded_forced_exact_{forced_chat_scope}_schema"
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
                    typed_child_report_action_sha256
                    or typed_child_report_compute_action_sha256
                    or leaf_compute_report_action_sha256
                    or leaf_inline_evidence_action_sha256
                    or typed_return_action_sha256
                    or typed_compute_action_sha256
                    or root_topology_action_sha256
                    or forced_chat_action_sha256
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
    parser.add_argument("--leak-document-manager-exact-action", action="store_true")
    parser.add_argument("--leak-coordinator-return-action", action="store_true")
    parser.add_argument("--typed-coordinator-return", action="store_true")
    parser.add_argument("--leak-child-exact-action", action="store_true")
    parser.add_argument("--strip-child-tool-choice", action="store_true")
    parser.add_argument("--strip-coordinator-tool-choice", action="store_true")
    parser.add_argument("--root-coordinator-contract", action="store_true")
    parser.add_argument("--leaf-reporter-contract", action="store_true")
    parser.add_argument("--leaf-inline-evidence", action="store_true")
    parser.add_argument("--leaf-compute-report-scaffold", action="store_true")
    parser.add_argument("--document-leaf-compute-report-scaffold", action="store_true")
    parser.add_argument("--document-manager-fanin-scaffold", action="store_true")
    parser.add_argument("--document-manager-wait-scaffold", action="store_true")
    parser.add_argument("--document-manager-termination-scaffold", action="store_true")
    parser.add_argument("--document-root-report-relay-scaffold", action="store_true")
    parser.add_argument(
        "--document-root-topology-normalization-scaffold", action="store_true"
    )
    parser.add_argument("--document-root-flat-fanin-scaffold", action="store_true")
    parser.add_argument("--typed-child-report", action="store_true")
    parser.add_argument("--child-authored-compute", action="store_true")
    parser.add_argument("--depth-default-child", action="store_true")
    parser.add_argument("--audit-log", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model or args.coordinator_model)
    private_evidence_token_ids = tokenizer.encode(PRIVATE_EVIDENCE_HEADER, add_special_tokens=False)
    recursive_coordinator_token_ids = tokenizer.encode(
        RECURSIVE_COORDINATOR_HEADER, add_special_tokens=False
    )
    document_coordinator_token_ids = tokenizer.encode(
        DOCUMENT_COORDINATOR_HEADER, add_special_tokens=False
    )
    root_depth_token_ids = tokenizer.encode("Recursive agent depth: 0", add_special_tokens=False)
    if not private_evidence_token_ids:
        raise RuntimeError("private-evidence role marker encoded to an empty token sequence")
    if not recursive_coordinator_token_ids:
        raise RuntimeError("recursive-coordinator role marker encoded to an empty token sequence")
    if not document_coordinator_token_ids:
        raise RuntimeError("document-coordinator role marker encoded to an empty token sequence")
    if not root_depth_token_ids:
        raise RuntimeError("root-depth role marker encoded to an empty token sequence")
    proxy = DualPolicyProxy(
        coordinator_url=args.coordinator_url,
        coordinator_model=args.coordinator_model,
        child_url=args.child_url,
        child_model=args.child_model,
        external_model=args.external_model,
        audit_log=args.audit_log.resolve(),
        private_evidence_token_ids=private_evidence_token_ids,
        recursive_coordinator_token_ids=recursive_coordinator_token_ids,
        document_coordinator_token_ids=document_coordinator_token_ids,
        root_depth_token_ids=root_depth_token_ids,
        depth_default_child=args.depth_default_child,
        tokenizer=tokenizer,
        leak_coordinator_exact_action=args.leak_coordinator_exact_action,
        leak_document_manager_exact_action=args.leak_document_manager_exact_action,
        leak_coordinator_return_action=args.leak_coordinator_return_action,
        typed_coordinator_return=args.typed_coordinator_return,
        leak_child_exact_action=args.leak_child_exact_action,
        strip_child_tool_choice=args.strip_child_tool_choice,
        strip_coordinator_tool_choice=args.strip_coordinator_tool_choice,
        root_coordinator_contract=args.root_coordinator_contract,
        leaf_reporter_contract=args.leaf_reporter_contract,
        leaf_inline_evidence=args.leaf_inline_evidence,
        leaf_compute_report_scaffold=args.leaf_compute_report_scaffold,
        document_leaf_compute_report_scaffold=args.document_leaf_compute_report_scaffold,
        document_manager_fanin_scaffold=args.document_manager_fanin_scaffold,
        document_manager_wait_scaffold=args.document_manager_wait_scaffold,
        document_manager_termination_scaffold=args.document_manager_termination_scaffold,
        document_root_report_relay_scaffold=args.document_root_report_relay_scaffold,
        document_root_topology_normalization_scaffold=(
            args.document_root_topology_normalization_scaffold
        ),
        document_root_flat_fanin_scaffold=args.document_root_flat_fanin_scaffold,
        typed_child_report=args.typed_child_report,
        child_authored_compute=args.child_authored_compute,
    )
    web.run_app(build_app(proxy), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
