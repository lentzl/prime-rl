import asyncio
import ctypes
import gc
import json
import logging
import math
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import orjson
import verifiers.v1 as vf

from prime_rl.configs.algorithm import SDPO_MAX_STUDENT_SUPPORT_TOPK
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.utils.client import setup_inference_pool
from prime_rl.utils.logger import InterceptHandler, get_logger, setup_logger
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_ckpt_dir,
    get_step_path,
)

MAX_PREFILL_CANDIDATE_TOKEN_IDS = SDPO_MAX_STUDENT_SUPPORT_TOPK
"""vLLM 0.22 ``SamplingParams.logprob_token_ids`` per-request limit."""


def _require_prefill_entry_mapping(entry: object, *, position: int, kind: str) -> Mapping:
    if not isinstance(entry, Mapping):
        raise ValueError(f"{kind} returned malformed entry at position {position}")
    return entry


def _parse_prefill_token_id_key(token_id: object, *, position: int, kind: str) -> int:
    if isinstance(token_id, bool):
        raise ValueError(f"{kind} returned malformed token id {token_id!r} at position {position}")
    if isinstance(token_id, int):
        parsed = token_id
    elif isinstance(token_id, str):
        try:
            parsed = int(token_id)
        except ValueError as exc:
            raise ValueError(f"{kind} returned malformed token id {token_id!r} at position {position}") from exc
    else:
        raise ValueError(f"{kind} returned malformed token id {token_id!r} at position {position}")
    if parsed < 0:
        raise ValueError(f"{kind} returned negative token id {parsed} at position {position}")
    return parsed


def _parse_prefill_logprob(value: object, *, token_id: object, position: int, kind: str) -> float:
    if hasattr(value, "logprob"):
        lp = value.logprob
    elif isinstance(value, Mapping):
        lp = value.get("logprob")
    else:
        raise ValueError(f"{kind} returned malformed logprob entry for token id {token_id} at position {position}")
    if lp is None:
        raise ValueError(f"{kind} missing logprob for token id {token_id} at position {position}")
    if isinstance(lp, bool) or not isinstance(lp, float) or not math.isfinite(lp):
        raise ValueError(
            f"{kind} returned non-finite or non-floating logprob for token id {token_id} at position {position}"
        )
    return lp


async def setup_policy_inference_pool(*, config: OrchestratorConfig, tokenizer):
    """Build the live policy inference pool + matching renderer. Returns
    ``(renderer, inference_pool)``.

    Training is renderer-only: the renderer object is the canonical
    messages → token ids path (sft backfill, opsd scoring prefixes, echo role
    attribution) and is always built. The renderer-client sampling path is
    wired onto the pool; when no train env samples from the live policy the
    renderer is still kept for client-side tokenization and the pool's evals
    use plain chat-completions."""
    from renderers.base import create_renderer

    client_config = config.model.client
    model_name = config.model.name
    renderer = create_renderer(tokenizer, config.renderer)
    get_logger().info(f"Initialized {type(renderer).__name__} for {model_name}")
    if config.any_policy_sourced:
        get_logger().info("Using direct renderer rollout client")
    else:
        get_logger().info("No policy-sourced train env — renderer kept for client-side tokenization only")
    inference_pool = await setup_inference_pool(
        client_config,
        model_name=model_name,
        train_client_type="renderer",
        eval_client_type="openai_chat_completions",
        renderer_config=config.renderer,
        pool_size=config.pool_size,
    )
    return renderer, inference_pool


async def setup_sdpo_teacher_inference_pool(*, config: OrchestratorConfig):
    """Build the optional live SDPO teacher inference pool.

    The policy renderer remains the canonical local tokenization path, but the
    teacher pool still uses renderer train clients so SDPO prefill scoring can
    call the token-in/token-out server route.
    """
    if config.sdpo_teacher is None:
        return None

    teacher = config.sdpo_teacher
    get_logger().info(
        f"Initializing SDPO teacher inference pool (base_url={', '.join(teacher.client.base_url)}, "
        f"model={teacher.name})"
    )
    return await setup_inference_pool(
        teacher.client,
        model_name=teacher.name,
        train_client_type="renderer",
        eval_client_type="openai_chat_completions",
        renderer_config=config.renderer,
        pool_size=config.pool_size,
    )


def save_rollouts(rollouts: list[dict], path: Path, exclude_keys: set[str] | None = None) -> None:
    """Save rollouts (Trace dicts, already JSON-serializable) to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY
    with open(path, "wb") as f:
        for rollout in rollouts:
            row = {k: v for k, v in rollout.items() if k not in exclude_keys} if exclude_keys else rollout
            f.write(orjson.dumps(row, default=str, option=opts))


def intercept_vf_logging(logger: str = "verifiers", level: str = "DEBUG", prefix: str | None = None):
    """Intercepts verifiers logging and routes through prime-rl logger with optional prefix."""
    vf_logger = logging.getLogger(logger)
    vf_logger.handlers.clear()
    vf_logger.addHandler(InterceptHandler(prefix=prefix))
    vf_logger.setLevel(level.upper())
    vf_logger.propagate = False


def setup_env_server_logging(log_level: str, json_logging: bool = False) -> None:
    """Configure logging for an env-server process: prime-rl's logger + routing v1's stdlib
    logs through it. Passed to verifiers' ``serve_env`` so it runs in the broker and in every
    spawned worker — fresh ``spawn`` processes that otherwise have no handlers and would drop
    their per-rollout logs."""
    setup_logger(log_level, json_logging=json_logging)
    intercept_vf_logging(logger="verifiers.v1", level=log_level)


def set_default_executor(max_workers: int = 64) -> None:
    """Scale the default asyncio thread pool so asyncio.to_thread has enough capacity."""
    get_logger().info(f"Setting default executor to ThreadPoolExecutor(max_workers={max_workers})")
    asyncio.get_event_loop().set_default_executor(ThreadPoolExecutor(max_workers=max_workers))


def trim_process_memory() -> None:
    """Return freed heap pages to the OS on glibc systems."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as exc:
        get_logger().debug(f"malloc_trim(0) failed: {exc!r}")


def get_model_completion_len(trace: vf.Trace) -> int:
    """All model-generated (completion) tokens across the rollout — excludes
    env-injected tokens between turns."""
    return trace.completion_len


def get_tool_response_len(trace: vf.Trace) -> int:
    """Total tool-response tokens consumed across the whole rollout, read from a
    harness-emitted metric (e.g. RLM's `rlm_total_tool_response_tokens`, deduped
    across turns/branches/sub-RLMs). Returns 0 when no such metric is present."""
    for key, value in (trace.metrics or {}).items():
        if key.endswith("total_tool_response_tokens") and isinstance(value, (int, float)):
            return int(value)
    return 0


async def compute_prefill_logprobs(
    client_config: vf.ClientConfig,
    model_name: str,
    token_ids: list[int],
) -> list[float]:
    """Score ``token_ids`` under ``model_name`` via prefill; returns one
    logprob per token (0.0 for the leading token, which has no context)."""
    import os

    import httpx
    from openai import AsyncOpenAI

    _validate_prefill_token_ids(token_ids)

    # Build the OpenAI client directly from the v1 ``vf.ClientConfig``
    # (base_url / api_key_var / headers). The frozen/teacher pool yields
    # verifiers.v1 client configs, which the v0 ``setup_openai_client`` can't
    # consume (it expects the v0 endpoint-config shape).
    api_key = os.environ.get(client_config.api_key_var) or "EMPTY"
    client = AsyncOpenAI(
        base_url=client_config.base_url,
        api_key=api_key,
        default_headers=client_config.headers or None,
    )

    # Two escape hatches from ``AsyncOpenAI.post``:
    #   1. URL — ``/inference/v1/generate`` is mounted at server root, not
    #      under ``/v1``. Pass an absolute URL so the SDK's
    #      ``_prepare_url`` skips the base-url merge (it short-circuits
    #      when the path passes ``httpx.URL.is_relative_url`` as False).
    #   2. Parse — vLLM's ``GenerateResponse`` is a plain
    #      ``pydantic.BaseModel`` and the SDK's parse layer rejects any
    #      ``cast_to`` that doesn't subclass ``openai.BaseModel``. Use
    #      ``cast_to=httpx.Response`` so the SDK still builds the request
    #      (preserving ``auth_headers``, retries, timeouts, idempotency
    #      keys) and just hands us the raw response to validate ourselves.
    base = str(client.base_url).rstrip("/").removesuffix("/v1")
    http_response = await client.post(
        f"{base}/inference/v1/generate",
        cast_to=httpx.Response,
        body={
            "model": model_name,
            "token_ids": token_ids,
            "sampling_params": {
                "max_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "prompt_logprobs": 1,
            },
        },
    )
    http_response.raise_for_status()
    response = _parse_generate_response(http_response.content)
    # ``prompt_logprobs[i]`` is a ``{token_id: Logprob}`` dict for tokens
    # the engine could score, or ``None`` for the leading token which has no
    # preceding context. vLLM may include the target token and top-k
    # alternatives, so select the exact token id at each position.
    prompt_logprobs = response.prompt_logprobs or []
    if len(prompt_logprobs) != len(token_ids):
        raise ValueError(f"prefill logprobs length != token length ({len(prompt_logprobs)} != {len(token_ids)})")
    flat: list[float] = []
    for i, (token_id, entry) in enumerate(zip(token_ids, prompt_logprobs)):
        if not entry:
            if i != 0:
                raise ValueError(f"prefill logprobs missing entry at position {i} for token id {token_id}")
            flat.append(0.0)
            continue
        entry = _require_prefill_entry_mapping(entry, position=i, kind="prefill logprobs")
        target = entry.get(str(token_id))
        if target is None:
            target = entry.get(token_id)
        if target is None:
            raise ValueError(f"prefill logprobs missing token id {token_id}")
        flat.append(_parse_prefill_logprob(target, token_id=token_id, position=i, kind="prefill logprobs"))
    return flat


async def compute_prefill_topk_logprobs(
    client_config: vf.ClientConfig,
    model_name: str,
    token_ids: list[int],
    topk: int,
) -> tuple[list[list[int]], list[list[float]]]:
    """Score ``token_ids`` under ``model_name`` via prefill and return the
    top-k token ids/logprobs at each position. The leading token has no
    context, so it returns neutral zeros."""
    import os

    import httpx
    from openai import AsyncOpenAI

    _validate_prefill_token_ids(token_ids)
    if isinstance(topk, bool) or not isinstance(topk, int):
        raise ValueError("prefill top-k logprobs requires integer topk")
    if topk <= 0:
        raise ValueError("prefill top-k logprobs requires topk > 0")

    api_key = os.environ.get(client_config.api_key_var) or "EMPTY"
    client = AsyncOpenAI(
        base_url=client_config.base_url,
        api_key=api_key,
        default_headers=client_config.headers or None,
    )
    base = str(client.base_url).rstrip("/").removesuffix("/v1")
    http_response = await client.post(
        f"{base}/inference/v1/generate",
        cast_to=httpx.Response,
        body={
            "model": model_name,
            "token_ids": token_ids,
            "sampling_params": {
                "max_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "prompt_logprobs": topk,
            },
        },
    )
    http_response.raise_for_status()
    response = _parse_generate_response(http_response.content)
    prompt_logprobs = response.prompt_logprobs or []
    if len(prompt_logprobs) != len(token_ids):
        raise ValueError(f"prefill logprobs length != token length ({len(prompt_logprobs)} != {len(token_ids)})")

    token_rows: list[list[int]] = []
    logprob_rows: list[list[float]] = []
    for i, entry in enumerate(prompt_logprobs):
        if not entry:
            if i != 0:
                raise ValueError(f"prefill top-k logprobs missing entry at position {i}")
            token_rows.append([0] * topk)
            logprob_rows.append([0.0] * topk)
            continue
        entry = _require_prefill_entry_mapping(entry, position=i, kind="prefill top-k logprobs")
        parsed = []
        for token_id, value in entry.items():
            parsed_token_id = _parse_prefill_token_id_key(token_id, position=i, kind="prefill top-k logprobs")
            lp = _parse_prefill_logprob(value, token_id=token_id, position=i, kind="prefill top-k logprobs")
            parsed.append((parsed_token_id, lp))
        parsed.sort(key=lambda item: item[1], reverse=True)
        if len(parsed) < topk:
            raise ValueError(f"prefill top-k logprobs returned {len(parsed)} entries at position {i}; expected {topk}")
        top = parsed[:topk]
        token_rows.append([token_id for token_id, _ in top])
        logprob_rows.append([logprob for _, logprob in top])
    return token_rows, logprob_rows


async def compute_prefill_candidate_logprobs(
    client_config: vf.ClientConfig,
    model_name: str,
    token_ids: list[int],
    candidate_token_ids: list[list[int]],
) -> list[list[float]]:
    """Score caller-provided candidate token ids under ``model_name`` via
    prefill.

    vLLM 0.22 exposes ``SamplingParams.logprob_token_ids`` as one global
    candidate list per request, so this helper is intentionally a single-span
    primitive. If the per-position candidate union grows beyond vLLM's current
    limit, callers should split the span before calling this function.
    """
    import os

    import httpx
    from openai import AsyncOpenAI

    _validate_prefill_token_ids(token_ids)
    if not isinstance(candidate_token_ids, list):
        raise ValueError(
            f"prefill candidate token ids must be a list of rows, got {type(candidate_token_ids).__name__}"
        )
    if len(candidate_token_ids) != len(token_ids):
        raise ValueError(
            f"candidate row count must match token length ({len(candidate_token_ids)} != {len(token_ids)})"
        )
    if candidate_token_ids and candidate_token_ids[0]:
        raise ValueError("prefill candidate logprobs cannot score candidate ids at position 0 without context")
    for row_idx, row in enumerate(candidate_token_ids):
        if not isinstance(row, list):
            raise ValueError(
                f"prefill candidate token ids rows must be lists (row={row_idx}, got={type(row).__name__})"
            )
        for token_id in row:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError(f"prefill candidate token ids must be integers (row={row_idx})")
            if token_id < 0:
                raise ValueError(f"prefill candidate token ids must be non-negative (row={row_idx})")

    candidate_union = sorted({token_id for row in candidate_token_ids[1:] for token_id in row})
    if len(candidate_union) > MAX_PREFILL_CANDIDATE_TOKEN_IDS:
        raise ValueError(
            f"prefill candidate logprobs received {len(candidate_union)} unique candidate token ids; "
            f"vLLM 0.22 supports at most {MAX_PREFILL_CANDIDATE_TOKEN_IDS} per request. "
            "Split the span before scoring."
        )

    api_key = os.environ.get(client_config.api_key_var) or "EMPTY"
    client = AsyncOpenAI(
        base_url=client_config.base_url,
        api_key=api_key,
        default_headers=client_config.headers or None,
    )
    base = str(client.base_url).rstrip("/").removesuffix("/v1")
    sampling_params = {
        "max_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "prompt_logprobs": max(len(candidate_union), 1),
    }
    if candidate_union:
        sampling_params["logprob_token_ids"] = candidate_union
    http_response = await client.post(
        f"{base}/inference/v1/generate",
        cast_to=httpx.Response,
        body={
            "model": model_name,
            "token_ids": token_ids,
            "sampling_params": sampling_params,
        },
    )
    http_response.raise_for_status()
    response = _parse_generate_response(http_response.content)
    prompt_logprobs = response.prompt_logprobs or []
    if len(prompt_logprobs) != len(token_ids):
        raise ValueError(f"prefill logprobs length != token length ({len(prompt_logprobs)} != {len(token_ids)})")

    logprob_rows: list[list[float]] = []
    for i, (candidates, entry) in enumerate(zip(candidate_token_ids, prompt_logprobs)):
        if not candidates:
            logprob_rows.append([])
            continue
        if not entry:
            raise ValueError(f"prefill candidate logprobs missing entry at position {i}")
        entry = _require_prefill_entry_mapping(entry, position=i, kind="prefill candidate logprobs")
        row: list[float] = []
        for token_id in candidates:
            target = entry.get(str(token_id))
            if target is None:
                target = entry.get(token_id)
            if target is None:
                raise ValueError(f"prefill candidate logprobs missing token id {token_id} at position {i}")
            row.append(_parse_prefill_logprob(target, token_id=token_id, position=i, kind="prefill candidate logprobs"))
        logprob_rows.append(row)
    return logprob_rows


def _validate_prefill_token_ids(token_ids: list[int]) -> None:
    if not isinstance(token_ids, list):
        raise ValueError(f"prefill token_ids must be a list, got {type(token_ids).__name__}")
    for idx, token_id in enumerate(token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(f"prefill token_ids must contain integers (position={idx})")
        if token_id < 0:
            raise ValueError(f"prefill token_ids must contain non-negative ids (position={idx})")


def _parse_generate_response(content: bytes):
    try:
        from vllm.entrypoints.serve.disagg.protocol import GenerateResponse
    except ModuleNotFoundError:
        payload = json.loads(content)
        return SimpleNamespace(prompt_logprobs=payload.get("prompt_logprobs"))
    return GenerateResponse.model_validate_json(content)


def get_weight_dir(output_dir: Path, step: int, check_exists: bool = True, wait_timeout: int | None = None) -> Path:
    """Get the weight directory for a given checkpoint step.

    Args:
        output_dir: The output directory for the run.
        step: The checkpoint step.
        check_exists: If True, raises FileNotFoundError if no weight directory exists.
            If False, returns the broadcast directory path without checking existence
            (useful for NCCL mode where weights are broadcasted, not stored on disk).
        wait_timeout: Maximum time in seconds to wait for a stable directory to appear.
            If None, no waiting is performed.
    """
    ckpt_weight_dir = get_step_path(get_ckpt_dir(output_dir), step) / "weight"
    broadcast_weight_dir = get_step_path(get_broadcast_dir(output_dir), step)

    def find_stable_dir() -> Path | None:
        # For checkpoint weights, check STABLE file in parent directory (checkpoints/step_{step}/STABLE)
        ckpt_step_dir = get_step_path(get_ckpt_dir(output_dir), step)
        if (ckpt_step_dir / "STABLE").exists() and ckpt_weight_dir.exists():
            return ckpt_weight_dir

        # For broadcast weights, check STABLE file in the broadcast directory itself
        if (broadcast_weight_dir / "STABLE").exists() and broadcast_weight_dir.exists():
            return broadcast_weight_dir

        return None

    # Check immediately, then wait if needed
    result = find_stable_dir()
    if result is None and wait_timeout:
        start_time = time.time()
        while time.time() - start_time < wait_timeout:
            time.sleep(1)
            result = find_stable_dir()
            if result:
                break

    if result:
        return result
    if not check_exists:
        return broadcast_weight_dir

    raise FileNotFoundError(f"No weight directory found for checkpoint step {step}")
