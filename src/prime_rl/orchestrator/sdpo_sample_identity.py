from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from prime_rl.transport.sdpo import has_active_sdpo_weights

if TYPE_CHECKING:
    from prime_rl.transport import TrainingSample


def ensure_sdpo_sample_ids(
    samples: list["TrainingSample"],
    *,
    step: int,
    prefix: str,
    phase: str,
) -> None:
    """Ensure active SDPO samples have unique step-local identities."""
    seen: set[str] = set()
    for idx, sample in enumerate(samples):
        if not _sample_has_sdpo(sample):
            continue
        env_name = _sample_env_name(sample, phase)
        sample_id = getattr(sample, "sample_id", None)
        if sample_id is None:
            sample_id = f"{prefix}-step-{step}-env-{_sample_id_env_fragment(env_name)}-sample-{idx}"
        elif not isinstance(sample_id, str):
            raise ValueError(f"SDPO {phase} sample_id must be a string, got {type(sample_id).__name__}")
        elif not sample_id.strip():
            raise ValueError(f"SDPO {phase} sample_id must be non-empty")
        if sample_id in seen:
            raise ValueError(f"duplicate SDPO {phase} sample_id {sample_id!r}")
        sample.sample_id = sample_id
        seen.add(sample_id)


def _sample_has_sdpo(sample: "TrainingSample") -> bool:
    weights = getattr(sample, "sdpo_weights", None)
    if weights is None:
        return False
    token_ids = getattr(sample, "token_ids", None)
    mask = getattr(sample, "mask", None)
    if not isinstance(token_ids, list):
        raise ValueError(f"SDPO sample_id assignment requires token_ids to be a list, got {type(token_ids).__name__}")
    if not isinstance(mask, list):
        raise ValueError(f"SDPO sample_id assignment requires mask to be a list, got {type(mask).__name__}")
    if not isinstance(weights, list):
        raise ValueError(f"SDPO sample_id assignment requires sdpo_weights to be a list, got {type(weights).__name__}")
    if len(mask) != len(token_ids):
        raise ValueError(
            "SDPO sample_id assignment requires mask length to match token_ids length "
            f"({len(mask)} != {len(token_ids)})"
        )
    if len(weights) != len(token_ids):
        raise ValueError(
            "SDPO sample_id assignment requires sdpo_weights length to match token_ids length "
            f"({len(weights)} != {len(token_ids)})"
        )
    for idx, (weight, trains) in enumerate(zip(weights, mask, strict=True)):
        if not isinstance(trains, bool):
            raise ValueError(f"SDPO sample_id assignment requires boolean mask values at token {idx}")
        if not _is_finite_number(weight):
            raise ValueError(f"SDPO sample_id assignment requires finite numeric sdpo_weights at token {idx}")
        if weight < 0:
            raise ValueError(f"SDPO sample_id assignment requires non-negative sdpo_weights at token {idx}")
        if has_active_sdpo_weights([weight]) and not trains:
            raise ValueError(f"SDPO sample_id assignment requires sdpo_weights to be zero outside mask at token {idx}")
    active = has_active_sdpo_weights(weights)
    if active and not _is_non_blank_string(getattr(sample, "env_name", None)):
        raise ValueError("SDPO sample_id assignment requires env_name to be a non-empty string")
    return active


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_non_blank_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sample_env_name(sample: "TrainingSample", phase: str) -> str:
    env_name = getattr(sample, "env_name", None)
    if not _is_non_blank_string(env_name):
        raise ValueError(f"SDPO {phase} sample_id assignment requires env_name to be a non-empty string")
    return env_name.strip()


def _sample_id_env_fragment(env_name: str) -> str:
    return quote(env_name, safe="-_.~")
