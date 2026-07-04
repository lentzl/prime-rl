import asyncio
import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prime_rl.orchestrator.sdpo_sample_identity import ensure_sdpo_sample_ids
from prime_rl.orchestrator.sdpo_student_support import (
    hydrate_student_support_from_records,
    load_student_support_records,
)
from prime_rl.transport import TrainingBatch
from prime_rl.transport.sdpo import has_active_sdpo_weights, is_active_sdpo_weight
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import wait_for_path

if TYPE_CHECKING:
    from prime_rl.transport import TrainingSample
    from prime_rl.transport.base import TrainingBatchSender


def token_export_step_dir(output_dir: Path, step: int) -> Path:
    """Trainer token-export directory for a run's logical step."""
    run_id = output_dir.name
    return output_dir.parent / run_id / "token_exports" / f"step_{step}"


async def run_sdpo_student_support_preflight(
    *,
    output_dir: Path,
    sender: "TrainingBatchSender",
    samples: list["TrainingSample"],
    step: int,
    expected_topk: int | None = None,
    export_timeout_s: int | None = None,
) -> int:
    """Run a forward/export-only trainer pass and hydrate student top-k ids."""
    if expected_topk is None:
        raise ValueError("SDPO student-support preflight requires expected_topk")
    if isinstance(expected_topk, bool) or not isinstance(expected_topk, int):
        raise ValueError("SDPO student-support preflight expected_topk must be an integer")
    if expected_topk <= 0:
        raise ValueError("SDPO student-support preflight expected_topk must be positive")
    _validate_export_timeout_s(export_timeout_s)
    export_dir = token_export_step_dir(output_dir, step)
    sdpo_samples = _validated_sdpo_samples(samples)
    if not sdpo_samples:
        raise ValueError(f"Step {step}: SDPO student-support preflight found no SDPO-weighted samples")
    if export_dir.exists():
        await asyncio.to_thread(shutil.rmtree, export_dir)
    get_logger().info(f"Step {step}: running SDPO student-support preflight export")
    ensure_sdpo_sample_ids(sdpo_samples, step=step, prefix="sdpo-preflight", phase="preflight")
    await sender.send(TrainingBatch(examples=sdpo_samples, step=step, preflight_only=True))
    stable_marker = export_dir / "STABLE"
    try:
        if export_timeout_s is None:
            await wait_for_path(stable_marker)
        else:
            await asyncio.wait_for(wait_for_path(stable_marker), timeout=export_timeout_s)
    except TimeoutError as exc:
        raise TimeoutError(
            f"Step {step}: timed out after {export_timeout_s}s waiting for SDPO student-support "
            f"preflight export marker {stable_marker}"
        ) from exc
    records = await asyncio.to_thread(load_student_support_records, export_dir, require_preflight_only=True)
    hydrated_rows = hydrate_student_support_from_records(
        sdpo_samples,
        records,
        expected_topk=expected_topk,
        require_sample_ids=True,
    )
    if hydrated_rows <= 0:
        raise ValueError(f"Step {step}: SDPO student-support preflight exported no usable support rows")
    get_logger().info(f"Step {step}: hydrated {hydrated_rows} SDPO student-support row(s) from {export_dir}")
    return hydrated_rows


def _validated_sdpo_samples(samples: list["TrainingSample"]) -> list["TrainingSample"]:
    sdpo_samples = []
    for sample in samples:
        if not _sample_has_sdpo(sample):
            continue
        _validate_sdpo_weights(sample)
        _validate_fresh_student_support(sample)
        sdpo_samples.append(sample)
    return sdpo_samples


def _validate_export_timeout_s(export_timeout_s: int | float | None) -> None:
    if export_timeout_s is None:
        return
    if not _is_finite_number(export_timeout_s):
        raise ValueError("SDPO student-support preflight export_timeout_s must be a finite numeric value when set")
    if export_timeout_s <= 0:
        raise ValueError("SDPO student-support preflight export_timeout_s must be positive when set")


def _validate_fresh_student_support(sample: "TrainingSample") -> None:
    env_name = getattr(sample, "env_name", "<unknown>")
    for name in ("sdpo_topk_token_ids", "sdpo_topk_logprobs", "sdpo_rollout_is_weights"):
        if getattr(sample, name, None) is not None:
            raise ValueError(
                f"SDPO preflight sample must not carry pre-existing {name} "
                f"before student-support hydration (env '{env_name}')."
            )


def _validate_sdpo_weights(sample: "TrainingSample") -> None:
    weights = getattr(sample, "sdpo_weights", None)
    token_ids = getattr(sample, "token_ids", None)
    mask = getattr(sample, "mask", None)
    env_name = getattr(sample, "env_name", "<unknown>")
    if not isinstance(token_ids, list):
        raise ValueError(f"SDPO preflight token_ids must be a list (env '{env_name}').")
    if not isinstance(mask, list):
        raise ValueError(f"SDPO preflight mask must be a list (env '{env_name}').")
    if not isinstance(weights, list):
        raise ValueError(f"SDPO preflight sdpo_weights must be a list (env '{env_name}').")
    for idx, token_id in enumerate(token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(f"SDPO preflight token_ids must contain integers (env '{env_name}', token={idx}).")
        if token_id < 0:
            raise ValueError(f"SDPO preflight token_ids must be non-negative (env '{env_name}', token={idx}).")
    if len(mask) != len(token_ids):
        raise ValueError(
            f"SDPO preflight mask length must match token_ids length "
            f"(env '{env_name}', mask={len(mask)}, tokens={len(token_ids)})."
        )
    if len(weights) != len(token_ids):
        raise ValueError(
            f"SDPO preflight sdpo_weights length must match token_ids length "
            f"(env '{env_name}', weights={len(weights)}, tokens={len(token_ids)})."
        )
    for idx, (weight, trains) in enumerate(zip(weights, mask, strict=True)):
        if not isinstance(trains, bool):
            raise ValueError(f"SDPO preflight mask must contain booleans (env '{env_name}', token={idx}).")
        if not _is_finite_number(weight):
            raise ValueError(
                f"SDPO preflight sdpo_weights must contain finite numeric values (env '{env_name}', token={idx})."
            )
        if weight < 0:
            raise ValueError(f"SDPO preflight sdpo_weights must be non-negative (env '{env_name}', token={idx}).")
        if is_active_sdpo_weight(weight) and not trains:
            raise ValueError(
                f"SDPO preflight sdpo_weights must be zero outside sampled tokens (env '{env_name}', token={idx})."
            )


def _sample_has_sdpo(sample: "TrainingSample") -> bool:
    weights = getattr(sample, "sdpo_weights", None)
    if weights is None:
        return False
    if not isinstance(weights, list):
        env_name = getattr(sample, "env_name", "<unknown>")
        raise ValueError(f"SDPO preflight sdpo_weights must be a list (env '{env_name}').")
    env_name = getattr(sample, "env_name", "<unknown>")
    for idx, weight in enumerate(weights):
        if isinstance(weight, bool):
            raise ValueError(
                f"SDPO preflight sdpo_weights must contain finite numeric values (env '{env_name}', token={idx})."
            )
    return has_active_sdpo_weights(weights)


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))
