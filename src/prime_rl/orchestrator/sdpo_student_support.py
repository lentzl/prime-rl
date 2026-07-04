from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prime_rl.transport.sdpo import is_active_sdpo_weight

if TYPE_CHECKING:
    from prime_rl.transport.types import TrainingSample


@dataclass(frozen=True)
class SDPOStudentSupportRecord:
    """Student-selected SDPO support exported by a trainer preflight forward."""

    env_name: str | None
    token_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    sdpo_weights: list[float | None]
    student_topk_token_ids: list[list[int] | None]
    student_topk_logprobs: list[list[float] | None]
    sample_id: str | None = None
    temperatures: list[float] | None = None
    preflight_only: bool | None = None


def load_student_support_records(path: Path, *, require_preflight_only: bool = False) -> list[SDPOStudentSupportRecord]:
    """Load schema-v2 token-export rows that carry nonzero SDPO support.

    The trainer writes one JSON record per packed sequence. In the exact
    student-support SDPO path, the algorithm runs a student-forward preflight,
    loads these records, hydrates the matching samples with student-selected
    candidate ids, then teacher-scores those ids under the hindsight prompt.
    """
    files = _find_token_export_files(path)
    records: list[SDPOStudentSupportRecord] = []
    for file in files:
        with file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                context = f"{file}:{line_number}"
                raw = _loads_json_object_no_duplicate_keys(line, context)
                _require(isinstance(raw, dict), context, "expected JSON object record")
                if not _is_sdpo_record(raw, context):
                    if require_preflight_only:
                        _require_inactive_record_has_no_support(raw, context)
                    continue
                record = _parse_record(raw, context, require_unweighted_placeholders=require_preflight_only)
                if require_preflight_only:
                    _require(
                        record.preflight_only is True, context, "student-support records must have preflight_only=true"
                    )
                    _require(
                        record.sample_id is not None,
                        context,
                        "student-support preflight records must carry a non-empty sample_id",
                    )
                    _require(
                        _is_non_blank_string(record.env_name),
                        context,
                        "student-support preflight records must carry a non-empty env_name",
                    )
                records.append(record)
    return records


def hydrate_student_support_from_records(
    samples: Iterable[TrainingSample],
    records: Iterable[SDPOStudentSupportRecord],
    *,
    expected_topk: int | None = None,
    require_sample_ids: bool = False,
) -> int:
    """Attach exported student-selected top-k ids to matching SDPO samples.

    New exports carry ``sample_id`` so matching does not depend on rank/file
    ordering. Legacy records without ids fall back to order-preserving strict
    matching.
    """
    sdpo_samples = [sample for sample in samples if _sample_has_sdpo(sample)]
    records = list(records)
    if require_sample_ids:
        _require_hydration_sample_ids(sdpo_samples, records)
        return _hydrate_student_support_by_sample_id(sdpo_samples, records, expected_topk=expected_topk)
    if _can_match_by_sample_id(sdpo_samples, records):
        return _hydrate_student_support_by_sample_id(sdpo_samples, records, expected_topk=expected_topk)
    if any(record.sample_id for record in records) or any(
        getattr(sample, "sample_id", None) for sample in sdpo_samples
    ):
        raise ValueError("student-support sample_id coverage mismatch")
    return _hydrate_student_support_in_order(sdpo_samples, records, expected_topk=expected_topk)


def _loads_json_object_no_duplicate_keys(line: str, context: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in pairs:
            if key in decoded:
                raise ValueError(f"{context}: duplicate JSON object key: {key}")
            decoded[key] = value
        return decoded

    return json.loads(line, object_pairs_hook=reject_duplicate_keys)


def _require_hydration_sample_ids(
    samples: list[TrainingSample],
    records: list[SDPOStudentSupportRecord],
) -> None:
    missing_sample_indices = [
        idx for idx, sample in enumerate(samples) if not _is_non_blank_string(getattr(sample, "sample_id", None))
    ]
    if missing_sample_indices:
        raise ValueError(
            "student-support hydration requires sample_id on every SDPO sample "
            f"(missing sample index(es): {missing_sample_indices})"
        )
    missing_sample_env_indices = [
        idx for idx, sample in enumerate(samples) if not _is_non_blank_string(getattr(sample, "env_name", None))
    ]
    if missing_sample_env_indices:
        raise ValueError(
            "student-support hydration requires env_name on every SDPO sample "
            f"(missing sample index(es): {missing_sample_env_indices})"
        )
    missing_record_indices = [idx for idx, record in enumerate(records) if not _is_non_blank_string(record.sample_id)]
    if missing_record_indices:
        raise ValueError(
            "student-support hydration requires sample_id on every export record "
            f"(missing record index(es): {missing_record_indices})"
        )
    missing_record_env_indices = [
        idx for idx, record in enumerate(records) if not _is_non_blank_string(record.env_name)
    ]
    if missing_record_env_indices:
        raise ValueError(
            "student-support hydration requires env_name on every export record "
            f"(missing record index(es): {missing_record_env_indices})"
        )


def _hydrate_student_support_in_order(
    samples: Iterable[TrainingSample],
    records: Iterable[SDPOStudentSupportRecord],
    *,
    expected_topk: int | None,
) -> int:
    record_iter = iter(records)
    hydrated_rows = 0
    for sample in samples:
        try:
            record = next(record_iter)
        except StopIteration as exc:
            raise ValueError("missing student-support export record for SDPO sample") from exc
        hydrated_rows += hydrate_sample_student_support(sample, record, expected_topk=expected_topk)

    try:
        extra = next(record_iter)
    except StopIteration:
        return hydrated_rows
    raise ValueError(
        "student-support export contains more SDPO records than samples "
        f"(first extra env={extra.env_name!r}, tokens={len(extra.token_ids)})."
    )


def _can_match_by_sample_id(samples: list[TrainingSample], records: list[SDPOStudentSupportRecord]) -> bool:
    return (
        bool(samples)
        and all(_is_non_blank_string(getattr(sample, "sample_id", None)) for sample in samples)
        and all(_is_non_blank_string(record.sample_id) for record in records)
    )


def _hydrate_student_support_by_sample_id(
    samples: list[TrainingSample],
    records: list[SDPOStudentSupportRecord],
    *,
    expected_topk: int | None,
) -> int:
    sample_ids: set[str] = set()
    for sample in samples:
        sample_id = getattr(sample, "sample_id", None)
        if not _is_non_blank_string(sample_id):
            raise ValueError("student-support sample_id matching requires sample_id on every SDPO sample")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate SDPO sample_id {sample_id!r}")
        sample_ids.add(sample_id)

    records_by_id: dict[str, SDPOStudentSupportRecord] = {}
    for record in records:
        if not _is_non_blank_string(record.sample_id):
            raise ValueError("student-support sample_id matching requires sample_id on every export record")
        if record.sample_id in records_by_id:
            raise ValueError(f"duplicate student-support export sample_id {record.sample_id!r}")
        records_by_id[record.sample_id] = record

    hydrated_rows = 0
    for sample in samples:
        sample_id = getattr(sample, "sample_id", None)
        if not _is_non_blank_string(sample_id):
            raise ValueError("student-support sample_id matching requires sample_id on every SDPO sample")
        try:
            record = records_by_id.pop(sample_id)
        except KeyError as exc:
            raise ValueError(f"missing student-support export record for sample_id {sample_id!r}") from exc
        hydrated_rows += hydrate_sample_student_support(sample, record, expected_topk=expected_topk)

    if records_by_id:
        first_extra = next(iter(records_by_id))
        raise ValueError(f"student-support export contains extra sample_id {first_extra!r}")
    return hydrated_rows


def hydrate_sample_student_support(
    sample: TrainingSample,
    record: SDPOStudentSupportRecord,
    *,
    expected_topk: int | None = None,
) -> int:
    """Hydrate one sample from one exported student-support record.

    The preflight student logprobs are only validation/debug evidence. The
    final SDPO batch transports teacher logprobs on the hydrated student ids,
    while the trainer recomputes current student logprobs during the training
    forward.
    """
    context = f"env={sample.env_name!r}"
    sample_id = getattr(sample, "sample_id", None)
    if record.sample_id is not None or sample_id is not None:
        _require(record.sample_id == sample_id, context, f"record sample_id={record.sample_id!r} mismatch")
    _require(record.env_name in (None, "", sample.env_name), context, f"record env_name={record.env_name!r} mismatch")
    _require(record.token_ids == list(sample.token_ids), context, "record token_ids do not match sample")
    expected_position_ids = list(getattr(sample, "position_ids", range(len(sample.token_ids))))
    _require(len(expected_position_ids) == len(sample.token_ids), context, "sample position_ids length mismatch")
    _require(record.position_ids == expected_position_ids, context, "record position_ids do not match sample")
    _require(record.loss_mask == list(sample.mask), context, "record loss_mask does not match sample mask")
    if record.temperatures is not None:
        _require(len(record.temperatures) == len(sample.token_ids), context, "record temperatures length mismatch")
        sample_temperatures = list(sample.temperatures)
        _require(len(sample_temperatures) == len(sample.token_ids), context, "sample temperatures length mismatch")
        for idx, (sample_temperature, record_temperature) in enumerate(zip(sample_temperatures, record.temperatures)):
            _require(
                math.isclose(float(sample_temperature), record_temperature, rel_tol=1e-6, abs_tol=1e-6),
                context,
                f"temperature mismatch at token {idx}",
            )
    _require(len(record.sdpo_weights) == len(sample.token_ids), context, "record sdpo_weights length mismatch")

    sample_weights = sample.sdpo_weights
    _require(sample_weights is not None, context, "sample has no sdpo_weights")
    _require(len(sample_weights) == len(sample.token_ids), context, "sample sdpo_weights length mismatch")
    for idx, (sample_weight, record_weight) in enumerate(zip(sample_weights, record.sdpo_weights)):
        _require(
            sample_weight is None or _is_finite_number(sample_weight),
            context,
            f"sample sdpo_weights must contain finite numeric values at token {idx}",
        )
        _require(
            sample_weight is None or sample_weight >= 0,
            context,
            f"sample sdpo_weights must be non-negative at token {idx}",
        )
        _require(
            is_active_sdpo_weight(sample_weight) == is_active_sdpo_weight(record_weight),
            context,
            f"sdpo weight membership mismatch at token {idx}",
        )
        if sample_weight is None or record_weight is None:
            _require(sample_weight is record_weight, context, f"sdpo weight value mismatch at token {idx}")
        else:
            _require(
                math.isclose(float(sample_weight), float(record_weight), rel_tol=1e-6, abs_tol=1e-6),
                context,
                f"sdpo weight value mismatch at token {idx}",
            )

    topk = _support_width(
        record.student_topk_token_ids,
        record.student_topk_logprobs,
        sample_weights,
        context,
        expected_topk=expected_topk,
    )
    sample.sdpo_topk_token_ids = [[0] * topk for _ in sample.token_ids]
    sample.sdpo_topk_logprobs = None
    hydrated_rows = 0
    for idx, weight in enumerate(sample_weights):
        if not is_active_sdpo_weight(weight):
            continue
        row = record.student_topk_token_ids[idx]
        if row is None:
            raise ValueError(f"{context}: student top-k ids missing at token {idx}")
        sample.sdpo_topk_token_ids[idx] = list(row)
        hydrated_rows += 1
    return hydrated_rows


def _find_token_export_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    direct = sorted(path.glob("rank_*.jsonl"))
    if direct:
        return direct
    from_token_exports = sorted(path.glob("step_*/rank_*.jsonl"))
    if from_token_exports:
        return from_token_exports
    nested = sorted(path.glob("**/token_exports/step_*/rank_*.jsonl"))
    if nested:
        return nested
    raise FileNotFoundError(f"No token export rank_*.jsonl files found under {path}")


def _parse_record(
    raw: dict[str, Any],
    context: str,
    *,
    require_unweighted_placeholders: bool = False,
) -> SDPOStudentSupportRecord:
    _require(raw.get("schema_version") == 2, context, "expected schema_version == 2")
    raw_sample_id = raw.get("sample_id")
    raw_env_name = raw.get("env_name")
    raw_preflight_only = raw.get("preflight_only")
    if raw_sample_id is not None:
        _require(isinstance(raw_sample_id, str), context, "sample_id must be a string when present")
        _require(bool(raw_sample_id.strip()), context, "sample_id must be a non-empty string when present")
    if raw_env_name is not None:
        _require(isinstance(raw_env_name, str), context, "env_name must be a string when present")
    if raw_preflight_only is not None:
        _require(isinstance(raw_preflight_only, bool), context, "preflight_only must be a boolean when present")
    token_ids = _require_list(raw.get("token_ids"), context, "token_ids")
    position_ids = _require_list(raw.get("position_ids"), context, "position_ids")
    loss_mask = _require_list(raw.get("loss_mask"), context, "loss_mask")
    temperatures = _optional_list(raw.get("temperatures"), context, "temperatures")
    weights = _require_list(raw.get("sdpo_weights"), context, "sdpo_weights")
    topk_ids = _require_list(raw.get("sdpo_student_topk_token_ids"), context, "sdpo_student_topk_token_ids")
    topk_logprobs = _require_list(raw.get("sdpo_student_topk_logprobs"), context, "sdpo_student_topk_logprobs")
    expected_len = len(token_ids)
    for name, value in (
        ("loss_mask", loss_mask),
        ("position_ids", position_ids),
        ("temperatures", temperatures),
        ("sdpo_weights", weights),
        ("sdpo_student_topk_token_ids", topk_ids),
        ("sdpo_student_topk_logprobs", topk_logprobs),
    ):
        if value is None:
            continue
        _require(len(value) == expected_len, context, f"{name} length {len(value)} != token_ids length {expected_len}")
    _require_int_values(token_ids, context, "token_ids")
    _require_int_values(position_ids, context, "position_ids")
    _require(all(isinstance(value, bool) for value in loss_mask), context, "loss_mask must contain booleans")
    _require_numeric_values(weights, context, "sdpo_weights", allow_none=True)
    _require_non_negative_values(weights, context, "sdpo_weights")
    if temperatures is not None:
        _require(
            all(_is_finite_number(value) and value > 0 for value in temperatures),
            context,
            "temperatures must contain positive finite numbers",
        )
    _require_topk_id_rows(topk_ids, context, "sdpo_student_topk_token_ids")
    _require_topk_logprob_rows(topk_logprobs, context, "sdpo_student_topk_logprobs")
    _require_paired_topk_row_widths(topk_ids, topk_logprobs, context, "sdpo_student_topk")
    _require_active_topk_logprob_rows_are_floats(topk_logprobs, weights, context, "sdpo_student_topk_logprobs")
    _require_sdpo_weights_follow_loss_mask(weights, loss_mask, context)
    if require_unweighted_placeholders:
        _require_unweighted_support_placeholders(topk_ids, topk_logprobs, weights, context)
        _require_no_transported_teacher_support(raw, weights, context)
    return SDPOStudentSupportRecord(
        sample_id=None if raw_sample_id is None else raw_sample_id,
        env_name=raw_env_name,
        token_ids=[int(value) for value in token_ids],
        position_ids=[int(value) for value in position_ids],
        loss_mask=[bool(value) for value in loss_mask],
        temperatures=None if temperatures is None else [float(value) for value in temperatures],
        preflight_only=raw_preflight_only,
        sdpo_weights=[None if value is None else float(value) for value in weights],
        student_topk_token_ids=[None if row is None else [int(value) for value in row] for row in topk_ids],
        student_topk_logprobs=[None if row is None else [float(value) for value in row] for row in topk_logprobs],
    )


def _support_width(
    token_rows: list[list[int] | None],
    logprob_rows: list[list[float] | None],
    weights: list[float],
    context: str,
    *,
    expected_topk: int | None,
) -> int:
    width: int | None = None
    for idx, weight in enumerate(weights):
        if not is_active_sdpo_weight(weight):
            continue
        ids = token_rows[idx]
        logprobs = logprob_rows[idx]
        _require(isinstance(ids, list) and ids, context, f"student top-k ids missing at token {idx}")
        _require(isinstance(logprobs, list) and logprobs, context, f"student top-k logprobs missing at token {idx}")
        _require(len(ids) == len(logprobs), context, f"student top-k row width mismatch at token {idx}")
        _require(
            not _is_placeholder_logprob_row(logprobs),
            context,
            f"student top-k logprobs look like an unfilled placeholder row at token {idx}",
        )
        _require(len(set(ids)) == len(ids), context, f"student top-k ids contain duplicate token ids at token {idx}")
        _require(
            all(_is_finite_float_number(logprob) for logprob in logprobs),
            context,
            f"student top-k logprobs must contain floating-point values at token {idx}",
        )
        row_mass = math.fsum(math.exp(logprob) for logprob in logprobs)
        _require(
            row_mass <= 1.0 + 1e-5,
            context,
            f"student top-k logprob probability mass exceeds 1 at token {idx}",
        )
        if width is None:
            width = len(ids)
        _require(len(ids) == width, context, f"student top-k width changed at token {idx}")
    _require(width is not None, context, "sample has no nonzero SDPO support rows")
    if expected_topk is not None:
        _require(width == expected_topk, context, f"student top-k width {width} != expected {expected_topk}")
    return width


def _sample_has_sdpo(sample: TrainingSample) -> bool:
    weights = sample.sdpo_weights
    if weights is None:
        return False
    if not isinstance(weights, list):
        raise ValueError(f"env={sample.env_name!r}: sample sdpo_weights must be a list")
    for idx, weight in enumerate(weights):
        if isinstance(weight, bool):
            raise ValueError(
                f"env={sample.env_name!r}: sample sdpo_weights must contain finite numeric values at token {idx}"
            )
    return any(is_active_sdpo_weight(weight) for weight in weights)


def _is_sdpo_record(record: dict[str, Any], context: str) -> bool:
    weights = record.get("sdpo_weights")
    if weights is None:
        return False
    _require(isinstance(weights, list), context, "sdpo_weights must be a list")
    for idx, weight in enumerate(weights):
        _require(
            not isinstance(weight, bool), context, f"sdpo_weights must contain finite numeric values at token {idx}"
        )
    return any(is_active_sdpo_weight(weight) for weight in weights)


def _require_inactive_record_has_no_support(record: dict[str, Any], context: str) -> None:
    for prefix in ("sdpo", "sdpo_student"):
        token_rows = record.get(f"{prefix}_topk_token_ids")
        logprob_rows = record.get(f"{prefix}_topk_logprobs")
        _require(
            token_rows is None and logprob_rows is None,
            context,
            f"inactive SDPO record must not carry {prefix} top-k support rows",
        )


def _is_placeholder_logprob_row(logprobs: list[Any]) -> bool:
    return all(not isinstance(value, bool) and value in (0, 0.0) for value in logprobs)


def _require_int_values(values: list[Any], context: str, name: str) -> None:
    _require(
        all(not isinstance(value, bool) and isinstance(value, int) for value in values),
        context,
        f"{name} must contain integer token ids",
    )
    _require(
        all(value >= 0 for value in values),
        context,
        f"{name} must contain non-negative token ids",
    )


def _require_numeric_values(values: list[Any], context: str, name: str, *, allow_none: bool) -> None:
    _require(
        all((value is None and allow_none) or _is_finite_number(value) for value in values),
        context,
        f"{name} must contain finite numeric values",
    )


def _require_non_negative_values(values: list[Any], context: str, name: str) -> None:
    for idx, value in enumerate(values):
        if value is None:
            continue
        _require(value >= 0, context, f"{name}[{idx}] must be non-negative")


def _require_topk_id_rows(rows: list[Any], context: str, name: str) -> None:
    for idx, row in enumerate(rows):
        if row is None:
            continue
        _require(isinstance(row, list), context, f"{name}[{idx}] must be a list or null")
        _require(bool(row), context, f"{name}[{idx}] must be non-empty")
        _require_int_values(row, context, f"{name}[{idx}]")


def _require_topk_logprob_rows(rows: list[Any], context: str, name: str) -> None:
    for idx, row in enumerate(rows):
        if row is None:
            continue
        _require(isinstance(row, list), context, f"{name}[{idx}] must be a list or null")
        _require(bool(row), context, f"{name}[{idx}] must be non-empty")
        _require_numeric_values(row, context, f"{name}[{idx}]", allow_none=False)


def _require_paired_topk_row_widths(
    token_rows: list[Any],
    logprob_rows: list[Any],
    context: str,
    name: str,
) -> None:
    for idx, (token_row, logprob_row) in enumerate(zip(token_rows, logprob_rows, strict=True)):
        if token_row is None and logprob_row is None:
            continue
        _require(
            isinstance(token_row, list) and isinstance(logprob_row, list),
            context,
            f"{name} row {idx} ids/logprobs must both be lists or both be null",
        )
        _require(
            len(token_row) == len(logprob_row),
            context,
            f"{name} row {idx} width mismatch",
        )


def _require_active_topk_logprob_rows_are_floats(
    rows: list[Any],
    weights: list[Any],
    context: str,
    name: str,
) -> None:
    for idx, (row, weight) in enumerate(zip(rows, weights, strict=True)):
        if not is_active_sdpo_weight(weight) or row is None or _is_placeholder_logprob_row(row):
            continue
        _require(
            all(_is_finite_float_number(value) for value in row),
            context,
            f"{name}[{idx}] must contain floating-point logprob values",
        )


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_finite_float_number(value: Any) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _is_non_blank_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_sdpo_weights_follow_loss_mask(weights: list[Any], loss_mask: list[Any], context: str) -> None:
    for idx, (weight, trains) in enumerate(zip(weights, loss_mask, strict=True)):
        _require(
            not is_active_sdpo_weight(weight) or bool(trains),
            context,
            f"sdpo weight at token {idx} is nonzero outside loss_mask",
        )


def _require_unweighted_support_placeholders(
    token_rows: list[Any],
    logprob_rows: list[Any],
    weights: list[Any],
    context: str,
) -> None:
    for idx, (token_row, logprob_row, weight) in enumerate(zip(token_rows, logprob_rows, weights, strict=True)):
        if is_active_sdpo_weight(weight):
            continue
        if token_row is None and logprob_row is None:
            continue
        _require(
            isinstance(token_row, list) and isinstance(logprob_row, list),
            context,
            f"unweighted student top-k row {idx} must be null or a placeholder",
        )
        _require(
            len(token_row) == len(logprob_row),
            context,
            f"unweighted student top-k row {idx} width mismatch",
        )
        _require(
            _is_placeholder_token_id_row(token_row) and _is_placeholder_logprob_row(logprob_row),
            context,
            f"unweighted student top-k row {idx} must be an all-zero placeholder",
        )


def _require_no_transported_teacher_support(raw: dict[str, Any], weights: list[Any], context: str) -> None:
    token_rows = raw.get("sdpo_topk_token_ids")
    logprob_rows = raw.get("sdpo_topk_logprobs")
    if token_rows is None and logprob_rows is None:
        return
    _require(
        isinstance(token_rows, list) and isinstance(logprob_rows, list),
        context,
        "preflight transported teacher top-k streams must be absent or paired lists",
    )
    _require(
        len(token_rows) == len(weights),
        context,
        f"sdpo_topk_token_ids length {len(token_rows)} != token_ids length {len(weights)}",
    )
    _require(
        len(logprob_rows) == len(weights),
        context,
        f"sdpo_topk_logprobs length {len(logprob_rows)} != token_ids length {len(weights)}",
    )
    for idx, (token_row, logprob_row, _weight) in enumerate(zip(token_rows, logprob_rows, weights, strict=True)):
        if token_row is None and logprob_row is None:
            continue
        _require(
            isinstance(token_row, list) and isinstance(logprob_row, list),
            context,
            f"preflight transported teacher top-k row {idx} must be null or an all-zero placeholder",
        )
        _require(
            len(token_row) == len(logprob_row),
            context,
            f"preflight transported teacher top-k row {idx} width mismatch",
        )
        _require(
            _is_placeholder_token_id_row(token_row) and _is_placeholder_logprob_row(logprob_row),
            context,
            f"preflight transported teacher top-k row {idx} must be null or an all-zero placeholder",
        )


def _is_placeholder_token_id_row(token_ids: list[Any]) -> bool:
    return all(value in (0, 0.0) and not isinstance(value, bool) for value in token_ids)


def _require_list(value: Any, context: str, name: str) -> list[Any]:
    _require(isinstance(value, list), context, f"{name} must be a list")
    return value


def _optional_list(value: Any, context: str, name: str) -> list[Any] | None:
    if value is None:
        return None
    return _require_list(value, context, name)


def _require(condition: bool, context: str, message: str) -> None:
    if not condition:
        raise ValueError(f"{context}: {message}")
