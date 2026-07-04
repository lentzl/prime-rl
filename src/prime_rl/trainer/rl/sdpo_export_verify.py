from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prime_rl.transport.sdpo import has_active_sdpo_weights, is_active_sdpo_weight


@dataclass(frozen=True)
class SDPOExportStats:
    files: int
    records: int
    sdpo_records: int
    transported_rows: int
    student_rows: int
    paired_rows: int
    matching_support_rows: int
    distinct_teacher_logprob_rows: int
    importance_ratio_rows: int
    rollout_is_weight_rows: int
    student_preflight_rows: int
    temperature_rows: int
    sample_id_records: int
    stable_steps: int = 0
    step_names: tuple[str, ...] = ()
    paired_step_names: tuple[str, ...] = ()
    matching_support_step_names: tuple[str, ...] = ()
    student_preflight_step_names: tuple[str, ...] = ()
    matching_support_sample_keys: tuple[str, ...] = ()
    student_preflight_sample_keys: tuple[str, ...] = ()
    matched_support_sample_keys: tuple[str, ...] = ()
    matching_support_row_keys: tuple[str, ...] = ()
    importance_ratio_row_keys: tuple[str, ...] = ()
    rollout_is_weight_row_keys: tuple[str, ...] = ()
    student_preflight_row_keys: tuple[str, ...] = ()
    distinct_teacher_logprob_row_keys: tuple[str, ...] = ()
    matched_support_row_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class SDPOEMABroadcastStats:
    steps: int
    teacher_steps: int
    role: str
    step_names: tuple[str, ...] = ()
    step_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class SDPOSmokeArtifactStats:
    token_exports: SDPOExportStats
    ema_broadcasts: SDPOEMABroadcastStats | None = None
    matched_steps: tuple[str, ...] = ()
    matched_step_keys: tuple[str, ...] = ()


def find_token_export_files(path: Path) -> list[Path]:
    """Find token-export JSONL files below an output, token_exports, or sibling broadcast dir."""
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

    sibling_files = _sibling_token_export_files(path)
    if sibling_files:
        return sibling_files

    raise FileNotFoundError(f"No token export rank_*.jsonl files found under {path}")


def find_broadcast_step_dirs(path: Path) -> list[Path]:
    """Find filesystem-broadcast step directories below an output, broadcasts, or sibling token_exports dir."""
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir() and path.name.startswith("step_") and (path / "STABLE").exists():
        return [path]

    candidates: list[Path] = []
    for root in (path, path / "broadcasts", path / "run_default" / "broadcasts"):
        if root.exists():
            candidates.extend(root.glob("step_*"))
    if not candidates:
        candidates.extend(_sibling_broadcast_step_dirs(path))
    if not candidates:
        candidates.extend(path.glob("**/broadcasts/step_*"))

    step_dirs = sorted(
        {candidate for candidate in candidates if candidate.is_dir() and (candidate / "STABLE").exists()}
    )
    if not step_dirs:
        raise FileNotFoundError(f"No stable filesystem broadcast step directories found under {path}")
    return step_dirs


def _sibling_token_export_files(path: Path) -> list[Path]:
    if path.name == "broadcasts":
        return sorted((path.parent / "token_exports").glob("step_*/rank_*.jsonl"))
    if path.name.startswith("step_") and path.parent.name == "broadcasts":
        return sorted((path.parent.parent / "token_exports" / path.name).glob("rank_*.jsonl"))
    return []


def _sibling_broadcast_step_dirs(path: Path) -> list[Path]:
    if path.name == "token_exports":
        return sorted((path.parent / "broadcasts").glob("step_*"))
    if path.name.startswith("step_") and path.parent.name == "token_exports":
        return [path.parent.parent / "broadcasts" / path.name]
    if path.is_file() and path.parent.name.startswith("step_") and path.parent.parent.name == "token_exports":
        return [path.parent.parent.parent / "broadcasts" / path.parent.name]
    return []


def verify_sdpo_token_exports(
    path: Path,
    *,
    require_stable: bool = False,
    require_student_preflight: bool = False,
    require_importance_ratio_evidence: bool = False,
    expected_topk: int | None = None,
    rollout_is_threshold: float | None = None,
    rollout_is: str | None = None,
) -> SDPOExportStats:
    if expected_topk is not None:
        _validate_expected_topk(expected_topk, str(path))
    if rollout_is_threshold is not None:
        _validate_rollout_is_threshold(rollout_is_threshold, str(path))
    if rollout_is is not None:
        _validate_rollout_is(rollout_is, str(path))
        _require(
            rollout_is_threshold is not None,
            str(path),
            "rollout_is matching requires rollout_is_threshold",
        )
    files = find_token_export_files(path)
    step_dirs = _token_export_step_dirs(files)
    stable_steps = _count_stable_token_export_steps(step_dirs, require_stable=require_stable)
    records = 0
    sdpo_records = 0
    transported_rows = 0
    student_rows = 0
    paired_rows = 0
    matching_support_rows = 0
    distinct_teacher_logprob_rows = 0
    importance_ratio_rows = 0
    rollout_is_weight_rows = 0
    student_preflight_rows = 0
    temperature_rows = 0
    sample_id_records = 0
    paired_step_dirs: set[Path] = set()
    matching_support_step_dirs: set[Path] = set()
    student_preflight_step_dirs: set[Path] = set()
    matching_support_sample_keys: set[str] = set()
    student_preflight_sample_keys: set[str] = set()
    matching_support_row_keys: set[str] = set()
    distinct_teacher_logprob_row_keys: set[str] = set()
    importance_ratio_row_keys: set[str] = set()
    rollout_is_weight_row_keys: set[str] = set()
    student_preflight_row_keys: set[str] = set()
    matching_support_sample_signatures: dict[str, tuple[Any, ...]] = {}
    student_preflight_sample_signatures: dict[str, tuple[Any, ...]] = {}
    matching_support_row_signatures: dict[str, tuple[Any, ...]] = {}
    student_preflight_row_signatures: dict[str, tuple[Any, ...]] = {}
    final_sdpo_sample_keys: set[str] = set()

    for file in files:
        with file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                context = f"{file}:{line_number}"
                record = _loads_json_object_no_duplicate_keys(line, context)
                records += 1
                _require(isinstance(record, dict), context, "expected JSON object record")
                _require_inactive_records_do_not_carry_sdpo_support(record, context)
                if _is_sdpo_record(record, context):
                    _require(record.get("schema_version") == 2, context, "expected schema_version == 2")
                    _require_sdpo_record_env_name(record, context)
                    sdpo_records += 1
                    _require_preflight_flag(record, context)
                    _require_sdpo_weights_follow_loss_mask(record, context)
                    _require_sdpo_rollout_is_weights_follow_sdpo_weights(
                        record,
                        context,
                        rollout_is_threshold=rollout_is_threshold,
                    )
                    sample_id_records += _count_sample_id_record(record, context)
                    if record.get("preflight_only") is not True:
                        _add_unique_sample_key(
                            final_sdpo_sample_keys,
                            _sample_key(file.parent, record, context),
                            context,
                            "final SDPO",
                        )
                    temperature_rows += _count_weighted_temperature_rows(record, context)
                    transported_rows += _count_optional_supported_rows(
                        record, "sdpo", context, expected_topk=expected_topk
                    )
                    student_rows += _count_optional_supported_rows(
                        record, "sdpo_student", context, expected_topk=expected_topk
                    )
                    record_paired_rows = _count_paired_supported_rows(record, context)
                    paired_rows += record_paired_rows
                    if record_paired_rows > 0:
                        paired_step_dirs.add(file.parent)
                    record_matching_indices = _matching_support_row_indices(record, context)
                    matching_support_rows += len(record_matching_indices)
                    record_distinct_teacher_indices = _distinct_teacher_logprob_row_indices(
                        record,
                        record_matching_indices,
                        context,
                    )
                    distinct_teacher_logprob_rows += len(record_distinct_teacher_indices)
                    record_importance_ratio_indices = _importance_ratio_evidence_row_indices(record, context)
                    if require_importance_ratio_evidence and record.get("preflight_only") is not True:
                        missing_ratio_indices = tuple(
                            sorted(
                                set(_weighted_sdpo_row_indices(record, context)) - set(record_importance_ratio_indices)
                            )
                        )
                        _require(
                            not missing_ratio_indices,
                            context,
                            "final weighted SDPO rows are missing rollout-IS ratio evidence "
                            f"at token position(s) {list(missing_ratio_indices)}",
                        )
                    _require_sdpo_rollout_is_weights_match_ratio(
                        record,
                        context,
                        rollout_is_threshold=rollout_is_threshold,
                        rollout_is=rollout_is,
                    )
                    importance_ratio_rows += len(record_importance_ratio_indices)
                    record_rollout_is_weight_indices = _rollout_is_weight_row_indices(record, context)
                    rollout_is_weight_rows += len(record_rollout_is_weight_indices)
                    if record_matching_indices:
                        matching_support_step_dirs.add(file.parent)
                        sample_key = _sample_key(file.parent, record, context)
                        matching_support_sample_signatures[sample_key] = _sample_signature(record, context)
                        _add_unique_sample_key(
                            matching_support_sample_keys,
                            sample_key,
                            context,
                            "matching final SDPO support",
                        )
                        matching_support_row_keys.update(
                            _sample_row_key(sample_key, idx) for idx in record_matching_indices
                        )
                        distinct_teacher_logprob_row_keys.update(
                            _sample_row_key(sample_key, idx) for idx in record_distinct_teacher_indices
                        )
                        importance_ratio_row_keys.update(
                            _sample_row_key(sample_key, idx) for idx in record_importance_ratio_indices
                        )
                        rollout_is_weight_row_keys.update(
                            _sample_row_key(sample_key, idx) for idx in record_rollout_is_weight_indices
                        )
                        matching_support_row_signatures.update(
                            _support_row_signatures(record, "sdpo", sample_key, record_matching_indices, context)
                        )
                    record_preflight_indices = _student_preflight_row_indices(
                        record,
                        context,
                        require_preflight_flag=require_student_preflight,
                    )
                    student_preflight_rows += len(record_preflight_indices)
                    if record_preflight_indices:
                        student_preflight_step_dirs.add(file.parent)
                        sample_key = _sample_key(file.parent, record, context)
                        student_preflight_sample_signatures[sample_key] = _sample_signature(record, context)
                        _add_unique_sample_key(
                            student_preflight_sample_keys,
                            sample_key,
                            context,
                            "preflight-only student support",
                        )
                        student_preflight_row_keys.update(
                            _sample_row_key(sample_key, idx) for idx in record_preflight_indices
                        )
                        student_preflight_row_signatures.update(
                            _support_row_signatures(
                                record, "sdpo_student", sample_key, record_preflight_indices, context
                            )
                        )
                    _require_final_transported_support_rows(record, context)

    _require(records > 0, str(path), "found no token export records")
    _require(sdpo_records > 0, str(path), "found no records with nonzero sdpo_weights")
    _require(sample_id_records > 0, str(path), "found no SDPO records with exported sample_id")
    _require(temperature_rows > 0, str(path), "found no supported SDPO records with exported temperatures")
    _require(transported_rows > 0, str(path), "found no supported transported sdpo_topk rows")
    _require(student_rows > 0, str(path), "found no supported trainer-forward sdpo_student_topk rows")
    _require(
        paired_rows > 0,
        str(path),
        "found no records with transported and trainer-forward SDPO support on the same weighted token positions",
    )
    if require_student_preflight:
        _require(
            student_preflight_rows > 0,
            str(path),
            "found no preflight-only trainer-forward student support rows",
        )
        _require(
            matching_support_rows > 0,
            str(path),
            "found no final SDPO rows where transported support ids match trainer-forward student support ids",
        )
        matched_support_sample_keys = tuple(sorted(student_preflight_sample_keys & matching_support_sample_keys))
        _require(
            bool(matched_support_sample_keys),
            str(path),
            "preflight-only student support rows and matching final SDPO support rows do not overlap on "
            "any same-step sample_id",
        )
        missing_final_support_sample_keys = tuple(sorted(student_preflight_sample_keys - matching_support_sample_keys))
        _require(
            not missing_final_support_sample_keys,
            str(path),
            "preflight-only student support rows are missing matching final SDPO support for same-step sample_id(s): "
            f"{list(missing_final_support_sample_keys)}",
        )
        unexpected_final_support_sample_keys = tuple(
            sorted(matching_support_sample_keys - student_preflight_sample_keys)
        )
        _require(
            not unexpected_final_support_sample_keys,
            str(path),
            "matching final SDPO support rows have no preflight-only student support for same-step sample_id(s): "
            f"{list(unexpected_final_support_sample_keys)}",
        )
        matched_support_row_keys = tuple(sorted(student_preflight_row_keys & matching_support_row_keys))
        missing_final_support_row_keys = tuple(sorted(student_preflight_row_keys - matching_support_row_keys))
        _require(
            not missing_final_support_row_keys,
            str(path),
            "preflight-only student support rows are missing matching final SDPO support for same-step token row(s): "
            f"{list(missing_final_support_row_keys)}",
        )
        unexpected_final_support_row_keys = tuple(sorted(matching_support_row_keys - student_preflight_row_keys))
        _require(
            not unexpected_final_support_row_keys,
            str(path),
            "matching final SDPO support rows have no preflight-only student support for same-step token row(s): "
            f"{list(unexpected_final_support_row_keys)}",
        )
        signature_mismatches = [
            sample_key
            for sample_key in matched_support_sample_keys
            if student_preflight_sample_signatures[sample_key] != matching_support_sample_signatures[sample_key]
        ]
        _require(
            not signature_mismatches,
            str(path),
            "preflight-only student support rows and matching final SDPO support rows have different "
            f"sample signatures for same-step sample_id(s): {signature_mismatches}",
        )
        row_signature_mismatches = [
            row_key
            for row_key in matched_support_row_keys
            if student_preflight_row_signatures[row_key] != matching_support_row_signatures[row_key]
        ]
        _require(
            not row_signature_mismatches,
            str(path),
            "preflight-only student support rows and matching final SDPO support rows have different "
            f"support ids for same-step token row(s): {row_signature_mismatches}",
        )
        matched_distinct_teacher_logprob_row_keys = tuple(
            sorted(set(matched_support_row_keys) & distinct_teacher_logprob_row_keys)
        )
        _require(
            bool(matched_distinct_teacher_logprob_row_keys),
            str(path),
            "found no matching final SDPO rows where transported teacher logprobs differ from "
            "trainer-forward student logprobs; this does not prove teacher-conditioned scoring ran.",
        )
        if require_importance_ratio_evidence:
            missing_importance_ratio_row_keys = tuple(sorted(set(matched_support_row_keys) - importance_ratio_row_keys))
            _require(
                not missing_importance_ratio_row_keys,
                str(path),
                "matching final SDPO rows are missing rollout-IS ratio evidence "
                f"(log_importance_ratio, importance_ratio, prob_delta): {list(missing_importance_ratio_row_keys)}",
            )
    else:
        matched_support_sample_keys = tuple(sorted(student_preflight_sample_keys & matching_support_sample_keys))
        matched_support_row_keys = tuple(sorted(student_preflight_row_keys & matching_support_row_keys))

    if require_importance_ratio_evidence:
        _require(
            importance_ratio_rows > 0,
            str(path),
            "found no final SDPO rows with rollout-IS ratio evidence "
            "(log_importance_ratio, importance_ratio, prob_delta)",
        )

    return SDPOExportStats(
        files=len(files),
        records=records,
        sdpo_records=sdpo_records,
        transported_rows=transported_rows,
        student_rows=student_rows,
        paired_rows=paired_rows,
        matching_support_rows=matching_support_rows,
        distinct_teacher_logprob_rows=distinct_teacher_logprob_rows,
        importance_ratio_rows=importance_ratio_rows,
        rollout_is_weight_rows=rollout_is_weight_rows,
        student_preflight_rows=student_preflight_rows,
        temperature_rows=temperature_rows,
        sample_id_records=sample_id_records,
        stable_steps=stable_steps,
        step_names=_step_names(step_dirs),
        paired_step_names=_step_names(paired_step_dirs),
        matching_support_step_names=_step_names(matching_support_step_dirs),
        student_preflight_step_names=_step_names(student_preflight_step_dirs),
        matching_support_sample_keys=tuple(sorted(matching_support_sample_keys)),
        student_preflight_sample_keys=tuple(sorted(student_preflight_sample_keys)),
        matched_support_sample_keys=matched_support_sample_keys,
        matching_support_row_keys=tuple(sorted(matching_support_row_keys)),
        importance_ratio_row_keys=tuple(sorted(importance_ratio_row_keys)),
        rollout_is_weight_row_keys=tuple(sorted(rollout_is_weight_row_keys)),
        student_preflight_row_keys=tuple(sorted(student_preflight_row_keys)),
        distinct_teacher_logprob_row_keys=tuple(sorted(distinct_teacher_logprob_row_keys)),
        matched_support_row_keys=matched_support_row_keys,
    )


def verify_sdpo_ema_broadcasts(path: Path, *, role: str = "sdpo_teacher") -> SDPOEMABroadcastStats:
    step_dirs = find_broadcast_step_dirs(path)
    teacher_steps = 0

    for step_dir in step_dirs:
        teacher_dir = step_dir / role
        teacher_stable = teacher_dir / "STABLE"
        _require(teacher_stable.exists(), str(step_dir), f"missing {role}/STABLE")
        _require(
            any(
                child.name != "STABLE" and child.is_file() and child.stat().st_size > 0
                for child in teacher_dir.iterdir()
            ),
            str(teacher_dir),
            "found STABLE marker but no non-empty teacher model artifacts",
        )
        teacher_steps += 1

    return SDPOEMABroadcastStats(
        steps=len(step_dirs),
        teacher_steps=teacher_steps,
        role=role,
        step_names=_step_names(step_dirs),
        step_keys=_broadcast_step_keys(step_dirs),
    )


def verify_sdpo_smoke_artifacts(
    path: Path,
    *,
    require_ema_teacher: bool = False,
    expected_topk: int | None = None,
) -> SDPOSmokeArtifactStats:
    _require(
        expected_topk is not None,
        str(path),
        "SDPO smoke artifact verification requires expected_topk",
    )
    _validate_expected_topk(expected_topk, str(path))
    token_exports = verify_sdpo_token_exports(
        path,
        require_stable=True,
        require_student_preflight=True,
        require_importance_ratio_evidence=True,
        expected_topk=expected_topk,
        rollout_is_threshold=2.0,
        rollout_is="token",
    )
    if not require_ema_teacher:
        return SDPOSmokeArtifactStats(token_exports=token_exports)

    ema_broadcasts = verify_sdpo_ema_broadcasts(path)
    required_export_step_keys = _ema_required_token_export_step_keys(token_exports.matching_support_sample_keys)
    broadcast_step_keys = set(ema_broadcasts.step_keys)
    matched_step_keys = tuple(step_key for step_key in required_export_step_keys if step_key in broadcast_step_keys)
    missing_broadcast_step_keys = tuple(
        step_key for step_key in required_export_step_keys if step_key not in broadcast_step_keys
    )
    matched_steps = tuple(sorted({_step_name_from_key(step_key) for step_key in matched_step_keys}))
    _require(
        bool(required_export_step_keys),
        str(path),
        "found no post-initial matching SDPO token-export steps to compare with EMA teacher broadcasts "
        f"(matching_token_export_step_keys={list(_ema_token_export_step_keys(token_exports.matching_support_sample_keys))})",
    )
    _require(
        bool(matched_step_keys),
        str(path),
        "matching SDPO token exports and EMA teacher broadcasts do not overlap on any run/step "
        f"(matching_token_export_step_keys={list(_ema_token_export_step_keys(token_exports.matching_support_sample_keys))}, "
        f"broadcast_step_keys={list(ema_broadcasts.step_keys)})",
    )
    _require(
        not missing_broadcast_step_keys,
        str(path),
        "matching SDPO token-export run/steps are missing EMA teacher broadcasts: "
        f"{list(missing_broadcast_step_keys)} "
        f"(matching_token_export_step_keys={list(_ema_token_export_step_keys(token_exports.matching_support_sample_keys))}, "
        f"broadcast_step_keys={list(ema_broadcasts.step_keys)})",
    )
    return SDPOSmokeArtifactStats(
        token_exports=token_exports,
        ema_broadcasts=ema_broadcasts,
        matched_steps=matched_steps,
        matched_step_keys=matched_step_keys,
    )


def _loads_json_object_no_duplicate_keys(line: str, context: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in pairs:
            if key in decoded:
                raise ValueError(f"{context}: duplicate JSON object key: {key}")
            decoded[key] = value
        return decoded

    return json.loads(line, object_pairs_hook=reject_duplicate_keys)


def _validate_expected_topk(expected_topk: int, context: str) -> None:
    _require(
        not isinstance(expected_topk, bool) and isinstance(expected_topk, int),
        context,
        "expected_topk must be an integer",
    )
    _require(expected_topk > 0, context, "expected_topk must be positive")


def _validate_rollout_is_threshold(rollout_is_threshold: float, context: str) -> None:
    _require(
        not isinstance(rollout_is_threshold, bool) and isinstance(rollout_is_threshold, (int, float)),
        context,
        "rollout_is_threshold must be numeric",
    )
    _require(math.isfinite(float(rollout_is_threshold)), context, "rollout_is_threshold must be finite")
    _require(rollout_is_threshold > 0, context, "rollout_is_threshold must be positive")


def _validate_rollout_is(rollout_is: str, context: str) -> None:
    _require(rollout_is in {"token", "sequence"}, context, "rollout_is must be 'token' or 'sequence'")


def _ema_token_export_step_keys(sample_keys: tuple[str, ...]) -> tuple[str, ...]:
    step_keys = {_sample_key_to_step_key(sample_key) for sample_key in sample_keys}
    return tuple(sorted(step_keys))


def _ema_required_token_export_step_keys(sample_keys: tuple[str, ...]) -> tuple[str, ...]:
    # Filesystem broadcasts start after the first optimizer update. A matching
    # SDPO export at step_0 can be valid, but later matching exports must have
    # the corresponding EMA teacher broadcast from the same run.
    return tuple(
        step_key for step_key in _ema_token_export_step_keys(sample_keys) if _step_name_from_key(step_key) != "step_0"
    )


def _sample_key_to_step_key(sample_key: str) -> str:
    run_id, step_name, *_ = sample_key.split(":", 2)
    return f"{run_id}:{step_name}"


def _step_name_from_key(step_key: str) -> str:
    return step_key.split(":", 1)[1]


def _is_sdpo_record(record: dict[str, Any], context: str) -> bool:
    weights = record.get("sdpo_weights")
    if weights is None:
        return False
    _require(isinstance(weights, list), context, "sdpo_weights must be a list")
    _require_no_boolean_sdpo_weights(weights, context)
    return has_active_sdpo_weights(weights)


def _require_no_boolean_sdpo_weights(weights: list[Any], context: str) -> None:
    for idx, weight in enumerate(weights):
        _require(
            not isinstance(weight, bool),
            context,
            f"sdpo_weights must contain finite numeric values at token {idx}",
        )


def _require_inactive_records_do_not_carry_sdpo_support(record: dict[str, Any], context: str) -> None:
    weights = record.get("sdpo_weights")
    if weights is not None:
        _require(isinstance(weights, list), context, "sdpo_weights must be a list")
        _require_no_boolean_sdpo_weights(weights, context)
        if has_active_sdpo_weights(weights):
            return
    for prefix in ("sdpo", "sdpo_student"):
        topk_token_ids = record.get(f"{prefix}_topk_token_ids")
        topk_logprobs = record.get(f"{prefix}_topk_logprobs")
        if topk_token_ids is None and topk_logprobs is None:
            continue
        _require(isinstance(topk_token_ids, list), context, f"{prefix}_topk_token_ids must be a list when present")
        _require(isinstance(topk_logprobs, list), context, f"{prefix}_topk_logprobs must be a list when present")
        _require(
            not any(row is not None for row in topk_token_ids + topk_logprobs),
            context,
            f"inactive SDPO record must not carry {prefix} top-k support rows",
        )


def _require_sdpo_weights_follow_loss_mask(record: dict[str, Any], context: str) -> None:
    token_ids = record.get("token_ids")
    position_ids = record.get("position_ids")
    loss_mask = record.get("loss_mask")
    weights = record.get("sdpo_weights")
    _require(isinstance(token_ids, list), context, "token_ids must be a list")
    _require(isinstance(position_ids, list), context, "position_ids must be a list")
    _require(isinstance(loss_mask, list), context, "loss_mask must be a list")
    _require(isinstance(weights, list), context, "sdpo_weights must be a list")
    _require_int_values(token_ids, context, "token_ids")
    _require_int_values(position_ids, context, "position_ids")
    _require(all(isinstance(value, bool) for value in loss_mask), context, "loss_mask must contain booleans")
    _require_numeric_values(weights, context, "sdpo_weights", allow_none=True)
    _require_non_negative_values(weights, context, "sdpo_weights")
    expected_len = len(token_ids)
    _require(
        len(position_ids) == expected_len,
        context,
        f"position_ids length {len(position_ids)} != token_ids length {expected_len}",
    )
    _require(
        len(loss_mask) == expected_len, context, f"loss_mask length {len(loss_mask)} != token_ids length {expected_len}"
    )
    _require(
        len(weights) == expected_len,
        context,
        f"sdpo_weights length {len(weights)} != token_ids length {expected_len}",
    )
    for idx, (weight, trains) in enumerate(zip(weights, loss_mask, strict=True)):
        _require(
            not is_active_sdpo_weight(weight) or bool(trains),
            context,
            f"sdpo weight at token {idx} is nonzero outside loss_mask",
        )


def _require_sdpo_rollout_is_weights_follow_sdpo_weights(
    record: dict[str, Any],
    context: str,
    *,
    rollout_is_threshold: float | None,
) -> None:
    token_ids = record.get("token_ids")
    weights = record.get("sdpo_weights")
    rollout_is_weights = record.get("sdpo_rollout_is_weights")
    require_final_rollout_is_weights = rollout_is_threshold is not None and record.get("preflight_only") is not True
    if rollout_is_weights is None:
        if require_final_rollout_is_weights:
            missing_indices = _weighted_sdpo_row_indices(record, context)
            _require(
                not missing_indices,
                context,
                "final weighted SDPO rows are missing sdpo_rollout_is_weights "
                f"at token position(s) {list(missing_indices)}",
            )
        return
    _require(isinstance(token_ids, list), context, "token_ids must be a list")
    _require(isinstance(weights, list), context, "sdpo_weights must be a list")
    _require(isinstance(rollout_is_weights, list), context, "sdpo_rollout_is_weights must be a list when present")
    expected_len = len(token_ids)
    _require(
        len(rollout_is_weights) == expected_len,
        context,
        f"sdpo_rollout_is_weights length {len(rollout_is_weights)} != token_ids length {expected_len}",
    )
    _require_numeric_values(rollout_is_weights, context, "sdpo_rollout_is_weights", allow_none=True)
    for idx, (rollout_is_weight, sdpo_weight) in enumerate(zip(rollout_is_weights, weights, strict=True)):
        if require_final_rollout_is_weights and is_active_sdpo_weight(sdpo_weight):
            _require(
                is_active_sdpo_weight(rollout_is_weight),
                context,
                f"sdpo_rollout_is_weights[{idx}] is missing on final SDPO component token",
            )
        if not is_active_sdpo_weight(rollout_is_weight):
            continue
        _require_finite_float_value(rollout_is_weight, context, f"sdpo_rollout_is_weights[{idx}]")
        _require(
            rollout_is_weight >= 0,
            context,
            f"sdpo_rollout_is_weights[{idx}] must be non-negative",
        )
        if rollout_is_threshold is not None:
            _require(
                rollout_is_weight <= float(rollout_is_threshold) + 1e-6,
                context,
                f"sdpo_rollout_is_weights[{idx}] exceeds rollout_is_threshold={float(rollout_is_threshold)}",
            )
        _require(
            is_active_sdpo_weight(sdpo_weight),
            context,
            f"sdpo_rollout_is_weights[{idx}] is nonzero outside SDPO component",
        )


def _require_sdpo_rollout_is_weights_match_ratio(
    record: dict[str, Any],
    context: str,
    *,
    rollout_is_threshold: float | None,
    rollout_is: str | None,
) -> None:
    if rollout_is is None:
        return
    if record.get("preflight_only") is True:
        return
    _require(
        rollout_is_threshold is not None,
        context,
        "rollout_is matching requires rollout_is_threshold",
    )
    token_ids = record.get("token_ids")
    weights = record.get("sdpo_weights")
    rollout_is_weights = record.get("sdpo_rollout_is_weights")
    log_importance_ratio = record.get("log_importance_ratio")
    _require(isinstance(token_ids, list), context, "token_ids must be a list")
    _require(isinstance(weights, list), context, "sdpo_weights must be a list")
    _require(isinstance(rollout_is_weights, list), context, "sdpo_rollout_is_weights must be a list when present")
    _require(isinstance(log_importance_ratio, list), context, "log_importance_ratio must be a list")
    expected_len = len(token_ids)
    for name, values in (
        ("sdpo_weights", weights),
        ("sdpo_rollout_is_weights", rollout_is_weights),
        ("log_importance_ratio", log_importance_ratio),
    ):
        _require(
            len(values) == expected_len,
            context,
            f"{name} length {len(values)} != token_ids length {expected_len}",
        )

    active_indices = _weighted_sdpo_row_indices(record, context)
    if not active_indices:
        return

    if rollout_is == "token":
        expected_by_idx = {
            idx: min(
                math.exp(_finite_row_value(log_importance_ratio[idx], "log_importance_ratio", idx, context)),
                float(rollout_is_threshold),
            )
            for idx in active_indices
        }
    else:
        seq_log_ratio = math.fsum(
            _finite_row_value(log_importance_ratio[idx], "log_importance_ratio", idx, context) for idx in active_indices
        )
        expected = min(math.exp(seq_log_ratio), float(rollout_is_threshold))
        expected_by_idx = {idx: expected for idx in active_indices}

    for idx, expected in expected_by_idx.items():
        rollout_is_weight = _finite_row_value(rollout_is_weights[idx], "sdpo_rollout_is_weights", idx, context)
        _require(
            _close_float(rollout_is_weight, expected),
            context,
            f"sdpo_rollout_is_weights[{idx}] does not match {rollout_is} rollout-IS "
            f"min(exp(log_importance_ratio), rollout_is_threshold)",
        )


def _token_export_step_dirs(files: list[Path]) -> set[Path]:
    return {file.parent for file in files}


def _count_stable_token_export_steps(step_dirs: set[Path], *, require_stable: bool) -> int:
    stable_steps = 0
    for step_dir in step_dirs:
        stable = (step_dir / "STABLE").exists()
        if stable:
            stable_steps += 1
        elif require_stable:
            raise ValueError(f"{step_dir}: missing token export STABLE")
    return stable_steps


def _step_names(step_dirs: set[Path] | list[Path]) -> tuple[str, ...]:
    return tuple(sorted(step_dir.name for step_dir in step_dirs))


def _broadcast_step_keys(step_dirs: set[Path] | list[Path]) -> tuple[str, ...]:
    return tuple(sorted(_broadcast_step_key(step_dir) for step_dir in step_dirs))


def _broadcast_step_key(step_dir: Path) -> str:
    run_id = step_dir.parent.parent.name if step_dir.parent.name == "broadcasts" else "<unknown-run>"
    return f"{run_id}:{step_dir.name}"


def _count_optional_supported_rows(
    record: dict[str, Any],
    prefix: str,
    context: str,
    *,
    expected_topk: int | None,
) -> int:
    token_ids = record.get("token_ids")
    weights = record.get("sdpo_weights")
    topk_token_ids = record.get(f"{prefix}_topk_token_ids")
    topk_logprobs = record.get(f"{prefix}_topk_logprobs")

    _require(isinstance(token_ids, list), context, "token_ids must be a list")
    _require(isinstance(weights, list), context, "sdpo_weights must be a list")
    if topk_token_ids is None and topk_logprobs is None:
        return 0
    _require(isinstance(topk_token_ids, list), context, f"{prefix}_topk_token_ids must be a list when present")
    _require(isinstance(topk_logprobs, list), context, f"{prefix}_topk_logprobs must be a list when present")
    expected_len = len(token_ids)
    for name, rows in (
        ("sdpo_weights", weights),
        (f"{prefix}_topk_token_ids", topk_token_ids),
        (f"{prefix}_topk_logprobs", topk_logprobs),
    ):
        _require(len(rows) == expected_len, context, f"{name} length {len(rows)} != token_ids length {expected_len}")

    supported = 0
    missing_weighted_rows: list[int] = []
    for idx, (ids, logprobs) in enumerate(zip(topk_token_ids, topk_logprobs, strict=True)):
        _require_optional_support_row(ids, logprobs, prefix, idx, context, expected_topk=expected_topk)
    for idx, weight in enumerate(weights):
        ids = topk_token_ids[idx]
        logprobs = topk_logprobs[idx]
        if not is_active_sdpo_weight(weight):
            _require_unweighted_support_placeholder(ids, logprobs, prefix, idx, context)
            continue
        if ids is None and logprobs is None:
            missing_weighted_rows.append(idx)
            continue
        _require(
            not _is_placeholder_logprob_row(logprobs),
            context,
            f"{prefix}_topk_logprobs[{idx}] looks like an unfilled placeholder row",
        )
        supported += 1
    if prefix == "sdpo" and supported > 0:
        _require(
            record.get("preflight_only") is not True,
            context,
            "transported teacher SDPO support rows require preflight_only=false",
        )
    _require(
        supported == 0 or not missing_weighted_rows,
        context,
        f"{prefix} support is missing at weighted token positions {missing_weighted_rows}",
    )
    return supported


def _require_final_transported_support_rows(record: dict[str, Any], context: str) -> None:
    if record.get("preflight_only") is True:
        return
    token_ids = record.get("token_ids")
    weights = record.get("sdpo_weights")
    _require(isinstance(token_ids, list), context, "token_ids must be a list")
    _require(isinstance(weights, list), context, "sdpo_weights must be a list")
    expected_len = len(token_ids)
    _require(
        len(weights) == expected_len,
        context,
        f"sdpo_weights length {len(weights)} != token_ids length {expected_len}",
    )
    transported = _optional_support_columns(record, "sdpo", expected_len, context)
    _require(
        transported is not None,
        context,
        "final SDPO records require transported teacher support columns",
    )
    transported_ids, transported_logprobs = transported
    missing_weighted_rows = [
        idx
        for idx, weight in enumerate(weights)
        if is_active_sdpo_weight(weight) and not _has_support_row(transported_ids[idx], transported_logprobs[idx])
    ]
    _require(
        not missing_weighted_rows,
        context,
        f"final SDPO records require transported teacher support at weighted token positions {missing_weighted_rows}",
    )


def _require_unweighted_support_placeholder(
    ids: Any,
    logprobs: Any,
    prefix: str,
    idx: int,
    context: str,
) -> None:
    if ids is None and logprobs is None:
        return
    _require(
        isinstance(ids, list) and isinstance(logprobs, list),
        context,
        f"{prefix} support at unweighted token {idx} must be null or an all-zero placeholder",
    )
    _require(
        _is_placeholder_token_id_row(ids) and _is_placeholder_logprob_row(logprobs),
        context,
        f"{prefix} support at unweighted token {idx} must be null or an all-zero placeholder",
    )


def _count_sample_id_record(record: dict[str, Any], context: str) -> int:
    sample_id = record.get("sample_id")
    _require(_is_non_blank_string(sample_id), context, "sample_id must be a non-empty string")
    return 1


def _require_sdpo_record_env_name(record: dict[str, Any], context: str) -> None:
    _require(_is_non_blank_string(record.get("env_name")), context, "env_name must be a non-empty string")


def _sample_key(step_dir: Path, record: dict[str, Any], context: str) -> str:
    sample_id = record.get("sample_id")
    _require(_is_non_blank_string(sample_id), context, "sample_id must be a non-empty string")
    run_id = step_dir.parent.parent.name if step_dir.parent.name == "token_exports" else "<unknown-run>"
    return f"{run_id}:{step_dir.name}:{sample_id}"


def _sample_signature(record: dict[str, Any], context: str) -> tuple[Any, ...]:
    env_name = record.get("env_name")
    token_ids = record.get("token_ids")
    position_ids = record.get("position_ids")
    loss_mask = record.get("loss_mask")
    temperatures = record.get("temperatures")
    sdpo_weights = record.get("sdpo_weights")
    _require(_is_non_blank_string(env_name), context, "env_name must be a non-empty string")
    _require(isinstance(token_ids, list), context, "token_ids must be a list")
    _require(isinstance(position_ids, list), context, "position_ids must be a list")
    _require(isinstance(loss_mask, list), context, "loss_mask must be a list")
    _require(isinstance(temperatures, list), context, "temperatures must be a list")
    _require(isinstance(sdpo_weights, list), context, "sdpo_weights must be a list")
    return (
        env_name,
        tuple(token_ids),
        tuple(position_ids),
        tuple(loss_mask),
        tuple(temperatures),
        tuple(sdpo_weights),
    )


def _sample_row_key(sample_key: str, idx: int) -> str:
    return f"{sample_key}:token-{idx}"


def _support_row_signatures(
    record: dict[str, Any],
    prefix: str,
    sample_key: str,
    indices: tuple[int, ...],
    context: str,
) -> dict[str, tuple[Any, ...]]:
    token_ids = record.get("token_ids")
    _require(isinstance(token_ids, list), context, "token_ids must be a list")
    support = _optional_support_columns(record, prefix, len(token_ids), context)
    _require(support is not None, context, f"{prefix} support columns must be present")
    topk_token_ids, topk_logprobs = support
    signatures: dict[str, tuple[Any, ...]] = {}
    for idx in indices:
        _require(
            _has_support_row(topk_token_ids[idx], topk_logprobs[idx]),
            context,
            f"{prefix} support row {idx} must be present",
        )
        signatures[_sample_row_key(sample_key, idx)] = tuple(topk_token_ids[idx])
    return signatures


def _add_unique_sample_key(keys: set[str], key: str, context: str, category: str) -> None:
    _require(key not in keys, context, f"duplicate {category} record for {key}")
    keys.add(key)


def _count_weighted_temperature_rows(record: dict[str, Any], context: str) -> int:
    token_ids = record.get("token_ids")
    weights = record.get("sdpo_weights")
    temperatures = record.get("temperatures")

    _require(isinstance(token_ids, list), context, "token_ids must be a list")
    _require(isinstance(weights, list), context, "sdpo_weights must be a list")
    _require(isinstance(temperatures, list), context, "temperatures must be a list")
    expected_len = len(token_ids)
    _require(
        len(weights) == expected_len, context, f"sdpo_weights length {len(weights)} != token_ids length {expected_len}"
    )
    _require(
        len(temperatures) == expected_len,
        context,
        f"temperatures length {len(temperatures)} != token_ids length {expected_len}",
    )

    weighted_rows = 0
    for idx, weight in enumerate(weights):
        if not is_active_sdpo_weight(weight):
            continue
        temperature = temperatures[idx]
        _require(
            _is_finite_number(temperature) and temperature > 0,
            context,
            f"temperatures[{idx}] must be a positive number",
        )
        weighted_rows += 1
    return weighted_rows


def _count_paired_supported_rows(record: dict[str, Any], context: str) -> int:
    token_ids = record.get("token_ids")
    weights = record.get("sdpo_weights")
    _require(isinstance(token_ids, list), context, "token_ids must be a list")
    _require(isinstance(weights, list), context, "sdpo_weights must be a list")
    expected_len = len(token_ids)
    _require(
        len(weights) == expected_len, context, f"sdpo_weights length {len(weights)} != token_ids length {expected_len}"
    )

    transported = _optional_support_columns(record, "sdpo", expected_len, context)
    student = _optional_support_columns(record, "sdpo_student", expected_len, context)
    if transported is None or student is None:
        return 0
    transported_ids, transported_logprobs = transported
    student_ids, student_logprobs = student

    paired = 0
    for idx, weight in enumerate(weights):
        if not is_active_sdpo_weight(weight):
            continue
        if not _has_support_row(transported_ids[idx], transported_logprobs[idx]):
            continue
        if not _has_support_row(student_ids[idx], student_logprobs[idx]):
            continue
        _require_same_support_width(transported_ids[idx], student_ids[idx], idx, context)
        paired += 1
    return paired


def _count_matching_support_rows(record: dict[str, Any], context: str) -> int:
    return len(_matching_support_row_indices(record, context))


def _matching_support_row_indices(record: dict[str, Any], context: str) -> tuple[int, ...]:
    token_ids = record.get("token_ids")
    weights = record.get("sdpo_weights")
    _require(isinstance(token_ids, list), context, "token_ids must be a list")
    _require(isinstance(weights, list), context, "sdpo_weights must be a list")
    expected_len = len(token_ids)
    _require(
        len(weights) == expected_len, context, f"sdpo_weights length {len(weights)} != token_ids length {expected_len}"
    )

    transported = _optional_support_columns(record, "sdpo", expected_len, context)
    student = _optional_support_columns(record, "sdpo_student", expected_len, context)
    if transported is None or student is None:
        return ()
    transported_ids, transported_logprobs = transported
    student_ids, student_logprobs = student

    matched: list[int] = []
    for idx, weight in enumerate(weights):
        if not is_active_sdpo_weight(weight):
            continue
        if not _has_support_row(transported_ids[idx], transported_logprobs[idx]):
            continue
        if not _has_support_row(student_ids[idx], student_logprobs[idx]):
            continue
        _require_same_support_width(transported_ids[idx], student_ids[idx], idx, context)
        if transported_ids[idx] == student_ids[idx]:
            matched.append(idx)
    return tuple(matched)


def _distinct_teacher_logprob_row_indices(
    record: dict[str, Any],
    matching_indices: tuple[int, ...],
    context: str,
) -> tuple[int, ...]:
    token_ids = record.get("token_ids")
    _require(isinstance(token_ids, list), context, "token_ids must be a list")
    expected_len = len(token_ids)
    transported = _optional_support_columns(record, "sdpo", expected_len, context)
    student = _optional_support_columns(record, "sdpo_student", expected_len, context)
    if transported is None or student is None:
        return ()
    _, transported_logprobs = transported
    _, student_logprobs = student

    distinct: list[int] = []
    for idx in matching_indices:
        if not _logprob_rows_close(transported_logprobs[idx], student_logprobs[idx]):
            distinct.append(idx)
    return tuple(distinct)


def _importance_ratio_evidence_row_indices(record: dict[str, Any], context: str) -> tuple[int, ...]:
    if record.get("preflight_only") is True:
        return ()
    token_ids = record.get("token_ids")
    weights = record.get("sdpo_weights")
    _require(isinstance(token_ids, list), context, "token_ids must be a list")
    _require(isinstance(weights, list), context, "sdpo_weights must be a list")
    expected_len = len(token_ids)
    _require(
        len(weights) == expected_len,
        context,
        f"sdpo_weights length {len(weights)} != token_ids length {expected_len}",
    )
    columns = {
        name: record.get(name)
        for name in (
            "log_importance_ratio",
            "importance_ratio",
            "prob_delta",
        )
    }
    if all(values is None for values in columns.values()):
        return ()
    trainer_logprobs = record.get("trainer_logprobs")
    inference_logprobs = record.get("inference_logprobs")
    _require(isinstance(trainer_logprobs, list), context, "trainer_logprobs must be a list")
    _require(isinstance(inference_logprobs, list), context, "inference_logprobs must be a list")
    _require(
        len(trainer_logprobs) == expected_len,
        context,
        f"trainer_logprobs length {len(trainer_logprobs)} != token_ids length {expected_len}",
    )
    _require(
        len(inference_logprobs) == expected_len,
        context,
        f"inference_logprobs length {len(inference_logprobs)} != token_ids length {expected_len}",
    )
    for name, values in columns.items():
        _require(isinstance(values, list), context, f"{name} must be a list")
        _require(
            len(values) == expected_len, context, f"{name} length {len(values)} != token_ids length {expected_len}"
        )

    supported: list[int] = []
    for idx, weight in enumerate(weights):
        if not is_active_sdpo_weight(weight):
            continue
        row = {name: values[idx] for name, values in columns.items()}
        if all(value is None for value in row.values()):
            continue
        for name, value in row.items():
            _require_finite_float_value(value, context, f"{name}[{idx}]")
        trainer_logprob = trainer_logprobs[idx]
        inference_logprob = inference_logprobs[idx]
        _require_finite_float_value(trainer_logprob, context, f"trainer_logprobs[{idx}]")
        _require_finite_float_value(inference_logprob, context, f"inference_logprobs[{idx}]")
        expected_log_ratio = float(trainer_logprob) - float(inference_logprob)
        _require(
            _close_float(row["log_importance_ratio"], expected_log_ratio),
            context,
            f"log_importance_ratio[{idx}] does not match trainer_logprobs - inference_logprobs",
        )
        _require(
            _close_float(row["importance_ratio"], math.exp(expected_log_ratio)),
            context,
            f"importance_ratio[{idx}] does not match exp(log_importance_ratio)",
        )
        expected_prob_delta = math.exp(float(trainer_logprob)) - math.exp(float(inference_logprob))
        _require(
            _close_float(row["prob_delta"], expected_prob_delta),
            context,
            f"prob_delta[{idx}] does not match exp(trainer_logprobs) - exp(inference_logprobs)",
        )
        _require(row["importance_ratio"] >= 0, context, f"importance_ratio[{idx}] must be non-negative")
        supported.append(idx)
    return tuple(supported)


def _rollout_is_weight_row_indices(record: dict[str, Any], context: str) -> tuple[int, ...]:
    if record.get("preflight_only") is True:
        return ()
    token_ids = record.get("token_ids")
    weights = record.get("sdpo_weights")
    rollout_is_weights = record.get("sdpo_rollout_is_weights")
    if rollout_is_weights is None:
        return ()
    _require(isinstance(token_ids, list), context, "token_ids must be a list")
    _require(isinstance(weights, list), context, "sdpo_weights must be a list")
    _require(isinstance(rollout_is_weights, list), context, "sdpo_rollout_is_weights must be a list when present")
    expected_len = len(token_ids)
    _require(
        len(weights) == expected_len,
        context,
        f"sdpo_weights length {len(weights)} != token_ids length {expected_len}",
    )
    _require(
        len(rollout_is_weights) == expected_len,
        context,
        f"sdpo_rollout_is_weights length {len(rollout_is_weights)} != token_ids length {expected_len}",
    )
    _require_numeric_values(rollout_is_weights, context, "sdpo_rollout_is_weights", allow_none=True)
    return tuple(
        idx
        for idx, (sdpo_weight, rollout_is_weight) in enumerate(zip(weights, rollout_is_weights, strict=True))
        if is_active_sdpo_weight(sdpo_weight) and is_active_sdpo_weight(rollout_is_weight)
    )


def _weighted_sdpo_row_indices(record: dict[str, Any], context: str) -> tuple[int, ...]:
    token_ids = record.get("token_ids")
    weights = record.get("sdpo_weights")
    _require(isinstance(token_ids, list), context, "token_ids must be a list")
    _require(isinstance(weights, list), context, "sdpo_weights must be a list")
    expected_len = len(token_ids)
    _require(
        len(weights) == expected_len,
        context,
        f"sdpo_weights length {len(weights)} != token_ids length {expected_len}",
    )
    return tuple(idx for idx, weight in enumerate(weights) if is_active_sdpo_weight(weight))


def _count_student_preflight_rows(record: dict[str, Any], context: str) -> int:
    return len(_student_preflight_row_indices(record, context, require_preflight_flag=False))


def _student_preflight_row_indices(
    record: dict[str, Any],
    context: str,
    *,
    require_preflight_flag: bool = False,
) -> tuple[int, ...]:
    token_ids = record.get("token_ids")
    weights = record.get("sdpo_weights")
    _require(isinstance(token_ids, list), context, "token_ids must be a list")
    _require(isinstance(weights, list), context, "sdpo_weights must be a list")
    expected_len = len(token_ids)
    _require(
        len(weights) == expected_len, context, f"sdpo_weights length {len(weights)} != token_ids length {expected_len}"
    )

    transported = _optional_support_columns(record, "sdpo", expected_len, context)
    student = _optional_support_columns(record, "sdpo_student", expected_len, context)
    if student is None:
        return ()
    transported_ids: list[Any] | None = None
    transported_logprobs: list[Any] | None = None
    if transported is not None:
        transported_ids, transported_logprobs = transported
    student_ids, student_logprobs = student

    rows: list[int] = []
    for idx, weight in enumerate(weights):
        if not is_active_sdpo_weight(weight):
            continue
        if transported_ids is not None and _has_support_row(transported_ids[idx], transported_logprobs[idx]):
            continue
        if _has_support_row(student_ids[idx], student_logprobs[idx]):
            if require_preflight_flag:
                _require(
                    record.get("preflight_only") is True,
                    context,
                    "preflight-only student support rows require preflight_only=true",
                )
            rows.append(idx)
    return tuple(rows)


def _require_preflight_flag(record: dict[str, Any], context: str) -> None:
    preflight_only = record.get("preflight_only")
    _require(
        isinstance(preflight_only, bool),
        context,
        "preflight_only must be a boolean",
    )


def _optional_support_columns(
    record: dict[str, Any], prefix: str, expected_len: int, context: str
) -> tuple[list[Any], list[Any]] | None:
    topk_token_ids = record.get(f"{prefix}_topk_token_ids")
    topk_logprobs = record.get(f"{prefix}_topk_logprobs")
    if topk_token_ids is None and topk_logprobs is None:
        return None
    _require(isinstance(topk_token_ids, list), context, f"{prefix}_topk_token_ids must be a list when present")
    _require(isinstance(topk_logprobs, list), context, f"{prefix}_topk_logprobs must be a list when present")
    _require(
        len(topk_token_ids) == expected_len,
        context,
        f"{prefix}_topk_token_ids length {len(topk_token_ids)} != token_ids length {expected_len}",
    )
    _require(
        len(topk_logprobs) == expected_len,
        context,
        f"{prefix}_topk_logprobs length {len(topk_logprobs)} != token_ids length {expected_len}",
    )
    return topk_token_ids, topk_logprobs


def _require_optional_support_row(
    ids: Any,
    logprobs: Any,
    prefix: str,
    idx: int,
    context: str,
    *,
    expected_topk: int | None,
) -> None:
    if ids is None and logprobs is None:
        return
    _require(isinstance(ids, list) and ids, context, f"{prefix}_topk_token_ids[{idx}] must be non-empty")
    _require(isinstance(logprobs, list) and logprobs, context, f"{prefix}_topk_logprobs[{idx}] must be non-empty")
    _require(
        len(ids) == len(logprobs),
        context,
        f"{prefix} row width mismatch at {idx}: ids={len(ids)} logprobs={len(logprobs)}",
    )
    _require_int_values(ids, context, f"{prefix}_topk_token_ids[{idx}]")
    _require_numeric_values(logprobs, context, f"{prefix}_topk_logprobs[{idx}]", allow_none=False)
    if expected_topk is not None:
        _require(
            len(ids) == expected_topk,
            context,
            f"{prefix} row width {len(ids)} at {idx} != expected_topk {expected_topk}",
        )
    if _is_placeholder_logprob_row(logprobs):
        _require(
            _is_placeholder_token_id_row(ids),
            context,
            f"{prefix}_topk_token_ids[{idx}] must be a zero placeholder row when logprobs are placeholders",
        )
        return
    _require_distinct_token_ids(ids, context, f"{prefix}_topk_token_ids[{idx}]")
    _require_logprob_row(logprobs, prefix, idx, context)


def _has_support_row(ids: Any, logprobs: Any) -> bool:
    return (
        isinstance(ids, list)
        and bool(ids)
        and isinstance(logprobs, list)
        and bool(logprobs)
        and len(ids) == len(logprobs)
        and not _is_placeholder_logprob_row(logprobs)
    )


def _require_same_support_width(transported_ids: list[Any], student_ids: list[Any], idx: int, context: str) -> None:
    _require(
        len(transported_ids) == len(student_ids),
        context,
        "transported and trainer-forward SDPO support widths differ at weighted token "
        f"{idx}: sdpo={len(transported_ids)} sdpo_student={len(student_ids)}",
    )


def _logprob_rows_close(left: list[Any], right: list[Any]) -> bool:
    if len(left) != len(right):
        return False
    return all(math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-8) for a, b in zip(left, right, strict=True))


def _finite_row_value(value: Any, name: str, idx: int, context: str) -> float:
    _require(_is_finite_number(value), context, f"{name}[{idx}] must be a finite number")
    return float(value)


def _close_float(left: Any, right: float) -> bool:
    return math.isclose(float(left), right, rel_tol=1e-5, abs_tol=1e-6)


def _require_logprob_row(logprobs: list[Any], prefix: str, idx: int, context: str) -> None:
    _require_numeric_values(logprobs, context, f"{prefix}_topk_logprobs[{idx}]", allow_none=False)
    _require(
        all(_is_finite_float_number(logprob) for logprob in logprobs),
        context,
        f"{prefix}_topk_logprobs[{idx}] must contain floating-point logprob values",
    )
    row_mass = math.fsum(math.exp(float(logprob)) for logprob in logprobs)
    _require(
        row_mass <= 1.0 + 1e-5,
        context,
        f"{prefix}_topk_logprobs[{idx}] probability mass exceeds 1",
    )


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


def _require_distinct_token_ids(values: list[int], context: str, name: str) -> None:
    _require(
        len(set(values)) == len(values),
        context,
        f"{name} must contain distinct token ids",
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


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_finite_float_number(value: Any) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _require_finite_float_value(value: Any, context: str, name: str) -> None:
    _require(_is_finite_number(value), context, f"{name} must be a finite number")
    _require(_is_finite_float_number(value), context, f"{name} must be a floating-point number")


def _is_non_blank_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_placeholder_logprob_row(logprobs: list[Any]) -> bool:
    return all(not isinstance(value, bool) and value in (0, 0.0) for value in logprobs)


def _is_placeholder_token_id_row(ids: list[Any]) -> bool:
    return all(not isinstance(value, bool) and value == 0 for value in ids)


def _require(condition: bool, context: str, message: str) -> None:
    if not condition:
        raise ValueError(f"{context}: {message}")
