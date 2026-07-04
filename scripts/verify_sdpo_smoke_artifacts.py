#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from prime_rl.trainer.rl.sdpo_export_verify import verify_sdpo_smoke_artifacts

REFERENCE_PROVENANCE_FIELDS = {
    "orchestrator.algo.distillation_topk_support": "student",
    "orchestrator.algo.teacher_update_rate": "0.05",
    "orchestrator.algo.success_reward_threshold": "0.5",
    "orchestrator.algo.successful_demonstration_selection": "batch_order",
    "orchestrator.algo.dont_reprompt_on_self_success": "True",
    "orchestrator.algo.remove_thinking_from_demonstration": "True",
    "orchestrator.algo.include_environment_feedback": "True",
    "orchestrator.algo.environment_feedback_only_without_solution": "True",
    "orchestrator.algo.max_reprompt_len": "10240",
    "orchestrator.algo.reprompt_truncation": "right",
    "orchestrator.algo.assistant_prefix": "",
    "orchestrator.algo.multi_turn": "False",
    "orchestrator.algo.template_target": "first_user",
    "trainer.sdpo_loss.full_logit_distillation": "True",
    "trainer.sdpo_loss.distillation_add_tail": "True",
    "trainer.sdpo_loss.alpha": "0.5",
    "trainer.sdpo_loss.is_clip": "2.0",
    "trainer.sdpo_loss.rollout_is": "token",
    "trainer.sdpo_loss.rollout_is_threshold": "2.0",
    "trainer.sdpo_loss.rollout_is_batch_normalize": "False",
    "trainer.sdpo_runtime.teacher_update_rate": "0.05",
}

REFERENCE_TEACHER_REGULARIZATION_BY_MODE = {
    "live": "live-policy",
    "ema": "ema",
}


def _read_smoke_provenance(path: Path) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    untracked_manifest_lines: list[str] = []
    in_untracked_manifest = False
    in_git_status = False
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if line == "git_status_short_begin":
                if in_git_status:
                    raise ValueError(f"{path} has nested git_status_short section")
                if "git_status_short_begin" in values:
                    raise ValueError(f"{path} repeats provenance field: git_status_short_begin")
                values["git_status_short_begin"] = "1"
                in_git_status = True
                continue
            if line == "git_status_short_end":
                if not in_git_status:
                    raise ValueError(f"{path} closes git_status_short before opening it")
                if "git_status_short_end" in values:
                    raise ValueError(f"{path} repeats provenance field: git_status_short_end")
                values["git_status_short_end"] = "1"
                in_git_status = False
                continue
            if in_git_status:
                continue
            if line == "git_untracked_manifest_begin":
                if in_untracked_manifest:
                    raise ValueError(f"{path} has nested git_untracked_manifest section")
                if "git_untracked_manifest_begin" in values:
                    raise ValueError(f"{path} repeats provenance field: git_untracked_manifest_begin")
                values["git_untracked_manifest_begin"] = "1"
                in_untracked_manifest = True
                continue
            if line == "git_untracked_manifest_end":
                if not in_untracked_manifest:
                    raise ValueError(f"{path} closes git_untracked_manifest before opening it")
                if "git_untracked_manifest_end" in values:
                    raise ValueError(f"{path} repeats provenance field: git_untracked_manifest_end")
                values["git_untracked_manifest_end"] = "1"
                in_untracked_manifest = False
                continue
            if in_untracked_manifest:
                untracked_manifest_lines.append(line)
                continue
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"{path} line {line_number} is malformed: {line!r}")
            key, value = line.split("=", 1)
            if key in values:
                raise ValueError(f"{path} repeats provenance field: {key}")
            values[key] = value
    if in_git_status:
        raise ValueError(f"{path} is missing git_status_short_end")
    if in_untracked_manifest:
        raise ValueError(f"{path} is missing git_untracked_manifest_end")
    return values, untracked_manifest_lines


def _hash_manifest_lines(lines: list[str]) -> str:
    payload = "" if not lines else "\n".join(lines) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _find_smoke_provenance(path: Path) -> Path | None:
    """Find run provenance from an output dir or a nested token/broadcast artifact path."""
    start = path.parent if path.is_file() else path
    candidates = [start / "sdpo_smoke_provenance.txt"]

    if start.name == "run_default":
        candidates.append(start.parent / "sdpo_smoke_provenance.txt")
    elif start.name in {"token_exports", "broadcasts"} and start.parent.name == "run_default":
        candidates.append(start.parent.parent / "sdpo_smoke_provenance.txt")
    elif start.name.startswith("step_") and start.parent.name in {"token_exports", "broadcasts"}:
        candidates.append(start.parent.parent.parent / "sdpo_smoke_provenance.txt")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _require_fresh_smoke_provenance_fields(parser: argparse.ArgumentParser, values: dict[str, str]) -> None:
    required_fields = (
        "git_commit",
        "git_branch",
        "git_diff_sha256",
        "git_cached_diff_sha256",
        "git_untracked_manifest_sha256",
        "git_untracked_manifest_begin",
        "git_untracked_manifest_end",
        "git_status_short_begin",
        "git_status_short_end",
        "python_runner",
        "rl_runner",
    )
    for field in required_fields:
        if not values.get(field):
            parser.error(f"SDPO smoke provenance is missing required field: {field}")
    for field in (
        "git_commit",
        "git_branch",
        "git_diff_sha256",
        "git_cached_diff_sha256",
        "git_untracked_manifest_sha256",
    ):
        if values[field] in {"unknown", "unavailable"}:
            parser.error(f"SDPO smoke provenance field {field} must not be {values[field]!r}")
    for field in ("git_diff_sha256", "git_cached_diff_sha256", "git_untracked_manifest_sha256"):
        if not _is_sha256_hex(values[field]):
            parser.error(f"SDPO smoke provenance field {field} must be a lowercase SHA-256 hex digest")


def _require_reference_smoke_provenance_fields(
    parser: argparse.ArgumentParser,
    values: dict[str, str],
    *,
    expected_topk: int,
) -> None:
    expected_fields = {
        **REFERENCE_PROVENANCE_FIELDS,
        "orchestrator.algo.distillation_topk": str(expected_topk),
        "trainer.sdpo_loss.distillation_topk": str(expected_topk),
    }
    mode = values.get("mode")
    teacher_regularization = REFERENCE_TEACHER_REGULARIZATION_BY_MODE.get(mode)
    if teacher_regularization is not None:
        expected_fields["orchestrator.algo.teacher_regularization"] = teacher_regularization
        expected_fields["trainer.sdpo_runtime.teacher_regularization"] = teacher_regularization
    for key, expected in expected_fields.items():
        actual = values.get(key)
        if actual != expected:
            parser.error(f"SDPO smoke provenance mismatch for {key}: expected {expected!r}, got {actual!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify strict SDPO smoke-run artifacts: stable token exports, preflight-only "
            "student support, and final transported SDPO support matched on run, step, "
            "sample_id, sample signature, and weighted token rows. The reference smoke "
            "contract also requires rollout-IS ratio evidence and final sdpo_rollout_is_weights "
            "matching token-level truncated rollout-IS with rollout_is_threshold=2.0."
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Run output directory, run_default directory, token_exports/broadcasts directory, "
            "or a token_exports/broadcasts step directory."
        ),
    )
    parser.add_argument(
        "--require-ema-teacher",
        action="store_true",
        help="Also require filesystem broadcasts with sdpo_teacher/STABLE markers and non-empty teacher artifacts.",
    )
    parser.add_argument(
        "--expected-topk",
        type=int,
        required=True,
        help="Required. Require every supported SDPO top-k row to have this width.",
    )
    parser.add_argument(
        "--require-provenance",
        action="store_true",
        help="Also require sdpo_smoke_provenance.txt in or above the artifact path.",
    )
    parser.add_argument(
        "--expected-provenance-mode",
        choices=("live", "ema"),
        default=None,
        help="Require sdpo_smoke_provenance.txt to record this smoke mode.",
    )
    parser.add_argument(
        "--expected-provenance-config",
        default=None,
        help="Require sdpo_smoke_provenance.txt to record this config path.",
    )
    args = parser.parse_args()

    provenance: Path | None = None
    provenance_values: dict[str, str] | None = None
    untracked_manifest_lines: list[str] = []
    if args.require_provenance or args.expected_provenance_mode or args.expected_provenance_config:
        provenance = _find_smoke_provenance(args.path)
        if provenance is None:
            parser.error(f"Missing SDPO smoke provenance file reachable from: {args.path}")
        try:
            provenance_values, untracked_manifest_lines = _read_smoke_provenance(provenance)
        except ValueError as exc:
            parser.error(f"Invalid SDPO smoke provenance: {exc}")
        expected_values = {
            "sdpo_smoke_provenance_version": "1",
            "expected_topk": str(args.expected_topk),
        }
        if args.expected_provenance_mode is not None:
            expected_values["mode"] = args.expected_provenance_mode
        if args.expected_provenance_config is not None:
            expected_values["config"] = args.expected_provenance_config
        for key, expected in expected_values.items():
            actual = provenance_values.get(key)
            if actual != expected:
                parser.error(f"SDPO smoke provenance mismatch for {key}: expected {expected!r}, got {actual!r}")
        if args.require_provenance:
            _require_fresh_smoke_provenance_fields(parser, provenance_values)
            _require_reference_smoke_provenance_fields(
                parser,
                provenance_values,
                expected_topk=args.expected_topk,
            )
            actual_manifest_hash = _hash_manifest_lines(untracked_manifest_lines)
            recorded_manifest_hash = provenance_values["git_untracked_manifest_sha256"]
            if actual_manifest_hash != recorded_manifest_hash:
                parser.error(
                    "SDPO smoke provenance mismatch for git_untracked_manifest_sha256: "
                    f"expected hash of embedded manifest {actual_manifest_hash!r}, "
                    f"got recorded value {recorded_manifest_hash!r}"
                )

    stats = verify_sdpo_smoke_artifacts(
        args.path,
        require_ema_teacher=args.require_ema_teacher,
        expected_topk=args.expected_topk,
    )
    export_stats = stats.token_exports
    if provenance is not None and provenance_values is not None:
        print(
            "Verified SDPO smoke provenance: "
            f"file={provenance}, mode={provenance_values.get('mode')}, "
            f"config={provenance_values.get('config')}, "
            f"expected_topk={provenance_values.get('expected_topk')}"
        )
    print(
        "Verified SDPO token exports: "
        f"files={export_stats.files}, records={export_stats.records}, "
        f"sdpo_records={export_stats.sdpo_records}, transported_rows={export_stats.transported_rows}, "
        f"student_rows={export_stats.student_rows}, paired_rows={export_stats.paired_rows}, "
        f"matching_support_rows={export_stats.matching_support_rows}, "
        f"distinct_teacher_logprob_rows={export_stats.distinct_teacher_logprob_rows}, "
        f"importance_ratio_rows={export_stats.importance_ratio_rows}, "
        f"rollout_is_weight_rows={export_stats.rollout_is_weight_rows}, "
        f"student_preflight_rows={export_stats.student_preflight_rows}, "
        f"temperature_rows={export_stats.temperature_rows}, "
        f"sample_id_records={export_stats.sample_id_records}, "
        f"stable_steps={export_stats.stable_steps}, "
        f"steps={list(export_stats.step_names)}, paired_steps={list(export_stats.paired_step_names)}, "
        f"matching_support_steps={list(export_stats.matching_support_step_names)}, "
        f"student_preflight_steps={list(export_stats.student_preflight_step_names)}, "
        f"matched_support_samples={len(export_stats.matched_support_sample_keys)}, "
        f"matched_support_token_rows={len(export_stats.matched_support_row_keys)}, "
        f"distinct_teacher_logprob_token_rows={len(export_stats.distinct_teacher_logprob_row_keys)}, "
        f"importance_ratio_token_rows={len(export_stats.importance_ratio_row_keys)}, "
        f"rollout_is_weight_token_rows={len(export_stats.rollout_is_weight_row_keys)}",
    )

    if stats.ema_broadcasts is not None:
        broadcast_stats = stats.ema_broadcasts
        print(
            "Verified SDPO EMA broadcasts: "
            f"steps={broadcast_stats.steps}, role={broadcast_stats.role}, "
            f"teacher_steps={broadcast_stats.teacher_steps}, matched_steps={list(stats.matched_steps)}, "
            f"matched_step_keys={list(stats.matched_step_keys)}"
        )


if __name__ == "__main__":
    main()
