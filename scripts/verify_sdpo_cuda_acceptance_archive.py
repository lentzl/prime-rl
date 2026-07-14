#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from prime_rl.trainer.rl.sdpo_export_verify import SDPOSmokeArtifactStats, verify_sdpo_smoke_artifacts

REQUIRED_FILES = (
    "sdpo_cuda_acceptance_summary.txt",
    "sdpo_cuda_acceptance_manifest.txt",
    "live/sdpo_smoke_provenance.txt",
    "live/sdpo_smoke_verify_report.txt",
    "ema/sdpo_smoke_provenance.txt",
    "ema/sdpo_smoke_verify_report.txt",
)
REQUIRED_NONEMPTY_FILE_PREFIXES = (
    "live/run_default/token_exports/",
    "ema/run_default/token_exports/",
    "ema/run_default/broadcasts/",
)
REPORT_MARKERS = {
    "live/sdpo_smoke_verify_report.txt": (
        "Verified SDPO smoke provenance:",
        "Verified SDPO token exports:",
    ),
    "ema/sdpo_smoke_verify_report.txt": (
        "Verified SDPO smoke provenance:",
        "Verified SDPO token exports:",
        "Verified SDPO EMA broadcasts:",
    ),
}
REQUIRED_TOKEN_EXPORT_REPORT_COUNTERS = (
    "sdpo_records",
    "transported_rows",
    "student_rows",
    "paired_rows",
    "matching_support_rows",
    "distinct_teacher_logprob_rows",
    "importance_ratio_rows",
    "rollout_is_weight_rows",
    "student_preflight_rows",
    "temperature_rows",
    "sample_id_records",
    "stable_steps",
    "matched_support_samples",
    "matched_support_token_rows",
    "distinct_teacher_logprob_token_rows",
    "importance_ratio_token_rows",
    "rollout_is_weight_token_rows",
)
REQUIRED_EMA_BROADCAST_REPORT_COUNTERS = ("steps", "teacher_steps")
REQUIRED_SUMMARY_FIELDS = (
    "output_root",
    "live_output_dir",
    "live_config",
    "live_provenance_file",
    "live_verify_report_file",
    "live_token_exports_dir",
    "ema_output_dir",
    "ema_config",
    "ema_provenance_file",
    "ema_verify_report_file",
    "ema_token_exports_dir",
    "ema_broadcasts_dir",
    "acceptance_manifest_file",
    "archive_path",
)
EXPECTED_SUMMARY_CONFIGS = {
    "live_config": "configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml",
    "ema_config": "configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml",
}
EXPECTED_PROVENANCE = {
    "live/sdpo_smoke_provenance.txt": {
        "mode": "live",
        "config": EXPECTED_SUMMARY_CONFIGS["live_config"],
        "orchestrator.algo.teacher_regularization": "live-policy",
        "trainer.sdpo_runtime.teacher_regularization": "live-policy",
    },
    "ema/sdpo_smoke_provenance.txt": {
        "mode": "ema",
        "config": EXPECTED_SUMMARY_CONFIGS["ema_config"],
        "orchestrator.algo.teacher_regularization": "ema",
        "trainer.sdpo_runtime.teacher_regularization": "ema",
    },
}
EXPECTED_REFERENCE_PROVENANCE_FIELDS = {
    "inference.vllm_extra.max_logprobs": "100",
    "orchestrator.train.sampling.temperature": "1.0",
    "orchestrator.algo.distillation_topk": "100",
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
    "trainer.sdpo_loss.distillation_topk": "100",
    "trainer.sdpo_loss.distillation_add_tail": "True",
    "trainer.sdpo_loss.alpha": "0.5",
    "trainer.sdpo_loss.is_clip": "2.0",
    "trainer.sdpo_loss.rollout_is": "token",
    "trainer.sdpo_loss.rollout_is_threshold": "2.0",
    "trainer.sdpo_loss.rollout_is_batch_normalize": "False",
    "trainer.sdpo_runtime.teacher_update_rate": "0.05",
}
REQUIRED_PROVENANCE_FIELDS = (
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
COHERENT_PROVENANCE_FIELDS = (
    "git_commit",
    "git_branch",
    "git_diff_sha256",
    "git_cached_diff_sha256",
    "git_untracked_manifest_sha256",
    "python_runner",
    "rl_runner",
)
SUMMARY_ARCHIVE_POINTERS = {
    "live_output_dir": ("live", "dir"),
    "live_provenance_file": ("live/sdpo_smoke_provenance.txt", "file"),
    "live_verify_report_file": ("live/sdpo_smoke_verify_report.txt", "file"),
    "live_token_exports_dir": ("live/run_default/token_exports", "dir"),
    "ema_output_dir": ("ema", "dir"),
    "ema_provenance_file": ("ema/sdpo_smoke_provenance.txt", "file"),
    "ema_verify_report_file": ("ema/sdpo_smoke_verify_report.txt", "file"),
    "ema_token_exports_dir": ("ema/run_default/token_exports", "dir"),
    "ema_broadcasts_dir": ("ema/run_default/broadcasts", "dir"),
    "acceptance_manifest_file": ("sdpo_cuda_acceptance_manifest.txt", "file"),
}


@dataclass(frozen=True)
class ManifestEntry:
    sha256: str
    size: int
    path: str


@dataclass(frozen=True)
class RawArtifactVerification:
    live: SDPOSmokeArtifactStats
    ema: SDPOSmokeArtifactStats


@dataclass(frozen=True)
class AcceptanceArchiveVerification:
    member_count: int
    manifest_entry_count: int
    acceptance_mode: str
    raw_artifacts: RawArtifactVerification


@dataclass(frozen=True)
class SmokeReportMetrics:
    token_exports: dict[str, str]
    ema_broadcasts: dict[str, str] | None = None


def _safe_member_name(name: str) -> bool:
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts and name not in {"", "."}


def _safe_member_type(member: tarfile.TarInfo) -> bool:
    return member.isfile() or member.isdir()


def _read_member(tar: tarfile.TarFile, name: str) -> bytes:
    member = tar.getmember(name)
    if not member.isfile():
        raise ValueError(f"archive member is not a regular file: {name}")
    handle = tar.extractfile(member)
    if handle is None:
        raise ValueError(f"archive member could not be read: {name}")
    return handle.read()


def _parse_manifest(data: bytes) -> tuple[dict[str, str], dict[str, ManifestEntry]]:
    fields: dict[str, str] = {}
    entries: dict[str, ManifestEntry] = {}
    saw_format = False
    for line_number, raw_line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not raw_line:
            continue
        if raw_line == "format=sha256 size_bytes relative_path":
            saw_format = True
            continue
        if not saw_format:
            if "=" not in raw_line:
                raise ValueError(f"manifest line {line_number} is malformed before format marker: {raw_line!r}")
            key, value = raw_line.split("=", 1)
            if key in fields:
                raise ValueError(f"manifest repeats field: {key}")
            fields[key] = value
            continue
        try:
            sha256, size_text, path = raw_line.split(" ", 2)
        except ValueError as exc:
            raise ValueError(f"manifest line {line_number} is malformed: {raw_line!r}") from exc
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError(f"manifest line {line_number} has invalid sha256: {sha256!r}")
        if not size_text.isdecimal():
            raise ValueError(f"manifest line {line_number} has invalid size: {size_text!r}")
        if path in entries:
            raise ValueError(f"manifest repeats path: {path}")
        entries[path] = ManifestEntry(sha256=sha256, size=int(size_text), path=path)
    if not saw_format:
        raise ValueError("manifest is missing format=sha256 size_bytes relative_path")
    if fields.get("sdpo_cuda_acceptance_manifest_version") != "1":
        raise ValueError("manifest is missing sdpo_cuda_acceptance_manifest_version=1")
    return fields, entries


def _parse_summary(data: bytes) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line_number, raw_line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not raw_line:
            continue
        if "=" not in raw_line:
            raise ValueError(f"summary line {line_number} is malformed: {raw_line!r}")
        key, value = raw_line.split("=", 1)
        if key in fields:
            raise ValueError(f"summary repeats field: {key}")
        fields[key] = value
    if fields.get("sdpo_cuda_acceptance_summary_version") != "1":
        raise ValueError("summary is missing sdpo_cuda_acceptance_summary_version=1")
    if fields.get("expected_topk") != "100":
        raise ValueError("summary is missing expected_topk=100")
    if fields.get("acceptance_mode") not in {"training", "no-run"}:
        raise ValueError("summary acceptance_mode must be 'training' or 'no-run'")
    for field in REQUIRED_SUMMARY_FIELDS:
        if not fields.get(field):
            raise ValueError(f"summary is missing {field}")
    for field, expected_value in EXPECTED_SUMMARY_CONFIGS.items():
        if fields[field] != expected_value:
            raise ValueError(f"summary {field} must be {expected_value!r}")
    for field in ("git_commit", "git_branch"):
        if not fields.get(field):
            raise ValueError(f"summary is missing {field}")
        if fields[field] in {"unknown", "unavailable"}:
            raise ValueError(f"summary field {field} must not be {fields[field]!r}")
    _verify_summary_archive_path_outside_output_root(fields)
    return fields


def _verify_summary_archive_path_outside_output_root(fields: dict[str, str]) -> None:
    output_root = fields["output_root"].rstrip("/")
    archive_path = fields["archive_path"].rstrip("/")
    if archive_path == output_root or archive_path.startswith(f"{output_root}/"):
        raise ValueError("summary archive_path must be outside output_root")


def _hash_manifest_lines(lines: list[str]) -> str:
    payload = "" if not lines else "\n".join(lines) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _parse_smoke_provenance(data: bytes, *, name: str) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    untracked_manifest_lines: list[str] = []
    in_git_status = False
    in_untracked_manifest = False
    for line_number, raw_line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if raw_line == "git_status_short_begin":
            if in_git_status:
                raise ValueError(f"{name} has nested git_status_short section")
            if "git_status_short_begin" in fields:
                raise ValueError(f"{name} repeats provenance field: git_status_short_begin")
            fields["git_status_short_begin"] = "1"
            in_git_status = True
            continue
        if raw_line == "git_status_short_end":
            if not in_git_status:
                raise ValueError(f"{name} closes git_status_short before opening it")
            if "git_status_short_end" in fields:
                raise ValueError(f"{name} repeats provenance field: git_status_short_end")
            fields["git_status_short_end"] = "1"
            in_git_status = False
            continue
        if in_git_status:
            continue
        if raw_line == "git_untracked_manifest_begin":
            if in_untracked_manifest:
                raise ValueError(f"{name} has nested git_untracked_manifest section")
            if "git_untracked_manifest_begin" in fields:
                raise ValueError(f"{name} repeats provenance field: git_untracked_manifest_begin")
            fields["git_untracked_manifest_begin"] = "1"
            in_untracked_manifest = True
            continue
        if raw_line == "git_untracked_manifest_end":
            if not in_untracked_manifest:
                raise ValueError(f"{name} closes git_untracked_manifest before opening it")
            if "git_untracked_manifest_end" in fields:
                raise ValueError(f"{name} repeats provenance field: git_untracked_manifest_end")
            fields["git_untracked_manifest_end"] = "1"
            in_untracked_manifest = False
            continue
        if in_untracked_manifest:
            untracked_manifest_lines.append(raw_line)
            continue
        if not raw_line:
            continue
        if "=" not in raw_line:
            raise ValueError(f"{name} line {line_number} is malformed: {raw_line!r}")
        key, value = raw_line.split("=", 1)
        if key in fields:
            raise ValueError(f"{name} repeats provenance field: {key}")
        fields[key] = value
    if in_git_status:
        raise ValueError(f"{name} is missing git_status_short_end")
    if in_untracked_manifest:
        raise ValueError(f"{name} is missing git_untracked_manifest_end")
    return fields, untracked_manifest_lines


def _verify_smoke_provenance(data: bytes, *, name: str) -> dict[str, str]:
    fields, untracked_manifest_lines = _parse_smoke_provenance(data, name=name)
    if fields.get("sdpo_smoke_provenance_version") != "1":
        raise ValueError(f"{name} is missing sdpo_smoke_provenance_version=1")
    if fields.get("expected_topk") != "100":
        raise ValueError(f"{name} is missing expected_topk=100")
    expected = EXPECTED_PROVENANCE[name]
    for field, expected_value in {**EXPECTED_REFERENCE_PROVENANCE_FIELDS, **expected}.items():
        if fields.get(field) != expected_value:
            raise ValueError(f"{name} mismatch for {field}: expected {expected_value!r}, got {fields.get(field)!r}")
    for field in REQUIRED_PROVENANCE_FIELDS:
        if not fields.get(field):
            raise ValueError(f"{name} is missing required provenance field: {field}")
    for field in ("git_commit", "git_branch", "git_diff_sha256", "git_cached_diff_sha256"):
        if fields[field] in {"unknown", "unavailable"}:
            raise ValueError(f"{name} field {field} must not be {fields[field]!r}")
    for field in ("git_diff_sha256", "git_cached_diff_sha256", "git_untracked_manifest_sha256"):
        if not _is_sha256_hex(fields[field]):
            raise ValueError(f"{name} field {field} must be a lowercase SHA-256 hex digest")
    actual_manifest_hash = _hash_manifest_lines(untracked_manifest_lines)
    if fields["git_untracked_manifest_sha256"] != actual_manifest_hash:
        raise ValueError(
            f"{name} mismatch for git_untracked_manifest_sha256: "
            f"expected hash of embedded manifest {actual_manifest_hash!r}, "
            f"got recorded value {fields['git_untracked_manifest_sha256']!r}"
        )
    return fields


def _verify_provenance_matches_summary(
    summary_fields: dict[str, str], provenance_fields: dict[str, str], name: str
) -> None:
    for field in ("git_commit", "git_branch"):
        if provenance_fields[field] != summary_fields[field]:
            raise ValueError(
                f"{name} {field} must match acceptance summary "
                f"({provenance_fields[field]!r} != {summary_fields[field]!r})"
            )


def _verify_live_and_ema_provenance_match(provenance_by_name: dict[str, dict[str, str]]) -> None:
    live_name = "live/sdpo_smoke_provenance.txt"
    ema_name = "ema/sdpo_smoke_provenance.txt"
    live = provenance_by_name[live_name]
    ema = provenance_by_name[ema_name]
    for field in COHERENT_PROVENANCE_FIELDS:
        if live[field] != ema[field]:
            raise ValueError(f"{ema_name} {field} must match {live_name} ({ema[field]!r} != {live[field]!r})")


def _report_fields(report: str, marker: str, report_name: str) -> dict[str, str]:
    matching_lines = [line for line in report.splitlines() if line.startswith(marker)]
    if not matching_lines:
        raise ValueError(f"{report_name} is missing success marker: {marker}")
    if len(matching_lines) > 1:
        raise ValueError(f"{report_name} repeats success marker: {marker}")
    fields: dict[str, str] = {}
    payload = matching_lines[0][len(marker) :].strip()
    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)=([^,\s]+)", payload):
        key = match.group(1)
        if key in fields:
            raise ValueError(f"{report_name} {marker} report repeats field: {key}")
        fields[key] = match.group(2)
    return fields


def _require_success_marker_once(report: str, marker: str, report_name: str) -> None:
    count = sum(1 for line in report.splitlines() if line.startswith(marker))
    if count == 0:
        raise ValueError(f"{report_name} is missing success marker: {marker}")
    if count > 1:
        raise ValueError(f"{report_name} repeats success marker: {marker}")


def _require_positive_report_counters(
    fields: dict[str, str],
    counters: tuple[str, ...],
    *,
    report_name: str,
    marker: str,
) -> None:
    for counter in counters:
        raw_value = fields.get(counter)
        if raw_value is None:
            raise ValueError(f"{report_name} {marker} report is missing counter: {counter}")
        if not raw_value.isdecimal() or int(raw_value) <= 0:
            raise ValueError(f"{report_name} {marker} counter {counter} must be positive, got {raw_value!r}")


def _verify_smoke_report_metrics(report: str, *, report_name: str) -> SmokeReportMetrics:
    token_fields = _report_fields(report, "Verified SDPO token exports:", report_name)
    _require_positive_report_counters(
        token_fields,
        REQUIRED_TOKEN_EXPORT_REPORT_COUNTERS,
        report_name=report_name,
        marker="Verified SDPO token exports:",
    )
    ema_fields = None
    if report_name.startswith("ema/"):
        ema_fields = _report_fields(report, "Verified SDPO EMA broadcasts:", report_name)
        _require_positive_report_counters(
            ema_fields,
            REQUIRED_EMA_BROADCAST_REPORT_COUNTERS,
            report_name=report_name,
            marker="Verified SDPO EMA broadcasts:",
        )
        if ema_fields.get("role") != "sdpo_teacher":
            raise ValueError(f"{report_name} EMA broadcast role must be 'sdpo_teacher'")
        if "matched_step_keys=[]" in report:
            raise ValueError(f"{report_name} EMA broadcast report must contain non-empty matched_step_keys")
    return SmokeReportMetrics(token_exports=token_fields, ema_broadcasts=ema_fields)


def _raw_token_export_counter(raw_artifacts: SDPOSmokeArtifactStats, counter: str) -> int:
    stats = raw_artifacts.token_exports
    if counter == "matched_support_samples":
        return len(stats.matched_support_sample_keys)
    if counter == "matched_support_token_rows":
        return len(stats.matched_support_row_keys)
    if counter == "distinct_teacher_logprob_token_rows":
        return len(stats.distinct_teacher_logprob_row_keys)
    if counter == "importance_ratio_token_rows":
        return len(stats.importance_ratio_row_keys)
    if counter == "rollout_is_weight_token_rows":
        return len(stats.rollout_is_weight_row_keys)
    value = getattr(stats, counter)
    if not isinstance(value, int):
        raise ValueError(f"raw SDPO token-export counter {counter} is not an integer")
    return value


def _raw_ema_broadcast_counter(raw_artifacts: SDPOSmokeArtifactStats, counter: str) -> int:
    stats = raw_artifacts.ema_broadcasts
    if stats is None:
        raise ValueError("raw EMA broadcast stats are missing")
    value = getattr(stats, counter)
    if not isinstance(value, int):
        raise ValueError(f"raw SDPO EMA broadcast counter {counter} is not an integer")
    return value


def _require_report_counter_matches_raw(
    fields: dict[str, str],
    counter: str,
    expected: int,
    *,
    report_name: str,
    marker: str,
) -> None:
    actual = int(fields[counter])
    if actual != expected:
        raise ValueError(
            f"{report_name} {marker} counter {counter} must match archived raw artifact value {expected}, got {actual}"
        )


def _verify_smoke_report_matches_raw_artifacts(
    *,
    report_name: str,
    report_metrics: SmokeReportMetrics,
    raw_artifacts: SDPOSmokeArtifactStats,
) -> None:
    for counter in REQUIRED_TOKEN_EXPORT_REPORT_COUNTERS:
        _require_report_counter_matches_raw(
            report_metrics.token_exports,
            counter,
            _raw_token_export_counter(raw_artifacts, counter),
            report_name=report_name,
            marker="Verified SDPO token exports:",
        )
    if report_name.startswith("ema/"):
        if report_metrics.ema_broadcasts is None:
            raise ValueError(f"{report_name} is missing EMA broadcast report metrics")
        for counter in REQUIRED_EMA_BROADCAST_REPORT_COUNTERS:
            _require_report_counter_matches_raw(
                report_metrics.ema_broadcasts,
                counter,
                _raw_ema_broadcast_counter(raw_artifacts, counter),
                report_name=report_name,
                marker="Verified SDPO EMA broadcasts:",
            )


def _regular_file_names(tar: tarfile.TarFile) -> set[str]:
    return {member.name for member in tar.getmembers() if member.isfile()}


def _extract_verified_archive(tar: tarfile.TarFile, destination: Path) -> None:
    for member in tar.getmembers():
        target = destination / member.name
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = tar.extractfile(member)
        if handle is None:
            raise ValueError(f"archive member could not be extracted: {member.name}")
        target.write_bytes(handle.read())


def _verify_archived_raw_artifacts(tar: tarfile.TarFile) -> RawArtifactVerification:
    with tempfile.TemporaryDirectory(prefix="sdpo-acceptance-archive-") as tmp_dir:
        extracted_root = Path(tmp_dir)
        _extract_verified_archive(tar, extracted_root)
        try:
            live = verify_sdpo_smoke_artifacts(extracted_root / "live", expected_topk=100)
            ema = verify_sdpo_smoke_artifacts(extracted_root / "ema", require_ema_teacher=True, expected_topk=100)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(f"archived raw SDPO artifacts failed verification: {exc}") from exc
        return RawArtifactVerification(live=live, ema=ema)


def _summary_pointer_relative_path(summary_fields: dict[str, str], field: str) -> str:
    output_root = summary_fields["output_root"].rstrip("/")
    value = summary_fields[field].rstrip("/")
    prefix = f"{output_root}/"
    if not value.startswith(prefix):
        raise ValueError(f"summary {field} must be under output_root")
    return value[len(prefix) :]


def _verify_summary_archive_pointers(summary_fields: dict[str, str], regular_files: set[str]) -> None:
    for field, (expected_member, expected_type) in SUMMARY_ARCHIVE_POINTERS.items():
        actual_member = _summary_pointer_relative_path(summary_fields, field)
        if actual_member != expected_member:
            raise ValueError(f"summary {field} must point to archive member {expected_member!r}")
        if expected_type == "file":
            if actual_member not in regular_files:
                raise ValueError(f"summary {field} points to a file missing from archive: {actual_member}")
        elif not any(name.startswith(f"{actual_member}/") for name in regular_files):
            raise ValueError(f"summary {field} points to an empty or missing archive directory: {actual_member}")


def _verify_expected_git_identity(
    summary_fields: dict[str, str],
    *,
    expected_git_commit: str | None,
    expected_git_branch: str | None,
) -> None:
    if expected_git_commit is not None and summary_fields["git_commit"] != expected_git_commit:
        raise ValueError(
            f"archive git_commit mismatch: expected {expected_git_commit!r}, got {summary_fields['git_commit']!r}"
        )
    if expected_git_branch is not None and summary_fields["git_branch"] != expected_git_branch:
        raise ValueError(
            f"archive git_branch mismatch: expected {expected_git_branch!r}, got {summary_fields['git_branch']!r}"
        )


def verify_archive(
    path: Path,
    *,
    expected_acceptance_mode: str | None = None,
    expected_git_commit: str | None = None,
    expected_git_branch: str | None = None,
) -> AcceptanceArchiveVerification:
    if not path.is_file():
        raise ValueError(f"archive does not exist: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"archive is empty: {path}")

    with tarfile.open(path, "r:gz") as tar:
        members = tar.getmembers()
        member_names: set[str] = set()
        for member in members:
            if not _safe_member_name(member.name):
                raise ValueError(f"archive contains unsafe member path: {member.name!r}")
            if not _safe_member_type(member):
                raise ValueError(f"archive contains unsupported member type: {member.name!r}")
            if member.name in member_names:
                raise ValueError(f"archive contains duplicate member path: {member.name!r}")
            member_names.add(member.name)
        for required in REQUIRED_FILES:
            if required not in member_names:
                raise ValueError(f"archive is missing required proof file: {required}")
            if not tar.getmember(required).isfile():
                raise ValueError(f"archive required proof member is not a regular file: {required}")
        regular_files = _regular_file_names(tar)
        for prefix in REQUIRED_NONEMPTY_FILE_PREFIXES:
            if not any(name.startswith(prefix) and tar.getmember(name).size > 0 for name in regular_files):
                raise ValueError(f"archive has no non-empty proof file under required prefix: {prefix}")

        summary_fields = _parse_summary(_read_member(tar, "sdpo_cuda_acceptance_summary.txt"))
        _verify_summary_archive_pointers(summary_fields, regular_files)

        provenance_by_name: dict[str, dict[str, str]] = {}
        for provenance_name in EXPECTED_PROVENANCE:
            provenance_fields = _verify_smoke_provenance(_read_member(tar, provenance_name), name=provenance_name)
            _verify_provenance_matches_summary(summary_fields, provenance_fields, provenance_name)
            provenance_by_name[provenance_name] = provenance_fields
        _verify_live_and_ema_provenance_match(provenance_by_name)

        report_metrics_by_name: dict[str, SmokeReportMetrics] = {}
        for report_name, markers in REPORT_MARKERS.items():
            report = _read_member(tar, report_name).decode("utf-8")
            for marker in markers:
                _require_success_marker_once(report, marker, report_name)
            report_metrics_by_name[report_name] = _verify_smoke_report_metrics(report, report_name=report_name)

        manifest_fields, manifest_entries = _parse_manifest(_read_member(tar, "sdpo_cuda_acceptance_manifest.txt"))
        if manifest_fields.get("acceptance_mode") not in {"training", "no-run"}:
            raise ValueError("manifest acceptance_mode must be 'training' or 'no-run'")
        if summary_fields["acceptance_mode"] != manifest_fields["acceptance_mode"]:
            raise ValueError(
                "summary/manifest acceptance_mode mismatch: "
                f"summary={summary_fields['acceptance_mode']!r}, "
                f"manifest={manifest_fields['acceptance_mode']!r}"
            )
        if expected_acceptance_mode is not None and summary_fields["acceptance_mode"] != expected_acceptance_mode:
            raise ValueError(
                "archive acceptance_mode mismatch: "
                f"expected {expected_acceptance_mode!r}, got {summary_fields['acceptance_mode']!r}"
            )
        _verify_expected_git_identity(
            summary_fields,
            expected_git_commit=expected_git_commit,
            expected_git_branch=expected_git_branch,
        )

        for manifest_path, entry in manifest_entries.items():
            if manifest_path not in regular_files:
                raise ValueError(f"manifest path is missing from archive: {manifest_path}")
            data = _read_member(tar, manifest_path)
            if len(data) != entry.size:
                raise ValueError(f"manifest size mismatch for {manifest_path}: expected {entry.size}, got {len(data)}")
            actual_sha = hashlib.sha256(data).hexdigest()
            if actual_sha != entry.sha256:
                raise ValueError(
                    f"manifest sha256 mismatch for {manifest_path}: expected {entry.sha256}, got {actual_sha}"
                )

        unmanifested_files = regular_files - {"sdpo_cuda_acceptance_manifest.txt"} - set(manifest_entries)
        if unmanifested_files:
            raise ValueError(f"archive contains regular files not listed in manifest: {sorted(unmanifested_files)}")
        raw_artifacts = _verify_archived_raw_artifacts(tar)
        _verify_smoke_report_matches_raw_artifacts(
            report_name="live/sdpo_smoke_verify_report.txt",
            report_metrics=report_metrics_by_name["live/sdpo_smoke_verify_report.txt"],
            raw_artifacts=raw_artifacts.live,
        )
        _verify_smoke_report_matches_raw_artifacts(
            report_name="ema/sdpo_smoke_verify_report.txt",
            report_metrics=report_metrics_by_name["ema/sdpo_smoke_verify_report.txt"],
            raw_artifacts=raw_artifacts.ema,
        )
        return AcceptanceArchiveVerification(
            member_count=len(members),
            manifest_entry_count=len(manifest_entries),
            acceptance_mode=summary_fields["acceptance_mode"],
            raw_artifacts=raw_artifacts,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify an SDPO CUDA acceptance proof tarball after download.",
    )
    parser.add_argument("archive", type=Path, help="Path to sdpo-cuda-acceptance-proof.tar.gz")
    parser.add_argument(
        "--expected-acceptance-mode",
        choices=("training", "no-run"),
        default=None,
        help="Require the archive summary and manifest to prove this acceptance mode.",
    )
    parser.add_argument(
        "--expected-git-commit",
        default=None,
        help="Require the archive summary/provenance to prove this exact git commit.",
    )
    parser.add_argument(
        "--expected-git-branch",
        default=None,
        help="Require the archive summary/provenance to prove this exact git branch.",
    )
    args = parser.parse_args()
    try:
        verification = verify_archive(
            args.archive,
            expected_acceptance_mode=args.expected_acceptance_mode,
            expected_git_commit=args.expected_git_commit,
            expected_git_branch=args.expected_git_branch,
        )
    except (KeyError, OSError, tarfile.TarError, UnicodeDecodeError, ValueError) as exc:
        parser.error(f"invalid SDPO CUDA acceptance archive: {exc}")
    live_exports = verification.raw_artifacts.live.token_exports
    ema_exports = verification.raw_artifacts.ema.token_exports
    ema_broadcasts = verification.raw_artifacts.ema.ema_broadcasts
    if ema_broadcasts is None:
        raise AssertionError("EMA raw artifact verification did not return EMA broadcast stats")
    print(
        "Verified SDPO CUDA acceptance archive: "
        f"file={args.archive}, acceptance_mode={verification.acceptance_mode}, "
        f"members={verification.member_count}, manifest_entries={verification.manifest_entry_count}, "
        "raw_artifacts=verified, "
        f"live_sdpo_records={live_exports.sdpo_records}, "
        f"live_matching_support_rows={live_exports.matching_support_rows}, "
        f"live_rollout_is_weight_rows={live_exports.rollout_is_weight_rows}, "
        f"ema_sdpo_records={ema_exports.sdpo_records}, "
        f"ema_matching_support_rows={ema_exports.matching_support_rows}, "
        f"ema_rollout_is_weight_rows={ema_exports.rollout_is_weight_rows}, "
        f"ema_teacher_steps={ema_broadcasts.teacher_steps}"
    )


if __name__ == "__main__":
    main()
