from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "run_sdpo_smoke_and_verify.sh"
ACCEPTANCE_SCRIPT = REPO_ROOT / "scripts" / "run_sdpo_cuda_acceptance.sh"
ACCEPTANCE_BACKGROUND_SCRIPT = REPO_ROOT / "scripts" / "start_sdpo_cuda_acceptance_background.sh"
LOCAL_VALIDATION_SCRIPT = REPO_ROOT / "scripts" / "run_sdpo_local_validation.sh"
VERIFY_SMOKE_ARTIFACTS_SCRIPT = REPO_ROOT / "scripts" / "verify_sdpo_smoke_artifacts.py"
VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT = REPO_ROOT / "scripts" / "verify_sdpo_cuda_acceptance_archive.py"
SDPO_LIVE_SMOKE_CONFIG = REPO_ROOT / "configs" / "debug" / "algorithms" / "sdpo_huebotter_reference_smoke.toml"
SDPO_EMA_SMOKE_CONFIG = REPO_ROOT / "configs" / "debug" / "algorithms" / "sdpo_huebotter_reference_ema_smoke.toml"
SDPO_TEST_GIT_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SDPO_TEST_GIT_BRANCH = "codex/sdpo-test"


def _current_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _current_git_branch() -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    branch = result.stdout.strip()
    if branch:
        return branch
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return f"detached-{result.stdout.strip()}"


def _sdpo_cli_env() -> dict[str, str]:
    pythonpath = os.pathsep.join(
        [
            str(REPO_ROOT / "src"),
            str(REPO_ROOT / "packages" / "prime-rl-configs" / "src"),
            str(REPO_ROOT / "deps" / "pydantic-config" / "src"),
        ]
    )
    return {**os.environ, "PYTHONPATH": pythonpath}


def test_sdpo_smoke_script_has_valid_shell_syntax():
    result = subprocess.run(["bash", "-n", str(SMOKE_SCRIPT)], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_sdpo_cuda_acceptance_script_has_valid_shell_syntax():
    result = subprocess.run(["bash", "-n", str(ACCEPTANCE_SCRIPT)], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_sdpo_cuda_acceptance_background_script_has_valid_shell_syntax():
    result = subprocess.run(["bash", "-n", str(ACCEPTANCE_BACKGROUND_SCRIPT)], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_sdpo_cuda_acceptance_script_help_documents_archive_option():
    result = subprocess.run(["bash", str(ACCEPTANCE_SCRIPT), "--help"], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "--archive PATH" in result.stdout
    assert "tar.gz archive of the acceptance proof artifacts" in result.stdout
    assert "combined summary and a SHA-256 manifest" in result.stdout
    assert "re-runs strict artifact verification" in result.stdout
    assert "proof" in result.stdout
    assert "files to be non-empty" in result.stdout
    assert "tarball must also be non-empty, listable, and pass the offline archive verifier" in result.stdout


def test_sdpo_cuda_acceptance_background_script_help_documents_remote_usage():
    result = subprocess.run(["bash", str(ACCEPTANCE_BACKGROUND_SCRIPT), "--help"], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "Start the full SDPO CUDA acceptance run in the background" in result.stdout
    assert "--dry-run" in result.stdout
    assert "--preflight-only" in result.stdout
    assert "--skip-host-preflight" in result.stdout
    assert "--skip-config-preflight" in result.stdout
    assert "--status" in result.stdout
    assert "Linux CUDA/vLLM box" in result.stdout
    assert "proof archive" in result.stdout
    assert "raw_artifacts=verified" in result.stdout
    assert "Recommended fresh-box flow:" in result.stdout
    assert "scripts/start_sdpo_cuda_acceptance_background.sh --preflight-only" in result.stdout
    assert "scripts/start_sdpo_cuda_acceptance_background.sh --status" in result.stdout


def test_sdpo_reference_smoke_configs_document_wrapper_first_verification():
    live_config = SDPO_LIVE_SMOKE_CONFIG.read_text(encoding="utf-8")
    ema_config = SDPO_EMA_SMOKE_CONFIG.read_text(encoding="utf-8")

    assert "scripts/run_sdpo_smoke_and_verify.sh --check-config" in live_config
    assert "scripts/run_sdpo_smoke_and_verify.sh" in live_config
    assert "--require-provenance" in live_config
    assert "--expected-provenance-mode live" in live_config
    assert f"--expected-provenance-config {SDPO_LIVE_SMOKE_CONFIG.relative_to(REPO_ROOT)}" in live_config
    assert "raw local rl launcher is useful for debugging" in live_config

    assert "scripts/run_sdpo_smoke_and_verify.sh --ema --check-config" in ema_config
    assert "scripts/run_sdpo_smoke_and_verify.sh --ema" in ema_config
    assert "--require-provenance" in ema_config
    assert "--expected-provenance-mode ema" in ema_config
    assert f"--expected-provenance-config {SDPO_EMA_SMOKE_CONFIG.relative_to(REPO_ROOT)}" in ema_config
    assert "--require-ema-teacher" in ema_config
    assert "raw local rl launcher is useful for debugging" in ema_config


def test_sdpo_cuda_acceptance_background_script_dry_run_prints_default_command():
    result = subprocess.run(
        ["bash", str(ACCEPTANCE_BACKGROUND_SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "scripts/run_sdpo_cuda_acceptance.sh" in result.stdout
    assert "--output-root outputs/sdpo-cuda-acceptance" in result.stdout
    assert "--archive outputs/sdpo-cuda-acceptance-proof.tar.gz" in result.stdout
    assert "--clean-output-dir" in result.stdout
    assert "Dry run only; not starting SDPO CUDA acceptance." in result.stdout
    assert "Started SDPO CUDA acceptance" not in result.stdout
    assert "Running SDPO CUDA acceptance host preflight" not in result.stdout


def test_sdpo_cuda_acceptance_background_script_dry_run_respects_custom_paths(tmp_path):
    output_root = tmp_path / "acceptance"
    archive_path = tmp_path / "proof.tar.gz"
    log_path = tmp_path / "acceptance.log"
    pid_file = tmp_path / "acceptance.pid"

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_BACKGROUND_SCRIPT),
            "--dry-run",
            "--no-clean-output-dir",
            "--output-root",
            str(output_root),
            "--archive",
            str(archive_path),
            "--log",
            str(log_path),
            "--pid-file",
            str(pid_file),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"Output root: {output_root}" in result.stdout
    assert f"Archive: {archive_path}" in result.stdout
    assert f"Log: {log_path}" in result.stdout
    assert f"PID file: {pid_file}" in result.stdout
    assert "--clean-output-dir" not in result.stdout
    assert not output_root.exists()
    assert not archive_path.exists()
    assert not log_path.exists()
    assert not pid_file.exists()


def test_sdpo_cuda_acceptance_background_script_dry_run_archive_guard_does_not_create_missing_dirs(tmp_path):
    output_root = tmp_path / "missing" / "acceptance"
    archive_path = tmp_path / "missing" / "proofs" / "proof.tar.gz"

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_BACKGROUND_SCRIPT),
            "--dry-run",
            "--output-root",
            str(output_root),
            "--archive",
            str(archive_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Dry run only; not starting SDPO CUDA acceptance." in result.stdout
    assert not output_root.exists()
    assert not archive_path.parent.exists()


def test_sdpo_cuda_acceptance_background_script_status_reports_missing_paths(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_BACKGROUND_SCRIPT),
            "--status",
            "--output-root",
            str(tmp_path / "acceptance"),
            "--archive",
            str(tmp_path / "proof.tar.gz"),
            "--log",
            str(tmp_path / "acceptance.log"),
            "--pid-file",
            str(tmp_path / "acceptance.pid"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Recorded PID: missing" in result.stdout
    assert "Process status: unknown" in result.stdout
    assert "Archive status: missing" in result.stdout
    assert "Log status: missing" in result.stdout
    assert "Dry run only" not in result.stdout
    assert "Started SDPO CUDA acceptance" not in result.stdout


def test_sdpo_cuda_acceptance_background_script_status_archive_guard_does_not_create_missing_dirs(tmp_path):
    output_root = tmp_path / "missing" / "acceptance"
    archive_path = tmp_path / "missing" / "proofs" / "proof.tar.gz"
    log_path = tmp_path / "missing" / "logs" / "acceptance.log"
    pid_file = tmp_path / "missing" / "pids" / "acceptance.pid"

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_BACKGROUND_SCRIPT),
            "--status",
            "--output-root",
            str(output_root),
            "--archive",
            str(archive_path),
            "--log",
            str(log_path),
            "--pid-file",
            str(pid_file),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Archive status: missing" in result.stdout
    assert not output_root.exists()
    assert not archive_path.parent.exists()
    assert not log_path.parent.exists()
    assert not pid_file.parent.exists()


def test_sdpo_cuda_acceptance_background_script_status_reports_existing_artifacts(tmp_path):
    archive_path = tmp_path / "proof.tar.gz"
    log_path = tmp_path / "acceptance.log"
    pid_file = tmp_path / "acceptance.pid"
    files = _minimal_sdpo_acceptance_archive_files(summary_mode="training")
    _write_sdpo_acceptance_tar(
        archive_path,
        _add_sdpo_acceptance_manifest(files, manifest_mode="training"),
    )
    log_path.write_text("first line\nlast line\n", encoding="utf-8")
    pid_file.write_text("not-a-pid\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_BACKGROUND_SCRIPT),
            "--status",
            "--output-root",
            str(tmp_path / "acceptance"),
            "--archive",
            str(archive_path),
            "--log",
            str(log_path),
            "--pid-file",
            str(pid_file),
        ],
        cwd=REPO_ROOT,
        env=os.environ | {"SDPO_ACCEPTANCE_PYTHON_RUNNER": sys.executable},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Recorded PID: not-a-pid" in result.stdout
    assert "Process status: not running" in result.stdout
    assert "Archive status: present non-empty" in result.stdout
    assert "Log status: present" in result.stdout
    assert "Last log lines:" in result.stdout
    assert "last line" in result.stdout
    assert "Verified SDPO CUDA acceptance archive:" in result.stdout
    assert "acceptance_mode=training" in result.stdout
    assert "raw_artifacts=verified" in result.stdout
    assert "ema_teacher_steps=1" in result.stdout
    assert "Archive verification: passed" in result.stdout


def test_sdpo_cuda_acceptance_background_script_status_fails_invalid_completed_archive(tmp_path):
    archive_path = tmp_path / "proof.tar.gz"
    pid_file = tmp_path / "acceptance.pid"
    archive_path.write_text("proof\n", encoding="utf-8")
    pid_file.write_text("not-a-pid\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_BACKGROUND_SCRIPT),
            "--status",
            "--output-root",
            str(tmp_path / "acceptance"),
            "--archive",
            str(archive_path),
            "--log",
            str(tmp_path / "acceptance.log"),
            "--pid-file",
            str(pid_file),
        ],
        cwd=REPO_ROOT,
        env=os.environ | {"SDPO_ACCEPTANCE_PYTHON_RUNNER": sys.executable},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Archive status: present non-empty" in result.stdout
    assert "Log status: missing" in result.stdout
    assert "Archive verification: failed" in result.stdout
    assert "invalid SDPO CUDA acceptance archive" in result.stderr


def test_sdpo_cuda_acceptance_background_script_status_fails_empty_orphan_archive(tmp_path):
    archive_path = tmp_path / "proof.tar.gz"
    archive_path.write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_BACKGROUND_SCRIPT),
            "--status",
            "--output-root",
            str(tmp_path / "acceptance"),
            "--archive",
            str(archive_path),
            "--log",
            str(tmp_path / "acceptance.log"),
            "--pid-file",
            str(tmp_path / "acceptance.pid"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Recorded PID: missing" in result.stdout
    assert "Process status: unknown" in result.stdout
    assert "Archive status: present empty" in result.stdout
    assert "Completion status: failed or incomplete; archive file is empty" in result.stdout
    assert "Archive verification:" not in result.stdout


def test_sdpo_cuda_acceptance_background_script_status_tolerates_empty_archive_while_running(tmp_path):
    archive_path = tmp_path / "proof.tar.gz"
    pid_file = tmp_path / "acceptance.pid"
    archive_path.write_text("", encoding="utf-8")
    sleeper = subprocess.Popen(["sleep", "60"])
    try:
        pid_file.write_text(f"{sleeper.pid}\n", encoding="utf-8")

        result = subprocess.run(
            [
                "bash",
                str(ACCEPTANCE_BACKGROUND_SCRIPT),
                "--status",
                "--output-root",
                str(tmp_path / "acceptance"),
                "--archive",
                str(archive_path),
                "--log",
                str(tmp_path / "acceptance.log"),
                "--pid-file",
                str(pid_file),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    finally:
        sleeper.terminate()
        try:
            sleeper.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sleeper.kill()
            sleeper.wait(timeout=5)

    assert result.returncode == 0, result.stderr
    assert "Process status: running" in result.stdout
    assert "Archive status: present empty" in result.stdout
    assert "Completion status: failed or incomplete" not in result.stdout
    assert "Archive verification:" not in result.stdout


def test_sdpo_cuda_acceptance_background_script_status_rejects_no_run_archive(tmp_path):
    archive_path = tmp_path / "proof.tar.gz"
    pid_file = tmp_path / "acceptance.pid"
    _write_sdpo_acceptance_tar(
        archive_path,
        _add_sdpo_acceptance_manifest(_minimal_sdpo_acceptance_archive_files()),
    )
    pid_file.write_text("not-a-pid\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_BACKGROUND_SCRIPT),
            "--status",
            "--output-root",
            str(tmp_path / "acceptance"),
            "--archive",
            str(archive_path),
            "--log",
            str(tmp_path / "acceptance.log"),
            "--pid-file",
            str(pid_file),
        ],
        cwd=REPO_ROOT,
        env=os.environ | {"SDPO_ACCEPTANCE_PYTHON_RUNNER": sys.executable},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Archive status: present non-empty" in result.stdout
    assert "Archive verification: failed" in result.stdout
    assert "archive acceptance_mode mismatch: expected 'training', got 'no-run'" in result.stderr


def test_sdpo_cuda_acceptance_background_script_status_fails_stopped_process_without_archive(tmp_path):
    pid_file = tmp_path / "acceptance.pid"
    pid_file.write_text("not-a-pid\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_BACKGROUND_SCRIPT),
            "--status",
            "--output-root",
            str(tmp_path / "acceptance"),
            "--archive",
            str(tmp_path / "proof.tar.gz"),
            "--log",
            str(tmp_path / "acceptance.log"),
            "--pid-file",
            str(pid_file),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Recorded PID: not-a-pid" in result.stdout
    assert "Process status: not running" in result.stdout
    assert "Archive status: missing" in result.stdout
    assert "Completion status: failed or incomplete" in result.stdout


def test_sdpo_cuda_acceptance_background_script_preflight_only_checks_host(tmp_path):
    fake_bin = _write_fake_sdpo_cuda_host_tools(tmp_path, gpu_count=3)
    fake_acceptance = _write_fake_sdpo_cuda_acceptance_runner(tmp_path)

    result = subprocess.run(
        ["bash", str(ACCEPTANCE_BACKGROUND_SCRIPT), "--preflight-only"],
        cwd=REPO_ROOT,
        env=os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}", "SDPO_ACCEPTANCE_RUNNER": str(fake_acceptance)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Running SDPO CUDA acceptance host preflight..." in result.stdout
    assert "Visible GPUs: 3" in result.stdout
    assert "Host preflight passed." in result.stdout
    assert "Running SDPO CUDA acceptance config preflight..." in result.stdout
    assert "ACCEPTANCE <--check-config>" in result.stdout
    assert "Config preflight passed." in result.stdout
    assert "Preflight only; not starting SDPO CUDA acceptance." in result.stdout
    assert "Started SDPO CUDA acceptance" not in result.stdout


def test_sdpo_cuda_acceptance_background_script_preflight_rejects_insufficient_gpus(tmp_path):
    fake_bin = _write_fake_sdpo_cuda_host_tools(tmp_path, gpu_count=2)
    fake_acceptance = _write_fake_sdpo_cuda_acceptance_runner(tmp_path)

    result = subprocess.run(
        ["bash", str(ACCEPTANCE_BACKGROUND_SCRIPT), "--preflight-only"],
        cwd=REPO_ROOT,
        env=os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}", "SDPO_ACCEPTANCE_RUNNER": str(fake_acceptance)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Running SDPO CUDA acceptance host preflight..." in result.stdout
    assert "Running SDPO CUDA acceptance config preflight..." not in result.stdout
    assert "requires at least 3 visible GPUs, found 2" in result.stderr


def test_sdpo_cuda_acceptance_background_script_preflight_rejects_bad_config(tmp_path):
    fake_bin = _write_fake_sdpo_cuda_host_tools(tmp_path, gpu_count=3)
    fake_acceptance = _write_fake_sdpo_cuda_acceptance_runner(tmp_path)

    result = subprocess.run(
        ["bash", str(ACCEPTANCE_BACKGROUND_SCRIPT), "--preflight-only"],
        cwd=REPO_ROOT,
        env=os.environ
        | {
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SDPO_ACCEPTANCE_RUNNER": str(fake_acceptance),
            "SDPO_FAKE_ACCEPTANCE_CONFIG_FAIL": "1",
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 7
    assert "Host preflight passed." in result.stdout
    assert "Running SDPO CUDA acceptance config preflight..." in result.stdout
    assert "fake acceptance config failed" in result.stderr
    assert "Preflight only; not starting SDPO CUDA acceptance." not in result.stdout
    assert "Started SDPO CUDA acceptance" not in result.stdout


def test_sdpo_cuda_acceptance_background_script_rejects_noop_preflight_only(tmp_path):
    fake_acceptance = _write_fake_sdpo_cuda_acceptance_runner(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_BACKGROUND_SCRIPT),
            "--preflight-only",
            "--skip-host-preflight",
            "--skip-config-preflight",
        ],
        cwd=REPO_ROOT,
        env=os.environ | {"SDPO_ACCEPTANCE_RUNNER": str(fake_acceptance)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "would not run any checks because both preflights are skipped" in result.stderr
    assert "Preflight only; not starting SDPO CUDA acceptance." not in result.stdout
    assert "Started SDPO CUDA acceptance" not in result.stdout


def test_sdpo_cuda_acceptance_background_script_rejects_running_pid_before_preflight(tmp_path):
    fake_bin = _write_fake_sdpo_cuda_host_tools(tmp_path, gpu_count=3)
    fake_acceptance = _write_fake_sdpo_cuda_acceptance_runner(tmp_path)
    pid_file = tmp_path / "acceptance.pid"
    sleeper = subprocess.Popen(["sleep", "60"])
    try:
        pid_file.write_text(f"{sleeper.pid}\n", encoding="utf-8")

        result = subprocess.run(
            [
                "bash",
                str(ACCEPTANCE_BACKGROUND_SCRIPT),
                "--output-root",
                str(tmp_path / "acceptance"),
                "--archive",
                str(tmp_path / "proof.tar.gz"),
                "--log",
                str(tmp_path / "acceptance.log"),
                "--pid-file",
                str(pid_file),
            ],
            cwd=REPO_ROOT,
            env=os.environ
            | {"PATH": f"{fake_bin}:{os.environ['PATH']}", "SDPO_ACCEPTANCE_RUNNER": str(fake_acceptance)},
            capture_output=True,
            text=True,
        )
    finally:
        sleeper.terminate()
        try:
            sleeper.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sleeper.kill()
            sleeper.wait(timeout=5)

    assert result.returncode == 2
    assert f"existing SDPO CUDA acceptance process appears to be running: pid={sleeper.pid}" in result.stderr
    assert f"PID file: {pid_file}" in result.stderr
    assert "Running SDPO CUDA acceptance host preflight..." not in result.stdout
    assert "Running SDPO CUDA acceptance config preflight..." not in result.stdout
    assert "Started SDPO CUDA acceptance" not in result.stdout


def test_sdpo_cuda_acceptance_background_script_start_prints_status_command(tmp_path):
    fake_acceptance = _write_fake_sdpo_cuda_acceptance_runner(tmp_path)
    output_root = tmp_path / "acceptance"
    archive_path = tmp_path / "proof.tar.gz"
    log_path = tmp_path / "acceptance.log"
    pid_file = tmp_path / "acceptance.pid"

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_BACKGROUND_SCRIPT),
            "--skip-host-preflight",
            "--skip-config-preflight",
            "--output-root",
            str(output_root),
            "--archive",
            str(archive_path),
            "--log",
            str(log_path),
            "--pid-file",
            str(pid_file),
        ],
        cwd=REPO_ROOT,
        env=os.environ | {"SDPO_ACCEPTANCE_RUNNER": str(fake_acceptance)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Started SDPO CUDA acceptance in background: pid=" in result.stdout
    assert "Status:" in result.stdout
    assert "scripts/start_sdpo_cuda_acceptance_background.sh --status" in result.stdout
    assert f"--output-root {output_root}" in result.stdout
    assert f"--archive {archive_path}" in result.stdout
    assert f"--log {log_path}" in result.stdout
    assert f"--pid-file {pid_file}" in result.stdout
    assert "Monitor:" in result.stdout
    assert f"scp USER@HOST:{archive_path} ." in result.stdout
    assert f"USER@HOST:{REPO_ROOT}/{archive_path}" not in result.stdout
    assert (
        "uv run python scripts/verify_sdpo_cuda_acceptance_archive.py "
        f"--expected-acceptance-mode training {archive_path.name}"
    ) in result.stdout
    assert "Expected verifier output includes: raw_artifacts=verified" in result.stdout
    assert pid_file.is_file()


def test_sdpo_cuda_acceptance_background_script_prints_configured_python_runner(tmp_path):
    fake_acceptance = _write_fake_sdpo_cuda_acceptance_runner(tmp_path)
    archive_path = tmp_path / "proof.tar.gz"

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_BACKGROUND_SCRIPT),
            "--skip-host-preflight",
            "--skip-config-preflight",
            "--output-root",
            str(tmp_path / "acceptance"),
            "--archive",
            str(archive_path),
            "--log",
            str(tmp_path / "acceptance.log"),
            "--pid-file",
            str(tmp_path / "acceptance.pid"),
        ],
        cwd=REPO_ROOT,
        env=os.environ
        | {
            "SDPO_ACCEPTANCE_RUNNER": str(fake_acceptance),
            "SDPO_ACCEPTANCE_PYTHON_RUNNER": "custom-python --verify",
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (
        f"custom-python --verify scripts/verify_sdpo_cuda_acceptance_archive.py "
        f"--expected-acceptance-mode training {archive_path}"
    ) in result.stdout
    assert (
        "custom-python --verify scripts/verify_sdpo_cuda_acceptance_archive.py "
        f"--expected-acceptance-mode training {archive_path.name}"
    ) in result.stdout


def test_sdpo_cuda_acceptance_background_script_removes_stale_archive_before_start(tmp_path):
    fake_acceptance = _write_fake_sdpo_cuda_acceptance_runner(tmp_path)
    output_root = tmp_path / "acceptance"
    archive_path = tmp_path / "proof.tar.gz"
    log_path = tmp_path / "acceptance.log"
    pid_file = tmp_path / "acceptance.pid"
    archive_path.write_text("stale proof\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_BACKGROUND_SCRIPT),
            "--skip-host-preflight",
            "--skip-config-preflight",
            "--output-root",
            str(output_root),
            "--archive",
            str(archive_path),
            "--log",
            str(log_path),
            "--pid-file",
            str(pid_file),
        ],
        cwd=REPO_ROOT,
        env=os.environ | {"SDPO_ACCEPTANCE_RUNNER": str(fake_acceptance)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"Removed existing SDPO CUDA acceptance archive before start: {archive_path}" in result.stdout
    assert "Started SDPO CUDA acceptance in background: pid=" in result.stdout
    assert not archive_path.exists()
    assert pid_file.is_file()


def test_sdpo_cuda_acceptance_background_script_rejects_non_file_archive_path_before_start(tmp_path):
    fake_acceptance = _write_fake_sdpo_cuda_acceptance_runner(tmp_path)
    archive_path = tmp_path / "proof.tar.gz"
    archive_path.mkdir()

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_BACKGROUND_SCRIPT),
            "--skip-host-preflight",
            "--skip-config-preflight",
            "--output-root",
            str(tmp_path / "acceptance"),
            "--archive",
            str(archive_path),
            "--log",
            str(tmp_path / "acceptance.log"),
            "--pid-file",
            str(tmp_path / "acceptance.pid"),
        ],
        cwd=REPO_ROOT,
        env=os.environ | {"SDPO_ACCEPTANCE_RUNNER": str(fake_acceptance)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "archive path exists and is not a regular file" in result.stderr
    assert str(archive_path) in result.stderr
    assert "Started SDPO CUDA acceptance" not in result.stdout


def test_sdpo_cuda_acceptance_background_script_rejects_broken_symlink_archive_path_before_start(tmp_path):
    fake_acceptance = _write_fake_sdpo_cuda_acceptance_runner(tmp_path)
    archive_path = tmp_path / "proof.tar.gz"
    pid_file = tmp_path / "acceptance.pid"
    archive_path.symlink_to(tmp_path / "missing-target.tar.gz")

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_BACKGROUND_SCRIPT),
            "--skip-host-preflight",
            "--skip-config-preflight",
            "--output-root",
            str(tmp_path / "acceptance"),
            "--archive",
            str(archive_path),
            "--log",
            str(tmp_path / "acceptance.log"),
            "--pid-file",
            str(pid_file),
        ],
        cwd=REPO_ROOT,
        env=os.environ | {"SDPO_ACCEPTANCE_RUNNER": str(fake_acceptance)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "archive path exists and is not a regular file" in result.stderr
    assert str(archive_path) in result.stderr
    assert "Started SDPO CUDA acceptance" not in result.stdout
    assert not pid_file.exists()


def test_sdpo_cuda_acceptance_background_script_rejects_archive_inside_output_root_before_start(tmp_path):
    fake_acceptance = _write_fake_sdpo_cuda_acceptance_runner(tmp_path)
    output_root = tmp_path / "acceptance"
    archive_path = output_root / "proof.tar.gz"
    pid_file = tmp_path / "acceptance.pid"

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_BACKGROUND_SCRIPT),
            "--skip-host-preflight",
            "--skip-config-preflight",
            "--output-root",
            str(output_root),
            "--archive",
            str(archive_path),
            "--log",
            str(tmp_path / "acceptance.log"),
            "--pid-file",
            str(pid_file),
        ],
        cwd=REPO_ROOT,
        env=os.environ | {"SDPO_ACCEPTANCE_RUNNER": str(fake_acceptance)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--archive must be outside --output-root" in result.stderr
    assert str(archive_path) in result.stderr
    assert str(output_root) in result.stderr
    assert "Started SDPO CUDA acceptance" not in result.stdout
    assert not pid_file.exists()


def test_sdpo_cuda_acceptance_background_script_rejects_archive_inside_symlinked_output_root(tmp_path):
    fake_acceptance = _write_fake_sdpo_cuda_acceptance_runner(tmp_path)
    real_output_root = tmp_path / "real-acceptance"
    output_root = tmp_path / "linked-acceptance"
    real_output_root.mkdir()
    output_root.symlink_to(real_output_root, target_is_directory=True)
    archive_path = real_output_root / "proof.tar.gz"
    pid_file = tmp_path / "acceptance.pid"

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_BACKGROUND_SCRIPT),
            "--skip-host-preflight",
            "--skip-config-preflight",
            "--output-root",
            str(output_root),
            "--archive",
            str(archive_path),
            "--log",
            str(tmp_path / "acceptance.log"),
            "--pid-file",
            str(pid_file),
        ],
        cwd=REPO_ROOT,
        env=os.environ | {"SDPO_ACCEPTANCE_RUNNER": str(fake_acceptance)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--archive must be outside --output-root" in result.stderr
    assert str(archive_path) in result.stderr
    assert str(output_root) in result.stderr
    assert "Started SDPO CUDA acceptance" not in result.stdout
    assert not pid_file.exists()


def test_sdpo_local_validation_script_has_valid_shell_syntax():
    result = subprocess.run(["bash", "-n", str(LOCAL_VALIDATION_SCRIPT)], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_help():
    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Verify an SDPO CUDA acceptance proof tarball" in result.stdout
    assert "--expected-acceptance-mode" in result.stdout


def test_sdpo_smoke_script_documents_no_run_output_dir_requirement():
    result = subprocess.run(["bash", str(SMOKE_SCRIPT), "--help"], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "--no-run" in result.stdout
    assert "requires --output-dir" in result.stdout


def test_sdpo_smoke_script_no_run_requires_explicit_output_dir():
    result = subprocess.run(["bash", str(SMOKE_SCRIPT), "--no-run"], capture_output=True, text=True)

    assert result.returncode == 2
    assert "--no-run requires --output-dir" in result.stderr
    assert "uv run" not in result.stderr


def test_sdpo_smoke_script_rejects_clean_output_dir_with_no_run():
    result = subprocess.run(
        [
            "bash",
            str(SMOKE_SCRIPT),
            "--no-run",
            "--clean-output-dir",
            "--output-dir",
            "outputs/existing-sdpo-smoke",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--clean-output-dir cannot be combined with --no-run" in result.stderr
    assert "uv run" not in result.stderr


def test_sdpo_smoke_script_check_config_prints_algorithm_topk_support(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--check-config"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "max_steps=20" in result.stdout
    assert "seq_len=2048" in result.stdout
    assert "orchestrator.algo.distillation_topk=100" in result.stdout
    assert "orchestrator.algo.distillation_topk_support=student" in result.stdout
    assert "orchestrator.algo.model=policy" in result.stdout
    assert "orchestrator.batch_size=32" in result.stdout
    assert "orchestrator.group_size=8" in result.stdout
    assert "orchestrator.renderer.name=qwen3" in result.stdout
    assert "orchestrator.train.sampling.max_completion_tokens=128" in result.stdout
    assert "orchestrator.train.env_ids=['reverse-text']" in result.stdout
    assert "orchestrator.eval.interval=1" in result.stdout
    assert "orchestrator.eval.num_examples=128" in result.stdout
    assert "orchestrator.eval.sampling.max_completion_tokens=128" in result.stdout
    assert "orchestrator.eval.env_ids=['reverse-text']" in result.stdout
    assert "orchestrator.algo.preflight_export_timeout_s=600" in result.stdout
    assert "orchestrator.algo.teacher_regularization=live-policy" in result.stdout
    assert "orchestrator.algo.success_reward_threshold=0.5" in result.stdout
    assert "orchestrator.algo.successful_demonstration_selection=batch_order" in result.stdout
    assert "orchestrator.algo.dont_reprompt_on_self_success=True" in result.stdout
    assert "orchestrator.algo.include_environment_feedback=True" in result.stdout
    assert "orchestrator.algo.max_reprompt_len=10240" in result.stdout
    assert "orchestrator.algo.reprompt_truncation=right" in result.stdout
    assert "orchestrator.algo.multi_turn=False" in result.stdout
    assert "orchestrator.algo.template_target=first_user" in result.stdout
    assert (
        "orchestrator.algo.template='{question}{successful_solution_block}{feedback_block}\\n\\n"
        "Correctly solve the original question.'"
    ) in result.stdout
    assert (
        "orchestrator.algo.solution_template='\\nCorrect solution:\\n\\n{successful_previous_attempt}'" in result.stdout
    )
    assert (
        "orchestrator.algo.feedback_template='\\nThe following is feedback from your unsuccessful "
        "earlier attempt:\\n\\n{feedback_raw}'"
    ) in result.stdout
    assert "trainer.model.cp=1" in result.stdout
    assert "trainer.sdpo_loss.distillation_add_tail=True" in result.stdout
    assert "trainer.sdpo_loss.rollout_is=token" in result.stdout
    assert "trainer.sdpo_runtime.teacher_update_rate=0.05" in result.stdout
    assert "SDPO smoke config checks passed." in result.stdout


def test_sdpo_smoke_script_check_config_ignores_no_run_verification_guards(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--check-config", "--no-run", "--clean-output-dir"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "clean_output_dir=False" in result.stdout
    assert "SDPO smoke config checks passed." in result.stdout
    assert "--no-run requires --output-dir" not in result.stderr
    assert "--clean-output-dir cannot be combined with --no-run" not in result.stderr
    assert "VERIFY" not in result.stdout
    assert "TRAIN" not in result.stdout


def test_sdpo_smoke_script_check_config_prints_ema_algorithm_mode(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--ema", "--check-config"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "uses_sdpo_internal_teacher_regularization=True" in result.stdout
    assert "deployment.num_sdpo_teacher_gpus=1" in result.stdout
    assert "orchestrator.algo.distillation_topk_support=student" in result.stdout
    assert "orchestrator.algo.model=policy" in result.stdout
    assert "orchestrator.algo.teacher_regularization=ema" in result.stdout
    assert "orchestrator.sdpo_teacher.base_url=['http://localhost:8001/v1']" in result.stdout
    assert "SDPO smoke config checks passed." in result.stdout


def test_sdpo_smoke_script_check_config_rejects_non_reference_values(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SDPO_FAKE_ALGO_DISTILLATION_TOPK": "64",
    }

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--check-config"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "orchestrator.algo.distillation_topk=100" in result.stdout
    assert "orchestrator.algo.distillation_topk to match trainer.sdpo_loss.distillation_topk" in result.stderr
    assert "SDPO smoke config checks passed." not in result.stdout
    assert "uv run rl" not in result.stdout


@pytest.mark.parametrize(
    ("env_var", "value", "message"),
    [
        ("SDPO_FAKE_MAX_STEPS", "2", "max_steps=20"),
        ("SDPO_FAKE_SEQ_LEN", "1024", "seq_len=2048"),
        ("SDPO_FAKE_BATCH_SIZE", "8", "orchestrator.batch_size=32"),
        ("SDPO_FAKE_GROUP_SIZE", "1", "orchestrator.group_size=8"),
        ("SDPO_FAKE_RENDERER_NAME", "auto", "orchestrator.renderer.name='qwen3'"),
        (
            "SDPO_FAKE_TRAIN_MAX_COMPLETION_TOKENS",
            "64",
            "orchestrator.train.sampling.max_completion_tokens=128",
        ),
        ("SDPO_FAKE_TRAIN_ENV_IDS", "['other-env']", "orchestrator.train.env_ids=['reverse-text']"),
        ("SDPO_FAKE_EVAL_INTERVAL", "5", "orchestrator.eval.interval=1"),
        ("SDPO_FAKE_EVAL_NUM_EXAMPLES", "32", "orchestrator.eval.num_examples=128"),
        (
            "SDPO_FAKE_EVAL_MAX_COMPLETION_TOKENS",
            "64",
            "orchestrator.eval.sampling.max_completion_tokens=128",
        ),
        ("SDPO_FAKE_EVAL_ENV_IDS", "['other-env']", "orchestrator.eval.env_ids=['reverse-text']"),
    ],
)
def test_sdpo_smoke_script_rejects_non_reference_run_shape(tmp_path, env_var, value, message):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        env_var: value,
    }

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--no-run", "--output-dir", "outputs/non-reference-run-shape"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert "VERIFY" not in result.stdout
    assert "uv run rl" not in result.stdout


def _write_fake_uv_for_no_run_smoke(tmp_path: Path) -> Path:
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
	if [[ "$1" != "run" ]]; then
	  echo "unexpected uv invocation: $*" >&2
	  exit 9
	fi
	if [[ "$2" == "rl" ]]; then
	  printf 'TRAIN'
	  shift 2
	  for arg in "$@"; do
	    printf ' <%s>' "$arg"
	  done
	  printf '\\n'
	  exit 0
	fi
	if [[ "$2" != "python" ]]; then
	  echo "unexpected uv invocation: $*" >&2
	  exit 9
	fi
if [[ "$3" == "-" ]]; then
  cat >/dev/null
  mode="live"
  for arg in "$@"; do
    if [[ "$arg" == *sdpo_huebotter_reference_ema_smoke.toml ]]; then
      mode="ema"
    fi
  done
  if [[ "$4" == "@" ]]; then
    teacher_regularization="live-policy"
    uses_internal="False"
    teacher_base_url="None"
    if [[ "$mode" == "ema" ]]; then
      teacher_regularization="ema"
      uses_internal="True"
      teacher_base_url="['http://localhost:8001/v1']"
    fi
    cat <<EOF
config_output_dir=outputs/fake
clean_output_dir=False
max_steps=${SDPO_FAKE_MAX_STEPS:-20}
seq_len=${SDPO_FAKE_SEQ_LEN:-2048}
uses_sdpo_student_support=True
uses_sdpo_internal_teacher_regularization=${uses_internal}
deployment.num_sdpo_teacher_gpus=$([[ "${mode}" == "ema" ]] && echo "${SDPO_FAKE_NUM_SDPO_TEACHER_GPUS:-1}" || echo "${SDPO_FAKE_NUM_SDPO_TEACHER_GPUS:-0}")
orchestrator.batch_size=${SDPO_FAKE_BATCH_SIZE:-32}
orchestrator.group_size=${SDPO_FAKE_GROUP_SIZE:-8}
orchestrator.renderer.name=${SDPO_FAKE_RENDERER_NAME:-qwen3}
orchestrator.train.sampling.max_completion_tokens=${SDPO_FAKE_TRAIN_MAX_COMPLETION_TOKENS:-128}
orchestrator.train.env_ids=${SDPO_FAKE_TRAIN_ENV_IDS:-['reverse-text']}
orchestrator.eval.interval=${SDPO_FAKE_EVAL_INTERVAL:-1}
orchestrator.eval.num_examples=${SDPO_FAKE_EVAL_NUM_EXAMPLES:-128}
orchestrator.eval.sampling.max_completion_tokens=${SDPO_FAKE_EVAL_MAX_COMPLETION_TOKENS:-128}
orchestrator.eval.env_ids=${SDPO_FAKE_EVAL_ENV_IDS:-['reverse-text']}
orchestrator.algo.distillation_topk=100
orchestrator.algo.distillation_topk_support=student
orchestrator.algo.model=policy
orchestrator.algo.preflight_export_timeout_s=600
orchestrator.algo.teacher_regularization=${teacher_regularization}
orchestrator.algo.success_reward_threshold=0.5
orchestrator.algo.successful_demonstration_selection=${SDPO_FAKE_SUCCESSFUL_DEMONSTRATION_SELECTION:-batch_order}
orchestrator.algo.dont_reprompt_on_self_success=True
orchestrator.algo.remove_thinking_from_demonstration=True
orchestrator.algo.include_environment_feedback=True
orchestrator.algo.environment_feedback_only_without_solution=True
orchestrator.algo.max_reprompt_len=10240
orchestrator.algo.reprompt_truncation=right
orchestrator.algo.assistant_prefix=
orchestrator.algo.multi_turn=False
orchestrator.algo.template_target=${SDPO_FAKE_TEMPLATE_TARGET:-first_user}
orchestrator.algo.template='{question}{successful_solution_block}{feedback_block}\\n\\nCorrectly solve the original question.'
orchestrator.algo.solution_template='\\nCorrect solution:\\n\\n{successful_previous_attempt}'
orchestrator.algo.feedback_template='\\nThe following is feedback from your unsuccessful earlier attempt:\\n\\n{feedback_raw}'
trainer.enable_token_export=True
trainer.model.cp=1
trainer.model.fused_lm_head_token_chunk_size=disabled
trainer.sdpo_loss.full_logit_distillation=True
trainer.sdpo_loss.distillation_topk=100
trainer.sdpo_loss.distillation_add_tail=True
trainer.sdpo_loss.alpha=0.5
trainer.sdpo_loss.is_clip=2.0
trainer.sdpo_loss.rollout_is=token
trainer.sdpo_loss.rollout_is_threshold=2.0
trainer.sdpo_loss.rollout_is_batch_normalize=False
trainer.sdpo_runtime.teacher_regularization=${teacher_regularization}
trainer.sdpo_runtime.teacher_update_rate=${SDPO_FAKE_TRAINER_TEACHER_UPDATE_RATE:-0.05}
orchestrator.algo.teacher_update_rate=${SDPO_FAKE_ALGO_TEACHER_UPDATE_RATE:-0.05}
orchestrator.sdpo_teacher.base_url=${teacher_base_url}
EOF
    exit 0
  fi
  field="$4"
  case "$field" in
    max_steps)
      if [[ -n "${SDPO_FAKE_MAX_STEPS:-}" ]]; then echo "${SDPO_FAKE_MAX_STEPS}"; else echo "20"; fi
      ;;
    seq_len)
      if [[ -n "${SDPO_FAKE_SEQ_LEN:-}" ]]; then echo "${SDPO_FAKE_SEQ_LEN}"; else echo "2048"; fi
      ;;
    orchestrator.batch_size)
      if [[ -n "${SDPO_FAKE_BATCH_SIZE:-}" ]]; then echo "${SDPO_FAKE_BATCH_SIZE}"; else echo "32"; fi
      ;;
    orchestrator.group_size)
      if [[ -n "${SDPO_FAKE_GROUP_SIZE:-}" ]]; then echo "${SDPO_FAKE_GROUP_SIZE}"; else echo "8"; fi
      ;;
    orchestrator.renderer.name)
      if [[ -n "${SDPO_FAKE_RENDERER_NAME:-}" ]]; then echo "${SDPO_FAKE_RENDERER_NAME}"; else echo "qwen3"; fi
      ;;
    orchestrator.train.sampling.max_completion_tokens)
      if [[ -n "${SDPO_FAKE_TRAIN_MAX_COMPLETION_TOKENS:-}" ]]; then echo "${SDPO_FAKE_TRAIN_MAX_COMPLETION_TOKENS}"; else echo "128"; fi
      ;;
    orchestrator.train.env_ids)
      if [[ -n "${SDPO_FAKE_TRAIN_ENV_IDS:-}" ]]; then echo "${SDPO_FAKE_TRAIN_ENV_IDS}"; else echo "['reverse-text']"; fi
      ;;
    orchestrator.eval.interval)
      if [[ -n "${SDPO_FAKE_EVAL_INTERVAL:-}" ]]; then echo "${SDPO_FAKE_EVAL_INTERVAL}"; else echo "1"; fi
      ;;
    orchestrator.eval.num_examples)
      if [[ -n "${SDPO_FAKE_EVAL_NUM_EXAMPLES:-}" ]]; then echo "${SDPO_FAKE_EVAL_NUM_EXAMPLES}"; else echo "128"; fi
      ;;
    orchestrator.eval.sampling.max_completion_tokens)
      if [[ -n "${SDPO_FAKE_EVAL_MAX_COMPLETION_TOKENS:-}" ]]; then echo "${SDPO_FAKE_EVAL_MAX_COMPLETION_TOKENS}"; else echo "128"; fi
      ;;
    orchestrator.eval.env_ids)
      if [[ -n "${SDPO_FAKE_EVAL_ENV_IDS:-}" ]]; then echo "${SDPO_FAKE_EVAL_ENV_IDS}"; else echo "['reverse-text']"; fi
      ;;
    trainer.sdpo_loss.distillation_topk) echo "100" ;;
    orchestrator.algo.distillation_topk)
      if [[ -n "${SDPO_FAKE_ALGO_DISTILLATION_TOPK:-}" ]]; then echo "${SDPO_FAKE_ALGO_DISTILLATION_TOPK}"; else echo "100"; fi
      ;;
    orchestrator.algo.distillation_topk_support)
      if [[ -n "${SDPO_FAKE_ALGO_DISTILLATION_TOPK_SUPPORT:-}" ]]; then echo "${SDPO_FAKE_ALGO_DISTILLATION_TOPK_SUPPORT}"; else echo "student"; fi
      ;;
    orchestrator.algo.model)
      if [[ -n "${SDPO_FAKE_ALGO_MODEL:-}" ]]; then echo "${SDPO_FAKE_ALGO_MODEL}"; else echo "policy"; fi
      ;;
    orchestrator.algo.preflight_export_timeout_s)
      if [[ -n "${SDPO_FAKE_PREFLIGHT_EXPORT_TIMEOUT_S:-}" ]]; then echo "${SDPO_FAKE_PREFLIGHT_EXPORT_TIMEOUT_S}"; else echo "600"; fi
      ;;
    orchestrator.algo.success_reward_threshold)
      if [[ -n "${SDPO_FAKE_SUCCESS_REWARD_THRESHOLD:-}" ]]; then echo "${SDPO_FAKE_SUCCESS_REWARD_THRESHOLD}"; else echo "0.5"; fi
      ;;
    orchestrator.algo.successful_demonstration_selection)
      if [[ -n "${SDPO_FAKE_SUCCESSFUL_DEMONSTRATION_SELECTION:-}" ]]; then echo "${SDPO_FAKE_SUCCESSFUL_DEMONSTRATION_SELECTION}"; else echo "batch_order"; fi
      ;;
    orchestrator.algo.dont_reprompt_on_self_success)
      if [[ -n "${SDPO_FAKE_DONT_REPROMPT_ON_SELF_SUCCESS:-}" ]]; then echo "${SDPO_FAKE_DONT_REPROMPT_ON_SELF_SUCCESS}"; else echo "True"; fi
      ;;
    orchestrator.algo.remove_thinking_from_demonstration)
      if [[ -n "${SDPO_FAKE_REMOVE_THINKING_FROM_DEMONSTRATION:-}" ]]; then echo "${SDPO_FAKE_REMOVE_THINKING_FROM_DEMONSTRATION}"; else echo "True"; fi
      ;;
    orchestrator.algo.include_environment_feedback)
      if [[ -n "${SDPO_FAKE_INCLUDE_ENVIRONMENT_FEEDBACK:-}" ]]; then echo "${SDPO_FAKE_INCLUDE_ENVIRONMENT_FEEDBACK}"; else echo "True"; fi
      ;;
    orchestrator.algo.environment_feedback_only_without_solution)
      if [[ -n "${SDPO_FAKE_ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION:-}" ]]; then echo "${SDPO_FAKE_ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION}"; else echo "True"; fi
      ;;
    orchestrator.algo.max_reprompt_len)
      if [[ -n "${SDPO_FAKE_MAX_REPROMPT_LEN:-}" ]]; then echo "${SDPO_FAKE_MAX_REPROMPT_LEN}"; else echo "10240"; fi
      ;;
    orchestrator.algo.reprompt_truncation)
      if [[ -n "${SDPO_FAKE_REPROMPT_TRUNCATION:-}" ]]; then echo "${SDPO_FAKE_REPROMPT_TRUNCATION}"; else echo "right"; fi
      ;;
    orchestrator.algo.assistant_prefix)
      if [[ -n "${SDPO_FAKE_ASSISTANT_PREFIX:-}" ]]; then echo "${SDPO_FAKE_ASSISTANT_PREFIX}"; else echo ""; fi
      ;;
    orchestrator.algo.multi_turn)
      if [[ -n "${SDPO_FAKE_MULTI_TURN:-}" ]]; then echo "${SDPO_FAKE_MULTI_TURN}"; else echo "False"; fi
      ;;
    orchestrator.algo.template_target)
      if [[ -n "${SDPO_FAKE_TEMPLATE_TARGET:-}" ]]; then echo "${SDPO_FAKE_TEMPLATE_TARGET}"; else echo "first_user"; fi
      ;;
    orchestrator.algo.template)
      if [[ -n "${SDPO_FAKE_TEMPLATE:-}" ]]; then
        printf '%s\n' "${SDPO_FAKE_TEMPLATE}"
      else
        printf '%s\n' $'{question}{successful_solution_block}{feedback_block}\n\nCorrectly solve the original question.'
      fi
      ;;
    orchestrator.algo.solution_template)
      if [[ -n "${SDPO_FAKE_SOLUTION_TEMPLATE:-}" ]]; then
        printf '%s\n' "${SDPO_FAKE_SOLUTION_TEMPLATE}"
      else
        printf '%s\n' $'\nCorrect solution:\n\n{successful_previous_attempt}'
      fi
      ;;
    orchestrator.algo.feedback_template)
      if [[ -n "${SDPO_FAKE_FEEDBACK_TEMPLATE:-}" ]]; then
        printf '%s\n' "${SDPO_FAKE_FEEDBACK_TEMPLATE}"
      else
        printf '%s\n' $'\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}'
      fi
      ;;
    trainer.sdpo_loss.full_logit_distillation)
      if [[ "${SDPO_FAKE_FULL_LOGIT_DISTILLATION:-True}" == "False" ]]; then echo "False"; else echo "True"; fi
      ;;
    trainer.sdpo_loss.distillation_add_tail)
      if [[ -n "${SDPO_FAKE_DISTILLATION_ADD_TAIL:-}" ]]; then echo "${SDPO_FAKE_DISTILLATION_ADD_TAIL}"; else echo "True"; fi
      ;;
    trainer.sdpo_loss.alpha)
      if [[ -n "${SDPO_FAKE_SDPO_ALPHA:-}" ]]; then echo "${SDPO_FAKE_SDPO_ALPHA}"; else echo "0.5"; fi
      ;;
    trainer.sdpo_loss.is_clip)
      if [[ -n "${SDPO_FAKE_IS_CLIP:-}" ]]; then echo "${SDPO_FAKE_IS_CLIP}"; else echo "2.0"; fi
      ;;
    trainer.sdpo_loss.rollout_is)
      if [[ -n "${SDPO_FAKE_ROLLOUT_IS:-}" ]]; then echo "${SDPO_FAKE_ROLLOUT_IS}"; else echo "token"; fi
      ;;
    trainer.sdpo_loss.rollout_is_threshold)
      if [[ -n "${SDPO_FAKE_ROLLOUT_IS_THRESHOLD:-}" ]]; then echo "${SDPO_FAKE_ROLLOUT_IS_THRESHOLD}"; else echo "2.0"; fi
      ;;
    trainer.sdpo_loss.rollout_is_batch_normalize)
      if [[ -n "${SDPO_FAKE_ROLLOUT_IS_BATCH_NORMALIZE:-}" ]]; then echo "${SDPO_FAKE_ROLLOUT_IS_BATCH_NORMALIZE}"; else echo "False"; fi
      ;;
    uses_sdpo_student_support) echo "True" ;;
    uses_sdpo_internal_teacher_regularization)
      if [[ "$mode" == "ema" ]]; then echo "True"; else echo "False"; fi
      ;;
    deployment.num_sdpo_teacher_gpus)
      if [[ "$mode" == "ema" ]]; then echo "${SDPO_FAKE_NUM_SDPO_TEACHER_GPUS:-1}"; else echo "${SDPO_FAKE_NUM_SDPO_TEACHER_GPUS:-0}"; fi
      ;;
    orchestrator.sdpo_teacher.base_url)
      if [[ -n "${SDPO_FAKE_SDPO_TEACHER_BASE_URL:-}" ]]; then
        echo "${SDPO_FAKE_SDPO_TEACHER_BASE_URL}"
      elif [[ "$mode" == "ema" ]]; then
        echo "['http://localhost:8001/v1']"
      else
        echo "None"
      fi
      ;;
    trainer.enable_token_export) echo "True" ;;
    trainer.model.cp)
      if [[ -n "${SDPO_FAKE_TRAINER_CP:-}" ]]; then echo "${SDPO_FAKE_TRAINER_CP}"; else echo "1"; fi
      ;;
    trainer.model.fused_lm_head_token_chunk_size) echo "disabled" ;;
    trainer.sdpo_runtime.teacher_regularization)
      if [[ "$mode" == "ema" ]]; then echo "ema"; else echo "live-policy"; fi
      ;;
    trainer.sdpo_runtime.teacher_update_rate)
      if [[ -n "${SDPO_FAKE_TRAINER_TEACHER_UPDATE_RATE:-}" ]]; then echo "${SDPO_FAKE_TRAINER_TEACHER_UPDATE_RATE}"; else echo "0.05"; fi
      ;;
    orchestrator.algo.teacher_regularization)
      if [[ -n "${SDPO_FAKE_ALGO_TEACHER_REGULARIZATION:-}" ]]; then
        echo "${SDPO_FAKE_ALGO_TEACHER_REGULARIZATION}"
      elif [[ "$mode" == "ema" ]]; then
        echo "ema"
      else
        echo "live-policy"
      fi
      ;;
    orchestrator.algo.teacher_update_rate)
      if [[ -n "${SDPO_FAKE_ALGO_TEACHER_UPDATE_RATE:-}" ]]; then echo "${SDPO_FAKE_ALGO_TEACHER_UPDATE_RATE}"; else echo "0.05"; fi
      ;;
    *)
      echo "unexpected config field: $field" >&2
      exit 10
      ;;
  esac
  exit 0
fi
if [[ "$3" == "scripts/verify_sdpo_smoke_artifacts.py" ]]; then
  printf 'VERIFY'
  shift 3
  for arg in "$@"; do
    printf ' <%s>' "$arg"
  done
  printf '\\n'
  exit 0
fi
echo "unexpected uv python target: $3" >&2
exit 11
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    return fake_uv


def _write_fake_sdpo_smoke_wrapper(tmp_path: Path) -> Path:
    fake_wrapper = tmp_path / "fake_sdpo_smoke_wrapper.sh"
    token_export_fixture = tmp_path / "sdpo_acceptance_token_export.jsonl"
    token_export_fixture.write_bytes(_sdpo_acceptance_token_export_bytes())
    fake_wrapper.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf 'SMOKE_PYTHON_RUNNER=<%s>\\n' "${SDPO_SMOKE_PYTHON_RUNNER:-}"
mode="live"
output_dir=""
for ((i = 1; i <= $#; i++)); do
  arg="${!i}"
  if [[ "$arg" == "--ema" ]]; then
    mode="ema"
  fi
  if [[ "$arg" == "--output-dir" ]]; then
    next=$((i + 1))
    output_dir="${!next}"
  fi
done
printf 'SMOKE'
for arg in "$@"; do
  printf ' <%s>' "$arg"
done
printf '\\n'
if [[ -n "$output_dir" ]]; then
  config="configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml"
  if [[ "$mode" == "ema" ]]; then
    config="configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml"
  fi
  mkdir -p "$output_dir"
  if [[ "${SDPO_FAKE_SMOKE_SKIP_PROOF_ARTIFACTS:-0}" != "1" ]]; then
    if [[ "${SDPO_FAKE_SMOKE_FULL_PROVENANCE:-0}" == "1" ]]; then
      git_commit="$(git rev-parse HEAD 2>/dev/null || true)"
      git_branch="$(git branch --show-current 2>/dev/null || true)"
      if [[ -z "$git_branch" ]]; then
        git_short_commit="$(git rev-parse --short HEAD 2>/dev/null || true)"
        git_branch="detached-$git_short_commit"
      fi
      teacher_regularization="live-policy"
      if [[ "$mode" == "ema" ]]; then
        teacher_regularization="ema"
      fi
      cat > "$output_dir/sdpo_smoke_provenance.txt" <<EOF
sdpo_smoke_provenance_version=1
mode=$mode
config=$config
output_dir=$output_dir
expected_topk=100
orchestrator.algo.distillation_topk=100
orchestrator.algo.distillation_topk_support=student
orchestrator.algo.teacher_regularization=$teacher_regularization
orchestrator.algo.teacher_update_rate=0.05
orchestrator.algo.success_reward_threshold=0.5
orchestrator.algo.successful_demonstration_selection=batch_order
orchestrator.algo.dont_reprompt_on_self_success=True
orchestrator.algo.remove_thinking_from_demonstration=True
orchestrator.algo.include_environment_feedback=True
orchestrator.algo.environment_feedback_only_without_solution=True
orchestrator.algo.max_reprompt_len=10240
orchestrator.algo.reprompt_truncation=right
orchestrator.algo.assistant_prefix=
orchestrator.algo.multi_turn=False
orchestrator.algo.template_target=first_user
trainer.sdpo_loss.full_logit_distillation=True
trainer.sdpo_loss.distillation_topk=100
trainer.sdpo_loss.distillation_add_tail=True
trainer.sdpo_loss.alpha=0.5
trainer.sdpo_loss.is_clip=2.0
trainer.sdpo_loss.rollout_is=token
trainer.sdpo_loss.rollout_is_threshold=2.0
trainer.sdpo_loss.rollout_is_batch_normalize=False
trainer.sdpo_runtime.teacher_regularization=$teacher_regularization
trainer.sdpo_runtime.teacher_update_rate=0.05
git_commit=$git_commit
git_branch=$git_branch
git_diff_sha256=df087996dd8d479802e485eb659e5f059fa914a26b0ee763d173934039a1d087
git_cached_diff_sha256=9a1de98e34e4d5518f4d04f49b51c003979f98b22f519d84cda22b2c66565f62
git_untracked_manifest_sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
python_runner=fake python
rl_runner=fake rl
git_untracked_manifest_begin
git_untracked_manifest_end
git_status_short_begin
 M src/prime_rl/orchestrator/algo/sdpo.py
git_status_short_end
EOF
    else
      echo "sdpo_smoke_provenance_version=1" > "$output_dir/sdpo_smoke_provenance.txt"
    fi
    mkdir -p "$output_dir/run_default/token_exports/step_1"
    cp __TOKEN_EXPORT_FIXTURE__ "$output_dir/run_default/token_exports/step_1/rank_0.jsonl"
    : > "$output_dir/run_default/token_exports/step_1/STABLE"
    if [[ "$mode" == "ema" ]]; then
      mkdir -p "$output_dir/run_default/broadcasts/step_1"
      : > "$output_dir/run_default/broadcasts/step_1/STABLE"
      mkdir -p "$output_dir/run_default/broadcasts/step_1/sdpo_teacher"
      echo "teacher" > "$output_dir/run_default/broadcasts/step_1/sdpo_teacher/model.bin"
      : > "$output_dir/run_default/broadcasts/step_1/sdpo_teacher/STABLE"
    fi
  fi
  {
    echo "Verified SDPO smoke provenance: file=$output_dir/sdpo_smoke_provenance.txt, mode=$mode, config=$config, expected_topk=100"
    echo "Verified SDPO token exports: files=1, records=2, sdpo_records=2, transported_rows=2, student_rows=4, paired_rows=2, matching_support_rows=2, distinct_teacher_logprob_rows=2, importance_ratio_rows=2, rollout_is_weight_rows=2, student_preflight_rows=2, temperature_rows=4, sample_id_records=2, stable_steps=1, steps=['step_1'], paired_steps=['step_1'], matching_support_steps=['step_1'], student_preflight_steps=['step_1'], matched_support_samples=1, matched_support_token_rows=2, distinct_teacher_logprob_token_rows=2, importance_ratio_token_rows=2, rollout_is_weight_token_rows=2"
    if [[ "$mode" == "ema" ]]; then
      echo "Verified SDPO EMA broadcasts: steps=1, role=sdpo_teacher, teacher_steps=1, matched_steps=['step_1'], matched_step_keys=['run_default:step_1']"
    fi
  } > "$output_dir/sdpo_smoke_verify_report.txt"
fi
""".replace("__TOKEN_EXPORT_FIXTURE__", shlex.quote(str(token_export_fixture))),
        encoding="utf-8",
    )
    fake_wrapper.chmod(0o755)
    return fake_wrapper


def _write_fake_sdpo_acceptance_python_runner(tmp_path: Path) -> Path:
    fake_runner = tmp_path / "fake_sdpo_acceptance_python.sh"
    real_python = shlex.quote(sys.executable)
    fake_runner.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${{1:-}}" == "scripts/verify_sdpo_cuda_acceptance_archive.py" ]]; then
  if [[ -n "${{SDPO_FAKE_ACCEPTANCE_ARG_LOG:-}}" ]]; then
    {{
      printf 'ARCHIVE_VERIFY'
      for arg in "$@"; do
        printf ' <%s>' "$arg"
      done
      printf '\\n'
    }} >> "${{SDPO_FAKE_ACCEPTANCE_ARG_LOG}}"
  fi
  exec {real_python} "$@"
fi
require_ema=0
printf 'PYTHON'
for arg in "$@"; do
  if [[ "$arg" == "--require-ema-teacher" ]]; then
    require_ema=1
  fi
  printf ' <%s>' "$arg"
done
printf '\\n'
if [[ "${{SDPO_FAKE_ACCEPTANCE_SKIP_VERIFY_MARKERS:-0}}" != "1" ]]; then
  echo "Verified SDPO smoke provenance: file=$2/sdpo_smoke_provenance.txt, mode=unknown, config=unknown, expected_topk=100"
  echo "Verified SDPO token exports: files=1, records=2, sdpo_records=2, transported_rows=2, student_rows=4, paired_rows=2, matching_support_rows=2, distinct_teacher_logprob_rows=2, importance_ratio_rows=2, rollout_is_weight_rows=2, student_preflight_rows=2, temperature_rows=4, sample_id_records=2, stable_steps=1, steps=['step_1'], paired_steps=['step_1'], matching_support_steps=['step_1'], student_preflight_steps=['step_1'], matched_support_samples=1, matched_support_token_rows=2, distinct_teacher_logprob_token_rows=2, importance_ratio_token_rows=2, rollout_is_weight_token_rows=2"
  if [[ "${{SDPO_FAKE_ACCEPTANCE_DUPLICATE_TOKEN_MARKER:-0}}" == "1" ]]; then
    echo "Verified SDPO token exports: files=1, records=2, sdpo_records=2, transported_rows=2, student_rows=4, paired_rows=2, matching_support_rows=2, distinct_teacher_logprob_rows=2, importance_ratio_rows=2, rollout_is_weight_rows=2, student_preflight_rows=2, temperature_rows=4, sample_id_records=2, stable_steps=1, steps=['step_1'], paired_steps=['step_1'], matching_support_steps=['step_1'], student_preflight_steps=['step_1'], matched_support_samples=1, matched_support_token_rows=2, distinct_teacher_logprob_token_rows=2, importance_ratio_token_rows=2, rollout_is_weight_token_rows=2"
  fi
  if [[ "$require_ema" -eq 1 ]]; then
    echo "Verified SDPO EMA broadcasts: steps=1, role=sdpo_teacher, teacher_steps=1, matched_steps=['step_1'], matched_step_keys=['run_default:step_1']"
  fi
fi
""",
        encoding="utf-8",
    )
    fake_runner.chmod(0o755)
    return fake_runner


def _write_fake_sdpo_cuda_host_tools(tmp_path: Path, *, gpu_count: int) -> Path:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name in ("uv", "tar"):
        tool = fake_bin / name
        tool.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        tool.chmod(0o755)
    names = "\n".join(f"Fake GPU {idx}" for idx in range(gpu_count))
    memory_rows = "\n".join(f"{idx}, Fake GPU {idx}, 80000, 81920" for idx in range(gpu_count))
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
args="$*"
if [[ "$args" == *"--query-gpu=name"* ]]; then
  cat <<'EOF'
{names}
EOF
  exit 0
fi
if [[ "$args" == *"--query-gpu=index,name,memory.free,memory.total"* ]]; then
  cat <<'EOF'
{memory_rows}
EOF
  exit 0
fi
echo "fake nvidia-smi"
""",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    return fake_bin


def _write_fake_sdpo_cuda_acceptance_runner(tmp_path: Path) -> Path:
    fake_runner = tmp_path / "fake_sdpo_cuda_acceptance.sh"
    fake_runner.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${SDPO_FAKE_ACCEPTANCE_CONFIG_FAIL:-0}" == "1" ]]; then
  echo "fake acceptance config failed" >&2
  exit 7
fi
printf 'ACCEPTANCE'
for arg in "$@"; do
  printf ' <%s>' "$arg"
done
printf '\\n'
""",
        encoding="utf-8",
    )
    fake_runner.chmod(0o755)
    return fake_runner


def _write_minimal_sdpo_acceptance_artifacts(
    output_root: Path,
    *,
    git_commit: str | None = None,
    git_branch: str | None = None,
) -> None:
    git_commit = _current_git_commit() if git_commit is None else git_commit
    git_branch = _current_git_branch() if git_branch is None else git_branch
    for mode in ("live", "ema"):
        run_dir = output_root / mode / "run_default"
        (run_dir / "token_exports" / "step_1").mkdir(parents=True)
        (run_dir / "token_exports" / "step_1" / "rank_0.jsonl").write_bytes(_sdpo_acceptance_token_export_bytes())
        (run_dir / "token_exports" / "step_1" / "STABLE").write_text("", encoding="utf-8")
        config = (
            "configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml"
            if mode == "ema"
            else "configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml"
        )
        (output_root / mode / "sdpo_smoke_provenance.txt").write_text(
            _sdpo_acceptance_provenance_bytes(
                mode=mode,
                config=config,
                git_commit=git_commit,
                git_branch=git_branch,
            ).decode("utf-8"),
            encoding="utf-8",
        )
    (output_root / "ema" / "run_default" / "broadcasts" / "step_1").mkdir(parents=True)
    (output_root / "ema" / "run_default" / "broadcasts" / "step_1" / "STABLE").write_text("", encoding="utf-8")
    (output_root / "ema" / "run_default" / "broadcasts" / "step_1" / "sdpo_teacher").mkdir(parents=True)
    (output_root / "ema" / "run_default" / "broadcasts" / "step_1" / "sdpo_teacher" / "model.bin").write_text(
        "teacher\n",
        encoding="utf-8",
    )
    (output_root / "ema" / "run_default" / "broadcasts" / "step_1" / "sdpo_teacher" / "STABLE").write_text(
        "",
        encoding="utf-8",
    )


def _sdpo_acceptance_provenance_bytes(
    *,
    mode: str,
    config: str,
    expected_topk: int = 100,
    git_commit: str = SDPO_TEST_GIT_COMMIT,
    git_branch: str = SDPO_TEST_GIT_BRANCH,
    untracked_manifest_lines: list[str] | None = None,
) -> bytes:
    manifest_lines = [] if untracked_manifest_lines is None else untracked_manifest_lines
    manifest_payload = "" if not manifest_lines else "\n".join(manifest_lines) + "\n"
    manifest_sha = hashlib.sha256(manifest_payload.encode("utf-8")).hexdigest()
    teacher_regularization = "ema" if mode == "ema" else "live-policy"
    return (
        "sdpo_smoke_provenance_version=1\n"
        f"mode={mode}\n"
        f"config={config}\n"
        f"output_dir=outputs/sdpo-cuda-acceptance/{mode}\n"
        f"expected_topk={expected_topk}\n"
        f"orchestrator.algo.distillation_topk={expected_topk}\n"
        "orchestrator.algo.distillation_topk_support=student\n"
        f"orchestrator.algo.teacher_regularization={teacher_regularization}\n"
        "orchestrator.algo.teacher_update_rate=0.05\n"
        "orchestrator.algo.success_reward_threshold=0.5\n"
        "orchestrator.algo.successful_demonstration_selection=batch_order\n"
        "orchestrator.algo.dont_reprompt_on_self_success=True\n"
        "orchestrator.algo.remove_thinking_from_demonstration=True\n"
        "orchestrator.algo.include_environment_feedback=True\n"
        "orchestrator.algo.environment_feedback_only_without_solution=True\n"
        "orchestrator.algo.max_reprompt_len=10240\n"
        "orchestrator.algo.reprompt_truncation=right\n"
        "orchestrator.algo.assistant_prefix=\n"
        "orchestrator.algo.multi_turn=False\n"
        "orchestrator.algo.template_target=first_user\n"
        "trainer.sdpo_loss.full_logit_distillation=True\n"
        f"trainer.sdpo_loss.distillation_topk={expected_topk}\n"
        "trainer.sdpo_loss.distillation_add_tail=True\n"
        "trainer.sdpo_loss.alpha=0.5\n"
        "trainer.sdpo_loss.is_clip=2.0\n"
        "trainer.sdpo_loss.rollout_is=token\n"
        "trainer.sdpo_loss.rollout_is_threshold=2.0\n"
        "trainer.sdpo_loss.rollout_is_batch_normalize=False\n"
        f"trainer.sdpo_runtime.teacher_regularization={teacher_regularization}\n"
        "trainer.sdpo_runtime.teacher_update_rate=0.05\n"
        f"git_commit={git_commit}\n"
        f"git_branch={git_branch}\n"
        f"git_diff_sha256={hashlib.sha256(b'diff').hexdigest()}\n"
        f"git_cached_diff_sha256={hashlib.sha256(b'cached').hexdigest()}\n"
        f"git_untracked_manifest_sha256={manifest_sha}\n"
        "python_runner=uv run python\n"
        "rl_runner=uv run rl\n"
        "git_untracked_manifest_begin\n"
        f"{manifest_payload}"
        "git_untracked_manifest_end\n"
        "git_status_short_begin\n"
        " M src/prime_rl/orchestrator/algo/sdpo.py\n"
        "git_status_short_end\n"
    ).encode("utf-8")


def _sdpo_acceptance_token_export_report_line() -> str:
    return (
        "Verified SDPO token exports: "
        "files=1, records=2, sdpo_records=2, transported_rows=2, student_rows=4, paired_rows=2, "
        "matching_support_rows=2, distinct_teacher_logprob_rows=2, importance_ratio_rows=2, "
        "rollout_is_weight_rows=2, student_preflight_rows=2, temperature_rows=4, sample_id_records=2, "
        "stable_steps=1, steps=['step_1'], paired_steps=['step_1'], matching_support_steps=['step_1'], "
        "student_preflight_steps=['step_1'], matched_support_samples=1, matched_support_token_rows=2, "
        "distinct_teacher_logprob_token_rows=2, importance_ratio_token_rows=2, rollout_is_weight_token_rows=2\n"
    )


def _sdpo_acceptance_smoke_report_bytes(*, ema: bool = False) -> bytes:
    report = "Verified SDPO smoke provenance:\n" + _sdpo_acceptance_token_export_report_line()
    if ema:
        report += (
            "Verified SDPO EMA broadcasts: steps=1, role=sdpo_teacher, teacher_steps=1, "
            "matched_steps=['step_1'], matched_step_keys=['run_default:step_1']\n"
        )
    return report.encode("utf-8")


def _sdpo_acceptance_topk_token_ids(base: int) -> list[int]:
    return [base + idx for idx in range(100)]


def _sdpo_acceptance_topk_logprobs(offset: float) -> list[float]:
    return [-(10.0 + offset + idx) for idx in range(100)]


def _sdpo_acceptance_token_export_bytes() -> bytes:
    student_rows = [
        [0] * 100,
        _sdpo_acceptance_topk_token_ids(1000),
        _sdpo_acceptance_topk_token_ids(2000),
    ]
    preflight = _minimal_sdpo_smoke_record(
        preflight_only=True,
        sdpo_topk_token_ids=[None, None, None],
        sdpo_topk_logprobs=[None, None, None],
        sdpo_student_topk_token_ids=student_rows,
        sdpo_student_topk_logprobs=[
            [0.0] * 100,
            _sdpo_acceptance_topk_logprobs(0.0),
            _sdpo_acceptance_topk_logprobs(0.5),
        ],
    )
    final = _minimal_sdpo_smoke_record(
        sdpo_topk_token_ids=student_rows,
        sdpo_topk_logprobs=[
            [0.0] * 100,
            _sdpo_acceptance_topk_logprobs(1.0),
            _sdpo_acceptance_topk_logprobs(1.5),
        ],
        sdpo_student_topk_token_ids=student_rows,
        sdpo_student_topk_logprobs=[
            [0.0] * 100,
            _sdpo_acceptance_topk_logprobs(0.0),
            _sdpo_acceptance_topk_logprobs(0.5),
        ],
    )
    return f"{json.dumps(preflight)}\n{json.dumps(final)}\n".encode("utf-8")


def _minimal_sdpo_acceptance_archive_files(
    *,
    summary_mode: str = "no-run",
    summary_archive_path: str = "outputs/sdpo-cuda-acceptance-proof.tar.gz",
    summary_git_commit: str = SDPO_TEST_GIT_COMMIT,
    summary_git_branch: str = SDPO_TEST_GIT_BRANCH,
    include_live_provenance: bool = True,
) -> dict[str, bytes]:
    files = {
        "sdpo_cuda_acceptance_summary.txt": (
            "sdpo_cuda_acceptance_summary_version=1\n"
            f"acceptance_mode={summary_mode}\n"
            "output_root=outputs/sdpo-cuda-acceptance\n"
            "live_output_dir=outputs/sdpo-cuda-acceptance/live\n"
            "live_config=configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml\n"
            "live_provenance_file=outputs/sdpo-cuda-acceptance/live/sdpo_smoke_provenance.txt\n"
            "live_verify_report_file=outputs/sdpo-cuda-acceptance/live/sdpo_smoke_verify_report.txt\n"
            "live_token_exports_dir=outputs/sdpo-cuda-acceptance/live/run_default/token_exports\n"
            "ema_output_dir=outputs/sdpo-cuda-acceptance/ema\n"
            "ema_config=configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml\n"
            "ema_provenance_file=outputs/sdpo-cuda-acceptance/ema/sdpo_smoke_provenance.txt\n"
            "ema_verify_report_file=outputs/sdpo-cuda-acceptance/ema/sdpo_smoke_verify_report.txt\n"
            "ema_token_exports_dir=outputs/sdpo-cuda-acceptance/ema/run_default/token_exports\n"
            "ema_broadcasts_dir=outputs/sdpo-cuda-acceptance/ema/run_default/broadcasts\n"
            "acceptance_manifest_file=outputs/sdpo-cuda-acceptance/sdpo_cuda_acceptance_manifest.txt\n"
            "expected_topk=100\n"
            f"archive_path={summary_archive_path}\n"
            f"git_commit={summary_git_commit}\n"
            f"git_branch={summary_git_branch}\n"
        ).encode("utf-8"),
        "live/sdpo_smoke_verify_report.txt": _sdpo_acceptance_smoke_report_bytes(),
        "ema/sdpo_smoke_provenance.txt": _sdpo_acceptance_provenance_bytes(
            mode="ema",
            config="configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml",
        ),
        "ema/sdpo_smoke_verify_report.txt": _sdpo_acceptance_smoke_report_bytes(ema=True),
        "live/run_default/token_exports/step_1/rank_0.jsonl": _sdpo_acceptance_token_export_bytes(),
        "live/run_default/token_exports/step_1/STABLE": b"",
        "ema/run_default/token_exports/step_1/rank_0.jsonl": _sdpo_acceptance_token_export_bytes(),
        "ema/run_default/token_exports/step_1/STABLE": b"",
        "ema/run_default/broadcasts/step_1/STABLE": b"",
        "ema/run_default/broadcasts/step_1/sdpo_teacher/model.bin": b"teacher\n",
        "ema/run_default/broadcasts/step_1/sdpo_teacher/STABLE": b"",
    }
    if include_live_provenance:
        files["live/sdpo_smoke_provenance.txt"] = _sdpo_acceptance_provenance_bytes(
            mode="live",
            config="configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml",
        )
    return files


def _add_sdpo_acceptance_manifest(
    files: dict[str, bytes],
    *,
    manifest_mode: str = "no-run",
    rows: list[str] | None = None,
) -> dict[str, bytes]:
    manifest_lines = [
        "sdpo_cuda_acceptance_manifest_version=1",
        f"acceptance_mode={manifest_mode}",
        "format=sha256 size_bytes relative_path",
    ]
    if rows is None:
        for name, data in files.items():
            manifest_lines.append(f"{hashlib.sha256(data).hexdigest()} {len(data)} {name}")
    else:
        manifest_lines.extend(rows)
    files["sdpo_cuda_acceptance_manifest.txt"] = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    return files


def _write_sdpo_acceptance_tar(
    archive_path: Path,
    files: dict[str, bytes],
    *,
    extra_members: list[tuple[tarfile.TarInfo, bytes | None]] | None = None,
) -> None:
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        for info, data in extra_members or []:
            archive.addfile(info, None if data is None else io.BytesIO(data))


def _minimal_sdpo_smoke_record(**overrides) -> dict:
    record = {
        "schema_version": 2,
        "sample_id": "sample-a",
        "env_name": "sdpo_env",
        "token_ids": [10, 11, 12],
        "position_ids": [0, 1, 2],
        "loss_mask": [False, True, True],
        "temperatures": [1.0, 0.75, 0.5],
        "trainer_logprobs": [None, -1.0, -1.5],
        "inference_logprobs": [None, -1.25, -1.0],
        "log_importance_ratio": [None, 0.25, -0.5],
        "importance_ratio": [None, 1.2840254, 0.6065307],
        "prob_delta": [None, 0.08137433, -0.14474928],
        "preflight_only": False,
        "sdpo_weights": [0.0, 1.0, 1.0],
        "sdpo_rollout_is_weights": [0.0, 1.2840254, 0.6065307],
        "sdpo_topk_token_ids": [[0, 0], [111, 112], [211, 212]],
        "sdpo_topk_logprobs": [[0.0, 0.0], [-3.0, -13.0], [-4.0, -14.0]],
        "sdpo_student_topk_token_ids": [[0, 0], [111, 112], [211, 212]],
        "sdpo_student_topk_logprobs": [[0.0, 0.0], [-0.75, -1.25], [-0.625, -1.5]],
    }
    record.update(overrides)
    return record


def _minimal_sdpo_smoke_provenance(*, extra: str = "") -> str:
    return (
        "sdpo_smoke_provenance_version=1\n"
        "mode=live\n"
        "config=configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml\n"
        "output_dir=outputs/sdpo-smoke\n"
        "expected_topk=2\n"
        "orchestrator.algo.distillation_topk=2\n"
        "orchestrator.algo.distillation_topk_support=student\n"
        "orchestrator.algo.teacher_regularization=live-policy\n"
        "orchestrator.algo.teacher_update_rate=0.05\n"
        "orchestrator.algo.success_reward_threshold=0.5\n"
        "orchestrator.algo.successful_demonstration_selection=batch_order\n"
        "orchestrator.algo.dont_reprompt_on_self_success=True\n"
        "orchestrator.algo.remove_thinking_from_demonstration=True\n"
        "orchestrator.algo.include_environment_feedback=True\n"
        "orchestrator.algo.environment_feedback_only_without_solution=True\n"
        "orchestrator.algo.max_reprompt_len=10240\n"
        "orchestrator.algo.reprompt_truncation=right\n"
        "orchestrator.algo.assistant_prefix=\n"
        "orchestrator.algo.multi_turn=False\n"
        "orchestrator.algo.template_target=first_user\n"
        "trainer.sdpo_loss.full_logit_distillation=True\n"
        "trainer.sdpo_loss.distillation_topk=2\n"
        "trainer.sdpo_loss.distillation_add_tail=True\n"
        "trainer.sdpo_loss.alpha=0.5\n"
        "trainer.sdpo_loss.is_clip=2.0\n"
        "trainer.sdpo_loss.rollout_is=token\n"
        "trainer.sdpo_loss.rollout_is_threshold=2.0\n"
        "trainer.sdpo_loss.rollout_is_batch_normalize=False\n"
        "trainer.sdpo_runtime.teacher_regularization=live-policy\n"
        "trainer.sdpo_runtime.teacher_update_rate=0.05\n"
        f"git_commit={SDPO_TEST_GIT_COMMIT}\n"
        f"git_branch={SDPO_TEST_GIT_BRANCH}\n"
        f"git_diff_sha256={hashlib.sha256(b'diff').hexdigest()}\n"
        f"git_cached_diff_sha256={hashlib.sha256(b'cached').hexdigest()}\n"
        f"git_untracked_manifest_sha256={hashlib.sha256(b'').hexdigest()}\n"
        "python_runner=uv run python\n"
        "rl_runner=uv run rl\n"
        "git_untracked_manifest_begin\n"
        "git_untracked_manifest_end\n"
        "git_status_short_begin\n"
        " M src/prime_rl/orchestrator/algo/sdpo.py\n"
        "git_status_short_end\n"
        f"{extra}"
    )


def _write_minimal_sdpo_smoke_artifacts(output_dir: Path, *, provenance_extra: str = "") -> None:
    export_dir = output_dir / "run_default" / "token_exports" / "step_1"
    export_dir.mkdir(parents=True)
    preflight = _minimal_sdpo_smoke_record(
        preflight_only=True,
        sdpo_topk_token_ids=[None, None, None],
        sdpo_topk_logprobs=[None, None, None],
    )
    final = _minimal_sdpo_smoke_record()
    with (export_dir / "rank_0.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(preflight) + "\n")
        handle.write(json.dumps(final) + "\n")
    (export_dir / "STABLE").write_text("", encoding="utf-8")
    (output_dir / "sdpo_smoke_provenance.txt").write_text(
        _minimal_sdpo_smoke_provenance(extra=provenance_extra),
        encoding="utf-8",
    )


def _write_fake_sdpo_local_validation_pytest_runner(tmp_path: Path) -> Path:
    fake_runner = tmp_path / "fake_sdpo_local_validation_pytest.sh"
    fake_runner.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf 'LOCAL_PYTEST_PYTHONPATH=<%s>\\n' "${PYTHONPATH:-}"
printf 'LOCAL_PYTEST'
for arg in "$@"; do
  printf ' <%s>' "$arg"
done
printf '\\n'
""",
        encoding="utf-8",
    )
    fake_runner.chmod(0o755)
    return fake_runner


def _write_fake_sdpo_local_validation_python_runner(tmp_path: Path) -> Path:
    fake_runner = tmp_path / "fake_sdpo_local_validation_python.sh"
    fake_runner.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf 'LOCAL_PYTHON_PYTHONPATH=<%s>\\n' "${PYTHONPATH:-}"
printf 'LOCAL_PYTHON'
for arg in "$@"; do
  printf ' <%s>' "$arg"
done
printf '\\n'
""",
        encoding="utf-8",
    )
    fake_runner.chmod(0o755)
    return fake_runner


def _write_fake_sdpo_local_validation_ruff_runner(tmp_path: Path) -> Path:
    fake_runner = tmp_path / "fake_sdpo_local_validation_ruff.sh"
    fake_runner.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf 'LOCAL_RUFF'
for arg in "$@"; do
  printf ' <%s>' "$arg"
done
printf '\\n'
""",
        encoding="utf-8",
    )
    fake_runner.chmod(0o755)
    return fake_runner


def _write_fake_sdpo_local_validation_smoke_runner(tmp_path: Path) -> Path:
    fake_runner = tmp_path / "fake_sdpo_local_validation_smoke.sh"
    fake_runner.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf 'LOCAL_SMOKE_PYTHON_RUNNER=<%s>\\n' "${SDPO_SMOKE_PYTHON_RUNNER:-}"
printf 'LOCAL_SMOKE_PYTHONPATH=<%s>\\n' "${PYTHONPATH:-}"
printf 'LOCAL_SMOKE'
for arg in "$@"; do
  printf ' <%s>' "$arg"
done
printf '\\n'
""",
        encoding="utf-8",
    )
    fake_runner.chmod(0o755)
    return fake_runner


def _write_fake_sdpo_local_validation_acceptance_runner(tmp_path: Path) -> Path:
    fake_runner = tmp_path / "fake_sdpo_local_validation_acceptance.sh"
    fake_runner.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf 'LOCAL_ACCEPTANCE_PYTHON_RUNNER=<%s>\\n' "${SDPO_SMOKE_PYTHON_RUNNER:-}"
printf 'LOCAL_ACCEPTANCE_PYTHONPATH=<%s>\\n' "${PYTHONPATH:-}"
printf 'LOCAL_ACCEPTANCE'
for arg in "$@"; do
  printf ' <%s>' "$arg"
done
printf '\\n'
""",
        encoding="utf-8",
    )
    fake_runner.chmod(0o755)
    return fake_runner


def test_sdpo_local_validation_script_help_documents_mac_gate():
    result = subprocess.run(["bash", str(LOCAL_VALIDATION_SCRIPT), "--help"], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "broad local SDPO validation gate" in result.stdout
    assert "--skip-hygiene" in result.stdout
    assert "Linux-only lockfile" in result.stdout


def test_sdpo_local_validation_script_runs_broad_gate_with_verifier_env_paths(tmp_path):
    fake_runner = _write_fake_sdpo_local_validation_pytest_runner(tmp_path)
    env = {**os.environ, "SDPO_LOCAL_VALIDATION_PYTEST_RUNNER": str(fake_runner)}

    result = subprocess.run(
        ["bash", str(LOCAL_VALIDATION_SCRIPT), "--skip-hygiene"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Running broad local SDPO validation gate..." in result.stdout
    assert "LOCAL_PYTEST_PYTHONPATH=<" in result.stdout
    assert "src:packages/prime-rl-configs/src:deps/pydantic-config/src" in result.stdout
    assert "deps/verifiers/environments/gsm8k_v1" in result.stdout
    assert "tests/unit/orchestrator/test_algorithms.py" in result.stdout
    assert "tests/unit/orchestrator/test_sdpo_preflight.py" in result.stdout
    assert "tests/unit/train/rl/test_sdpo_loss.py" in result.stdout
    assert "tests/unit/transport" in result.stdout
    assert "<-q>" in result.stdout
    assert "Running SDPO script syntax checks..." not in result.stdout
    assert "SDPO local validation gate passed." in result.stdout


def test_sdpo_local_validation_script_uses_configured_python_for_hygiene(tmp_path):
    fake_pytest = _write_fake_sdpo_local_validation_pytest_runner(tmp_path)
    fake_python = _write_fake_sdpo_local_validation_python_runner(tmp_path)
    fake_ruff = _write_fake_sdpo_local_validation_ruff_runner(tmp_path)
    fake_smoke = _write_fake_sdpo_local_validation_smoke_runner(tmp_path)
    fake_acceptance = _write_fake_sdpo_local_validation_acceptance_runner(tmp_path)
    env = {
        **os.environ,
        "SDPO_LOCAL_VALIDATION_PYTEST_RUNNER": str(fake_pytest),
        "SDPO_LOCAL_VALIDATION_PYTHON_RUNNER": str(fake_python),
        "SDPO_LOCAL_VALIDATION_RUFF_RUNNER": str(fake_ruff),
        "SDPO_LOCAL_VALIDATION_SMOKE_RUNNER": str(fake_smoke),
        "SDPO_LOCAL_VALIDATION_ACCEPTANCE_RUNNER": str(fake_acceptance),
    }

    result = subprocess.run(
        ["bash", str(LOCAL_VALIDATION_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Running SDPO script syntax checks..." in result.stdout
    assert "LOCAL_PYTHON_PYTHONPATH=<" in result.stdout
    assert "src:packages/prime-rl-configs/src:deps/pydantic-config/src" in result.stdout
    assert "LOCAL_PYTHON <-m> <py_compile>" in result.stdout
    assert "<scripts/verify_sdpo_smoke_artifacts.py>" in result.stdout
    assert "<scripts/verify_sdpo_token_exports.py>" in result.stdout
    assert "Running SDPO smoke config checks..." in result.stdout
    assert f"LOCAL_SMOKE_PYTHON_RUNNER=<{fake_python}>" in result.stdout
    assert "LOCAL_SMOKE_PYTHONPATH=<" in result.stdout
    assert "LOCAL_SMOKE <--check-config>" in result.stdout
    assert "LOCAL_SMOKE <--ema> <--check-config>" in result.stdout
    assert f"LOCAL_ACCEPTANCE_PYTHON_RUNNER=<{fake_python}>" in result.stdout
    assert "LOCAL_ACCEPTANCE_PYTHONPATH=<" in result.stdout
    assert "LOCAL_ACCEPTANCE <--check-config>" in result.stdout
    assert "Running SDPO Ruff checks..." in result.stdout
    assert "LOCAL_RUFF <check>" in result.stdout
    assert "LOCAL_RUFF <format> <--check>" in result.stdout
    assert "Running SDPO whitespace checks..." in result.stdout
    assert "<configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml>" in result.stdout
    assert "<configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml>" in result.stdout
    assert "<scripts/run_sdpo_local_validation.sh>" in result.stdout
    assert "python3 -m py_compile" not in result.stdout
    assert "SDPO local validation gate passed." in result.stdout


def test_sdpo_cuda_acceptance_script_check_config_runs_live_then_ema(tmp_path):
    fake_wrapper = _write_fake_sdpo_smoke_wrapper(tmp_path)
    fake_python = _write_fake_sdpo_acceptance_python_runner(tmp_path)
    env = {
        **os.environ,
        "SDPO_ACCEPTANCE_SMOKE_WRAPPER": str(fake_wrapper),
        "SDPO_ACCEPTANCE_PYTHON_RUNNER": str(fake_python),
    }

    result = subprocess.run(
        ["bash", str(ACCEPTANCE_SCRIPT), "--check-config"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Checking live-policy SDPO smoke config..." in result.stdout
    assert f"SMOKE_PYTHON_RUNNER=<{fake_python}>" in result.stdout
    assert "SMOKE <--check-config>" in result.stdout
    assert "Checking EMA SDPO smoke config..." in result.stdout
    assert "SMOKE <--ema> <--check-config>" in result.stdout
    assert "SDPO CUDA acceptance config checks passed." in result.stdout
    assert result.stdout.index("SMOKE <--check-config>") < result.stdout.index("SMOKE <--ema> <--check-config>")


def test_sdpo_cuda_acceptance_script_check_config_ignores_no_run_verification_guards(tmp_path):
    fake_wrapper = _write_fake_sdpo_smoke_wrapper(tmp_path)
    env = {**os.environ, "SDPO_ACCEPTANCE_SMOKE_WRAPPER": str(fake_wrapper)}

    result = subprocess.run(
        ["bash", str(ACCEPTANCE_SCRIPT), "--check-config", "--no-run", "--clean-output-dir"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Checking live-policy SDPO smoke config..." in result.stdout
    assert "SMOKE <--check-config>" in result.stdout
    assert "Checking EMA SDPO smoke config..." in result.stdout
    assert "SMOKE <--ema> <--check-config>" in result.stdout
    assert "SDPO CUDA acceptance config checks passed." in result.stdout
    assert "--no-run requires --output-root" not in result.stderr
    assert "--clean-output-dir cannot be combined with --no-run" not in result.stderr
    assert "--clean-output-dir" not in result.stdout


def test_sdpo_cuda_acceptance_script_runs_live_then_ema_with_fixed_dirs(tmp_path):
    fake_wrapper = _write_fake_sdpo_smoke_wrapper(tmp_path)
    fake_python = _write_fake_sdpo_acceptance_python_runner(tmp_path)
    output_root = tmp_path / "acceptance"
    env = {
        **os.environ,
        "SDPO_ACCEPTANCE_SMOKE_WRAPPER": str(fake_wrapper),
        "SDPO_ACCEPTANCE_PYTHON_RUNNER": str(fake_python),
    }

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_SCRIPT),
            "--output-root",
            str(output_root),
            "--clean-output-dir",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    live_verify = (
        "PYTHON <scripts/verify_sdpo_smoke_artifacts.py> "
        f"<{output_root / 'live'}> <--expected-topk> <100> <--require-provenance> "
        "<--expected-provenance-mode> <live> <--expected-provenance-config> "
        "<configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml>"
    )
    ema_verify = (
        "PYTHON <scripts/verify_sdpo_smoke_artifacts.py> "
        f"<{output_root / 'ema'}> <--expected-topk> <100> <--require-provenance> "
        "<--expected-provenance-mode> <ema> <--expected-provenance-config> "
        "<configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml> <--require-ema-teacher>"
    )
    assert "Running live-policy SDPO CUDA smoke..." in result.stdout
    assert f"SMOKE <--output-dir> <{output_root / 'live'}> <--clean-output-dir>" in result.stdout
    assert "Re-verifying completed live-policy SDPO CUDA smoke artifacts..." in result.stdout
    assert live_verify in result.stdout
    assert "Running EMA SDPO CUDA smoke..." in result.stdout
    assert f"SMOKE <--ema> <--output-dir> <{output_root / 'ema'}> <--clean-output-dir>" in result.stdout
    assert "Re-verifying completed EMA SDPO CUDA smoke artifacts..." in result.stdout
    assert ema_verify in result.stdout
    assert f"Wrote SDPO CUDA acceptance summary: {output_root / 'sdpo_cuda_acceptance_summary.txt'}" in result.stdout
    assert f"Wrote SDPO CUDA acceptance manifest: {output_root / 'sdpo_cuda_acceptance_manifest.txt'}" in result.stdout
    assert "SDPO CUDA acceptance complete." in result.stdout
    assert f"Live output: {output_root / 'live'}" in result.stdout
    assert f"EMA output: {output_root / 'ema'}" in result.stdout
    assert result.stdout.index("SMOKE <--output-dir>") < result.stdout.index("SMOKE <--ema> <--output-dir>")
    assert result.stdout.index(live_verify) < result.stdout.index("SMOKE <--ema> <--output-dir>")
    assert result.stdout.index("SMOKE <--ema> <--output-dir>") < result.stdout.index(ema_verify)
    summary = (output_root / "sdpo_cuda_acceptance_summary.txt").read_text(encoding="utf-8")
    assert "sdpo_cuda_acceptance_summary_version=1" in summary
    assert "acceptance_mode=training" in summary
    assert f"output_root={output_root}" in summary
    assert f"live_output_dir={output_root / 'live'}" in summary
    assert "live_config=configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml" in summary
    assert f"live_provenance_file={output_root / 'live' / 'sdpo_smoke_provenance.txt'}" in summary
    assert f"live_verify_report_file={output_root / 'live' / 'sdpo_smoke_verify_report.txt'}" in summary
    assert f"live_config_dir={output_root / 'live' / 'configs'}" in summary
    assert f"live_run_control_dir={output_root / 'live' / 'run_default' / 'control'}" in summary
    assert f"live_token_exports_dir={output_root / 'live' / 'run_default' / 'token_exports'}" in summary
    assert f"ema_output_dir={output_root / 'ema'}" in summary
    assert "ema_config=configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml" in summary
    assert f"ema_provenance_file={output_root / 'ema' / 'sdpo_smoke_provenance.txt'}" in summary
    assert f"ema_verify_report_file={output_root / 'ema' / 'sdpo_smoke_verify_report.txt'}" in summary
    assert f"ema_config_dir={output_root / 'ema' / 'configs'}" in summary
    assert f"ema_run_control_dir={output_root / 'ema' / 'run_default' / 'control'}" in summary
    assert f"ema_token_exports_dir={output_root / 'ema' / 'run_default' / 'token_exports'}" in summary
    assert f"ema_broadcasts_dir={output_root / 'ema' / 'run_default' / 'broadcasts'}" in summary
    assert f"acceptance_manifest_file={output_root / 'sdpo_cuda_acceptance_manifest.txt'}" in summary
    assert "archive_path=" in summary
    assert "expected_topk=100" in summary
    assert "git_commit=" in summary
    assert "git_branch=" in summary
    manifest = (output_root / "sdpo_cuda_acceptance_manifest.txt").read_text(encoding="utf-8")
    assert "sdpo_cuda_acceptance_manifest_version=1" in manifest
    assert "acceptance_mode=training" in manifest
    assert "format=sha256 size_bytes relative_path" in manifest
    assert " live/sdpo_smoke_verify_report.txt" in manifest
    assert " ema/sdpo_smoke_verify_report.txt" in manifest


def test_sdpo_cuda_acceptance_script_training_run_can_archive_verified_artifacts(tmp_path):
    fake_wrapper = _write_fake_sdpo_smoke_wrapper(tmp_path)
    fake_python = _write_fake_sdpo_acceptance_python_runner(tmp_path)
    output_root = tmp_path / "acceptance"
    archive_path = tmp_path / "archives" / "sdpo-training-acceptance.tar.gz"
    arg_log = tmp_path / "archive-verify-argv.log"
    env = {
        **os.environ,
        "SDPO_ACCEPTANCE_SMOKE_WRAPPER": str(fake_wrapper),
        "SDPO_ACCEPTANCE_PYTHON_RUNNER": str(fake_python),
        "SDPO_FAKE_ACCEPTANCE_ARG_LOG": str(arg_log),
        "SDPO_FAKE_SMOKE_FULL_PROVENANCE": "1",
    }

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_SCRIPT),
            "--output-root",
            str(output_root),
            "--clean-output-dir",
            "--archive",
            str(archive_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"Wrote SDPO CUDA acceptance summary: {output_root / 'sdpo_cuda_acceptance_summary.txt'}" in result.stdout
    assert f"Wrote SDPO CUDA acceptance manifest: {output_root / 'sdpo_cuda_acceptance_manifest.txt'}" in result.stdout
    assert f"Verifying SDPO CUDA acceptance archive: {archive_path}" in result.stdout
    assert "Verified SDPO CUDA acceptance archive:" in result.stdout
    assert "acceptance_mode=training" in result.stdout
    assert f"Wrote SDPO CUDA acceptance archive: {archive_path}" in result.stdout
    assert "SDPO CUDA acceptance complete." in result.stdout
    assert archive_path.is_file()
    summary = (output_root / "sdpo_cuda_acceptance_summary.txt").read_text(encoding="utf-8")
    assert "acceptance_mode=training" in summary
    assert f"archive_path={archive_path}" in summary
    manifest = (output_root / "sdpo_cuda_acceptance_manifest.txt").read_text(encoding="utf-8")
    assert "acceptance_mode=training" in manifest
    assert (
        "ARCHIVE_VERIFY <scripts/verify_sdpo_cuda_acceptance_archive.py> <--expected-acceptance-mode> <training>"
    ) in arg_log.read_text(encoding="utf-8")

    verify_result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT),
            "--expected-acceptance-mode",
            "training",
            str(archive_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert verify_result.returncode == 0, verify_result.stderr
    assert "Verified SDPO CUDA acceptance archive:" in verify_result.stdout
    assert "acceptance_mode=training" in verify_result.stdout
    assert "raw_artifacts=verified" in verify_result.stdout
    assert "live_matching_support_rows=2" in verify_result.stdout
    assert "ema_teacher_steps=1" in verify_result.stdout


def test_sdpo_cuda_acceptance_script_rejects_training_success_markers_without_proof_artifacts(tmp_path):
    fake_wrapper = _write_fake_sdpo_smoke_wrapper(tmp_path)
    fake_python = _write_fake_sdpo_acceptance_python_runner(tmp_path)
    output_root = tmp_path / "acceptance"
    env = {
        **os.environ,
        "SDPO_ACCEPTANCE_SMOKE_WRAPPER": str(fake_wrapper),
        "SDPO_ACCEPTANCE_PYTHON_RUNNER": str(fake_python),
        "SDPO_FAKE_SMOKE_SKIP_PROOF_ARTIFACTS": "1",
    }

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_SCRIPT),
            "--output-root",
            str(output_root),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "cannot summarize SDPO CUDA acceptance proof" in result.stderr
    assert str(output_root / "live" / "sdpo_smoke_provenance.txt") in result.stderr
    assert not (output_root / "sdpo_cuda_acceptance_summary.txt").exists()
    assert not (output_root / "sdpo_cuda_acceptance_manifest.txt").exists()


def test_sdpo_cuda_acceptance_script_rejects_empty_required_proof_directories(tmp_path):
    fake_wrapper = _write_fake_sdpo_smoke_wrapper(tmp_path)
    fake_python = _write_fake_sdpo_acceptance_python_runner(tmp_path)
    output_root = tmp_path / "acceptance"
    env = {
        **os.environ,
        "SDPO_ACCEPTANCE_SMOKE_WRAPPER": str(fake_wrapper),
        "SDPO_ACCEPTANCE_PYTHON_RUNNER": str(fake_python),
    }

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_SCRIPT),
            "--output-root",
            str(output_root),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    for file in (output_root / "live" / "run_default" / "token_exports").glob("**/*"):
        if file.is_file():
            file.unlink()

    verify_result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_SCRIPT),
            "--no-run",
            "--output-root",
            str(output_root),
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "SDPO_ACCEPTANCE_PYTHON_RUNNER": str(_write_fake_sdpo_acceptance_python_runner(tmp_path)),
        },
        capture_output=True,
        text=True,
    )

    assert verify_result.returncode == 2
    assert "cannot summarize SDPO CUDA acceptance proof" in verify_result.stderr
    assert "empty required artifact directory" in verify_result.stderr
    assert str(output_root / "live" / "run_default" / "token_exports") in verify_result.stderr


def test_sdpo_cuda_acceptance_script_no_run_requires_explicit_output_root():
    result = subprocess.run(
        ["bash", str(ACCEPTANCE_SCRIPT), "--no-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--no-run requires --output-root" in result.stderr
    assert "uv run" not in result.stderr


def test_sdpo_cuda_acceptance_script_rejects_clean_output_dir_with_no_run(tmp_path):
    output_root = tmp_path / "acceptance"

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_SCRIPT),
            "--no-run",
            "--output-root",
            str(output_root),
            "--clean-output-dir",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--clean-output-dir cannot be combined with --no-run" in result.stderr
    assert "uv run" not in result.stderr


def test_sdpo_cuda_acceptance_script_no_run_strictly_verifies_live_then_ema_artifacts(tmp_path):
    fake_wrapper = _write_fake_sdpo_smoke_wrapper(tmp_path)
    fake_python = _write_fake_sdpo_acceptance_python_runner(tmp_path)
    output_root = tmp_path / "acceptance"
    _write_minimal_sdpo_acceptance_artifacts(output_root)
    env = {
        **os.environ,
        "SDPO_ACCEPTANCE_SMOKE_WRAPPER": str(fake_wrapper),
        "SDPO_ACCEPTANCE_PYTHON_RUNNER": str(fake_python),
    }

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_SCRIPT),
            "--no-run",
            "--output-root",
            str(output_root),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    live_verify = (
        "PYTHON <scripts/verify_sdpo_smoke_artifacts.py> "
        f"<{output_root / 'live'}> <--expected-topk> <100> <--require-provenance> "
        "<--expected-provenance-mode> <live> <--expected-provenance-config> "
        "<configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml>"
    )
    ema_verify = (
        "PYTHON <scripts/verify_sdpo_smoke_artifacts.py> "
        f"<{output_root / 'ema'}> <--expected-topk> <100> <--require-provenance> "
        "<--expected-provenance-mode> <ema> <--expected-provenance-config> "
        "<configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml> <--require-ema-teacher>"
    )

    assert result.returncode == 0, result.stderr
    assert "Verifying existing live-policy SDPO CUDA smoke artifacts..." in result.stdout
    assert live_verify in result.stdout
    assert "Verifying existing EMA SDPO CUDA smoke artifacts..." in result.stdout
    assert ema_verify in result.stdout
    assert f"Wrote SDPO smoke verifier report: {output_root / 'live' / 'sdpo_smoke_verify_report.txt'}" in result.stdout
    assert f"Wrote SDPO smoke verifier report: {output_root / 'ema' / 'sdpo_smoke_verify_report.txt'}" in result.stdout
    assert f"Wrote SDPO CUDA acceptance summary: {output_root / 'sdpo_cuda_acceptance_summary.txt'}" in result.stdout
    assert f"Wrote SDPO CUDA acceptance manifest: {output_root / 'sdpo_cuda_acceptance_manifest.txt'}" in result.stdout
    assert "SDPO CUDA acceptance artifact verification passed." in result.stdout
    assert "SMOKE" not in result.stdout
    assert result.stdout.index(live_verify) < result.stdout.index(ema_verify)
    assert live_verify in (output_root / "live" / "sdpo_smoke_verify_report.txt").read_text(encoding="utf-8")
    assert ema_verify in (output_root / "ema" / "sdpo_smoke_verify_report.txt").read_text(encoding="utf-8")
    summary = (output_root / "sdpo_cuda_acceptance_summary.txt").read_text(encoding="utf-8")
    assert "sdpo_cuda_acceptance_summary_version=1" in summary
    assert "acceptance_mode=no-run" in summary
    assert f"output_root={output_root}" in summary
    assert f"live_output_dir={output_root / 'live'}" in summary
    assert f"ema_output_dir={output_root / 'ema'}" in summary
    assert f"live_provenance_file={output_root / 'live' / 'sdpo_smoke_provenance.txt'}" in summary
    assert f"ema_provenance_file={output_root / 'ema' / 'sdpo_smoke_provenance.txt'}" in summary
    assert f"live_verify_report_file={output_root / 'live' / 'sdpo_smoke_verify_report.txt'}" in summary
    assert f"ema_verify_report_file={output_root / 'ema' / 'sdpo_smoke_verify_report.txt'}" in summary
    assert f"live_config_dir={output_root / 'live' / 'configs'}" in summary
    assert f"ema_config_dir={output_root / 'ema' / 'configs'}" in summary
    assert f"live_run_control_dir={output_root / 'live' / 'run_default' / 'control'}" in summary
    assert f"ema_run_control_dir={output_root / 'ema' / 'run_default' / 'control'}" in summary
    assert f"live_token_exports_dir={output_root / 'live' / 'run_default' / 'token_exports'}" in summary
    assert f"ema_token_exports_dir={output_root / 'ema' / 'run_default' / 'token_exports'}" in summary
    assert f"ema_broadcasts_dir={output_root / 'ema' / 'run_default' / 'broadcasts'}" in summary
    assert f"acceptance_manifest_file={output_root / 'sdpo_cuda_acceptance_manifest.txt'}" in summary
    assert "archive_path=" in summary
    assert "expected_topk=100" in summary
    manifest = (output_root / "sdpo_cuda_acceptance_manifest.txt").read_text(encoding="utf-8")
    assert "sdpo_cuda_acceptance_manifest_version=1" in manifest
    assert "acceptance_mode=no-run" in manifest
    assert " live/sdpo_smoke_verify_report.txt" in manifest
    assert " ema/sdpo_smoke_verify_report.txt" in manifest


def test_sdpo_cuda_acceptance_script_rejects_placeholder_summary_git_identity(tmp_path):
    fake_python = _write_fake_sdpo_acceptance_python_runner(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    fake_git.chmod(0o755)
    output_root = tmp_path / "acceptance"
    _write_minimal_sdpo_acceptance_artifacts(output_root)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "SDPO_ACCEPTANCE_PYTHON_RUNNER": str(fake_python),
    }

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_SCRIPT),
            "--no-run",
            "--output-root",
            str(output_root),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "cannot write SDPO CUDA acceptance summary; git_commit must not be 'unknown'" in result.stderr
    assert not (output_root / "sdpo_cuda_acceptance_summary.txt").exists()
    assert not (output_root / "sdpo_cuda_acceptance_manifest.txt").exists()


def test_sdpo_cuda_acceptance_script_rejects_verify_report_without_success_markers(tmp_path):
    fake_python = _write_fake_sdpo_acceptance_python_runner(tmp_path)
    output_root = tmp_path / "acceptance"
    env = {
        **os.environ,
        "SDPO_ACCEPTANCE_PYTHON_RUNNER": str(fake_python),
        "SDPO_FAKE_ACCEPTANCE_SKIP_VERIFY_MARKERS": "1",
    }

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_SCRIPT),
            "--no-run",
            "--output-root",
            str(output_root),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "SDPO smoke verifier report for live-policy is missing success marker" in result.stderr
    assert "Verified SDPO smoke provenance:" in result.stderr
    assert not (output_root / "sdpo_cuda_acceptance_summary.txt").exists()


def test_sdpo_cuda_acceptance_script_rejects_duplicate_verify_report_success_marker(tmp_path):
    fake_python = _write_fake_sdpo_acceptance_python_runner(tmp_path)
    output_root = tmp_path / "acceptance"
    _write_minimal_sdpo_acceptance_artifacts(output_root)
    env = {
        **os.environ,
        "SDPO_ACCEPTANCE_PYTHON_RUNNER": str(fake_python),
        "SDPO_FAKE_ACCEPTANCE_DUPLICATE_TOKEN_MARKER": "1",
    }

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_SCRIPT),
            "--no-run",
            "--output-root",
            str(output_root),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "SDPO smoke verifier report for live-policy repeats success marker" in result.stderr
    assert "Verified SDPO token exports:" in result.stderr
    assert not (output_root / "sdpo_cuda_acceptance_summary.txt").exists()
    assert not (output_root / "sdpo_cuda_acceptance_manifest.txt").exists()


def test_sdpo_cuda_acceptance_script_no_run_can_archive_verified_artifacts(tmp_path):
    fake_python = _write_fake_sdpo_acceptance_python_runner(tmp_path)
    output_root = tmp_path / "acceptance"
    archive_path = tmp_path / "archives" / "sdpo-acceptance.tar.gz"
    arg_log = tmp_path / "archive-verify-argv.log"
    _write_minimal_sdpo_acceptance_artifacts(output_root)
    for mode in ("live", "ema"):
        (output_root / mode / "configs").mkdir(parents=True)
        (output_root / mode / "configs" / "rl.toml").write_text(f"# {mode} resolved rl config\n", encoding="utf-8")
        (output_root / mode / "run_default" / "control").mkdir(parents=True)
        (output_root / mode / "run_default" / "control" / "orch.toml").write_text(
            f"# {mode} run control\n",
            encoding="utf-8",
        )
    env = {
        **os.environ,
        "SDPO_ACCEPTANCE_PYTHON_RUNNER": str(fake_python),
        "SDPO_FAKE_ACCEPTANCE_ARG_LOG": str(arg_log),
    }

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_SCRIPT),
            "--no-run",
            "--output-root",
            str(output_root),
            "--archive",
            str(archive_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"Wrote SDPO CUDA acceptance manifest: {output_root / 'sdpo_cuda_acceptance_manifest.txt'}" in result.stdout
    assert f"Verifying SDPO CUDA acceptance archive: {archive_path}" in result.stdout
    assert f"Wrote SDPO CUDA acceptance archive: {archive_path}" in result.stdout
    assert result.stdout.index(f"Verifying SDPO CUDA acceptance archive: {archive_path}") < result.stdout.index(
        f"Wrote SDPO CUDA acceptance archive: {archive_path}"
    )
    assert "Verified SDPO CUDA acceptance archive:" in result.stdout
    assert "acceptance_mode=no-run" in result.stdout
    assert (
        "ARCHIVE_VERIFY <scripts/verify_sdpo_cuda_acceptance_archive.py> <--expected-acceptance-mode> <no-run>"
    ) in arg_log.read_text(encoding="utf-8")
    assert archive_path.is_file()
    summary = (output_root / "sdpo_cuda_acceptance_summary.txt").read_text(encoding="utf-8")
    assert f"archive_path={archive_path}" in summary
    manifest = (output_root / "sdpo_cuda_acceptance_manifest.txt").read_text(encoding="utf-8")
    assert "sdpo_cuda_acceptance_manifest_version=1" in manifest
    assert " live/run_default/token_exports/step_1/rank_0.jsonl" in manifest
    assert " ema/run_default/broadcasts/step_1/STABLE" in manifest
    assert " ema/run_default/broadcasts/step_1/sdpo_teacher/model.bin" in manifest
    with tarfile.open(archive_path) as archive:
        names = set(archive.getnames())
    assert "sdpo_cuda_acceptance_summary.txt" in names
    assert "sdpo_cuda_acceptance_manifest.txt" in names
    assert "live/sdpo_smoke_provenance.txt" in names
    assert "live/sdpo_smoke_verify_report.txt" in names
    assert "ema/sdpo_smoke_provenance.txt" in names
    assert "ema/sdpo_smoke_verify_report.txt" in names
    assert "live/run_default/token_exports" in names
    assert "ema/run_default/token_exports" in names
    assert "ema/run_default/broadcasts" in names
    assert "live/configs" in names
    assert "live/configs/rl.toml" in names
    assert "ema/configs" in names
    assert "ema/configs/rl.toml" in names
    assert "live/run_default/control" in names
    assert "live/run_default/control/orch.toml" in names
    assert "ema/run_default/control" in names
    assert "ema/run_default/control/orch.toml" in names
    verify_result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert verify_result.returncode == 0, verify_result.stderr
    assert "Verified SDPO CUDA acceptance archive:" in verify_result.stdout
    assert "acceptance_mode=no-run" in verify_result.stdout
    assert "raw_artifacts=verified" in verify_result.stdout
    assert "ema_rollout_is_weight_rows=2" in verify_result.stdout
    assert "ema_teacher_steps=1" in verify_result.stdout


def test_sdpo_cuda_acceptance_script_rejects_archive_inside_output_root(tmp_path):
    output_root = tmp_path / "acceptance"
    archive_path = output_root / "proof.tar.gz"

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_SCRIPT),
            "--no-run",
            "--output-root",
            str(output_root),
            "--archive",
            str(archive_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--archive must be outside --output-root" in result.stderr
    assert str(archive_path) in result.stderr
    assert str(output_root) in result.stderr
    assert not archive_path.exists()


def test_sdpo_cuda_acceptance_script_rejects_archive_inside_symlinked_output_root(tmp_path):
    real_output_root = tmp_path / "real-acceptance"
    output_root = tmp_path / "linked-acceptance"
    real_output_root.mkdir()
    output_root.symlink_to(real_output_root, target_is_directory=True)
    archive_path = real_output_root / "proof.tar.gz"

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_SCRIPT),
            "--no-run",
            "--output-root",
            str(output_root),
            "--archive",
            str(archive_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--archive must be outside --output-root" in result.stderr
    assert str(archive_path) in result.stderr
    assert str(output_root) in result.stderr
    assert not archive_path.exists()
    assert not (real_output_root / "sdpo_cuda_acceptance_summary.txt").exists()


def test_sdpo_cuda_acceptance_script_rejects_non_file_archive_path_before_verification(tmp_path):
    output_root = tmp_path / "acceptance"
    archive_path = tmp_path / "archives" / "sdpo-acceptance.tar.gz"
    archive_path.mkdir(parents=True)

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_SCRIPT),
            "--no-run",
            "--output-root",
            str(output_root),
            "--archive",
            str(archive_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--archive path exists and is not a regular file" in result.stderr
    assert str(archive_path) in result.stderr
    assert not (output_root / "sdpo_cuda_acceptance_summary.txt").exists()
    assert not (output_root / "sdpo_cuda_acceptance_manifest.txt").exists()


def test_sdpo_cuda_acceptance_script_rejects_broken_symlink_archive_path_before_verification(tmp_path):
    output_root = tmp_path / "acceptance"
    archive_path = tmp_path / "archives" / "sdpo-acceptance.tar.gz"
    archive_path.parent.mkdir(parents=True)
    archive_path.symlink_to(tmp_path / "missing-target.tar.gz")

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_SCRIPT),
            "--no-run",
            "--output-root",
            str(output_root),
            "--archive",
            str(archive_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--archive path exists and is not a regular file" in result.stderr
    assert str(archive_path) in result.stderr
    assert not (output_root / "sdpo_cuda_acceptance_summary.txt").exists()
    assert not (output_root / "sdpo_cuda_acceptance_manifest.txt").exists()


def test_sdpo_cuda_acceptance_archive_verifier_accepts_expected_training_mode(tmp_path):
    archive_path = tmp_path / "sdpo-acceptance-training.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files(summary_mode="training")
    files = _add_sdpo_acceptance_manifest(files, manifest_mode="training")
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT),
            "--expected-acceptance-mode",
            "training",
            str(archive_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Verified SDPO CUDA acceptance archive:" in result.stdout
    assert "acceptance_mode=training" in result.stdout
    assert "raw_artifacts=verified" in result.stdout


def test_sdpo_cuda_acceptance_archive_verifier_rejects_unexpected_acceptance_mode(tmp_path):
    archive_path = tmp_path / "sdpo-acceptance-no-run.tar.gz"
    files = _add_sdpo_acceptance_manifest(_minimal_sdpo_acceptance_archive_files())
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT),
            "--expected-acceptance-mode",
            "training",
            str(archive_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "archive acceptance_mode mismatch: expected 'training', got 'no-run'" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_marker_only_smoke_report(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-marker-only-report.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    files["live/sdpo_smoke_verify_report.txt"] = b"Verified SDPO smoke provenance:\nVerified SDPO token exports:\n"
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "live/sdpo_smoke_verify_report.txt Verified SDPO token exports: report is missing counter" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rechecks_archived_raw_token_exports(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-raw-token-exports.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    files["live/run_default/token_exports/step_1/rank_0.jsonl"] = b"{}\n"
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "archived raw SDPO artifacts failed verification" in result.stderr
    assert "found no records with nonzero sdpo_weights" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_report_counter_mismatching_raw_artifacts(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-report-counter-mismatch.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    files["live/sdpo_smoke_verify_report.txt"] = files["live/sdpo_smoke_verify_report.txt"].replace(
        b"sdpo_records=2, transported_rows=2",
        b"sdpo_records=3, transported_rows=2",
        1,
    )
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "live/sdpo_smoke_verify_report.txt Verified SDPO token exports:" in result.stderr
    assert "counter sdpo_records must match archived raw artifact value 2, got 3" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_zero_ema_broadcast_evidence(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-zero-ema-report.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    files["ema/sdpo_smoke_verify_report.txt"] = (
        "Verified SDPO smoke provenance:\n"
        f"{_sdpo_acceptance_token_export_report_line()}"
        "Verified SDPO EMA broadcasts: steps=1, role=sdpo_teacher, teacher_steps=0, "
        "matched_steps=[], matched_step_keys=[]\n"
    ).encode("utf-8")
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "ema/sdpo_smoke_verify_report.txt Verified SDPO EMA broadcasts: counter teacher_steps" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_duplicate_report_counter(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-duplicate-report-counter.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    files["live/sdpo_smoke_verify_report.txt"] = files["live/sdpo_smoke_verify_report.txt"].replace(
        b"sdpo_records=2, transported_rows=2",
        b"sdpo_records=2, sdpo_records=3, transported_rows=2",
        1,
    )
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "live/sdpo_smoke_verify_report.txt Verified SDPO token exports: report repeats field: sdpo_records" in (
        result.stderr
    )


def test_sdpo_cuda_acceptance_archive_verifier_rejects_duplicate_report_marker(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-duplicate-report-marker.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    files["ema/sdpo_smoke_verify_report.txt"] += b"Verified SDPO EMA broadcasts: steps=1, teacher_steps=1\n"
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "ema/sdpo_smoke_verify_report.txt repeats success marker: Verified SDPO EMA broadcasts:" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_duplicate_summary_field(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-duplicate-summary.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    files["sdpo_cuda_acceptance_summary.txt"] += b"git_commit=overwritten\n"
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "summary repeats field: git_commit" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_duplicate_manifest_field(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-duplicate-manifest.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    files = _add_sdpo_acceptance_manifest(files)
    files["sdpo_cuda_acceptance_manifest.txt"] = files["sdpo_cuda_acceptance_manifest.txt"].replace(
        b"format=sha256 size_bytes relative_path\n",
        b"acceptance_mode=training\nformat=sha256 size_bytes relative_path\n",
        1,
    )
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "manifest repeats field: acceptance_mode" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_duplicate_provenance_field(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-duplicate-provenance-field.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    files["ema/sdpo_smoke_provenance.txt"] = files["ema/sdpo_smoke_provenance.txt"].replace(
        b"git_branch=codex/sdpo-test\n",
        b"git_branch=codex/sdpo-test\ngit_branch=codex/overwritten\n",
        1,
    )
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "ema/sdpo_smoke_provenance.txt repeats provenance field: git_branch" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_repeated_provenance_section_marker(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-duplicate-provenance-section.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    files["live/sdpo_smoke_provenance.txt"] += b"git_status_short_begin\ngit_status_short_end\n"
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "live/sdpo_smoke_provenance.txt repeats provenance field: git_status_short_begin" in result.stderr


def test_sdpo_cuda_acceptance_archive_fails_with_clear_missing_artifact(tmp_path):
    fake_python = _write_fake_sdpo_acceptance_python_runner(tmp_path)
    output_root = tmp_path / "acceptance"
    archive_path = tmp_path / "archives" / "sdpo-acceptance.tar.gz"
    for mode in ("live", "ema"):
        run_dir = output_root / mode / "run_default"
        (run_dir / "token_exports" / "step_1").mkdir(parents=True)
        (run_dir / "token_exports" / "step_1" / "rank_0.jsonl").write_text("{}\n", encoding="utf-8")
        (output_root / mode / "sdpo_smoke_provenance.txt").write_text(
            f"sdpo_smoke_provenance_version=1\nmode={mode}\nexpected_topk=100\n",
            encoding="utf-8",
        )
    env = {**os.environ, "SDPO_ACCEPTANCE_PYTHON_RUNNER": str(fake_python)}

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_SCRIPT),
            "--no-run",
            "--output-root",
            str(output_root),
            "--archive",
            str(archive_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "cannot summarize SDPO CUDA acceptance proof" in result.stderr
    assert str(output_root / "ema" / "run_default" / "broadcasts") in result.stderr
    assert not archive_path.exists()


def test_sdpo_cuda_acceptance_archive_rejects_empty_tarball_after_successful_tar_exit(tmp_path):
    fake_python = _write_fake_sdpo_acceptance_python_runner(tmp_path)
    fake_tar = tmp_path / "tar"
    fake_tar.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${COPYFILE_DISABLE:-}" != "1" ]]; then
  echo "missing COPYFILE_DISABLE=1" >&2
  exit 3
fi
archive=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-czf" ]]; then
    archive="$2"
    shift 2
    continue
  fi
  shift
done
: > "$archive"
exit 0
""",
        encoding="utf-8",
    )
    fake_tar.chmod(0o755)
    output_root = tmp_path / "acceptance"
    archive_path = tmp_path / "archives" / "sdpo-acceptance.tar.gz"
    _write_minimal_sdpo_acceptance_artifacts(output_root)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SDPO_ACCEPTANCE_PYTHON_RUNNER": str(fake_python),
    }

    result = subprocess.run(
        [
            "bash",
            str(ACCEPTANCE_SCRIPT),
            "--no-run",
            "--output-root",
            str(output_root),
            "--archive",
            str(archive_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "SDPO CUDA acceptance archive is empty after writing" in result.stderr
    assert str(archive_path) in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_manifest_hash_mismatch(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    live_provenance_size = len(files["live/sdpo_smoke_provenance.txt"])
    files = _add_sdpo_acceptance_manifest(
        files,
        rows=[
            "0000000000000000000000000000000000000000000000000000000000000000 "
            f"{live_provenance_size} live/sdpo_smoke_provenance.txt"
        ],
    )
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "manifest sha256 mismatch for live/sdpo_smoke_provenance.txt" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_summary_manifest_mode_mismatch(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-mode.tar.gz"
    files = _add_sdpo_acceptance_manifest(_minimal_sdpo_acceptance_archive_files(summary_mode="training"))
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "summary/manifest acceptance_mode mismatch" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_placeholder_summary_git_identity(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-summary-git.tar.gz"
    files = _add_sdpo_acceptance_manifest(_minimal_sdpo_acceptance_archive_files(summary_git_commit="unknown"))
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "summary field git_commit must not be 'unknown'" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_placeholder_smoke_provenance(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-smoke-provenance.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    files["live/sdpo_smoke_provenance.txt"] = b"ok"
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "live/sdpo_smoke_provenance.txt line 1 is malformed" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_smoke_provenance_commit_mismatch(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-smoke-provenance-commit.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    files["ema/sdpo_smoke_provenance.txt"] = _sdpo_acceptance_provenance_bytes(
        mode="ema",
        config="configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml",
        git_commit="1" * 40,
    )
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "ema/sdpo_smoke_provenance.txt git_commit must match acceptance summary" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_live_ema_source_fingerprint_mismatch(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-smoke-provenance-diff.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    files["ema/sdpo_smoke_provenance.txt"] = files["ema/sdpo_smoke_provenance.txt"].replace(
        f"git_diff_sha256={hashlib.sha256(b'diff').hexdigest()}".encode("utf-8"),
        f"git_diff_sha256={hashlib.sha256(b'different-diff').hexdigest()}".encode("utf-8"),
    )
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "ema/sdpo_smoke_provenance.txt git_diff_sha256 must match live/sdpo_smoke_provenance.txt" in (result.stderr)


def test_sdpo_cuda_acceptance_archive_verifier_rejects_malformed_source_fingerprint(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-smoke-provenance-malformed-diff.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    files["live/sdpo_smoke_provenance.txt"] = files["live/sdpo_smoke_provenance.txt"].replace(
        f"git_diff_sha256={hashlib.sha256(b'diff').hexdigest()}".encode("utf-8"),
        b"git_diff_sha256=not-a-sha256",
    )
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "live/sdpo_smoke_provenance.txt field git_diff_sha256 must be a lowercase SHA-256 hex digest" in (
        result.stderr
    )


def test_sdpo_cuda_acceptance_archive_verifier_rejects_wrong_smoke_provenance_mode(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-smoke-provenance-mode.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    files["live/sdpo_smoke_provenance.txt"] = _sdpo_acceptance_provenance_bytes(
        mode="ema",
        config="configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml",
    )
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "live/sdpo_smoke_provenance.txt mismatch for mode" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_smoke_provenance_reference_knob_mismatch(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-smoke-provenance-reference-knob.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    files["live/sdpo_smoke_provenance.txt"] = files["live/sdpo_smoke_provenance.txt"].replace(
        b"orchestrator.algo.successful_demonstration_selection=batch_order",
        b"orchestrator.algo.successful_demonstration_selection=highest_reward",
    )
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "live/sdpo_smoke_provenance.txt mismatch for orchestrator.algo.successful_demonstration_selection" in (
        result.stderr
    )


def test_sdpo_cuda_acceptance_archive_verifier_rejects_smoke_provenance_manifest_hash_mismatch(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-smoke-provenance-manifest.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    provenance = files["ema/sdpo_smoke_provenance.txt"].decode("utf-8")
    files["ema/sdpo_smoke_provenance.txt"] = provenance.replace(
        "git_untracked_manifest_sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "git_untracked_manifest_sha256=0000000000000000000000000000000000000000000000000000000000000000",
    ).encode("utf-8")
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "ema/sdpo_smoke_provenance.txt mismatch for git_untracked_manifest_sha256" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_missing_summary_archive_path(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-summary-archive-path.tar.gz"
    files = _add_sdpo_acceptance_manifest(_minimal_sdpo_acceptance_archive_files(summary_archive_path=""))
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "summary is missing archive_path" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_summary_archive_path_under_output_root(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-summary-archive-under-output-root.tar.gz"
    files = _add_sdpo_acceptance_manifest(
        _minimal_sdpo_acceptance_archive_files(summary_archive_path="outputs/sdpo-cuda-acceptance/proof.tar.gz")
    )
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "summary archive_path must be outside output_root" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_missing_summary_proof_pointer(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-summary-proof-pointer.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    summary = files["sdpo_cuda_acceptance_summary.txt"].decode("utf-8")
    files["sdpo_cuda_acceptance_summary.txt"] = "\n".join(
        line for line in summary.splitlines() if not line.startswith("live_verify_report_file=")
    ).encode("utf-8")
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "summary is missing live_verify_report_file" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_mismatched_summary_proof_pointer(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-summary-mismatched-pointer.tar.gz"
    files = _minimal_sdpo_acceptance_archive_files()
    summary = files["sdpo_cuda_acceptance_summary.txt"].decode("utf-8")
    files["sdpo_cuda_acceptance_summary.txt"] = summary.replace(
        "live_verify_report_file=outputs/sdpo-cuda-acceptance/live/sdpo_smoke_verify_report.txt",
        "live_verify_report_file=outputs/sdpo-cuda-acceptance/live/wrong-report.txt",
    ).encode("utf-8")
    files = _add_sdpo_acceptance_manifest(files)
    _write_sdpo_acceptance_tar(archive_path, files)

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "summary live_verify_report_file must point to archive member" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_required_file_as_directory(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-required-dir.tar.gz"
    files = _add_sdpo_acceptance_manifest(_minimal_sdpo_acceptance_archive_files(include_live_provenance=False))
    required_dir = tarfile.TarInfo("live/sdpo_smoke_provenance.txt")
    required_dir.type = tarfile.DIRTYPE
    _write_sdpo_acceptance_tar(archive_path, files, extra_members=[(required_dir, None)])

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "archive required proof member is not a regular file: live/sdpo_smoke_provenance.txt" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_unsupported_member_type(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-symlink.tar.gz"
    files = _add_sdpo_acceptance_manifest(_minimal_sdpo_acceptance_archive_files())
    symlink = tarfile.TarInfo("live/run_default/token_exports/latest")
    symlink.type = tarfile.SYMTYPE
    symlink.linkname = "step_1"
    _write_sdpo_acceptance_tar(archive_path, files, extra_members=[(symlink, None)])

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "archive contains unsupported member type: 'live/run_default/token_exports/latest'" in result.stderr


def test_sdpo_cuda_acceptance_archive_verifier_rejects_duplicate_member_name(tmp_path):
    archive_path = tmp_path / "bad-sdpo-acceptance-duplicate.tar.gz"
    files = _add_sdpo_acceptance_manifest(_minimal_sdpo_acceptance_archive_files())
    duplicate_data = b"duplicate"
    duplicate = tarfile.TarInfo("live/sdpo_smoke_provenance.txt")
    duplicate.size = len(duplicate_data)
    _write_sdpo_acceptance_tar(archive_path, files, extra_members=[(duplicate, duplicate_data)])

    result = subprocess.run(
        [sys.executable, str(VERIFY_ACCEPTANCE_ARCHIVE_SCRIPT), str(archive_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "archive contains duplicate member path: 'live/sdpo_smoke_provenance.txt'" in result.stderr


def test_verify_sdpo_smoke_artifacts_accepts_strict_provenance(tmp_path):
    output_dir = tmp_path / "sdpo-smoke"
    _write_minimal_sdpo_smoke_artifacts(output_dir)

    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SMOKE_ARTIFACTS_SCRIPT),
            str(output_dir),
            "--expected-topk",
            "2",
            "--require-provenance",
            "--expected-provenance-mode",
            "live",
            "--expected-provenance-config",
            "configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml",
        ],
        cwd=REPO_ROOT,
        env=_sdpo_cli_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Verified SDPO smoke provenance:" in result.stdout
    assert "Verified SDPO token exports:" in result.stdout


@pytest.mark.parametrize(
    ("provenance_extra", "message"),
    [
        ("git_branch=overwritten\n", "repeats provenance field: git_branch"),
        ("git_status_short_begin\ngit_status_short_end\n", "repeats provenance field: git_status_short_begin"),
        ("not-key-value\n", "is malformed: 'not-key-value'"),
    ],
)
def test_verify_sdpo_smoke_artifacts_rejects_ambiguous_provenance(tmp_path, provenance_extra, message):
    output_dir = tmp_path / "sdpo-smoke"
    _write_minimal_sdpo_smoke_artifacts(output_dir, provenance_extra=provenance_extra)

    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SMOKE_ARTIFACTS_SCRIPT),
            str(output_dir),
            "--expected-topk",
            "2",
            "--require-provenance",
        ],
        cwd=REPO_ROOT,
        env=_sdpo_cli_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Invalid SDPO smoke provenance:" in result.stderr
    assert message in result.stderr


def test_verify_sdpo_smoke_artifacts_rejects_malformed_source_fingerprint(tmp_path):
    output_dir = tmp_path / "sdpo-smoke"
    _write_minimal_sdpo_smoke_artifacts(output_dir)
    provenance_file = output_dir / "sdpo_smoke_provenance.txt"
    provenance_file.write_text(
        provenance_file.read_text(encoding="utf-8").replace(
            f"git_cached_diff_sha256={hashlib.sha256(b'cached').hexdigest()}",
            "git_cached_diff_sha256=not-a-sha256",
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SMOKE_ARTIFACTS_SCRIPT),
            str(output_dir),
            "--expected-topk",
            "2",
            "--require-provenance",
        ],
        cwd=REPO_ROOT,
        env=_sdpo_cli_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "SDPO smoke provenance field git_cached_diff_sha256 must be a lowercase SHA-256 hex digest" in result.stderr


def test_sdpo_smoke_script_no_run_verifies_live_artifacts_with_expected_topk(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}
    output_dir = tmp_path / "existing-sdpo-smoke"
    output_dir.mkdir()

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--no-run", "--output-dir", str(output_dir)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"Skipping training; verifying existing output directory: {output_dir}" in result.stdout
    assert f"VERIFY <{output_dir}> <--expected-topk> <100>" in result.stdout
    assert f"Wrote SDPO smoke verifier report: {output_dir / 'sdpo_smoke_verify_report.txt'}" in result.stdout
    assert "--require-provenance" not in result.stdout
    assert "Wrote SDPO smoke provenance" not in result.stdout
    assert not (output_dir / "sdpo_smoke_provenance.txt").exists()
    assert f"VERIFY <{output_dir}> <--expected-topk> <100>" in (output_dir / "sdpo_smoke_verify_report.txt").read_text(
        encoding="utf-8"
    )
    assert "--require-ema-teacher" not in result.stdout
    assert "uv run rl" not in result.stdout


def test_sdpo_smoke_script_runs_training_then_strict_artifact_verifier(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}
    output_dir = tmp_path / "new-sdpo-smoke"

    result = subprocess.run(
        [
            "bash",
            str(SMOKE_SCRIPT),
            "--output-dir",
            str(output_dir),
            "--clean-output-dir",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Running SDPO smoke config: configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml" in result.stdout
    assert (
        "TRAIN <@> <configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml> "
        f"<--output-dir> <{output_dir}> <--clean-output-dir>"
    ) in result.stdout
    assert f"Wrote SDPO smoke provenance: {output_dir / 'sdpo_smoke_provenance.txt'}" in result.stdout
    assert (
        f"VERIFY <{output_dir}> <--expected-topk> <100> <--require-provenance> "
        "<--expected-provenance-mode> <live> <--expected-provenance-config> "
        "<configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml>"
    ) in result.stdout
    assert f"Wrote SDPO smoke verifier report: {output_dir / 'sdpo_smoke_verify_report.txt'}" in result.stdout
    assert result.stdout.index("TRAIN <@>") < result.stdout.index(f"VERIFY <{output_dir}>")
    verify_report = (output_dir / "sdpo_smoke_verify_report.txt").read_text(encoding="utf-8")
    assert f"VERIFY <{output_dir}> <--expected-topk> <100> <--require-provenance>" in verify_report
    provenance = (output_dir / "sdpo_smoke_provenance.txt").read_text(encoding="utf-8")
    provenance_values = dict(
        line.split("=", 1)
        for line in provenance.splitlines()
        if "=" in line and not line.startswith(" ") and not line.startswith(" M")
    )
    assert "sdpo_smoke_provenance_version=1" in provenance
    assert "mode=live" in provenance
    assert "config=configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml" in provenance
    assert f"output_dir={output_dir}" in provenance
    assert "expected_topk=100" in provenance
    assert "orchestrator.algo.distillation_topk=100" in provenance
    assert "orchestrator.algo.distillation_topk_support=student" in provenance
    assert "orchestrator.algo.teacher_regularization=live-policy" in provenance
    assert "orchestrator.algo.successful_demonstration_selection=batch_order" in provenance
    assert "orchestrator.algo.template_target=first_user" in provenance
    assert "trainer.sdpo_loss.distillation_topk=100" in provenance
    assert "trainer.sdpo_runtime.teacher_regularization=live-policy" in provenance
    assert provenance_values["git_commit"]
    assert provenance_values["git_branch"]
    assert "git_diff_sha256=" in provenance
    assert "git_cached_diff_sha256=" in provenance
    assert "git_untracked_manifest_sha256=" in provenance
    assert "python_runner=uv run python" in provenance
    assert "rl_runner=uv run rl" in provenance
    assert "git_untracked_manifest_begin" in provenance
    assert "git_untracked_manifest_end" in provenance
    assert provenance.index("git_untracked_manifest_begin") < provenance.index("git_untracked_manifest_end")
    assert "git_status_short_begin" in provenance
    assert "git_status_short_end" in provenance


def test_sdpo_smoke_script_rejects_non_full_logit_sdpo_smoke(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SDPO_FAKE_FULL_LOGIT_DISTILLATION": "False",
    }

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--no-run", "--output-dir", "outputs/not-full-logit"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "trainer.sdpo_loss.full_logit_distillation=true" in result.stderr
    assert "VERIFY" not in result.stdout
    assert "uv run rl" not in result.stdout


@pytest.mark.parametrize(
    ("env_var", "value", "message"),
    [
        ("SDPO_FAKE_DISTILLATION_ADD_TAIL", "False", "trainer.sdpo_loss.distillation_add_tail=true"),
        ("SDPO_FAKE_SDPO_ALPHA", "0.25", "trainer.sdpo_loss.alpha=0.5"),
        ("SDPO_FAKE_IS_CLIP", "None", "trainer.sdpo_loss.is_clip=2.0"),
        ("SDPO_FAKE_ROLLOUT_IS", "sequence", "trainer.sdpo_loss.rollout_is='token'"),
        ("SDPO_FAKE_ROLLOUT_IS_THRESHOLD", "1.5", "trainer.sdpo_loss.rollout_is_threshold=2.0"),
        ("SDPO_FAKE_ROLLOUT_IS_BATCH_NORMALIZE", "True", "trainer.sdpo_loss.rollout_is_batch_normalize=false"),
    ],
)
def test_sdpo_smoke_script_rejects_non_reference_sdpo_loss_knobs(tmp_path, env_var, value, message):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        env_var: value,
    }

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--no-run", "--output-dir", "outputs/non-reference-loss"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert "VERIFY" not in result.stdout
    assert "uv run rl" not in result.stdout


@pytest.mark.parametrize(
    ("env_var", "value", "message"),
    [
        ("SDPO_FAKE_SUCCESS_REWARD_THRESHOLD", "1.0", "orchestrator.algo.success_reward_threshold=0.5"),
        (
            "SDPO_FAKE_SUCCESSFUL_DEMONSTRATION_SELECTION",
            "highest_reward",
            "orchestrator.algo.successful_demonstration_selection='batch_order'",
        ),
        (
            "SDPO_FAKE_DONT_REPROMPT_ON_SELF_SUCCESS",
            "False",
            "orchestrator.algo.dont_reprompt_on_self_success=true",
        ),
        (
            "SDPO_FAKE_REMOVE_THINKING_FROM_DEMONSTRATION",
            "False",
            "orchestrator.algo.remove_thinking_from_demonstration=true",
        ),
        ("SDPO_FAKE_INCLUDE_ENVIRONMENT_FEEDBACK", "False", "orchestrator.algo.include_environment_feedback=true"),
        (
            "SDPO_FAKE_ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION",
            "False",
            "orchestrator.algo.environment_feedback_only_without_solution=true",
        ),
        ("SDPO_FAKE_MAX_REPROMPT_LEN", "2048", "orchestrator.algo.max_reprompt_len=10240"),
        ("SDPO_FAKE_REPROMPT_TRUNCATION", "left", "orchestrator.algo.reprompt_truncation='right'"),
        ("SDPO_FAKE_ASSISTANT_PREFIX", "Solve carefully.", "orchestrator.algo.assistant_prefix=''"),
        ("SDPO_FAKE_MULTI_TURN", "True", "orchestrator.algo.multi_turn=false"),
        ("SDPO_FAKE_TEMPLATE_TARGET", "last_user", "orchestrator.algo.template_target='first_user'"),
        (
            "SDPO_FAKE_TEMPLATE",
            "{question}{successful_solution_block}",
            "Hübotter outer template",
        ),
        (
            "SDPO_FAKE_SOLUTION_TEMPLATE",
            "Correct solution.",
            "Hübotter solution_template",
        ),
        (
            "SDPO_FAKE_FEEDBACK_TEMPLATE",
            "Feedback.",
            "Hübotter feedback_template",
        ),
    ],
)
def test_sdpo_smoke_script_rejects_non_reference_hindsight_conditioning_knobs(tmp_path, env_var, value, message):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        env_var: value,
    }

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--no-run", "--output-dir", "outputs/non-reference-conditioning"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert "VERIFY" not in result.stdout
    assert "uv run rl" not in result.stdout


def test_sdpo_smoke_script_rejects_split_algorithm_and_trainer_topk(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SDPO_FAKE_ALGO_DISTILLATION_TOPK": "64",
    }

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--no-run", "--output-dir", "outputs/split-topk"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "orchestrator.algo.distillation_topk to match trainer.sdpo_loss.distillation_topk" in result.stderr
    assert "VERIFY" not in result.stdout
    assert "uv run rl" not in result.stdout


def test_sdpo_smoke_script_rejects_non_student_algorithm_topk_support(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SDPO_FAKE_ALGO_DISTILLATION_TOPK_SUPPORT": "teacher",
    }

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--no-run", "--output-dir", "outputs/teacher-support"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "orchestrator.algo.distillation_topk_support='student'" in result.stderr
    assert "VERIFY" not in result.stdout
    assert "uv run rl" not in result.stdout


def test_sdpo_smoke_script_rejects_external_teacher_for_reference_smoke(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SDPO_FAKE_ALGO_MODEL": "FrozenModelConfig(name='teacher', base_url=['http://teacher/v1'])",
    }

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--no-run", "--output-dir", "outputs/external-teacher-sdpo"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "orchestrator.algo.model='policy' for self-distillation" in result.stderr
    assert "VERIFY" not in result.stdout
    assert "uv run rl" not in result.stdout


def test_sdpo_smoke_script_rejects_missing_preflight_export_timeout(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SDPO_FAKE_PREFLIGHT_EXPORT_TIMEOUT_S": "None",
    }

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--no-run", "--output-dir", "outputs/missing-preflight-timeout"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "orchestrator.algo.preflight_export_timeout_s" in result.stderr
    assert "VERIFY" not in result.stdout
    assert "uv run rl" not in result.stdout


def test_sdpo_smoke_script_rejects_context_parallelism(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SDPO_FAKE_TRAINER_CP": "2",
    }

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--no-run", "--output-dir", "outputs/context-parallel-sdpo"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "trainer.model.cp=1" in result.stderr
    assert "VERIFY" not in result.stdout
    assert "uv run rl" not in result.stdout


def test_sdpo_smoke_script_rejects_split_algorithm_and_trainer_teacher_modes(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SDPO_FAKE_ALGO_TEACHER_REGULARIZATION": "ema",
    }

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--no-run", "--output-dir", "outputs/split-teacher-mode"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert (
        "orchestrator.algo.teacher_regularization to match trainer.sdpo_runtime.teacher_regularization" in result.stderr
    )
    assert "VERIFY" not in result.stdout
    assert "uv run rl" not in result.stdout


def test_sdpo_smoke_script_rejects_split_algorithm_and_trainer_teacher_update_rates(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SDPO_FAKE_ALGO_TEACHER_UPDATE_RATE": "0.1",
    }

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--no-run", "--output-dir", "outputs/split-teacher-rate"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "orchestrator.algo.teacher_update_rate to match trainer.sdpo_runtime.teacher_update_rate" in result.stderr
    assert "VERIFY" not in result.stdout
    assert "uv run rl" not in result.stdout


def test_sdpo_smoke_script_rejects_non_reference_teacher_update_rate_even_when_matched(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SDPO_FAKE_TRAINER_TEACHER_UPDATE_RATE": "0.1",
        "SDPO_FAKE_ALGO_TEACHER_UPDATE_RATE": "0.1",
    }

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--ema", "--no-run", "--output-dir", "outputs/non-reference-teacher-rate"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "trainer.sdpo_runtime.teacher_update_rate=0.05" in result.stderr
    assert "VERIFY" not in result.stdout
    assert "uv run rl" not in result.stdout


def test_sdpo_smoke_script_rejects_missing_ema_teacher_base_url(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SDPO_FAKE_SDPO_TEACHER_BASE_URL": "None",
    }

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--ema", "--no-run", "--output-dir", "outputs/missing-ema-teacher"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "orchestrator.sdpo_teacher.base_url=['http://localhost:8001/v1']" in result.stderr
    assert "VERIFY" not in result.stdout
    assert "uv run rl" not in result.stdout


def test_sdpo_smoke_script_rejects_live_teacher_base_url(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SDPO_FAKE_SDPO_TEACHER_BASE_URL": "['http://localhost:8001/v1']",
    }

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--no-run", "--output-dir", "outputs/live-with-teacher"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "live SDPO smoke must not configure orchestrator.sdpo_teacher.base_url" in result.stderr
    assert "VERIFY" not in result.stdout
    assert "uv run rl" not in result.stdout


def test_sdpo_smoke_script_no_run_verifies_ema_artifacts_with_teacher_broadcasts(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--ema", "--no-run", "--output-dir", "outputs/existing-sdpo-ema-smoke"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Skipping training; verifying existing output directory: outputs/existing-sdpo-ema-smoke" in result.stdout
    assert "VERIFY <outputs/existing-sdpo-ema-smoke> <--expected-topk> <100> <--require-ema-teacher>" in result.stdout
    assert "uv run rl" not in result.stdout


def test_sdpo_smoke_script_rejects_ema_without_local_teacher_gpu(tmp_path):
    _write_fake_uv_for_no_run_smoke(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SDPO_FAKE_NUM_SDPO_TEACHER_GPUS": "0",
    }

    result = subprocess.run(
        ["bash", str(SMOKE_SCRIPT), "--ema", "--no-run", "--output-dir", "outputs/no-teacher-gpu"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "deployment.num_sdpo_teacher_gpus > 0" in result.stderr
    assert "VERIFY" not in result.stdout
    assert "uv run rl" not in result.stdout


def test_sdpo_smoke_script_checks_teacher_runtime_mode():
    script = SMOKE_SCRIPT.read_text()

    assert "uses_sdpo_internal_teacher_regularization" in script
    assert "--ema smoke requires an internal SDPO teacher runtime" in script
    assert "live SDPO smoke must not use an internal SDPO teacher runtime" in script


def test_sdpo_smoke_script_uses_strict_artifact_verifier():
    script = SMOKE_SCRIPT.read_text()

    assert "scripts/verify_sdpo_smoke_artifacts.py" in script
    assert "--expected-topk" in script
    assert "--require-provenance" in script
    assert "--expected-provenance-mode" in script
    assert "--expected-provenance-config" in script
    assert "--require-ema-teacher" in script


def test_sdpo_smoke_script_records_nonempty_git_identity_for_detached_checkouts():
    script = SMOKE_SCRIPT.read_text()

    assert "git_commit_sha()" in script
    assert "git_branch_name()" in script
    assert "detached-$short_commit" in script


def test_sdpo_cuda_acceptance_summary_records_nonempty_git_identity_for_detached_checkouts():
    script = ACCEPTANCE_SCRIPT.read_text()

    assert "git_commit_sha()" in script
    assert "git_branch_name()" in script
    assert "detached-$short_commit" in script
