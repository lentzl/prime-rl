#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Start the full SDPO CUDA acceptance run in the background.

Usage:
  scripts/start_sdpo_cuda_acceptance_background.sh [--output-root DIR] [--archive PATH] [--log PATH] [--pid-file PATH] [--no-clean-output-dir] [--skip-host-preflight] [--skip-config-preflight] [--preflight-only] [--status] [--dry-run]

Options:
  --output-root DIR       Write live and EMA smoke outputs below DIR. Defaults to
                          outputs/sdpo-cuda-acceptance.
  --archive PATH          Write the verified proof tarball to PATH. Defaults to
                          outputs/sdpo-cuda-acceptance-proof.tar.gz.
  --log PATH              Write stdout/stderr from the acceptance run to PATH.
                          Defaults to outputs/sdpo-cuda-acceptance.log.
  --pid-file PATH         Record the background PID at PATH. Defaults to
                          outputs/sdpo-cuda-acceptance.pid.
  --no-clean-output-dir   Preserve existing live/EMA output dirs. By default the
                          acceptance run starts from a clean output root.
  --skip-host-preflight   Do not check basic host CUDA/tool availability before
                          starting the background run.
  --skip-config-preflight Do not run the SDPO acceptance config check before
                          starting the background run.
  --preflight-only        Run host/config preflight checks and exit without
                          starting.
  --status                Print PID/log/archive status without starting a run.
  --dry-run               Print the command and paths without starting it.
  -h, --help              Show this help.

Run this on a Linux CUDA/vLLM box after syncing the SDPO branch. The wrapped
acceptance command verifies live-policy and EMA smokes, writes a manifest, builds
the proof archive, and verifies that archive before this helper reports it as the
artifact to download. Successful archive verification should print
raw_artifacts=verified with recomputed live/EMA token-export counters.

Recommended fresh-box flow:
  scripts/start_sdpo_cuda_acceptance_background.sh --preflight-only
  scripts/start_sdpo_cuda_acceptance_background.sh
  scripts/start_sdpo_cuda_acceptance_background.sh --status
EOF
}

output_root="outputs/sdpo-cuda-acceptance"
archive_path="outputs/sdpo-cuda-acceptance-proof.tar.gz"
log_path="outputs/sdpo-cuda-acceptance.log"
pid_file="outputs/sdpo-cuda-acceptance.pid"
clean_output_dir=1
dry_run=0
status_only=0
host_preflight=1
config_preflight=1
preflight_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      if [[ $# -lt 2 ]]; then
        echo "Error: --output-root requires a value" >&2
        exit 2
      fi
      output_root="$2"
      shift 2
      ;;
    --archive)
      if [[ $# -lt 2 ]]; then
        echo "Error: --archive requires a value" >&2
        exit 2
      fi
      archive_path="$2"
      shift 2
      ;;
    --log)
      if [[ $# -lt 2 ]]; then
        echo "Error: --log requires a value" >&2
        exit 2
      fi
      log_path="$2"
      shift 2
      ;;
    --pid-file)
      if [[ $# -lt 2 ]]; then
        echo "Error: --pid-file requires a value" >&2
        exit 2
      fi
      pid_file="$2"
      shift 2
      ;;
    --no-clean-output-dir)
      clean_output_dir=0
      shift
      ;;
    --skip-host-preflight)
      host_preflight=0
      shift
      ;;
    --skip-config-preflight)
      config_preflight=0
      shift
      ;;
    --preflight-only)
      preflight_only=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --status)
      status_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

read -r -a python_runner <<< "${SDPO_ACCEPTANCE_PYTHON_RUNNER:-${SDPO_SMOKE_PYTHON_RUNNER:-uv run --extra flash-attn python}}"
read -r -a acceptance_runner <<< "${SDPO_ACCEPTANCE_RUNNER:-scripts/run_sdpo_cuda_acceptance.sh}"
min_gpus="${SDPO_ACCEPTANCE_MIN_GPUS:-3}"

git_commit_sha() {
  git rev-parse HEAD 2>/dev/null || echo "unknown"
}

git_branch_name() {
  local branch
  branch="$(git branch --show-current 2>/dev/null || true)"
  if [[ -n "$branch" ]]; then
    echo "$branch"
    return
  fi
  local short_commit
  short_commit="$(git rev-parse --short HEAD 2>/dev/null || true)"
  if [[ -n "$short_commit" ]]; then
    echo "detached-$short_commit"
    return
  fi
  echo "unknown"
}

run_host_preflight() {
  echo "Running SDPO CUDA acceptance host preflight..."
  for required_cmd in uv tar nvidia-smi; do
    if ! command -v "$required_cmd" >/dev/null 2>&1; then
      echo "Error: missing required command for SDPO CUDA acceptance: $required_cmd" >&2
      return 2
    fi
  done
  if ! "${python_runner[@]}" -c "import flash_attn" >/dev/null 2>&1; then
    echo "Error: SDPO CUDA acceptance requires flash_attn; use the default uv runner or include prime-rl[flash-attn] in SDPO_ACCEPTANCE_PYTHON_RUNNER." >&2
    return 2
  fi
  if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
    echo "Error: missing required hashing command for SDPO CUDA acceptance: sha256sum or shasum" >&2
    return 2
  fi
  if ! [[ "$min_gpus" =~ ^[0-9]+$ ]] || [[ "$min_gpus" -le 0 ]]; then
    echo "Error: SDPO_ACCEPTANCE_MIN_GPUS must be a positive integer (got '$min_gpus')" >&2
    return 2
  fi
  gpu_count="$(
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
      | sed '/^[[:space:]]*$/d' \
      | wc -l \
      | tr -d '[:space:]'
  )"
  if [[ "$gpu_count" -lt "$min_gpus" ]]; then
    echo "Error: SDPO CUDA acceptance requires at least $min_gpus visible GPUs, found $gpu_count" >&2
    return 2
  fi
  echo "Git branch: $(git branch --show-current 2>/dev/null || echo unknown)"
  echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  if [[ -n "$(git status --short 2>/dev/null || true)" ]]; then
    echo "Git status: dirty (provenance will record source fingerprints)"
  else
    echo "Git status: clean"
  fi
  echo "Visible GPUs: $gpu_count"
  nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader,nounits || true
  echo "Disk space at repo root:"
  df -h "$repo_root" || true
  echo "Host preflight passed."
}

run_config_preflight() {
  echo "Running SDPO CUDA acceptance config preflight..."
  "${acceptance_runner[@]}" --check-config
  echo "Config preflight passed."
}

print_quoted_command() {
  printf ' %q' "$@"
  printf '\n'
}

print_command() {
  if [[ "$#" -eq 0 ]]; then
    printf '\n'
    return
  fi
  printf '%q' "$1"
  shift
  printf ' %q' "$@"
  printf '\n'
}

canonical_path_without_creating() {
  local path="$1"
  local trimmed_path="${path%/}"
  if [[ -z "$trimmed_path" ]]; then
    trimmed_path="."
  fi
  if [[ "$trimmed_path" != /* ]]; then
    trimmed_path="$PWD/$trimmed_path"
  fi

  local existing="$trimmed_path"
  local suffix=""
  while [[ ! -e "$existing" && "$existing" != "/" ]]; do
    suffix="/$(basename "$existing")$suffix"
    existing="$(dirname "$existing")"
  done

  if [[ -d "$existing" ]]; then
    echo "$(cd "$existing" && pwd -P)$suffix"
    return
  fi

  echo "$(cd "$(dirname "$existing")" && pwd -P)/$(basename "$existing")$suffix"
}

require_archive_outside_output_root() {
  local archive_abs
  local output_abs
  archive_abs="$(canonical_path_without_creating "$archive_path")"
  output_abs="$(canonical_path_without_creating "$output_root")"
  if [[ "$archive_abs" == "$output_abs" || "$archive_abs" == "$output_abs"/* ]]; then
    echo "Error: --archive must be outside --output-root so the proof tarball cannot overwrite or self-include acceptance artifacts." >&2
    echo "Archive path: $archive_path" >&2
    echo "Output root: $output_root" >&2
    exit 2
  fi
}

require_no_running_process() {
  if [[ -f "$pid_file" ]]; then
    old_pid="$(cat "$pid_file")"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Error: existing SDPO CUDA acceptance process appears to be running: pid=$old_pid" >&2
      echo "PID file: $pid_file" >&2
      exit 2
    fi
  fi
}

cmd=(
  "${acceptance_runner[@]}"
  --output-root
  "$output_root"
  --archive
  "$archive_path"
)
if [[ "$clean_output_dir" -eq 1 ]]; then
  cmd+=(--clean-output-dir)
fi

printf 'SDPO CUDA acceptance command:'
printf ' %q' "${cmd[@]}"
printf '\n'
echo "Output root: $output_root"
echo "Archive: $archive_path"
echo "Log: $log_path"
echo "PID file: $pid_file"
require_archive_outside_output_root

if [[ "$status_only" -eq 1 ]]; then
  pid_recorded=0
  process_running=0
  process_stopped=0
  archive_nonempty=0
  archive_empty=0
  status_exit=0
  if [[ -f "$pid_file" ]]; then
    pid_recorded=1
    old_pid="$(cat "$pid_file")"
    echo "Recorded PID: $old_pid"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Process status: running"
      process_running=1
      ps -p "$old_pid" -o pid,etime,stat,command || true
    else
      echo "Process status: not running"
      process_stopped=1
    fi
  else
    echo "Recorded PID: missing"
    echo "Process status: unknown"
  fi
  if [[ -s "$archive_path" ]]; then
    echo "Archive status: present non-empty"
    archive_nonempty=1
  elif [[ -e "$archive_path" ]]; then
    echo "Archive status: present empty"
    archive_empty=1
  else
    echo "Archive status: missing"
  fi
  if [[ -f "$log_path" ]]; then
    echo "Log status: present"
    echo "Last log lines:"
    tail -n 20 "$log_path"
  else
    echo "Log status: missing"
  fi
  if [[ "$pid_recorded" -eq 1 && "$process_stopped" -eq 1 && "$archive_nonempty" -eq 0 ]]; then
    echo "Completion status: failed or incomplete; recorded process is stopped without a non-empty archive"
    status_exit=2
  fi
  if [[ "$archive_empty" -eq 1 && "$process_running" -eq 0 ]]; then
    echo "Completion status: failed or incomplete; archive file is empty"
    status_exit=2
  fi
  if [[ "$archive_nonempty" -eq 1 ]]; then
    if [[ "$process_running" -eq 1 ]]; then
      echo "Archive verification: skipped while process is running"
    elif "${python_runner[@]}" scripts/verify_sdpo_cuda_acceptance_archive.py \
      --expected-acceptance-mode training \
      --expected-git-commit "$(git_commit_sha)" \
      --expected-git-branch "$(git_branch_name)" \
      "$archive_path"; then
      echo "Archive verification: passed"
    else
      echo "Archive verification: failed"
      status_exit=2
    fi
  fi
  exit "$status_exit"
fi

if [[ "$dry_run" -eq 1 ]]; then
  echo "Dry run only; not starting SDPO CUDA acceptance."
  exit 0
fi

if [[ "$preflight_only" -eq 1 && "$host_preflight" -eq 0 && "$config_preflight" -eq 0 ]]; then
  echo "Error: --preflight-only would not run any checks because both preflights are skipped." >&2
  exit 2
fi

if [[ "$preflight_only" -eq 0 ]]; then
  require_no_running_process
fi

if [[ "$host_preflight" -eq 1 ]]; then
  run_host_preflight
fi

if [[ "$config_preflight" -eq 1 ]]; then
  run_config_preflight
fi

if [[ "$preflight_only" -eq 1 ]]; then
  echo "Preflight only; not starting SDPO CUDA acceptance."
  exit 0
fi

if [[ -e "$archive_path" || -L "$archive_path" ]]; then
  if [[ -L "$archive_path" || ! -f "$archive_path" ]]; then
    echo "Error: cannot start SDPO CUDA acceptance; archive path exists and is not a regular file: $archive_path" >&2
    exit 2
  fi
  rm -f "$archive_path"
  echo "Removed existing SDPO CUDA acceptance archive before start: $archive_path"
fi

mkdir -p "$(dirname "$log_path")" "$(dirname "$pid_file")" "$(dirname "$archive_path")" "$output_root"
nohup "${cmd[@]}" > "$log_path" 2>&1 &
pid="$!"
echo "$pid" > "$pid_file"

archive_download_path="$archive_path"
case "$archive_download_path" in
  /*) ;;
  *) archive_download_path="$(pwd)/$archive_download_path" ;;
esac

echo "Started SDPO CUDA acceptance in background: pid=$pid"
echo "Status:"
printf '  '
print_quoted_command \
  scripts/start_sdpo_cuda_acceptance_background.sh \
  --status \
  --output-root \
  "$output_root" \
  --archive \
  "$archive_path" \
  --log \
  "$log_path" \
  --pid-file \
  "$pid_file"
echo "Monitor:"
echo "  tail -f $log_path"
echo "Check process:"
echo "  ps -p $pid -o pid,etime,stat,command"
echo "After completion, verify archive on the remote box:"
printf '  '
print_command \
  "${python_runner[@]}" \
  scripts/verify_sdpo_cuda_acceptance_archive.py \
  --expected-acceptance-mode \
  training \
  --expected-git-commit \
  "$(git_commit_sha)" \
  --expected-git-branch \
  "$(git_branch_name)" \
  "$archive_path"
echo "  Expected verifier output includes: raw_artifacts=verified"

echo "Then download and verify the same archive locally:"
echo "  scp USER@HOST:$archive_download_path ."
printf '  '
print_command \
  "${python_runner[@]}" \
  scripts/verify_sdpo_cuda_acceptance_archive.py \
  --expected-acceptance-mode \
  training \
  --expected-git-commit \
  "$(git_commit_sha)" \
  --expected-git-branch \
  "$(git_branch_name)" \
  "$(basename "$archive_path")"
echo "  Expected verifier output includes: raw_artifacts=verified"
