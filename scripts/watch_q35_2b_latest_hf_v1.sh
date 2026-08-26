#!/usr/bin/env bash
set -euo pipefail

remote=${1:?SSH remote required}
ssh_key=${2:?SSH key required}
remote_state_dir=${3:?remote controller state directory required}
checkpoint_root=${4:?local checkpoint root required}
local_state_dir=${5:?local sync state directory required}
env_file=${6:?local credential env file required}
coordinator_repo=${7:?coordinator HF repo required}
child_repo=${8:?child HF repo required}
prime_revision=${9:?Prime-RL source revision required}
verifier_revision=${10:?verifier source revision required}
poll_seconds=${11:-600}
remote_upload_helper=${12:-}
remote_uv_bin=${13:-}
retry_seconds=${14:-60}

if [[ "$poll_seconds" -lt 60 || "$retry_seconds" -lt 1 || "$retry_seconds" -gt "$poll_seconds" ]]; then
  echo "poll interval must be at least 60 seconds and retry interval must be between 1 and the poll interval" >&2
  exit 2
fi

mkdir -p "$local_state_dir"
log_file="$local_state_dir/watch.log"
script_dir=$(cd "$(dirname "$0")" && pwd)
uv_bin=${UV_BIN:-uv}

record() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >>"$log_file"
}

while true; do
  sleep_seconds=$poll_seconds
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
  export HF_TOKEN=${HF_TOKEN:-${HF_KEY:-}}
  if [[ -z "$HF_TOKEN" ]]; then
    record "sync_failed missing_hf_credential"
    sleep_seconds=$retry_seconds
  else
    extra_args=()
    if [[ -n "$remote_upload_helper" || -n "$remote_uv_bin" ]]; then
      if [[ -z "$remote_upload_helper" || -z "$remote_uv_bin" ]]; then
        record "sync_failed remote_upload_requires_helper_and_uv"
        sleep "$retry_seconds"
        continue
      fi
      extra_args+=(
        --remote-upload-helper "$remote_upload_helper"
        --remote-uv-bin "$remote_uv_bin"
      )
    fi
    if output=$(cd "${TMPDIR:-/tmp}" && PYTHONDONTWRITEBYTECODE=1 \
    "$uv_bin" run --no-project --with 'huggingface-hub>=0.34' python \
    "$script_dir/sync_q35_2b_latest_hf_v1.py" \
    --remote "$remote" \
    --ssh-key "$ssh_key" \
    --remote-state-dir "$remote_state_dir" \
    --checkpoint-root "$checkpoint_root" \
    --local-state-dir "$local_state_dir" \
    --coordinator-repo "$coordinator_repo" \
    --child-repo "$child_repo" \
    --prime-revision "$prime_revision" \
    --verifier-revision "$verifier_revision" \
    "${extra_args[@]}" 2>&1); then
      output=${output//$'\r'/}
      output=${output//$'\n'/\\n}
      record "sync_ok $output"
    else
      status=$?
      output=${output//$'\r'/}
      output=${output//$'\n'/\\n}
      record "sync_failed status=$status output=$output"
      sleep_seconds=$retry_seconds
    fi
  fi
  sleep "$sleep_seconds"
done
