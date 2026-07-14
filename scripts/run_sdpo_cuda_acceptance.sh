#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the full SDPO CUDA acceptance sequence.

Usage:
  scripts/run_sdpo_cuda_acceptance.sh [--output-root DIR] [--clean-output-dir] [--check-config] [--no-run] [--archive PATH]

Options:
  --output-root DIR  Write live and EMA smoke outputs below DIR. Defaults to
                     outputs/sdpo-cuda-acceptance-<utc timestamp>.
  --clean-output-dir
                     Delete existing live/EMA output dirs before training.
  --check-config     Run only the live and EMA smoke config checks.
  --no-run           Skip training and strictly verify existing live/EMA
                     artifacts under --output-root (requires --output-root).
  --archive PATH     After successful training or no-run verification, write a
                     tar.gz archive of the acceptance proof artifacts to PATH.
  -h, --help         Show this help.

This is a thin convenience wrapper around run_sdpo_smoke_and_verify.sh. It does
not weaken that wrapper's config, provenance, token-export, or EMA checks. A
successful run writes a combined summary and a SHA-256 manifest beside the live
and EMA output directories before optionally archiving the proof artifacts. It
re-runs strict artifact verification after each training smoke, requires proof
files to be non-empty and proof directories to contain at least one non-empty
file before writing that summary or archive. When --archive is set, the written
tarball must also be non-empty, listable, and pass the offline archive verifier.
EOF
}

output_root=""
clean_output_dir=0
check_config=0
no_run=0
archive_path=""

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
    --clean-output-dir)
      clean_output_dir=1
      shift
      ;;
    --check-config)
      check_config=1
      shift
      ;;
    --no-run)
      no_run=1
      shift
      ;;
    --archive)
      if [[ $# -lt 2 ]]; then
        echo "Error: --archive requires a value" >&2
        exit 2
      fi
      archive_path="$2"
      shift 2
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

read -r -a smoke_wrapper <<< "${SDPO_ACCEPTANCE_SMOKE_WRAPPER:-scripts/run_sdpo_smoke_and_verify.sh}"
read -r -a python_runner <<< "${SDPO_ACCEPTANCE_PYTHON_RUNNER:-${SDPO_SMOKE_PYTHON_RUNNER:-uv run --extra flash-attn python}}"
expected_topk=100
live_config="configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml"
ema_config="configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml"
required_proof_entries=(
  live/sdpo_smoke_provenance.txt
  live/sdpo_smoke_verify_report.txt
  ema/sdpo_smoke_provenance.txt
  ema/sdpo_smoke_verify_report.txt
  live/run_default/token_exports
  ema/run_default/token_exports
  ema/run_default/broadcasts
)
required_archive_entries=(
  sdpo_cuda_acceptance_summary.txt
  sdpo_cuda_acceptance_manifest.txt
  "${required_proof_entries[@]}"
)
optional_acceptance_entries=(
  live/configs
  live/run_default/control
  ema/configs
  ema/run_default/control
)

git_commit_sha() {
  local commit
  commit="$(git rev-parse HEAD 2>/dev/null || true)"
  if [[ -n "$commit" ]]; then
    echo "$commit"
    return
  fi
  echo "unknown"
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

require_acceptance_summary_git_identity() {
  local field="$1"
  local value="$2"
  if [[ -z "$value" || "$value" == "unknown" || "$value" == "unavailable" ]]; then
    echo "Error: cannot write SDPO CUDA acceptance summary; $field must not be '${value:-<empty>}'" >&2
    exit 2
  fi
}

write_acceptance_summary() {
  local acceptance_mode="$1"
  local summary_file="$output_root/sdpo_cuda_acceptance_summary.txt"
  local git_commit
  local git_branch
  git_commit="$(git_commit_sha)"
  git_branch="$(git_branch_name)"
  require_acceptance_summary_git_identity git_commit "$git_commit"
  require_acceptance_summary_git_identity git_branch "$git_branch"
  mkdir -p "$output_root"
  {
    echo "sdpo_cuda_acceptance_summary_version=1"
    echo "acceptance_mode=$acceptance_mode"
    echo "verified_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "output_root=$output_root"
    echo "live_output_dir=$live_output_dir"
    echo "live_config=$live_config"
    echo "live_provenance_file=$live_output_dir/sdpo_smoke_provenance.txt"
    echo "live_verify_report_file=$live_output_dir/sdpo_smoke_verify_report.txt"
    echo "live_config_dir=$live_output_dir/configs"
    echo "live_run_control_dir=$live_output_dir/run_default/control"
    echo "live_token_exports_dir=$live_output_dir/run_default/token_exports"
    echo "ema_output_dir=$ema_output_dir"
    echo "ema_config=$ema_config"
    echo "ema_provenance_file=$ema_output_dir/sdpo_smoke_provenance.txt"
    echo "ema_verify_report_file=$ema_output_dir/sdpo_smoke_verify_report.txt"
    echo "ema_config_dir=$ema_output_dir/configs"
    echo "ema_run_control_dir=$ema_output_dir/run_default/control"
    echo "ema_token_exports_dir=$ema_output_dir/run_default/token_exports"
    echo "ema_broadcasts_dir=$ema_output_dir/run_default/broadcasts"
    echo "acceptance_manifest_file=$output_root/sdpo_cuda_acceptance_manifest.txt"
    echo "archive_path=$archive_path"
    echo "expected_topk=$expected_topk"
    echo "git_commit=$git_commit"
    echo "git_branch=$git_branch"
  } > "$summary_file"
  echo "Wrote SDPO CUDA acceptance summary: $summary_file"
}

hash_file_sha256() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
    return
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
    return
  fi
  echo "Error: cannot write SDPO CUDA acceptance manifest; sha256sum or shasum is required." >&2
  exit 2
}

write_acceptance_manifest() {
  local acceptance_mode="$1"
  local manifest_file="$output_root/sdpo_cuda_acceptance_manifest.txt"
  mkdir -p "$output_root"
  {
    echo "sdpo_cuda_acceptance_manifest_version=1"
    echo "acceptance_mode=$acceptance_mode"
    echo "verified_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "output_root=$output_root"
    echo "format=sha256 size_bytes relative_path"
    for entry in "${required_archive_entries[@]}" "${optional_acceptance_entries[@]}"; do
      local path="$output_root/$entry"
      if [[ "$entry" == "sdpo_cuda_acceptance_manifest.txt" || ! -e "$path" ]]; then
        continue
      fi
      if [[ -f "$path" ]]; then
        local size
        size="$(wc -c < "$path" | tr -d ' ')"
        echo "$(hash_file_sha256 "$path") $size $entry"
      elif [[ -d "$path" ]]; then
        while IFS= read -r file; do
          local relative_path
          local size
          relative_path="${file#"$output_root"/}"
          size="$(wc -c < "$file" | tr -d ' ')"
          echo "$(hash_file_sha256 "$file") $size $relative_path"
        done < <(find "$path" -type f | LC_ALL=C sort)
      fi
    done
  } > "$manifest_file"
  echo "Wrote SDPO CUDA acceptance manifest: $manifest_file"
}

require_verify_report_marker() {
  local report_file="$1"
  local marker="$2"
  local label="$3"
  local marker_count
  if [[ ! -f "$report_file" ]]; then
    echo "Error: missing SDPO smoke verifier report for $label: $report_file" >&2
    exit 2
  fi
  marker_count="$(awk -v marker="$marker" 'index($0, marker) == 1 { count++ } END { print count + 0 }' "$report_file")"
  if [[ "$marker_count" -eq 0 ]]; then
    echo "Error: SDPO smoke verifier report for $label is missing success marker: $marker" >&2
    echo "Report file: $report_file" >&2
    exit 2
  fi
  if [[ "$marker_count" -ne 1 ]]; then
    echo "Error: SDPO smoke verifier report for $label repeats success marker: $marker" >&2
    echo "Report file: $report_file" >&2
    exit 2
  fi
}

require_smoke_verify_report_markers() {
  local report_file="$1"
  local label="$2"
  local require_ema="$3"
  require_verify_report_marker "$report_file" "Verified SDPO smoke provenance:" "$label"
  require_verify_report_marker "$report_file" "Verified SDPO token exports:" "$label"
  if [[ "$require_ema" -eq 1 ]]; then
    require_verify_report_marker "$report_file" "Verified SDPO EMA broadcasts:" "$label"
  fi
}

run_smoke_wrapper() {
  if [[ -n "${SDPO_ACCEPTANCE_PYTHON_RUNNER:-}" && -z "${SDPO_SMOKE_PYTHON_RUNNER:-}" ]]; then
    SDPO_SMOKE_PYTHON_RUNNER="$SDPO_ACCEPTANCE_PYTHON_RUNNER" "${smoke_wrapper[@]}" "$@"
    return
  fi
  "${smoke_wrapper[@]}" "$@"
}

run_acceptance_verifier() {
  local output_dir="$1"
  local report_file="$2"
  local mode="$3"
  local config="$4"
  local label="$5"
  local require_ema="$6"
  local message="$7"

  echo "$message"
  local verify_cmd=(
    "${python_runner[@]}"
    scripts/verify_sdpo_smoke_artifacts.py
    "$output_dir"
    --expected-topk
    "$expected_topk"
    --require-provenance
    --expected-provenance-mode
    "$mode"
    --expected-provenance-config
    "$config"
  )
  if [[ "$require_ema" -eq 1 ]]; then
    verify_cmd+=(--require-ema-teacher)
  fi
  "${verify_cmd[@]}" | tee "$report_file"
  echo "Wrote SDPO smoke verifier report: $report_file"
  require_smoke_verify_report_markers "$report_file" "$label" "$require_ema"
}

require_acceptance_artifact() {
  local action="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "Error: cannot $action SDPO CUDA acceptance proof; missing required artifact: $path" >&2
    exit 2
  fi
  if [[ -f "$path" && ! -s "$path" ]]; then
    echo "Error: cannot $action SDPO CUDA acceptance proof; empty required artifact: $path" >&2
    exit 2
  fi
  if [[ -d "$path" ]] && ! find "$path" -type f -size +0c -print -quit | grep -q .; then
    echo "Error: cannot $action SDPO CUDA acceptance proof; empty required artifact directory: $path" >&2
    exit 2
  fi
}

require_acceptance_proof_entries() {
  local action="$1"
  for required_entry in "${required_proof_entries[@]}"; do
    require_acceptance_artifact "$action" "$output_root/$required_entry"
  done
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
  if [[ -z "$archive_path" ]]; then
    return
  fi
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

require_archive_path_is_regular_file_or_absent() {
  if [[ -z "$archive_path" ]]; then
    return
  fi
  if [[ -L "$archive_path" || ( -e "$archive_path" && ! -f "$archive_path" ) ]]; then
    echo "Error: --archive path exists and is not a regular file: $archive_path" >&2
    exit 2
  fi
}

write_acceptance_archive() {
  local acceptance_mode="$1"
  if [[ -z "$archive_path" ]]; then
    return
  fi
  mkdir -p "$(dirname "$archive_path")"
  for required_entry in "${required_archive_entries[@]}"; do
    require_acceptance_artifact "archive" "$output_root/$required_entry"
  done
  local archive_entries=("${required_archive_entries[@]}")
  for optional_entry in "${optional_acceptance_entries[@]}"; do
    if [[ -e "$output_root/$optional_entry" ]]; then
      archive_entries+=("$optional_entry")
    fi
  done
  COPYFILE_DISABLE=1 tar -C "$output_root" -czf "$archive_path" "${archive_entries[@]}"
  if [[ ! -s "$archive_path" ]]; then
    echo "Error: SDPO CUDA acceptance archive is empty after writing: $archive_path" >&2
    exit 2
  fi
  if ! tar -tzf "$archive_path" >/dev/null; then
    echo "Error: SDPO CUDA acceptance archive is not listable after writing: $archive_path" >&2
    exit 2
  fi
  echo "Verifying SDPO CUDA acceptance archive: $archive_path"
  "${python_runner[@]}" scripts/verify_sdpo_cuda_acceptance_archive.py \
    --expected-acceptance-mode "$acceptance_mode" \
    --expected-git-commit "$(git_commit_sha)" \
    --expected-git-branch "$(git_branch_name)" \
    "$archive_path"
  echo "Wrote SDPO CUDA acceptance archive: $archive_path"
}

if [[ "$no_run" -eq 1 && "$check_config" -eq 0 && -z "$output_root" ]]; then
  echo "Error: --no-run requires --output-root so both completed smoke artifact directories are explicit." >&2
  exit 2
fi
if [[ "$no_run" -eq 1 && "$check_config" -eq 0 && "$clean_output_dir" -eq 1 ]]; then
  echo "Error: --clean-output-dir cannot be combined with --no-run; verification-only mode must preserve existing artifacts." >&2
  exit 2
fi

if [[ -z "$output_root" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  output_root="outputs/sdpo-cuda-acceptance-${timestamp}"
fi

if [[ "$check_config" -eq 1 ]]; then
  echo "Checking live-policy SDPO smoke config..."
  run_smoke_wrapper --check-config
  echo "Checking EMA SDPO smoke config..."
  run_smoke_wrapper --ema --check-config
  echo "SDPO CUDA acceptance config checks passed."
  exit 0
fi

live_output_dir="$output_root/live"
ema_output_dir="$output_root/ema"
live_verify_report_file="$live_output_dir/sdpo_smoke_verify_report.txt"
ema_verify_report_file="$ema_output_dir/sdpo_smoke_verify_report.txt"
require_archive_outside_output_root
require_archive_path_is_regular_file_or_absent

if [[ "$no_run" -eq 1 ]]; then
  mkdir -p "$live_output_dir" "$ema_output_dir"
  run_acceptance_verifier "$live_output_dir" "$live_verify_report_file" live "$live_config" "live-policy" 0 \
    "Verifying existing live-policy SDPO CUDA smoke artifacts..."
  run_acceptance_verifier "$ema_output_dir" "$ema_verify_report_file" ema "$ema_config" "EMA" 1 \
    "Verifying existing EMA SDPO CUDA smoke artifacts..."
  require_acceptance_proof_entries "summarize"
  write_acceptance_summary "no-run"
  write_acceptance_manifest "no-run"
  write_acceptance_archive "no-run"
  echo "SDPO CUDA acceptance artifact verification passed."
  exit 0
fi

echo "Running live-policy SDPO CUDA smoke..."
if [[ "$clean_output_dir" -eq 1 ]]; then
  run_smoke_wrapper --output-dir "$live_output_dir" --clean-output-dir
else
  run_smoke_wrapper --output-dir "$live_output_dir"
fi
run_acceptance_verifier "$live_output_dir" "$live_verify_report_file" live "$live_config" "live-policy" 0 \
  "Re-verifying completed live-policy SDPO CUDA smoke artifacts..."

echo "Running EMA SDPO CUDA smoke..."
if [[ "$clean_output_dir" -eq 1 ]]; then
  run_smoke_wrapper --ema --output-dir "$ema_output_dir" --clean-output-dir
else
  run_smoke_wrapper --ema --output-dir "$ema_output_dir"
fi
run_acceptance_verifier "$ema_output_dir" "$ema_verify_report_file" ema "$ema_config" "EMA" 1 \
  "Re-verifying completed EMA SDPO CUDA smoke artifacts..."

require_acceptance_proof_entries "summarize"
write_acceptance_summary "training"
write_acceptance_manifest "training"
write_acceptance_archive "training"
echo "SDPO CUDA acceptance complete."
echo "Live output: $live_output_dir"
echo "EMA output: $ema_output_dir"
