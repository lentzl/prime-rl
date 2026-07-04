#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the broad local SDPO validation gate.

Usage:
  scripts/run_sdpo_local_validation.sh [--skip-hygiene]

Options:
  --skip-hygiene  Run only the broad pytest slice, skipping shell/Python syntax
                  checks, ShellCheck, smoke config checks, Ruff, and whitespace
                  checks.
  -h, --help      Show this help.

This is the local, Mac-friendly validation gate for the SDPO port. It adds the
editable verifier environment packages to PYTHONPATH, uses an explicit uvx
runner so the Linux-only lockfile does not block local confidence checks, and
checks both the direct smoke config paths and the combined CUDA acceptance
wrapper's config path.
EOF
}

run_hygiene=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-hygiene)
      run_hygiene=0
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

env_pythonpath=""
if [[ -d deps/verifiers/environments ]]; then
  env_pythonpath="$(find deps/verifiers/environments -mindepth 1 -maxdepth 1 -type d | paste -sd: -)"
fi

base_pythonpath=".:src:packages/prime-rl-configs/src:deps/pydantic-config/src:deps/verifiers:deps/renderers:deps/research-environments"
validation_pythonpath="$base_pythonpath"
if [[ -n "$env_pythonpath" ]]; then
  validation_pythonpath="${validation_pythonpath}:${env_pythonpath}"
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
  validation_pythonpath="${validation_pythonpath}:${PYTHONPATH}"
fi

read -r -a pytest_runner <<< "${SDPO_LOCAL_VALIDATION_PYTEST_RUNNER:-uvx --python 3.12 --from pytest --with pytest-asyncio --with psutil --with setproctitle --with pydantic --with loguru --with torch --with torchdata --with numpy --with pandas --with transformers --with datasets --with jaxtyping --with beartype --with tomli --with tomli-w --with rich --with orjson --with anthropic --with openai --with tenacity --with requests --with aiohttp --with wandb --with msgspec --with pyzmq pytest}"
read -r -a python_runner <<< "${SDPO_LOCAL_VALIDATION_PYTHON_RUNNER:-uvx --python 3.12 --with pytest-asyncio --with psutil --with setproctitle --with pydantic --with loguru --with torch --with torchdata --with numpy --with pandas --with transformers --with datasets --with jaxtyping --with beartype --with tomli --with tomli-w --with rich --with orjson --with anthropic --with openai --with tenacity --with requests --with aiohttp --with wandb --with msgspec --with pyzmq python}"
read -r -a shellcheck_runner <<< "${SDPO_LOCAL_VALIDATION_SHELLCHECK_RUNNER:-uvx --from shellcheck-py shellcheck}"
read -r -a ruff_runner <<< "${SDPO_LOCAL_VALIDATION_RUFF_RUNNER:-uvx --from ruff==0.13.0 ruff}"
read -r -a smoke_runner <<< "${SDPO_LOCAL_VALIDATION_SMOKE_RUNNER:-scripts/run_sdpo_smoke_and_verify.sh}"
read -r -a acceptance_runner <<< "${SDPO_LOCAL_VALIDATION_ACCEPTANCE_RUNNER:-scripts/run_sdpo_cuda_acceptance.sh}"

shell_files=(
  scripts/run_sdpo_cuda_acceptance.sh
  scripts/run_sdpo_local_validation.sh
  scripts/run_sdpo_smoke_and_verify.sh
  scripts/start_sdpo_cuda_acceptance_background.sh
)

tests=(
  tests/unit/orchestrator/test_algorithms.py
  tests/unit/orchestrator/test_batch.py
  tests/unit/orchestrator/test_prefill_logprobs.py
  tests/unit/orchestrator/test_sdpo_preflight.py
  tests/unit/orchestrator/test_sdpo_student_support.py
  tests/unit/orchestrator/test_watcher.py
  tests/unit/test_configs.py
  tests/unit/test_rl_entrypoint.py
  tests/unit/train/rl/test_data.py
  tests/unit/train/rl/test_filesystem_broadcast.py
  tests/unit/train/rl/test_packer.py
  tests/unit/train/rl/test_sdpo_export_verify.py
  tests/unit/train/rl/test_sdpo_component_loss.py
  tests/unit/train/rl/test_sdpo_loss.py
  tests/unit/train/rl/test_sdpo_smoke_script.py
  tests/unit/train/rl/test_sdpo_student_topk_support.py
  tests/unit/train/rl/test_sdpo_teacher.py
  tests/unit/train/rl/test_sdpo_train_support.py
  tests/unit/train/rl/test_token_export.py
  tests/unit/train/test_ckpt.py
  tests/unit/train/test_optim.py
  tests/unit/transport
)

ruff_files=(
  packages/prime-rl-configs/src/prime_rl/configs/algorithm.py
  packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py
  packages/prime-rl-configs/src/prime_rl/configs/rl.py
  packages/prime-rl-configs/src/prime_rl/configs/trainer.py
  scripts/verify_sdpo_smoke_artifacts.py
  scripts/verify_sdpo_cuda_acceptance_archive.py
  scripts/verify_sdpo_token_exports.py
  src/prime_rl/entrypoints/rl.py
  src/prime_rl/orchestrator/algo/__init__.py
  src/prime_rl/orchestrator/algo/base.py
  src/prime_rl/orchestrator/algo/custom.py
  src/prime_rl/orchestrator/algo/echo.py
  src/prime_rl/orchestrator/algo/grpo.py
  src/prime_rl/orchestrator/algo/opd.py
  src/prime_rl/orchestrator/algo/opsd.py
  src/prime_rl/orchestrator/algo/routing.py
  src/prime_rl/orchestrator/algo/sdpo.py
  src/prime_rl/orchestrator/envs.py
  src/prime_rl/orchestrator/orchestrator.py
  src/prime_rl/orchestrator/sdpo_preflight.py
  src/prime_rl/orchestrator/sdpo_sample_identity.py
  src/prime_rl/orchestrator/sdpo_student_support.py
  src/prime_rl/orchestrator/utils.py
  src/prime_rl/orchestrator/watcher.py
  src/prime_rl/trainer/batch.py
  src/prime_rl/trainer/ckpt.py
  src/prime_rl/trainer/optim.py
  src/prime_rl/trainer/rl/broadcast/__init__.py
  src/prime_rl/trainer/rl/broadcast/base.py
  src/prime_rl/trainer/rl/broadcast/filesystem.py
  src/prime_rl/trainer/rl/broadcast/nccl.py
  src/prime_rl/trainer/rl/data.py
  src/prime_rl/trainer/rl/loss.py
  src/prime_rl/trainer/rl/packer.py
  src/prime_rl/trainer/rl/sdpo_export_verify.py
  src/prime_rl/trainer/rl/sdpo_loss.py
  src/prime_rl/trainer/rl/sdpo_teacher.py
  src/prime_rl/trainer/rl/sdpo_train_support.py
  src/prime_rl/trainer/rl/token_export.py
  src/prime_rl/trainer/rl/train.py
  src/prime_rl/transport/filesystem.py
  src/prime_rl/transport/sdpo.py
  src/prime_rl/transport/types.py
  src/prime_rl/transport/zmq.py
  tests/unit/orchestrator/test_algorithms.py
  tests/unit/orchestrator/test_batch.py
  tests/unit/orchestrator/test_prefill_logprobs.py
  tests/unit/orchestrator/test_sdpo_preflight.py
  tests/unit/orchestrator/test_sdpo_student_support.py
  tests/unit/orchestrator/test_watcher.py
  tests/unit/test_configs.py
  tests/unit/test_rl_entrypoint.py
  tests/unit/train/rl/test_data.py
  tests/unit/train/rl/test_filesystem_broadcast.py
  tests/unit/train/rl/test_packer.py
  tests/unit/train/rl/test_sdpo_export_verify.py
  tests/unit/train/rl/test_sdpo_component_loss.py
  tests/unit/train/rl/test_sdpo_loss.py
  tests/unit/train/rl/test_sdpo_smoke_script.py
  tests/unit/train/rl/test_sdpo_student_topk_support.py
  tests/unit/train/rl/test_sdpo_teacher.py
  tests/unit/train/rl/test_sdpo_train_support.py
  tests/unit/train/rl/test_token_export.py
  tests/unit/train/rl/sdpo_reference_cases.py
  tests/unit/train/test_ckpt.py
  tests/unit/train/test_optim.py
  tests/unit/transport/test_filesystem.py
  tests/unit/transport/test_sdpo.py
  tests/unit/transport/test_zmq.py
)

whitespace_files=(
  configs/debug/algorithms/README.md
  configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml
  configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml
  docs/algorithms.md
  docs/sdpo-huebotter-mapping.md
  scripts/run_sdpo_cuda_acceptance.sh
  scripts/run_sdpo_local_validation.sh
  scripts/run_sdpo_smoke_and_verify.sh
  scripts/start_sdpo_cuda_acceptance_background.sh
  skills/configs/SKILL.md
  "${ruff_files[@]}"
)

echo "Running broad local SDPO validation gate..."
PYTHONPATH="$validation_pythonpath" "${pytest_runner[@]}" "${tests[@]}" -q

if [[ "$run_hygiene" -eq 1 ]]; then
  echo "Running SDPO script syntax checks..."
  bash -n "${shell_files[@]}"
  echo "Running SDPO ShellCheck checks..."
  "${shellcheck_runner[@]}" "${shell_files[@]}"
  PYTHONPATH="$validation_pythonpath" "${python_runner[@]}" -m py_compile \
    scripts/verify_sdpo_smoke_artifacts.py \
    scripts/verify_sdpo_cuda_acceptance_archive.py \
    scripts/verify_sdpo_token_exports.py
  echo "Running SDPO smoke config checks..."
  PYTHONPATH="$validation_pythonpath" SDPO_SMOKE_PYTHON_RUNNER="${python_runner[*]}" \
    "${smoke_runner[@]}" --check-config
  PYTHONPATH="$validation_pythonpath" SDPO_SMOKE_PYTHON_RUNNER="${python_runner[*]}" \
    "${smoke_runner[@]}" --ema --check-config
  PYTHONPATH="$validation_pythonpath" SDPO_SMOKE_PYTHON_RUNNER="${python_runner[*]}" \
    "${acceptance_runner[@]}" --check-config
  echo "Running SDPO Ruff checks..."
  "${ruff_runner[@]}" check "${ruff_files[@]}"
  "${ruff_runner[@]}" format --check "${ruff_files[@]}"
  echo "Running SDPO whitespace checks..."
  PYTHONPATH="$validation_pythonpath" "${python_runner[@]}" - "${whitespace_files[@]}" <<'PY'
from pathlib import Path
import sys

failed = False
seen = set()
for raw_path in sys.argv[1:]:
    if raw_path in seen:
        continue
    seen.add(raw_path)
    path = Path(raw_path)
    if not path.exists():
        continue
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        print(f"{path}: missing final newline", file=sys.stderr)
        failed = True
    for line_number, line in enumerate(data.splitlines(), start=1):
        if line.rstrip(b" \t") != line:
            print(f"{path}:{line_number}: trailing whitespace", file=sys.stderr)
            failed = True

if failed:
    raise SystemExit(1)
PY
  git diff --check
fi

echo "SDPO local validation gate passed."
