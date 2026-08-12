#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=${PYTHON:-$root/.venv/bin/python}
uv=${UV:-$HOME/.local/bin/uv}

if [[ ! -x "$python" ]]; then
  echo "project virtualenv is missing; run 'uv sync --extra flash-attn' first" >&2
  exit 1
fi

environments=(
  oolong_synth_v1
  prime_agent_capabilities_v1
  subagent_communication_v1
  subagent_admission_v1
  ownership_invariant_v1
)
install_args=()
for environment in "${environments[@]}"; do
  path="$root/deps/verifiers/environments/$environment"
  if [[ ! -f "$path/pyproject.toml" ]]; then
    echo "missing Prime Agent environment package: $path" >&2
    exit 1
  fi
  install_args+=(--editable "$path")
done

"$uv" pip install --python "$python" --no-deps "${install_args[@]}"

"$python" - <<'PY'
import importlib

packages = (
    "oolong_synth_v1",
    "prime_agent_capabilities_v1",
    "subagent_communication_v1",
    "subagent_admission_v1",
    "ownership_invariant_v1",
)
for package in packages:
    importlib.import_module(package)
print(f"verified {len(packages)} Prime Agent environment packages")
PY
