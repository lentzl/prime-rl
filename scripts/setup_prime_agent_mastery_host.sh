#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
uv=${UV:-$HOME/.local/bin/uv}

if [[ ! -x "$uv" ]]; then
  echo "uv is required at $uv (override with UV=/path/to/uv)" >&2
  exit 1
fi

cd "$root"
"$uv" sync --extra flash-attn
bash scripts/install_prime_agent_mastery_envs.sh

test -x .venv/bin/evaluator
.venv/bin/python - <<'PY'
import flash_attn
import math_verify

print(f"flash_attn={flash_attn.__version__}")
print(f"math_verify={getattr(math_verify, '__version__', 'installed')}")
PY

echo "Prime Agent mastery host setup verified"
