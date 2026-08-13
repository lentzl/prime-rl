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

case "$(uname -m)" in
  x86_64|amd64)
    router_wheel=https://github.com/PrimeIntellect-ai/router/releases/download/v0.1.26/vllm_router-0.1.26-cp38-abi3-manylinux_2_28_x86_64.whl
    ;;
  aarch64|arm64)
    router_wheel=https://github.com/PrimeIntellect-ai/router/releases/download/v0.1.26/vllm_router-0.1.26-cp38-abi3-manylinux_2_28_aarch64.whl
    ;;
  *)
    echo "unsupported architecture for pinned vllm-router: $(uname -m)" >&2
    exit 1
    ;;
esac
# Local inference always starts the router, but Prime-RL currently exposes its
# pinned wheel only through the much broader disagg extra. Install just that
# wheel so a mastery host does not pull unrelated disaggregated-serving stacks.
"$uv" pip install --python .venv/bin/python "$router_wheel"
bash scripts/install_prime_agent_mastery_envs.sh

test -x .venv/bin/evaluator
test -x .venv/bin/vllm-router
.venv/bin/python - <<'PY'
import flash_attn
import math_verify

print(f"flash_attn={flash_attn.__version__}")
print(f"math_verify={getattr(math_verify, '__version__', 'installed')}")
PY

echo "Prime Agent mastery host setup verified"
