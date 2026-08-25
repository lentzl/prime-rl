#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_root=${RUNTIME_V2_ROOT:-/home/ubuntu/rlm/runtime-v2}
source_env=${RUNTIME_V1_ENV:-$root/.venv}
runtime_env=${RUNTIME_V2_ENV:-$runtime_root/envs/prime-rl-vllm-0.27.1-cu129}
wheel=$runtime_root/wheels/vllm-0.27.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
wheel_sha256=bf0d52faa2a51e7a01c6856a7a8a2d1307fd0ff711415d34168a67ffac0fa47b
authorization_url=${RUNTIME_V2_PACKAGE_ONLY_AUTHORIZATION_URL:-}

case "$authorization_url" in
  https://github.com/lentzl/rlm/issues/1#issuecomment-*) ;;
  *)
    echo "an explicit control-channel package-only authorization URL is required" >&2
    exit 1
    ;;
esac
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing package-only finalization while a GPU process is active" >&2
  exit 1
fi
if [[ -e "$runtime_env" ]]; then
  echo "refusing to overwrite runtime-v2 environment: $runtime_env" >&2
  exit 1
fi
mapfile -t partial_environments < <(compgen -G "$runtime_env.partial.*" || true)
if [[ ${#partial_environments[@]} -ne 1 ]]; then
  echo "expected exactly one preserved partial environment; found ${#partial_environments[@]}" >&2
  exit 1
fi
partial_env=${partial_environments[0]}
if [[ "$(sha256sum "$wheel" | awk '{print $1}')" != "$wheel_sha256" ]]; then
  echo "runtime-v2 wheel hash mismatch" >&2
  exit 1
fi

installed_version=$(
  "$partial_env/bin/python" -c \
    'import importlib.metadata; print(importlib.metadata.version("vllm"))'
)
if [[ "$installed_version" != "0.27.1+cu129" ]]; then
  echo "unexpected partial vLLM version: $installed_version" >&2
  exit 1
fi

source_packages=$runtime_root/V1-PACKAGES.tsv
candidate_packages=$runtime_root/V2-PACKAGES.tsv
for interpreter_and_output in \
  "$source_env/bin/python:$source_packages" \
  "$partial_env/bin/python:$candidate_packages"; do
  interpreter=${interpreter_and_output%%:*}
  package_output=${interpreter_and_output#*:}
  "$interpreter" - >"$package_output" <<'PY'
import importlib.metadata

packages = {
    (distribution.metadata["Name"].lower().replace("_", "-"), distribution.version)
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get("Name")
}
for name, version in sorted(packages):
    print(f"{name}\t{version}")
PY
done
if ! diff -u \
  <(grep -v $'^vllm\t' "$source_packages") \
  <(grep -v $'^vllm\t' "$candidate_packages"); then
  echo "package-only candidate changed a distribution other than vLLM" >&2
  exit 1
fi

# Exercise imports used by dense Qwen3.8 before promoting the prefix. This is
# deliberately not a substitute for the three independent GPU cold starts.
"$partial_env/bin/python" - <<'PY'
import vllm._C_stable_libtorch  # noqa: F401
import vllm._moe_C_stable_libtorch  # noqa: F401
import vllm.model_executor.layers.mamba.mamba_mixer2  # noqa: F401
import vllm.model_executor.models.qwen3_5  # noqa: F401
import vllm.model_executor.models.qwen3_next  # noqa: F401
import vllm.v1.worker.gpu_model_runner  # noqa: F401
PY

"${UV_BIN:-$HOME/.local/bin/uv}" pip check \
  --python "$source_env/bin/python" >"$runtime_root/V1-PIP-CHECK.txt" 2>&1 || true
"${UV_BIN:-$HOME/.local/bin/uv}" pip check \
  --python "$partial_env/bin/python" >"$runtime_root/V2-PIP-CHECK.txt" 2>&1 || true

manifest=$runtime_root/RUNTIME-V2.txt
if [[ -e "$manifest" ]]; then
  echo "refusing to overwrite runtime-v2 manifest: $manifest" >&2
  exit 1
fi
mv "$partial_env" "$runtime_env"
while IFS= read -r generated_file; do
  sed -i "s|$partial_env|$runtime_env|g" "$generated_file"
done < <(grep -Il "$partial_env" "$runtime_env"/bin/* 2>/dev/null || true)

{
  printf 'schema=q38-runtime-v2/v1\n'
  printf 'status=package_only_prepared_not_qualified\n'
  printf 'authorization_url=%s\n' "$authorization_url"
  printf 'prime_rl_commit=%s\n' "$(git -C "$root" rev-parse HEAD)"
  printf 'source_env=%s\n' "$source_env"
  printf 'runtime_env=%s\n' "$runtime_env"
  printf 'vllm_version=%s\n' "$installed_version"
  printf 'vllm_wheel_sha256=%s\n' "$wheel_sha256"
  printf 'dependency_policy=package_only_metadata_mismatch_recorded\n'
  printf 'v1_packages_sha256=%s\n' "$(sha256sum "$source_packages" | awk '{print $1}')"
  printf 'v2_packages_sha256=%s\n' "$(sha256sum "$candidate_packages" | awk '{print $1}')"
  printf 'v1_pip_check_sha256=%s\n' "$(sha256sum "$runtime_root/V1-PIP-CHECK.txt" | awk '{print $1}')"
  printf 'v2_pip_check_sha256=%s\n' "$(sha256sum "$runtime_root/V2-PIP-CHECK.txt" | awk '{print $1}')"
  printf 'prepared_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$manifest"
sha256sum "$manifest" >"$manifest.sha256"
echo "package-only runtime-v2 finalized but not GPU-qualified: $runtime_env"
