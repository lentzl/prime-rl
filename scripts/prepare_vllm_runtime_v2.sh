#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_env=${RUNTIME_V1_ENV:-$root/.venv}
runtime_root=${RUNTIME_V2_ROOT:-/home/ubuntu/rlm/runtime-v2}
runtime_env=${RUNTIME_V2_ENV:-$runtime_root/envs/prime-rl-vllm-0.27.1-cu129}
wheel_dir=$runtime_root/wheels
wheel=$wheel_dir/vllm-0.27.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
wheel_url=https://github.com/vllm-project/vllm/releases/download/v0.27.1/vllm-0.27.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl
wheel_sha256=bf0d52faa2a51e7a01c6856a7a8a2d1307fd0ff711415d34168a67ffac0fa47b
uv_bin=${UV_BIN:-$(command -v uv || true)}

if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
  uv_bin=$HOME/.local/bin/uv
fi
if [[ -z "$uv_bin" ]]; then
  echo "uv executable not found" >&2
  exit 1
fi
if [[ ! -x "$source_env/bin/python" ]]; then
  echo "source environment is incomplete: $source_env" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing runtime preparation while a GPU process is active" >&2
  exit 1
fi
if [[ -e "$runtime_env" ]]; then
  echo "refusing to overwrite existing runtime-v2 environment: $runtime_env" >&2
  exit 1
fi
if [[ "$(realpath "$source_env")" == "$(realpath -m "$runtime_env")" ]]; then
  echo "runtime-v2 environment must be distinct from v1" >&2
  exit 1
fi

mkdir -p "$wheel_dir" "$(dirname "$runtime_env")"
if [[ ! -f "$wheel" ]]; then
  curl -fL --retry 3 --retry-delay 2 -o "$wheel" "$wheel_url"
fi
actual_wheel_sha256=$(sha256sum "$wheel" | awk '{print $1}')
if [[ "$actual_wheel_sha256" != "$wheel_sha256" ]]; then
  echo "runtime-v2 wheel hash mismatch: $actual_wheel_sha256" >&2
  exit 1
fi

partial_env=$runtime_env.partial.$$
if [[ -e "$partial_env" ]]; then
  echo "refusing to overwrite partial environment: $partial_env" >&2
  exit 1
fi
cp -a --reflink=auto "$source_env" "$partial_env"

# A copied venv is usable, but console-script shebangs and activation helpers
# retain the original absolute prefix. Rewrite only generated text files under
# bin; site-packages remain byte-for-byte copied until the vLLM replacement.
while IFS= read -r generated_file; do
  sed -i "s|$source_env|$runtime_env|g" "$generated_file"
done < <(grep -Il "$source_env" "$partial_env"/bin/* 2>/dev/null || true)

"$uv_bin" pip install \
  --python "$partial_env/bin/python" \
  --no-deps \
  --reinstall \
  "$wheel"
"$uv_bin" pip check --python "$partial_env/bin/python"
installed_version=$(
  "$partial_env/bin/python" -c \
    'import importlib.metadata; print(importlib.metadata.version("vllm"))'
)
if [[ "$installed_version" != "0.27.1+cu129" ]]; then
  echo "unexpected installed vLLM version: $installed_version" >&2
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
  echo "runtime-v2 changed packages other than vLLM" >&2
  exit 1
fi

mv "$partial_env" "$runtime_env"
while IFS= read -r generated_file; do
  sed -i "s|$partial_env|$runtime_env|g" "$generated_file"
done < <(grep -Il "$partial_env" "$runtime_env"/bin/* 2>/dev/null || true)
manifest=$runtime_root/RUNTIME-V2.txt
if [[ -e "$manifest" ]]; then
  echo "refusing to overwrite runtime-v2 manifest: $manifest" >&2
  exit 1
fi
{
  printf 'schema=q38-runtime-v2/v1\n'
  printf 'status=prepared_not_qualified\n'
  printf 'prime_rl_commit=%s\n' "$(git -C "$root" rev-parse HEAD)"
  printf 'source_env=%s\n' "$source_env"
  printf 'runtime_env=%s\n' "$runtime_env"
  printf 'vllm_version=%s\n' "$installed_version"
  printf 'vllm_wheel_url=%s\n' "$wheel_url"
  printf 'vllm_wheel_sha256=%s\n' "$wheel_sha256"
  printf 'source_uv_lock_sha256=%s\n' "$(sha256sum "$root/uv.lock" | awk '{print $1}')"
  printf 'v1_packages_sha256=%s\n' "$(sha256sum "$source_packages" | awk '{print $1}')"
  printf 'v2_packages_sha256=%s\n' "$(sha256sum "$candidate_packages" | awk '{print $1}')"
  printf 'prepared_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$manifest"
sha256sum "$manifest" >"$manifest.sha256"
printf 'runtime-v2 environment prepared: %s\n' "$runtime_env"
