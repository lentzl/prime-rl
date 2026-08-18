#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dockerfile=$root/experiments/qwen35-27b-procedural-harness-master-v1/prime-agent-runtime.Dockerfile
prime_agent_version=${PRIME_AGENT_VERSION:-0.7.2-beta.495.1.97b994c}
node_version=${PRIME_AGENT_NODE_VERSION:-22.19.0}
image=${PRIME_AGENT_RUNTIME_IMAGE:-rlm-prime-agent-runtime:${prime_agent_version}-node${node_version}}

if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build \
    --file "$dockerfile" \
    --build-arg "NODE_VERSION=$node_version" \
    --build-arg "PRIME_AGENT_VERSION=$prime_agent_version" \
    --tag "$image" \
    "$root"
fi

docker run --rm "$image" sh -c \
  "test -x /var/tmp/vf-prime-agent/$prime_agent_version/bin/prime-agent && \
   test \"\$(/var/tmp/vf-prime-agent/$prime_agent_version/bin/prime-agent --version 2>&1)\" = $prime_agent_version && \
   test \"\$(/var/tmp/vf-node/bin/node --version)\" = v$node_version"

printf '%s\n' "$image"
