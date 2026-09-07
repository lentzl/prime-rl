#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dockerfile=$root/experiments/qwen35-2b-document-translation-prime-agent-v1/Dockerfile.kernel
image=${DOCUMENT_TRANSLATION_RUNTIME_IMAGE:-rlm-prime-agent-runtime:0.7.2-beta.495.1.97b994c-node22.19.0-document-translation-v1}

docker build --file "$dockerfile" --tag "$image" "$root"

docker run --rm --interactive --entrypoint /usr/local/bin/python "$image" - <<'PY'
import agent_message
import agent_observe
import ipykernel
import rlm

assert callable(rlm.run)
assert callable(rlm.host_request)
print("document translation Prime Agent kernel image validated")
PY
