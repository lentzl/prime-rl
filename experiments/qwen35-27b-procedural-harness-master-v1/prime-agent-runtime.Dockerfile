FROM python:3.11-slim

ARG NODE_VERSION=22.19.0
ARG PRIME_AGENT_VERSION=0.7.2-beta.495.1.97b994c
ARG PRIME_AGENT_INSTALL_URL=https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev/install.sh

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends ca-certificates curl util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    case "$(uname -m)" in \
        aarch64|arm64) node_arch=arm64 ;; \
        *) node_arch=x64 ;; \
    esac; \
    mkdir -p /var/tmp/vf-node; \
    curl -fsSL \
        "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.gz" \
        | tar -xz -C /var/tmp/vf-node --strip-components=1; \
    /var/tmp/vf-node/bin/node --version | grep -Fx "v${NODE_VERSION}"

ENV PATH="/var/tmp/vf-node/bin:${PATH}"

RUN set -eux; \
    prefix="/var/tmp/vf-prime-agent/${PRIME_AGENT_VERSION}"; \
    NPM_CONFIG_PREFIX="$prefix" \
    PRIME_AGENT_VERSION="$PRIME_AGENT_VERSION" \
    PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL=0 \
    PRIME_AGENT_INSTALLER_PLAIN=1 \
        sh -c "curl -fsSL '${PRIME_AGENT_INSTALL_URL}' | sh"; \
    test -x "$prefix/bin/prime-agent"; \
    test "$("$prefix/bin/prime-agent" --version 2>&1)" = "$PRIME_AGENT_VERSION"

COPY patches/prime-agent/agent-message-default-parent.py /tmp/agent-message-default-parent.py

RUN set -eux; \
    target="/var/tmp/vf-prime-agent/${PRIME_AGENT_VERSION}/lib/node_modules/prime-agent/dist/skills/agent-message/src/agent_message/__init__.py"; \
    test -f "$target"; \
    install -m 0644 /tmp/agent-message-default-parent.py "$target"; \
    grep -F 'receiver_role = "parent"' "$target"; \
    rm /tmp/agent-message-default-parent.py

LABEL ai.rlmlab.prime-agent.version="${PRIME_AGENT_VERSION}" \
      ai.rlmlab.node.version="${NODE_VERSION}" \
      ai.rlmlab.agent-message-default-parent="v1"
