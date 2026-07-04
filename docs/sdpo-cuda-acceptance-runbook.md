# SDPO CUDA Acceptance Runbook

This runbook records the proof procedure for the Hübotter-style SDPO port. Run
it on a Linux CUDA/vLLM machine, not on macOS. The local Mac validation gate is
useful for development confidence, but the acceptance proof requires the real
Prime runtime, vLLM prefill APIs, trainer preflight export, teacher rescoring,
final SDPO training batches, rollout-IS evidence, and EMA teacher broadcasts.

The contribution branch is stacked on `PrimeIntellect-ai/prime-rl`
`feat/algorithm-abstraction-v1` at
`b20427b6fc11283e59c3397950fc611e31e6d093`. If that upstream branch moves
before opening a PR, re-check the merge base and re-run this acceptance proof
after rebasing or confirming there is no integration drift.

## Fresh Box Setup

Clone the contribution branch:

```bash
git clone --branch codex/sdpo-algorithm-abstraction-v1-clean \
  https://github.com/lentzl/prime-rl.git
cd prime-rl
git rev-parse HEAD
```

The expected branch is `codex/sdpo-algorithm-abstraction-v1-clean`. Record the
commit SHA printed by `git rev-parse HEAD`; the acceptance archive verifier will
later check that the live-policy and EMA smoke provenance agree on commit and
branch identity.

For copy-pasteable verification commands later in the runbook, capture the
exact source identity once:

```bash
EXPECTED_SDPO_BRANCH="$(git branch --show-current)"
EXPECTED_SDPO_COMMIT="$(git rev-parse HEAD)"
```

Initialize submodules:

```bash
git submodule update --init --recursive \
  deps/pydantic-config \
  deps/renderers \
  deps/research-environments \
  deps/verifiers
```

If the machine cannot use GitHub SSH for Prime submodules, override the
submodule URLs locally and retry:

```bash
git config submodule.renderers.url \
  https://github.com/PrimeIntellect-ai/renderers.git
git config submodule.research-environments.url \
  https://github.com/PrimeIntellect-ai/research-environments.git
git config submodule.verifiers.url \
  https://github.com/PrimeIntellect-ai/verifiers.git
git submodule update --init --recursive \
  deps/pydantic-config \
  deps/renderers \
  deps/research-environments \
  deps/verifiers
```

## Preflight

Run host and config preflights before starting a long background job:

```bash
scripts/start_sdpo_cuda_acceptance_background.sh --preflight-only
```

The host preflight checks `uv`, `tar`, hashing support, `nvidia-smi`, visible GPU
count, git branch/commit/status, and disk space. By default it expects at least
3 visible GPUs: policy inference, EMA teacher inference, and trainer. To
intentionally run on a different topology, set `SDPO_ACCEPTANCE_MIN_GPUS`, but
record that deviation with the proof evidence.

The config preflight resolves both SDPO reference presets and should print:

```text
SDPO CUDA acceptance config checks passed.
```

Passing this only proves configuration fidelity. It is not the runtime proof.

## Background Acceptance Run

Start the full training, verification, manifest, and archive flow:

```bash
scripts/start_sdpo_cuda_acceptance_background.sh
```

Default artifacts:

```text
outputs/sdpo-cuda-acceptance/
outputs/sdpo-cuda-acceptance.log
outputs/sdpo-cuda-acceptance.pid
outputs/sdpo-cuda-acceptance-proof.tar.gz
```

Monitor the run:

```bash
scripts/start_sdpo_cuda_acceptance_background.sh --status
tail -f outputs/sdpo-cuda-acceptance.log
```

Status mode returns nonzero if the recorded process has stopped without a
non-empty archive, if the archive is empty, or if the completed archive fails
offline verification.

## Remote Verification

When the background process has stopped and the archive exists, verify it on the
remote box:

```bash
uv run python scripts/verify_sdpo_cuda_acceptance_archive.py \
  --expected-acceptance-mode training \
  --expected-git-branch "$EXPECTED_SDPO_BRANCH" \
  --expected-git-commit "$EXPECTED_SDPO_COMMIT" \
  outputs/sdpo-cuda-acceptance-proof.tar.gz
```

The success line must include:

```text
raw_artifacts=verified
```

This means the archive verifier re-extracted the proof bundle, re-ran strict
smoke artifact verification for the live-policy and EMA runs, checked verifier
report counters against raw token exports, and verified the archive manifest.

## Download And Local Re-Verification

Download the archive:

```bash
scp USER@HOST:/path/to/prime-rl/outputs/sdpo-cuda-acceptance-proof.tar.gz .
```

From a checkout of the same branch and commit, capture the expected identity
again and verify the downloaded archive:

```bash
EXPECTED_SDPO_BRANCH="$(git branch --show-current)"
EXPECTED_SDPO_COMMIT="$(git rev-parse HEAD)"
uv run python scripts/verify_sdpo_cuda_acceptance_archive.py \
  --expected-acceptance-mode training \
  --expected-git-branch "$EXPECTED_SDPO_BRANCH" \
  --expected-git-commit "$EXPECTED_SDPO_COMMIT" \
  sdpo-cuda-acceptance-proof.tar.gz
```

Keep the archive, the remote command output, and the local re-verification
output with PR evidence. Until this Linux CUDA/vLLM archive proof passes, the
SDPO port is locally well-covered but not fully accepted.
