# Hübotter SDPO and the Algorithm Abstraction

This note maps Hübotter-style SDPO onto Prime's algorithm abstraction. It is a
working design record for the implementation: the code now has a first-class
`sdpo` algorithm and loss component, but the contribution boundary should still
stay close to Prime's existing per-token stream model.

Contribution provenance: this SDPO stack is based on
`PrimeIntellect-ai/prime-rl` branch `feat/algorithm-abstraction-v1` at
`b20427b6fc11283e59c3397950fc611e31e6d093`, with SDPO-specific commits layered
on top in the fork branch `codex/sdpo-algorithm-abstraction-v1-clean`.

## Current Prime Pieces

The abstraction already has the important generic pieces for distillation-style
training:

- `opd` samples from the live policy, scores the sampled tokens under a teacher
  model, and routes action tokens into the `ref_kl` loss component.
- `opsd` uses the same `ref_kl` loss component, but rebuilds the reference
  scoring prompt with an expert demonstration. This is the SDFT-shaped path.
- The trainer is algorithm-blind. It consumes per-token component streams and
  applies `rl`, `ce`, `ref_kl`, and `sdpo` losses independently.

That means Hübotter-style SDPO should not be a separate trainer mode. It should
look like an `opd`/`opsd` sibling at the orchestration boundary: same
scored-sample abstraction and loss routing, but with feedback-conditioned
teacher contexts and an SDPO-specific distribution payload when sampled-token
`ref_kl` is no longer enough.

## Loss-Level Mapping

The sampled-token SDPO primitive has the per-token form:

```text
(log p_student(token) - log p_teacher(token)).detach() * log p_student(token)
```

The `ref_kl` component uses the teacher gap as the policy-gradient signal:

```text
ref_kl = log p_teacher(token) - log p_student(token)
loss ~= -ref_kl.detach() * importance_ratio
```

At the fully on-policy point where trainer and inference logprobs match, this
has the same gradient direction as Prime's sampled-token `ref_kl` component
with respect to the sampled token logprob. That equivalence is useful for
regression tests and for explaining the relationship between `ref_kl` and
sampled-token SDPO, but it is not the full Hübotter path.

The paper-aligned path distills distributions, not only sampled-token gaps. The
implementation therefore adds a narrow `sdpo` component rather than overloading
`ref_kl`:

- `full_logit_distillation=true` computes KL/JSD-style distillation over a
  shared distribution support.
- `distillation_topk_support="student"` follows the Hübotter top-k path: the
  trainer first exports student-selected top-k token ids, then the orchestrator
  asks the feedback-conditioned teacher to score exactly those ids.
- `distillation_topk_support="teacher"` remains a lighter smoke/ablation path
  where the teacher chooses the support directly.
- `sdpo_weights` plays the role of the self-distillation mask: rollouts without
  a valid hindsight target are kept out of the `sdpo` denominator instead of
  silently training against an unconditioned teacher.

The lower-level loss primitive keeps sampled-token and full-vocabulary
reference cases covered so the math remains tied to the Hübotter/`verl`
lineage. The launchable Prime `rl` integration supports full-logit top-k SDPO:
that path fits Prime's per-token stream boundary and vLLM candidate scoring
without requiring dense vocabulary transport through the orchestrator.

The `verl` path is the author-lineage executable specification; tests here
should guard that the Prime translation preserves those semantics while using
Prime's own component system and runtime ownership boundaries.

Reference-scoring correctness depends on exact token alignment. For `opd` and
`opsd`, the prefill helper must select the logprob for the token being scored at
each position, not an arbitrary top-k entry returned by the serving backend. For
`sdpo`, the stricter condition is support alignment: the student and teacher
logprobs must be evaluated on the same top-k token ids at every trainable
position.

The Prime port makes that alignment an artifact contract, not only an algorithm
convention. For the default `distillation_topk_support="student"` path:

- the SDPO algorithm first prunes the self-distillation mask to tokens with a
  valid hindsight target;
- explicit replay target-position lists must be unique in-range token indices,
  so malformed replay construction fails before teacher scoring rather than
  leaking through Python indexing behavior;
- the trainer preflight export chooses student top-k ids only for those
  nonzero `sdpo_weights`;
- unweighted rows must be absent or all-zero placeholders, so stale dense top-k
  rows cannot survive as accidental targets. This is checked again inside the
  combined trainer loss path before the `sdpo` component slices active tokens,
  so direct `compute_loss` callers cannot bypass the transport-layer guard;
- active rows treat the logprob row as the placeholder signal. Token id `0`
  remains a valid top-k candidate, which matters for `distillation_topk=1`
  tests and avoids baking tokenizer-specific assumptions into the contract;
- every final SDPO record must transport real teacher logprobs at each
  weighted token row, on the same student ids exported by preflight for the
  same step and `sample_id`;
- generated SDPO `sample_id` values include the non-empty env name, percent
  escaped, so default replay identities carry the same env dimension the
  verifier later requires;
- matching preflight/final records must also describe the same positioned token
  sequence through a non-empty env name, token ids, position ids, loss mask,
  and temperatures.
- strict student-support hydration rejects preflight records, export records,
  or active samples that lack a non-empty env name; legacy order-preserving
  loading remains permissive only outside the strict SDPO preflight path.
- matched final teacher logprobs must differ from the trainer-forward student
  logprobs on at least one matched row, so the smoke evidence proves that
  feedback-conditioned teacher rescoring ran instead of merely copying the
  preflight student scores.
- reference-smoke final rows must carry rollout-IS ratio evidence
  (`log_importance_ratio`, `importance_ratio`, and `prob_delta`) so the
  token-level truncated rollout correction path is exercised, not only
  configured.
- reference-smoke final rows must also carry `sdpo_rollout_is_weights` at every
  weighted SDPO token row, matching token-level truncated rollout-IS under the
  configured threshold, so the exported final batch proves the loss consumed
  the reference smoke's weighting path rather than only logging ratio
  diagnostics. The smoke verifier reports this as `rollout_is_weight_rows` and
  `rollout_is_weight_token_rows`.
- `trainer.sdpo_loss.rollout_is_batch_normalize=true` is intentionally not a
  launchable `sdpo` algorithm setting yet. The lower-level loss primitive keeps
  the knob for reference coverage, but the combined trainer evaluates packed
  sequences one at a time, so batch normalization there would be sequence-local
  rather than normalized over the global SDPO component batch.
  The primitive follows the Hübotter rollout-correction helper's normalization
  rule: token-level rollout-IS normalizes over active tokens, while
  sequence-level rollout-IS normalizes over active sequences so longer
  completions do not get extra weight in the normalization factor.

This invariant is checked at multiple boundaries: token export masks unweighted
rows, strict preflight hydration rejects leaked dense rows, the packer rejects
malformed incoming samples before batching, and the smoke verifier proves that
preflight-only student support rows line up with final transported teacher
support rows, while the final batch also exports the ratio evidence needed by
the configured rollout-IS path. The preflight trainer pass itself is
forward/export-only: it exists to export student-selected support and must not
perform loss/backward, optimizer or scheduler updates, EMA teacher updates, or
trainer-progress advancement. Step-boundary broadcasts/checkpoints can still
happen before the trainer sees whether the next batch is preflight-only; those
artifacts belong to earlier optimizer updates and are what the EMA smoke
verifier matches against. If a preflight batch re-enters the same
`progress.step`, step-boundary maintenance is reused rather than repeated, so a
single logical trainer step cannot duplicate broadcasts/checkpoints before the
hydrated final batch arrives.

## Current Proof Boundary

The local unit suite now covers the main Prime-native contracts independently:

- loss constants match the fixed Hübotter/`verl` reference cases for
  sampled-token SDPO, full-distribution KL/JSD variants, top-k tail handling,
  clipping, and rollout-importance weighting;
- sample packing rejects malformed token identity, malformed SDPO component
  identity, malformed optional position identity, malformed SDPO component
  weights, stale or partial top-k support, duplicate supported ids, and support
  without active `sdpo_weights`;
- preflight hydration rejects ambiguous duplicate JSON object keys, requires
  inactive records to carry no support payload, strict sample ids, matching
  env/token/position identity, and same-step preflight/final support matching
  for the student-selected top-k path;
- token export rejects malformed sequence lengths, token/position identity,
  masks, component weights, stale teacher support on preflight records, missing
  teacher support on final records, missing rollout-IS weights in strict smoke
  mode, and non-placeholder support on unweighted rows;
- the combined trainer loss rejects malformed top-k geometry and
  non-placeholder top-k rows outside the active SDPO component, while preserving
  inert placeholder rows for packed non-SDPO tokens;
- trainer preflight mode is a whole-step mode: every micro-batch must opt into
  `preflight_only`, token export must be enabled, and the pass exits before
  loss/backward/optimizer/scheduler/EMA/progress updates;
- the `rl` entrypoint writes a separate local SDPO teacher inference config
  for EMA runs while preserving sticky `X-Session-ID` routing in the
  orchestrator config;
- smoke artifact verification proves stable schema-v2 token exports,
  non-empty env names on active SDPO records, preflight-only student support,
  final teacher support on the same student ids, at least one same-support row
  with teacher logprobs distinct from the trainer-forward student logprobs,
  rollout-IS ratio evidence, token-level truncated `sdpo_rollout_is_weights`,
  and same-step EMA teacher broadcasts for post-initial matching-support steps. The
  distinct-logprob check is intentionally an evidence check, not an every-row
  requirement: exact equality on an individual row is possible in the
  live-policy smoke, but a run where all transported teacher logprobs are
  copied from the student path does not prove teacher-conditioned rescoring.

The smoke wrapper adds a local configuration gate before any expensive run. It
resolves the selected preset and fails if the reference knobs have drifted:
student-selected top-k support, `model = "policy"`, the three Hübotter prompt
templates, full-logit top-k SDPO with tail mass, sampled-token clipping,
token-level rollout-IS evidence, token exports, unfused trainer logits, and the
expected live-policy or EMA teacher mode. After a training smoke completes, the
wrapper also writes `sdpo_smoke_provenance.txt` into the output directory. That
file records the smoke mode, config path, expected top-k width, resolved
reference SDPO knobs, git commit, branch, runner commands, `git status --short`,
and SHA-256 fingerprints of the tracked diff, staged diff, and untracked-file
content manifest. The resolved knobs include student top-k support, live/EMA
teacher regularization, self-success masking, batch-order successful-sibling
selection, feedback inclusion, `template_target = "first_user"`, and trainer
SDPO loss settings. It also embeds the readable untracked-file manifest, with
one SHA-256/path row per file, so a dirty-tree CUDA proof can be audited after
download. That lets a successful CUDA run be traced back to the exact working
tree and reference contract that produced its artifacts, even while the
contribution is still being matured before a final commit.

The remaining proof is operational rather than architectural: the SDPO smoke
presets still need to be run on a Linux CUDA/vLLM machine. For the exact
fresh-box procedure, see `docs/sdpo-cuda-acceptance-runbook.md`. The acceptance
gates are:

```bash
scripts/run_sdpo_cuda_acceptance.sh --check-config
scripts/run_sdpo_cuda_acceptance.sh --output-root outputs/sdpo-cuda-acceptance --clean-output-dir
scripts/run_sdpo_cuda_acceptance.sh --no-run --output-root outputs/sdpo-cuda-acceptance
scripts/run_sdpo_cuda_acceptance.sh --no-run --output-root outputs/sdpo-cuda-acceptance --archive outputs/sdpo-cuda-acceptance-proof.tar.gz

# Expanded form:
scripts/run_sdpo_smoke_and_verify.sh --check-config
scripts/run_sdpo_smoke_and_verify.sh --ema --check-config
scripts/run_sdpo_smoke_and_verify.sh --clean-output-dir --output-dir outputs/sdpo-smoke
scripts/run_sdpo_smoke_and_verify.sh --ema --clean-output-dir --output-dir outputs/sdpo-ema-smoke
```

For the actual remote proof run, prefer producing the archive in one pass:

```bash
scripts/start_sdpo_cuda_acceptance_background.sh --preflight-only
scripts/start_sdpo_cuda_acceptance_background.sh
scripts/start_sdpo_cuda_acceptance_background.sh --status

# Equivalent foreground command:
scripts/run_sdpo_cuda_acceptance.sh \
  --output-root outputs/sdpo-cuda-acceptance \
  --clean-output-dir \
  --archive outputs/sdpo-cuda-acceptance-proof.tar.gz
```

After the command succeeds, download `outputs/sdpo-cuda-acceptance-proof.tar.gz`
and keep it with the PR evidence. For example:

```bash
scp USER@HOST:/path/to/prime-rl/outputs/sdpo-cuda-acceptance-proof.tar.gz .
uv run python scripts/verify_sdpo_cuda_acceptance_archive.py \
  --expected-acceptance-mode training \
  sdpo-cuda-acceptance-proof.tar.gz
```

Passing the config-check commands proves only configuration fidelity for the
live and EMA presets. Passing the combined CUDA acceptance command, or its
expanded live and EMA smoke commands, proves that the Prime
orchestration, vLLM prefill APIs, trainer preflight export, teacher rescoring,
final SDPO training batch, rollout importance evidence, and EMA teacher
broadcasts work together in the real runtime. The semantic proof is the verifier
output; `sdpo_smoke_provenance.txt` is the reproducibility breadcrumb for that
proof. The wrapper requires that provenance file for fresh training smokes via
`verify_sdpo_smoke_artifacts.py --require-provenance`, checks its recorded
top-k width against the verifier's `--expected-topk`, pins the expected
live/EMA mode plus preset config path, and rejects resolved SDPO reference-knob
drift such as `successful_demonstration_selection = "highest_reward"` in the
Hübotter reference smoke. In `--require-provenance` mode, it also requires the
commit/branch, runner commands, and source-tree fingerprints, and rejects
placeholder `unknown` / `unavailable` values for the git and hash fields. All
recorded `*_sha256` provenance fields must be lowercase SHA-256 hex digests. It
also recomputes the embedded untracked-file manifest hash and checks it against
the recorded `git_untracked_manifest_sha256`, and requires the bounded
`git status --short` section to include its end marker. The standalone smoke
verifier treats provenance as an unambiguous proof format: duplicate fields,
repeated or nested provenance sections, unterminated sections, and malformed
non-empty metadata lines are rejected instead of interpreted with "last value
wins" semantics. The token-export verifier applies the same principle to JSONL
records by rejecting duplicate JSON object keys before checking sample ids, env
names, support rows, and rollout-IS evidence. The `--require-provenance` flag is
optional so the verifier can still inspect artifact bundles that do not carry
that breadcrumb, but acceptance runs should keep it enabled. The CLI can find
that provenance from the run output directory or nested `run_default`,
`token_exports`, `broadcasts`, step, or rank-file paths. Until those Linux/CUDA
runs pass, the port should be treated as locally well-covered but not yet
upstream-ready.

On rented boxes, `scripts/start_sdpo_cuda_acceptance_background.sh` is the
operational wrapper: it launches the same combined acceptance command under
`nohup`, writes `outputs/sdpo-cuda-acceptance.log`, records
`outputs/sdpo-cuda-acceptance.pid`, and prints the monitor, remote verification,
download, and local verification commands. On a fresh machine, run it with
`--preflight-only` first to check basic host tooling, visible GPU count, and the
combined acceptance config without starting training; the helper runs those
preflights by default before `nohup` unless `--skip-host-preflight` or
`--skip-config-preflight` is passed. Immediately before starting the background
process, it removes any existing regular file at the requested archive path and
refuses non-file/symlink archive paths or archive paths inside the output root. This
keeps a failed fresh run from later appearing successful because `--status`
found and verified a stale archive from an earlier run, and prevents the
background helper from starting a foreground acceptance command that would later
reject a self-including proof tarball path. Re-run it with `--status` to inspect
the recorded PID, process state, archive state, log state, and latest log lines
without starting another run. When the recorded process is no longer running and
a non-empty archive
exists, status mode also runs the offline archive verifier and returns nonzero
for an invalid completed proof tarball. It also returns nonzero when a recorded
process is stopped and no non-empty archive exists, or when an empty archive is
present without a running process, making failed or incomplete overnight runs
automation-visible. A `--preflight-only` call must run at least one of the host
or config preflight checks; asking it to skip both is rejected as a no-op. The
status and post-launch instructions explicitly tell the operator to expect
`raw_artifacts=verified` in verifier output. Use the
foreground command above only when an attached shell is acceptable for the full
run.

The combined acceptance launcher also writes
`sdpo_cuda_acceptance_summary.txt` under the output root after both the
live-policy and EMA checks pass. That file is only a combined success marker;
it points at each smoke's provenance file, verifier report, token-export
directory, the EMA broadcast directory, and the requested archive path when one
is provided. The provenance files and saved `sdpo_smoke_verify_report.txt` files
remain the human-readable semantic proof. After training, the combined launcher
re-runs the strict verifier for each smoke and overwrites those reports with
acceptance-owned verification output. Before writing the combined summary or
archive, it checks those reports for the verifier's explicit `Verified SDPO smoke
provenance:`, `Verified SDPO token exports:`, and, for the EMA half,
`Verified SDPO EMA broadcasts:` success markers.
It also requires each mandatory proof file to be non-empty and each mandatory
proof directory to contain at least one non-empty file before writing the
summary, manifest, or archive; empty scaffolding is not accepted as proof.
It also writes `sdpo_cuda_acceptance_manifest.txt` with SHA-256 hashes and byte
sizes for the proof files included under the output root.
Pass `--archive /path/to/proof.tar.gz` after a successful training run or during
`--no-run` re-verification to bundle the summary, per-smoke provenance files,
the manifest, verifier reports, token exports, EMA broadcasts, and resolved
config/control metadata when present for download from a remote CUDA box. The
wrapper requires that archive path to live outside `--output-root`, so the
tarball cannot overwrite or self-include the proof artifacts it is meant to
preserve. It also rejects an archive path that already exists as a non-regular
file or symlink, plus any archive that is empty, cannot be listed, or fails the
offline archive verifier after writing. The offline verifier re-checks the summary's
recorded archive path is outside the recorded output root. The wrapper passes
the mode it just wrote to the offline verifier, so a training archive must
verify as `training` and a `--no-run` re-verification archive must verify as
`no-run`. After download,
`verify_sdpo_cuda_acceptance_archive.py` checks the required safe regular-file
proof members, verifier success markers, summary/manifest acceptance mode
agreement, non-placeholder summary git identity, live/EMA provenance
mode/config/top-k fields, resolved reference SDPO knobs, provenance source-tree
fingerprints including the embedded untracked-file manifest hash, matching
summary/provenance commit and branch identity across both smoke runs, and
matching live/EMA tracked-diff, staged-diff, untracked-manifest, and runner
fingerprints so a combined archive cannot mix evidence produced from different
local source states. It also checks positive smoke-report evidence counters for
matched support, distinct teacher logprobs, rollout-IS ratio/weight rows, and
EMA teacher broadcasts, and every manifest hash/size against the archived bytes.
It then extracts the already
path/type-checked archive into a temporary directory and re-runs the strict
smoke artifact verifier on the archived `live/` and `ema/` trees, so a forged
success report cannot hide malformed raw token exports, reference-knob drift, or
incomplete EMA teacher broadcasts. The saved verifier-report counters must also
match the recomputed raw artifact counters. It rejects archive members other
than regular files and directories, and rejects duplicate archive member paths.
It also rejects ambiguous proof metadata: duplicate summary or manifest header
fields, duplicate provenance fields or section markers, repeated verifier
report markers, and duplicate counters inside
verifier report lines.
The success line includes `raw_artifacts=verified` plus recomputed live/EMA
token-export counters and EMA teacher-step evidence from the archived files.
For PR evidence, pass
`--expected-acceptance-mode training`; verifying without that flag is useful for
integrity checks, but it intentionally also accepts `--no-run` archives and will
print the recorded `acceptance_mode` in its success line.

The broad local SDPO gate used during development is wrapped by:

```bash
scripts/run_sdpo_local_validation.sh
```

It runs the pytest slice below, then shell syntax checks, Python bytecode
compilation for the verifier CLIs, direct live-policy and EMA smoke
`--check-config` guards, the combined CUDA acceptance launcher's
`--check-config` guard, targeted Ruff lint/format checks over the SDPO
contribution files, an explicit whitespace/final-newline scan over tracked and
untracked SDPO contribution files, and `git diff --check`.

The current local evidence is that this full wrapper passes on macOS:
`1339 passed, 15 warnings`, followed by passing script syntax checks, live-policy
and EMA smoke config checks, CUDA acceptance config checks, Ruff lint, Ruff
format, whitespace, and diff hygiene.

The expanded pytest command is:

```bash
ENV_PYTHONPATH="$(find deps/verifiers/environments -mindepth 1 -maxdepth 1 -type d | paste -sd: -)"
PYTHONPATH=".:src:packages/prime-rl-configs/src:deps/pydantic-config/src:deps/verifiers:deps/renderers:deps/research-environments:${ENV_PYTHONPATH}" \
  uvx --python 3.12 --from pytest \
    --with pytest-asyncio --with psutil --with setproctitle \
    --with pydantic --with loguru --with torch --with torchdata \
    --with numpy --with pandas --with transformers --with datasets \
    --with jaxtyping --with beartype --with tomli --with tomli-w \
    --with rich --with orjson --with anthropic --with openai \
    --with tenacity --with requests --with aiohttp --with wandb \
    --with msgspec --with pyzmq \
    pytest \
      tests/unit/orchestrator/test_algorithms.py \
      tests/unit/orchestrator/test_batch.py \
      tests/unit/orchestrator/test_prefill_logprobs.py \
      tests/unit/orchestrator/test_sdpo_preflight.py \
      tests/unit/orchestrator/test_sdpo_student_support.py \
      tests/unit/orchestrator/test_watcher.py \
      tests/unit/test_configs.py \
      tests/unit/test_rl_entrypoint.py \
      tests/unit/train/rl/test_data.py \
      tests/unit/train/rl/test_filesystem_broadcast.py \
      tests/unit/train/rl/test_packer.py \
      tests/unit/train/rl/test_sdpo_export_verify.py \
      tests/unit/train/rl/test_sdpo_component_loss.py \
      tests/unit/train/rl/test_sdpo_loss.py \
      tests/unit/train/rl/test_sdpo_smoke_script.py \
      tests/unit/train/rl/test_sdpo_student_topk_support.py \
      tests/unit/train/rl/test_sdpo_teacher.py \
      tests/unit/train/rl/test_sdpo_train_support.py \
      tests/unit/train/rl/test_token_export.py \
      tests/unit/train/test_ckpt.py \
      tests/unit/train/test_optim.py \
      tests/unit/transport \
      -q
```

On macOS, full `tests/unit` collection is not a substitute for the Linux locked
environment: unrelated model/inference/SFT tests require Linux-side dependencies
such as `torchtitan`, `vllm`, `torchdata`, and Prime CLI packages. Those
collection failures are outside the SDPO proof boundary; run the full suite in
Prime's normal Linux environment when preparing an upstream PR.

## Reference Constants

The loss tests use fixed numeric constants rather than importing the
author-lineage implementation at test time. That keeps Prime's unit suite
self-contained while still anchoring the translation to executable reference
behavior. The current constants are based on `lasgroup/SDPO@c52586b`,
specifically `verl.trainer.ppo.core_algos.compute_self_distillation_loss` and
`verl.trainer.ppo.rollout_corr_helper.compute_rollout_correction_weights` from
the Hübotter implementation lineage. They cover the minimal semantic branches
that matter for the trainer primitive:

- sampled-token SDPO and its self-distillation mask
- sampled-token importance clipping
- full-distribution forward KL, reverse KL, and JSD interpolation
- top-k distillation with the tail-mass bucket
- token-level and sequence-level rollout-importance truncation
- token-level and sequence-level rollout-importance batch-normalization
  semantics

The copied constants are generated from the local RLM provenance fixture
(`scripts/generate_sdpo_reference_constants.py` writing
`docs/prime-rl-sdpo-reference-constants.json`) before being copied into
`tests/unit/train/rl/sdpo_reference_cases.py`. The primitive loss tests and
trainer-component aggregation tests consume that shared helper rather than
importing constants from each other. The rollout-importance case follows the
`verl` loss boundary by passing explicit `rollout_is_weights`; Prime tests its
own helper for computing those weights separately, then checks that the SDPO
loss consumes them with clipping, top-k tail mass, and JSD enabled.
The RLM JSON fixture also keeps those direct rollout helper cases separate under
`rollout_is_weight_cases`, so the copied constants preserve the same boundary:
loss consumption and rollout-weight construction are related but independently
tested.

The constants are intentionally small tensor cases. They are not a statistical
test of training quality and they do not prove the orchestrator's hindsight
construction. They prove that the Prime loss primitive matches the reference
math on representative deterministic inputs, while the orchestrator, transport,
and smoke-artifact tests cover the Prime-specific wiring around that primitive.

## Algorithm-Level Mapping

Hübotter-style SDPO is different from plain `opd` and from `opsd` mainly in how
the teacher context is constructed:

- `opd` scores the sampled completion under the original rollout context and a
  separate teacher model.
- `opsd` scores the sampled completion under a demo-conditioned prompt.
- SDPO needs a hindsight/feedback-conditioned self-teacher context: the policy
  first acts, the environment returns feedback, and the teacher distribution is
  obtained from the policy family conditioned on that feedback.

In Prime terms, SDPO is an `opd`/`opsd`-family algorithm whose `score_batch`
hook rebuilds the reference prefix from rollout metadata such as:

- original prompt
- sampled rollout or failed attempt
- environment feedback
- optional successful previous rollout
- sibling/group context

The prompt construction follows the reference three-template shape: a
successful-demonstration `solution_template`, an environment `feedback_template`,
and an outer `template` that combines the original prompt with the prebuilt
`successful_solution_block` and `feedback_block`. The defaults match the
Hübotter YAML spacing while keeping the section templates configurable for
environment-specific natural feedback.

Like `opsd`, SDPO exposes `template_target = "first_user" | "last_user"` for the
user message that receives the rewritten prompt. The default is `first_user`,
which keeps later user-role feedback messages intact in Prime multi-turn
traces. `last_user` is available for raw prompt-style reprompting when the
latest user message is the original question, matching the boundary used by the
vendored single-turn `verl` path before appending the sampled response tokens
for teacher scoring.

One fidelity distinction is intentional: Table 2 in the paper says a successful
original attempt is passed as the correct solution for that same attempt. The
active Hübotter/`verl` training YAML instead sets
`dont_reprompt_on_self_success: True`, so a successful rollout is trained only
when a different successful sibling can provide the solution block; otherwise
its SDPO target is masked out. Prime follows that executable reference config
by default, while keeping `dont_reprompt_on_self_success=false` available for
the literal paper-template behavior.

The presence of a successful sibling rollout is the semantic signal. Prime
therefore preserves the decoded demonstration text exactly after any configured
`<think>...</think>` removal, including an empty or whitespace-only string, and
still treats that sibling as the `Correct solution` source. This keeps "no
successful solution" (`None`) distinct from "successful solution with empty
decoded text" and preserves the reference rule that feedback is skipped when a
solution is available.

When several successful siblings are available, the vendored Hübotter/`verl`
code selects the first successful sample in batch order. Prime follows that
behavior by default with
`successful_demonstration_selection = "batch_order"`. The optional
`"highest_reward"` setting is available as a Prime-native shaped-reward
ablation, where preferring the strongest successful sibling is more meaningful
than whichever candidate arrived first.

`opsd` is still the closest existing mechanism: it rewrites a prompt, scores the
sampled completion under that reference context, and scatters the result back
onto the original sample. The SDPO gap is what `opsd` deliberately does not own:
sibling selection, failure-feedback selection, top-k support construction, and
self-distillation teacher lifecycle.

The default `opsd` setting is single-step, which is a good fit for SDFT-style
prompt/response tasks. Multi-turn scoring is possible, but feedback-conditioned
SDPO owns the hindsight object at sampled-turn granularity. In a multi-turn
trace, sampled assistant tokens are interleaved with environment observations,
user feedback, tool outputs, and later sampled assistant turns. A
feedback-conditioned SDPO scorer must decide which context each sampled turn is
scored under and preserve that stepwise alignment. Simply taking all trainable
tokens and scoring them after one rewritten prompt would lose the observations
between turns.

Even in the single-turn setting, Prime selects the trainable branch rather than
assuming the first trace leaf is the sampled branch. Verifier traces can contain
non-trainable side leaves, and those must not become the SDPO replay target.

The `sdpo` algorithm therefore supports two fidelity levels:

- Single-turn scoring, matching prompt/response environments where the
  hindsight target is attached to the completed rollout.
- Optional multi-turn scoring, where each sampled assistant segment is scored
  under a rewritten prefix while preserving the intervening observations before
  that segment.

## Review Shape

A review-friendly upstream contribution can still keep the branch pieces
separable:

1. The narrow `sdpo` loss primitive, thin component wrapper, and reference
   constants do not require Prime to import the author-lineage repo.
2. The `compute_loss` aggregation keeps component routing weights separate from
   rollout-importance weights.
3. Token export / hydration for student-selected top-k support is a generic
   per-token stream seam.
4. The `sdpo` algorithm is a feedback-conditioned reference scorer that reuses
   Prime's algorithm abstraction rather than introducing a trainer mode.
5. EMA teacher regularization stays behind an explicit live-teacher endpoint
   and filesystem broadcast path.
6. The smoke artifact verifier proves the student-support preflight, teacher
   rescoring, and final SDPO batch agree on step, sample identity, non-empty
   env-aware sample signature, token row, top-k width, and rollout-IS ratio
   evidence before calling the path end-to-end. For EMA teacher smoke runs,
   every post-initial matching-support export step should also have the
   same-step `sdpo_teacher` filesystem broadcast.
7. Environment-specific replay construction should stay in `verifiers` or task
   packages unless Prime already has a matching generic abstraction.

The CUDA smoke presets, acceptance wrapper, provenance files, and archive
builder are proof tooling for this branch. They make the end-to-end contract
auditable while the path is validated. For upstream review, the same work can
be presented as smaller PRs: loss primitive plus reference constants, trainer
component aggregation, student-support stream seam, algorithm runtime, EMA
teacher lifecycle, and smoke/acceptance tooling.

## Working Hypothesis

The core SDPO contribution should live at the reference-scoring/replay boundary
plus a narrow trainer loss component, not in a separate trainer mode.

In other words: preserve Prime's algorithm abstraction, reuse the existing
component machinery wherever it matches the paper, add only the SDPO-specific
distribution payload that the paper requires, and make the long-term research
frontier the construction of high-quality hindsight-conditioned teacher
contexts.
