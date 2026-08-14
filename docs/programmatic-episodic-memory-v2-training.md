# Programmatic Episodic Memory v2 training comparison

This lane tests whether demonstration-conditioned on-policy self-distillation
(Prime-RL `algo.type = "opsd"`) teaches natural retrieval policy more robustly
than cloning the same verified demonstrations with plain SFT. It is a bootstrap
experiment inside the broader Prime Agent mastery program, not a replacement
for GRPO or for the frozen Harness Mastery battery.

## Frozen inputs

- Common start: untouched thinking-mode
  `Qwen/Qwen3.5-27B@fc05daec18b0a78c049392ed2e771dde82bdf654`.
  Run 331 remains an independent broad-RL experiment; its Step 4 fast-screen
  candidate is not allowed to confound the algorithm comparison before passing
  the frozen 74-task battery.
- Train split: 1,200 examples across 25 familiar families.
- Familiar heldout: 300 examples; admission samples one frozen instance from
  every family twice (50 traces per arm).
- Semantic OOD: 96 examples across eight unseen families; admission samples one
  frozen instance from every family twice (16 traces per arm).
- Neither heldout split is used for weight updates.
- Demonstrations contain observable assistant tool calls and answers, never
  fabricated `reasoning_content`.

The frozen JSONL SHA-256 values are `9bd50adc54d472897aa385f7cc1c7c8f436abc9c1a535d6247003dc57380aaa5`
for train, `15fc541de65494d2c866bcb3da68654bda8a7b2e526a9808f7193b21d7f4e3fc`
for familiar heldout, and
`0984bfe9b376fc1398a8c82547a2087fe64c4380cbd7b146d79b8766bcc9f2e0`
for semantic OOD. An audit over canonicalized `messages_json`, `tools`, and
`workspace_files_json` found 1,200, 300, and 96 unique full-input fingerprints,
with zero cross-split overlap. Do not use `(family, instance)` as a global task
identifier: familiar heldout deliberately reuses 300 train instance numbers
with different complete inputs.

## Teacher admission

The gate was preregistered while only unconditioned traces existed. Both
conditioned arms must meet every core behavior minimum (0.90 familiar, 0.80
OOD), may not regress any core component by more than 0.05, and must improve
aggregate strict success by at least 0.08 absolute or by at least 0.04 absolute
with at least 50% relative error reduction. Every planned trace must complete
cleanly. Each conditioned/unconditioned pair must also contain the exact same
multiset of frozen semantic task payloads; the identity check excludes only the
system prompt where the demonstration intervention is intentionally applied.
Core components include answer and grounding, retrieve/no-retrieve,
tool validity, bounded retrieval, no repeated cells, stale-note resolution,
context-reset recovery, persistent-index reuse, and current-turn override.

`answer_correct` and strict success enforce the requested exact output contract.
The admission report also records non-gating `expected_value_present`, which
distinguishes a semantically recovered value wrapped in unwanted prose from a
wrong value. This prevents a conditioning-induced concision gain from being
misreported as a retrieval-reasoning gain.

Both admission arms use Verifiers' renderer-backed train client. The conditioned
arm prepends the demonstration as an independent rendered system block before
the unchanged ordinary prompt. This exactly matches Prime-RL OPSD's token-prefix
teacher context; concatenating the hint into the task's existing system message
is not equivalent under the Qwen3.5 chat template and is not admissible evidence.

Run the assessment with:

```bash
.venv/bin/python scripts/assess_programmatic_memory_teacher_admission.py \
  /ephemeral/subagent-rung/evals/336-339-qwen35-27b-memory-v2-teacher-admission-base-r3 \
  --output /ephemeral/subagent-rung/evals/336-339-qwen35-27b-memory-v2-teacher-admission-base-r3/admission.json \
  --require-pass
```

The SDFT launcher repeats this check and refuses to train on a failed or
incomplete admission. The matched SFT/SDFT comparison proceeds only if the
teacher is admitted; running only its SFT arm would answer a different question.

## Matched exposure

The renderer audit measured one training pass as 869,372 rendered sequence
tokens and 113,338 assistant/action tokens. Four passes produce 3,477,488
rendered tokens. With 4,096-token packs and global batch six, the plain-SFT lane
uses 154 updates at about 91.9% packing utilization.

The SDFT lane uses one on-policy rollout per example, group size one, global
batch 24, and 200 updates: exactly 4,800 dispatched episodes or four nominal
passes. Its sampled-token budget is inherently policy-dependent and must be
reported from admitted training samples. Equal examples and epochs are exact;
equal token exposure is not claimed. Checkpoint comparisons must include actual
trainable-token counts, not only optimizer steps.

Both lanes use full-weight BF16 AdamW on six trainer GPUs, learning rate 5e-7,
the same untouched-base start, and epoch-scale checkpoints. SFT trains only observable
assistant/tool actions. SDFT samples the unconditioned live policy and applies
dense demo-conditioned reverse-KL on those student-visited tokens.

## Execution

After all evaluation GPU processes have stopped:

```bash
scripts/run_programmatic_memory_v2_training.sh sft
scripts/run_programmatic_memory_v2_training.sh sdft
```

Run one lane at a time. Evaluate their epoch checkpoints on the unchanged
familiar/OOD environment and the frozen 74-task Harness Mastery battery. Prefer
SDFT only if it improves natural memory behavior at comparable exposure without
retrieval shortcuts or broader harness regressions.

## Admission result

The exact-prefix admission rejected the untouched 27B self-teacher before any
gradients. The unconditioned familiar arm completed all 50 traces cleanly with
`0.200` strict success and `0.780` expected-value presence. On the exact same 34
task/rollout identities observed before the prospective early stop, the
conditioned policy improved strict success from `8/34` (`0.235`) to `20/34`
(`0.588`) and expected-value presence from `27/34` (`0.794`) to `32/34`
(`0.941`). This is real evidence that the demonstration changes the policy in a
useful direction, but it is not a reliable teacher.

With only 16 familiar traces remaining, even a perfect tail could raise strict
success to at most `36/50` (`0.720`), answer correctness to `37/50` (`0.740`),
and grounded correctness to `37/50` (`0.740`). All are below the preregistered
`0.90` familiar minimum. The run therefore stopped before the OOD arms rather
than spending compute on an already impossible admission. The matched SFT/SDFT
jobs remain unstarted, and this outcome must not be described as an SDFT
training failure. The exact result and trace hashes are recorded in
`qwen35-27b-memory-v2-teacher-admission-results-v1.json`.

## Executable causal feedback

The next environment slice gives an unsuccessful request one bounded retry
with a diagnostic that identifies the violated retrieval, persistence,
traceback-repair, output-contract, or event-semantics rule. The diagnostic
never contains the expected answer. Final per-request answers and emitted
feedback are retained in the trace so repairs and unresolved failures remain
auditable.

The four-family untouched-27B smoke completed 4/4 traces without runtime
errors. Two tasks were native strict successes. The other two received
answer-free feedback for missing-history, output-contract, or event-semantics
failures; their unresolved errors remained failures rather than being promoted
by the scorer.

This smoke does **not** admit plain trajectory-level GRPO with feedback retries.
A final repaired reward would be broadcast across the failed pre-feedback
action as well as the repair, contrary to the preregistered routing rule. The
training route must instead keep native-clean GRPO credit separate from
feedback-conditioned SDPO on diagnostically understood failures. No gradients
start until that loss routing is executable and covered by the Huebotter
reference tests.

## Failure-routed SDPO CUDA acceptance

Run 344-r11 qualifies the exact diagnostic-failure path end to end on untouched
thinking-mode Qwen3.5-27B. All six admitted trajectories were strict failures
with one answer-free causal diagnostic, no feedback retry, valid bounded tool
behavior, and no ordinary RL, CE, or reference-KL tokens. The trainer selected
4,733 SDPO tokens and completed separate student and EMA-teacher sparse scoring,
backward, full CPU-offloaded AdamW at `1e-7`, the EMA update, scheduler update,
and a stable 12-shard weight checkpoint.

The step remained finite: mean loss `0.023958`, mean SDPO loss `0.172211`, mean
student/teacher mismatch KL `0.000211`, throughput `103.27` tokens/s, and peak
trainer memory `37.24` GiB. The implementation avoids full `[sequence, vocab]`
logits by selecting exact student top-k support and evaluating those same token
IDs under the teacher in sequence and vocabulary chunks. Distributed replay is
rank-aligned, and the no-reshard LM-head/final-norm state is explicitly restored
before shard-local EMA interpolation.

This is an engineering admission, not evidence that the one-step checkpoint
learned useful memory behavior. The checkpoint is neither a teacher nor a
continuation candidate. It admits the next controlled experiment: ordinary GRPO
only on native-clean successful trajectories, feedback-conditioned SDPO only on
failed trajectories with trustworthy causal diagnostics, and no manufactured
target for the remainder. Exact metrics, failure history, source revisions, and
checkpoint hashes are recorded in
`qwen35-27b-memory-v2-failure-sdpo-cuda-acceptance-results-v1.json`.

## Hybrid GRPO/SDPO CUDA acceptance

Run 345-r2 qualifies both routes in one heterogeneous full-weight update from
untouched thinking-mode Qwen3.5-27B. The batch contained eight native,
no-feedback GRPO trajectories and four diagnostic SDPO trajectories. Every
trajectory completed without runtime error or truncation. The diagnostic arm
remained strict-failure-only: zero correct or grounded answers, exactly one
answer-free causal diagnostic per trajectory, no retry, and valid bounded tool
behavior. The native arm contained real verifier variation, including 3/8
strict successes and rewards from `0.63` to `1.0`.

Loss routing stayed disjoint. The trainer selected 4,083 ordinary RL tokens and
2,165 SDPO tokens, with zero CE and reference-KL tokens. Full CPU-offloaded
AdamW and the EMA teacher update completed at `1e-7`; mean loss was
`0.016453`, mean SDPO loss `0.190409`, mismatch KL `0.000220`, throughput
`158.73` tokens/s, and peak trainer memory `37.25` GiB. The launcher wrote a
stable 12-shard checkpoint and exited cleanly.

The first attempt exposed a configuration-level group-completion deadlock:
mixing one-sample SDPO groups with four-sample GRPO groups could fill a
12-sample dispatch window with an incomplete final GRPO group. Both sources now
use four-sample groups, and the config test requires the batch size to be
divisible by that shared group size.

This promotes the heterogeneous training mechanism, not Run 345-r2's one-step
weights. Untouched 27B remains the canonical start for the first controlled
multi-step tranche. Exact routing metrics, source revisions, failure history,
artifact hashes, and checkpoint hashes are recorded in
`qwen35-27b-memory-v2-hybrid-grpo-sdpo-cuda-acceptance-results-v1.json`.

Run 345-r3 then reproduced the same heterogeneous path after typed causal
diagnostics were integrated into the executable environment. All four
diagnostic trajectories carried the exact
`programmatic-episodic-memory-v2/causal-feedback/v1` contract with
`answer_free=true`, `retryable=true`, a stable code/category, and a rendered
message identical to the legacy feedback string. All eight native trajectories
remained contract-free. The fail-closed Prime router selected 4,684 ordinary RL
tokens and 3,107 SDPO tokens, with zero CE or reference-KL tokens; full-weight
AdamW and the EMA update completed, and a stable 12-shard checkpoint was
written. This is semantic-equivalence acceptance only. Run 345-r3's weights are
not behavioral evidence and are not the start of the tranche.

## Iterative hybrid tranche qualification

Run 346-v1 started again from the untouched thinking-mode 27B revision. Its
first update completed, but the intermediate HF weight gather overlapped with
new policy-v1 rollout workers after broadcast. Host virtual memory reached
91.5%, rank 0 was killed, and the incomplete `weights/step_1` had no `STABLE`
marker. No checkpoint from this attempt is admissible.

The replacement tranche opts into saving intermediate checkpoints before
broadcast, keeping rollout workers paused during the high-memory full-weight
gather. It also opts into deterministic weighted-round-robin source selection:
each complete three-group update contains two native GRPO groups and one typed
diagnostic SDPO group, rather than relying on a favorable random draw. Prime's
default checkpoint overlap and weighted-random source selection remain
unchanged for other runs. A two-step CUDA re-acceptance must prove that Step 1
becomes stable before policy-v1 rollout dispatch resumes.

After that gate, the replacement run performs eight full-weight updates.
Checkpoints 1, 2, 4, and 8 are selected prospectively, not after looking at
rewards. Qualification compares each one with the untouched base on all 300
familiar-heldout and 96 semantic-OOD tasks, with one rollout, no demonstration,
no feedback retry, record-only typed diagnostics, and a shared sampling seed of
`20260814`. The historical 74-task Harness Mastery battery remains byte-for-byte
unchanged.

The combined qualification driver keeps memory and Harness Mastery outputs
separate, writes typed failure-code/category mass, and marks a model complete
only after both batteries and both summaries finish. It is resumable only at
whole-model boundaries: a completed model is skipped, while a partial model
directory is rejected rather than blended with a rerun. Promotion requires
held-out/OOD memory transfer without a retrieval-always shortcut or systematic
regression in foundations, direct execution, communication, ownership, or
Oolong. Otherwise classify the checkpoint as continuation-eligible or reject
the branch; training reward alone cannot promote it.
