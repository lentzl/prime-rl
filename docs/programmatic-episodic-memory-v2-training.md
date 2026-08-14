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
