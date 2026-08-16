# Qwen3.5 27B Prime Agent SDPO V1

This experiment tests feedback-conditioned SDPO on natural child-owned Prime Agent
ownership failures without dropping ordinary reinforcement learning from the same
update. The diagnostic source emits a typed, answer-free correction only after an
unsuccessful first decision. Prime-RL requires that exact contract and restricts the
SDPO loss to the first serialized coordinator tool call. Five disjoint GRPO sources
retain coordinator-owned, direct, single-child, parallel-child, follow-up, and
handshake behavior.

`zero-lr-audit.toml` is a mechanism audit, not training. It runs one complete update
with `lr = 0`, creates no checkpoint, and proves that genuine on-policy failures can
flow through feedback admission, teacher replay, token filtering, and the analytic
SDPO loss alongside group-relative RL before any parameter-changing run is considered.
Its fixed-seed 16-trace batch is the minimum complete screen: four diagnostic traces,
one two-rollout group from each ordinary retention source, and two causal groups so
both follow-up and handshake remain represented. The nonuniform source ratios exist
only to make that exact audit allocation deterministic; they are not proposed as a
training curriculum. The audit caps concurrency at eight episodes to match vLLM's
active sequence capacity while collecting the complete batch in two waves.

The launcher validates the completed run before returning success and writes
`AUDIT.json` plus `AUDIT.txt` into the run directory. A passing verdict requires
the pinned model revision across all resolved services, typed answer-free feedback
only on the diagnostic failures, at least two diagnostic codes and two resource and
phrasing families, both causal communication families, positive RL and SDPO token
mass, zero CE and reference-KL mass, a successful optimizer step at exactly zero
learning rate, and no checkpoint or model-weight artifact. Stable token exports are
matched one-to-one with reconstructed Verifiers branches to prove that SDPO reaches
only the first serialized coordinator tool call, child branches receive zero SDPO,
and retention sources receive only GRPO.

## Invalid first nonzero attempt

The first `1e-7` full-weight launch completed its optimizer step but is not an
experimental candidate. Numerically it was healthy (`loss=0.0060`, mismatch
`KL=0.0002`, gradient norm `10.625`, peak memory `43.2 GiB`), and one strict
diagnostic success correctly emitted no feedback contract and received zero SDPO
weight. The original validator incorrectly rejected that valid zero-target case;
commit `199dc9ac6` now distinguishes diagnosed failures from strict successes and
requires the latter to remain entirely unweighted.

The repaired validator then found the actual invalidation: both sampled parallel
rollouts had active branches beyond the qualified 8,192-token trainer window, so
the enforcing window filter removed them. The 16-rollout trainer cohort therefore
contained no parallel-retention signal. The checkpoint was rejected before any
behavioral evaluation and must not be promoted or used as a starting point.

Commit `a01d59f26` closes the admission gap generically. A rollout with no nonzero
loss-component signal no longer consumes a trainer batch slot, and a rejected group
is refilled from the same source instead of the global weighted sampler. A fresh
zero-LR audit on that scheduler is required before repeating the nonzero update; the
completed optimizer step above cannot substitute for it.

## Invalid second zero-LR audit

The source-refill audit completed a healthy full backward pass (`loss=0.0006`,
mismatch `KL=0.0002`, gradient norm `10.25`, peak memory `42.9 GiB`) and recovered
valid parallel-retention signal. It was still rejected because direct-retention
groups repeatedly produced no nonzero GRPO signal, while already in-flight valid
causal and parallel groups filled the 16 admitted slots before the queued direct
replacement could enter the cohort. No model artifact was written.

The follow-up introduces opt-in `orchestrator.batch_source_minimums`. Synchronous
rollout batches schedule the declared source groups before weighted sampling, wait
until every minimum is admitted, and reserve those rollouts when selecting the fixed
batch. Admitted overflow remains associated with a future trainer batch so exported
tokens and effective traces stay one-to-one. The audit declares its complete fixed
allocation (4 diagnostic, 2 each ordinary retention source, and 4 causal) rather
than relying on ratios to approximate it.

The first runtime of that allocation was stopped before its backward pass after it
revealed a second accounting edge. A source could reach its quota while replacements
requested by earlier filtered groups were still queued or in flight; those valid but
now-superfluous rollouts could exhaust the synchronous scheduler budget before a
different missing source ran. TrainSink now reports satisfied sources so Dispatcher
can remove stale queued work. If a nominally full buffer still lacks a quota, the sink
reopens only enough slots for one fresh group from that source. Focused replacement
requests remain bounded while already-running trajectories are retained as observable
overflow rather than silently discarded.

A second pre-backward runtime reached 15 admitted rollouts with every quota except
coordinator retention satisfied. A valid parallel overflow had consumed the globally
returned coordinator slots, leaving the coordinator replacement queued behind a zero
synchronous budget. Dispatcher now treats an explicit source-replacement queue like
an already-open group: it may open that bounded group after the nominal policy budget
is exhausted. Ordinary weighted work remains capped, while a rejected source cannot
be permanently starved by unrelated overflow.

## Qualified zero-LR audit

The first source-complete audit on commit `6ddefc170` passed. Its effective cohort
matched the declared 4/2/2/2/2/4 source allocation exactly and included both causal
families. The full-weight BF16 step completed with loss `0.0066686`, SDPO loss
`0.2775186`, mismatch KL `0.0002`, gradient norm `8.5`, and peak memory `43.0 GiB`.
The trainer exported 19,252 RL tokens and 294 SDPO tokens across 45 branch samples.
Validation reconstructed every effective Verifiers trace one-to-one: four coordinator
samples received causal SDPO, three child samples received zero SDPO, and 38 retention
samples received RL without CE, reference-KL, or SDPO leakage. No model artifact was
written because the learning rate was zero. This report is the prerequisite for the
fresh `1e-7` full-weight candidate update; it is mechanism evidence, not a capability
claim.

## PCIe-only runtime qualification

The experiment launchers keep NCCL P2P disabled but enable SHM by default. This
combination is required on the tested eight-L40S host: CUDA peer reads and writes
are unsupported, while the six-rank SHM path completes the audit's exact
383,418,612-element BF16 FSDP reduce-scatter in 0.48-0.90 seconds. Disabling both
P2P and SHM forced NCCL onto its socket transport; four ranks stalled in that
collective until the one-hour process-group timeout while two ranks reached the
following barrier.

Run `scripts/probe_nccl_reduce_scatter.py` under `torchrun` before reusing this
configuration on a different GPU topology. The launch defaults remain explicitly
overridable through `NCCL_P2P_DISABLE` and `NCCL_SHM_DISABLE`.

## First qualified nonzero update

The first source-complete `1e-7` full-weight update from the untouched pinned 27B
snapshot passed its mechanism and artifact validators. Its exact effective cohort
contained four diagnostic SDPO traces, two each of coordinator, direct, single, and
parallel retention, and four causal retention traces. The BF16 AdamW step processed
21,791 RL tokens and 503 SDPO tokens with loss `0.0069`, mismatch KL `0.0002`,
gradient norm `17.5`, and peak memory `42.5 GiB`. The resulting 51 GiB sharded
checkpoint is valid method evidence.

The checkpoint is not a teacher candidate. The unchanged 74-task mastery and
12-task resilience gates classified it `BRANCH-REJECTED`. Direct behavior remained
clean at 8/8 and Oolong remained 3/8. Single answer/protocol alignment improved from
3/8 to 6/8, parallel answer accuracy from 2/8 to 3/8, and follow-up answer accuracy
from 0/8 to 3/8. Those partial gains came with hard regressions: child and
coordinator strict ownership each fell from 1/8 to 0/8, handshake answer/protocol
alignment fell from 5/8 and 8/8 to 2/8 and 5/8, both child-result-delivery foundation
tasks became unclean, and child-owned path leakage increased. Resilience remained
0/12 strict. Mastery issue count rose from 51 to 55.

Trace inspection showed a broader interventionist shift rather than corrupted
weights. For example, a lost child-owned case copied the delegated path into
coordinator code, while failed result-delivery cases introduced polling or API
introspection after spawn. Because the comparison uses one temperature-0.4 sample
per task, a paired 66-task non-Oolong replication was launched before selecting a
new curriculum. The original frozen results remain authoritative; the replication
estimates evaluation variance and cannot retroactively promote this branch.

The first evaluation startup also exposed an export-completeness bug. Prime-RL's
weights-only checkpoint included model and tokenizer files but omitted Qwen3.5's
image/video processor metadata. `finalize_hf_processor_metadata.py` now copies the
immutable processor files from the pinned source snapshot, constructs the local
`AutoProcessor`, and runs before artifact validation. The validator rejects any
future multimodal checkpoint without both processor configurations.
