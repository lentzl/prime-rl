---
name: monitor-run
description: Monitor an ongoing prime-rl training run — find the output directory, tail logs, check key metrics, inspect SLURM jobs, and restart safely. Use when asked to check on a run, debug training, or investigate performance.
---

# Monitor a run

## Runbook

### On launch

1. Find the run dir and read the resolved configs at `{run_dir}/configs/` (start with `rl.json`). The run dir is `{output_dir}/{run_name}` — `run.name` auto-generates as `<envs>--<model>--<short-id>`, so if you only know the output dir, pick the most recently modified subdirectory (`ls -t {output_dir} | head -1`) or read `run.name` from the launch command.
2. Confirm all processes are alive and the run is making progress.
3. Write the initial summary into `{run_dir}/STATUS.md`.

### Recurring check-ins

Default cadence: **1 hour** (researcher can override). At each check-in:

1. Confirm processes are alive.
2. Grep logs for errors/warnings; note current step and key metrics.
3. **Append** an entry to `{run_dir}/STATUS.md` (never overwrite):

```markdown
## YYYY-MM-DD HH:MM UTC

**Step**: {current_step} / {max_steps}
**Health**: {Healthy | Degraded | Down}

**Progress**: reward/mean, seq_len, truncation, eval scores, env-specific metrics.
**Stability**: entropy, mismatch_kl, grad_norm — flag spikes.
**Performance**: trainer vs orchestrator step time, env lag, inference pressure.

**Notes**: anything unusual (errors, restarts, hangs). Omit if nothing notable.
```

### Validation on a live host

Do not run repository pytest commands on a host with a live inference, training,
or evaluation process until you have audited every discovered `conftest.py`.
Prime-RL's root test fixture runs `pkill -f VLLM` during module setup to clean CI
zombies, which also terminates legitimate live vLLM services. Run focused tests
off-host, wait for the GPU lane to drain, or use a test invocation whose fixture
discovery is demonstrably isolated from the repository root. Checking only the
selected test body is insufficient because pytest loads `conftest.py` first.

For admission-gated autonomous evaluations, distinguish behavioral failures from
provider failures before advancing the curriculum. Inspect nested trace calls,
not only the aggregate summary; a harness rescorer can report zero qualifiers
without surfacing the underlying provider exception:

```bash
jq -s 'map(.traces[]?.calls[]?.error?.type // empty) | group_by(.) |
  map({type: .[0], count: length})' traces.jsonl
```

Any bank containing a `ProviderError` is invalid evidence. Preserve its artifacts,
append the controller's invalidation event, and retry the same phase on fresh keys.
Do not use it to trigger an easier curriculum rung, candidate rollback, or an
optimizer update.

Track the evaluator process's RSS as well as GPU health. A dead agent/container
stream can make the evaluator busy-loop and retain tens of gigabytes even after
the behavioral gate is closed. For the dual-dense harness, set
`QWEN38_QUALIFICATION_EVAL_MAX_ADDRESS_SPACE_BYTES` (32 GiB on a 94 GiB host).
If an evaluation terminates early, a partial bank may trigger an easier rung or
rejection only when its complete, error-free trajectories mathematically close
the four-qualifier gate. It must never be admitted as a training source.
Snapshot the partial traces and routing audit to immutable files before hashing
the abort event. A late evaluator child can append to its original output after
the wrapper returns, invalidating an otherwise correct event-path hash. After
the runner stops, verify that no task containers from the aborted bank remain;
remove only containers whose exact IDs, creation times, and runtime labels match
that bank.

In W&B, each project auto-gets an **"overview" saved view** (train / eval / stability / performance sections) on its first run — use it for a quick check instead of the auto-generated default workspace.

### Restarting a run

**Never restart unless the researcher explicitly asked.** Confirm the exact restart command and the conditions that warrant one.

For detached launches (`nohup`, service managers, or non-interactive SSH), resolve
`uv` before launch and pass its absolute path. Also prepend the repository
`.venv/bin` when a wrapper starts `vllm-router` or another virtualenv console
script. Detached shells may not load the interactive `PATH`, so a command that
works in a login shell can fail before the run begins. After launch, verify the
real runner process and its first durable journal/log event; a background-shell
PID alone does not prove startup.

Before launching a procedural curriculum, validate that its task selector is
reachable. The `families` filter applies only to families emitted directly by
the base generator; named curricula such as `natural_n1a` must be selected with
`curriculum_rung`. Filtering `families` to a non-emitted rung makes the taskset
materializer's `while len(tasks) < count` loop spin at 100% CPU while the run is
stuck at `Loading training environments`. Treat this as a pre-rollout config
failure: preserve the partial run, stop it cleanly, fix and audit the selector,
and relaunch under a fresh immutable run label.

When a privileged bootstrap or hint artifact is attached, its split, curriculum
rung, start index, count, master seed, and private-payload mode must match the
live taskset generator exactly. Schema validation is insufficient because task
keys include generator coordinates. Materialize the complete configured bank
with the artifact in a CPU-only preflight before starting trainer or inference;
an identity mismatch must fail before GPU allocation and optimizer activity.

For role-scoped GRPO with one inference model, stabilize the role that is not
receiving gradient. A coordinator/root update must use the exact-child phase so
weak child behavior cannot flatten every group before root credit is measured.
A child/non-root update uses an exact root spawn with natural child evidence
handling. Confirm the trace contains the anchored role event and reward variance
within one task group before accepting the optimizer receipt. This is temporary
training-time leakage; promotion remains a separate unassisted evaluation gate.
Prime Agent children may execute in external sessions that are absent from the
parent verifier trace, so a parent-side exact-child request interceptor alone is
not sufficient evidence of anchoring. Carry a value-free `replace VALUE` send
template in the spawned child prompt as the cross-session fallback; never expose
the oracle integer in coordinator-visible prompt tokens.

**Never** run kill or launch commands from your own shell. Dispatch them to the tmux **Launcher** window so the researcher sees what was executed:

```bash
SESSION=$(tmux display-message -p '#S')
tmux send-keys -t "$SESSION:Launcher" 'your command here' Enter
```

After a restart, verify all processes are back up and progress resumed before the next check-in.

---

## Reference

### Where to find things

- `scripts/tmux.sh` launches the run with a `Launcher` window in the named tmux session. The Claude window receives the run dir and session name in its appended prompt — if either is missing, **ask** rather than guess.
- `{run_dir}/configs/` — resolved configs, written as JSON so explicit None settings round-trip (`rl.json` has the full picture).
- `{run_dir}/logs/latest/` — the current attempt's logs (each launch gets `logs/attempt_<n>/`; resumes never overwrite earlier attempts). See below.
- `{run_dir}/rollouts/step_N/{train,eval}/` — saved rollout traces (see Traces below).

### Logs

```
{run_dir}/logs/latest/
├── trainer.log                # rank 0 stdout
├── orchestrator.log           # orchestrator stdout
├── inference.log              # vLLM stdout
├── trainer/
│   ├── node_*.log             # per-node (multi-node only)
│   └── torchrun/              # per-rank stdout/stderr
├── inference/
│   ├── node_*.log             # per-node (multi-node only)
│   └── router.log             # the single global router (multi-node only; single-node logs it in inference.log)
└── envs/{train,eval}/{env_name}.log    # one log file per env
```

Usually tailing `trainer.log`, `orchestrator.log`, and `inference.log` is enough. Drop into per-node or per-rank logs only when debugging. All logs are loguru with `HH:mm:ss  LEVEL  message`; levels: `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`.

Scan for problems:

```bash
grep -E "WARNING|ERROR" {run_dir}/logs/latest/{trainer,orchestrator,inference}.log
grep -E "WARNING|ERROR" {run_dir}/logs/latest/envs/{train,eval}/*.log
```

### Metrics

All metrics print to the console log (and W&B when configured).

**Progress** — orchestrator log. Rollout metrics mirror the episode/trace hierarchy, at two levels:

- `{scope}/{subset}/<metric>/<stat>` — episode-level facts only: the token/turn/branch counts, summed over an episode's traces.
- `{scope}/{subset}/<agent>/<metric>/<stat>` — every trace-level metric (reward, truncation, errors, timing, env metrics, filter verdicts, eval scores), keyed by agent name so seats never mix. Flat over that agent's traces: one sample is one trace, so an in-episode fan-out like n solvers contributes n samples.

`scope` is `train/agg` (all train envs) or `train/<env>` (`eval/<env>` for eval); `subset` is `all` (every rollout) or `effective` (post-filter). Single-agent envs have one agent — usually `agent` — and one trace per episode, so both levels agree; multi-agent envs name each seat (`proposer`, `solver`, `judge`, …).

| Metric | Description |
|--------|-------------|
| `train/agg/effective/<agent>/reward/mean` | mean training reward for that agent (per env: `train/<env>/effective/<agent>/reward/mean`) |
| `train/agg/effective/num_total_tokens/mean` | avg tokens per episode, summed over its agents (also `num_input_tokens`, `num_output_tokens`) |
| `train/agg/effective/num_turns/mean` | avg turns per episode, summed over its agents |
| `train/<env>/effective/<agent>/num_turns/mean` | avg turns for that agent alone (also token counts, `num_branches`) |
| `train/agg/effective/<agent>/is_truncated/mean` | fraction of that agent's rollouts truncated |
| `train/agg/all/<agent>/has_error/mean` | fraction of that agent's rollouts errored (per-type under `train/agg/all/<agent>/error/<type>`; also `dispatcher/errored/{train,eval}`) |
| `train/agg/all/<agent>/is_trainable/mean` | fraction carrying a training signal — 0.0 for a frozen seat like a judge (also `is_filtered`, `filters/<name>`) |
| `train/<env>/effective/<agent>/metrics/<name>/mean` | env-specific metrics for that agent (e.g. pass rate) |
| `train/<env>/effective/<agent>/timing/agent/model/mean` | model vs harness share of that agent's phase |
| `eval/<env>/effective/<agent>/{avg@k,pass@k}` | eval scores for that agent, when configured |

**Stability** — trainer log:

| Metric | Description |
|--------|-------------|
| `mismatch_kl/{all,env}/{mean,std,max}` | KL between trainer and (old) inference policy over trainable tokens |
| `entropy/{all,env}/{mean,std,max}` | policy entropy over trainable tokens |
| `masked_advantage_{positive,negative}/mean` | fraction of DPPO-masked tokens with +/- advantage |
| `optim/grad_norm` | spikes may precede divergence |

**Performance** — trainer and orchestrator step independently, so comparing step times shows who's waiting on whom.

| Source | Metric | Description |
|--------|--------|-------------|
| trainer | `time/step` | total trainer step |
| trainer | `time/wait_for_batch` | **high → orchestrator is bottleneck** |
| trainer | `time/forward_backward`, `time/broadcast_weights`, `time/save_ckpt` | phase timings |
| trainer | `perf/throughput`, `perf/mfu` | tokens/s and MFU % |
| orchestrator | `time/step`, `time/save_ckpt` | phase timings |
| orchestrator | `time/wait_for_policy` | **high → trainer is bottleneck** |
| orchestrator | `dispatcher/off_policy_level_{mean,max}`, `dispatcher/inflight_{train,eval}`, `dispatcher/groups_in_flight`, `dispatcher/queued/eval` | dispatcher / async state |
| env server | event loop lag (min/mean/p90/p99/max), active task distribution | periodic |

For live vLLM stats, query Prometheus directly:

```bash
curl -s http://localhost:8100/metrics | grep -E "num_requests|gpu_cache_usage"  # engine port (8000 is the router)
# vllm:num_requests_running, vllm:num_requests_waiting, vllm:gpu_cache_usage_perc (→1.0 = KV cache saturated)
```

### Traces

```
{run_dir}/rollouts/step_N/{train,eval}/all/traces.jsonl        # appended per rollout as it completes
{run_dir}/rollouts/step_N/{train,eval}/effective/traces.jsonl  # written per finalized batch / eval epoch
```

JSONL files of `vf.Trace` records (training tensors excluded), one line per trace — a
multi-agent env's episode contributes several lines sharing one `info.episode_id`. `all`
gets every completed rollout the moment it arrives — errored, filtered, and never-batched
ones included — so it's crash-durable; `effective` gets the clean trainable subset that went
into the step's train batch (eval: the non-errored trainable epoch cohort; multiple eval envs
share the step file) — untrainable traces (a frozen judge's) appear only in `all`. Each record carries `run` (`{type, id, step}`; for eval, `step` is the trigger step),
`verifiers` (producing build), `agent` (model, sampling, harness, `name`, `trainable`), `ok`
(the success sentinel — `errors` alone keeps retry history even after a recovery), and
`runtime` (config + provisioned resource id, e.g. the sandbox id), plus `env_name`,
`group_id`, `episode_id`, and `policy_version` under `info`.

```bash
wc -l {run_dir}/rollouts/step_42/train/{all,effective}/traces.jsonl
jq '.rewards' {run_dir}/rollouts/step_42/train/effective/traces.jsonl
jq 'select(.ok | not) | {id, env: .info.env_name, runtime}' {run_dir}/rollouts/step_*/train/all/traces.jsonl
```

The batches consumed by the trainer are shipped over ZMQ by default, so nothing binary is written. With `rollout_transport.type = "filesystem"` they land at `{run_dir}/rollouts/step_N/rank_<rank>.bin` (one packed micro-batch file per trainer DP rank), next to the trace subtrees.

### Common failure modes

A few warnings are normal. Escalate when errors are persistent, growing, or hit a large fraction of rollouts.

- **Env workers**: exceptions in env code, timeouts, sandbox errors, OOM kills (most common source — runs user code).
- **Orchestrator**: empty/errored rollout spikes, weight-broadcast failures, checkpoint errors.
- **Trainer**: NCCL/CUDA errors, OOM, NaN loss or gradients.
- **Inference**: NCCL/CUDA errors, OOM, request timeouts.

### Process tree

All processes use `setproctitle` so they're visible in `ps`/`htop`/`pstree`:

```
PRIME-RL::Launcher
├── PRIME-RL::Inference          (vLLM server, GPU 0)
├── PRIME-RL::EnvServer          (verifiers' ZMQ env server, run in-process; one per train/eval source)
│   └── Verifiers::EnvWorker0..N
├── PRIME-RL::Orchestrator       (CPU-only; connects to each env server)
├── torchrun
│   └── PRIME-RL::Trainer        (GPU 1+)
└── tail trainer.log
```

For multi-node runs, trainer and inference processes are on separate nodes — use `srun` or `ssh` to inspect them.

### Dual-dense SPADE coevolution loop

For terminal document-summary comparisons, the paired summary evaluation driver
can reuse the two frozen engines while running one model per GPU concurrently.
Its receipt explicitly records direct engine routing, not owner/child delegation
or role-proxy behavior. `DOCUMENT_SUMMARY_PROBES` selects space-separated existing
probe config names; defaults are `exceptions city-shade`. Inspect every episode
in the receipt's declared probe list and both checkpoint hashes;
driver completion and file-presence rewards do not establish semantic utility.
For owner-led summarization, select `smoke-owner-direct.toml` with the existing
summary smoke launcher. It selects the owner checkpoint at depth zero only for
owner modes; terminal probes continue to use the worker there. Verify both role
routes and actual child receipts before claiming delegated capability. The
Markdown artifact reward checks matching handoffs and unchanged assembly, not
semantic fidelity or all protocol actions. Inspect exported per-chapter text and
the native trace; never infer those behaviors from a configuration dry-run.
`DOCUMENT_SUMMARY_PROBE_CONCURRENCY` optionally runs multiple independent probes
per frozen model in bounded waves (default 1). Use the same value for both arms
and label the changed scheduling when comparing with an earlier run. The driver
waits for every wave member, records concurrency in its receipt and rejects
duplicate probe labels. This can reduce an idle GPU tail without adding models
or changing the chapter task; measure actual throughput and timeout effects.
Initialize the routing audit as an empty event log before starting the proxy:
direct-engine runs legitimately leave it empty, and final receipt hashing must
not fail solely because no role-routing event was emitted.
The role proxy records a forwarded request after its upstream response returns.
An empty routing log during generation does not prove no model call was issued.
Check the engine's running requests/token throughput and native processes;
the evaluator's last installation log can remain unchanged during live generation.
The summary evaluation driver exports exact evidence-workflow text to its
`document/artifacts` directory. Missing files remain absent; readable exports are
diagnostic artifacts and do not change the native trace or semantic verdict.
When exporting teacher episodes from intercepted traces, select the recorded
model-visible continuation rather than its raw pre-interception sibling. Record
the feedback style in the dataset manifest so later scaffold changes do not
silently leave supervision aligned to obsolete gate framing.
For direct-summary SFT, export the observed direct runtime/task prefix with
`export_q35_2b_document_summary_direct_sft_v1.py`; do not reuse the staged-notes
continuation. Interleave the 20 retained TRAIN cases with reviewed public
chapters. The source preparer accepts `--additional-chapters-per-book`, and the
exporter accepts a separately reviewed `--teacher-additions` file so the corpus
can grow without overwriting the earlier `--teacher-labels` file; do not abbreviate
that flag to `--teacher`, which is ambiguous. Counts derive from the source
and case manifests rather than requiring exactly 40 episodes. Preserve the
earlier data and exclude all evaluation chapters, including prospective ones.
The optional `--include-book-of-tea` adds its pinned complete chapters I and II
without changing the original four-book selection. Review them as authored
cultural/historical arguments, distinguishing quoted stereotypes, myths and
the author's judgments from established events. The exporter permits only the
explicit TRAIN book allowlist and rejects duplicate or evaluation book IDs.
The direct exporter excludes numbered chapter probe files (`*-ch[0-9]*.md`),
not only first chapters; keep new probe books outside the TRAIN source list.
When extending the public TRAIN corpus, inspect both chapter boundaries. A new
part heading can fall between the chapter's final prose and the next chapter
marker (for example, Treasure Island VI/VII); omit that transition heading,
preserve the complete prose, and verify earlier chapter files remain unchanged.
Run `audit_q35_2b_document_summary_direct_sft_v1.py` against the
source checkpoint's tokenizer before launch. Pass it with `--tokenizer`; the
separate owner audit uses `--tokenizer-path`. The audit replays teacher file
operations, checks complete untruncated trainer sequences, and verifies
assistant-only loss.
The training wrapper requires the matching `RENDERER-AUDIT.json`, including
dataset and tokenizer hashes. Format validation alone is not semantic admission.
For correction-response training, `--include-format-repairs` appends one authored
repair episode per TRAIN chapter while retaining the base episodes unchanged.
The initial prose copy and premature Done are fixed context with
`trainable=false`; the useful source read, corrected reviewed bullets and final
stop are supervised. The correction text is the observed model-visible native
format feedback, not a raw gate diagnostic. Replay all file observations and use
the real trainer/tokenizer audit to verify zero loss on the incorrect prefix,
user/source text and tools, with no truncation. Never infer masking from parquet
annotations alone or treat these authored drafts as on-policy model failures.
For source-grounded semantic repairs, pass a reviewed TRAIN-only
`--semantic-repairs` specification to the same exporter. It pins the existing
chapter source, creates a format-valid but deliberately incorrect draft, and
supervises the original reviewed summary with explicit correction reasoning.
The revision request is authored user feedback, not an observed native semantic
gate. Record that distinction; a format gate does not supply factual review.
Both the draft and premature stop are masked. The real trainer audit covers
`semantic_repair` as well as `format_repair`; annotation checks and file replay
are preparation, not substitutes for verifying token loss and truncation.
For child integration teaching, the same exporter accepts `--native-child-trace`.
It takes only an observed depth-one runtime prefix and IPython schema, replaces
the inherited task instruction with the current role-aware instruction, and
uses existing reviewed TRAIN chapters for read/write/parent-receipt/stop episodes.
No evaluation task or source enters those examples. Selected count-repair states
mask a repeated successful write with mistaken word-count reasoning, then teach
the actual bullet-word count and one parent send. `Path.write_text` returns a
character count, not a word count or a failure. The audit replays real file I/O
with a declared exact-receipt stub; a scripted queued status is not live delivery.
Check native-role context, source/target preservation and incorrect-retry loss
masks in the real trainer audit before the next update. Native session logs can
omit model-facing interception rewrites, so inspect the finalized trace as well
when diagnosing whether recovery feedback reached a child.
When retaining native session logs, start the bounded capture observer at trial
startup and bind it to the newly created runtime's verified ID, image and creation
time. Short episodes can finish between path discovery and a later copy call.
Inspect `/tmp/vf-prime-agent-runs` recursively; child JSONL files live below
`agent/session-artifacts`. A path listing or empty capture directory is not an
archive, and even live snapshots may omit the final events. Preserve the complete
intercepted trace separately and state the coverage actually recovered.
For native source-copy or semantic failures, `--include-native-revisions` adds
authored self-review before the receipt: masked incorrect draft write, actual
saved-draft read, reviewed correction, then one send and stop. It reuses existing
TRAIN summaries and semantic-repair specifications, not evaluation answers or an
invented native semantic gate. Verify both writes and the intervening read in
file replay, and correction reasoning plus zero incorrect-draft loss with the
real tokenizer audit. These variants add a teaching boundary, not new documents.
For native owner acquisition, `export_q35_2b_document_summary_owner_sft_v1.py`
combines TRAIN-only scripted index/delegation/receipt/assembly episodes with the
acquired 48-row role-decision rehearsal. Its runtime trace supplies only the
owner prompt prefix and IPython schema, not evaluation chapter contents or
successful child experience. Wrong index lookups are masked context; receipt
validation, passive yielding and unchanged index-order assembly are supervised.
For owner waiting repairs, `--include-wait-repairs` adds masked finite polling
analogues before any receipt and after a partial fan-in, followed by passive
turn endings. These are authored executable lessons, not a claim that an observed
infinite cell returned. Retain handles and track received receipts separately;
do not clear handles to make a waiting condition pass. Regenerate owner examples
against the current role-aware task prompt while preserving reviewed TRAIN
source/summary pairs and acquired rehearsal. The audit derives owner-family
counts from the cases and checks each repair's action mask before token replay.
Use explicitly printed results for these teacher cells so replayed stdout is
exact without assuming Python repr matches IPython's object pretty-printer.
Keep all original rehearsal fields when forming a mixed Arrow dataset: building
from the first row's keys alone silently drops fields unique to the other family.
The owner audit replays actual file I/O with an explicitly declared admission
stub, verifies preserved rehearsal and the real trainer/tokenizer's complete
sequences and loss masks. Stubbed admission is not evidence of live delegation.
The existing training wrapper requires this matching audit before an owner update.
For owner task-start failures, `--include-start-repairs` adds authored short
repetition and premature-wait contexts before any tool action. Mask the entire
incorrect assistant response, then teach index reading and the complete existing
handoff sequence. These are TRAIN-only corrective analogues, not native successes
or copied evaluation output. A routed child request alone does not establish a
valid chapter assignment: inspect its actual name and prompt before attributing
missing summaries to the worker's summarization ability.
For retained-handle misuse, `--include-handle-repairs` adds masked join/await
failures before receipts and immediately after the final message arrives but
before its payload is stored. Supervise passive turn ending or explicit receipt
validation followed by reading assigned files, respectively. A queued send,
message delivered to context, receipt stored in the kernel, and assembled file
remain separate observations. Replay the declared admission stub's error type;
do not represent it as the full native traceback or treat a handle as a result.
For thinking-enabled owner SFT, use `--decision-prefixes` to supervise each
correct assistant decision at its own generation boundary, masking earlier
assistant history. Whole-episode Qwen rendering strips reasoning before later
user/child messages; `enable_thinking=true` alone does not preserve those earlier
targets. Keep full authored episodes in CASES for file replay, not student input.
The audit must check exact prefixes, all intended decision boundaries, preserved
current-target reasoning in the actual supervised tokens, zero history loss and
complete source/rehearsal provenance. Do not change the inference renderer to
make this training-only check pass.
The owner and native-child exporters/audits import the summary
taskset. When using the shared host environment with `--no-sync`, include both
the checkout's `deps/verifiers/environments/document_summary_v1` and
`deps/verifiers` in `PYTHONPATH`; the former package is not necessarily installed
in that environment. Do not modify the environment while training is live.
Run it only while the GPU host is idle, then continue from the current acquired
owner descendant; keep the worker weights and semantic promotion separate.
For local exporters using an isolated taskset runtime, invoke
`uv run python /absolute/path/to/script.py`. Passing the script directly to
`uv run` can rediscover its enclosing workspace and resolve unrelated,
incompatible environment dependencies instead of using the selected runtime.
The document training wrapper accepts a positive explicit update count. Use that
per-run count and timeout for bounded reassessment; there is no campaign-wide
GPU-hour or eight-update quota on the Owner's existing allocation.

For the document-summary campaign, the Owner explicitly prefers aggressive,
successive weight updates and accepts the risk of behavioral collapse. Continue
from the latest valid experimental checkpoint by default, even when the full
summary still fails evaluation. Small gains and diagnosed failures should guide
additional training, not create a minimum capability threshold for permission to
train. Keep semantic promotion separate from the training frontier. Train useful
partial behavior and corrected targets without labeling erroneous suffixes as
successes; complete successful rollouts are not the only possible supervision.
Preserve recoverable checkpoints and compare periodically for regressions, but
do not turn those checks into an admission gate for every next update. Numerical
corruption, invalid data or broken infrastructure remain reasons to repair the
run. Behavioral regression alone is evidence to investigate, not an automatic
rollback or campaign stop. Never change numerical dtype settings implicitly.
The [standing domain-training charter](../../../docs/continual-domain-training-charter.md)
also distinguishes an evolving curated mixture from obligatory append-only data
growth and describes recovery and optimizer continuity at practical boundaries.

For `run_q35_2b_spade_dual_dense_autonomous_v1.py --coevolution`, treat a
generated batch as complete only when all of these exist and agree:

- `generation/GENERATION.json`, `NO_HINT_BOOTSTRAP.json`, and
  `HINT_BOOTSTRAP.json`;
- both six-episode result trees and routing audits;
- `PAIRED_EVALUATIONS_COMPLETE`, the two interaction summaries, and
  `SCORE.json`;
- the corresponding hash-chained rows in `coevolution-memory.jsonl`.

Check that the generation records the current coordinator weight hash, exposes
no oracle/private values, and assigns the same fresh task keys to both arms.
The first batch after a checkpoint has no eligible Designer update by design.
Only a positive-reward batch generated by an older coordinator hash may appear
as a delayed rewarded Designer row. A better-arm interaction source contains
complete qualifying rows. Ordinary evaluation failures may additionally
produce `positive-prefix-source` rows, but only when their sampled tool actions
hash-match the verifier event audit, cardinality is exact, and no forbidden
atom fired. Confirm that the replay reports these separately as
`new_partial_rows`; the incorrect suffix of a failed trajectory must never
appear in the exported messages. Preserve the four-qualifier champion
threshold even when the aggressive exploratory frontier advances from one
complete trajectory or one validated positive prefix.

If every Designer proposal fails schema or safety validation, the expected
durable outcome is `generation/REJECTIONS.json` plus `DESIGNER_REJECTED`. The
controller must record `coevolution_batch_repaired`, export one or two
`scaffolded_schema_and_safety_repair` rows for the coordinator, skip the paired
arms, and continue role training. This is not an infrastructure failure.

Replay rows use raw curriculum phases or a three-part wrapped phase. Rewarded
Designer rows use `spade:<track>:<phase>` and scaffolded repair rows use
`spade-repair:<track>:<phase>`. Before restarting after a replay-build failure,
run the exact failed combine command against a fresh temporary output directory
and confirm both namespaces rank against the embedded track. Preserve the
failed output directory for diagnosis; the autonomous runner will refuse to
overwrite it.

For multi-day rentals, run `watch_q35_2b_spade_dual_dense_v1.sh` in a separate
tmux window. It may restart only when the runner is absent, the explicit stop
file is absent, and no GPU compute process remains. Keep its default fuse of
three restarts at the same hash of the durable controller head; an open fuse is
a deterministic blocker requiring diagnosis, not permission to delete partial
artifacts or replay evaluations. The watcher must exclude its own PID when
matching the runner pattern because its argv contains the full restart command,
which ordinarily repeats that same pattern.

When two independent vLLM role engines share one GPU, do not rely only on
`gpu_memory_utilization` values whose sum appears to fit. Each process profiles
the device independently, so the second engine can report no available cache
blocks after the first has reserved its cache. Set an explicit
`kv_cache_memory_bytes` cap for both engines, verify that the cap supports the
configured concurrency and context length, then confirm both health endpoints
and actual aggregate GPU memory before allowing rollouts. An engine startup
failure before rollout generation is a zero-update infrastructure attempt; stop
the waiting trainer and preserve the unique run label rather than reusing it.
For renderer-mode training, the role proxy must also forward the root-mounted
`/inference/v1/generate` endpoint; forwarding only `/v1/chat/completions`
produces an all-404 rollout group. Because generate requests contain token ids
rather than messages, classify the role with a tokenizer-derived subsequence
for the private-evidence marker, rewrite the logical model to the selected
upstream model, and include the endpoint, role, model, payload hash, and status
in the routing audit.

If a batch fails for any other reason before either `DESIGNER_REJECTED` or
`PAIRED_EVALUATIONS_COMPLETE`, leave the controller event head unchanged, stop
GPU services, and archive the partial batch with a reason suffix before
retrying. Never silently reuse or overwrite a partial generation or one arm of
a pair.

### Autonomous role-GRPO loop

For `run_q35_2b_role_grpo_autonomous_v1.py`, the durable authority is the
controller state directory, especially its append-only `events.jsonl`. Verify
the SHA-256 link from every row to its predecessor before trusting the
frontier. The controller lock must have exactly one owner, and the explicit
`STOP` file must be absent while work is expected to continue.

Each training or evaluation launch has a fresh sequence-derived label and a
fresh, non-overlapping deterministic task bank. Never reuse a label after a
failed, interrupted, or partially completed action. On restart, reconcile a
recorded `train_started` event only from its attempt and success receipts, and
reconcile `eval_started` only from the complete result envelope and routing
audit. If those artifacts do not prove completion, record the interrupted
action as failed and advance to a new label; do not replay it under the old
identity.

Treat the two-GPU role-GRPO host as single-tenant while an action is live.
Do not run repository tests, validation launchers, or even a nominal `--dry-run`
beside the live stack; repeated validation commands have coincided with external
`SIGTERM` delivery to the trainer. Make read-only log/process checks only, and
run operational validation in a controller maintenance gap or on another host.

Role-GRPO is full-dense and strictly role-scoped. A coordinator update samples
only root tokens while the child checkpoint is a frozen anchor; a child update
samples only non-root tokens while the coordinator checkpoint is frozen. The
role filter masks wholly unscoped auxiliary graph roots such as Prime Agent's
`/refine` calls from both policies. Missing or conflicting lineage inside an
actual coordinator or child client-session graph still fails closed.
For depth-zero role-persistence acquisition, set
`DUAL_ROOT_COORDINATOR_CONTRACT=1` on the dual-policy mastery launcher. The
proxy then adds the explicit root contract only to depth-zero Chat Completions:
the root has no parent, retains its coordinator identity across child traffic,
and alone finalizes the user answer. Keep it disabled for unscaffolded
admission measurements, and never apply it to private-evidence or depth-positive
child sessions.
Structured Chat Completions carrying the full bounded-leaf contract
(`is_root=false`, delegation/finalization disabled, and exactly one parent
report) route to the child policy even when the private-evidence marker is not
serialized into the request. A recursive coordinator lacking that complete
leaf contract remains on the coordinator policy. Verify both roles appear in
the routing audit before accepting a dual-policy screen.
For natural leaf-report acquisition, set `DUAL_LEAF_REPORTER_CONTRACT=1`.
This adds a private-child-only contract with one canonical sequence: compute
from inline evidence, send `str(result)` once to the default parent, and remain
terminal after success. It does not disclose the answer or force a completion.
Keep coordinator and leaf contracts independently switchable so screens can
attribute gains to the correct role.
For a weak leaf that understands the protocol but mishandles evidence paths, add
`DUAL_LEAF_INLINE_EVIDENCE=1`. The proxy then makes the already-visible evidence
available as `INLINE_EVIDENCE` inside the model-authored IPython cell while
preserving the model's computation and native parent send. This is an
answer-free acquisition scaffold: verify the routing audit records
`forwarded_leaf_inline_evidence_repaired`, and never combine it with the typed
child-report scaffold.
When isolating natural leaf reporting, combine the leaf contract with
`DUAL_LEAK_COORDINATOR_EXACT_ACTION=1`. This guarantees only the disclosed
root spawn; do not enable the coordinator-return leak, because that would force
the child report and invalidate the natural leaf measurement.
The exact coordinator action flag applies to both token-in training requests
and live depth-zero Chat Completions. Confirm the routing audit records
`forwarded_forced_exact_coordinator_schema` before treating an evaluation as a
root-spawn-controlled child isolation.
Live chat extraction may use the exact-action prompt phrase from an
`action_scaffold` bootstrap even when the token-in marker is absent; the
explicit launcher flag remains mandatory.
If the leaf repeatedly computes the right value but invalidates the trajectory
with a path read, malformed IPython, or a second send, set
`DUAL_TYPED_CHILD_REPORT=1`. The model still computes the payload from the
private prompt, but a typed one-shot return schema fixes parent routing and the
proxy terminates the child session after the native send. This is a scaffolded
acquisition measurement, not an unscaffolded admission result.
For a child-only consolidation update, build from hard-success forced-return
traces with `build_q35_2b_recursive_return_trace_sft_v1.py --child-only`.
This excludes root anchors so the candidate can be trained from the protected
child checkpoint while the coordinator remains frozen during evaluation.
The early coordinator curriculum preserves the harness's first named IPython action
and disables thinking. Because the 2B model cannot yet copy that action reliably,
the proxy supplies one synthetic exact retained-spawn completion per coordinator
session; receipts label this `first_action_sampling=synthetic_exact_spawn` rather
than presenting it as a strict policy-distribution sample. Later coordinator turns
are naturally sampled. Child
GRPO keeps thinking enabled and strips the broken named tool-choice constraint;
the frozen counterpart retains its curriculum mediation. A completion-only notice about a child action is not child
evidence. Child action shaping must come from an
observable, parseable non-root IPython action and awaited
`agent_message.send(..., receiver_role='parent')`; forbidden behavior must
still receive zero reward.

Zero-advantage filtering is mandatory. An all-equal group may cause the
orchestrator to sample another group, so `0/8` followed by a fresh set of eight
in-flight rollouts is not by itself a stall. The controller's bounded training
deadline is the terminal guard: after it expires, terminate the whole process
group, preserve the unique attempt receipt and logs, record a failed/no-update
event, and alternate to the other role. Keep the complete eight-rollout GRPO
group logically in flight, but serialize complete coordinator episodes at the
EnvServer boundary. Two simultaneous coordinator episodes have driven a 94-GiB
host into global OOM with the EnvServer at roughly 89 GiB RSS. Child updates may
use two-episode waves when observed memory remains bounded. Allow up to three
hours for a serialized coordinator update so a zero-advantage group can be
replaced without terminating valid work.

Serialization does not contain an individual scorer allocation loop. Invalid
sampled Python such as `from agent_message import agent_message` can create a
self-prefixed static alias. Alias resolution must detect cycles by the repeatedly
resolved head, not by the ever-growing full dotted name; otherwise task scoring
can concatenate the alias until the EnvServer or evaluator reaches its memory
limit. This is a scorer robustness failure, not evidence against ordinary IPython
computation. Keep ACP output caps and non-interactive pagers as independent
defenses. During a coordinator soak, sample EnvServer RSS and host available
memory as well as container count; if RSS grows continuously toward the host
limit, place the controller STOP sentinel and terminate the exact EnvServer
worker before global OOM, then record the action as failed with no update.

Advance a role's exploratory training frontier after every validated optimizer
update, even if held-out admission fails. Advance its promoted frontier only
after at least four distinct complete qualifying held-out trajectories. The
evaluation envelope must contain the expected episode count, zero errors,
distinct task keys, hard success, successful coordinator and child routes, and
the exact hashes of both evaluated checkpoints. Never weaken the four-trajectory
promotion floor.

For disk pressure, prune only completed `grpo-auto-*` checkpoint directories
that are explicitly absent from the initial, current, and promoted frontiers.
Record every deletion in the hash-chained event log and retain enough recent
unpromoted history for diagnosis. Never delete an in-flight output, receipt,
evaluation result, routing audit, or controller state artifact.

Prime Agent Docker runtimes may survive after their owning training or
evaluation command exits and can consume tens of GiB in writable layers. The
controller snapshots running container IDs before each action and may force-remove
only newly created containers whose image is either the purpose-built
`rlm-prime-agent-runtime:*` image or the configured `python:3.11-slim` runtime
after that exact action has terminated. Every removal
must be recorded as `runtime_containers_pruned` in the hash-chained event log.
Never infer cleanup scope from age alone and never remove a pre-existing or
in-flight container.

### Publishing code when local Git ref scans stall

Use `GIT_OPTIONAL_LOCKS=0` for read-only status/diff checks on cloud-managed
checkouts; an optional index refresh can hold `index.lock` while file reads stall.
Prefer explicitly scoped paths; the flag does not prevent all cloud-read stalls
or guarantee that a later phase of the Git operation cannot acquire a lock.
If a lock blocks publication, identify its owning process and wait for or resolve
that exact operation. Never remove a lock while its owner is still live.

Before deleting a remote checkpoint in favor of a local archive, repeat the
local byte-count and content-hash checks. An iCloud `dataless` placeholder and a
historical matching hash do not prove that recovery bytes are currently readable.
If the fresh read fails, retain the remote checkpoint and verify a complete copy
outside the cloud-managed folder before cleanup. Do not claim data loss merely
from placeholder metadata.

When local space is tight, stream a complete run and its standalone receipt into
a compressed archive outside iCloud instead of retaining a second unpacked copy.
Before remote cleanup, stream-decompress through the gzip checksum, compare the
exact regular-file set and every content hash with the host, and check symlink
targets. Hash the archive itself and record its recovery path. Do not confuse a
completed transfer or successful archive listing with verified recovery bytes.
An idle host can also perform CPU preparation when local source imports stall
on cloud-file reads. Pin the same source/data hashes and exact script snapshot;
keep that provenance distinct from a clean checkout, and never do this beside
live training or inference. Preserve any partially completed local attempt.

If a push or bundle stalls while enumerating local refs, inspect its exact
process before retrying. A temporary bare repository can reuse the existing
object store through `GIT_ALTERNATE_OBJECT_DIRECTORIES`, hold only a transfer
branch pointing to the exact committed head, and use
`-c core.alternateRefsCommand=true` to avoid enumerating the source refs. Push
that ref to the existing campaign branch, or create and verify an incremental
bundle from the GPU checkout's exact head before transferring it. Fast-forward
only, verify both commit IDs, and keep live GPU checkouts unchanged. This
preserves original commits; do not present an uncommitted file copy as a commit.
For an incoming collaborator commit, the same isolated repository can fetch the
specific branch with `--no-tags` and `GIT_OBJECT_DIRECTORY` pointing to the
existing object store. Inspect the fetched commit and merge it normally in the
local checkout before publishing; never force-push over concurrent work or sync
the live GPU checkout merely to resolve a publishing divergence.

If an ordinary commit fails while refreshing cloud-managed worktree files,
first confirm that process is terminal and its index lock is absent. Inspect the
staged paths and diff, and obtain the exact staged tree with `git write-tree`.
`git commit-tree` can commit that inspected tree with the current HEAD as parent;
advance only the current branch with `git update-ref BRANCH NEW OLD`, guarding the old
HEAD. This preserves unstaged changes and avoids another full-worktree refresh.
Verify the new parent and tree before publishing; this bypasses commit hooks,
so retain the separately executed checks and never claim hooks ran.
