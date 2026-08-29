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
For natural leaf-report acquisition, set `DUAL_LEAF_REPORTER_CONTRACT=1`.
This adds a private-child-only contract with one canonical sequence: compute
from inline evidence, send `str(result)` once to the default parent, and remain
terminal after success. It does not disclose the answer or force a completion.
Keep coordinator and leaf contracts independently switchable so screens can
attribute gains to the correct role.
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
