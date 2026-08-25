# Qwen3.5-2B dual-dense SPADE loop

This experiment measures whether a pair of independently updated Qwen3.5-2B policies can climb the Prime Agents interaction curriculum. It is a capability-ramp experiment, not a production training recipe.

## Policies

- The coordinator policy serves root turns and learns spawn, passive yield, resumption, synthesis, and final-answer behavior.
- The child policy serves requests containing the audited private-evidence marker and learns evidence analysis, IPython/tool use, and typed reports to its parent.
- Both policies begin from the same dense seed produced by merging the admitted R6Y6 adapter into its hash-locked base. After the first update their checkpoints must have different SHA-256 values.
- Evaluation runs one policy per GPU. Training runs one role at a time with both GPUs, so the update is a full-parameter FSDP update rather than LoRA.

## Autonomous cycle

1. Evaluate the current pair on fresh `natural_n1a` train-generation keys with no optimizer update.
2. Admit a role source for learning when at least one complete, error-free
   qualifying trajectory covers one distinct task key. Track four complete
   qualifiers on four distinct keys separately as promotion-grade evidence.
3. If a gate fails, move that role to the next more supportive environment. Keep the other role frozen once its gate is open.
4. Export between one and four admitted rows for the relevant role. Failed
   trajectories are never trainable.
5. Combine the new rows with a deduplicated role-local replay capped at eight
   child rows or twelve coordinator rows.
6. Run exactly one full-parameter optimizer step for each role authorized in
   the lineage. A frozen role receives exactly zero steps. No LoRA update is
   authorized.
7. Verify that every updated dense hash changed, that the two role hashes remain
   distinct, and that updated checkpoints are complete and `STABLE`. Tighten
   the updated role's environment by one rung and continue on fresh keys.

After an update, each role must also retain the exact rung that admitted its
parent. A candidate that fails its parent rung is rejected for that role even if
it passes an easier rung; the other role remains frozen. Partial banks can prove
rejection only when every missing episode passing would still leave the bank
below the active learning minimum. Partial banks can never open a gradient
gate: an admitted row must come from a completed, summarized bank.

The runner's dense learning rate is an explicit command-line parameter and is
recorded in both the command journal and update receipt. This permits a rejected
role to roll back independently and retry from its last viable checkpoint with
a smaller full-parameter step without changing dtype or using LoRA. Every
candidate is still evaluated in the real coordinator/child interaction before
promotion.

The controller state is an append-only SHA-256-chained event log. The executor also writes an append-only command journal, immutable configs, hash-locked source/replay manifests, update receipts, and per-request routing audits. A partial evaluation or training directory is never overwritten; a restart advances to a new attempt suffix.

## Learning-ramp evidence

A dense update is not itself evidence of learning. The role-local ramp is the sequence of fresh-key post-update admissions at successively stricter phases. The joint harness result is valid only when the routing audit proves that both checkpoints served successful requests in the same evaluation. Report coordinator and child phase/qualifier histories separately, alongside the paired checkpoint hashes.

The first proof target is one complete pair update followed by fresh-key
evaluation at the controller's child and yield targets. The open-ended runner
continues until its explicit stop file is created.

## Recovered frontier (2026-08-23)

The protected pair is two distinct full-weight checkpoints:

- Coordinator `C4`: `d3c852479335151840ed901f61a2997cc7d5235569ab0e14dff0332201657507`
- Child `K4`: `55cebb265ddc513b173dab2c5e4f4366f7d59ff3daa24f0bbf4d2bdfc00b89a1`

Fresh-key requalification admitted `K4` at `e0c25_inline_evidence` with four
complete qualifying trajectories on four distinct task keys (five of six raw
harness passes). `C4` remained asymmetric: strict uncapped yield scored one of
six in the first post-update bank and zero of six in the final fresh
requalification, while capped exact-child yield scored six of six. This is a child-side
curriculum ramp, not yet a coordinator-side strict-yield ramp.

A second `1e-6` full-update pair was rejected. `K5` produced only three
qualifying/distinct trajectories at its parent `e0c2` rung, versus `K4`'s six of
six. Independent composition was then tested with `C5 + K4`; five complete
strict-yield trajectories all failed, so even a perfect missing sixth episode
could not reach the four-trajectory floor. The result rejects `C5` independently
and demonstrates why the role checkpoints should be trained and assessed
separately even though final harness promotion remains pair-aware.

The follow-on coordinator-only lineage starts from C4 with K4 frozen. Its event
log records `trainable_roles=["coordinator"]`; every authorized update receipt
must therefore contain coordinator `1`, child `0`. It begins at strict `e0d3`,
leaks down to capped `e0d2` only after a failed gate, and then retests strict
yield plus the capped parent-retention rung on fresh keys.

## Stop and recovery

Creating the configured stop file ends the runner after its current atomic action. A non-blocking lock prevents duplicate runners. On restart, the event log is authoritative; existing artifacts are reused only when their contents match the planned action exactly.

## Open-flow continuation (2026-08-24)

The owner authorized an open-ended capability-growth run and separated learning
admission from canonical promotion evidence:

- one complete, error-free success on one distinct task key may authorize a
  full-parameter role step;
- four complete successes on four distinct keys remain the promotion-grade
  evidence threshold;
- failed, incomplete, provider-error, or routing-invalid rows remain
  untrainable;
- `--open-ended` ignores numeric action/evaluation/update budgets and stops only
  at the explicit stop-file boundary.

The coordinator model initially also trained a scaffold-imitation Environment Designer objective. Each
admitted coordinator bank exports the same number of answer-free designer rows
as interaction rows. The target is the exact training scaffold that produced the
admitted trajectories. Designer leakage fades by exactly one adjacent stage per
admitted coordinator update:

1. exact action scaffold;
2. exact root action with a less scripted child contract;
3. spawn contract without exact code;
4. ownership and passive-yield contract;
5. minimal strategy hint.

While a smaller designer step remains, the controller holds the interaction
phase fixed instead of tightening both axes at once. Parent-rung retention and
role-local rollback continue to apply. This static leak ladder was the
capability-growth bootstrap. It is superseded by the causal coevolution
continuation below; its rows remain only as historical replay anchors.

The continuation started from C5/K5 after one fresh 4/4-admitted pair update:

- C5: `9941864dbde68d93fa75e58182fba06a2fd7ddff7aca807d5e4d1a25810c70d2`
- K5: `0996038492ef7bc18cdee92779ca91baac1dc4e46c3b3f56aa68cb4025d0acc7`

K5 then scored 4/6 on fresh `e0c275_inline_location` keys before the open-flow
policy was activated. The append-only policy event head is
`fd6c0c866507d45690eee99d8b51ee5a2ecc02e01b43ad3b0fb50c1ddf5c142f`.

## Causal environment coevolution continuation (2026-08-24)

The `--coevolution` executor implements a constrained, executable SPADE-style
loop around the existing Prime Agents environment:

1. The current coordinator checkpoint acts as Environment Designer and emits
   two answer-free declarative overlays with root/child contracts and separate
   privileged hints. It sees public task prompts, environment memory, and three
   deterministically sampled protocol excerpts from a hash-locked snapshot of
   the official Prime Agent documentation. The snapshot is pinned to upstream
   commit `9e49b73dd46908b3e400f4780b46a90daef69052`; oracle values and private
   evidence are excluded. Every generation records the corpus hash and sampled
   document IDs.
2. Validation rejects malformed schemas, reserved/private headers, task or
   reward replacement, exact terminal-answer leakage, and duplicate designs.
   The verifier propagates the generated child contract into the actual forced
   child-spawn request, so this is a causal intervention on both role contexts.
3. Each overlay receives three fresh tasks. The same task/model/seed pairing is
   executed once without the hint and once with it, with no optimizer update
   between arms.
4. Designer reward is `0.4 * positive_hint_regret + 0.6 * difficulty_plateau`.
   Raw regret is floored at zero and normalized at `0.15`; the flat-top
   no-hint success band is `[0.4, 0.6]` with a `0.25` ramp.
5. All valid scored overlays enter an append-only, hash-chained environment
   memory. Positive designs and too-easy/too-hard negatives condition later
   generations.
6. Complete qualifying trajectories from the better paired arm may augment the
   matching role's replay. Failed or incomplete trajectories remain
   untrainable. A positive-reward Designer generation enters coordinator replay
   only after at least one coordinator checkpoint change, giving a delayed
   reward-filtered SFT update.

The role updates remain one-step full dense updates. This is deliberately a
causal coevolution experiment, but it is not an exact reproduction of SPADE's
GRPO update and it does not permit arbitrary generated Python verifier code.
The generated DSL is compiled into the production Prime Agents task bootstrap
and child-spawn hook so it stays safe, paired, and recoverable.

The continuation started at the clean stopped boundary C18/K13:

- Coordinator C18: `cd61be4fbe3acec29bf5ef16af0b10a2c5fed21133ace4760cabf33d5a263f21`
- Child K13: `71d7e55e9f0a402a8e13c4a2d6e238f6aa330d3d4932f8015a0f99d5e9168465`

At that boundary the admitted child source contained one qualifying trajectory
and the final in-flight yield bank contained three. The existing open-flow
training rule therefore authorized both role updates, while the unchanged
four-trajectory promotion threshold remained unsatisfied.

Durable coevolution state lives in `spade-coevolution-*/generation`, paired
result directories, `SCORE.json`, and `coevolution-memory.jsonl`. A batch is
atomic only after `PAIRED_EVALUATIONS_COMPLETE`; a partial batch must be
archived or explicitly recovered before restarting the same event head.

## Role-scoped dense GRPO continuation (2026-08-25)

`scripts/run_q35_2b_role_grpo_autonomous_v1.py` is the current open-ended
controller. It alternates one coordinator update and one child update. Each
cycle samples a GRPO group of eight and authorizes at most one full-parameter
optimizer update for that cycle's role; LoRA and mixed-role optimizer steps are
not part of this continuation. Coordinator rollouts are serialized to contain
runtime memory, while child rollouts use concurrency two. The zero-advantage
filter remains enforced, so the executor may sample replacement groups until
it has one trainable group or reaches a terminal no-update/failure receipt.

A successful dense update always advances that role's **frontier**, even if its
following admission fails. This is the deliberately aggressive coevolution
rule: the other role trains against the most recent learned counterpart rather
than waiting through repeated static rungs. **Promotion** is separate and
advances only after a six-episode fresh-key admission contains at least four
complete, error-free qualifying trajectories on four distinct task keys. The
constant `PROMOTION_MINIMUM = 4` is not relaxed by the open-flow policy.

The authoritative state is
`grpo-autonomous-v1/events.jsonl`, an fsynced, SHA-256-chained event stream.
Cycle and task identities are deterministic: training starts at
`start_index + 100 * cycle`; its admission starts fifty keys later. Before
starting work, the controller holds a non-blocking state lock, requires idle
GPUs, and refuses every pre-existing config, receipt, output, or router target.
On restart it reconciles an unterminated `train_started` or
`evaluation_started` event first:

- if the exact labeled process is still active, wait;
- if it has exited and its immutable terminal artifact validates, append the
  missing terminal event;
- otherwise append a failure event and advance to the other role on a fresh
  cycle identity.

Never manually relaunch the training or admission command for an event whose
start record already exists. Restart only the controller with the original
arguments and state directory; reconciliation prevents a second evaluation or
optimizer update for that identity.

The companion watchdog is
`scripts/watch_q35_2b_spade_dual_dense_v1.sh`. The live configuration polls
every ten seconds, permits at most three restart dispatches for one unchanged
event head, waits while any GPU process remains, and guards the aggregate
`PRIME-RL::EnvServer` process at 48 GiB RSS. Crossing the guard creates the
explicit stop file before killing the offending EnvServer, so the watchdog
cannot silently loop on a memory fault. The scorer alias-expansion fix bounds
each alias head to one expansion; ACP stderr/session-update caps and
`PAGER=PYTHONPAGER=cat` are secondary containment.

Safe recovery order:

1. Inspect the event-chain tail, controller/watchdog processes, GPU PIDs,
   runtime containers, and the exact labeled train/evaluation artifacts.
2. Do not launch anything while the labeled action or any GPU process remains.
3. Validate the whole event chain with `load_events`; do not edit or truncate
   it. Confirm the stop file is absent only when continuation is intended.
4. Restore the exact source tree, model/output/result/artifact roots, and
   original controller arguments. Start the controller through the dedicated
   tmux launcher pane; let `_reconcile` decide the unfinished action.
5. Start one watchdog only after the controller is present. Confirm the next
   appended event links to the prior `event_sha256` and that no cycle contains
   duplicate start or terminal kinds.

### Rolling Hugging Face recovery slots

The companion `scripts/sync_q35_2b_latest_hf_v1.py` mirrors the newest role
cycle that has both `train_completed` and `evaluation_completed` events. It
validates the remote `STABLE` checkpoint and event-ledger SHA-256, verifies the
local copy and Hub LFS SHA-256, and then super-squashes each private model repo
to one commit. The coordinator and child repositories are latest-only recovery
slots rather than historical checkpoint archives. A local non-blocking lock and
the Hub hash check prevent duplicate uploads after watcher or host restarts.

Run the long-lived companion through
`scripts/watch_q35_2b_latest_hf_v1.sh`; credentials stay in the local env file
and are not copied to the rented training host. A sync is eligible only after
the admission terminal event exists, so it cannot race or duplicate the live
evaluation. The manifest in the local sync-state directory pins the event
sequence, model hash, single HF commit, and admission result for each role.
On macOS, install the launchd runtime and its HF-only credential file outside
privacy-protected `Documents` (for example under `~/.local/share` and
`~/.config`) instead of granting launchd Full Disk Access.
