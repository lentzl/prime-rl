# Latent Coordinator Architecture v1

**Status:** design and prospective experimental plan  
**Execution authorization:** **none** — documentation only until the current specialist-worker lane reaches a preserved checkpoint and the Owner/Strategist explicitly opens this branch  
**Prepared:** 2026-09-04  
**Target branch:** `exp/q35-2b-root-role-contract`  
**Current protected cognitive substrate:**

- coordinator C: `e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47`
- terminal worker H: `77980e247bbccd6463ddda02cd42d2c357e15f8ec1ad0ea84627e008a8674a1e`

This document defines the next architectural program after the current specialist-routing and specialist-worker work is stabilized. It does not supersede that work and does not authorize an optimizer run.

---

## 1. Executive decision

The next architecture should be developed as **two independent experiments followed by one combination**:

1. **Coordinator-to-coordinator latent workspace handoff** — move useful continuous cognition from a frozen parent C to a frozen child C while preserving the typed textual control contract.
2. **Recurrent coordinator reasoning sidecar** — give a frozen C additional local latent depth through a small trainable recurrent module.
3. **Unified latent recursive coordinator** — use the same workspace as both the local reasoning substrate and the inter-node communication substrate.

A later fourth experiment may add strict **Cache-to-Cache (C2C)** fusion on Qwen3.5's conventional full-attention layers. Raw KV-cache C2C is not the first implementation because Qwen3.5 uses a hybrid Gated DeltaNet/full-attention backbone; soft continuous workspace vectors are a cleaner interface that every layer can consume.

The intended long-term separation is:

```text
CONTROL PLANE — harness-owned
identity, role, parent/child relation, objective, ownership, permissions,
budget, legal routing, exactly-once delivery, lifecycle, provenance

COGNITIVE PLANE — learned
small continuous workspace, task representation, hypotheses, abstractions,
local recurrent refinement, high-bandwidth handoff
```

Text remains the authoritative and auditable control/result channel. Latent state augments it; it does not silently replace protocol semantics.

---

## 2. Why this belongs in the applied program

The system's north star is not merely to prove that tiny models can communicate. It is to maximize useful capability under scarce local inference resources.

The present text-only recursive path repeatedly pays a reconstruction tax:

```text
parent understands task
→ serializes a child prompt
→ child reconstructs context and intent
→ child performs work
→ serializes a report
→ parent reconstructs the child's state
```

That tax grows with organizational depth. A high-bandwidth latent channel could preserve more of the cognition already paid for, while a recurrent sidecar could let a coordinator spend more sequential compute locally before allocating another node or loading a larger model.

The two axes are complementary:

```text
LOCAL DEPTH
one coordinator + recurrent latent refinement

DISTRIBUTED BREADTH / ORGANIZATIONAL DEPTH
multiple C/H/specialist nodes + latent handoffs
```

The applied question is therefore:

> Can a fixed, massively pretrained small language model be surrounded by trainable latent machinery that improves task success or reduces GPU-seconds per success, without changing the protected language-model weights?

If yes, the system can grow useful cognition around a stable pretrained core rather than retraining a foundation model for every architectural improvement.

---

## 3. Evidence and design constraints

### 3.1 Latent Recurrent Thoughts makes the core mechanism concrete

[Latent Recurrent Thoughts (LRT)](https://arxiv.org/abs/2609.01117) keeps a pretrained Qwen decoder frozen while training small modules that produce and recurrently refine continuous latent vectors. The projected vectors are inserted into the frozen decoder as soft input embeddings. The official implementation is available in [`czl-david/latent-recurrent-thoughts`](https://github.com/czl-david/latent-recurrent-thoughts).

Implementation details directly relevant to this plan:

- the decoder weights are frozen;
- continuous vectors are inserted through `inputs_embeds` during training and an embedding-input serving path during evaluation;
- ordinary next-token cross-entropy backpropagates through the frozen decoder into the trainable latent modules;
- the projection normalizes solver states, maps them into the decoder embedding dimension, and optionally rescales them to the mean norm of the decoder's token embeddings;
- the recurrent solver is adapted from TRM/HRM and supports a fast/slow iterative latent state.

LRT is important evidence, not a drop-in solution. Its released pipelines are task-specific rather than a universal coordinator reasoner. Our target is a reusable coordinator-side module trained across decomposition, routing, synthesis, and recovery tasks.

### 3.2 Cache-to-Cache proves direct semantic communication is trainable

[Cache-to-Cache: Direct Semantic Communication Between Large Language Models](https://arxiv.org/abs/2510.03215) and the official [`thu-nics/C2C`](https://github.com/thu-nics/C2C) implementation project and fuse a source model's KV cache into a receiver model while both LLMs remain frozen. The framework supports different model sizes and families, layer-wise projectors and gates, and preliminary multi-sharer fusion.

This supports the broader thesis that learned latent communication can outperform text-only communication. It also provides a later reference implementation for full-attention cache fusion.

The term **C2C** is overloaded in this project:

- `C → C` means coordinator-to-coordinator;
- **Cache-to-Cache** means the specific KV-cache communication method.

This document uses **workspace handoff** for the first coordinator-to-coordinator mechanism.

### 3.3 Qwen3.5's hybrid backbone favors soft workspace vectors first

The official [Qwen3.5-2B configuration](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json) has hidden size 2048 and 24 layers. The official [Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5) describes a hybrid stack with a 3:1 pattern of Gated DeltaNet linear-attention layers to conventional full-attention layers.

Conventional Cache-to-Cache naturally addresses KV state in the six full-attention layers. It does not by itself define how to transfer the recurrent/convolutional states of the other eighteen layers. Soft workspace embeddings avoid that mismatch: all Qwen layers consume them through the ordinary input path.

### 3.4 Recurrent depth needs its own optimization treatment

[DeepLoop: Depth Scaling for Looped Transformers](https://arxiv.org/abs/2607.13491) and [`lszshu/DeepLoop`](https://github.com/lszshu/DeepLoop) show that revisiting shared residual blocks changes the optimization dynamics. Their loop-aware residual parameterization uses, for effective unrolled depth `N`,

```text
alpha = (2N)^(1/2)
beta  = (8N)^(-1/2)
```

in the studied Post-LN sandwich architecture. DeepLoop is relevant only to a recurrent sidecar trained from initialization. It is not a post-hoc fix for pretrained e33, and it is not assumed to be the winning parameterization until ablated in our setting.

### 3.5 HRM/TRM supply useful recurrent and halting ideas, not a ready coordinator

The [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734) and [`sapientinc/HRM`](https://github.com/sapientinc/HRM) use fast and slow recurrent states and a halt/continue Q head. The public implementation repeatedly reinjects the task embedding into the fast state and uses truncated one-step gradients through a longer forward recurrence.

A critical implementation detail is that the released HRM code forces maximum steps at evaluation for batching, while learned halting is applied during training. Our production target requires real inference-time per-example halting or an equivalent bucketed implementation; copying the HRM head is not sufficient.

[Tiny Recursive Models](https://arxiv.org/abs/2510.04871) and LRT further suggest that a small recurrent module can be valuable without becoming another language model.

### 3.6 Frozen-LLM coprocessors are an established pattern

[Deliberation in Latent Space via Differentiable Cache Augmentation](https://proceedings.mlr.press/v267/liu25bc.html) trains a coprocessor around a frozen language model and writes latent information into the model's cache. [Coconut](https://arxiv.org/abs/2412.06769) treats hidden states as continuous thoughts. Together with LRT and C2C, these works support the central engineering premise: a frozen language model can remain the semantic/world-model substrate while trainable latent modules supply additional computation or communication.

### 3.7 What this plan deliberately does not assume

- It does not assume rumors about any closed frontier model are correct.
- It does not assume latent transfer is automatically safe or useful merely because the endpoints share an architecture.
- It does not assume a fixed two-pass loop is the desired endpoint.
- It does not assume a learned workspace is interpretable.
- It does not assume raw KV cache transfer is superior to soft workspace vectors for Qwen3.5.
- It does not assume local recurrent depth should be applied to every worker. The first target is root and non-root coordinators, where one better decision affects an entire downstream graph.

---

## 4. Current baseline and invariants

All first-generation experiments use the preserved Qwen3.5-2B pair:

```text
C = e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47
H = 77980e247bbccd6463ddda02cd42d2c357e15f8ec1ad0ea84627e008a8674a1e
```

Both hashes remain immutable reference points.

The current intended tight stack remains part of the product. The harness owns:

- role and session identity;
- parent/caller provenance;
- legal depth and budget;
- resource ownership and permissions;
- typed delegation and return;
- exactly-once delivery;
- passive waiting and fan-in;
- lifecycle closure and timeout behavior;
- artifact and checkpoint audit.

The latent branch may not weaken those invariants. It is allowed to add a cognitive payload only after the runtime has accepted the corresponding control-plane action.

Protected D0-D3 and topology suites remain regression tests. A bridge or sidecar candidate that fails its own prerequisite gate does not spend the protected full regression battery.

---

## 5. Common latent workspace

Define a versioned latent object

```text
W ∈ R^(K × d_w)
```

with initial defaults:

```text
K   = 8 workspace slots
d_w = 256 dimensions
```

These values are deliberately small. Eight 256-dimensional BF16 vectors are only 4 KiB of raw tensor payload before metadata. The purpose is to test whether a compact cognitive packet can carry useful information, not to copy the full parent context invisibly.

### 5.1 Workspace encoder

The first encoder should use learned-query resampling rather than naive mean pooling:

```text
parent hidden sequence H ∈ R^(T × 2048)
        ↓  source LayerNorm / RMSNorm
project keys and values 2048 → 256
        ↓
8 learned queries cross-attend to H
        ↓
W ∈ R^(8 × 256)
```

For v1, use the final hidden states from a bounded, deterministic capture span. A practical first choice is the last 64-128 non-padding positions at the accepted delegation boundary.

After the basic channel works, ablate features from the six full-attention boundaries, approximately layers 3, 7, 11, 15, 19, and 23 under zero-based indexing, and combinations of intermediate/final states. Intermediate extraction is not part of the first gate.

### 5.2 Workspace decoder

Map the workspace back into e33's 2048-dimensional embedding space:

```text
W
 ↓ LayerNorm
linear 256 → 2048
 ↓ optional embedding-shell rescaling
receiver gate g
 ↓
8 soft latent input vectors
```

Follow the LRT implementation pattern of rescaling projected vectors toward the mean norm of the frozen Qwen input embeddings. Use a receiver gate initialized at or very near zero so the untrained module begins as a no-op rather than corrupting the baseline.

### 5.3 Injection point

The child continues to receive the normal typed textual objective. Insert the soft vectors after the child-visible objective and before the assistant generation opening:

```text
system/control contract
user child objective
<8 continuous workspace vectors>
assistant generation begins
```

The exact chat-template boundary must be verified token-for-token against the live Qwen3.5 template. Training and inference must use the same placement, sampling contract, and thinking/non-thinking mode.

### 5.4 Workspace metadata

Every workspace object must carry non-neural metadata:

```text
workspace_id
workspace_schema_version
producer_session_id
producer_role
producer_checkpoint_hash
bridge_checkpoint_hash
scope_id
parent_session_id
creation_depth
allowed_receiver_sessions or receiver predicate
source_resource_labels / taint labels
created_at / TTL
shape, dtype, checksum
```

The runtime validates metadata before tensor delivery. A tensor without valid provenance is unusable.

---

# Phase A — coordinator-to-coordinator latent workspace handoff

## 6. A0: implementation and no-op equivalence

### Question

Can the runtime construct, store, validate, and inject a workspace without changing baseline behavior when its gate is disabled?

### Frozen

- parent e33;
- child e33;
- H176;
- current tight runtime semantics.

### Trainable

None.

### Required checks

1. `gate = 0` reproduces text-only receiver logits within a preregistered numerical tolerance.
2. Only the intended eight positions are replaced by continuous embeddings.
3. Padding and attention masks remain correct.
4. Invalid scope, receiver, checkpoint, shape, dtype, or checksum fails closed.
5. No workspace from one task can be attached to another without an explicit test-only override.
6. Model hashes remain unchanged.
7. Captured tensors and metadata receive deterministic artifact hashes.

### Prototype runtime

Do not begin by modifying the production vLLM path. The bridge training prototype needs hidden-state access and gradient flow through the frozen receiver. Use Hugging Face Transformers or an equivalent instrumented local path first.

The official LRT code demonstrates two useful deployment patterns:

- `inputs_embeds` for frozen-decoder training;
- SGLang's embedding-input `/generate` path for evaluation.

A separate prototype backend is acceptable until the channel passes causal admission. Production integration is a later optimization.

---

## 7. A1: causal semantic transfer unit test

### Question

Can correct parent latent state convey information that the child cannot obtain from text, and does the effect disappear when the latent is shuffled or invalid?

This is the most important first gate. It is not a production benchmark; it is a semantic unit test for the channel.

### Dataset design

Use a split-information construction that prevents trivial final-answer encoding:

```text
parent sees evidence X but does not see the downstream query f
child sees query f but does not see X
correct answer is f(X)
```

The parent is asked to prepare a reusable handoff representation of `X`. Several different child queries should be paired with the same evidence packet. Therefore the parent cannot merely emit the answer to one known query.

Candidate families:

- keyed numerical records with unseen query operators;
- small tables with later join/filter/aggregate requests;
- source/config snippets with later structural questions;
- graph or ownership facts with later routing/composition questions;
- symbolic mappings with held-out labels and transformations.

Use randomized values, identifiers, orderings, and surface language. Keep held-out structures and query templates out of training.

### Frozen

- parent e33 feature extractor;
- child e33 decoder;
- all current operational policies.

### Trainable

- workspace encoder `E_C`;
- workspace decoder `D_C`;
- one receiver gate.

Expected scale is approximately 1-3M parameters for the first implementation, depending on the resampler block.

### Objective

Detach the parent hidden states. Backpropagate ordinary teacher-forced next-token cross-entropy through the frozen child into `E_C`, `D_C`, and the gate.

No gradient reaches e33. Log parameter names and verify that every non-bridge parameter has zero update.

Start with the simplest loss. A contrastive or margin term between correct and shuffled workspaces may be added only if ordinary causal training is connected but weak; it should be a separately preregistered rung.

### Evaluation arms

For every held-out key run:

1. **TEXT/NO-LATENT:** child receives only its query.
2. **ZERO:** correctly shaped all-zero workspace.
3. **NOISE:** random workspace matched to the correct workspace's norm distribution.
4. **SHUFFLED:** workspace from another task in the same family.
5. **CORRECT:** workspace generated from the matching parent evidence.

The child text is otherwise identical.

### Proposed admission floor

Freeze exact values before the first model call. The initial recommendation is:

- at least 16 held-out tasks across at least four semantic families;
- at least four distinct `NO-LATENT fail → CORRECT success` paired recoveries across at least two families;
- a prospectively fixed, material CORRECT-vs-SHUFFLED success gap;
- SHUFFLED and NOISE may not reproduce most of the CORRECT gain;
- all scope/provenance checks pass;
- zero operational-model updates;
- no answer-bearing text enters the child.

If CORRECT and SHUFFLED perform similarly, the channel has not demonstrated semantic specificity and is rejected.

### Stronger anti-shortcut probe

After the first pass, reuse the same parent evidence with a held-out child query. If the latent only encoded a memorized answer, transfer should fail; if it encoded useful evidence/structure, the new query should remain solvable.

---

## 8. A2: real recursive coordinator handoff

### Question

Does a latent workspace improve genuine e33 parent-to-e33 child recursion when the full typed text objective is already present?

### Task bank

Use fresh tasks from the existing causal recursive families plus new coordinator-heavy variants:

- D2 and D3 decomposition decisions;
- non-root manager continuation decisions;
- ambiguous but causally resolvable decomposition;
- manager fan-in and synthesis;
- tasks where all resources are reachable but e33 currently makes a local planning error;
- later, utility-based specialist selection after that lane is stable.

Include easy/direct anchors to detect indiscriminate latent interference.

### Evaluation arms

```text
T    text-only production control
L    latent-only where the control contract can remain unambiguous
T+L  normal text contract plus matching latent workspace
T+S  normal text contract plus shuffled latent workspace
```

`T` remains the authoritative production baseline. `L` is diagnostic and must never remove identity, ownership, budget, or lifecycle metadata.

### Separate claims

#### Capability admission

A capability claim requires fresh paired cases in which `T` hard-fails and `T+L` cleanly succeeds, while `T+S` does not reproduce the gain and protected behavior remains intact. A proposed first useful signal is at least four distinct paired recoveries.

#### Efficiency admission

An efficiency claim may be admitted without a capability gap if success is non-inferior under a prospectively frozen margin and the latent arm materially reduces one or more of:

- handoff text tokens;
- child re-orientation tokens;
- GPU-seconds per successful task;
- wall-clock latency;
- unnecessary recursive nodes.

Candidate provisional thresholds are 20% less communication/re-orientation token work or 15% fewer GPU-seconds per success. These are planning values, not post-hoc gates.

### Required metrics

- exact answer and protocol success;
- parent and child action correctness;
- root versus non-root cells;
- D0/D1/D2/D3 family breakdown;
- correct-vs-shuffled causal gap;
- generated text tokens by node and phase;
- prefill and decode time;
- bridge time;
- GPU-seconds;
- peak VRAM;
- workspace bytes;
- number and depth of activated nodes;
- failure taxonomy: ignored latent, harmful latent, wrong scope, malformed action, synthesis failure, lifecycle failure.

---

## 9. A3: ownership and information-flow policy

Latent tensors are opaque and may contain more source information than an explicit report. Treat them as at least as sensitive as the full producer context.

### Initially allowed

```text
C_parent → C_child
```

only when the information represented by the workspace is permitted in the child's declared scope.

### Initially forbidden

```text
H with private delegated evidence → parent C full latent state
private child workspace → sibling
workspace from one user/task → another user/task
unversioned or unprovenanced tensor injection
```

A worker-to-parent latent return risks bypassing the information-reduction boundary by carrying a near-lossless representation of a private document. Keep the explicit typed result as the only authoritative upward payload until a separate abstraction/sanitization mechanism is designed and tested.

### Fail-closed rules

- scope mismatch rejects delivery;
- expired workspace rejects delivery;
- unknown bridge/model hash rejects delivery;
- wrong receiver rejects delivery;
- cross-task reuse is prohibited outside explicit shuffled-control runs;
- latent artifacts are not persisted by default;
- diagnostic persistence must be encrypted or otherwise handled as sensitive task data in a production deployment.

---

# Phase B — recurrent coordinator reasoning sidecar

## 10. B0: objective

### Question

Can a small trainable recurrent module give frozen e33 useful **local depth** at root and non-root coordinator nodes?

Do not combine this with inter-node handoff initially. Phase B uses a workspace generated from the coordinator's own current task state.

### Why coordinators first

A worker improvement affects one bounded leaf. A coordinator improvement can change decomposition, topology, specialist selection, evidence flow, and synthesis across the whole graph. The expected return per additional recurrent cycle is therefore likely highest at root and non-root C nodes.

Workers remain unchanged in the first recurrent experiment.

---

## 11. B1: architecture

Initial target size: **10-30M trainable parameters**.

```text
frozen e33 hidden states
        ↓
workspace encoder E_C
        ↓
U = 8 × 256 task anchor
        ↓
recurrent reasoner R_phi
   fast state z_L
   slow state z_H
   original U reinjected every fast update
        ↓
refined workspace W_T
        ↓
workspace decoder D_C
        ↓
8 soft vectors
        ↓
frozen e33 emits action / synthesis
```

A candidate update schedule is:

```text
for outer cycle t = 1..T:
    for inner step l = 1..L:
        z_L ← R_L(z_L, z_H + U, time_features)
    z_H ← R_H(z_H, z_L, time_features)
```

`U` is reinjected rather than supplied only at initialization, following the useful pattern in HRM/LRT. This reduces drift away from the original problem.

### Pass specialization without hard-coded phase labels

The same recurrent weights may perform different effective computations because the state changes across cycles. Permit, but do not prescribe, pass specialization.

Supply small time features such as:

- normalized cycle index;
- normalized used-budget fraction;
- root/non-root role embedding;
- optional current depth and remaining system budget.

Avoid a rigid rule like “cycle 1 understands, cycle 2 plans, cycle 3 verifies.” The model should discover useful phases. Avoid large per-cycle parameter sets that turn recurrence back into untied depth.

### Output safety

Use a bounded residual correction and a gate initialized near zero:

```text
W_out = U + g · clamp_or_normalize(delta_phi)
```

This gives the sidecar a stable no-op starting point and lets the frozen e33 baseline remain reachable.

---

## 12. B2: training ladder

### Frozen

- e33;
- H176;
- runtime contracts.

### Trainable

- workspace encoder/decoder if not imported from admitted Phase A;
- recurrent reasoner;
- output gate;
- later, halting/value head.

### Training signal

Use coordinator-local targets and executable outcomes:

- correct `solve_owned` / `delegate_terminal` / `delegate_coordinator` decisions;
- decomposition/scoping content;
- non-root continuation decisions;
- manager synthesis and return;
- utility-based expert/resource choices after those interfaces stabilize.

The first rung may use teacher-forced accepted outputs through frozen e33. Later rungs may use verifier/outcome learning while keeping e33 frozen.

Prefer examples near the current frontier: tasks that e33 sometimes solves, tasks where extra local planning should help, and tasks with clean causal verification. Do not begin with structurally unreachable tasks.

### Gradient strategy

1. Begin with full backpropagation through 2-4 recurrent cycles to establish correctness.
2. Use activation checkpointing as needed through the frozen decoder.
3. Compare full and truncated gradients at a depth where full BPTT is still feasible.
4. Only then extend to 8-16 cycles using truncated or one-step gradient methods.

The HRM/LRT pattern of long no-gradient recurrence plus a final gradient-visible cycle is a useful reference, not an assumption.

### DeepLoop ablation

If the reasoner uses a weight-shared Transformer transition, compare:

- standard Pre-LN recurrent block;
- a Post-LN sandwich block;
- DeepLoop-style loop-aware scaling.

Keep the reasoner size, data, optimizer, and recurrent depth matched. Do not attribute a gain to recurrence if it is actually caused by a different parameter count or training budget.

---

## 13. B3: controls for a local-depth claim

Use matched trainable parameter budgets:

1. **BASE:** e33 alone.
2. **STATIC:** e33 plus static latent proposer/adapter.
3. **FFN:** e33 plus a non-recurrent sidecar with approximately the same parameters as the recurrent module.
4. **RECURRENT:** e33 plus the recurrent sidecar.

A true local-depth result requires `RECURRENT > FFN` at matched parameter count and comparable training data/compute.

Then evaluate the same recurrent checkpoint at fixed cycle budgets:

```text
T ∈ {1, 2, 4, 8, 12, 16}
```

The desired pattern is:

- easy tasks saturate early;
- harder coordinator tasks continue improving for additional cycles;
- some tasks eventually overthink or oscillate, motivating halting;
- additional cycles do not merely produce longer text, because all extra work occurs before decoding in latent space.

Track state-change norms, output stability, task utility, and cycle-to-cycle action changes.

### Proposed admission

The first sidecar is admitted only if:

- recurrent matched-parameter performance exceeds the non-recurrent control on fresh coordinator-heavy tasks;
- gains appear at both root and non-root C positions or the limitation is explicitly recorded;
- shuffled/ablated task anchors remove the gain;
- e33/H176 remain bitwise unchanged;
- protected recursive behavior is retained;
- measured GPU cost is recorded so a capability gain cannot hide an impractical systems cost.

---

## 14. B4: adaptive inference-time halting

Do not begin with learned halting. First measure whether additional cycles buy capability.

### Stage 1: deterministic fixed-depth curves

Collect per-task trajectories across cycle budgets. Identify:

- earliest cycle yielding a stable correct action;
- tasks that improve late;
- tasks that regress after being correct;
- convergence or oscillation patterns in `z_H` and `z_L`.

A simple diagnostic halt rule may use several consecutive small relative changes:

```text
||z_H(t) - z_H(t-1)|| / max(||z_H(t-1)||, eps) < threshold
```

This is not assumed to be semantically reliable; it is an engineering baseline.

### Stage 2: cost-aware learned halting

Train a head to estimate halt versus continue utility:

```text
continue iff E[additional task utility | current state]
             > lambda × expected next-cycle cost
```

The reward should include verified task outcome and public cycle cost. The target is not “think as long as possible”; it is “spend the cheapest amount of local cognition that works.”

### Stage 3: real inference behavior

Unlike the public HRM evaluation shortcut, production evaluation must actually stop examples at different cycle counts. Implement one of:

- per-example active masks with compacted batches;
- bucketing by next cycle;
- asynchronous per-request recurrence;
- a bounded batch scheduler that removes halted states.

Report the actual cycle distribution by task family, root/non-root position, and difficulty.

---

# Phase C — unified latent recursive coordinator

## 15. Combine only after A and B independently pass

Once workspace handoff and local recurrence each have causal evidence, use one common workspace schema:

```text
root e33 task state
        ↓ E_C
U_root
        ↓ recurrent refinement
W_root
        ↓ typed delegation + authorized workspace transfer
child e33 receives text objective + W_root
        ↓ recurrent refinement
W_child
        ↓ D_C
child e33 acts
```

The same workspace becomes:

- a compact representation of current task understanding;
- the state operated on by the recurrent sidecar;
- the cognitive payload transferred between coordinator nodes.

Because root and child use the same e33 checkpoint, the same `E_C`, `D_C`, and recurrent reasoner should be shared initially. Role and depth are metadata/features, not separate large policies.

### Core Phase C question

> Can local latent depth and high-bandwidth inter-node handoff compose, producing a recursive coordinator system whose capability or efficiency exceeds either mechanism alone?

Use a factorial comparison:

```text
text only, no sidecar
text + workspace handoff
sidecar + text only
sidecar + workspace handoff
```

This separates additive from interaction effects.

### Success criteria

- fresh D2/D3 and applied coordinator-heavy tasks;
- correct-latent causal controls;
- matched system budgets;
- no ownership/provenance violation;
- no regression into indiscriminate recursion;
- measured local-cycle allocation and external-node allocation;
- evidence that the combined system chooses between thinking more locally and organizing more externally.

---

# Phase D — strict Cache-to-Cache extension

## 16. Why this is later

The workspace interface is architecture-agnostic and naturally covers Qwen3.5's hybrid layers. Strict Cache-to-Cache may still add value because the full-attention layers contain richer position-specific context than eight workspace slots.

After Phase A is admitted, test direct cache fusion on the six full-attention layers only.

### Same-model advantages

For e33 → e33:

- architecture is identical;
- hidden/head dimensions are identical;
- tokenizer is identical;
- weights are identical.

Initialize source-to-target mappings close to identity and learn only fusion/projector corrections and gates.

### Comparison

```text
text only
soft workspace only
full-attention Cache-to-Cache only
soft workspace + Cache-to-Cache
shuffled-cache controls
```

### Questions

- Does cache fusion add capability beyond the compact workspace?
- Does it reduce the number of workspace slots needed?
- Is the latency/memory cost worthwhile?
- Can preliminary C2C multi-sharer fusion help coordinator fan-in?
- Does ignoring DeltaNet state leave significant information on the table?

DeltaNet-state transfer is a separate future research problem. Do not block the first architecture on it.

---

## 17. Implementation plan in `prime-rl`

Proposed code organization, subject to adaptation to current conventions:

```text
prime_rl/latent/
    workspace.py          # tensor + metadata schema and validation
    encoder.py            # learned-query resampler
    decoder.py            # projection, norm-shell, receiver gate
    recurrent.py          # fast/slow sidecar and fixed-depth runner
    halting.py            # later value/halt head
    policy_adapter.py     # frozen-Qwen inputs_embeds integration
    provenance.py         # scope/receiver/hash checks

scripts/latent/
    build_transfer_bank.py
    cache_parent_features.py
    train_workspace_bridge.py
    eval_workspace_bridge.py
    train_recurrent_sidecar.py
    eval_recurrent_depth.py

experiments/qwen35-2b-latent-coordinator-v1/
    *-plan.json
    *-result.json
    manifests/
```

Do not vendor external repositories wholesale. Reuse small, attributed components only where licensing and architecture fit. LRT is MIT; C2C and DeepLoop are Apache-2.0.

### 17.1 Feature extraction

For early bridge training, load the protected e33 checkpoint through an instrumented Transformers path and request hidden states. Cache detached parent features for the synthetic A1 bank to reduce repeated compute.

The cache manifest must record:

- exact parent checkpoint hash;
- tokenizer/chat-template hash;
- capture-layer and token-span specification;
- source task ID and split;
- dtype/shape/checksum;
- code commit.

Do not reuse a feature cache after any of these change.

### 17.2 Receiver training

Load e33 frozen. Build child prefixes with `inputs_embeds`, preserving the exact chat template around the continuous positions. Compute loss only on accepted target outputs.

Use automatic assertions:

- all e33 parameters `requires_grad=False`;
- no e33 optimizer entries;
- bridge gradient finite and nonzero;
- exact checkpoint rehash after training;
- gate-zero baseline equivalence.

### 17.3 Inference prototype

Use direct Transformers generation or an embedding-input SGLang server first. Only integrate into the live recursive runtime after A1 passes.

The production interface should eventually resemble:

```python
workspace = encode_workspace(
    producer_session=parent_session,
    hidden_states=parent_hidden,
    scope=child_scope,
)

child = spawn_coordinator(
    objective=typed_objective,
    workspace=workspace,
)
```

The runtime, not the model, validates whether `workspace` may be delivered.

### 17.4 Versioning

Treat the workspace schema and bridge checkpoint as protocol versions. A receiver must reject an incompatible version rather than silently projecting with the wrong weights.

---

## 18. Experiment ledgers and reproducibility

Every run should have a prospective plan JSON and immutable result JSON.

Minimum plan fields:

```text
schema_version
experiment_id
status = preregistered
base code commit
parent/receiver checkpoint hashes
tokenizer/template hash
bridge architecture and parameter count
capture layers/span
workspace K and d_w
injection position
train split and held-out split hashes
optimizer and schedule
sampling boundary
evaluation arms
admission criteria
protected regression policy
```

Minimum result fields:

```text
status = admitted | rejected | infrastructure_invalid
all input hashes
bridge checkpoint hash
model rehashes
per-arm task outcomes
paired causal gaps
per-family/root-nonroot breakdown
GPU-seconds, latency, peak VRAM, tokens
workspace norm/statistics
failure taxonomy
artifact tree hash
promotion decision and reason
```

Persist correct and shuffled provenance so a later audit can reconstruct exactly which workspace was delivered to which session.

---

## 19. Failure branches

### A1 fails because the child ignores the workspace

- verify that text alone is genuinely insufficient;
- verify injection placement and norm shell;
- inspect receiver gate and gradient;
- reduce target length to a small typed decision;
- consider a paired correct-vs-shuffled contrastive objective;
- do not unfreeze e33 as the first response.

### A1 succeeds only by final-answer encoding

- enforce split-information tasks where the parent does not know the downstream query;
- pair multiple unseen queries with the same parent evidence;
- test held-out transformations;
- reject any claim of general communication until this passes.

### Correct and shuffled workspaces both help

Likely causes include text leakage, family priors, or a generic learned prompt effect. Reject the semantic-channel claim and repair the bank.

### Workspace harms text-only successes

- check zero-init/no-op gate;
- reduce injected norm;
- add receiver gating conditioned on task/role;
- require the text-only path to remain available;
- do not accept aggregate gains that hide broad regressions.

### Recurrent sidecar equals the matched FFN

There is no evidence for useful local depth. The static latent adapter may still be valuable, but do not claim recurrence. Examine harder yet reachable coordinator tasks before increasing parameters.

### More recurrent cycles worsen performance

Record overthinking/oscillation. This motivates halting or better recurrent training; it does not justify post-hoc selection by answer correctness. Use answer-free convergence/value criteria.

### Sidecar overfits one synthetic family

Hold promotion. Expand to semantically distinct coordinator tasks and real applied workloads. The target is a reusable coordinator reasoner, not a task-dedicated puzzle module.

### Systems overhead erases task gains

A scientifically valid capability result may still be an applied rejection. Preserve it as evidence, then optimize batching, feature reuse, and serving only if the capability margin warrants the engineering cost.

---

## 20. Relationship to specialist population work

The current specialist-worker lane remains first in execution order. The latent branch should begin only after that lane has a preserved checkpoint or a clearly documented hard boundary.

The first latent experiments deliberately use C → C only, avoiding a routing/worker-competence confound.

Later integration:

1. C → C shared workspace.
2. Recurrent sidecar for root and sub-C.
3. C → specialist H adapters from the common workspace.
4. Sanitized/typed H → C latent summaries, if ownership controls can be proven.
5. Heterogeneous 2B/4B/8B workspace adapters.
6. Multi-child latent fan-in.

Specialists derived from the same H lineage should require smaller adapter corrections than unrelated model families, but this is a hypothesis to test.

The utility router should eventually see latent task/workspace features in addition to public capability metadata, but expert selection remains judged by verified outcome and public cost, not hidden identity labels.

---

## 21. Applied resource accounting

Record from the first prototype:

- trainable bridge/sidecar parameter count;
- frozen model parameter footprint;
- active model instances;
- feature-extraction GPU-seconds;
- bridge training GPU-hours;
- inference GPU-seconds per successful task;
- peak VRAM and host RAM;
- communication text tokens;
- latent bytes;
- local recurrent cycles;
- external node count and depth.

The desired system may contain many stored experts and adapters. The optimization target is not minimal total stored parameters; it is useful capability per scarce **active** inference resource.

A likely practical split is:

- precompute parent features for A1;
- train the small bridge on one GPU while a frozen receiver occupies the same device with activation checkpointing as necessary;
- use the available two-GPU machine for parent/receiver separation or throughput once the single-GPU path is correct;
- postpone inference-engine optimization until causal value is demonstrated.

LRT's official code runs its stages on a single GPU, which supports feasibility of the general pattern, but Qwen3.5's hybrid implementation and our sequence lengths must be measured rather than assumed.

---

## 22. Execution order and decision points

```text
CURRENT
finish/preserve specialist routing + specialist worker competence lane

A0
workspace transport and gate-zero equivalence

A1
causal split-information latent transfer
    fail → repair channel/bank; do not add recurrence
    pass → preserve bridge

A2
real C→C recursive handoff
    capability/efficiency admitted separately

B1/B2
static vs matched-FFN vs recurrent coordinator sidecar
    no recurrent advantage → keep bridge, stop local-depth claim
    recurrent advantage → depth curve

B4
cost-aware real inference-time halting

C
combine sidecar + inter-node workspace

D
optional full-attention Cache-to-Cache extension

LATER
specialist/larger-model adapters, sanitized upward latent reports,
multi-sharer fan-in, workload-driven training and pruning
```

No phase is advanced merely because training loss falls. Advancement requires fresh causal outcome evidence.

---

## 23. The intended architectural endpoint

```text
                           ROOT C — frozen pretrained Qwen
                                  │
                         latent workspace U_root
                                  │
                   recurrent local reasoner ↻ ↻ ↻
                                  │
                    solve / H / C / larger expert?
                                  │
                  ┌───────────────┴───────────────┐
                  │ typed control + latent W      │
                  ▼                               ▼
          SUB-COORDINATOR C                 SPECIALIST H
          local recurrence ↻ ↻              bounded work
                  │                               │
                  └──── explicit, governed returns ┘
```

The language model supplies semantic understanding, world knowledge, code/tool priors, and generation. The recurrent sidecar supplies adaptive local depth. The workspace supplies high-bandwidth cognitive communication. The harness supplies legal organization and resource governance.

The system should ultimately learn four resource choices:

```text
1. how many local recurrent cycles this node deserves;
2. whether another node should be instantiated;
3. which specialist/capacity should be activated;
4. when the expected gain no longer justifies local GPU cost.
```

The one-sentence north star remains:

> Use the cheapest local cognition that works, dynamically assemble or grow more cognition only where the task demands it, and later compress recurring expensive capability when experience makes that possible.

---

## 24. References

### Direct implementation references

- Chen, Z. and Fu, J. **Latent Recurrent Thoughts: Recurrent Refinement of Proposed Latents for Reasoning with Frozen LLMs.** 2026. [Paper](https://arxiv.org/abs/2609.01117) · [Official code](https://github.com/czl-david/latent-recurrent-thoughts)
- Fu, T. et al. **Cache-to-Cache: Direct Semantic Communication Between Large Language Models.** ICLR 2026. [Paper](https://arxiv.org/abs/2510.03215) · [Official code](https://github.com/thu-nics/C2C)
- Li, S. et al. **DeepLoop: Depth Scaling for Looped Transformers.** 2026. [Paper](https://arxiv.org/abs/2607.13491) · [Official code](https://github.com/lszshu/DeepLoop)
- Wang, G. et al. **Hierarchical Reasoning Model.** 2025. [Paper](https://arxiv.org/abs/2506.21734) · [Official code](https://github.com/sapientinc/HRM)
- Jolicoeur-Martineau, A. **Less is More: Recursive Reasoning with Tiny Networks / Tiny Recursive Models.** 2025. [Paper](https://arxiv.org/abs/2510.04871) · [Code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)

### Frozen-LLM latent computation

- Liu, Z. et al. **Deliberation in Latent Space via Differentiable Cache Augmentation.** ICML 2025. [Paper](https://proceedings.mlr.press/v267/liu25bc.html)
- Hao, S. et al. **Training Large Language Models to Reason in a Continuous Latent Space (Coconut).** 2024. [Paper](https://arxiv.org/abs/2412.06769)
- Xu, Y. et al. **SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs.** 2025. [Paper](https://arxiv.org/abs/2502.12134) · [Code](https://github.com/xuyige/SoftCoT)

### Backbone and comparison references

- Qwen Team. **Qwen3.5-2B.** [Model configuration](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json) · [Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- Nanbeige LLM Lab. **Nanbeige4.2-3B: Unlocking Agentic Capabilities in a Compact Model.** 2026. [Paper](https://arxiv.org/abs/2607.22083). Relevant as evidence that loop-aware pretraining matters; its fixed two-pass architecture is not the dynamic-depth target of this plan.
- Sapient. **HRM-Text.** 2026. [Paper](https://arxiv.org/abs/2605.20613) · [Code](https://github.com/sapientinc/HRM-Text). Useful as a recurrent-language-model engineering reference, not the immediate coordinator sidecar.

### Motivating but not yet reproducible enough to anchor gates

- Mostik. **Read More.** [Project description](https://mostik.ai/read-more). Motivates learned latent bridges between models; claims should remain separate from the reproducible LRT/C2C evidence until fuller technical details are available.
