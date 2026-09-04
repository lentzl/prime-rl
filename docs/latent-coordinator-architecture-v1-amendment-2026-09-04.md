# Latent Coordinator Architecture v1 — Research Amendment 2026-09-04

**Status:** prospective design amendment; documentation only  
**Parent plan:** `docs/latent-coordinator-architecture-v1.md`  
**Execution authorization:** none; current specialist-worker lane and protected e33/H176 remain unchanged  
**Prepared:** 2026-09-04

This amendment incorporates three developments that materially sharpen, but do not reorder, the latent-coordinator program:

1. **Full-bandwidth Transformer** (`arXiv:2608.08888`) strengthens the within-node recurrent-depth axis and shows a second path to latent recurrence beyond the small sidecar.
2. **Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning** (`arXiv:2609.01449`, submitted 2026-09-01) motivates timestep-free recurrence and a corruption curriculum for learning an anytime recurrent operator.
3. **Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM Communication** (`arXiv:2607.26773`, submitted 2026-07-29) strengthens Phase A's causal evaluation and makes compute-matched self-substitution mandatory.

The architectural thesis remains:

```text
LOCAL DEPTH
inside a coordinator
recurrent/iterative latent computation

DISTRIBUTED BREADTH / ORGANIZATIONAL DEPTH
across coordinators and specialists
governed latent handoff + typed control plane
```

Neither axis replaces the other. The long-term runtime must learn when to spend GPU-seconds on another local update, another coordinator, a specialist, a larger model, or a combination.

---

## 1. Trainer alignment after Full-bandwidth Transformer

The Trainer's current architectural interpretation is adopted:

- full-bandwidth feedback is an **additional local-depth candidate**, not a substitute for C→C workspace handoff;
- the frozen-e33 sidecar remains the smallest reversible first local-depth experiment;
- a future full-bandwidth Qwen branch is a different experiment because it modifies the model's input pathway and requires continued training;
- recurrence-stability ideas such as mostly shallow training with a small deeper-pass fraction, bounded/normalized recurrent state, noise regularization, train/inference boundary randomization, and long-horizon contraction/oscillation diagnostics should be imported where applicable;
- standard/no-feedback operation remains an explicit control;
- post-training must eventually be on-policy under the actual recurrent inference mode;
- latent reasoning does not replace tools, exact computation, resource ownership, or typed control actions.

The eventual systems experiment remains factorial:

```text
local recurrence OFF / latent handoff OFF
local recurrence ON  / latent handoff OFF
local recurrence OFF / latent handoff ON
local recurrence ON  / latent handoff ON
```

This directly asks whether local depth and distributed breadth are additive, substitutive, or interactive.

---

# Amendment A — causal audit for latent communication

## 2. Replace the primary Phase A message controls

The parent plan currently emphasizes `NO-LATENT / ZERO / NOISE / SHUFFLED / CORRECT`. The new causal-audit paper shows that this is not sufficient to identify **other-agent value**.

The primary Phase A audit is amended to use four mandatory semantic arms:

```text
M0      NO MESSAGE
MOTH    OTHER-EXAMPLE MESSAGE
MSELF   SELF-GENERATED MESSAGE
MCUR    CURRENT-EXAMPLE / CORRECT PARENT MESSAGE
```

`ZERO` and `NOISE` remain useful diagnostics for numerical sensitivity and gating behavior, but they are no longer substitutes for `MOTH` or `MSELF`.

### Definitions

**NO MESSAGE (`M0`)**  
The receiver gets the normal typed objective and no latent cognitive payload.

**OTHER-EXAMPLE (`MOTH`)**  
The receiver gets a latent message generated through the same sender/bridge path but from a different, approximately matched task. This preserves message presence, shape, norm, and sender/interface effects while removing current-example content.

**SELF-GENERATED (`MSELF`)**  
The receiver itself receives an equal or prospectively matched amount of extra latent-compute budget and generates a message through the same interface from the current example. This tests whether a separate parent node contributes anything beyond what the receiver could have generated locally.

**CURRENT-EXAMPLE (`MCUR`)**  
The receiver gets the actual parent-generated message for the current task.

### Mandatory causal contrasts

```text
Overall pathway effect:
OPE = U(MCUR) - U(M0)

Other-example/message-presence effect:
OME = U(MOTH) - U(M0)

Current-example content-attributable gain:
CAG = U(MCUR) - U(MOTH)

Self-substitution gap / other-agent value:
SSG = U(MCUR) - U(MSELF)

Receiver-self contribution relative to generic message:
DSC = U(MSELF) - U(MOTH)
```

Use task utility rather than hidden expert/message labels. When sampling variance matters, use paired confidence intervals and a prospectively chosen practical-equivalence margin.

---

## 3. Why `MSELF` is architecturally decisive

A correct-vs-shuffled gain proves that example-specific information matters. It does **not** prove that another cognitive node was worth instantiating.

Our runtime ultimately faces this resource decision:

```text
OPTION 1
same C
+ another local recurrent cycle / self-generated latent computation

vs.

OPTION 2
instantiate another C
+ separate cognition
+ latent handoff
```

Therefore `MCUR` vs `MSELF` is not merely an evaluation refinement. It is a direct experimental proxy for the system's eventual economic decision:

> **Think more locally, or organize more externally?**

If `MSELF ≈ MCUR` at matched GPU-seconds, external coordination has not demonstrated marginal cognitive value for that task regime.

If `MCUR > MSELF` under matched public cost, the separate coordinator contributes information/computation the receiver did not reproduce itself.

If `MSELF > MCUR`, the system should prefer local depth unless external organization has another benefit such as parallelism, ownership isolation, latency, or specialist access.

---

## 4. Compute matching for self-substitution

The self-generated arm must not be a rhetorical control. Its compute budget must be measured and prospectively specified.

At minimum record:

```text
sender feature-extraction GPU-seconds
bridge encoding GPU-seconds
receiver self-message GPU-seconds
receiver decode GPU-seconds
peak VRAM
latent bytes
generated text tokens
```

Two admissible comparison styles:

1. **Equal-operation control:** sender and receiver generate messages using the same workspace encoder/interface and same number of forward/recurrent operations.
2. **Equal-cost control:** allow implementations to differ, but match measured GPU-seconds within a frozen tolerance.

The first is cleaner mechanistically; the second is closer to the eventual runtime utility question. Report both when feasible.

---

## 5. Phase A1 revised admission logic

A latent channel may support several different claims. Do not collapse them.

### Claim A — receiver sensitivity

`MCUR` changes receiver behavior versus `M0`.

This alone does not establish useful communication.

### Claim B — example-specific causal content

`MCUR > MOTH` on task utility with a prospectively meaningful gap.

This establishes that current-example content matters.

### Claim C — separate-agent value

`MCUR > MSELF` under matched compute/cost.

This is the strongest architectural communication claim and should be required before saying that another C node is superior to equivalent local latent compute in that task regime.

### Claim D — applied utility

The complete sender+handoff path beats self/local compute on a public utility metric incorporating task success, GPU-seconds, latency, memory, or parallel critical path.

This is the production admission criterion.

The first A1 semantic-channel gate can still pass on Claim B before Claim C is demonstrated, because the primitive may be useful for scoped information transfer even when another-agent value is not yet positive. Promotion into the adaptive systems architecture should distinguish these levels explicitly.

---

## 6. Model-scale audits do not transfer automatically

The causal-audit paper reports qualitatively different component effects for Qwen3-4B and Qwen3-8B. Therefore:

- an admitted e33-2B bridge/audit does not automatically admit a 4B or 8B adapter;
- every new model-size or model-family workspace adapter receives its own `M0/MOTH/MSELF/MCUR` causal audit;
- multi-sharer/fan-in variants receive another audit because message-presence and generic-message effects can change with composition;
- router utility estimates should be model/adapter specific rather than assuming a globally stable latent-channel benefit.

---

# Amendment B — timestep-free recurrent coordinator

## 7. Default to no explicit loop clock

The parent plan proposed small time features such as normalized cycle index and used-budget fraction. The new timestep-free iterative-reasoning result weakens the case for making those features part of the default recurrent architecture.

The revised primary arm is:

```text
R_phi(z, U, role/context)
```

with **no explicit cycle/timestep embedding**.

The cycle-conditioned version becomes an ablation:

```text
R_phi(z, U, role/context, cycle_index)
```

Rationale: a genuinely reusable anytime update should infer progress from its persistent state and current problem representation rather than specialize behavior to an absolute loop number.

Do not remove non-temporal context that is causally part of the task, such as root/non-root role, public resource budget, or ownership scope. The removed feature is specifically the hidden loop clock.

---

## 8. Add an anytime-depth extrapolation gate

Training and evaluation depths must be separated prospectively.

Recommended first design after basic recurrence is connected:

```text
TRAIN rollout lengths:
randomized across a bounded range

BPTT window:
short and fixed initially

EVAL depths:
inside training range
+ moderately beyond
+ far beyond
```

The new paper demonstrates a particularly strong version: rollout lengths sampled from 20-160, truncated BPTT through four steps, and continued improvement out to 10,000 recurrent steps on Sudoku-Extreme. We should not expect those magnitudes to transfer to coordinator cognition, but the experimental principle is directly applicable.

A candidate recurrent coordinator should therefore report:

- success as a function of cycle depth;
- fraction of tasks newly solved at each depth;
- fraction of previously correct tasks that regress;
- action/solution stability;
- hidden-state change norms;
- oscillation/divergence rate;
- compute-adjusted utility.

The desired evidence is not “T=8 is best.” It is:

> a shared update remains useful and stable when run beyond its typical training horizon.

---

## 9. Add a latent corruption curriculum as a Phase B training axis

The diffusion paper suggests that the corruption schedule can be valuable mainly as a **training curriculum**, while inference can use a much simpler stationary/noisy process.

This does not imply that a Qwen-attached workspace should literally become a diffusion model. The proposed adaptation is narrower:

> During recurrent-sidecar training, repeatedly perturb or partially replace the **visible/proposal workspace** while preserving the recurrent hidden memory, and require the shared update to recover task-useful cognition.

### Proposed state split

```text
U          immutable task anchor extracted from frozen Qwen
v_t        visible/proposal workspace, intentionally corruptible
h_t        persistent recurrent hidden memory

(v_{t+1}, h_{t+1}) = R_phi(corrupt(v_t), h_t, U)
```

This makes `h_t` the only reliable path for accumulated recurrent progress when `v_t` is strongly perturbed, analogous to the paper's memory intervention result.

### Training arms

At minimum compare:

1. **CLEAN:** ordinary recurrent training, no artificial corruption.
2. **FIXED:** fixed corruption strength.
3. **IID:** independent randomly sampled corruption strength each recurrent step.
4. **ORDERED CURRICULUM:** prospectively ordered corruption schedule across an episode.

The source paper's striking result is specific to Sudoku: ordered annealed corruption strongly outperformed fixed, i.i.d., and clean-only training. In our project that motivates the ablation; it does not predetermine the winner.

Candidate corruption operators for a compact latent workspace:

- additive Gaussian noise after normalization;
- stochastic slot replacement;
- feature/channel dropout;
- random masking of a subset of workspace slots;
- convex mixing with a fresh proposal/noise vector.

Start with one simple operator. Do not create a combinatorial sweep before establishing connectivity.

---

## 10. Persistent hidden state becomes a first-class object

The source paper's intervention evidence indicates that its persistent hidden state, not the repeatedly corrupted visible state, carries iterative progress. That suggests a useful distinction for our sidecar:

```text
COMMUNICABLE WORKSPACE W
compact, governed, transferable, potentially corrupted/reconstructed

PRIVATE RECURRENT MEMORY H
node-local, persistent across local cycles, not automatically transferable
```

This separation is attractive for our ownership model.

A coordinator can have rich private recurrent memory while exposing only the smaller governed workspace to another node. That reduces the pressure to dump full internal state across ownership boundaries.

Future experiments can ask whether a compressed function of `H` should influence `W`, but `H` itself should not become an ungoverned cross-node channel.

---

## 11. Truncated BPTT should be the default engineering hypothesis

The timestep-free paper achieves its result with a four-step truncated BPTT window despite much longer persistent rollouts. This reinforces our existing plan to avoid full unrolling as the default for deep recurrence.

Revised ladder:

1. establish gradient connectivity with very short full BPTT;
2. move to a short truncated window while carrying detached hidden state across much longer episodes;
3. compare a few larger windows prospectively;
4. reject the assumption that deeper BPTT is automatically better.

Track gradient norm by recurrent distance and hidden-state sensitivity. If longer windows worsen optimization, preserve that result rather than increasing the window for conceptual purity.

---

## 12. Halting: stability is a feature, not the criterion

The revised halting view is:

```text
state convergence / action stability
        +
solution/value confidence
        +
expected marginal utility of another cycle
        -
public next-cycle compute cost
```

A simple norm threshold is only a diagnostic baseline.

For coordinator tasks, a state may appear locally stable while still reflecting a wrong decomposition or wrong resource choice. Conversely, stochastic/anytime recurrence may temporarily move between candidate solutions before settling.

The eventual halting head should therefore estimate something like:

```text
continue iff
E[U_after_next_cycle - U_now | current recurrent state, public task context]
    > cycle_cost
```

Training targets can be derived retrospectively from fixed-depth evaluation trajectories without exposing answer correctness at inference.

---

# Amendment C — full-bandwidth recurrence in the local-depth ladder

## 13. Distinguish three local-depth mechanisms

The revised architecture contains three candidate local-depth price points:

```text
TIER 0 — ordinary e33
no extra latent recurrence

TIER 1 — small recurrent sidecar
~10-30M trainable recurrent module
cheap repeated local updates
frozen e33

TIER 2 — recurrent full-Qwen prefill / full-bandwidth-style backbone
another pass through the full 2B semantic core
expensive but much higher-capacity local update
requires a separately trained recurrent-capable Qwen lineage
```

Decode-time full-bandwidth feedback is another capability of Tier 2: once trained, it can carry top-layer state across generated tokens at very small extra per-token overhead.

This suggests a future escalation policy:

```text
ordinary C
  ↓ if insufficient
cheap sidecar cycles
  ↓ if insufficient
full-Qwen recurrent prefill
  ↓ if insufficient
spawn C / specialist / larger model
```

The ordering is a hypothesis. The utility router should ultimately learn it from measured success/cost.

---

## 14. Do not conflate sidecar recurrence with full-bandwidth backbone adaptation

The frozen-e33 sidecar remains first because it is reversible and isolates the local-depth question.

A full-bandwidth Qwen branch requires:

- a separate checkpoint lineage;
- modified input construction/fusion;
- continued language-model training with mixed feedback-pass schedules;
- standard/no-feedback retention tests;
- long-horizon stability diagnostics;
- later on-policy post-training under recurrent decoding.

The Full-bandwidth paper demonstrates that recurrence can be introduced after ordinary pretraining has begun. It does **not** establish the minimum adaptation budget for a mature Qwen3.5-2B checkpoint. Therefore no assumption is made that e33 can be cheaply converted.

Canonical e33 remains frozen.

---

# Amendment D — the combined resource-allocation experiment

## 15. Turn the new controls into the system's core economic experiment

Once Phase A handoff and Phase B local recurrence are independently functional, build a matched-budget task bank with at least these arms:

```text
A. BASE
receiver C only

B. LOCAL
receiver C + extra local recurrent compute

C. EXTERNAL-TEXT
parent/sub-C external compute + governed text handoff

D. EXTERNAL-LATENT
parent/sub-C external compute + governed latent handoff

E. COMBINED
local recurrence + external latent handoff
```

Within the latent arms retain the causal message interventions:

```text
NO MESSAGE
OTHER-EXAMPLE
SELF-GENERATED
CURRENT-EXAMPLE
```

The resulting decomposition answers two different questions:

1. **Communication question:** does current-example latent content from another node causally improve the receiver?
2. **Resource question:** is that improvement worth more than spending the same compute locally?

This is the cleanest experimental bridge from today's architecture research to the long-term adaptive runtime.

---

## 16. Revised execution order

```text
CURRENT
finish/preserve specialist routing + specialist-worker competence lane

PHASE A0/A1
workspace transport + causal message audit
M0 / MOTH / MSELF / MCUR mandatory

PHASE A2
real C→C recursive handoff
capability, other-agent value, and utility claims separated

PHASE B1
frozen-e33 recurrent sidecar
no explicit timestep by default
static/FFN/recurrent matched controls

PHASE B2
variable-depth extrapolation
short truncated BPTT + long persistent rollouts

PHASE B3
latent corruption-curriculum ablation
only after ordinary recurrence is connected

PHASE B4
cost-aware inference-time halting

PARALLEL LATER LOCAL-DEPTH BRANCH
full-bandwidth/recurrent-Qwen feasibility on a separate lineage

PHASE C
factorial combination of local recurrence × latent handoff

PHASE D
optional strict Cache-to-Cache/full-attention extension
```

No execution-order change is implied by this amendment.

---

## 17. Updated promotion vocabulary

Future reports should use these terms precisely:

**Latent-pathway effect**  
A latent message changes task behavior versus no message.

**Example-specific latent value**  
The current-example message beats an other-example message.

**Other-agent latent value**  
The separate sender's current-example message beats a compute-matched receiver self-message.

**Local-depth value**  
A recurrent sidecar beats a matched-parameter non-recurrent sidecar and/or additional local cycles improve fresh task utility.

**Anytime-depth value**  
The same recurrent update remains stable/useful beyond its ordinary training depth without requiring a loop clock.

**Applied population value**  
External organization, latent handoff, recurrence, or their combination improves verified end-to-end utility after measured compute/resource cost.

These distinctions prevent a generic latent perturbation, prompt-like effect, or extra-compute effect from being mislabeled as communication or cognitive breadth.

---

## 18. References added by this amendment

- Mariia Drozdova, Aidan Sirbu, Pietro Miotti, Robert Obryk, Mayalen Etcheverry, Eyvind Niklasson, Blake Richards. **Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning.** arXiv:2609.01449, 2026. https://arxiv.org/abs/2609.01449
- Huixiang Zhang, Mahzabeen Emu. **Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM Communication.** arXiv:2607.26773, 2026. https://arxiv.org/abs/2607.26773
- Xi Wang et al. **Full-bandwidth Transformer.** arXiv:2608.08888, 2026. https://arxiv.org/abs/2608.08888

Existing references in the parent plan to LRT, C2C, DeepLoop, HRM/TRM, Differentiable Cache Augmentation, Coconut, Qwen3.5, Nanbeige, and HRM-Text remain applicable.

---

## 19. Updated one-sentence architectural criterion

> **A useful latent architecture must prove not only that more hidden-state compute helps, but which information is causal, whether another node adds value beyond compute-matched self-reasoning, and whether the runtime can buy that value more cheaply by thinking deeper locally or organizing cognition externally.**
