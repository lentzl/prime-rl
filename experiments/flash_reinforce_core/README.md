# FlashREINFORCE: isolated reference core

**Status:** mathematical implementation and CPU checks, not a live Prime-RL integration,
model-training result, or permission to change the running campaign.

This directory implements FlashREINFORCE Eqs. 5–11 independently of a trainer. It follows
[the paper](https://github.com/yifanzhang-pro/FlashREINFORCE) and the
[Trainer handoff](https://github.com/lentzl/rlm/issues/1#issuecomment-5669869979).
No Molt runtime or model is imported, and no existing loss dispatch, sampler, optimizer,
configuration, dependency pin, or numerical training dtype is changed.

The review branch starts from published Prime-RL `fa21395a68728ffb80e61cb660e9b921e4c76430`
only to transport these independent files. It is **not** a port against the frozen host
`a050d3938144e670d75ffbacea0db75c27cd9056`, its local campaign branch, or uncommitted overlays.
Trainer owns that separate integration.

## Objective

Given one finite reward `R[i]` for each of the original `B` logical trajectories:

- Centre once: `A[i] = R[i] - mean(R)` (Eq. 5).
- At every sampled policy action, compute the Bernoulli KL from recorded behavior
  probability to current probability (Eq. 6).
- Average KL across **all** action tokens of each trajectory and apply one detached
  `mean_KL <= threshold` decision (Eqs. 7–8).
- Minimize `-sum_i accepted[i] * A[i] / (B*T[i]) * sum_t stopgrad(ratio[i,t]) * current_logp[i,t]`
  (Eq. 11), where `ratio = exp(current_logp - recorded_behavior_logp)`.

Rejected trajectories stay in the original reward mean and denominator `B`. No
ratio clipping, reward whitening, critic, reference-model forward, entropy filtering,
length-weighted batch mean, or accepted-count renormalization is applied.

The sampled-action KL is only a drift proxy, not a certificate of full-policy closeness.
A small proxy can coexist with an extreme importance ratio. An admitted ratio overflow
raises rather than silently changing the objective by clipping.

## Interface and two-phase execution

Each fragment supplies aligned, one-dimensional tensors:

| Tensor | Meaning |
|---|---|
| `current_logprobs` | Differentiable current-policy log-probability of each recorded action, evaluated at the actual call context. |
| `behavior_logprobs` | Fixed log-probabilities recorded by the sampler, not retrospectively generated estimates. |
| `trajectory_index` | `int64` global-batch indices `0..B-1`. All turns of the selected role's episode use the same index. |
| `action_mask` | Boolean; selects each genuinely sampled action once. External/context/repeated-history tokens are false. |

Arbitrary sentinels, including NaNs, are permitted **only in masked positions**.
Selected log-probabilities must be finite and <= 0. Probability-one (`logp=0`)
endpoints are supported; selected `-inf`/NaN values fail explicitly.

1. `trajectory_statistics(...)` returns detached **sums and counts**, never fragment-local
   accept/reject decisions. A fragment may be empty or contain parts of many trajectories.
2. SUM these statistics across the entire update. `merge_statistics(...)` is an
   in-process reference; production collectives are Trainer's responsibility.
3. `prepare_update(rewards, global_statistics, threshold=...)` constructs the global
   plan once. Rewards contain exactly one entry per original trajectory. The original
   batch size is their vector length, not the number of fragments or accepted episodes.
4. `loss_fragment(..., plan)` produces a fragment contribution that is **already
   normalized by global B and T**. Sum fragment losses or accumulate their gradients,
   then perform one optimizer update. Do not divide again by token/pack/microbatch counts.

Example for one complete batch (all literals below are synthetic):

```python
import torch
from flash_reinforce import trajectory_statistics, prepare_update, loss_fragment

current = torch.tensor([-0.4, -0.6, -0.8], requires_grad=True)
behavior = torch.tensor([-0.5, -0.6, -0.7])
trajectory = torch.tensor([0, 0, 1], dtype=torch.int64)
mask = torch.ones(3, dtype=torch.bool)
rewards = torch.tensor([1.0, 0.0])

stats = trajectory_statistics(current, behavior, trajectory, mask, batch_size=2)
plan = prepare_update(rewards, stats, threshold=1e-3,
                      expected_token_counts=torch.tensor([2, 1]))
loss = loss_fragment(current, behavior, trajectory, mask, plan)
loss.backward()
```

For memory-limited execution, the detached-statistics pass can precede a differentiable
replay of each fragment. Keep learner weights, tokens, contexts, forward conventions and
relevant stochastic state fixed between these passes. Alternatively retain a complete
forward graph. Changing the learner between trust statistics and loss is not supported.
The core cannot verify that a caller supplied identical model computations in both passes.

`expected_token_counts` is an optional completeness check. It detects missing/duplicate
counts, not replacement of one action by another. The adapter must maintain action IDs,
policy/role/episode identities and original batch ownership; this file cannot reconstruct
them from log-probability tensors. Do not treat a truncated rollout as a complete one.
Use the existing shared-node loss masking rather than counting repeated prefixes again.

`UpdatePlan.has_reward_signal` tells whether any admitted centred advantage is nonzero.
It does not guarantee a nonzero parameter gradient. If false, momentum or decoupled
weight decay could still change parameters when `optimizer.step()` is called. Trainer
owns optimizer/scheduler skip semantics and fresh-batch consumption. No optimizer runs
inside the core.

## Numerical scope

Scalar probability/loss arithmetic promotes FP16/BF16 tensors to FP32 and retains FP64
inputs. This does **not** change model or optimizer dtypes and cannot restore bits already
lost when probabilities were recorded. `log1p`/`expm1` handle complementary probabilities
near one; exact endpoints are handled separately. Only negative roundoff in a theoretical
nonnegative KL is clamped to zero. No upper KL or ratio cap is hidden in the implementation.
All-rejected and equal-reward losses are finite graph-connected zeros. Rejected and
zero-advantage actions are removed before importance ratios are exponentiated.

## Tests actually run

In an environment with PyTorch and pytest installed, from this directory:

```sh
uv run --no-project python -m pytest -q
```

2026-09-14: **22 tests passed** on CPU, Python 3.13.5, PyTorch 2.10.0+cpu, pytest 9.0.2.
The focused checks cover:

- gradients against an independently constructed ratio objective;
- gradient signs and detached behavior/reward/trust terms;
- original batch denominator and rejected-trajectory centering;
- repeated-token length neutrality and fragmented accumulation;
- a global trajectory rejection even though one of its fragments has zero local drift;
- masked NaN placeholders, empty fragments, equal rewards and all-rejected batches;
- near-one probabilities against a 70-digit Decimal reference, exact endpoints,
  low-precision promotion, overflow rejection and input/count errors.

The historical 13 behavior log-probabilities from Trainer's handoff are used **only as
numerical test values**. The learner perturbation in that test is synthetic. We do not
have that historical model, its full context or a current model forward. This is not a
new RL trajectory, a use of its old GRPO advantage, or a Qwen numerical-parity result.

CPU fragmentation checks simulate summing shards. They do **not** certify actual FSDP,
context-parallel collectives, GPU kernels, collector behavior, checkpoint replay or learning.

## Trainer integration work still required

The inspected frozen interface invokes per-sample losses and applies global selected-token
normalization. A simple custom-hook substitution is therefore insufficient. Carry trajectory
metadata through rendering/branch conversion/packing; retain raw reward population; aggregate
trust over all fragments; preserve the objective's normalization through distributed backward;
and prevent reuse for multiple optimizer updates. CE/ref-KL/SDPO component normalization
must remain separate rather than adopting this denominator accidentally.

Current R13 summarization exports have empty sampled-token ID, mask and log-probability
arrays. They are **not** inputs to this objective. Enable the actual collector path for fresh
rollouts and inspect the recorded sampling distribution. Temperature/top-p/top-k metadata
alone neither proves probability correctness nor repairs missing support. Preserve per-call
rendered inputs: the chat template can rewrite preceding reasoning at later turns. A final
flattened transcript need not represent earlier decisions' real conditioning histories.

Define a small outcome-bearing TRAIN episode under the current inbox contract. Reward must
measure the intended outcome, not only file presence or a completion claim. R13's artifact
reward of one coexisted with approval/protocol violations. Keep assistance explicit and keep
supervised teacher continuations separate from actual policy samples. All-identical rewards
supply no centred reward signal; adjust reachable teaching/tasks instead of fabricating it.

No threshold, optimizer, policy checkpoint, batch schedule or deployment recommendation is
encoded as a campaign default here. The example threshold is illustrative. No live trainer,
GPU host, runtime, Hugging Face artifact or checkpoint was modified for this contribution.
