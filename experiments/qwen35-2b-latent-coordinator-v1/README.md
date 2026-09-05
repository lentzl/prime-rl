# Qwen3.5-2B latent coordinator v1

This experiment keeps the protected coordinator and worker policies immutable while testing two
orthogonal extensions: local recurrent depth and governed inter-node workspace handoff. This branch
implements only the Phase B local-depth primitive.

## Phase B implementation boundary

`prime_rl.latent.recurrent.TimestepFreeRecurrentSidecar` operates on a compact task anchor with shape
`[batch, 8, 256]`. Its transition has no cycle argument or timestep embedding. Each update reinjects
the immutable task anchor and evolves two different objects:

- `visible_workspace`: the bounded cognitive payload that may later be exported through the governed
  workspace protocol;
- `private_memory`: persistent node-local state that is never returned by `export_workspace()` and
  is not eligible for C→C transfer.

The visible update is a bounded residual around the task anchor. Its residual scale is zero-initialized,
so the untrained sidecar is an exact visible no-op while private memory can still evolve. The memory
update is a convex gated interpolation with a bounded candidate. These are conservative stability
choices; they do not assert that the learned transition is contractive.

`OneShotFeedForwardSidecar` is the primary matched-parameter control. Its default 10,562,155 active
parameters differ from the recurrent module's 10,562,048 by 107 parameters (about 0.001%). It receives
the same task anchor once, has the same bounded zero-initialized output form, and has no persistent
state or repeated update.

`LocalDepthCodec` keeps Phase B independent from inter-node handoff. It detaches the last eight visible
states from the same coordinator's own prompt, projects them to the local task anchor, and decodes the
refined workspace onto the frozen embedding-norm shell. `compose_local_depth_inputs` inserts those
vectors at a caller-verified assistant boundary and excludes them from the token loss. Neither object
constructs workspace provenance or accepts another node's state.

The local PrimeRL dense Qwen3.5 language body already supports both required prototype interfaces:

- `Qwen3_5Model.forward(..., inputs_embeds=..., seq_lens=...)` accepts continuous embeddings;
- its `BaseModelOutputWithPast.last_hidden_state` supplies the frozen feature capture used to form the
  task anchor.

The causal-LM wrapper accepts `inputs_embeds` for text-only Qwen3.5, but returns logits rather than
hidden states. For a VLM-shaped Qwen3.5 config, its current VLM branch ignores the wrapper's
`inputs_embeds` argument and reconstructs embeddings from `input_ids`. The preflight must inspect the
staged e33 config. If it is VLM-shaped, the prototype should call the frozen language body with
`inputs_embeds` and apply the existing frozen `lm_head` directly; it must not silently route through
the incompatible wrapper branch. No change to the Qwen backbone, checkpoint keys, vLLM serving path,
or Gated DeltaNet state is required for the first Phase B experiment.

## Deliberate exclusions

- No full-bandwidth/backbone adaptation.
- No inter-node workspace transport or provenance implementation on this branch.
- No learned halt policy.
- No GPU run, model call, checkpoint mutation, or external publication is authorized by this plan.
- No corruption curriculum until clean recurrence is connected and evaluated.

The prospective controls, training hypothesis, diagnostics, and resource accounting are frozen in
`phase-b-recurrent-sidecar-v1-plan.json`. Evaluation seeds, held-out hashes, and final quantitative
admission thresholds remain owned by the independent evaluation strand or root and must be filled in
before the first model call.

The fixed-depth host runner is present at `scripts/latent/run_phase_b_fixed_depth_smoke_v1.py`. The old
fixed-depth plan remains intentionally non-launchable. The separate A0C-bound plan references an exact
committed four-of-four carrier-only receipt and binding before importing Torch or Transformers. It is
still subject to independent gatekeeper review and an explicit root schedule. Rejected A0R evidence is
preserved as a blocker for cache/generation/A0/A1 claims and is never accepted through this path.

The first authorized A0C-bound start failed before its first useful forward because the preserved
OpenAI-wire assistant target stores `function.arguments` as a JSON string while the pinned Qwen chat
template iterates over it as a mapping. Its exact FAILURE, log, and post-failure immutable audit are
preserved by the failed-start binding. The prospective B-R runner converts only that one final target
path with strict JSON-string-to-object parsing, proves the source unchanged elsewhere, and renders and
tokenizes all 12 exact selected targets before importing the model class or entering any CUDA-facing
path. B-R received a separate reviewed plan/launcher freeze; the prior output namespace remains
immutable and must never be reused.

That first B-R tokenizer-only preflight then rejected before model/CUDA/output because the generic
generation prompt and the full assistant target did not have the required prefix relationship. The
preserved target has nonempty `reasoning_content`: the pinned Qwen template's default opening emits an
empty closed think block, while the full target renders the preserved reasoning inside its think block.
The prospective B-R2 repair does not use a longest-common-prefix boundary, disable thinking, or alter
reasoning. It passes `enable_thinking=True` explicitly for plain, opening, and full renders, requires
both exact string and token prefix relations, hashes the nonempty reasoning bytes before/after, and
continues to inject before and mask through the verified explicit-thinking assistant opening. B-R2 is
again nonlaunchable until its own mechanism/freeze pair receives independent review.

B-R2 then passed that real tokenizer-only gate on all 12 frozen rows with balanced 4/4/4 actions and
reached the protected model load. It failed before its first useful forward in provenance hashing:
e33's state dictionary contains a zero-dimensional BF16 tensor, and PyTorch cannot reinterpret that
scalar directly as `uint8` when the element sizes differ. The prospective B-R3 repair changes only
`_module_tensor_sha256`: it reshapes the contiguous CPU tensor to one dimension before the byte view,
while continuing to hash the original tensor name, dtype, and shape separately. The raw tensor helper
is unchanged because each of its actual call sites hashes a rank-2 token tensor or rank-3 hidden-state
capture. B-R3 changes no tensor value, model path, sidecar, arm, or numerical predicate.

B-R3 passed those boundaries and entered the real four-arm loop, but the forward-metric phase retained
autograd graphs that are useful only in the later single-row backward probes. A completed latent
forward still owned nearly all of GPU0 when the next forward requested another 516 MiB, producing an
atomic OOM failure with exact post-failure e33 and no-update evidence. B-R4 runs only the unchanged
12-row BASE/STATIC/FFN/RECURRENT metric phase under `torch.no_grad()`, converts each loss and recurrent
diagnostic to detached Python values, and releases each model output before the next arm. The canonical
and hypothetical-open-gate backward probes remain separate, gradient-enabled, and numerically
unchanged. B-R4 does not use checkpointing, offload, fewer rows, shorter sequences, or altered arms.
