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

The fixed-depth host runner is present at `scripts/latent/run_phase_b_fixed_depth_smoke_v1.py`, but its
checked-in state is intentionally non-launchable. It requires both a newly authorized plan and an exact
binding to a prospective four-of-four A0C carrier-only success receipt before importing Torch or
Transformers. `phase-b-a0c-binding-placeholder-v1.json` documents that interface and remains marked
`unresolved`; rejected A0R evidence is never accepted through this path.
