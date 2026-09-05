# Phase B fixed-depth smoke assessment

**Status:** frozen design assessment; no GPU/model authorization

The smallest useful Phase B smoke is a same-node, teacher-forced connectivity test. Frozen e33 reads a
coordinator prompt once to provide detached local features. A small codec produces an eight-slot task
anchor. The STATIC arm decodes that anchor directly, the FFN arm applies one matched-parameter update,
and the RECURRENT arm applies the same timestep-free transition exactly four times while retaining
private memory. The first smoke performs forward probes plus one backward diagnostic but constructs no
optimizer and changes no parameter. No coordinator is spawned, no governed workspace is delivered,
H176 is not loaded, and no generated action is executed.

## Relative simplicity and safety

Phase B is simpler than Phase A1 as a systems experiment: it has one prompt view, one frozen model,
one ownership scope, and no sender/receiver pairing, provenance policy, split-information bank,
message substitution, or cross-node information-flow risk.

It is **not materially simpler or safer than Phase A0**. A0 is a zero-update transport/no-op test.
Phase B adds gradient-bearing parameters, repeated differentiable state, fixed-depth activation cost,
and the possibility of recurrence instability even though its first smoke performs no update. Both phases still share the difficult low-level
boundary: exact Qwen3.5 hidden-state capture, `inputs_embeds` insertion, tokenizer/template placement,
loss masking, embedding-norm calibration, and equivalence to the standard path. B removes protocol
surface but adds optimization surface.

The independent ordering recommendation is therefore unchanged:

1. Run the prospective A0C carrier-only probe because its hard-bypass/no-update result isolates the
   exact model-interface predicates B consumes with less destructive power.
2. Bind a newly frozen B smoke to A0C's immutable four-of-four success receipt, then run B without
   waiting for A1 capability training. B consumes A0C's model-interface proof, not an A workspace or
   bridge checkpoint. Rejected A0R control-flow is not a substitute for that prospective receipt.
3. Keep A1 semantic transfer and B local-depth capability as independent claims. Do not combine them
   until each has passed its own causal evaluation.

## Why the smoke is deliberately weak

The 12 selected examples come from the already preserved e33 coordinator-action curriculum, balanced
four per action. Every row receives a forward probe; one prospectively named row receives a backward
diagnostic. The dataset includes all three established action targets, but target text is never executed
and no worker is instantiated. This can expose broken masks, recurrence, no-op behavior, or frozen-weight
boundaries; it cannot establish trainability behind the closed zero gate, generalization, or useful local
depth.

The smoke therefore has no `RECURRENT > FFN` threshold. A later independent evaluation must supply
fresh held-out tasks, hashes, and a quantitative admission rule before any capability claim.

## Host fit

The host's two 48GB A6000s are ample for the one frozen 2B instance plus approximately 11.6M
gradient-bearing parameters in the active arm. Disk, not VRAM, is the immediate constraint: only 88GB is free. The smoke
keeps feature caches in RAM, disables every checkpoint save, forbids tensor persistence, caps new
artifacts at 512MiB, and refuses to start below 60GiB free. A two-hour hard stop is conservative for
the zero-update probe while preventing an integration fault from occupying the host indefinitely.

The staged coordinator is `Qwen3_5ForConditionalGeneration` with a 2048-wide text model under
Transformers 5.6.2 and Torch 2.11. Both the upstream wrapper and text model accept `inputs_embeds`.
The smoke freezes the upstream Transformers wrapper as its backend; the repository's custom PrimeRL
VLM wrapper has a different branch boundary and is deliberately outside this first integration test.

All arms run sequentially on GPU 0 with one frozen e33 instance because the soft-input loss backward
must retain e33 activations for the currently active arm. GPU 1 remains idle. Parallel copies would add
memory and reproducibility surface without making a twelve-row no-update probe meaningfully faster.

Completion is transactional: the run writes either `SUCCESS.json` or `FAILURE.json` through a temporary
file and an atomic no-replace hard link, never both, and never overwrites either. The receipt includes
immutable pre/post hashes for e33 and every codec/sidecar parameter tensor plus the bound A0C receipt
hash.

## Runner readiness boundary

`scripts/latent/run_phase_b_fixed_depth_smoke_v1.py` contains the host execution path. The historical
`phase-b-fixed-depth-smoke-v1-plan.json` remains permanently non-launchable. After A0C completed all
four probes, `phase-b-fixed-depth-smoke-a0c-v1-plan.json` was prospectively frozen as a separate plan
bound to the exact committed receipt and binding. The receipt's file SHA-256 and its internal canonical
SHA-256 (computed while omitting only `receipt_sha256`) are checked independently. Binding files live in
the deployed worktree, outside the fresh terminal output directory.

The new freeze still does not schedule itself: independent gatekeeper review and an explicit root
schedule are required. A run stops compute at 114 minutes, reserves five minutes to rehash e33 weights
and metadata, revalidate the A0C binding and receipt, and rehash the plan and selection, then retains one
minute for atomic terminal-receipt publication inside an independent 120-minute `timeout` wrapper.
Scientific connectivity failures are recorded as `mechanism_rejected`; provenance, timeout, OOM, and
resource failures are `infrastructure_invalid`.
