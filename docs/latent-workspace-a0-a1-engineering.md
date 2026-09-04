# Latent Workspace A0/A1 Engineering Substrate

**Status:** CPU-safe implementation substrate; no model call or optimizer authorization

**Protocol:** `prime-rl/latent-workspace/v1`

**Protected checkpoints:** coordinator e33 and worker H176 remain immutable

This document makes Phase A0/A1 launchable after an independent evaluator freezes the held-out bank and admission thresholds. It does not connect the latent path to the production runtime.

## 1. Runtime boundary

The v1 workspace is a rank-two floating tensor plus immutable metadata. The tensor is not deliverable until `validate_workspace_delivery` accepts all of:

- protocol version;
- tensor shape, dtype, finiteness, and checksum;
- deterministic workspace identity;
- producer, parent, intended receiver, task, and scope;
- exact producer checkpoint, bridge checkpoint, and capture-spec hashes;
- creation time and expiry;
- delivery mode and, for causal tests, the exact audit arm.

Operational workspaces must have identical source and intended task IDs. Cross-task tensors are representable only as `MOTH` in `causal_audit` mode, and a receiver must independently expect that exact arm. The runtime has no general cross-task override.

V1 allows only `coordinator → coordinator`. Worker returns, sibling reuse, unprovenanced tensors, and unknown schemas fail closed. Metadata is serializable; tensor persistence is deliberately not implemented.

Implementation: `src/prime_rl/latent/workspace.py`.

## 2. Parent hidden-state capture

The initial capture spec is fixed structurally, but its hash is recorded per run:

```text
layer: final (-1)
boundary: accepted_delegation
span: last <=128 non-padding positions, left-padded to exactly 128 bridge positions
gradient: detached
batch: one parent sequence in A0/A1
```

The harness must first accept the typed delegation action. Only then may the instrumented Transformers backend read `outputs.hidden_states[-1]`. `capture_parent_features` selects the bounded span using the attention mask, detaches it, left-pads it to a fixed bridge-compute shape, and emits the capture-spec hash. Cached features must additionally record the parent checkpoint, tokenizer/chat-template, evidence ID and split, tensor checksum, dtype/shape, and code commit.

No hook should capture an earlier speculative decode state. The parent sees an evidence packet but never the downstream query; one cached parent feature must be reused bitwise for every query paired with that evidence packet.

## 3. Receiver `inputs_embeds` integration

The prototype uses Transformers, not the production vLLM path:

1. Render the ordinary child system/control contract and child-visible query with the exact Qwen3.5 chat template.
2. Tokenize through, but not beyond, the assistant-generation opening.
3. Obtain ordinary token embeddings from frozen e33.
4. Project the governed workspace to eight 2048-dimensional vectors.
5. Use `compose_receiver_inputs` to insert those vectors at the recorded assistant-opening index.
6. Insert attention-mask ones, shift subsequent position IDs, and insert `-100` labels for workspace positions.
7. Pass only `inputs_embeds` to frozen e33 and compute loss on the accepted target response.

The exact-zero control is a hard standard-path bypass: it materializes no extra positions and returns the original embeddings, mask, positions, and labels unchanged. A trainable gate, even if initialized to zero, retains latent positions so it can receive gradient. A0 must separately compare actual e33 logits under the hard bypass before training is allowed.

The live template boundary, generation cache behavior, thinking mode, and norm-shell projection remain GPU integration checks; they are not guessed in the pure-logic layer.

Implementation: `src/prime_rl/latent/policy_adapter.py`.

The initial learned-query bridge is also defined locally: source LayerNorm, 2048→256 projection, eight learned queries over multi-head attention, output LayerNorm, and a 256→2048 decoder. The decoder rescales each projected slot to the frozen embedding shell and applies a bounded scalar gate initialized to 0.001, leaving a gradient path into the whole bridge. The exact-zero A0 control remains the hard bypass described above. With production dimensions the bridge has about 1.3M trainable parameters. It is not attached to a model or optimizer by this change.

Implementation: `src/prime_rl/latent/bridge.py`.

## 4. Split-information bank

The deterministic generator defines four semantic families:

1. keyed numerical records with later aggregation operators;
2. relational joins from entity through group to code;
3. structured configuration lookup;
4. ownership graphs requiring a two-edge traversal.

Every parent evidence packet supports three child queries. The parent view contains evidence but no query or answer. The child view contains a query but no evidence or answer. The answer key is a third, evaluator-only projection. Train, validation, and held-out splits use disjoint surface-template IDs; held-out structures are larger or differently rendered. Random opaque identifiers, values, order, and wording prevent a stable final-answer slot.

For `MOTH`, donors are a deterministic family-matched derangement, ordered by approximate evidence length. The donor ID is recorded. For `MCUR`, every query for an evidence packet must receive the same cached workspace. This is the critical anti-answer-encoding check: the sender cannot tailor its latent to a downstream query it never sees.

The generator intentionally has no default seed. The independent evaluation owner must choose held-out seeds, generate the manifests, and freeze their hashes before any model call.

Implementation: `src/prime_rl/latent/transfer_bank.py` and `scripts/latent/build_transfer_bank.py`.

## 5. Mandatory causal audit

Primary semantic arms, run for every held-out task with otherwise identical receiver text:

```text
M0     no latent message
MOTH   family/length-matched other-example message
MSELF  current receiver self-message
MCUR   current-example parent message
```

`ZERO` and norm-matched `NOISE` remain numerical/gating diagnostics. They cannot substitute for `MOTH` or `MSELF`.

The primary MSELF control is equal-operation:

- one frozen-Qwen feature-extraction forward for parent and self;
- one identical workspace-encoder forward for parent and self;
- identical, independently frozen padded input-token budgets for feature extraction;
- identical 128-position bridge inputs, with padding masked;
- identical workspace decoder and receiver decode budget;
- arms measured sequentially on the same physical GPU;
- prompt/prefill/decode tokens, GPU-seconds, latency, and peak VRAM logged separately.

The receiver self-message is built only from its visible current query. It cannot access parent evidence. A separately frozen relative GPU-seconds tolerance supplies the secondary equal-cost audit. Runs outside that tolerance are infrastructure-invalid, not negative causal evidence.

Report, without collapsing them:

```text
OPE = U(MCUR)  - U(M0)
OME = U(MOTH)  - U(M0)
CAG = U(MCUR)  - U(MOTH)
SSG = U(MCUR)  - U(MSELF)
DSC = U(MSELF) - U(MOTH)
```

A1 can establish example-specific latent value from `CAG` before it establishes other-agent value from `SSG`. Applied promotion requires the later cost-aware system test.

The checked-in plan is deliberately a draft. `validate_launch_plan` refuses it until the independent evaluator supplies and hashes the held-out bank, all practical thresholds, the decode budget, and a canonical plan hash.

Implementation: `src/prime_rl/latent/audit.py`, `scripts/latent/validate_a0_a1_preregistration.py`, and `experiments/qwen35-2b-latent-workspace-v1/a0-a1-preregistration-draft.json`.

## 6. Launch sequence after authorization

1. Independent evaluator generates and seals train/validation/held-out bank manifests and thresholds.
2. Run the preregistration validator; any failure blocks launch.
3. Rehash e33 and H176, record tokenizer/template hash, and verify no trainable or optimizer-owned base-model parameters.
4. On one GPU, run A0 hard-bypass logit equivalence and nonzero-position/mask inspection.
5. Cache detached parent features by evidence ID.
6. Train only encoder, decoder, and receiver gate through frozen e33.
7. Rehash e33 and H176.
8. Run all six arms in paired order with randomized, recorded arm ordering.
9. Emit immutable per-task outcomes and cost records; classify any provenance or compute-match failure as infrastructure-invalid.

No production runtime integration occurs before A1 passes its independently frozen causal gate.

## 7. Requested resources

No GPU is required for the checked-in substrate. After explicit launch approval, request exactly:

```text
GPU:       1 × NVIDIA A6000 48 GB
Host RAM:  128 GB
Local NVMe: 250 GB free
Lease:     24 hours, hard stop
Software:  existing pinned uv environment; no dependency update
```

Use the same physical GPU sequentially for `M0/MOTH/MSELF/MCUR/ZERO/NOISE`. One 2B parent pass is cached and the same frozen 2B receiver is reused, so a second GPU is not required for causal validity. If the 24-hour rung cannot finish, preserve the exact manifest and timing evidence and request a new bounded lease; do not relax the audit or silently change hardware.

## 8. Remaining launch risks

- Hard-bypass embedding equivalence is proven locally, but actual e33 logit equivalence still requires a GPU/model test.
- Qwen3.5 chat-template placement and hybrid-cache generation with `inputs_embeds` require token-for-token integration verification.
- The first bank tests channel semantics, not real recursive coordination or production utility.
- MSELF equal operations may still differ slightly in sequence length; both operation counts and measured cost must be reported.
- Opaque latent tensors remain potentially sensitive even at eight slots; operational persistence is still prohibited.
- A successful `MCUR > MOTH` result does not by itself prove `MCUR > MSELF` or justify production activation.
