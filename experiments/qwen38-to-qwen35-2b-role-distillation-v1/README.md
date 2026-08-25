# Qwen3.8-27B to Qwen3.5-2B role distillation

This experiment trains two independent rank-16 LoRA adapters from verified
Qwen3.8-27B Prime Agent trajectories:

- `orchestrator`: only sampled spans from the primary/root agent session;
- `child`: only sampled spans from non-root child sessions.

The base student is the immutable private snapshot
`lentzl/rlm-prime-agent-qwen35-orchestrator-candidate-r1-20260809` at revision
`c90f5a78e81f59cd62c2e3d0661a2ece1eb8a428`. Its dense weight file must hash to
`c75915dd41cd4fc9b1a1ef5582c6fd14913fc6f9971a58feca3b72b4bfcad406` before
either run may start. Qwen3.8-27B revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` is the frozen teacher.

The corpus is not a raw trace dump. A checked admission manifest names every
accepted trace and pins each source JSONL plus its run `VERSIONS.txt` hash. The
run metadata must identify the exact teacher model and immutable revision and
must contain the frozen causal-template, dialect, generator, taskset, and
boundary-verifier hashes. Pre-causal qualification traces fail this gate.
Asynchronous axes admit only
complete hard successes. The direct axis uses the owner-aligned audit: correct
in-context arithmetic and one clean coordinator-local IPython calculation are
both valid; unnecessary delegation, unrelated resource access, repeated work,
or completion-gate inspection are excluded. Each required asynchronous axis
must contribute at least four complete trajectories. This floor is not
relaxable.

`scripts/export_prime_agent_role_sft_v1.py` validates the admission manifest,
splits the message graph by its primary and child roots, checks model-call
session lineage, and emits a separate Parquet dataset and audit manifest for
one role. The student renderer reconstructs the chat in the student's token
space; only assistant messages contribute to CE. The renderer explicitly uses
the 2B checkpoint's thinking-disabled polarity because filesystem checkpoint
paths otherwise inherit the Qwen3.5 large-model default. Recorded teacher
reasoning remains part of historical assistant turns.

Both training runs are one-step, rank-16 LoRA probes. They are sequential and
start from the same untouched dense snapshot. They do not merge weights, alter
R7 or Qwen3.8, or co-serve teacher and student. The active frozen Q38/Q35 matrix
and the untouched 2B baseline must finish before the first optimizer step.
`scripts/build_q35_2b_baseline_manifest_v1.py` freezes that baseline only when
all four 16-task axes are infrastructure-complete under a qualified runtime,
the student and frozen-harness hashes match, and all runtime provenance is
recorded. The role-SFT launcher requires this manifest and rechecks every
referenced artifact hash before it can invoke the trainer. Each exported role
corpus is also bound to the exact teacher admission manifest and source root.
`scripts/validate_q35_2b_role_training_inputs_v1.py` revalidates that entire
chain, checks every Parquet row against an admitted trace and role, requires
complete axis representation, and structurally freezes the one-step rank-16
assistant-only training template before SFT can start.

Candidate evaluation keeps the dense student immutable and serves one adapter
at a time through Prime-RL's runtime LoRA endpoint. The dedicated
`scripts/run_q35_2b_role_adapter_eval_v1.sh` wrapper accepts only a stable
rank-16 checkpoint whose adapter config points back to the exact student and
whose target-module set matches the preregistration. The harness requests the
content-addressed adapter name, while `VERSIONS.txt` records and hashes the
base model, adapter config, adapter weights, resolved inference config, and
frozen harness separately. No dense merge is required for evaluation.
`scripts/build_q35_2b_role_eval_manifest_v1.py` then requires all 64 candidate
traces to be infrastructure-complete and paired to the untouched baseline's
exact task keys under the same qualified runtime. It reports strict task-level
gains and losses without changing the preregistered promotion policy.

The later 4B lane, including merging its accepted LoRA into dense weights, is
out of scope for this experiment.

The original Q38 qualification runtime is frozen after reproducible vLLM
infrastructure failures. `runtime-v2-proposal.json` records a separately
versioned vLLM 0.27.1 qualification hypothesis, its exact official wheel hash,
and fail-closed gates. The strategist authorized only this runtime
qualification and its conditional recovery sequence in control-channel comment
`5348735850`; optimizer authorization remains gated by the frozen corpus,
baseline, provenance, and hash requirements.

New qualification runs also record the installed vLLM version and distribution
URL plus hashes of `uv.lock` and the resolved inference configuration in
`VERSIONS.txt`. The admission manifest carries these fields per source. Older
frozen runs are labeled `legacy_unrecorded`; their files are not rewritten.

`scripts/prepare_vllm_runtime_v2.sh` clones the v1 environment into a distinct
prefix, replaces only vLLM with the hash-verified official wheel, and fails if
any other installed distribution changes. `scripts/run_q38_runtime_v2_coldstart_qualification_v1.sh`
runs exact task 3806011 from three independent server cold starts and writes a
qualification manifest only if all three traces are infrastructure-valid and
the implicated JIT/CUDA/RPC signatures are absent.

The first isolated preparation verified the wheel hash but found that the
official 0.27.1 wheel requires newer Torch, torchvision, FlashInfer, TVM FFI,
TileLang, and Quack Kernels than the v1 environment. Control-channel comment
`5348832970` authorized a package-only stable-ABI falsification while retaining
the complete dependency-mismatch record. The finalizer verified that vLLM was
the only changed distribution, exercised the stable-ABI/Qwen imports, and
hashed both dependency inventories and `pip check` evidence.

Runtime-v2a failed its first exact-task cold start with a TP0 CUDA illegal
memory access followed by EngineCore death and evaluator ProviderErrors. The
trace envelope was incomplete and is permanently non-admitted. The fail-closed
rule stopped cold starts two and three, missing-teacher recovery, the untouched
2B baseline, corpus admission, and training. Control-channel comment
`5348926929` records the result. Qualifying teacher counts remain
direct/N1a/N1a-local/N1b = `15/5/3/0` against unchanged floors `12/4/4/4`, and
optimizer steps remain zero. A separately labeled CUDA-blocking attribution
run is proposed in comment `5348942266`; it is not authorized by runtime-v2a.

The strategist authorized exactly one diagnostic-only reproduction in comment
`5352117943`. With only `CUDA_LAUNCH_BLOCKING=1` added, exact task 3806011
completed cleanly: one complete error-free trace, 10/10 HTTP 200 model calls,
hard harness score 1.0, and final-answer exactness 1.0. The same inference-time
JIT warnings appeared for layer norm, causal-convolution update, and packed GDN
decode, while `_compute_slot_mapping_kernel` did not recur. Because launch
blocking removed the illegal-memory failure instead of pinning it to a
synchronous site, this is timing-sensitivity evidence only. The trace is
permanently non-admitted, runtime-v2a remains rejected, teacher counts remain
`15/5/3/0`, the untouched 2B baseline remains unstarted, and optimizer steps
remain zero. `cuda-blocking-diagnostic-result.json` records the exact result and
control-channel report `5352279980`; no follow-on mitigation is authorized.

Control-channel comments `5353871747` and `5356066259` then authorized a
separately versioned `runtime-v2b-blocking` lane. Its three fresh-server
qualification attempts all completed exact task 3806011 without infrastructure
errors: three complete traces, 30/30 HTTP 200 model calls, distinct serving
processes, and clean teardown. All three produced exact final answers but hard
score 0, which was allowed for serving qualification; all three traces remain
permanently non-admitted. Comment `5357367145` records that screen.

The operational hypothesis failed immediately afterward. The first N1a-local
teacher-recovery task, again exact task 3806011, returned one HTTP 200 response
and then raised a synchronous `CUBLAS_STATUS_EXECUTION_FAILED` in the
Qwen3-next MLP `gate_up_proj` linear call. CUDAEvent cleanup then reported an
illegal memory access, TP0 died, EngineCore stopped, and the evaluator received
provider errors. No trace envelope was serialized. The fail-closed wrapper did
not start the remaining N1a-local tasks, N1b, the untouched 2B baseline, corpus
admission, or training. `runtime-v2b-result.json` and control-channel report
`5357455076` preserve the result. Counts remain `15/5/3/0`, optimizer steps
remain zero.

Control-channel comment `5359331913` broadened the autonomous runtime mandate
and required a byte/config comparison before the next intervention. The
successful v2b qualification run and failed recovery run have byte-identical
`inference.toml` files (SHA-256 `ab04c120320321ca4126d53248f884baf76a52b52456a5c189d8830427e1438a`).
Their recorded configs differ only by the intended task count, one versus
five; the generator seeds each absolute task index independently, so task
3806011 is unchanged. The 3/3 fresh-start-only gate is therefore retired as
insufficient.

`runtime-v2c-proposal.json` preregisters the next one-variable runtime-only
hypothesis: retain `CUDA_LAUNCH_BLOCKING=1` and set
`VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0`. The installed vLLM 0.27.1 release
then bypasses its packed recurrent GDN decode fast path in favor of its
existing standard recurrent update. The stronger gate requires three fresh
sentinel starts plus two independent fresh-server, five-task sequential soaks
through N1a-local indices 3806011-3806015. All diagnostic traces are
permanently non-admitted; every leg must be complete, infrastructure-clean,
and cleanly torn down before this lane can be called stable.

Runtime-v2c was rejected on its second fresh start. Sentinel r1 completed one
error-free 11-call trace through the standard recurrent path and tore down
cleanly. The immediately following r2 restart failed before server health in
vLLM's `determine_available_memory` profiling pass: TP0 raised
`CUBLAS_STATUS_EXECUTION_FAILED` in an attention `qkv_proj`, and the kernel
recorded NVIDIA Xid 31 (an MMU virtual-read fault) before CUDA illegal-memory
cleanup and EngineCore shutdown. No request completed and neither sustained
soak started. `runtime-v2c-result.json` records the fail-closed result; the
durable incident copy verifies against manifest SHA-256
`592314abb8a0fb31cc65c971d3318bedd6c9e97b3cc572b9cc5d772e2216a696`.

`runtime-v2d-proposal.json` preregisters the next lifecycle-only hypothesis.
It retains every v2c runtime setting but requires 90 consecutive seconds with
no GPU compute process before every server start. This directly tests the
same-second teardown/restart boundary observed between v2c r1 and r2. The gate
remains three sentinel starts plus two restarted five-task soaks, with all
qualification traces permanently non-admitted.

Runtime-v2d also failed closed on sentinel r2. After the full quiescence
barrier, the server became healthy and returned one chat response, but the next
call raised a Triton illegal instruction specifically in
`fused_sigmoid_gating_delta_rule_update_kernel`, the non-packed GDN recurrence
selected by v2c. The host recorded Xid 13 (out-of-range register and multiple
warp errors) followed by Xid 43. Thus restart timing is not sufficient, and the
failure is now localized to both available vLLM 0.27.1 GDN decode kernels.
`runtime-v2d-result.json` records the result; the local incident manifest hash
is `20069d6277e4257a229907d12c999e7803355190e0eacf4e4ce0571c3a95cb41`.

`runtime-v2e-proposal.json` preregisters the final same-host vLLM 0.27.1
kernel hypothesis. A small reversible patch replaces only non-speculative,
one-token GDN recurrence with equivalent PyTorch tensor operations; packed and
fused-sigmoid Triton decode kernels are both bypassed. An exact-shape synthetic
comparison against the vLLM kernel passed with maximum output error
`3.814697265625e-06` and maximum recurrent-state error
`3.725290298461914e-09`. The wrapper restores the byte-identical original
vLLM module on exit. The full three-start/two-soak gate is unchanged; a v2e
failure reaches the preregistered boundary for declaring this host/runtime
family nonviable and requesting a materially different serving substrate.

Runtime-v2e reached that boundary. Sentinel r1 completed a clean 13-call trace
with both recurrent-decode Triton kernels absent. After restart, sentinel r2
returned one response and then failed visibly in an attention output-projection
`cublasGemmEx`; the host recorded Xid 13 out-of-range-register/multiple-warp
exceptions on two SMs and Xid 43. The preceding JIT warning was only the
causal-convolution update kernel. The wrapper stopped before r3 or either soak,
restored the original vLLM module to SHA-256
`1227d6f385a52296e9f08223544b1c5fdc7e8d9aa09a848e7a8e522a8dc51214`,
and left both GPUs idle. `runtime-v2e-result.json` records the result; the local
incident manifest hash is
`2ada7c6cd35b18afe1872d06ab5d30333800a48c2363b326fada0dfabec8b7a7`.
The current RTX 6000 Ada / vLLM 0.27.1 / Torch 2.11 / CUDA 12.8 serving family
is therefore abandoned for this pilot. Any next test requires a materially
different serving substrate or dependency-family decision.
