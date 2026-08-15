# Qwen3.5 27B Prime Agent SDPO V1

This experiment tests feedback-conditioned SDPO on natural child-owned Prime Agent
ownership failures without dropping ordinary reinforcement learning from the same
update. The diagnostic source emits a typed, answer-free correction only after an
unsuccessful first decision. Prime-RL requires that exact contract and restricts the
SDPO loss to the first serialized coordinator tool call. Five disjoint GRPO sources
retain coordinator-owned, direct, single-child, parallel-child, follow-up, and
handshake behavior.

`zero-lr-audit.toml` is a mechanism audit, not training. It runs one complete update
with `lr = 0`, creates no checkpoint, and proves that genuine on-policy failures can
flow through feedback admission, teacher replay, token filtering, and the analytic
SDPO loss alongside group-relative RL before any parameter-changing run is considered.
Its fixed-seed 16-trace batch is the minimum complete screen: four diagnostic traces,
one two-rollout group from each ordinary retention source, and two causal groups so
both follow-up and handshake remain represented. The nonuniform source ratios exist
only to make that exact audit allocation deterministic; they are not proposed as a
training curriculum. The audit caps concurrency at eight episodes to match vLLM's
active sequence capacity while collecting the complete batch in two waves.

The launcher validates the completed run before returning success and writes
`AUDIT.json` plus `AUDIT.txt` into the run directory. A passing verdict requires
the pinned model revision across all resolved services, typed answer-free feedback
only on the diagnostic failures, at least two diagnostic codes and two resource and
phrasing families, both causal communication families, positive RL and SDPO token
mass, zero CE and reference-KL mass, a successful optimizer step at exactly zero
learning rate, and no checkpoint or model-weight artifact. Stable token exports are
matched one-to-one with reconstructed Verifiers branches to prove that SDPO reaches
only the first serialized coordinator tool call, child branches receive zero SDPO,
and retention sources receive only GRPO.

## PCIe-only runtime qualification

The experiment launchers keep NCCL P2P disabled but enable SHM by default. This
combination is required on the tested eight-L40S host: CUDA peer reads and writes
are unsupported, while the six-rank SHM path completes the audit's exact
383,418,612-element BF16 FSDP reduce-scatter in 0.48-0.90 seconds. Disabling both
P2P and SHM forced NCCL onto its socket transport; four ranks stalled in that
collective until the one-hour process-group timeout while two ranks reached the
following barrier.

Run `scripts/probe_nccl_reduce_scatter.py` under `torchrun` before reusing this
configuration on a different GPU topology. The launch defaults remain explicitly
overridable through `NCCL_P2P_DISABLE` and `NCCL_SHM_DISABLE`.
