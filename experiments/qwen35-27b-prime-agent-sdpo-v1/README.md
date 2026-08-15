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
training curriculum.

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
