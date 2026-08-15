# Qwen3.5 27B Prime Agent SDPO V1

This experiment tests feedback-conditioned SDPO on natural child-owned Prime Agent
ownership failures. The environment emits a typed, answer-free diagnostic only after
an unsuccessful first decision. Prime-RL requires that exact contract and restricts
the SDPO loss to the first serialized coordinator tool call.

`zero-lr-audit.toml` is a mechanism audit, not training. It runs one complete update
with `lr = 0`, creates no checkpoint, and proves that genuine on-policy failures can
flow through feedback admission, teacher replay, token filtering, and the analytic
SDPO loss before any parameter-changing run is considered.

The launcher validates the completed run before returning success and writes
`AUDIT.json` plus `AUDIT.txt` into the run directory. A passing verdict requires
the pinned model revision across all resolved services, typed answer-free feedback
on every effective failure trace, positive finite SDPO gradients with no competing
loss component, a successful optimizer step at exactly zero learning rate, and no
checkpoint or model-weight artifact.
