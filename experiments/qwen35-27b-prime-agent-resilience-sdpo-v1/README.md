# Qwen3.5 27B Prime Agent Resilience SDPO V1

This experiment asks whether feedback-conditioned SDPO can learn from the natural
recoverable errors already present in Prime Agent trajectories. It starts from the
untouched pinned Qwen3.5-27B thinking model and uses `prime-agent-resilience-v1`
calibration tasks. The SDPO teacher receives only the ordinary tool output, traceback,
or child message that followed a sampled action. It receives no expected answer,
golden action, demonstration, typed policy label, or separately authored correction.

`zero-lr-audit.toml` runs one four-trace full-weight BF16 step at exactly zero learning
rate and writes no checkpoint. Its filter selects only serialized IPython calls whose
matching tool response contains a recoverable error or omitted-`await` coroutine. It
does not train successful setup, inspection, messaging, or waiting calls. The delayed
result family is excluded because waiting feedback is not a failure-local traceback
and needs a different credit mechanism. The audit is successful only if token exports
prove positive SDPO mass on genuine failed calls, zero RL/CE/reference-KL mass, and
position-aligned teacher spans whose prefixes contain the observed failure. A passing
mechanism audit does not authorize a parameter-changing run or imply improved repair
behavior.

Validate a completed audit with:

```bash
.venv/bin/python scripts/validate_prime_agent_resilience_sdpo_zero_lr_audit_v1.py \
  /ephemeral/outputs/qwen35-27b-prime-agent-resilience-sdpo-v1/zero-lr-audit \
  --output /ephemeral/outputs/qwen35-27b-prime-agent-resilience-sdpo-v1/zero-lr-audit/AUDIT.json
```

This audit is retained as method evidence only. The procedural Harness Master benchmark
supersedes the proposed isolated mixed update as the next weight-changing objective.
Any later reuse must restart from untouched 27B and route failure-local SDPO inside a
generated end-to-end cohort whose conjunctive HarnessScore is the promotion target.
