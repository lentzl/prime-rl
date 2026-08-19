# Constrained Yield Ramp V1

Purpose: cross the Prime Agent post-spawn passive-yield support barrier without weakening the frozen Harness Master benchmark or hand-writing a golden response.

The scaffold is **training/collection only**. Model-visible natural N1 prompts and the ordinary hard verifier are unchanged. At one hidden-oracle-eligible root state — correct retained child spawn, no child message yet, and no coordinator-local work remaining — the next model request has its tools removed. The model still samples the response. Prime Agent still decides whether the no-tool turn yields/resumes correctly. The unchanged verifier still decides whether the complete episode is a hard success.

Local-work N1 variants are never scaffolded. They are the paired control against the wrong rule `after rlm always stop`.

## Y0 — constrained-exploration admission, no gradient

Run `scripts/run_qwen35_27b_natural_yield_scaffold_admission_v1.sh` from canonical R7 with the scaffold enabled only inside that launcher.

Admission requires error-free traces, at least 8 scaffold fires, at least 4 hard successes, at least half of scaffolded episodes free of the post-spawn forbidden-tool event, and at least 3 semantic families. Failure means the scaffold representation itself is not a useful support bridge; do not train.

## Y1 — diverse success harvest, no gradient

Run `scripts/run_qwen35_27b_natural_yield_scaffold_harvest_v1.sh` on a disjoint window. Require at least 16 verified scaffolded hard successes across at least 6 semantic families before distillation.

No handcrafted completion or reasoning is ever created. The harvest contains only model-generated trajectories that the unchanged harness scorer accepts.

## Y2a — zero-LR routing audit

Run `scripts/run_qwen35_27b_natural_yield_scaffold_sft_v1.sh y2-routing-audit` with the default `NATURAL_YIELD_SCAFFOLD_TRAIN_LR=0`.

The SFT filter is `procedural_harness_master_v1.natural_yield_scaffold.keep_scaffolded_natural_yield_response`. Before any nonzero run, decode/export-audit every active token and require:

- 100% of active CE tokens belong to the root/coordinator session;
- 100% belong to the single model-generated no-tool scaffold response, including its turn-ending token when represented;
- 0 spawn/action tokens, 0 child tokens, 0 reasoning before the selected response, 0 final-answer tokens, 0 RL/ref-KL/SDPO mass;
- every active trajectory has hard reward 1 and a valid scaffold audit marker.

Any mismatch blocks Y2b.

## Y2b — first distillation step

Only after Y0, Y1, and Y2a pass, rerun Y2 from exact canonical R7 with `NATURAL_YIELD_SCAFFOLD_TRAIN_LR=2.5e-8` and a fresh start window. One full-weight step only. Do not continue from the candidate before Y3.

The unusually low first dose is intentional: the mask is concentrated on a tiny behavioral decision and prior broader CE at larger doses interfered with harness prerequisites.

## Y3 — native unscaffolded same-draw screen

Run `scripts/run_qwen35_27b_natural_yield_scaffold_postflight_v1.sh R7_PATH CANDIDATE_PATH`.

The launcher forcibly unsets `PROCEDURAL_NATURAL_YIELD_SCAFFOLD`. It evaluates R7 and candidate on identical frozen draws for:

- natural immediate yield;
- natural spawn + required local work + yield;
- atomic state;
- atomic send.

The existing comparator authorizes **replication only** when native yield improves (hard success or >=2/8 reduction of the exact forbidden post-spawn-tool event), state/send are non-inferior, local-work behavior does not gain premature yield, and exact target answers do not regress.

A positive Y3 must be replicated on a second disjoint fixed draw before continuation. R7 remains canonical until then. Once native N1 successes exist at useful frequency, disable the scaffold and return to broad natural N1/N2 benchmark-native optimization.

## Safety / invariants

- Never enable the scaffold in frozen VALID/OOD, Y3, or promotion runs.
- Never train from a scaffolded trajectory that does not pass the ordinary hard verifier.
- Never use scaffolded child-session tokens as targets.
- Do not weaken forbidden-control, ownership, ordering, cardinality, or exact-answer gates.
- The scaffold is a temporary exploration bridge, not part of the deployed policy or benchmark definition.
